#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
"""End-to-end tests of the automatic solve-mode decision.

The automatic policy keeps the in-core solve when the node-total estimate fits
and switches to the staged chunked solve when it does not.  Both sides are
node totals: the requirement and the capacity are summed over the ranks of the
shared-memory domain.  Three single-rank cases pin that decision against the
budget:

* a budget far above the in-core requirement keeps the run in-core,
* a node capacity between the chunked and in-core aggregates switches to
  chunked and must reproduce the in-core field to round-off,
* a capacity below the chunked aggregate refuses with the required and
  capacity totals in the message.

Case (d) runs two ranks with equal budgets whose node total lands between the
in-core and chunked aggregates, so both ranks must agree on chunked.  Case (e)
runs four ranks with budgets whose sum lands in the same window and requires
the chunked field to match the four-rank in-core reference.  Cases (f) and (g)
exercise the ``[solve] solve_mode`` TOML key: once without a CLI flag and once
with the CLI override winning.  Case (h) gives the two ranks unequal budgets so
their per-rank decisions differ; the job must abort with the divergence
message instead of hanging, which exercises the local-bound/disagreement path
end to end.

The budget is forced with the test-only ``PICBUNCH_TEST_MEMORY_BUDGET_BYTES``
environment variable (see ``evaluateDeviceFit``).  ``PICBUNCH_TEST_SOLVE_MODE_RANKS``
makes every rank print its local decision before the collective broadcast.

usage: test_solve_mode_auto_switch.py <mpiexec> <picbunch-bin> <workdir>
"""
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_chunked_config_mpi import (  # noqa: E402
    TOL,
    compare_fields,
    failure_tail,
    mpi_command,
    write_config,
    write_input,
)

GRID = (48, 48, 48)
PADDING = 1
CHUNK_CELLS = 512
N_PARTICLES = 256
GIB = 1024.0**3

DECISION_RE = re.compile(r"^picBunch: solve mode: (auto -> \S+)(?: \((.*)\))?$")
RANK_DECISION_RE = re.compile(r"^picBunch: solve mode: rank (\d+): (.*)$")


def format_bytes(value):
    return f"{float(value) / GIB:.3f} GiB"


def padded_extents(grid, padding):
    # TSC needs two halo nodes; fft-padding >= 2 supplies them itself.
    halo = 2 if padding == 1 else 0
    return tuple(int(n) * padding + halo for n in grid)


def virtual_ranks(padded, chunk_cells):
    mx, my, mz = padded
    planes_per_rank = max(1, chunk_cells // (mx * my))
    wanted = (mz + planes_per_rank - 1) // planes_per_rank
    return max(1, min(wanted, min(mx, mz)))


def chunked_particle_bytes(particles):
    # The chunked gate counts the resident Bunch arrays (x, y, z, ux, uy, uz,
    # weight, charge, restMass: nine doubles) plus the boxOf/members
    # bookkeeping (two size_t).  Keep this in sync with Compute.cpp.
    return particles * (9 * 8 + 2 * 8)


def chunked_requirement(grid, padding, particles, chunk_cells, rank_count):
    mx, my, mz = padded_extents(grid, padding)
    ranks = virtual_ranks((mx, my, mz), chunk_cells)
    nz = (mz + ranks - 1) // ranks
    nx = (mx + ranks - 1) // ranks
    workspace = 16 * (mx * my * nz + 2 * nx * my * mz + 2 * max(mx * nz, mz * nx))
    particle_bytes = chunked_particle_bytes(particles)
    out_nz = grid[2]
    slab_planes = max(
        ((r + 1) * out_nz) // rank_count - (r * out_nz) // rank_count
        for r in range(rank_count)
    )
    plane_elems = grid[0] * grid[1]
    # Six resident output components plus the per-component emitter temporary
    # of ceil(Mz / virtual_ranks) planes (``nz``), alive with the slab and the
    # workspace.
    emitter_planes = nz
    return (
        workspace
        + particle_bytes
        + 6 * slab_planes * plane_elems * 8
        + emitter_planes * plane_elems * 8
    )


def chunked_aggregate(grid, padding, particles, chunk_cells, rank_count):
    """Sum of the per-rank chunked peaks over ``rank_count`` ranks."""
    mx, my, mz = padded_extents(grid, padding)
    ranks = virtual_ranks((mx, my, mz), chunk_cells)
    nz = (mz + ranks - 1) // ranks
    nx = (mx + ranks - 1) // ranks
    workspace = 16 * (mx * my * nz + 2 * nx * my * mz + 2 * max(mx * nz, mz * nx))
    plane_elems = grid[0] * grid[1]
    # The per-rank output slabs sum to the whole output volume; each rank also
    # holds its own emitter temporary of ceil(Mz / virtual_ranks) planes.
    slab_total = 6 * grid[2] * plane_elems * 8
    emitter_total = rank_count * nz * plane_elems * 8
    return (
        rank_count * (workspace + chunked_particle_bytes(particles))
        + slab_total
        + emitter_total
    )


def in_core_aggregate(grid, padding, particles, rank_count):
    """Sum of the per-rank in-core estimates over ``rank_count`` ranks."""
    distributed = rank_count > 1
    return rank_count * in_core_requirement(
        grid, padding, particles, rank_count, distributed
    )


def in_core_requirement(grid, padding, particles, rank_count, distributed):
    mx, my, mz = padded_extents(grid, padding)
    cp = mx // 2 + 1
    padded_nodes = mx * my * mz
    spectrum = mz * my * cp
    out_nodes = grid[0] * grid[1] * grid[2]
    if distributed:
        ranks = rank_count
        padded = (padded_nodes + ranks - 1) // ranks
        spec = (spectrum + ranks - 1) // ranks
        out = (out_nodes + ranks - 1) // ranks
        grid_bytes = padded * 5 * 16 + spec * 2 * 16 + out * 8
    else:
        grid_bytes = padded_nodes * 8 + spectrum * 3 * 16 + out_nodes * 7 * 8
    particle_bytes = particles * 32
    if padded_nodes <= (1 << 26):
        particle_bytes += padded_nodes * 4
    return grid_bytes + particle_bytes


def parse_decision(line):
    match = DECISION_RE.match(line.strip())
    if match is None:
        return None
    decision = {"mode": match.group(1), "detail": match.group(2) or ""}
    detail = decision["detail"]
    fields = re.search(
        r"required ([0-9.]+) GiB, capacity ([0-9.]+) GiB over (\d+) ranks?, "
        r"reason ([^)]+)",
        detail,
    )
    if fields is not None:
        decision["required"] = fields.group(1)
        decision["capacity"] = fields.group(2)
        decision["ranks"] = fields.group(3)
        decision["reason"] = fields.group(4)
        return decision
    reason = re.search(r"reason ([^)]+)", detail)
    if reason is not None:
        decision["reason"] = reason.group(1)
    return decision


def only_decision(stderr, label):
    lines = [
        line.strip()
        for line in stderr.splitlines()
        if line.strip().startswith("picBunch: solve mode:")
    ]
    if len(lines) != 1:
        raise RuntimeError(f"{label}: expected one decision line, got {lines!r}")
    decision = parse_decision(lines[0])
    if decision is None:
        raise RuntimeError(f"{label}: unparsable decision line {lines[0]!r}")
    return lines[0], decision


def rank_decisions(stderr, label):
    found = {}
    for line in stderr.splitlines():
        match = RANK_DECISION_RE.match(line.strip())
        if match is not None:
            found[int(match.group(1))] = match.group(2)
    if set(found) != {0, 1}:
        raise RuntimeError(f"{label}: expected decisions from ranks 0 and 1, got {found!r}")
    return found


def child_environment(env_extra=None):
    """Environment for a child run with inherited PICBUNCH_TEST_* removed.

    A stale PICBUNCH_TEST_MEMORY_BUDGET_BYTES in the caller's environment
    would silently change every case, so scrub the test-only namespace and
    re-add only the values the case sets explicitly.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("PICBUNCH_TEST_")
    }
    if env_extra:
        env.update(env_extra)
    return env


def run(command, label, env_extra=None):
    env = child_environment(env_extra)
    result = subprocess.run(
        command, text=True, capture_output=True, env=env, timeout=600
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{label}: command failed with exit {result.returncode}: "
            + failure_tail(result)
        )
    return result


def run_expect_failure(command, label, env_extra=None):
    env = child_environment(env_extra)
    return subprocess.run(
        command, text=True, capture_output=True, env=env, timeout=600
    )


def main():
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: test_solve_mode_auto_switch.py <mpiexec> <picbunch-bin> <workdir>"
        )
    mpiexec, binary, workdir = sys.argv[1:4]
    binary = os.path.abspath(binary)
    workdir = os.path.abspath(workdir)
    root = os.path.join(workdir, f"solve-mode-auto-{os.getpid()}")
    keep = os.environ.get("PICBUNCH_KEEP_TEST_OUTPUT") == "1"
    os.makedirs(root, exist_ok=True)

    chunked_one = chunked_requirement(GRID, PADDING, N_PARTICLES, CHUNK_CELLS, 1)
    in_core_one = in_core_requirement(GRID, PADDING, N_PARTICLES, 1, False)
    chunked_two_total = chunked_aggregate(GRID, PADDING, N_PARTICLES, CHUNK_CELLS, 2)
    in_core_two_total = in_core_aggregate(GRID, PADDING, N_PARTICLES, 2)
    chunked_four_total = chunked_aggregate(GRID, PADDING, N_PARTICLES, CHUNK_CELLS, 4)
    in_core_four_total = in_core_aggregate(GRID, PADDING, N_PARTICLES, 4)
    if not chunked_one < in_core_one:
        raise SystemExit(
            f"test setup: chunked ({chunked_one}) must stay below in-core ({in_core_one})"
        )
    for ranks, chunked_total, in_core_total in (
        (2, chunked_two_total, in_core_two_total),
        (4, chunked_four_total, in_core_four_total),
    ):
        if not chunked_total < in_core_total:
            raise SystemExit(
                f"test setup: {ranks}-rank chunked aggregate ({chunked_total}) must stay "
                f"below the in-core aggregate ({in_core_total})"
            )

    failures = 0
    try:
        input_path = os.path.join(root, "IN.h5")
        config_path = os.path.join(root, "CFG.toml")
        write_input(input_path, GRID)
        write_config(config_path, GRID)
        common = [
            "--input",
            input_path,
            "--config",
            config_path,
            "--fft-padding",
            str(PADDING),
            "--deterministic-deposit",
            "--shape",
            "tsc",
            "--input-iteration",
            "0",
            "--solve-mode",
            "auto",
            "--chunk-cells",
            str(CHUNK_CELLS),
        ]
        # Same launch without the CLI solve-mode flag, for the TOML cases.
        common_no_solve = list(common)
        solve_flag = common_no_solve.index("--solve-mode")
        del common_no_solve[solve_flag:solve_flag + 2]

        def case_dir(name):
            path = os.path.join(root, name)
            os.makedirs(path, exist_ok=True)
            return path

        # (a) A budget far above the in-core requirement keeps the run in-core.
        large = 1 << 40
        destination = case_dir("a_in_core")
        command = mpi_command(
            mpiexec,
            1,
            binary,
            [*common, "--destination", destination, "--staging-dir", destination + "-staging"],
        )
        result = run(
            command,
            "case a",
            {"PICBUNCH_TEST_MEMORY_BUDGET_BYTES": str(large)},
        )
        line, decision = only_decision(result.stderr, "case a")
        expected_line = (
            "picBunch: solve mode: auto -> in-core "
            f"(required {format_bytes(in_core_one)}, capacity {format_bytes(large)} "
            "over 1 rank, reason in-core-fits)"
        )
        if line != expected_line:
            print(f"case a status=FAIL decision {line!r} != {expected_line!r}")
            failures += 1
        else:
            print(f"case a status=OK decision={line}")

        # (b) A capacity between the chunked and in-core aggregates switches.
        # The override replaces the reported capacity for the decision, so a
        # larger value would not exercise a genuinely reduced budget.
        between = (chunked_one + in_core_one) // 2
        auto_dir = case_dir("b_auto_chunked")
        reference_dir = case_dir("b_reference_in_core")
        command = mpi_command(
            mpiexec,
            1,
            binary,
            [*common, "--destination", auto_dir, "--staging-dir", auto_dir + "-staging"],
        )
        result = run(
            command,
            "case b auto",
            {"PICBUNCH_TEST_MEMORY_BUDGET_BYTES": str(between)},
        )
        line, decision = only_decision(result.stderr, "case b auto")
        expected_line = (
            "picBunch: solve mode: auto -> chunked "
            f"(required {format_bytes(chunked_one)}, capacity {format_bytes(between)} "
            "over 1 rank, reason in-core-does-not-fit;chunked-fits)"
        )
        if line != expected_line or decision["mode"] != "auto -> chunked":
            print(f"case b status=FAIL decision {line!r} != {expected_line!r}")
            failures += 1
        else:
            print(f"case b status=OK decision={line}")
        reference = run(
            mpi_command(
                mpiexec,
                1,
                binary,
                [
                    *common,
                    "--solve-mode",
                    "in-core",
                    "--destination",
                    reference_dir,
                ],
            ),
            "case b reference",
        )
        if "reason explicit-in-core" not in reference.stderr:
            print(
                "case b status=FAIL reference did not resolve in-core: "
                + failure_tail(reference)
            )
            failures += 1
        else:
            worst, details = compare_fields(reference_dir, auto_dir)
            status = "OK" if worst <= TOL else "FAIL"
            print(f"case b fields worst={worst:.3e} status={status}")
            if worst > TOL:
                from compare_chunked_config_mpi import diff_summary

                print(diff_summary(reference_dir, auto_dir, details))
                failures += 1

        # (c) A capacity below the chunked aggregate refuses and names both.
        too_small = chunked_one // 2
        refused_destination = case_dir("c_refused")
        result = run_expect_failure(
            mpi_command(
                mpiexec,
                1,
                binary,
                [*common, "--destination", refused_destination],
            ),
            "case c",
            {"PICBUNCH_TEST_MEMORY_BUDGET_BYTES": str(too_small)},
        )
        message = result.stderr
        required_text = f"required {format_bytes(chunked_one)}"
        budget_text = f"capacity {format_bytes(too_small)} over 1 rank"
        if (
            result.returncode == 0
            or required_text not in message
            or budget_text not in message
            or "in-core-does-not-fit;chunked-exceeds-budget" not in message
        ):
            print(
                f"case c status=FAIL exit={result.returncode} "
                f"stderr={failure_tail(result)}"
            )
            failures += 1
        else:
            print(f"case c status=OK exit={result.returncode}")

        # (d) Two ranks, equal budgets whose node total lies between the
        # chunked and in-core aggregates: both ranks must agree on chunked.
        budget_two = ((chunked_two_total + in_core_two_total) // 2) // 2
        if not chunked_two_total < 2 * budget_two < in_core_two_total:
            raise SystemExit(
                "test setup: 2-rank budget total "
                f"{2 * budget_two} is not between chunked {chunked_two_total} and "
                f"in-core {in_core_two_total}"
            )
        destination = case_dir("d_two_ranks")
        result = run(
            mpi_command(
                mpiexec,
                2,
                binary,
                [*common, "--destination", destination, "--staging-dir", destination + "-staging"],
            ),
            "case d",
            {
                "PICBUNCH_TEST_MEMORY_BUDGET_BYTES": f"{budget_two},{budget_two}",
                "PICBUNCH_TEST_SOLVE_MODE_RANKS": "1",
            },
        )
        decisions = rank_decisions(result.stderr, "case d")
        if decisions[0] != decisions[1]:
            print(
                "case d status=FAIL ranks resolved different modes: "
                f"rank0={decisions[0]!r} rank1={decisions[1]!r}"
            )
            failures += 1
        elif not decisions[0].startswith("auto -> chunked"):
            print(f"case d status=FAIL unexpected shared decision {decisions[0]!r}")
            failures += 1
        elif "divergent capacity or solver state" in result.stderr:
            print("case d status=FAIL job reported divergent capacity")
            failures += 1
        else:
            print(f"case d status=OK both ranks: {decisions[0]}")

        # (e) Four ranks with equal budgets whose sum stays between the chunked
        # and in-core aggregates.  The auto decision must switch to chunked and
        # the field must match the four-rank in-core reference.
        budget_four = ((chunked_four_total + in_core_four_total) // 2) // 4
        if not chunked_four_total < 4 * budget_four < in_core_four_total:
            raise SystemExit(
                "test setup: 4-rank budget total "
                f"{4 * budget_four} is not between chunked {chunked_four_total} and "
                f"in-core {in_core_four_total}"
            )
        auto_four_dir = case_dir("e_auto_chunked")
        reference_four_dir = case_dir("e_reference_in_core")
        result = run(
            mpi_command(
                mpiexec,
                4,
                binary,
                [
                    *common,
                    "--destination",
                    auto_four_dir,
                    "--staging-dir",
                    auto_four_dir + "-staging",
                ],
            ),
            "case e auto",
            {"PICBUNCH_TEST_MEMORY_BUDGET_BYTES": ",".join([str(budget_four)] * 4)},
        )
        line, decision = only_decision(result.stderr, "case e auto")
        expected_line = (
            "picBunch: solve mode: auto -> chunked "
            f"(required {format_bytes(chunked_four_total)}, "
            f"capacity {format_bytes(4 * budget_four)} over 4 ranks, "
            "reason in-core-does-not-fit;chunked-fits)"
        )
        if line != expected_line or decision["mode"] != "auto -> chunked":
            print(f"case e status=FAIL decision {line!r} != {expected_line!r}")
            failures += 1
        else:
            print(f"case e status=OK decision={line}")
        reference = run(
            mpi_command(
                mpiexec,
                4,
                binary,
                [
                    *common,
                    "--solve-mode",
                    "in-core",
                    "--fft-mode",
                    "distributed",
                    "--destination",
                    reference_four_dir,
                ],
            ),
            "case e reference",
        )
        if "reason explicit-in-core" not in reference.stderr:
            print(
                "case e status=FAIL reference did not resolve in-core: "
                + failure_tail(reference)
            )
            failures += 1
        else:
            worst, details = compare_fields(reference_four_dir, auto_four_dir)
            status = "OK" if worst <= TOL else "FAIL"
            print(f"case e fields worst={worst:.3e} status={status}")
            if worst > TOL:
                from compare_chunked_config_mpi import diff_summary

                print(diff_summary(reference_four_dir, auto_four_dir, details))
                failures += 1

        # (f) The optional [solve] solve_mode TOML key selects chunked with no
        # CLI solve-mode flag.
        config_toml = os.path.join(root, "CFG-chunked.toml")
        write_config(config_toml, GRID)
        with open(config_toml, "a") as handle:
            handle.write('\n[solve]\nsolve_mode = "chunked"\n')
        toml_common = list(common_no_solve)
        toml_common[toml_common.index("--config") + 1] = config_toml
        destination = case_dir("f_toml_chunked")
        result = run(
            mpi_command(
                mpiexec,
                1,
                binary,
                [*toml_common, "--destination", destination,
                 "--staging-dir", destination + "-staging"],
            ),
            "case f",
        )
        if (
            "reason explicit-chunked" not in result.stderr
            or "picBunch: staged solve:" not in result.stderr
        ):
            print("case f status=FAIL TOML solve_mode did not select chunked: "
                  + failure_tail(result))
            failures += 1
        else:
            print("case f status=OK TOML solve_mode=chunked with no CLI flag")

        # (g) The CLI --solve-mode wins over the TOML value.
        config_toml_in_core = os.path.join(root, "CFG-in-core.toml")
        write_config(config_toml_in_core, GRID)
        with open(config_toml_in_core, "a") as handle:
            handle.write('\n[solve]\nsolve_mode = "in-core"\n')
        cli_common = list(common_no_solve)
        cli_common[cli_common.index("--config") + 1] = config_toml_in_core
        destination = case_dir("g_cli_wins")
        result = run(
            mpi_command(
                mpiexec,
                1,
                binary,
                [*cli_common, "--solve-mode", "chunked", "--destination", destination,
                 "--staging-dir", destination + "-staging"],
            ),
            "case g",
        )
        if (
            "reason explicit-chunked" not in result.stderr
            or "picBunch: staged solve:" not in result.stderr
            or "reason explicit-in-core" in result.stderr
        ):
            print("case g status=FAIL CLI did not override the TOML value: "
                  + failure_tail(result))
            failures += 1
        else:
            print("case g status=OK CLI solve_mode overrides the TOML value")

        # (h) Two ranks with asymmetric budgets: rank 0's budget covers its own
        # in-core requirement and the node total, so it resolves in-core, while
        # rank 1's budget does not cover its own in-core requirement, so the
        # local bound makes it resolve chunked (or refuse).  The root broadcast
        # then disagrees with rank 1 and the job must abort instead of hanging.
        # Both budgets keep the node total above the in-core aggregate, so the
        # aggregate criterion alone would let both ranks run in-core.
        local_in_core_two = in_core_requirement(
            GRID, PADDING, N_PARTICLES, 2, distributed=True
        )
        large_budget = 4 * local_in_core_two
        small_budget = chunked_two_total // 2
        if not local_in_core_two > small_budget:
            raise SystemExit(
                "test setup: rank-1 budget "
                f"{small_budget} does not fall below its local in-core requirement "
                f"{local_in_core_two}"
            )
        if not 2 * local_in_core_two <= large_budget + small_budget:
            raise SystemExit(
                "test setup: asymmetric budgets do not keep the node total above "
                "the in-core aggregate"
            )
        if not chunked_two_total <= large_budget + small_budget:
            raise SystemExit(
                "test setup: asymmetric budgets do not let rank 1 reach chunked"
            )
        destination = case_dir("h_divergent_budgets")
        try:
            result = run_expect_failure(
                mpi_command(
                    mpiexec,
                    2,
                    binary,
                    [
                        *common,
                        "--destination",
                        destination,
                        "--staging-dir",
                        destination + "-staging",
                    ],
                ),
                "case h",
                {
                    "PICBUNCH_TEST_MEMORY_BUDGET_BYTES": f"{large_budget},{small_budget}",
                    "PICBUNCH_TEST_SOLVE_MODE_RANKS": "1",
                },
            )
        except subprocess.TimeoutExpired:
            print("case h status=FAIL timed out after 600 s (deadlock)")
            failures += 1
        else:
            divergence = "the rank set has divergent capacity or solver state"
            if result.returncode == 0 or divergence not in result.stderr:
                print(
                    f"case h status=FAIL exit={result.returncode} "
                    f"stderr={failure_tail(result)}"
                )
                failures += 1
            else:
                print(
                    f"case h status=OK exit={result.returncode} "
                    f"message={divergence!r}"
                )

        return 1 if failures else 0
    finally:
        if keep:
            print(f"kept test output in {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
