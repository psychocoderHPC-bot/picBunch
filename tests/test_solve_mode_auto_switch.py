#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
"""End-to-end tests of the automatic solve-mode decision.

The automatic policy keeps the in-core solve when the per-rank estimate fits
and switches to the staged chunked solve when it does not.  Three cases pin
that decision against the per-rank memory budget:

* a budget far above the in-core requirement keeps the run in-core,
* a budget between the chunked and in-core requirements switches to chunked
  and must reproduce the in-core field to round-off,
* a budget below the chunked requirement refuses with required and budget
  bytes in the message.

A fourth case runs two ranks with unequal budgets.  The Allreduce must reconcile
the local decisions, so both ranks either pick the same mode or abort together;
the test captures each rank's local decision and requires them to be identical.

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


def chunked_requirement(grid, padding, particles, chunk_cells, rank_count):
    mx, my, mz = padded_extents(grid, padding)
    ranks = virtual_ranks((mx, my, mz), chunk_cells)
    nz = (mz + ranks - 1) // ranks
    nx = (mx + ranks - 1) // ranks
    workspace = 16 * (mx * my * nz + 2 * nx * my * mz + 2 * max(mx * nz, mz * nx))
    particle_bytes = particles * 2 * 8
    out_nz = grid[2]
    slab_planes = max(
        ((r + 1) * out_nz) // rank_count - (r * out_nz) // rank_count
        for r in range(rank_count)
    )
    plane_elems = grid[0] * grid[1]
    return workspace + particle_bytes + 6 * slab_planes * plane_elems * 8


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
        r"required ([0-9.]+) GiB, budget ([0-9.]+) GiB per rank, reason ([^)]+)",
        detail,
    )
    if fields is not None:
        decision["required"] = fields.group(1)
        decision["budget"] = fields.group(2)
        decision["reason"] = fields.group(3)
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


def run(command, label, env_extra=None):
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    result = subprocess.run(command, text=True, capture_output=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"{label}: command failed with exit {result.returncode}: "
            + failure_tail(result)
        )
    return result


def run_expect_failure(command, label, env_extra=None):
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    return subprocess.run(command, text=True, capture_output=True, env=env)


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
    chunked_two = chunked_requirement(GRID, PADDING, N_PARTICLES, CHUNK_CELLS, 2)
    in_core_two = in_core_requirement(GRID, PADDING, N_PARTICLES, 2, True)
    if not chunked_one < in_core_one:
        raise SystemExit(
            f"test setup: chunked ({chunked_one}) must stay below in-core ({in_core_one})"
        )
    if not chunked_two < in_core_two:
        raise SystemExit(
            f"test setup: 2-rank chunked ({chunked_two}) must stay below "
            f"in-core ({in_core_two})"
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
        expected_line = "picBunch: solve mode: auto -> in-core (reason in-core-fits)"
        if line != expected_line:
            print(f"case a status=FAIL decision {line!r} != {expected_line!r}")
            failures += 1
        else:
            print(f"case a status=OK decision={line}")

        # (b) A budget between the chunked and in-core requirements switches.
        # Precondition: ``between`` must stay below the real device capacity.
        # The decision budget is shareBytes = min(capacity, override), so a
        # larger override would not exercise a genuinely reduced budget.
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
            f"(required {format_bytes(chunked_one)}, budget {format_bytes(between)} "
            "per rank, reason in-core-does-not-fit;chunked-fits)"
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

        # (c) A budget below the chunked requirement refuses and names both.
        # Precondition: ``too_small`` must stay below the real device capacity.
        # The decision budget is shareBytes = min(capacity, override), so a
        # larger override would not exercise a genuinely reduced budget.
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
        required_text = f"required {format_bytes(chunked_one)} per rank"
        budget_text = f"budget {format_bytes(too_small)} per rank"
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

        # (d) Unequal budgets across two ranks: the Allreduce must reconcile.
        small_two = (chunked_two + in_core_two) // 2
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
                "PICBUNCH_TEST_MEMORY_BUDGET_BYTES": f"{large},{small_two}",
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

        return 1 if failures else 0
    finally:
        if keep:
            print(f"kept test output in {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
