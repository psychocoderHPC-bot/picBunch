#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
"""Compare the chunked solve on the --config path at one or more MPI ranks.

The reference is the in-core distributed FFT at the same rank count.  The
candidate is ``--solve-mode chunked`` with a deliberately small chunk buffer,
so a padded grid of a few thousand cells is split into many chunks/planes and
the staged transpose is exercised instead of a single pass.

The script first validates the reference itself: the rank-1 in-core field and
every multi-rank in-core field must agree.  Only then does it compare the
chunked checkpoints against the reference.  One line is printed per case; the
first mismatch (or failed run) fails the script unless
``PICBUNCH_CONTINUE_ON_MISMATCH=1`` is set, which keeps running so a baseline
matrix can be collected.  Every generated file, including the staging
directories, is removed at the end; ``PICBUNCH_KEEP_TEST_OUTPUT=1`` keeps the
per-run directory for debugging.

usage: compare_chunked_config_mpi.py <mpiexec> <picbunch-bin> <workdir>
"""
import os
import shutil
import subprocess
import sys

import h5py
import numpy as np

TOL = 1.0e-12
CELL = 1.0e-6
N_PARTICLES = 256
Q_E = 1.602176634e-19
M_E = 9.1093837015e-31

# Small grids that still exercise odd extents, an elongated z axis, and a
# non-cubic transverse plane.  Rank 3 is only used where it adds a non-power-
# of-two decomposition, as required by the matrix.
GRIDS = {
    "A": (16, 16, 16),
    "B": (11, 9, 13),
    "C": (8, 8, 17),
}
RANKS = {"A": (1, 4), "B": (1, 3, 4), "C": (1, 4)}
PADDINGS = (1, 2)
CHUNK_CELLS = (512, 4096)

DIM_POS = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
DIM_MOM = np.array([1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
DIM_SCALAR = np.zeros(7)
DIM_CHARGE = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
DIM_MASS = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def _series_attrs(f):
    """Write the openPMD 2.0 series attributes the reader requires.

    This is the h5py-only layout used by tools/e2e/gen_inputs.py; the test
    deliberately avoids the python openpmd binding so it also runs in a CPU
    CI job that did not install it.
    """
    f.attrs["openPMD"] = "2.0.0"
    f.attrs["openPMDextension"] = np.uint32(0)
    f.attrs["basePath"] = "/data/%T/"
    f.attrs["meshesPath"] = "meshes/"
    f.attrs["particlesPath"] = "particles/"
    f.attrs["iterationEncoding"] = "groupBased"
    f.attrs["iterationFormat"] = "/data/%T/"
    f.attrs["software"] = "picBunch-chunked-config-test"
    f.attrs["softwareVersion"] = "1.0"


def _vec_record(species, name, comps, unit_si, dim):
    rec = species.create_group(name)
    rec.attrs["unitDimension"] = dim
    rec.attrs["timeOffset"] = 0.0
    for cname, values in comps:
        ds = rec.create_dataset(cname, data=values, dtype=np.float64)
        ds.attrs["unitSI"] = float(unit_si)
        ds.attrs["unitDimension"] = dim
        ds.attrs["timeOffset"] = 0.0


def _scalar_record(species, name, values, unit_si, dim):
    rec = species.create_group(name)
    rec.attrs["unitSI"] = float(unit_si)
    rec.attrs["unitDimension"] = dim
    rec.attrs["timeOffset"] = 0.0
    ds = rec.create_dataset("value", data=values, dtype=np.float64)
    ds.attrs["unitSI"] = float(unit_si)
    ds.attrs["unitDimension"] = dim
    ds.attrs["timeOffset"] = 0.0


def write_input(path, grid, n=N_PARTICLES, cell=CELL):
    """Write a deterministic openPMD HDF5 bunch with h5py only.

    Positions are stored in cell units with ``unitSI = cell`` (the layout of
    tools/e2e/gen_inputs.py), so the physical bunch is centred in the requested
    grid.  Seed 7 and the 0.8-cell spread match tests/compare_chunked_mpi.py.
    """
    rng = np.random.default_rng(7)
    cells = np.asarray(grid, dtype=np.float64)
    pos = [0.5 * cells[a] + 0.8 * rng.standard_normal(n) for a in range(3)]
    zeros = np.zeros(n, dtype=np.float64)
    with h5py.File(path, "w") as f:
        _series_attrs(f)
        it = f.create_group("data/0")
        it.attrs["time"] = 0.0
        it.attrs["dt"] = 0.0
        it.attrs["timeUnitSI"] = 1.0
        sp = it.create_group("particles/electrons")
        sp.attrs["speciesType"] = np.uint32(1)
        _vec_record(
            sp,
            "position",
            [("x", pos[0]), ("y", pos[1]), ("z", pos[2])],
            cell,
            DIM_POS,
        )
        _vec_record(
            sp,
            "momentum",
            [("x", zeros), ("y", zeros), ("z", zeros)],
            1.0,
            DIM_MOM,
        )
        _scalar_record(sp, "weighting", np.ones(n, dtype=np.float64), 1.0, DIM_SCALAR)
        _scalar_record(
            sp, "charge", np.full(n, Q_E, dtype=np.float64), 1.0, DIM_CHARGE
        )
        _scalar_record(
            sp, "mass", np.ones(n, dtype=np.float64), M_E, DIM_MASS
        )


def write_config(path, grid, cell=CELL):
    with open(path, "w") as f:
        f.write(
            "[picongpu]\n"
            "delta_t = 1.0e-15\n"
            f"cell_size = [{cell!r}, {cell!r}, {cell!r}]\n"
            "base_density = 1.0e24\n"
            "step = 0\n"
            'precision = "float64"\n'
            'species = "electrons"\n'
            "\n"
            "[domain]\n"
            f"cells = [{grid[0]}, {grid[1]}, {grid[2]}]\n"
            "\n"
            "[placement]\n"
            'center = "centroid"\n'
            "position = [0.5, 0.5, 0.5]\n"
            'offset_unit = "fraction"\n'
            'patch = "domain"\n'
            "\n"
            "[accuracy]\n"
            "tolerance = 1.0e-3\n"
            'near_shell = "none"\n'
        )


def _is_openmpi(mpiexec):
    try:
        out = subprocess.run(
            [mpiexec, "--version"], text=True, capture_output=True
        ).stdout
    except OSError:
        return False
    return "Open MPI" in out or "OpenRTE" in out


def mpi_command(mpiexec, ranks, binary, arguments):
    """Build the launch command; rank 1 runs the binary directly."""
    if ranks == 1:
        return [binary, *arguments]
    command = [mpiexec]
    if os.geteuid() == 0 and _is_openmpi(mpiexec):
        command.append("--allow-run-as-root")
    command += ["-n", str(ranks), binary, *arguments]
    return command


def failure_tail(result):
    lines = [
        line.strip()
        for line in (result.stdout + result.stderr).splitlines()
        if line.strip()
    ]
    interesting = [line for line in lines if "picBunch:" in line]
    return " | ".join((interesting or lines)[-3:]) or "no output"


def run(command, label):
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{label}: command failed with exit {result.returncode}: "
            + failure_tail(result)
        )
    return result


def stderr_tail(result):
    lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
    return " | ".join(lines[-5:]) or "no stderr"


# Printed by the staged solver itself after it ran.  The solve-mode reason line
# is emitted from argv parsing before any work, so the reason alone cannot
# prove the chunked path executed.  The configured path does not call
# ``printTimings`` (that header is direct-path only), so it uses the staged
# solver's own line instead of "chunked solve (chunk=".
STAGED_MARKER = "picBunch: staged solve:"


def require_reason(result, marker, label):
    """Fail when the run did not log the expected solve-mode reason.

    The reference must resolve in-core and the candidate must honour the hard
    chunked override.  Without this check a memory-constrained runner could
    switch the reference to chunked on its own and compare chunked against
    chunked while still printing OK.
    """
    if marker in result.stderr:
        return None
    return f"{label}: stderr lacks {marker!r}: " + stderr_tail(result)


def require_chunked_path(result, expected, label):
    """Require the solver's own marker exactly where the mode says it ran."""
    present = STAGED_MARKER in result.stderr
    if present == expected:
        return None
    wanted = "must print" if expected else "must not print"
    return (
        f"{label}: solver output {wanted} {STAGED_MARKER!r} "
        f"(stderr={result.stderr[-200:]!r})"
    )


def _checkpoint(path):
    return os.path.join(path, "checkpoint_000000.h5")


def compare_fields(reference, candidate):
    """Return (worst_relative, details) over data/0/fields/{E,B}/{x,y,z}."""
    ref_path = _checkpoint(reference)
    cand_path = _checkpoint(candidate)
    for path in (ref_path, cand_path):
        if not os.path.isfile(path):
            raise RuntimeError(f"missing checkpoint {path}")
    worst = 0.0
    details = []
    with h5py.File(ref_path, "r") as ref, h5py.File(cand_path, "r") as cand:
        for record in ("E", "B"):
            for component in ("x", "y", "z"):
                key = f"data/0/fields/{record}/{component}"
                if key not in ref or key not in cand:
                    raise RuntimeError(f"missing dataset {key}")
                a = np.asarray(ref[key]).astype(np.float64)
                b = np.asarray(cand[key]).astype(np.float64)
                if a.shape != b.shape:
                    raise RuntimeError(f"{key}: shape {a.shape} != {b.shape}")
                scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-300)
                absolute = float(np.max(np.abs(a - b)))
                relative = absolute / scale
                details.append((key, relative, absolute, scale, a.shape))
                worst = max(worst, relative)
    return worst, details


def diff_summary(reference, candidate, details):
    lines = [f"  reference: {_checkpoint(reference)}",
             f"  candidate: {_checkpoint(candidate)}"]
    for key, relative, absolute, scale, shape in details:
        if relative > TOL:
            lines.append(
                f"  {key}: shape={shape} max|a-b|={absolute:.3e} "
                f"scale={scale:.3e} relative={relative:.3e}"
            )
    return "\n".join(lines)


def main():
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: compare_chunked_config_mpi.py <mpiexec> <picbunch-bin> <workdir>"
        )
    mpiexec, binary, workdir = sys.argv[1:4]
    binary = os.path.abspath(binary)
    mpiexec = (
        mpiexec
        if os.path.sep in mpiexec or os.path.isfile(mpiexec)
        else shutil.which(mpiexec) or mpiexec
    )
    workdir = os.path.abspath(workdir)
    root = os.path.join(workdir, f"compare_chunked_config_mpi-{os.getpid()}")
    keep = os.environ.get("PICBUNCH_KEEP_TEST_OUTPUT") == "1"
    continue_on_mismatch = os.environ.get("PICBUNCH_CONTINUE_ON_MISMATCH") == "1"

    os.makedirs(root, exist_ok=True)
    failures = 0
    try:
        inputs = {}
        configs = {}
        for name, grid in GRIDS.items():
            case_root = os.path.join(root, name)
            os.makedirs(case_root, exist_ok=True)
            input_path = os.path.join(case_root, "IN.h5")
            config_path = os.path.join(case_root, "CFG.toml")
            write_input(input_path, grid)
            write_config(config_path, grid)
            inputs[name] = input_path
            configs[name] = config_path

        for name, grid in GRIDS.items():
            for padding in PADDINGS:
                tag = f"grid={name} pad={padding}"
                common = [
                    "--input",
                    inputs[name],
                    "--config",
                    configs[name],
                    "--fft-padding",
                    str(padding),
                    "--deterministic-deposit",
                    "--shape",
                    "tsc",
                    "--input-iteration",
                    "0",
                ]
                references = {}
                for ranks in RANKS[name]:
                    reference = os.path.join(
                        root, name, f"pad{padding}", f"ref_r{ranks}"
                    )
                    os.makedirs(reference, exist_ok=True)
                    try:
                        result = run(
                            mpi_command(
                                mpiexec,
                                ranks,
                                binary,
                                [
                                    *common,
                                    "--destination",
                                    reference,
                                    "--fft-mode",
                                    "distributed",
                                    "--solve-mode",
                                    "in-core",
                                ],
                            ),
                            f"reference {tag} ranks={ranks}",
                        )
                    except RuntimeError as error:
                        print(f"refcheck {tag} ranks=1_vs_{ranks} FAIL {error}")
                        return 1
                    mismatch = require_reason(
                        result,
                        "reason explicit-in-core",
                        f"reference {tag} ranks={ranks}",
                    )
                    if mismatch is None:
                        mismatch = require_chunked_path(
                            result, False, f"reference {tag} ranks={ranks}"
                        )
                    if mismatch is not None:
                        print(f"refcheck {tag} ranks=1_vs_{ranks} FAIL {mismatch}")
                        return 1
                    references[ranks] = reference

                for ranks in RANKS[name]:
                    if ranks == 1:
                        continue
                    try:
                        worst, details = compare_fields(
                            references[1], references[ranks]
                        )
                    except RuntimeError as error:
                        print(f"refcheck {tag} ranks=1_vs_{ranks} FAIL {error}")
                        return 1
                    status = "OK" if worst <= TOL else "FAIL"
                    print(
                        f"refcheck {tag} ranks=1_vs_{ranks} worst={worst:.3e} "
                        f"status={status}"
                    )
                    if worst > TOL:
                        print(diff_summary(references[1], references[ranks], details))
                        return 1

                for ranks in RANKS[name]:
                    for chunk in CHUNK_CELLS:
                        label = f"grid={name} pad={padding} ranks={ranks} chunk={chunk}"
                        candidate = os.path.join(
                            root, name, f"pad{padding}", f"chunk_r{ranks}_c{chunk}"
                        )
                        staging = os.path.join(candidate + "-staging")
                        os.makedirs(candidate, exist_ok=True)
                        os.makedirs(staging, exist_ok=True)
                        try:
                            result = run(
                                mpi_command(
                                    mpiexec,
                                    ranks,
                                    binary,
                                    [
                                        *common,
                                        "--destination",
                                        candidate,
                                        "--solve-mode",
                                        "chunked",
                                        "--chunk-cells",
                                        str(chunk),
                                        "--staging-dir",
                                        staging,
                                    ],
                                ),
                                label,
                            )
                        except RuntimeError as error:
                            print(f"case {label} status=FAIL {error}")
                            failures += 1
                            if not continue_on_mismatch:
                                return 1
                            continue
                        mismatch = require_reason(
                            result, "reason explicit-chunked", label
                        )
                        if mismatch is None:
                            mismatch = require_chunked_path(result, True, label)
                        if mismatch is not None:
                            print(f"case {label} status=FAIL {mismatch}")
                            failures += 1
                            if not continue_on_mismatch:
                                return 1
                            continue
                        try:
                            worst, details = compare_fields(
                                references[ranks], candidate
                            )
                        except RuntimeError as error:
                            print(f"case {label} status=FAIL {error}")
                            failures += 1
                            if not continue_on_mismatch:
                                return 1
                            continue
                        status = "OK" if worst <= TOL else "FAIL"
                        print(f"case {label} worst={worst:.3e} status={status}")
                        if worst > TOL:
                            print(diff_summary(references[ranks], candidate, details))
                            failures += 1
                            if not continue_on_mismatch:
                                return 1
        return 1 if failures else 0
    finally:
        if keep:
            print(f"kept test output in {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
