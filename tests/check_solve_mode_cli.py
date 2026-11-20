#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
"""Exercise the --solve-mode aliases on the direct --input/--output path.

No MPI is needed.  A tiny deterministic bunch is deposited once per mode; the
default (no flag), ``auto``, ``none`` and ``in-core`` must all produce the
same six field components.  ``chunked`` must also run at one rank and match
the in-core field on a host build; a device build must refuse it with the
host/CPU-only message, which is treated as the expected outcome.  The last
three checks cover the CLI contract of the
option itself: a bogus value exits 2 with all four accepted spellings in its
stderr, ``incore`` exits 2 as a removed alias, and ``--help`` documents
exactly the four accepted spellings.

Every generated file is removed at the end; ``PICBUNCH_KEEP_TEST_OUTPUT=1``
keeps the per-run directory for debugging.

usage: check_solve_mode_cli.py <picbunch-bin> <workdir>
"""
import os
import re
import shutil
import subprocess
import sys

import h5py
import numpy as np

TOL = 1.0e-12
CELL = 1.0e-6
# The staged chunked solve hands host buffers to the FFT backend, so a
# CUDA/HIP build refuses it at runtime.  That refusal is the expected result
# on a device build; any other failure still fails the test.
DEVICE_CHUNKED_REFUSAL = (
    "chunked solve is implemented for the host/CPU backend only; "
    "the staging passes hand host buffers to the FFT backend"
)
# Printed by ``printTimings`` only after the staged solver actually ran.  The
# solve-mode decision line is emitted from argv parsing before any work, so a
# binary that prints "explicit-chunked" and then solves in-core would pass a
# reason-only check.
CHUNKED_MARKER = "chunked solve (chunk="
N_PARTICLES = 128
Q_E = 1.602176634e-19
M_E = 9.1093837015e-31
GRID = (8, 8, 8)
SPELLINGS = ("auto", "chunked", "none", "in-core")

DIM_POS = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
DIM_MOM = np.array([1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
DIM_SCALAR = np.zeros(7)
DIM_CHARGE = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
DIM_MASS = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def _series_attrs(f):
    """h5py-only openPMD 2.0 series layout (see tools/e2e/gen_inputs.py)."""
    f.attrs["openPMD"] = "2.0.0"
    f.attrs["openPMDextension"] = np.uint32(0)
    f.attrs["basePath"] = "/data/%T/"
    f.attrs["meshesPath"] = "meshes/"
    f.attrs["particlesPath"] = "particles/"
    f.attrs["iterationEncoding"] = "groupBased"
    f.attrs["iterationFormat"] = "/data/%T/"
    f.attrs["software"] = "picBunch-solve-mode-cli-test"
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


def write_input(path, grid=GRID, n=N_PARTICLES, cell=CELL):
    """Write a deterministic openPMD HDF5 bunch without the python binding."""
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


def run(command):
    return subprocess.run(command, text=True, capture_output=True)


def is_device_chunked_refusal(result):
    """True only for the host/CPU-only refusal a device build must produce."""
    return (
        result.returncode != 0
        and DEVICE_CHUNKED_REFUSAL in (result.stdout + result.stderr)
    )


def failure_tail(result):
    lines = [
        line.strip()
        for line in (result.stdout + result.stderr).splitlines()
        if line.strip()
    ]
    interesting = [line for line in lines if "picBunch:" in line]
    return " | ".join((interesting or lines)[-3:]) or "no output"


def compare_meshes(reference, candidate):
    """Return the worst relative difference over data/0/meshes/{E,B}/{x,y,z}."""
    worst = 0.0
    details = []
    with h5py.File(reference, "r") as ref, h5py.File(candidate, "r") as cand:
        for record in ("E", "B"):
            for component in ("x", "y", "z"):
                key = f"data/0/meshes/{record}/{component}"
                if key not in ref or key not in cand:
                    raise RuntimeError(f"missing dataset {key}")
                a = np.asarray(ref[key]).astype(np.float64)
                b = np.asarray(cand[key]).astype(np.float64)
                if a.shape != b.shape:
                    raise RuntimeError(f"{key}: shape {a.shape} != {b.shape}")
                scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-300)
                absolute = float(np.max(np.abs(a - b)))
                relative = absolute / scale
                details.append((key, relative, absolute, scale))
                worst = max(worst, relative)
    return worst, details


def solve_mode_section(text):
    """Text from the --solve-mode help entry up to the next option."""
    match = re.search(r"--solve-mode\b", text)
    if match is None:
        return ""
    tail = text[match.start():]
    next_option = re.search(r"(?m)^\s*--[A-Za-z]", tail[len("--solve-mode"):])
    if next_option is None:
        return tail
    return tail[: len("--solve-mode") + next_option.start()]


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: check_solve_mode_cli.py <picbunch-bin> <workdir>")
    binary = os.path.abspath(sys.argv[1])
    workdir = os.path.abspath(sys.argv[2])
    root = os.path.join(workdir, f"check_solve_mode_cli-{os.getpid()}")
    keep = os.environ.get("PICBUNCH_KEEP_TEST_OUTPUT") == "1"
    os.makedirs(root, exist_ok=True)
    failures = 0
    try:
        input_path = os.path.join(root, "IN.h5")
        write_input(input_path)
        common = [
            "--input",
            input_path,
            "--grid",
            "8,8,8",
            "--origin",
            "0,0,0",
            "--cell-size",
            "1e-6,1e-6,1e-6",
            "--fft-padding",
            "1",
            "--deterministic-deposit",
        ]

        outputs = {}
        device_chunked_refused = False
        modes = (None, "auto", "none", "in-core", "chunked")
        for mode in modes:
            name = "default" if mode is None else mode
            output = os.path.join(root, f"{name}.h5")
            command = [binary, *common, "--output", output]
            if mode is not None:
                command += ["--solve-mode", mode]
            if mode == "chunked":
                command += [
                    "--chunk-cells",
                    "512",
                    "--staging-dir",
                    os.path.join(root, "staging-chunked"),
                ]
            result = run(command)
            label = f"mode={name}"
            if result.returncode != 0:
                if mode == "chunked" and is_device_chunked_refusal(result):
                    device_chunked_refused = True
                    print(
                        f"run {label} status=EXPECTED-REFUSAL exit={result.returncode} "
                        "device build refuses chunked by design: "
                        + DEVICE_CHUNKED_REFUSAL
                    )
                    continue
                print(
                    f"run {label} status=FAIL exit={result.returncode} "
                    + failure_tail(result)
                )
                failures += 1
                continue
            ran_chunked = CHUNKED_MARKER in result.stdout
            if ran_chunked != (mode == "chunked"):
                wanted = "must print" if mode == "chunked" else "must not print"
                print(
                    f"run {label} status=FAIL solver output {wanted} "
                    f"{CHUNKED_MARKER!r}"
                )
                failures += 1
                continue
            outputs[name] = output
            print(f"run {label} status=OK")

        reference = outputs.get("default")
        for mode in ("auto", "none", "in-core", "chunked"):
            if mode == "chunked" and device_chunked_refused:
                print(
                    "compare default_vs_chunked status=NOT-APPLICABLE "
                    "device build refuses chunked by design"
                )
                continue
            if reference is None or mode not in outputs:
                print(f"compare default_vs_{mode} status=SKIP missing run")
                failures += 1
                continue
            try:
                worst, details = compare_meshes(reference, outputs[mode])
            except RuntimeError as error:
                print(f"compare default_vs_{mode} status=FAIL {error}")
                failures += 1
                continue
            status = "OK" if worst <= TOL else "FAIL"
            print(f"compare default_vs_{mode} worst={worst:.3e} status={status}")
            if worst > TOL:
                for key, relative, absolute, scale in details:
                    if relative > TOL:
                        print(
                            f"  {key}: max|a-b|={absolute:.3e} "
                            f"scale={scale:.3e} relative={relative:.3e}"
                        )
                failures += 1

        bogus = run([binary, "--input", input_path, "--solve-mode", "bogus"])
        missing = [word for word in SPELLINGS if word not in bogus.stderr]
        if bogus.returncode != 2:
            print(
                f"bogus status=FAIL exit={bogus.returncode} expected=2"
            )
            failures += 1
        elif missing:
            print(
                "bogus status=FAIL stderr lacks " + ", ".join(missing)
            )
            failures += 1
        else:
            print("bogus status=OK exit=2 stderr lists all four spellings")

        # ``incore`` was a removed alias; keep it pinned as a rejection.
        incore = run([binary, "--input", input_path, "--solve-mode", "incore"])
        missing = [word for word in SPELLINGS if word not in incore.stderr]
        if incore.returncode != 2:
            print(
                f"incore status=FAIL exit={incore.returncode} expected=2"
            )
            failures += 1
        elif missing:
            print(
                "incore status=FAIL stderr lacks " + ", ".join(missing)
            )
            failures += 1
        else:
            print("incore status=OK exit=2 stderr lists all four spellings")

        helped = run([binary, "--help"])
        section = solve_mode_section(helped.stdout + helped.stderr)
        missing = [word for word in SPELLINGS if word not in section]
        stale = "incore" in section
        if helped.returncode != 0:
            print(f"help status=FAIL exit={helped.returncode}")
            failures += 1
        elif missing or stale:
            problems = []
            if missing:
                problems.append("lacks " + ", ".join(missing))
            if stale:
                problems.append("still lists incore")
            print("help status=FAIL --solve-mode section " + "; ".join(problems))
            failures += 1
        else:
            print(
                "help status=OK --solve-mode section lists exactly the four "
                "accepted spellings"
            )

        return 1 if failures else 0
    finally:
        if keep:
            print(f"kept test output in {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
