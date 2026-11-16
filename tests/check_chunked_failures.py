#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
"""Assert the staged-failure diagnostics for the truncation and read-only cases.

The first case removes the final y-slice from the forward exchange file and
requires the missing-slice message.  The second case points the staging
directory at a directory without write permission and requires the file-create
message.  Both runs must fail; a successful run is a test failure.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

import openpmd_api as io


def write_input(path, n=256, cell=1.0e-6, grid=16):
    rng = np.random.default_rng(7)
    pos = 0.5 * grid * cell + 0.8 * cell * rng.standard_normal((3, n))
    weight = np.ones(n)
    charge = np.full(n, 1.602176634e-19)
    zero = np.zeros(n)
    series = io.Series(path, io.Access.create)
    it = series.iterations[0]
    it.time = 0.0
    it.time_unit_SI = 1.0
    sp = it.particles["electrons"]
    dims = {
        "position": {io.Unit_Dimension.L: 1},
        "positionOffset": {io.Unit_Dimension.L: 1},
        "momentum": {
            io.Unit_Dimension.M: 1,
            io.Unit_Dimension.L: 1,
            io.Unit_Dimension.T: -1,
        },
        "weighting": {io.Unit_Dimension.L: 0},
        "charge": {io.Unit_Dimension.I: 1, io.Unit_Dimension.T: 1},
    }
    for name, data in [
        ("position", pos),
        ("positionOffset", np.zeros((3, n))),
        ("momentum", np.stack([zero, zero, zero])),
        ("weighting", weight),
        ("charge", charge),
    ]:
        rec = sp[name]
        rec.unit_dimension = dims[name]
        if name in ("position", "positionOffset", "momentum"):
            for a, comp in enumerate(["x", "y", "z"]):
                c = rec[comp]
                c.unit_SI = 1.0
                c.reset_dataset(io.Dataset(data[a].dtype, [n]))
                c.store_chunk(np.ascontiguousarray(data[a]))
        else:
            rec.unit_SI = 1.0
            rec.reset_dataset(io.Dataset(data.dtype, [n]))
            rec.store_chunk(data)
    series.flush()
    series.close()


def run_case(binary, workdir, name, env_extra, staging_dir, expected):
    out = os.path.join(workdir, name + ".h5")
    cmd = [
        binary,
        "--input",
        os.path.join(workdir, "chunked_failure_input.h5"),
        "--output",
        out,
        "--grid",
        "16,16,16",
        "--origin",
        "0,0,0",
        "--cell-size",
        "1e-6,1e-6,1e-6",
        "--fft-padding",
        "2",
        "--deterministic-deposit",
        "--solve-mode",
        "chunked",
        "--virtual-ranks",
        "4",
        "--chunk-cells",
        "4096",
        "--staging-dir",
        staging_dir,
    ]
    env = os.environ.copy()
    env.update(env_extra)
    result = subprocess.run(cmd, text=True, capture_output=True, env=env)
    text = result.stdout + result.stderr
    if result.returncode == 0:
        raise SystemExit(f"{name}: run unexpectedly succeeded\n{text}")
    if not re.search(expected, text):
        raise SystemExit(f"{name}: expected diagnostic not found\n{text}")
    if "staged failure: phase=source" not in text:
        raise SystemExit(f"{name}: staged phase diagnostic not found\n{text}")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: check_chunked_failures.py <picbunch> <workdir>")
    binary, workdir = sys.argv[1:3]
    os.makedirs(workdir, exist_ok=True)
    write_input(os.path.join(workdir, "chunked_failure_input.h5"))

    truncate_dir = os.path.join(workdir, "chunked_failure_truncate")
    shutil.rmtree(truncate_dir, ignore_errors=True)
    os.makedirs(truncate_dir)
    run_case(
        binary,
        workdir,
        "truncate",
        {"PICBUNCH_TEST_TRUNCATE_STAGING": "1"},
        truncate_dir,
        r"picBunch: error: staged exchange: missing y-slice 31 in '[^']*"
        r"source-forward-0\.bin': requested bytes \[\d+, \d+\), file holds \d+ bytes",
    )

    read_only_dir = os.path.join(workdir, "chunked_failure_read_only")
    shutil.rmtree(read_only_dir, ignore_errors=True)
    os.makedirs(read_only_dir)
    os.chmod(read_only_dir, 0o555)
    try:
        if os.geteuid() == 0:
            print("chunked failure diagnostics: read-only case skipped as root")
            return
        run_case(
            binary,
            workdir,
            "read_only",
            {},
            read_only_dir,
            r"picBunch: error: staging: cannot create '[^']*"
            r"source-forward-0\.bin': Permission denied",
        )
    finally:
        os.chmod(read_only_dir, 0o755)
    print("chunked failure diagnostics: ok")


if __name__ == "__main__":
    main()
