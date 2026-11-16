#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
"""Compare the staged solve at N virtual ranks with N real MPI ranks.

The staged path reuses DistributedFft3D and its pack/unpack arithmetic; only
the MPI_Alltoallv call is replaced by the file exchange.  This test runs the
same small case through both paths and requires the six components to agree to
FFT round-off.
"""
import os
import subprocess
import sys

import h5py
import numpy as np

import openpmd_api as io


def write_input(path, n=2048, cell=1.0e-6, grid=16):
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


def run(cmd):
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(f"command failed: {' '.join(cmd)}")


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: compare_chunked_mpi.py <mpiexec> <picbunch> <workdir>")
    mpiexec, binary, workdir = sys.argv[1:4]
    os.makedirs(workdir, exist_ok=True)
    input_path = os.path.join(workdir, "chunked_mpi_input.h5")
    mpi_path = os.path.join(workdir, "chunked_mpi_real.h5")
    virt_path = os.path.join(workdir, "chunked_mpi_virtual.h5")
    write_input(input_path)

    common = [
        binary,
        "--input",
        input_path,
        "--grid",
        "16,16,16",
        "--origin",
        "0,0,0",
        "--cell-size",
        "1e-6,1e-6,1e-6",
        "--fft-padding",
        "2",
        "--deterministic-deposit",
    ]
    run([mpiexec, "-n", "4", *common, "--output", mpi_path, "--fft-mode", "distributed"])
    run(
        [
            *common,
            "--output",
            virt_path,
            "--solve-mode",
            "chunked",
            "--virtual-ranks",
            "4",
            "--chunk-cells",
            "4096",
            "--staging-dir",
            os.path.join(workdir, "chunked_mpi_staging"),
        ]
    )

    worst = 0.0
    with h5py.File(mpi_path, "r") as real, h5py.File(virt_path, "r") as virt:
        for record in ("E", "B"):
            for component in ("x", "y", "z"):
                key = f"data/0/meshes/{record}/{component}"
                a = np.asarray(real[key])
                b = np.asarray(virt[key])
                if a.shape != b.shape:
                    raise SystemExit(f"{key}: shape {a.shape} != {b.shape}")
                scale = max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-300)
                worst = max(worst, float(np.max(np.abs(a - b))) / scale)
    print(f"virtual-vs-real-rank worst relative difference: {worst:.3e}")
    if worst > 1.0e-12:
        raise SystemExit("staged solve does not match the MPI distributed path")
    for path in (input_path, mpi_path, virt_path):
        try:
            os.remove(path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
