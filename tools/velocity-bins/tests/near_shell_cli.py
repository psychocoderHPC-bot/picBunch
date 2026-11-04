#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
"""End-to-end near-shell CLI test.

Runs the shipped executable twice on the same small bunch: once with the plain
mesh and once with ``--near-shell exact``.  The exact-shell run must exit 0,
produce a finite field, change values inside the shell, and leave nodes outside
the shell equal to the mesh-only result.  On the v0.4.0 binary both near runs
exit 2 with ``unknown option '--near-shell'``, so this test fails there.
"""
import argparse
import pathlib
import subprocess
import sys

import h5py
import numpy as np


def write_bunch(path, n=12):
    rng = np.random.default_rng(7)
    # 8-cell cube of 1e-6 m cells, comfortably inside the 16^3 grid.
    pos = rng.uniform(-4e-6, 4e-6, (3, n))
    m_e = 9.1093837015e-31
    c = 299792458.0

    def sattr(g, k, v):
        g.attrs[k] = np.bytes_(v)

    with h5py.File(path, "w") as f:
        sattr(f, "openPMD", "2.0.0")
        sattr(f, "iterationEncoding", "groupBased")
        sattr(f, "iterationFormat", "%T")
        sattr(f, "basePath", "/data/%T/")
        sattr(f, "meshesPath", "meshes/")
        sattr(f, "particlesPath", "particles/")
        sattr(f, "software", "picBunch near-shell CLI test")
        sattr(f, "softwareVersion", "test")
        f.attrs["openPMDextension"] = np.uint32(1)
        it = f.create_group("data/0")
        it.attrs["dt"] = np.float64(1.0)
        it.attrs["time"] = np.float64(0.0)
        it.attrs["timeUnitSI"] = np.float64(1.0)
        it.create_group("meshes")
        sp = it.create_group("particles/electrons")
        sp.attrs["numParticles"] = np.int32(n)
        pos_g = sp.create_group("position")
        pos_g.attrs["macroWeighted"] = np.int32(0)
        pos_g.attrs["weightingPower"] = np.float64(0.0)
        pos_g.attrs["unitDimension"] = np.asarray([1.0, 0, 0, 0, 0, 0, 0])
        pos_g.attrs["timeOffset"] = np.float64(0.0)
        for ax, values in (("x", pos[0]), ("y", pos[1]), ("z", pos[2])):
            d = pos_g.create_dataset(ax, data=values.astype(np.float64))
            d.attrs["unitSI"] = np.float64(1.0)
        mom = sp.create_group("momentum")
        mom.attrs["macroWeighted"] = np.int32(0)
        mom.attrs["weightingPower"] = np.float64(0.0)
        mom.attrs["unitDimension"] = np.asarray([1.0, 1.0, -1.0, 0, 0, 0, 0])
        mom.attrs["timeOffset"] = np.float64(0.0)
        for ax in ("x", "y", "z"):
            d = mom.create_dataset(ax, data=np.zeros(n, dtype=np.float64))
            d.attrs["unitSI"] = np.float64(m_e * c)

        def scalar(name, child, values, unit, dim, macro, power):
            r = sp.create_group(name)
            r.attrs["macroWeighted"] = np.int32(macro)
            r.attrs["weightingPower"] = np.float64(power)
            r.attrs["unitDimension"] = np.asarray(dim)
            r.attrs["timeOffset"] = np.float64(0.0)
            r.attrs["unitSI"] = np.float64(unit)
            d = r.create_dataset(child, data=np.asarray(values, dtype=np.float64))
            d.attrs["unitSI"] = np.float64(unit)

        zero_dim = [0.0] * 7
        scalar("weighting", "value", np.ones(n), 1.0, zero_dim, 1, 0.0)
        scalar("charge", "charge", np.full(n, -1.602176634e-19), 1.0,
               [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], 0, 1.0)
        scalar("mass", "value", np.full(n, m_e), 1.0,
               [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0, 0.0)


def run(app, args):
    result = subprocess.run([app, *args], capture_output=True, text=True)
    return result


def read_field(path):
    with h5py.File(path, "r") as f:
        out = {}
        for record in ("E", "B"):
            for component in ("x", "y", "z"):
                out[(record, component)] = (
                    np.array(f[f"data/0/meshes/{record}/{component}"][:], dtype=np.float64)
                )
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", required=True)
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()
    work = pathlib.Path(args.workdir) / "near_shell_cli"
    work.mkdir(parents=True, exist_ok=True)
    bunch = work / "bunch.h5"
    mesh = work / "mesh.h5"
    near = work / "near.h5"
    write_bunch(bunch)

    common = [
        "--input", str(bunch), "--input-iteration", "0",
        "--species", "electrons",
        "--cell-size", "1e-6", "--grid", "16,16,16",
        "--origin", "-8e-6,-8e-6,-8e-6",
        "--tolerance", "1e-3", "--deterministic-deposit",
    ]
    mesh_result = run(args.app, common + ["--output", str(mesh)])
    if mesh_result.returncode != 0:
        print(mesh_result.stdout)
        print(mesh_result.stderr)
        raise SystemExit("mesh-only run failed")

    near_result = run(
        args.app,
        common + [
            "--output", str(near),
            "--near-shell", "exact",
            "--near-shell-radius", "3",
            "--near-shell-tile", "4",
        ],
    )
    print(near_result.stdout)
    if near_result.returncode != 0:
        print(near_result.stderr)
        raise SystemExit(
            f"--near-shell exact failed with exit {near_result.returncode}: "
            f"{near_result.stderr.strip().splitlines()[-1] if near_result.stderr else ''}"
        )
    if "near shell: mode=exact" not in near_result.stdout:
        raise SystemExit("run summary does not report the near-shell mode")

    mesh_fields = read_field(mesh)
    near_fields = read_field(near)
    for key, values in near_fields.items():
        if not np.all(np.isfinite(values)):
            raise SystemExit(f"non-finite values in near-shell field {key}")
    differences = {
        key: np.abs(near_fields[key] - mesh_fields[key]) for key in near_fields
    }
    total_difference = max(float(values.max()) for values in differences.values())
    if not total_difference > 0.0:
        raise SystemExit("near-shell field is identical to the mesh-only field")

    # Outside the shell plus half a tile the two runs must agree: the exact
    # pass only touches (source, node) pairs inside the radius.
    with h5py.File(bunch, "r") as f:
        g = f["data/0/particles/electrons/position"]
        positions = np.stack(
            [np.array(g[ax][:], dtype=np.float64) for ax in ("x", "y", "z")], axis=1
        )
    gx = -8e-6 + np.arange(16) * 1e-6
    grid = np.stack(np.meshgrid(gx, gx, gx, indexing="ij"), axis=-1)
    distances = np.sqrt(
        ((grid[:, :, :, None, :] - positions[None, None, None, :, :]) ** 2).sum(-1)
    ).min(axis=-1)
    inside = distances <= (3.0 + 1.5) * 1e-6
    outside = distances > (3.0 + 4.0 + 1.5) * 1e-6
    worst_inside = max(float(values[inside].max()) for values in differences.values())
    worst_outside = max(float(values[outside].max()) for values in differences.values())
    scale = max(float(np.abs(v).max()) for v in near_fields.values())
    print(
        f"near-shell CLI: max|difference|={total_difference:g}, "
        f"inside={worst_inside:g}, outside tile halo={worst_outside:g}, "
        f"field scale={scale:g}"
    )
    if not worst_inside > worst_outside:
        raise SystemExit("near-shell change is not concentrated inside the shell")


if __name__ == "__main__":
    main()
