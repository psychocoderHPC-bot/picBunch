#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
"""Generate bunch-only openPMD 2.0 HDF5 inputs for the end-to-end campaign.

The original campaign inputs were not checked into the repository.  This
script is the reconstruction used by ``tools/e2e/README.md``: it mirrors the
H100 campaign generator (gamma=10, 1 mrad divergence, uniform positions in a
20-cell box) and writes the two file families the harness expects:

* ``input_N<N>.h5`` for the main table (cell 1e-6 m, positions 0..20 cells);
* ``input_N2000000_cost.h5`` for the cost table (cell 2.5e-7 m, a 20-cell
  bunch centred in the 66x68x66 grid; the grid origin is inferred, see
  README).

The openPMD positions are stored in ``unitSI``-scaled units with no
``positionOffset`` record, so the reader maps a stored value ``p`` to the
physical coordinate ``p * unitSI``. The cost drawer therefore stores plain
``uniform(0, 20)`` cell indices, which places the bunch at physical
``[0, 20*cell)``. With the grid origin ``COST_ORIGIN_M`` that is the interior
of the 66x68x66 grid, centred in physical space (the earlier ``(grid-20)/2``
offset assumed an origin of 0 and pinned the bunch to the far corner).

Requires numpy and h5py in the interpreter that runs it.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import h5py

M_E = 9.1093837015e-31
C = 299792458.0
Q_E = 1.602176634e-19

DIM_POS = np.array([1.0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
DIM_MOM = np.array([1.0, 1.0, -1.0, 0, 0, 0, 0], dtype=np.float64)
DIM_SCALAR = np.zeros(7, dtype=np.float64)
DIM_CHARGE = np.array([0.0, 0.0, 1.0, 1.0, 0, 0, 0], dtype=np.float64)
DIM_MASS = np.array([0.0, 1.0, 0, 0, 0, 0, 0], dtype=np.float64)

# Cost-table geometry.  The grid origin is a CLI argument (--origin) and is not
# stored in the openPMD file, so it lives here next to the drawer that has to
# agree with it.  A 20-cell bunch stored as uniform(0, 20) * COST_CELL_M maps to
# physical [0, 20 * COST_CELL_M), whose centroid equals the grid centroid
# COST_ORIGIN_M + 0.5 * grid * COST_CELL_M.
COST_N = 2_000_000
COST_GRID = (66, 68, 66)
COST_CELL_M = 2.5e-7
COST_ORIGIN_M = (-5.75e-6, -6.0e-6, -5.75e-6)
COST_BUNCH_CELLS = 20.0


def series_attrs(f):
    f.attrs["openPMD"] = "2.0.0"
    f.attrs["openPMDextension"] = np.uint32(0)
    f.attrs["basePath"] = "/data/%T/"
    f.attrs["meshesPath"] = "meshes/"
    f.attrs["particlesPath"] = "particles/"
    f.attrs["iterationEncoding"] = "groupBased"
    f.attrs["iterationFormat"] = "/data/%T/"
    f.attrs["software"] = "picBunch-e2e"
    f.attrs["softwareVersion"] = "1.0"


def vec_record(species, name, comps, unit_si, dim):
    rec = species.create_group(name)
    rec.attrs["unitDimension"] = dim
    rec.attrs["timeOffset"] = 0.0
    for cname, values in comps:
        ds = rec.create_dataset(cname, data=values, dtype=np.float64)
        ds.attrs["unitSI"] = float(unit_si)
        ds.attrs["unitDimension"] = dim
        ds.attrs["timeOffset"] = 0.0


def scalar_record(species, name, value, unit_si, dim):
    rec = species.create_group(name)
    rec.attrs["unitSI"] = float(unit_si)
    rec.attrs["unitDimension"] = dim
    rec.attrs["timeOffset"] = 0.0
    ds = rec.create_dataset("value", data=value, dtype=np.float64)
    ds.attrs["unitSI"] = float(unit_si)
    ds.attrs["unitDimension"] = dim
    ds.attrs["timeOffset"] = 0.0


def write_bunch(path, n, cell_m, ux, uy, uz, pos_frac):
    rng = np.random.default_rng(1234 + n)
    with h5py.File(path, "w") as f:
        series_attrs(f)
        it = f.create_group("data/0")
        it.attrs["time"] = 0.0
        it.attrs["dt"] = 0.0
        it.attrs["timeUnitSI"] = 1.0
        sp = it.create_group("particles/e")
        sp.attrs["speciesType"] = np.uint32(1)
        vec_record(
            sp,
            "position",
            [("x", pos_frac[0]), ("y", pos_frac[1]), ("z", pos_frac[2])],
            cell_m,
            DIM_POS,
        )
        vec_record(sp, "momentum", [("x", ux), ("y", uy), ("z", uz)], M_E * C, DIM_MOM)
        ones = np.ones(n, dtype=np.float64)
        scalar_record(sp, "weighting", ones, 1.0, DIM_SCALAR)
        scalar_record(sp, "charge", -np.ones(n, dtype=np.float64), Q_E, DIM_CHARGE)
        scalar_record(sp, "mass", np.ones(n, dtype=np.float64), M_E, DIM_MASS)


def gamma10_bunch(rng, n):
    sig = 10.0 * 1.0e-3
    return rng.normal(0.0, sig, n), rng.normal(0.0, sig, n), np.full(n, 10.0)


def write_main(out_dir, n):
    rng = np.random.default_rng(42 + n)
    ux, uy, uz = gamma10_bunch(rng, n)
    pos = [rng.uniform(0.0, 20.0, n) for _ in range(3)]
    path = out_dir / f"input_N{n}.h5"
    write_bunch(path, n, 1.0e-6, ux, uy, uz, pos)
    return path


def write_cost(out_dir, n):
    # The cost table records "20-cell bunch" and grid 66x68x66 at 2.5e-7 m but
    # not the drawer or origin.  Store the cell indices directly (no
    # (grid-20)/2 offset): the openPMD position is p*unitSI, so uniform(0, 20)
    # is the physical interval [0, 20*cell), which is the interior and the
    # physical centre of the grid whose origin is COST_ORIGIN_M.
    rng = np.random.default_rng(42 + n)
    ux, uy, uz = gamma10_bunch(rng, n)
    pos = [rng.uniform(0.0, COST_BUNCH_CELLS, n) for _ in range(3)]
    path = out_dir / f"input_N{n}_cost.h5"
    write_bunch(path, n, COST_CELL_M, ux, uy, uz, pos)
    return path


def write_multivel(out_dir, n=200):
    rng = np.random.default_rng(7)
    gamma = 10.0
    u = gamma * rng.normal(1.0, 0.05, n)
    theta = np.deg2rad(5.0) * np.sqrt(rng.uniform(0.0, 1.0, n))
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    ux = u * np.sin(theta) * np.cos(phi)
    uy = u * np.sin(theta) * np.sin(phi)
    uz = u * np.cos(theta)
    pos = [rng.uniform(0.0, 20.0, n) for _ in range(3)]
    path = out_dir / "input_mv_N200.h5"
    write_bunch(path, n, 1.0e-6, ux, uy, uz, pos)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--case", choices=["main", "cost", "mv", "all"], default="all")
    parser.add_argument("--particles", type=int, nargs="*", default=[10000, 100000, 1000000, 2000000])
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if args.case in ("main", "all"):
        for n in args.particles:
            written.append(write_main(out_dir, n))
    if args.case in ("cost", "all"):
        n = COST_N if COST_N in args.particles else max(args.particles)
        written.append(write_cost(out_dir, n))
    if args.case in ("mv", "all"):
        written.append(write_multivel(out_dir))
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
