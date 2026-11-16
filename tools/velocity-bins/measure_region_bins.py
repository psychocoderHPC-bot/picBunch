#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""
Reproduce the real-bunch region-aware in-bunch calibration table.

Wraps the committed TSC-convolved continuum reference
``indfield.integrate_point`` (order 8) and measures the peak-normalised
relative field error, per component and as the driver (max over E and B),
in a small in-bunch cell patch around a chosen cell (the By-peak cell for
the D6 bunch).  This is the reference that produced
``docs/measurements/velocity_bin_real_bunch_calibration.txt``; the exact
command is stated in that document.

The reference is the per-particle continuum field T (each particle keeps its
own velocity) versus the binned field Tb (each particle is replaced by its
velocity-bin representative).  The error is normalised by the in-bunch PEAK
of the reference per component (peak-normalised, not per-cell), which is the
metric the auto policy drives.

Because the per-particle continuum reference is slow (~20 ms per particle per
cell in-bunch), the measurement is bounded by a cell patch (``--cell-radius``)
and a stratified particle subset (``--subset``).  The full
``--grid`` x ``N`` run is intractable; the validated region is the 27-cell
patch around the By-peak cell at the 391-particle subset.

Usage (D6 real bunch, the command that reproduces the committed table):

  python3 tools/velocity-bins/measure_region_bins.py \
      --bunch particles_000000.bp5 --species b \
      --cell 2e-7 --origin 1.5149e-05,9.7255e-05,1.2683e-05 \
      --grid 298,122,318 --bins 1,2,4,8,16,64,256 \
      --cell-region 163,70,23 --cell-radius 3 --subset 391

A plain text bunch file (``x y z ux uy uz weight charge boxOf``) is also
accepted via ``--bunch file.txt`` (no ``--species`` needed); the text reader
skips a comment header and the same metric is applied.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import indfield  # noqa: E402  (committed reference; same directory)

K_Coulomb = 8.9875517873681764e9  # N m^2 / C^2
k_SpeedOfLight = 299792458.0  # m / s
COMPONENTS = ("Ex", "Ey", "Ez", "Bx", "By", "Bz")


def _split_ints(text: str, name: str) -> list[int]:
    parts = [p for p in text.split(",") if p.strip() != ""]
    try:
        vals = [int(p) for p in parts]
    except ValueError as exc:
        raise SystemExit(f"{name}: expected comma-separated integers, got {text!r}") from exc
    return vals


def _split_floats(text: str, name: str) -> list[float]:
    parts = [p for p in text.split(",") if p.strip() != ""]
    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        raise SystemExit(f"{name}: expected comma-separated floats, got {text!r}") from exc


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--bunch", required=True, type=pathlib.Path,
                   help="OpenPMD (.bp5) particle series or a text bunch file")
    p.add_argument("--species", default="b",
                   help="OpenPMD species name (default b); ignored for text input")
    p.add_argument("--cell", required=True,
                   help="cell size dx[,dy,dz] in metres, comma-separated")
    p.add_argument("--origin", required=True,
                   help="grid origin x0,y0,z0 in metres, comma-separated")
    p.add_argument("--grid", required=True,
                   help="grid nx,ny,nz, comma-separated")
    p.add_argument("--bins", required=True,
                   help="comma-separated velocity-bin counts, e.g. 1,2,4,8,16,64,256")
    p.add_argument("--cell-region", required=True,
                   help="reference cell (ix,iy,iz), x-fastest, e.g. the By-peak cell")
    p.add_argument("--cell-radius", type=int, default=3,
                   help="patch radius around --cell-region (3 -> 7x7x7 = 343 cells; "
                        "the D6 record uses radius 1 -> 27 cells; see the doc)")
    p.add_argument("--subset", type=int, default=391,
                   help="stratified particle subset size (of N); bounds the reference cost")
    p.add_argument("--order", type=int, default=8,
                   help="indfield integration order (default 8, the validated order)")
    p.add_argument("--out", type=pathlib.Path, default=None,
                   help="optional: write the per-component table to this .npy "
                        "(shape (n_bins, n_cells, 6))")
    return p.parse_args(argv)


def _load_text(path: pathlib.Path):
    """Read a ``x y z ux uy uz weight charge boxOf`` text bunch file."""
    d = np.loadtxt(path, ndmin=2)
    if d.ndim == 1:
        d = d[None, :]
    if d.shape[1] < 8:
        raise SystemExit(f"{path}: expected >= 8 columns (x y z ux uy uz weight charge boxOf)")
    x = d[:, 0:3].astype(np.float64)
    u = d[:, 3:6].astype(np.float64)
    wq = d[:, 6].astype(np.float64) * d[:, 7].astype(np.float64)
    return x, u, wq


def _load_openpmd(path: pathlib.Path, species: str):
    """Read an OpenPMD particle series via openpmd-api (best effort)."""
    try:
        import openpmd_api as openpmd
    except ImportError as exc:
        raise SystemExit(
            "openpmd-api is required to read an OpenPMD bunch; pass a text bunch "
            "file or install openpmd-api"
        ) from exc
    s = openpmd.Series(str(path), openpmd.AccessKind.READ_ONLY)
    its = sorted(s.iterations.keys())
    if not its:
        raise SystemExit(f"{path}: no iterations")
    sp = s.iterations[its[0]].particles[species]
    rec = sp["positionOffset" if "positionOffset" in sp else "position"]
    x = np.stack(
        [np.asarray(rec[i]) if False else np.asarray(sp["positionOffset"][i])
         for i in range(3)], axis=-1
    )
    u = np.stack([np.asarray(sp["momentum"][i]) for i in range(3)], axis=-1)
    w = np.asarray(sp["weighting"][0])
    charge = sp["charge"][0] if "charge" in sp else np.full(w.shape, -1.602176634e-19)
    wq = w * charge
    return x, u, wq


def _beam_frame(u: np.ndarray) -> np.ndarray:
    """Beam-frame components.  The D6 beam axis is lab +y: e1=-ux, e2=+uz,
    e3=+uy (the charge-weighted mean direction is along lab y for this bunch)."""
    return np.column_stack([-u[:, 0], u[:, 2], u[:, 1]])


def _split(B: int) -> tuple[int, int, int]:
    """Per-axis split for the transverse box budget (axis 3 / beam axis is 1)."""
    if B == 1:
        return (1, 1, 1)
    if B == 2:
        return (1, 2, 1)
    if B == 4:
        return (2, 2, 1)
    if B == 8:
        return (2, 4, 1)
    if B == 16:
        return (4, 4, 1)
    if B == 64:
        return (8, 8, 1)
    if B == 256:
        return (16, 16, 1)
    # nearest power-of-two factorisation (transverse only)
    import math
    n = int(round(math.sqrt(B)))
    n = max(1, 2 ** (n.bit_length() - 1))
    return (n, max(1, B // n), 1)


def _box_of(ubeam: np.ndarray, per_axis) -> np.ndarray:
    lo = ubeam.min(axis=0)
    hi = ubeam.max(axis=0)
    rng = hi - lo
    pa = per_axis
    idx = np.zeros((ubeam.shape[0], 3), dtype=np.int64)
    for a in range(3):
        if pa[a] > 1 and rng[a] > 0:
            t = np.floor((ubeam[:, a] - lo[a]) / (rng[a] / pa[a]))
            idx[:, a] = np.clip(t, 0, pa[a] - 1)
    return idx[:, 0] + pa[0] * (idx[:, 1] + pa[1] * idx[:, 2])


def _representative(u: np.ndarray, ubeam: np.ndarray, per_axis, wq: np.ndarray):
    """Charge-weighted box-centroid representative per particle."""
    pa = per_axis
    boxes = pa[0] * pa[1] * pa[2]
    bb = _box_of(ubeam, pa)
    rep = np.zeros((boxes, 3))
    sw = np.zeros(boxes)
    np.add.at(rep, bb, wq[:, None] * u)
    np.add.at(sw, bb, wq)
    used = sw != 0.0
    rep[used] /= sw[used, None]
    return rep[bb]


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    cell = _split_floats(args.cell, "--cell")
    while len(cell) < 3:
        cell.append(cell[-1])
    origin = np.array(_split_floats(args.origin, "--origin"))
    grid = _split_ints(args.grid, "--grid")
    bins = _split_ints(args.bins, "--bins")
    c0 = _split_ints(args.cell_region, "--cell-region")
    r = args.cell_radius
    nx, ny, nz = grid

    if args.bunch.suffix == ".txt" or args.bunch.suffix == ".dat":
        x, u, wq = _load_text(args.bunch)
    else:
        x, u, wq = _load_openpmd(args.bunch, args.species)
    N = len(x)
    k = max(1, int(round(N / args.subset)))
    idx = np.arange(0, N, k)
    xs, us, wqs = x[idx], u[idx], wq[idx]
    Ns = len(xs)

    cells = []
    for i in range(c0[0] - r, c0[0] + r + 1):
        for j in range(c0[1] - r, c0[1] + r + 1):
            for kk in range(c0[2] - r, c0[2] + r + 1):
                if 0 <= i < nx and 0 <= j < ny and 0 <= kk < nz:
                    cells.append((i, j, kk))
    cells = np.array(cells)
    print(f"{Ns} particles x {len(cells)} cells (patch {c0} radius {r}) "
          f"order {args.order}", flush=True)

    yee = indfield.OFFSETS

    def field_at(u_pp: np.ndarray) -> np.ndarray:
        acc = np.zeros((len(cells), 6))
        for ci in range(len(cells)):
            i, j, kk = cells[ci]
            node = origin + np.array([i, j, kk], dtype=np.float64) * np.array(cell)
            for c in range(6):
                x_obs = node + yee[c] * cell
                tot = 0.0
                for p in range(Ns):
                    e, b = indfield.integrate_point(
                        x_obs, xs[p], u_pp[p], cell[0], cell[1], cell[2],
                        order=args.order, refine=0)
                    tot += wqs[p] * (K_Coulomb if c < 3 else K_Coulomb / k_SpeedOfLight) \
                        * (e[c] if c < 3 else b[c - 3])
                acc[ci, c] = tot
        return acc

    t0 = time.time()
    T = field_at(us)
    print(f"T (per-particle continuum) in {time.time() - t0:.0f}s", flush=True)

    peaks = np.abs(T).max(axis=0)  # in-bunch peak per component
    ubeam = _beam_frame(us)
    results = []
    for B in bins:
        pa = _split(B)
        rep = _representative(us, ubeam, pa, wqs)
        t0 = time.time()
        Tb = field_at(rep)
        rel = np.abs(Tb - T) / peaks[None, :]  # peak-normalised, per cell/component
        bycomp = rel.max(axis=0)  # max over cells per component
        driver = float(bycomp.max())
        print(f"B={B:<4d} " + "  ".join(f"{bycomp[c]:.3e}" for c in range(6))
              + f"  DRIVER={driver:.3e}  ({time.time() - t0:.0f}s)", flush=True)
        results.append((B, pa, bycomp, driver))
    if args.out is not None:
        # store the last B's per-cell per-component table as a sanity artifact
        np.save(args.out, rel)

    print("\npeaks (in-bunch reference, per component):", flush=True)
    for c in range(6):
        print(f"  {COMPONENTS[c]}: {peaks[c]:.6e}", flush=True)
    print("\ntable (peak-normalised relative error, per component):", flush=True)
    print("   B | " + " | ".join(f"{COMPONENTS[c]:>9s}" for c in range(6)) + " |   DRIVER")
    for B, pa, bycomp, driver in results:
        print(f"{B:4d} | " + " | ".join(f"{bycomp[c]:9.3e}" for c in range(6))
              + f" | {driver:8.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
