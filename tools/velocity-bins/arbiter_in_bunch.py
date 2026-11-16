#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""
Arbiter: exact Liénard-Wiechert direct-sum reference for the velocity-binning
in-bunch field error.

This is the "truth" the region-aware metric is checked against: it computes
the field at a set of in-bunch probe nodes by direct summation of the
Liénard-Wiechert field of EVERY particle (no proxy, no fitted kernel, no TSC
convolution), at the actual per-particle gamma distribution, and reports the
per-component relative error (normalised by the reference amplitude at each
probe node, maximum over nodes) for a list of velocity-bin counts.

The reference is the per-particle field T (each particle keeps its own
velocity); the binned field Tb replaces every particle by the charge-weighted
representative of its velocity bin.  The relative error is
|Tb - T| / |T| per component per node, maximised over the probe nodes.

Usage (D6 real bunch, the command that reproduces the arbiter table in
docs/measurements/velocity_bin_real_bunch_calibration.txt):

  python3 tools/velocity-bins/arbiter_in_bunch.py \
      --bunch <bunch.bp5-or-txt> --species b \
      --bins 1,2,4,8,16,32,64,256

A text bunch file (``x y z ux uy uz weight charge boxOf``) is read directly;
an OpenPMD (``.bp5``) series is read via openpmd-api.  The probe nodes are the
bunch core plus points at 2, 5 and 10 sigma in each transverse direction.

The result for the D6 bunch: the tolerance (1e-3) is NOT reached by any
component at any box count up to 256; the per-node relative error is dominated
by the beam-axis components (Ez, Bx) and the core 1/r singularity, and the
absolute in-bunch By error saturates (~3.5e-1 T at 1 box to ~6.4e-2 T at 256
boxes).  This is why the auto policy saturates rather than returning the cap.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np


def _split_ints(text: str, name: str) -> list[int]:
    parts = [p for p in text.split(",") if p.strip() != ""]
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise SystemExit(f"{name}: expected comma-separated integers, got {text!r}") from exc


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--bunch", required=True, type=pathlib.Path,
                   help="OpenPMD (.bp5) series or a text bunch file")
    p.add_argument("--species", default="b",
                   help="OpenPMD species name (default b); ignored for text input")
    p.add_argument("--bins", default="1,2,4,8,16,32,64,256",
                   help="comma-separated velocity-bin counts")
    p.add_argument("--sigma-max", type=int, default=10,
                   help="largest sigma probe (default 10: probes at 0, 2, 5, 10)")
    return p.parse_args(argv)


def _load_text(path: pathlib.Path):
    d = np.loadtxt(path, ndmin=2)
    if d.ndim == 1:
        d = d[None, :]
    if d.shape[1] < 8:
        raise SystemExit(f"{path}: expected >= 8 columns (x y z ux uy uz weight charge boxOf)")
    d = d.astype(np.float64)
    return d[:, 0:3], d[:, 3:6], d[:, 6] * d[:, 7]


def _load_openpmd(path: pathlib.Path, species: str):
    try:
        import openpmd_api as openpmd
    except ImportError as exc:
        raise SystemExit("openpmd-api is required to read an OpenPMD bunch") from exc
    s = openpmd.Series(str(path), openpmd.AccessKind.READ_ONLY)
    its = sorted(s.iterations.keys())
    sp = s.iterations[its[0]].particles[species]
    x = np.stack([np.asarray(sp["positionOffset"][i]) for i in range(3)], axis=-1)
    u = np.stack([np.asarray(sp["momentum"][i]) for i in range(3)], axis=-1)
    w = np.asarray(sp["weighting"][0])
    charge = sp["charge"][0] if "charge" in sp else np.full(w.shape, -1.602176634e-19)
    return x.astype(np.float64), u.astype(np.float64), (w * charge).astype(np.float64)


K_Coulomb = 8.9875517873681764e9
k_SpeedOfLight = 299792458.0
COMPONENTS = ("Ex", "Ey", "Ez", "Bx", "By", "Bz")


def lw_field(xo: np.ndarray, x: np.ndarray, u: np.ndarray, wq: np.ndarray):
    """Liénard-Wiechert E, B of the full particle set at observation point xo."""
    R = xo[None, :] - x
    uu = (u * u).sum(axis=1)
    g = np.sqrt(1.0 + uu)
    b = u / g[:, None]
    D = np.einsum("ni,ni->n", R, b)
    r2 = (R * R).sum(axis=1)
    beta2 = uu / (g * g)
    denom = r2 - (beta2 * r2 - D * D)
    Eb = (wq * K_Coulomb * (1.0 - beta2) / (denom ** 1.5))[:, None] * R
    Bb = np.cross(b, Eb) / k_SpeedOfLight
    return Eb.sum(axis=0), Bb.sum(axis=0)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    bins = _split_ints(args.bins, "--bins")
    x, u, wq = (_load_text(args.bunch) if args.bunch.suffix in (".txt", ".dat")
                else _load_openpmd(args.bunch, args.species))
    N = len(x)
    print(f"arbiter: N={N} particles, bins {bins}", flush=True)

    ac = np.abs(wq)
    tot = ac.sum()
    core = np.array([(ac * x[:, 0]).sum(), (ac * x[:, 1]).sum(), (ac * x[:, 2]).sum()]) / tot
    sx = np.sqrt((ac * (x[:, 0] - core[0]) ** 2).sum() / tot)
    sz = np.sqrt((ac * (x[:, 2] - core[2]) ** 2).sum() / tot)
    print(f"core {core[0]:.6e} {core[1]:.6e} {core[2]:.6e}  "
          f"sigma x {sx:.3e} z {sz:.3e}", flush=True)

    # Probe nodes: core + 2, 5, 10 sigma in each transverse direction.
    sigmas = [s for s in (2, 5, 10) if s <= args.sigma_max]
    probes = [("core", core.copy())]
    for m in sigmas:
        for axis, s in (("x", sx), ("z", sz)):
            p = core.copy()
            p[0 if axis == "x" else 2] += m * s
            probes.append((f"{axis}+{m}", p))
            p2 = core.copy()
            p2[0 if axis == "x" else 2] -= m * s
            probes.append((f"{axis}-{m}", p2))
    nodes = [v for _, v in probes]
    names = [n for n, _ in probes]
    print(f"{len(nodes)} probe nodes", flush=True)

    # Reference T (per-particle, full set) at each node.
    T = np.zeros((len(nodes), 6))
    for ni in range(len(nodes)):
        E, Bf = lw_field(nodes[ni], x, u, wq)
        T[ni] = [E[0], E[1], E[2], Bf[0], Bf[1], Bf[2]]
    print("REFERENCE (full-set LW direct sum):", flush=True)
    print("node     " + "".join("%12s" % c for c in COMPONENTS))
    for ni, nm in enumerate(names):
        print("%-6s " % nm + "".join("%12.3e" % T[ni, c] for c in range(6)), flush=True)

    # Beam frame (D6 beam axis is lab +y): e1=-ux, e2=+uz, e3=+uy.
    ubeam = np.column_stack([-u[:, 0], u[:, 2], u[:, 1]])
    lo = ubeam.min(axis=0)
    hi = ubeam.max(axis=0)
    rng = hi - lo

    def split(B: int):
        return {1: (1, 1, 1), 2: (1, 2, 1), 4: (2, 2, 1), 8: (2, 4, 1),
                16: (4, 4, 1), 32: (8, 4, 1), 64: (8, 8, 1), 256: (16, 16, 1)}.get(
            B, (int(round(B ** 0.5)) or 1, 1, 1))

    print("\nRELATIVE ERROR per component (max over nodes); By absolute (T):", flush=True)
    print("   B  |" + "".join("%12s" % c for c in COMPONENTS) + "|  By_abs_max")
    for B in bins:
        pa = split(B)
        idx = np.zeros((N, 3), dtype=np.int64)
        for a in range(3):
            if pa[a] > 1 and rng[a] > 0:
                idx[:, a] = np.clip(np.floor((ubeam[:, a] - lo[a]) / (rng[a] / pa[a])), 0, pa[a] - 1)
        box = idx[:, 0] + pa[0] * (idx[:, 1] + pa[1] * idx[:, 2])
        boxes = pa[0] * pa[1] * pa[2]
        rep = np.zeros((boxes, 3))
        sw = np.zeros(boxes)
        np.add.at(rep, box, wq[:, None] * u)
        np.add.at(sw, box, wq)
        used = sw != 0
        rep[used] /= sw[used, None]
        u_pp = rep[box]
        Tb = np.zeros((len(nodes), 6))
        for ni in range(len(nodes)):
            E, Bf = lw_field(nodes[ni], x, u_pp, wq)
            Tb[ni] = [E[0], E[1], E[2], Bf[0], Bf[1], Bf[2]]
        rel = np.abs(Tb - T)
        amp = np.abs(T).copy()
        amp[amp < 1e-30] = np.nan
        maxnode = np.nanmax(rel / amp, axis=0)
        by_abs = float(np.abs(Tb[:, 4] - T[:, 4]).max())
        print("B=%-4d|%s|  %12.3e" %
              (B, "".join("%12.3e" % float(maxnode[c]) for c in range(6)), by_abs),
              flush=True)
    print("\nIf no count brings the per-component relative error below 1e-3, the "
          "tolerance is unreachable in-bunch and the auto policy saturates.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
