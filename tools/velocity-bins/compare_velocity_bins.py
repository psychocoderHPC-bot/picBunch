#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
#
# Compare the binned mesh fields against the TSC-convolved continuum reference
# for a mixed-velocity bunch (tools/velocity-bins/indfield.py is the arbiter,
# the same module the paper's N2b/W4b measurements used).
#
# For a bunch with per-particle velocities u_p the continuum target is
#
#   T(x) = sum_p q_p w_p I(x - x_p; u_p),   I = int S(s) K(x - s*cell) ds^3
#
# with the TSC shape S and the moving-point kernel K.  The binned mesh
# replaces u_p by the box representative u_b, so the pure binning error is
#   Tb(x) - T(x) = sum_b (sum_{p in b} q_p w_p) I(x - x_p; u_b) - T(x)
# and the mesh error decomposes as
#   M_B - T = (M_B - Tb) + (Tb - T)   (discretization + binning).
#
# Usage:
#   compare_velocity_bins.py PREFIX [--bins 1,2,4,8,16] [--workers 32]
#       [--order 12] [--refine 1] [--refine100 2] [--fit] [--bands 0-1,1-2,2-4,4-8,8-16,16+]
#
# Reads PREFIX_bunch.txt, PREFIX_field_B<B>.bin, PREFIX_meta_B<B>.txt and
# writes PREFIX_compare.md / PREFIX_compare.json.  ``--fit`` reports the
# per-box-count ratio measured/modelSum (the calibration constant).

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import indfield  # noqa: E402

K_COULOMB = 8.9875517873681764e9
C_LIGHT = 299792458.0
YEE = indfield.OFFSETS  # (6, 3), cell units
COMPONENTS = ("Ex", "Ey", "Ez", "Bx", "By", "Bz")
UNITS = ("V/m", "V/m", "V/m", "T", "T", "T")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prefix")
    p.add_argument("--bins", default="1,2,4,8,16")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--order", type=int, default=12)
    p.add_argument("--refine", type=int, default=1)
    p.add_argument("--refine100", type=int, default=2,
                   help="refine level used when gamma >= 100 (beamed integrands)")
    p.add_argument("--fit", action="store_true")
    p.add_argument("--bands", default="0-1,1-2,2-4,4-8,8-16,16+")
    return p.parse_args()


def parse_bands(text: str):
    bands = []
    for part in text.split(","):
        part = part.strip()
        if part.endswith("+"):
            lo = float(part[:-1])
            bands.append((lo, math.inf, f"[{lo:g}, inf)"))
        else:
            lo, hi = part.split("-")
            bands.append((float(lo), float(hi), f"[{lo:g}, {hi:g})"))
    return bands


def read_meta(prefix: str, B: int) -> dict:
    meta: dict = {}
    with open(f"{prefix}_meta_B{B}.txt") as f:
        for line in f:
            tok = line.split()
            meta[tok[0]] = tok[1:]
    return meta


def read_bunch(prefix: str):
    x, y, z, ux, uy, uz, w, q, box = [], [], [], [], [], [], [], [], []
    with open(f"{prefix}_bunch.txt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            v = line.split()
            x.append(float(v[0])); y.append(float(v[1])); z.append(float(v[2]))
            ux.append(float(v[3])); uy.append(float(v[4])); uz.append(float(v[5]))
            w.append(float(v[6])); q.append(float(v[7])); box.append(int(v[8]))
    return {k: np.asarray(v) for k, v in
            dict(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, w=w, q=q).items()}


def read_field(prefix: str, B: int, n) -> np.ndarray:
    data = np.fromfile(f"{prefix}_field_B{B}.bin", dtype=np.float64)
    return data.reshape(6, n[0], n[1], n[2]).copy()


def node_windows(bunch: dict, n, origin, cell, margin: float = 2.0):
    """Per particle, the (i, j, k) node tuples whose Yee positions are within
    the TSC support (|s| < 1.5 per axis, plus the 0.5-cell Yee shift)."""
    xs = bunch["x"]; ys = bunch["y"]; zs = bunch["z"]
    windows = []
    for p in range(len(xs)):
        lo = []
        hi = []
        for pos, c, o, nn in ((xs[p], cell[0], origin[0], n[0]),
                              (ys[p], cell[1], origin[1], n[1]),
                              (zs[p], cell[2], origin[2], n[2])):
            a = math.floor((pos - margin * c - o) / c) - 1
            b = math.ceil((pos + margin * c - o) / c) + 1
            lo.append(max(0, a))
            hi.append(min(nn, b + 1))
        triples = [(i, j, k) for i in range(lo[0], hi[0])
                   for j in range(lo[1], hi[1]) for k in range(lo[2], hi[2])]
        windows.append(triples)
    return windows


def worker(args):
    (p_lo, p_hi, bunch, u_use, n, origin, cell, order, refine) = args
    n = tuple(n)
    origin = np.asarray(origin); cell = np.asarray(cell)
    acc = np.zeros((6, n[0], n[1], n[2]))
    xs = bunch["x"]; ys = bunch["y"]; zs = bunch["z"]
    ws = bunch["w"]; qs = bunch["q"]
    for p in range(p_lo, p_hi):
        x_obs_src = np.array([xs[p], ys[p], zs[p]])
        wq = ws[p] * qs[p]
        if wq == 0.0:
            continue
        lo = []
        hi = []
        for pos, c, o, nn in ((xs[p], cell[0], origin[0], n[0]),
                              (ys[p], cell[1], origin[1], n[1]),
                              (zs[p], cell[2], origin[2], n[2])):
            a = math.floor((pos - 2.0 * c - o) / c) - 1
            b = math.ceil((pos + 2.0 * c - o) / c) + 1
            lo.append(max(0, a))
            hi.append(min(nn, b + 1))
        for i in range(lo[0], hi[0]):
            for j in range(lo[1], hi[1]):
                for k in range(lo[2], hi[2]):
                    node = origin + np.array([i, j, k]) * cell
                    for c in range(6):
                        x_obs = node + YEE[c] * cell
                        e, b = indfield.integrate_point(
                            x_obs, x_obs_src, u_use[p], cell[0], cell[1], cell[2],
                            order=order, refine=refine)
                        val = e[c] if c < 3 else b[c - 3]
                        acc[c, i, j, k] += wq * (K_COULOMB if c < 3 else K_COULOMB / C_LIGHT) * val
    return acc


def compute_reference(bunch, u_use, n, origin, cell, order, refine, workers):
    nparts = len(bunch["x"])
    chunks = max(1, nparts // workers)
    jobs = []
    for lo in range(0, nparts, chunks):
        jobs.append((lo, min(nparts, lo + chunks), bunch, u_use, tuple(n),
                     [origin[0], origin[1], origin[2]], [cell[0], cell[1], cell[2]],
                     order, refine))
    if workers <= 1:
        return sum(worker(j) for j in jobs)
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        return sum(ex.map(worker, jobs))


def nearest_distance(bunch, n, origin, cell, dmax: float):
    """Distance (cells) from every node within dmax of some particle; -1 beyond."""
    d = np.full((n[0], n[1], n[2]), -1.0)
    xs = bunch["x"]; ys = bunch["y"]; zs = bunch["z"]
    cell = np.asarray(cell)
    origin = np.asarray(origin)
    nodes = np.stack(np.meshgrid(
        np.arange(n[0]), np.arange(n[1]), np.arange(n[2]), indexing="ij"), axis=-1)
    pos = origin + nodes * cell  # (n0, n1, n2, 3)
    best = np.full(pos.shape[:-1], np.inf)
    for p in range(len(xs)):
        src = np.array([xs[p], ys[p], zs[p]])
        diff = pos - src
        dist = np.sqrt(np.sum(diff * diff, axis=-1)) / cell[0]
        best = np.minimum(best, dist)
    mask = best <= dmax
    d[mask] = best[mask]
    return d


def band_index(dist, bands):
    idx = np.full(dist.shape, -1, dtype=int)
    for bi, (lo, hi, _label) in enumerate(bands):
        sel = (dist >= lo) & (dist < hi)
        idx[sel] = bi
    return idx


def error_stats(mesh, ref, dist, bands, band_idx):
    """Worst per-component relative (unfloored, over the reference window) and
    absolute error, overall and per distance band."""
    window = dist >= 0
    rel_all = np.zeros(6)
    abs_all = np.zeros(6)
    for c in range(6):
        m = mesh[c][window]
        r = ref[c][window]
        denom = np.maximum(np.maximum(np.abs(r), np.abs(m)), 1e-30)
        rel_all[c] = np.max(np.abs(m - r) / denom)
        abs_all[c] = np.max(np.abs(m - r))
    per_band = []
    for bi, (lo, hi, label) in enumerate(bands):
        sel = band_idx == bi
        if not sel.any():
            per_band.append({"band": label, "n": 0, "rel": [np.nan] * 6,
                             "abs": [np.nan] * 6, "worst_rel": np.nan, "worst_abs": np.nan})
            continue
        rel = np.zeros(6)
        absv = np.zeros(6)
        for c in range(6):
            m = mesh[c][sel]
            r = ref[c][sel]
            denom = np.maximum(np.maximum(np.abs(r), np.abs(m)), 1e-30)
            rel[c] = np.max(np.abs(m - r) / denom)
            absv[c] = np.max(np.abs(m - r))
        per_band.append({"band": label, "n": int(sel.sum()), "rel": rel.tolist(),
                         "abs": absv.tolist(), "worst_rel": float(np.max(rel)),
                         "worst_abs": float(np.max(absv))})
    return {
        "window_nodes": int(window.sum()),
        "rel": rel_all.tolist(),
        "abs": abs_all.tolist(),
        "worst_rel": float(np.max(rel_all)),
        "worst_abs": float(np.max(abs_all)),
        "worst_rel_component": int(np.argmax(rel_all)),
        "bands": per_band,
        "dominant_band": per_band[int(np.argmax([b["worst_rel"] for b in per_band]))]["band"]
        if any(b["n"] for b in per_band) else None,
    }


def fmt(v, sig=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.{sig}e}"


def main() -> None:
    args = parse_args()
    prefix = args.prefix
    bins = [int(b) for b in args.bins.split(",") if b.strip()]
    bands = parse_bands(args.bands)

    bunch = read_bunch(prefix)
    first_meta = read_meta(prefix, bins[0])
    n = [int(v) for v in first_meta["grid"]]
    origin = [float(v) for v in first_meta["origin"]]
    cell = [float(v) for v in first_meta["cell"]]
    gamma = float(first_meta["gamma"][0])
    refine = args.refine100 if gamma >= 100.0 else args.refine

    u_p = np.stack([bunch["ux"], bunch["uy"], bunch["uz"]], axis=1)
    print(f"bunch: {len(bunch['x'])} particles, gamma={gamma}, grid={n}, cell={cell[0]} m, "
          f"reference order={args.order} refine={refine}, workers={args.workers}")

    dist = nearest_distance(bunch, tuple(n), origin, cell, dmax=16.0)
    band_idx = band_index(dist, bands)

    # Continuum target with per-particle velocities (independent of B).
    print("computing T (per-particle continuum reference) ...", flush=True)
    T = compute_reference(bunch, u_p, n, origin, cell, args.order, refine, args.workers)

    results = {"bunch": {"sources": len(bunch["x"]), "gamma": gamma,
                         "energy_spread": float(first_meta["energy_spread"][0]),
                         "angle_degrees": float(first_meta["angle_degrees"][0]),
                         "grid": n, "cell": cell[0], "origin": origin,
                         "order": args.order, "refine": refine},
               "runs": {}}
    lines = []
    lines.append(f"# MV1 comparison: `{os.path.basename(prefix)}`")
    lines.append("")
    lines.append(f"Bunch: {len(bunch['x'])} particles, gamma = {gamma:g}, "
                 f"sigma_g/gamma = {first_meta['energy_spread'][0]}, "
                 f"sigma_theta = {first_meta['angle_degrees'][0]} deg; "
                 f"grid {n[0]}^3, cell {cell[0]:g} m.")
    lines.append(f"Reference: TSC-convolved continuum (indfield, order {args.order}, "
                 f"refine {refine}); errors unfloored, over nodes within 16 cells of a particle.")
    lines.append("")

    for B in bins:
        meta = read_meta(prefix, B)
        B_used = int(meta["bins_used"][0])
        model_sum = float(meta["model_sum"][0])
        model_err = float(meta["model_error"][0])
        per_axis = meta["per_axis"]
        rep = np.array(meta["representative_u"], dtype=float).reshape(-1, 3)
        box_of = np.array(meta["box_of"], dtype=int)
        u_b = u_p.copy()
        u_b[box_of >= 0] = rep[box_of[box_of >= 0]] if len(rep) > 1 else rep[0]

        mesh = read_field(prefix, B_used, n)
        print(f"B={B_used}: mesh vs T ...", flush=True)
        stats_T = error_stats(mesh, T, dist, bands, band_idx)
        Tb = compute_reference(bunch, u_b, n, origin, cell, args.order, refine, args.workers)
        stats_bin = error_stats(Tb, T, dist, bands, band_idx)
        # discretization part: mesh vs Tb (needs Tb as "ref"; mesh error vs Tb)
        mesh_disp = mesh.copy()
        stats_disc = error_stats(mesh_disp, Tb, dist, bands, band_idx)
        outside = dist < 0
        roundoff = float(np.max(np.abs(mesh[outside])) if outside.any() else 0.0)

        results["runs"][str(B)] = {
            "bins_used": B_used, "per_axis": per_axis, "model_sum": model_sum,
            "model_error": model_err,
            "mesh_vs_T": stats_T, "Tb_vs_T": stats_bin, "mesh_vs_Tb": stats_disc,
            "outside_window_roundoff": roundoff,
        }

        lines.append(f"## B = {B_used} (requested {B}, per-axis {per_axis[0]}x{per_axis[1]}x{per_axis[2]}, "
                     f"modelSum {model_sum:.4e}, model error {model_err:.4e})")
        lines.append("")
        lines.append("| metric | Ex | Ey | Ez | Bx | By | Bz | worst |")
        lines.append("|---|---|---|---|---|---|---|---|")
        lines.append("| mesh-T rel | " + " | ".join(fmt(v) for v in stats_T["rel"])
                     + f" | {stats_T['worst_rel']:.3e} |")
        lines.append("| mesh-T abs | " + " | ".join(fmt(v) for v in stats_T["abs"])
                     + f" | {stats_T['worst_abs']:.3e} |")
        lines.append("| Tb-T rel (binning) | " + " | ".join(fmt(v) for v in stats_bin["rel"])
                     + f" | {stats_bin['worst_rel']:.3e} |")
        lines.append("| mesh-Tb rel (discr.) | " + " | ".join(fmt(v) for v in stats_disc["rel"])
                     + f" | {stats_disc['worst_rel']:.3e} |")
        lines.append("")
        lines.append("| band (cells) | n | worst rel mesh-T | worst abs mesh-T |")
        lines.append("|---|---|---|---|")
        for b in stats_T["bands"]:
            lines.append(f"| {b['band']} | {b['n']} | {fmt(b['worst_rel'])} | {fmt(b['worst_abs'])} |")
        lines.append("")
        lines.append(f"Dominant band: {stats_T['dominant_band']}; "
                     f"outside-window max|mesh| = {roundoff:.3e}.")
        lines.append("")

    if args.fit:
        lines.append("## Calibration fit")
        lines.append("")
        lines.append("| B | mesh-T worst rel | Tb-T worst rel | modelSum | C (mesh-T) | C (Tb-T) |")
        lines.append("|---|---|---|---|---|---|")
        c_vals_T, c_vals_B = [], []
        for B in bins:
            r = results["runs"][str(B)]
            ms = r["model_sum"]
            if ms <= 0:
                continue
            cT = r["mesh_vs_T"]["worst_rel"] / ms
            cB = r["Tb_vs_T"]["worst_rel"] / ms
            c_vals_T.append(cT)
            c_vals_B.append(cB)
            lines.append(f"| {r['bins_used']} | {r['mesh_vs_T']['worst_rel']:.3e} | "
                         f"{r['Tb_vs_T']['worst_rel']:.3e} | {ms:.4e} | {cT:.3e} | {cB:.3e} |")
        if c_vals_T:
            lines.append("")
            lines.append(f"C median: mesh-T {np.median(c_vals_T):.3e}, "
                         f"Tb-T {np.median(c_vals_B):.3e}; "
                         f"geomean: mesh-T {math.exp(np.mean(np.log(np.maximum(c_vals_T, 1e-300)))):.3e}, "
                         f"Tb-T {math.exp(np.mean(np.log(np.maximum(c_vals_B, 1e-300)))):.3e}")
        lines.append("")

    with open(f"{prefix}_compare.md", "w") as f:
        f.write("\n".join(lines))
    with open(f"{prefix}_compare.json", "w") as f:
        json.dump(results, f, indent=1, default=float)
    print(f"wrote {prefix}_compare.md / .json")


if __name__ == "__main__":
    main()
