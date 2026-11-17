#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
#
# Orchestrate the adaptive edge validation: for each synthetic bunch case, run
# the measurement driver with uniform, adaptive and auto edges at fixed counts
# (2, 4, 8), compute the in-bunch per-component relative error against the
# TSC-convolved direct-sum reference (tools/velocity-bins/compare_velocity_bins.py
# machinery), and assemble the uniform/adaptive/auto table (transverse B included).
#
# Usage:
#   validate_adaptive.py CASES_DIR OUTDIR [--sources 512] [--gamma 50] [--workers 24]

from __future__ import annotations
import argparse, os, sys, json, math, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_velocity_bins as cvb  # noqa: E402
import indfield  # noqa: E402

COMPONENTS = ("Ex", "Ey", "Ez", "Bx", "By", "Bz")


def run_driver(measure, case, edge, count, prefix, sources, gamma, cell, grid, origin, bunch_file):
    cmd = [measure,
           "--bunch", bunch_file,
           "--velocity-bin-edges", edge,
           "--bins", str(count),
           "--gamma", str(gamma),
           "--cell", f"{cell},{cell},{cell}",
           "--grid", f"{grid},{grid},{grid}",
           "--origin", f"{origin[0]},{origin[1]},{origin[2]}",
           "--cloud-cells", "3.0",
           "--fft-padding", "2",
           "--near-shell", "none",
           "--output", prefix,
           "--sources", str(sources)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return prefix


def load_bunch(path):
    """Read a bunch file with 8 or 9 columns (x y z ux uy uz weight charge [boxOf])."""
    import numpy as np
    d = {k: [] for k in ("x", "y", "z", "ux", "uy", "uz", "w", "q")}
    with open(path) as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            v = line.split()
            d["x"].append(float(v[0])); d["y"].append(float(v[1])); d["z"].append(float(v[2]))
            d["ux"].append(float(v[3])); d["uy"].append(float(v[4])); d["uz"].append(float(v[5]))
            d["w"].append(float(v[6])); d["q"].append(float(v[7]))
    return {k: np.asarray(vv) for k, vv in d.items()}


def inbunch_error(prefix, B, n, origin, cell, bunch, maxdist=10.0, order=12):
    """Per-component max in-bunch relative error (within maxdist cells of a
    particle) between the binned mesh field and the TSC-convolved direct-sum
    reference, plus the absolute By."""
    mesh = cvb.read_field(prefix, B, n)
    # reference: per-particle direct sum with the per-particle u
    ref = cvb.compute_reference(bunch, np.stack([bunch["ux"], bunch["uy"], bunch["uz"]], axis=1),
                                tuple(n), [origin[0], origin[1], origin[2]],
                                [cell[0], cell[1], cell[2]], order, 1, 8)
    dist = cvb.nearest_distance(bunch, tuple(n), [origin[0], origin[1], origin[2]],
                                [cell[0], cell[1], cell[2]], maxdist)
    mask = dist >= 0.0
    err = np.zeros(6)
    for c in range(6):
        denom = np.abs(ref[c])
        num = np.abs(mesh[c] - ref[c])
        sel = mask & (denom > 1e-30)
        rel = np.where(sel, num / np.where(denom > 1e-30, denom, 1.0), 0.0)
        err[c] = float(np.max(rel[sel])) if sel.any() else 0.0
    by_abs = float(np.max(np.abs(mesh[4] - ref[4])[mask])) if mask.any() else 0.0
    return err, by_abs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cases_dir")
    p.add_argument("outdir")
    p.add_argument("--measure", default=None)
    p.add_argument("--sources", type=int, default=512)
    p.add_argument("--gamma", type=float, default=50.0)
    p.add_argument("--cell", type=float, default=1.0e-6)
    p.add_argument("--grid", type=int, default=48)
    p.add_argument("--origin", default="0,0,0")
    p.add_argument("--workers", type=int, default=24)
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    origin = [float(v) for v in args.origin.split(",")]
    n = (args.grid, args.grid, args.grid)
    cell = [args.cell, args.cell, args.cell]
    cell_scalar = args.cell
    measure = args.measure or os.path.join(HERE, "..", "..", "build-cpu", "picBunchVelocityBinsMeasure")
    cases = sorted(f[:-6] for f in os.listdir(args.cases_dir) if f.endswith(".bunch"))
    counts = [2, 4, 8]
    results = {}
    for case in cases:
        bunch_file = os.path.join(args.cases_dir, case + ".bunch")
        bunch = load_bunch(bunch_file)
        row = {}
        for edge in ("uniform", "adaptive", "auto"):
            for count in counts:
                prefix = os.path.join(args.outdir, f"{case}_{edge}_B{count}")
                run_driver(measure, case, edge, count, prefix, args.sources, args.gamma, cell_scalar, args.grid, origin, bunch_file)
                # find the actual bins_used from meta
                meta = cvb.read_meta(prefix, count)
                B = int(meta["bins_used"][0])
                err, by_abs = inbunch_error(prefix, B, n, origin, cell, bunch)
                key = f"{edge}_B{count}"
                row[key] = {"err": err, "by_abs": by_abs, "B": B,
                            "per_axis": meta.get("per_axis"), "bin_derive_ms": float(meta.get("bin_derive_ms", ["0"])[0])}
                print(f"{case} {edge} B{count}: " + " ".join(f"{c}={e:.3g}" for c, e in zip(COMPONENTS, err)), flush=True)
        results[case] = row
    with open(os.path.join(args.outdir, "adaptive_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n=== TABLE (max in-bunch relative error per component) ===")
    for case in cases:
        print(f"\n{case}:")
        for edge in ("uniform", "adaptive", "auto"):
            for count in counts:
                r = results[case][f"{edge}_B{count}"]
                print(f"  {edge:9s} B{count}: " + " ".join(f"{c}={e:.3g}" for c, e in zip(COMPONENTS, r["err"])) + f"  By_abs={r['by_abs']:.3g}")
        print(f"  (adaptive bin_derive_ms B8: {results[case]['adaptive_B8']['bin_derive_ms']:.3f} ms)")


if __name__ == "__main__":
    main()
