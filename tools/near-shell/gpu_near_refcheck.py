#!/usr/bin/env python3
"""GPU1 accuracy check: compare gpu_near per-pair output against the independent indfield reference.

Usage: ref_check.py <sample_base> [more bases ...]
Reads <base>.bin (rel[3n] f64, comp[n] i32, out[n] f64) + <base>.meta.txt, computes the
TSC-convolved field at each staggered point with indfield, and reports the worst per-component
relative error  |gpu - ref| / (1 + max(|gpu|,|ref|)).
"""
import sys, os, hashlib
import numpy as np

sys.path.insert(0, os.path.expanduser("~/pr24/pr24w53-ref"))
import indfield as I  # noqa: E402

OFFSETS = np.array([
    [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5],
    [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])

ARB_ORDER = 24
ARB_REFINE = 0


def read_meta(base):
    m = {}
    with open(base + ".meta.txt") as f:
        for line in f:
            parts = line.split()
            m[parts[0]] = parts[1:]
    return m


def load_sample(base):
    meta = read_meta(base)
    n = int(meta["npairs"][0])
    raw = open(base + ".bin", "rb").read()
    off = 0
    rel = np.frombuffer(raw[off:off + 3 * n * 8], dtype=np.float64).reshape(n, 3).copy()
    off += 3 * n * 8
    comp = np.frombuffer(raw[off:off + n * 4], dtype=np.int32).copy()
    off += n * 4
    out = np.frombuffer(raw[off:off + n * 8], dtype=np.float64).copy()
    u = np.array([float(meta["ux"][0]), float(meta["uy"][0]), float(meta["uz"][0])])
    cell = np.array([float(x) for x in meta.get("cell", ["1", "1", "1"])])
    return meta, rel, comp, out, u, cell


def reference(rel, comp, u, cell, cachekey):
    """Converged-ish arbiter (order 24 refine 0), cached. Returns (E[n,3], B[n,3])."""
    cache = "/tmp/gpu1_ref_%s.npz" % cachekey
    if os.path.exists(cache):
        z = np.load(cache)
        return z["E"], z["B"]
    n = rel.shape[0]
    E = np.zeros((n, 3)); B = np.zeros((n, 3))
    for i in range(n):
        xo = rel[i] + OFFSETS[comp[i]] * cell
        e, b = I.integrate_point(xo, np.zeros(3), u, cell[0], cell[1], cell[2],
                                 order=ARB_ORDER, refine=ARB_REFINE, chunk=8)
        E[i] = e; B[i] = b
        if i % 50 == 0:
            print("   ref %d/%d" % (i, n), flush=True)
    np.savez(cache, E=E, B=B)
    return E, B


def validate_arbiter(rel, comp, u, cell):
    """Bound the arbiter (order 24 refine 0) by comparing to order 40 refine 1 on a few pairs.
    integrate_point_converged is NOT used: for singular pairs it cannot reach 1e-10 (in-support
    floor ~1e-4) and escalates to infeasible node counts. Two nearby high orders that agree to
    <1e-6 establish the arbiter is converged."""
    worst = 0.0
    detail = None
    for i in range(0, min(6, len(comp))):
        xo = rel[i] + OFFSETS[comp[i]] * cell
        ea, ba = I.integrate_point(xo, np.zeros(3), u, cell[0], cell[1], cell[2],
                                   order=ARB_ORDER, refine=ARB_REFINE, chunk=8)
        eh, bh = I.integrate_point(xo, np.zeros(3), u, cell[0], cell[1], cell[2],
                                   order=40, refine=1, chunk=8)
        a = np.concatenate([ea, ba]); h = np.concatenate([eh, bh])
        e = np.max(np.abs(a - h)) / (1.0 + np.max(np.abs(h)))
        worst = max(worst, e)
        detail = e
    return worst, {"detail_last": detail}


def main():
    bases = sys.argv[1:]
    for base in bases:
        meta, rel, comp, out, u, cell = load_sample(base)
        key = hashlib.md5(("".join(map(str, rel.tolist())) + "|" + str(tuple(u)).replace(" ", "")).encode()).hexdigest()[:12]
        print("=== %s (kernel=%s band=%s n=%d gamma=%s) ===" % (
            os.path.basename(base), meta["kernel"][0], meta["band"][0], len(comp), meta["gamma"][0]))
        arb_err, info = validate_arbiter(rel, comp, u, cell)
        print("   arbiter (o%d/r%d) vs o40/r1 bound: worst rel diff = %.3e  (%s)" % (
            ARB_ORDER, ARB_REFINE, arb_err, info))
        E, B = reference(rel, comp, u, cell, key)
        ref = np.where(comp[:, None] < 3, E, B)  # (n,3)
        refc = np.array([ (ref[i, comp[i] % 3] if comp[i] < 3 else ref[i, comp[i] % 3]) for i in range(len(comp)) ])
        # per-component worst relative error
        names = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]
        worst = {}
        nnan = 0
        for c in range(6):
            sel = comp == c
            if not sel.any():
                continue
            g = out[sel]; r = refc[sel]
            err = np.abs(g - r) / (1.0 + np.maximum(np.abs(g), np.abs(r)))
            nnan += int(np.sum(~np.isfinite(err)))
            val = float(np.nanmax(err)) if np.any(np.isfinite(err)) else float("nan")
            worst[names[c]] = val
        overall = float(np.nanmax(list(worst.values()))) if worst else float("nan")
        print("   worst per-component rel err: " + "  ".join("%s=%.2e" % (k, v) for k, v in worst.items()))
        print("   (NaN pairs skipped: %d of %d)" % (nnan, len(comp)))
        print("   OVERALL worst rel err = %.3e   (1e-4 gate: %s ; 1e-3 gate: %s)" % (
            overall, "PASS" if overall < 1e-4 else "FAIL", "PASS" if overall < 1e-3 else "FAIL"))
    print("REFCHECK_DONE")


if __name__ == "__main__":
    main()
