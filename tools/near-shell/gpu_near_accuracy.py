#!/usr/bin/env python3
"""Fast targeted accuracy: worst-case rel err vs converged ref at the hardest pairs (corners)."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.expanduser("~/pr24/pr24w53-ref"))
import indfield as I

CONV = (10, 3)
CFGS = [(8, 0), (8, 1), (8, 2)]
u = np.array([0.0, 0.0, 9.95])  # gamma=10 axial


def one(R0, c, order, refine):
    e, b = I.integrate_point(R0, np.zeros(3), u, 1.0, 1.0, 1.0, order=order, refine=refine, chunk=8)
    return e[c] if c < 3 else b[c - 3]


in_pairs = []
for R0 in [np.array([1.49, 1.49, 1.49]), np.array([0.0, 0.0, 0.0])]:
    for c in range(6):
        in_pairs.append((R0, c))
band_pairs = []
for R0 in [np.array([1.51, 0.5, 0.5]), np.array([2.5, 1.5, 0.0])]:
    for c in range(6):
        band_pairs.append((R0, c))

for name, pairs in [("IN", in_pairs), ("BAND", band_pairs)]:
    npan = 4 if name == "IN" else 3
    print("=== %s (%d pairs, %d panels/axis) converged=%s ===" % (name, len(pairs), npan, CONV), flush=True)
    conv = np.array([one(R0, c, *CONV) for (R0, c) in pairs])
    dconv = 0.0
    for (R0, c) in pairs[:2]:
        a = one(R0, c, 10, 3); h = one(R0, c, 10, 4)
        dconv = max(dconv, abs(a - h) / (1.0 + max(abs(a), abs(h))))
    print("   converged o10r3 vs o10r4 (2 pairs): %.3e" % dconv, flush=True)
    for (o, r) in CFGS:
        worst = 0.0; wp = None
        for i, (R0, c) in enumerate(pairs):
            v = one(R0, c, o, r)
            e = abs(v - conv[i]) / (1.0 + max(abs(v), abs(conv[i])))
            if e > worst:
                worst = e; wp = (R0, c)
        nodes_axis = npan * (2 ** r) * o
        print("   o%d r%d (%d^3) : worst rel err = %.3e  %s   @R0=%s c=%d" % (
            o, r, nodes_axis, worst, "<=1e-3" if worst < 1e-3 else ("<=1e-4" if worst < 1e-4 else "FAIL"),
            np.array2string(wp[0], precision=2), wp[1]), flush=True)
print("ACC2_DONE", flush=True)
