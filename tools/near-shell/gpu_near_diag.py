#!/usr/bin/env python3
"""Diagnostic: isolate whether the sing GPU kernel has a bug vs order-8 discretization error.
Compare gpu1_sing_{in,band}.bin (order 8 per panel) against indfield.integrate_point at order 8
(same scheme+order -> should agree to ~1e-9 if the kernel is correct) and at orders 12/16/24
(convergence)."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.expanduser("~/pr24/pr24w53-ref"))
import indfield as I

OFFSETS = np.array([[.5,0,0],[0,.5,0],[0,0,.5],[0,.5,.5],[.5,0,.5],[.5,.5,0]])

def load(base):
    m = {}
    for line in open(base + ".meta.txt"):
        p = line.split(); m[p[0]] = p[1:]
    n = int(m["npairs"][0]); raw = open(base + ".bin", "rb").read()
    rel = np.frombuffer(raw[:3*n*8], dtype=np.float64).reshape(n,3).copy()
    comp = np.frombuffer(raw[3*n*8:3*n*8+n*4], dtype=np.int32).copy()
    out = np.frombuffer(raw[3*n*8+n*4:], dtype=np.float64).copy()
    u = np.array([float(m["ux"][0]),float(m["uy"][0]),float(m["uz"][0])])
    return n, rel, comp, out, u

for base in ["gpu1_sing_in", "gpu1_sing_band"]:
    n, rel, comp, out, u = load(base)
    print("=== %s (n=%d) ===" % (base, n))
    # compare GPU to indfield at several orders, worst per-component rel err
    for order in (8, 12, 16, 24):
        worst = 0.0
        for i in range(0, n, max(1, n//40)):
            xo = rel[i] + OFFSETS[comp[i]]
            e, b = I.integrate_point(xo, np.zeros(3), u, 1.0,1.0,1.0, order=order, refine=0, chunk=8)
            ref = e[comp[i]] if comp[i]<3 else b[comp[i]-3]
            g = out[i]
            if not (np.isfinite(ref) and np.isfinite(g)): continue
            err = abs(g-ref)/(1.0+max(abs(g),abs(ref)))
            worst = max(worst, err)
        print("   GPU(sing o8) vs indfield o%d r0 : worst rel err = %.3e" % (order, worst))
    # also: indfield o8 vs o24 (pure discretization, no GPU)
    d8=0.0; d24=0.0
    for i in range(0, n, max(1, n//40)):
        xo = rel[i] + OFFSETS[comp[i]]
        e8,b8 = I.integrate_point(xo,np.zeros(3),u,1,1,1,order=8,refine=0,chunk=8)
        e24,b24 = I.integrate_point(xo,np.zeros(3),u,1,1,1,order=24,refine=0,chunk=8)
        r8 = e8[comp[i]] if comp[i]<3 else b8[comp[i]-3]
        r24 = e24[comp[i]] if comp[i]<3 else b24[comp[i]-3]
        if not (np.isfinite(r8) and np.isfinite(r24)): continue
        d8 = max(d8, abs(r8-r24)/(1.0+max(abs(r8),abs(r24))))
    print("   indfield o8 vs o24 (discretization only) : worst rel err = %.3e" % d8)
print("DIAG_DONE")
