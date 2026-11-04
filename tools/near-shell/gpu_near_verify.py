#!/usr/bin/env python3
"""Verify the GPU sing kernel (with refinement) matches indfield at the SAME (order, refine).
Usage: verify_ref.py <order> <refine>"""
import sys, os, subprocess
import numpy as np
sys.path.insert(0, os.path.expanduser("~/pr24/pr24w53-ref"))
import indfield as I

order = int(sys.argv[1]); refine = int(sys.argv[2])
for band in ["in", "band"]:
    base = "vref_%s_o%d_r%d" % (band, order, refine)
    env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
    env["CUDA_VISIBLE_DEVICES"] = "1"
    subprocess.run(["./gpu_near", "--kernel", "sing", "--band", band, "--order", str(order),
                    "--refine", str(refine), "--npairs", "120", "--iters", "1",
                    "--sample", "--out", base, "--nocheck", "--nobs"],
                   env=env, check=True, stdout=subprocess.DEVNULL)
    m = {}
    for line in open(base + ".meta.txt"):
        p = line.split(); m[p[0]] = p[1:]
    n = int(m["npairs"][0]); raw = open(base + ".bin", "rb").read()
    rel = np.frombuffer(raw[:3 * n * 8], dtype=np.float64).reshape(n, 3).copy()
    comp = np.frombuffer(raw[3 * n * 8:3 * n * 8 + n * 4], dtype=np.int32).copy()
    out = np.frombuffer(raw[3 * n * 8 + n * 4:], dtype=np.float64).copy()
    u = np.array([float(m["ux"][0]), float(m["uy"][0]), float(m["uz"][0])])
    worst = 0.0
    for i in range(n):
        xo = rel[i] + I.OFFSETS[comp[i]]
        e, b = I.integrate_point(xo, np.zeros(3), u, 1, 1, 1, order=order, refine=refine, chunk=8)
        ref = e[comp[i]] if comp[i] < 3 else b[comp[i] - 3]
        err = abs(out[i] - ref) / (1.0 + max(abs(out[i]), abs(ref)))
        worst = max(worst, err)
    print("verify GPU(o%d r%d) vs indfield(o%d r%d) [%s]: worst = %.3e" % (
        order, refine, order, refine, band, worst), flush=True)
print("VERIFY_DONE", flush=True)
