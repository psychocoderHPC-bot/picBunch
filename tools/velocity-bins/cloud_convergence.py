import numpy as np, sys, time
import cloud_reference as cr
from cloud_reference import cloud_field, K, C, CELL, NQUAD

def load(path):
    d = np.loadtxt(path).astype(np.float64); d = d[1:]
    return d[:, 0:3], d[:, 3:6], d[:, 6] * d[:, 7]

def probes(x, wq):
    ac = np.abs(wq); tot = ac.sum()
    core = np.array([(ac * x[:, 0]).sum(), (ac * x[:, 1]).sum(), (ac * x[:, 2]).sum()]) / tot
    sx = np.sqrt((ac * (x[:, 0] - core[0]) ** 2).sum() / tot)
    sz = np.sqrt((ac * (x[:, 2] - core[2]) ** 2).sum() / tot)
    ps = [("core", core.copy())]
    for m in (2, 5, 10):
        for axis, s in (("x", sx), ("z", sz)):
            p = core.copy(); p[0 if axis == "x" else 2] += m * s
            ps.append(("%s+%d" % (axis, m), p.copy()))
            p2 = core.copy(); p2[0 if axis == "x" else 2] -= m * s
            ps.append(("%s-%d" % (axis, m), p2.copy()))
    return core, sx, sz, ps

def binning(x, u, wq, B):
    ubeam = np.column_stack([-u[:, 0], u[:, 2], u[:, 1]])
    lo = ubeam.min(0); hi = ubeam.max(0); rng = hi - lo
    pa = {1: (1,1,1), 2: (1,2,1), 4: (2,2,1), 8: (2,4,1), 16: (4,4,1),
          32: (8,4,1), 64: (8,8,1), 256: (16,16,1)}.get(B, (1,1,1))
    N = len(x); idx = np.zeros((N, 3), np.int64)
    for a in range(3):
        if pa[a] > 1 and rng[a] > 0:
            idx[:, a] = np.clip(np.floor((ubeam[:, a] - lo[a]) / (rng[a] / pa[a])), 0, pa[a] - 1)
    box = idx[:, 0] + pa[0] * idx[:, 1] + pa[0] * pa[1] * idx[:, 2]
    boxes = pa[0] * pa[1] * pa[2]
    rep = np.zeros((boxes, 3)); sw = np.zeros(boxes)
    np.add.at(rep, box, wq[:, None] * u); np.add.at(sw, box, wq)
    used = sw != 0
    rep[used] /= sw[used, None]
    return rep[box]

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'real_B1_bunch.txt'
    x, u, wq = load(path)
    N = len(x)
    core, sx, sz, ps = probes(x, wq)
    names = [n for n, _ in ps]; nodes = [v for _, v in ps]
    print("CLOUD REFERENCE (coarse quad NQUAD=%d), N=%d" % (NQUAD, N), flush=True)
    print("core % .6e % .6e % .6e  sigma x %.3e z %.3e" % (core[0], core[1], core[2], sx, sz), flush=True)
    t0 = time.time()
    T = np.zeros((len(nodes), 6))
    for ni in range(len(nodes)):
        T[ni] = cloud_field(nodes[ni], x, u, wq)
        print("  reference node %-6s (%.1f s)" % (names[ni], time.time() - t0), flush=True)
    print("REFERENCE (cloud, per-particle):", flush=True)
    hdr = "node"
    for c in "Ex Ey Ez Bx By Bz".split():
        hdr += "%12s" % c
    print(hdr, flush=True)
    for ni, nm in enumerate(names):
        print("%-6s" % nm + "".join("%12.3e" % T[ni, c] for c in range(6)), flush=True)
    amp = np.abs(T).copy(); amp[amp < 1e-30] = np.nan
    print("\nRELATIVE ERROR (cloud-binned / cloud-reference, per component, max over nodes); By abs:", flush=True)
    print("   B  |" + "".join("%12s" % c for c in "Ex Ey Ez Bx By Bz".split()) + "|  By_abs_max", flush=True)
    for B in [1, 2, 4, 8, 16, 32, 64, 256]:
        t0 = time.time()
        u_pp = binning(x, u, wq, B)
        Tb = np.zeros((len(nodes), 6))
        for ni in range(len(nodes)):
            Tb[ni] = cloud_field(nodes[ni], x, u_pp, wq)
        rel = np.abs(Tb - T)
        maxnode = np.nanmax(rel / amp, axis=0)
        by_abs = float(np.abs(Tb[:, 4] - T[:, 4]).max())
        print("B=%-4d|%s|  %12.3e   (%.0f s)" % (B, "".join("%12.3e" % float(maxnode[c]) for c in range(6)), by_abs, time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
