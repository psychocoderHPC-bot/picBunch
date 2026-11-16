"""Cloud reference with COARSE quadrature (order 2, refine 0, NQUAD=216),
validated to machine precision for the single cloud.  Fast enough for the full
convergence study."""
import numpy as np
import sys
import indfield

K = 8.9875517873681764e9
C = 299792458.0
CELL = 2e-7
ORDER = 2
REFINE = 0

def _axis_rule(order, refine):
    xi, wi = np.polynomial.legendre.leggauss(order)
    knots = indfield.TSC_KNOTS
    s, w = [], []
    for a, b in zip(knots[:-1], knots[1:]):
        nseg = 2 ** refine
        for seg in range(nseg):
            t0 = seg / nseg; t1 = (seg + 1) / nseg
            mid = 0.5 * (t0 + t1); half = 0.5 * (t1 - t0)
            t = mid + half * xi; sn = a + (b - a) * t
            ww = wi * half * (b - a) * indfield.tsc_shape(sn)
            s.append(sn); w.append(ww)
    return np.concatenate(s), np.concatenate(w)

SX, WX = _axis_rule(ORDER, REFINE)
SY, WY = _axis_rule(ORDER, REFINE)
SZ, WZ = _axis_rule(ORDER, REFINE)
_sxx, _syy, _szz = np.meshgrid(SX, SY, SZ, indexing='ij')
WXv = (WX[:, None, None] * WY[None, :, None] * WZ[None, None, :]).ravel()
Sxx = _sxx.ravel(); Syy = _syy.ravel(); Szz = _szz.ravel()
NQUAD = len(Sxx)
SOFF = np.stack([Sxx * CELL, Syy * CELL, Szz * CELL], axis=-1)

def cloud_field(node, xs, us, wqs, batch=256):
    """Sum over all clouds at the Yee-staggered node (6 components), vectorized."""
    out = np.zeros(6)
    n = len(xs)
    for c in range(6):
        xobs = node + indfield.OFFSETS[c] * CELL
        tot = 0.0
        for lo in range(0, n, batch):
            hi = min(n, lo + batch)
            xb = xs[lo:hi]; ub = us[lo:hi]; wb = wqs[lo:hi]
            R = xobs[None, None, :] - xb[:, None, :] - SOFF[None, :, :]
            uu = (ub * ub).sum(1)
            g = np.sqrt(1.0 + uu)
            b = ub / g[:, None]
            D = np.einsum('pi,pni->pn', b, R)
            r2 = (R * R).sum(2)
            b2 = uu / (g * g)
            den = r2 - (b2[:, None] * r2 - D * D)
            with np.errstate(divide="ignore", invalid="ignore"):
                E = np.where(uu[:, None, None] == 0.0,
                             R / (r2 ** 1.5)[:, :, None],
                             (1.0 - b2)[:, None, None] * R / (den ** 1.5)[:, :, None])
            B = np.zeros_like(E)
            B[:, :, 0] = b[:, 0:1] * E[:, :, 2] - b[:, 1:2] * E[:, :, 1]
            B[:, :, 1] = b[:, 1:2] * E[:, :, 0] - b[:, 0:1] * E[:, :, 2]
            B[:, :, 2] = b[:, 0:1] * E[:, :, 1] - b[:, 1:2] * E[:, :, 0]
            E = np.where(np.isfinite(E), E, 0.0)
            B = np.where(np.isfinite(B), B, 0.0)
            val = (E if c < 3 else B)[:, :, (c if c < 3 else c - 3)]
            Ei = (val * WXv[None, :]).sum(1)
            tot += float((wb * Ei).sum())
        out[c] = tot * (K if c < 3 else K / C)
    return out
