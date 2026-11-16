#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""
CI revalidation of the velocity-bin in-bunch saturation claim.

Runs the Liénard-Wiechert direct-sum arbiter on the committed real-bunch
fixture (tests/data/velocity_bins_real_bunch.txt, 49931 particles, gamma ~ 490)
with the FULL set of bin counts (1..256), and checks two things:

  1. The published per-component relative-error table
     (docs/measurements/velocity_bin_real_bunch_calibration.txt) is reproduced
     within a stated relative tolerance.  This is the reproducibility gate: if
     the arbiter, the fixture, or the binning changes, the table changes and
     the test fails.

  2. The SATURATION invariant: no component reaches the 1e-3 in-bunch
     tolerance at any box count up to 256.  This is the claim the auto policy
     rests on (the in-bunch error saturates above the tolerance, so `auto`
     caps at the practical cost ceiling instead of returning the 256 cap).  If
     a code change makes the binned model meet 1e-3 in-bunch, this test fails
     so the auto policy and the documentation are revisited in the same change.

The test is the arbiter requirement: the central claim of the velocity-bin
auto policy (tolerance unreachable, error saturates) is revalidated by CI from
a committed fixture, not only from a 4-particle synthetic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arbiter_in_bunch as arbiter  # noqa: E402

# The full set of bin counts the auto policy considers.
BINS = [1, 2, 4, 8, 16, 32, 64, 256]
# The 1e-3 in-bunch tolerance the auto policy targets.
TOLERANCE = 1e-3
# Relative tolerance for the table reproduction check.  The arbiter is
# deterministic (a direct sum), so a tight tolerance catches any change to the
# binning, the probe nodes, or the reference.  5% is generous to absorb
# platform floating-point differences while still catching a real change.
TABLE_REL_TOL = 0.05

# Published per-component relative error (max over probe nodes) and By_abs_max
# (T), from docs/measurements/velocity_bin_real_bunch_calibration.txt, for the
# 49931-particle fixture.  Order: Ex, Ey, Ez, Bx, By, Bz, By_abs_max.
PUBLISHED = {
    1:   (1.620e+01, 2.521e+00, 1.026e+01, 1.025e+01, 2.447e+00, 1.621e+01, 3.511e-01),
    2:   (9.348e+00, 2.360e+00, 1.135e+01, 1.135e+01, 6.729e+00, 9.352e+00, 5.209e-01),
    4:   (4.049e+00, 5.850e+00, 1.317e+00, 1.317e+00, 1.293e+01, 4.034e+00, 3.935e-01),
    8:   (3.621e+00, 7.014e+00, 2.492e+00, 2.491e+00, 1.692e+01, 3.607e+00, 2.169e-01),
    16:  (3.598e+00, 1.085e+01, 2.132e+00, 2.131e+00, 1.272e+01, 3.584e+00, 2.271e-01),
    32:  (4.099e+00, 6.540e+00, 3.102e+00, 3.101e+00, 2.180e+01, 4.082e+00, 1.126e-01),
    64:  (2.383e+00, 5.301e+00, 3.191e+00, 3.190e+00, 9.622e+00, 2.373e+00, 1.394e-01),
    256: (1.020e+00, 4.649e+00, 2.041e+00, 2.041e+00, 3.703e+00, 1.020e+00, 6.441e-02),
}
COMPONENTS = ("Ex", "Ey", "Ez", "Bx", "By", "Bz")


def _relative_error(bunch_path: Path) -> dict[int, np.ndarray]:
    """Compute the per-component relative error (max over nodes) + By_abs for
    each bin count, using the arbiter's own reference, probe, and binning."""
    x, u, wq = arbiter._load_text(bunch_path)
    N = len(x)
    ac = np.abs(wq)
    tot = ac.sum()
    core = np.array([(ac * x[:, 0]).sum(), (ac * x[:, 1]).sum(), (ac * x[:, 2]).sum()]) / tot
    sx = np.sqrt((ac * (x[:, 0] - core[0]) ** 2).sum() / tot)
    sz = np.sqrt((ac * (x[:, 2] - core[2]) ** 2).sum() / tot)
    sigmas = [s for s in (2, 5, 10) if s <= 10]
    probes = [core.copy()]
    for m in sigmas:
        for axis, s in (("x", sx), ("z", sz)):
            p = core.copy()
            p[0 if axis == "x" else 2] += m * s
            probes.append(p.copy())
            p2 = core.copy()
            p2[0 if axis == "x" else 2] -= m * s
            probes.append(p2.copy())
    # Reference T (per-particle, full set).
    T = np.zeros((len(probes), 6))
    for ni in range(len(probes)):
        E, Bf = arbiter.lw_field(probes[ni], x, u, wq)
        T[ni] = [E[0], E[1], E[2], Bf[0], Bf[1], Bf[2]]
    amp = np.abs(T).copy()
    amp[amp < 1e-30] = np.nan

    result = {}
    for B in BINS:
        # Same beam-frame split as the arbiter.
        ubeam = np.column_stack([-u[:, 0], u[:, 2], u[:, 1]])
        lo = ubeam.min(axis=0)
        hi = ubeam.max(axis=0)
        rng = hi - lo
        pa = {1: (1, 1, 1), 2: (1, 2, 1), 4: (2, 2, 1), 8: (2, 4, 1),
              16: (4, 4, 1), 32: (8, 4, 1), 64: (8, 8, 1),
              256: (16, 16, 1)}[B]
        idx = np.zeros((N, 3), dtype=np.int64)
        for a in range(3):
            if pa[a] > 1 and rng[a] > 0:
                idx[:, a] = np.clip(np.floor((ubeam[:, a] - lo[a]) / (rng[a] / pa[a])),
                                    0, pa[a] - 1)
        box = idx[:, 0] + pa[0] * (idx[:, 1] + pa[1] * idx[:, 2])
        boxes = pa[0] * pa[1] * pa[2]
        rep = np.zeros((boxes, 3))
        sw = np.zeros(boxes)
        np.add.at(rep, box, wq[:, None] * u)
        np.add.at(sw, box, wq)
        used = sw != 0
        rep[used] /= sw[used, None]
        u_pp = rep[box]
        Tb = np.zeros((len(probes), 6))
        for ni in range(len(probes)):
            E, Bf = arbiter.lw_field(probes[ni], x, u_pp, wq)
            Tb[ni] = [E[0], E[1], E[2], Bf[0], Bf[1], Bf[2]]
        rel = np.abs(Tb - T)
        maxnode = np.nanmax(rel / amp, axis=0)
        by_abs = float(np.abs(Tb[:, 4] - T[:, 4]).max())
        result[B] = np.array(list(maxnode) + [by_abs])
    return result


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: validate_saturation.py <bunch.txt>", file=sys.stderr)
        return 2
    bunch_path = Path(argv[1])
    if not bunch_path.exists():
        print(f"FAIL: fixture {bunch_path} not found", file=sys.stderr)
        return 1

    print(f"validate_saturation: bunch {bunch_path} (bins {BINS})", flush=True)
    result = _relative_error(bunch_path)

    # 1. Reproduce the published table within the stated tolerance.
    failures = []
    for B in BINS:
        measured = result[B]
        published = PUBLISHED[B]
        for i in range(6):  # Ex..Bz relative error
            m, pub = measured[i], published[i]
            if not (abs(m - pub) <= TABLE_REL_TOL * abs(pub)):
                failures.append(
                    f"  B={B} {COMPONENTS[i]}: measured {m:.4e} vs published {pub:.4e} "
                    f"(rel tol {TABLE_REL_TOL})")
        mby, pubby = measured[6], published[6]
        if not (abs(mby - pubby) <= TABLE_REL_TOL * abs(pubby)):
            failures.append(
                f"  B={B} By_abs: measured {mby:.4e} T vs published {pubby:.4e} T "
                f"(rel tol {TABLE_REL_TOL})")
    if failures:
        print("FAIL: table reproduction mismatch:\n" + "\n".join(failures), file=sys.stderr)
        return 1
    print("  table reproduction: OK (within %.0f%% relative tolerance)"
          % (100 * TABLE_REL_TOL), flush=True)

    # 2. The saturation invariant: no component reaches the 1e-3 in-bunch
    #    tolerance at any box count up to 256.  This is the claim the auto
    #    policy rests on; if it ever becomes false, the auto policy and the
    #    documentation must be revisited.
    for B in BINS:
        for i in range(6):
            if result[B][i] <= TOLERANCE:
                print(f"FAIL: B={B} {COMPONENTS[i]} relative error "
                      f"{result[B][i]:.4e} <= {TOLERANCE}: the in-bunch "
                      f"tolerance is met, so the saturation claim is false; "
                      f"revisit the auto policy and the docs", file=sys.stderr)
                return 1
    print("  saturation invariant: OK (no component reaches %.0e at any count "
          "up to 256)" % TOLERANCE, flush=True)

    # 3. Report the achieved accuracy at the practical cap (16) for the record.
    print(f"  achieved in-bunch error at the practical cap (16): "
          f"max component rel {max(result[16][:6]):.4e}, "
          f"By_abs {result[16][6]:.4e} T", flush=True)
    print("PASS: velocity-bin in-bunch saturation revalidated from the fixture",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
