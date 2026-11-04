#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
"""Generate the converged reference table used by refkernel_port accuracy.

The reference is the independently formulated TSC-convolution integrator
``indfield.py`` (knot split, power transform at the singular corner, runtime
Gauss-Legendre rules).  It is not derived from the ported code.

Usage:
    python3 make_reference.py /path/to/indfield.py > accuracy_ref.txt
"""
import os
import sys

import numpy as np


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: make_reference.py /path/to/indfield.py\n")
        return 2
    sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[1])))
    import indfield as ind  # noqa: E402

    u10 = np.sqrt(99.0)  # gamma = 10
    oblique = (u10 / np.sqrt(3.0),) * 3
    points = [
        ("int", (0.3, 0.2, 0.1)),
        ("face", (1.5, 0.4, 0.3)),
        ("out", (2.2, 0.4, 0.3)),
    ]
    # (name, u, order, refine).  The oblique interior point needs a higher order
    # because the power-transformed peak is narrower there; the values below
    # agree with order+4 to <=1e-6.
    cases = [
        ("g1", (0.0, 0.0, 0.0), 12, 3),
        ("g10_ax", (0.0, 0.0, u10), 12, 3),
        ("g10_ob", oblique, 16, 3),
    ]
    print("# label x y z ux uy uz Ex Ey Ez Bx By Bz")
    for point_name, x in points:
        for case_name, u, order, refine in cases:
            E, B = ind.integrate_point(
                np.array(x), np.zeros(3), np.array(u), 1.0, 1.0, 1.0, order=order, refine=refine
            )
            values = [*x, *u, *E, *B]
            print(f"{point_name}_{case_name} " + " ".join(f"{v:.17g}" for v in values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
