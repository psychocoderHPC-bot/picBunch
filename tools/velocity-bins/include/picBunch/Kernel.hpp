// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#    define PICBUNCH_HD __host__ __device__ inline
#else
#    define PICBUNCH_HD inline
#endif

namespace picBunch
{
    /** TSC assignment function, argument in cell units. */
    PICBUNCH_HD double tscShape(double x)
    {
        double const ax = fabs(x);
        if(ax < 0.5)
            return 0.75 - ax * ax;
        if(ax < 1.5)
        {
            double const t = 1.5 - ax;
            return 0.5 * t * t;
        }
        return 0.0;
    }

    /** Field of a uniformly moving point charge, reference normalization.
     *
     * Returns
     *   E_n = 4*pi*eps0 * E / q
     *   B_n = 4*pi*eps0 * c * B / q
     * for displacement ``R`` (physical) and dimensionless momentum ``u``.
     * This is the algebraic form of the independently validated reference.
     * E is odd in ``R``; B is odd in ``R`` and odd in ``u``.
     */
    PICBUNCH_HD void pointFieldNormalized(
        double Rx,
        double Ry,
        double Rz,
        double ux,
        double uy,
        double uz,
        double* out)
    {
        double const uu2 = ux * ux + uy * uy + uz * uz;
        double const invG2 = 1.0 / (1.0 + uu2);
        double const gamma = sqrt(1.0 + uu2);
        double const bx = ux / gamma;
        double const by = uy / gamma;
        double const bz = uz / gamma;

        double const r2 = Rx * Rx + Ry * Ry + Rz * Rz;
        double const rb = Rx * bx + Ry * by + Rz * bz;
        // |R|^2 - |beta x R|^2 = |R|^2 (1-beta^2) + (R.beta)^2
        double const D = r2 * invG2 + rb * rb;

        double Ex, Ey, Ez;
        if(uu2 == 0.0)
        {
            // stationary limit: exact Coulomb field
            double const r3 = r2 * sqrt(r2);
            double const f = (r3 > 0.0) ? 1.0 / r3 : 0.0;
            Ex = f * Rx;
            Ey = f * Ry;
            Ez = f * Rz;
        }
        else
        {
            double const f = (D > 0.0) ? invG2 / (D * sqrt(D)) : 0.0;
            Ex = f * Rx;
            Ey = f * Ry;
            Ez = f * Rz;
        }
        out[0] = Ex;
        out[1] = Ey;
        out[2] = Ez;
        if(uu2 == 0.0)
        {
            out[3] = out[4] = out[5] = 0.0;
        }
        else
        {
            out[3] = by * Ez - bz * Ey;
            out[4] = bz * Ex - bx * Ez;
            out[5] = bx * Ey - by * Ex;
        }
    }

} // namespace picBunch
