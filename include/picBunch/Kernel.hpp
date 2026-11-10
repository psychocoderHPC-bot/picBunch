// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/Types.hpp"

#include <cmath>

// Use alpaka's host/device annotation and the SIMD-aware math functors from
// the single alpaka math header.  Include it unconditionally so ALPAKA_FN_ACC
// is defined for host and device translation units alike and no device-callable
// function reaches for std:: math.
#include <alpaka/core/common.hpp>
#include <alpaka/math.hpp>

namespace picBunch
{
    //! Support radius of a shape in cells, compile-time per shape.
    template<Shape S>
    ALPAKA_FN_ACC inline constexpr double shapeSupport()
    {
        if constexpr(S == Shape::CIC)
            return 1.0;
        else if constexpr(S == Shape::TSC)
            return 1.5;
        else
            return 2.0;
    }

    //! Assignment weight of a shape, argument in cell units.  Compile-time shape.
    template<Shape S>
    ALPAKA_FN_ACC inline double shapeWeight(double x)
    {
        double const ax = alpaka::math::abs(x);
        if constexpr(S == Shape::CIC)
        {
            return ax < 1.0 ? 1.0 - ax : 0.0;
        }
        else if constexpr(S == Shape::TSC)
        {
            if(ax < 0.5)
                return 0.75 - ax * ax;
            if(ax < 1.5)
                return 0.5 * (1.5 - ax) * (1.5 - ax);
            return 0.0;
        }
        else
        {
            if(ax < 1.0)
                return (4.0 - 6.0 * ax * ax + 3.0 * ax * ax * ax) / 6.0;
            if(ax < 2.0)
                return (2.0 - ax) * (2.0 - ax) * (2.0 - ax) / 6.0;
            return 0.0;
        }
    }

    //! Runtime wrapper for non-hot paths and tests.
    ALPAKA_FN_ACC inline double shapeWeight(Shape shape, double x)
    {
        if(shape == Shape::CIC)
            return shapeWeight<Shape::CIC>(x);
        if(shape == Shape::TSC)
            return shapeWeight<Shape::TSC>(x);
        return shapeWeight<Shape::PQS>(x);
    }

    ALPAKA_FN_ACC inline constexpr double shapeSupport(Shape shape)
    {
        return shape == Shape::CIC ? shapeSupport<Shape::CIC>()
            : (shape == Shape::TSC ? shapeSupport<Shape::TSC>() : shapeSupport<Shape::PQS>());
    }

    /** TSC assignment function, argument in cell units. */
    ALPAKA_FN_ACC inline double tscShape(double x)
    {
        double const ax = alpaka::math::abs(x);
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
    ALPAKA_FN_ACC inline void pointFieldNormalized(
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
        double const gamma = alpaka::math::sqrt(1.0 + uu2);
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
            double const r3 = r2 * alpaka::math::sqrt(r2);
            double const f = (r3 > 0.0) ? 1.0 / r3 : 0.0;
            Ex = f * Rx;
            Ey = f * Ry;
            Ez = f * Rz;
        }
        else
        {
            double const f = (D > 0.0) ? invG2 / (D * alpaka::math::sqrt(D)) : 0.0;
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
