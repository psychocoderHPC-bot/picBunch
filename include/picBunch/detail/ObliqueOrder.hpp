// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Compatibility ordering knob and the shared host/device `auto` rule.
//
// The oblique in-support path is now an exact radial / polar-aligned angular
// integral in Lorentz-contracted coordinates and does not depend on this
// order.  The values are retained so existing CLI, TOML, restart, MPI and
// device-table plumbing keeps working.  The caller still resolves one order
// per tile from the largest oblique-overlapping gamma, which keeps the
// kernel launch coherent.  Axial pairs never enter the oblique path.
#pragma once

#include <cmath>

// Only the host/device annotation on quadratureOrderFor is accelerator
// specific.  Pulling the full alpaka header into every CUDA translation unit
// that includes this ordering knob is unnecessary; the compiler builtins keep
// this header dependency-free.
#if defined(__CUDACC__) || defined(__HIPCC__)
#    define PICBUNCH_ORDER_HOST_ACC __host__ __device__
#else
#    define PICBUNCH_ORDER_HOST_ACC
#endif

namespace picBunch::pbfast
{
    constexpr int kObliqueOrderMin = 4;
    constexpr int kObliqueOrderMax = 12;
    constexpr int kObliqueOrderCount = 5;
    //! The orders with a precomputed Gauss-Legendre rule.
    constexpr int kObliqueOrders[kObliqueOrderCount] = {4, 6, 8, 10, 12};

    //! Map an explicit request to an allowed even order in [4, 12].
    constexpr int normalizeObliqueOrder(int requested)
    {
        int order = requested;
        if(order < kObliqueOrderMin)
            order = kObliqueOrderMin;
        if(order > kObliqueOrderMax)
            order = kObliqueOrderMax;
        if(order % 2 != 0)
            ++order;
        return order;
    }

    /*! Gamma-driven order for one output tile.
     *
     * Thresholds: gamma <= 12 keeps order 4.  The 12 rather than 10 leaves
     * the 5%-spread gamma-10 control (max gamma 11.2) on the historical order
     * 4 path bit for bit.  12 < gamma <= 40 uses 6, and gamma > 40 uses 8.
     * At gamma 100 the contracted support is about 1/100 of a cell across,
     * which order 4 cannot resolve; order 8 is the requested minimum there
     * and stays well below the safe cap.
     */
    constexpr int autoObliqueOrder(double gamma)
    {
        if(!(gamma > 12.0))
            return 4;
        if(gamma <= 40.0)
            return 6;
        return 8;
    }

    //! Gauss-order cap for a remaining smooth-remainder quadrature.  Above the
    //! cap the pair is routed to the exact s-space evaluator instead of paying
    //! an ever higher rule (device tables stop at 32 as well).
    constexpr int kQuadratureOrderCap = 32;

    /*! Route out-of-support pairs whose contracted d/gamma tube the fixed-order
     * remainder cannot resolve.  At gamma 100 the tube is 1/100 of a cell
     * across; the order-8 threshold is the validated gamma-100 design rule and
     * larger gamma scales the requirement with it.
     */
    constexpr double kOutOfSupportExactGamma = 100.0;

    /*! Whether a pair is in the gamma regime where legacy quadrature is
     * unusable.  The tiny slack absorbs the floating-point construction of
     * gamma=100, whose sqrt(1+u^2) lands a few ulp below the integer.
     */
    constexpr bool isExactGamma(double gamma)
    {
        return gamma >= kOutOfSupportExactGamma - 1.0e-9;
    }

    /*! Gauss order of the smooth remainder for one pair.
     *
     * At or above the default 1e-3 budget this is exactly the caller's
     * requested order, i.e. the historical base remainder rule.  It does not
     * grow with gamma: the gamma-10 oblique control and every gamma below the
     * exact-only threshold keep the base rule, and gamma >= 100 is routed to
     * the exact evaluator instead of paying a larger rule.  Only a tolerance
     * tighter than 1e-3 raises the order, one node per decade.  ``gamma`` is
     * kept in the signature for host/device call-site compatibility.
     */
    inline PICBUNCH_ORDER_HOST_ACC int quadratureOrderFor(
        double gamma, double tolerance, int requested)
    {
        (void)gamma;
        int const order = requested >= 2 ? requested : 5;
        double tol = (tolerance > 0.0 && std::isfinite(tolerance)) ? tolerance : 1e-3;
        int extra = 0;
        while(tol < 1e-3 && extra < 16)
        {
            tol *= 10.0;
            ++extra;
        }
        return order + extra;
    }
} // namespace picBunch::pbfast
