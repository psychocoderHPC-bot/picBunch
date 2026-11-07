// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Allowed orders of the Duffy-mapped oblique singular box and the shared
// host/device `auto` selection rule.
//
// The oblique box is evaluated in Lorentz-contracted coordinates.  At high
// gamma the beamed peak narrows by 1/gamma, so an order-4 quadrature
// under-resolves it.  The rule below is intentionally cheap: it depends only
// on the pair's gamma.  The caller applies it per tile to the largest gamma of
// the *oblique, support-overlapping* sources, which keeps one coherent order
// for the whole kernel launch.  Axial pairs never enter this path (they use
// the exact gamma-stretched box), so raising the tile order cannot change
// them.
#pragma once

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
} // namespace picBunch::pbfast
