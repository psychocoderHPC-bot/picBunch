// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group: host/device consistency of the fast near-field evaluator.
//
// The exact s-space evaluator exists twice: include/picBunch/detail/FastNear.hpp
// (host, public shape_field) and src/FastNearDevice.cpp (generic alpaka device
// kernel, public evalFastNearDevice).  This test exercises both public
// evaluators on the same grid of gamma, geometry and distance cases and fails
// when the backend agreement exceeds the documented condition-number bound.
//
// It also pins the knot constants: the host aliases in pbfast::detail and the
// device aliases in pbfast::dconst must be the same values that both
// implementations read from FastNearConstants.hpp.

#include "picBunch/Types.hpp"
#include "picBunch/detail/FastNear.hpp"
#include "picBunch/detail/FastNearConstants.hpp"
#include "picBunch/detail/FastNearDevice.hpp"
#include "picBunch/detail/ObliqueOrder.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace
{
    using nbref::Vec3;
    namespace pbfast = picBunch::pbfast;

    struct FastParams
    {
        pbfast::DeviceParams params{};
        pbfast::Settings settings{};
    };

    FastParams makeFastParams(int order, int obliqueOrder, int exactCellReach, double tolerance)
    {
        FastParams out;
        out.settings.order = order;
        out.settings.obliqueOrder = obliqueOrder;
        out.settings.exactCellReach = exactCellReach;
        out.settings.tolerance = tolerance;

        auto& p = out.params;
        p.order = order;
        p.obliqueOrder = obliqueOrder;
        p.exactCellReach = exactCellReach;
        p.tolerance = tolerance;
        auto const& r = pbfast::detail::rule(order);
        for(int i = 0; i < order; ++i)
        {
            p.xi[i] = r.x[static_cast<std::size_t>(i)];
            p.wi[i] = r.w[static_cast<std::size_t>(i)];
        }
        for(int slot = 0; slot < pbfast::kObliqueOrderCount; ++slot)
        {
            int const n = pbfast::kObliqueOrders[slot];
            auto const& o = pbfast::detail::rule(n);
            for(int i = 0; i < n; ++i)
            {
                p.oxi[slot][i] = o.x[static_cast<std::size_t>(i)];
                p.owi[slot][i] = o.w[static_cast<std::size_t>(i)];
            }
        }
        p.unitE = 1.0;
        p.unitB = 1.0;
        return out;
    }

    picBunch::Bunch singleSource(Vec3 const& u)
    {
        picBunch::Bunch source;
        source.resize(1);
        source.x[0] = source.y[0] = source.z[0] = 0.0;
        source.ux[0] = u.x;
        source.uy[0] = u.y;
        source.uz[0] = u.z;
        source.weight[0] = 1.0;
        source.charge[0] = 1.0;
        return source;
    }

    /*! Combined L2 error of the six field components, normalised by the
     * larger of the reference magnitude and the unit field scale.
     *
     * Cells and volume are 1 in this test, so |F| = 1 is the natural O(1)
     * field scale of one unit charge.  The floor keeps a poorly resolved
     * cancellation tail (|F| << 1) from turning a small absolute backend
     * difference into a meaningless relative error, while keeping full
     * relative sensitivity for every O(1) field.
     */
    double relativeError(std::array<double, 6> const& got, std::array<double, 6> const& ref)
    {
        double d2 = 0.0;
        double r2 = 0.0;
        for(int c = 0; c < 6; ++c)
        {
            double const d = got[static_cast<std::size_t>(c)] - ref[static_cast<std::size_t>(c)];
            d2 += d * d;
            r2 += ref[static_cast<std::size_t>(c)] * ref[static_cast<std::size_t>(c)];
        }
        return std::sqrt(d2) / std::sqrt(std::max(r2, 1.0));
    }

    //! Dimensionless momentum for a requested gamma.  0 = axial (z), 1 = 45 deg.
    Vec3 momentum(double gamma, bool oblique)
    {
        double const p = std::sqrt(std::max(gamma * gamma - 1.0, 0.0));
        if(!oblique)
            return {0.0, 0.0, p};
        double const q = p / std::sqrt(2.0);
        return {q, 0.0, q};
    }

    Vec3 normalized(Vec3 const& v)
    {
        double const n = std::sqrt(nbref::dot(v, v));
        return {v.x / n, v.y / n, v.z / n};
    }

    struct Geometry
    {
        char const* name;
        Vec3 obs;
        bool inSupport; //!< |s| < 1.5 on every axis
        bool knotCorner; //!< origin near a TSC face/edge/corner intersection
    };

    std::vector<Geometry> geometries()
    {
        std::vector<Geometry> out;
        // A generic in-support ray and a ray that leaves through the +z face.
        Vec3 const dirIn = normalized(Vec3{0.42, -0.55, 0.72});
        Vec3 const dirOut = normalized(Vec3{0.22, -0.34, 0.91});
        // d is the source-relative distance in cells (cells = 1 here).
        for(double d : {0.05, 0.2, 0.7, 1.2, 1.45})
            out.push_back({"in-support", dirIn * d, true, false});
        for(double d : {1.8, 2.0})
            out.push_back({"out-of-support", dirOut * d, false, false});
        // On-knot faces, edges and corners (TSC knots at +-0.5 and +-1.5).
        out.push_back({"knot-face", Vec3{0.5, 0.13, -0.22}, true, false});
        out.push_back({"knot-face-2", Vec3{0.5, -0.4, 0.3}, true, false});
        out.push_back({"knot-edge", Vec3{0.5, -0.5, 0.11}, true, false});
        out.push_back({"knot-edge-2", Vec3{-0.5, 0.5, 0.27}, true, false});
        out.push_back({"knot-corner", Vec3{0.5, -0.5, 0.5}, true, true});
        out.push_back({"knot-corner-2", Vec3{-0.5, 0.5, -0.5}, true, true});
        out.push_back({"knot-outer-face", Vec3{1.5, 0.21, -0.33}, true, false});
        return out;
    }

    /*! Backend agreement bound for one gamma/geometry class.
     *
     * The exact s-space recurrence is evaluated in double precision by two
     * independent code paths.  Its input-to-output condition number grows
     * with gamma, so the achievable backend agreement degrades:
     *
     *  - gamma <= 10: 1e-10 for regular points.  An exact TSC corner (three
     *    knot planes meet at the origin) is a removable but ill-conditioned
     *    one-sided limit; there the measured floor is about 6e-9, so that
     *    class gets 1e-8.
     *  - gamma = 100: 1e-6, the same floor the shipped gamma-100 device/host
     *    test uses (condition number ~1e8).  With the O(1) field floor in
     *    relativeError this also covers out-of-support tails whose tiny
     *    reference magnitude defeats a pure relative comparison.
     *  - gamma = 1000: 5e-2.  The measured absolute disagreement grows from
     *    about 8e-7 at gamma 100 to 2.5e-2 at gamma 1000 (roughly the gamma^4
     *    scaling of a gamma^2-conditioned term times its gamma^2 magnitude);
     *    the bound is 2x the observed worst case.
     */
    double agreementBound(double gamma, bool knotCorner)
    {
        if(gamma <= 10.0)
            return knotCorner ? 1.0e-8 : 1.0e-10;
        if(gamma <= 100.0)
            return 1.0e-6;
        return 5.0e-2;
    }
} // namespace

//! Host and device must read the same knot constants.  A change on one side
//! without the other fails here at compile time (aliases) and at run time.
TEST(FastNearConsistency, SharedKnotConstantsMatch)
{
    static_assert(pbfast::detail::kKnotDetect == pbfast::dconst::kKnotDetect);
    static_assert(
        pbfast::detail::kKnotEvalSnapLowGamma == pbfast::dconst::kKnotEvalSnapLowGamma);
    static_assert(
        pbfast::detail::kKnotEvalSnapHighGamma == pbfast::dconst::kKnotEvalSnapHighGamma);
    static_assert(
        pbfast::detail::kKnotHighGammaThreshold == pbfast::dconst::kKnotHighGammaThreshold);
    static_assert(pbfast::detail::kKnotGateSnap == pbfast::dconst::kKnotGateSnap);

    // Shared routing constants used by both implementations.
    static_assert(pbfast::kKnotEvalSnapLowGamma == 1.0e-9);
    static_assert(pbfast::kKnotEvalSnapHighGamma == 1.0e-7);
    static_assert(pbfast::kKnotHighGammaThreshold == 20.0);
    static_assert(pbfast::kQuadratureOrderCap == 32);
    static_assert(pbfast::kOutOfSupportExactGamma == 100.0);
    static_assert(pbfast::DeviceParams::kMaxOrder == pbfast::kQuadratureOrderCap);
}

//! Public host evaluator vs public device evaluator on the exact path.
TEST(FastNearConsistency, DeviceMatchesHostOnExactGrid)
{
    if(!pbfast::fastNearDeviceAvailable())
        GTEST_SKIP() << "compiled backend has no available device";

    for(double gamma : {1.0, 10.0, 100.0, 1000.0})
    {
        for(bool oblique : {false, true})
        {
            Vec3 const u = momentum(gamma, oblique);
            picBunch::Bunch const source = singleSource(u);
            // Reach 2 selects every in-support pair; out-of-support pairs at
            // gamma >= 100 route exact through the shared gamma predicate.
            FastParams const fast = makeFastParams(5, 8, 2, 1.0e-3);
            auto const geometry = geometries();

            for(Geometry const& g : geometry)
            {
                auto const device = pbfast::evalFastNearDevice(
                    source,
                    source,
                    false,
                    1,
                    1,
                    1,
                    g.obs.x,
                    g.obs.y,
                    g.obs.z,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    fast.params);
                std::array<double, 6> got{};
                std::copy(device.begin(), device.end(), got.begin());

                Vec3 E{}, B{};
                pbfast::shape_field(g.obs, Vec3{}, u, 1.0, 1.0, 1.0, fast.settings, E, B);
                std::array<double, 6> const ref{
                    E.x,
                    E.y,
                    E.z,
                    B.x,
                    B.y,
                    B.z};

                double const err = relativeError(got, ref);
                double const bound = agreementBound(gamma, g.knotCorner);
                EXPECT_LT(err, bound) << "gamma=" << gamma << " oblique=" << oblique << " geom="
                                      << g.name << " obs=" << g.obs.x << "," << g.obs.y << ","
                                      << g.obs.z << " inSupport=" << g.inSupport << " err=" << err
                                      << " bound=" << bound;
            }
        }
    }
}
