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
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
     *  - gamma = 100: 2e-6 (condition number ~1e8).  With the O(1) field
     *    floor in relativeError this also covers out-of-support tails whose
     *    tiny reference magnitude defeats a pure relative comparison.  The
     *    measured worst case on gfx1100/ROCm 7.2 is 1.16e-6 after the host
     *    (&x)[i] member-subobject indexing was made defined; the previous
     *    1e-6 was below the achievable device/host agreement there.
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
            return 2.0e-6;
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

//! Host bit-stability pin for the fast evaluator.
//!
//! The host shape_field is the fallback and the reference the device functor
//! mirrors.  These are the exact bit patterns emitted by the pre-alpaka-math
//! base (commit 048ca547, `-O2`, GCC 13.3, host libm) at the near-shell
//! oracle/golden points; they cover the exact s-space path, the axial
//! gamma-stretched box, the oblique d-space ray path, the legacy remainder
//! quadrature and the static branch.  A change of algorithm, threshold,
//! quadrature order or math backend that moves any host bit fails here.
TEST(FastNearConsistency, HostResultsAreBitStable)
{
    Vec3 const uZero{0.0, 0.0, 0.0};
    Vec3 const uAxialG10{0.0, 0.0, 9.9498743710661995};
    double const b10 = 9.9498743710661995 / std::sqrt(2.0);
    double const b100 = 99.99499987499375 / std::sqrt(2.0);
    Vec3 const uOb45G10{b10, 0.0, b10};
    Vec3 const uOb45G100{b100, 0.0, b100};
    double const t5 = 5.0 * 3.141592653589793238462643383279502884 / 180.0;
    Vec3 const uTilt5G100{99.99499987499375 * std::sin(t5), 0.0, 99.99499987499375 * std::cos(t5)};

    struct Pin
    {
        char const* name;
        Vec3 obs;
        Vec3 u;
        int order;
        int reach;
        double tolerance;
        std::array<double, 3> E;
        std::array<double, 3> B;
    };
    std::array<Pin, 21> const pins{{
        {"exact-axial-g10-a", {-0.6, 0.4, -0.3}, uAxialG10, 8, 2, 1e-3, {-0x1.de82d2e9dde58p-1, 0x1.3b47ac459f45ep-1, -0x1.75f4dba398725p-6}, {-0x1.39b319c0f4adbp-1, -0x1.dc1cca734ec28p-1, 0x0p+0}},
        {"exact-axial-g10-b", {1.0, 0.0, 0.0}, uAxialG10, 8, 2, 1e-3, {0x1.42182a7ceab75p+0, -0x1.dfp-50, -0x1.5e147ae147ae1p-56}, {0x1.dc9956e8ab83ep-50, 0x1.407ad9687eb14p+0, -0x0p+0}},
        {"exact-axial-g10-c", {0.0, 0.5, 0.0}, uAxialG10, 8, 2, 1e-3, {0x1.1d7p-44, 0x1.17f405c232bb3p+0, 0x1.a70a3d70a3d71p-56}, {-0x1.168cc832be483p+0, 0x1.1c01b8d545568p-44, 0x0p+0}},
        {"exact-ob45-g10-a", {-0.6, 0.4, -0.3}, uOb45G10, 8, 2, 1e-3, {-0x1.ae9dc7c8ffaf4p-3, 0x1.d7d6345ee076bp-2, 0x1.419f36392a1bcp-3}, {-0x1.4bf7711ea7d3ap-2, -0x1.089f8e08d17e8p-2, 0x1.4bf7711ea7d3ap-2}},
        {"exact-ob45-g10-b", {1.0, 0.0, 0.0}, uOb45G10, 8, 2, 1e-3, {0x1.b057dd843f823p-2, 0x1.414p-47, -0x1.7fced57dac661p-2}, {-0x1.c409f0084807cp-48, 0x1.1f1b4ba33c6a8p-1, 0x1.c409f0084807cp-48}},
        {"exact-ob45-g10-c", {0.0, 0.5, 0.0}, uOb45G10, 8, 2, 1e-3, {0x1.1cc1c28f5c29p-45, 0x1.1bc332ac2dfdfp+0, -0x1.171e3d70a3d7p-45}, {-0x1.8f4a0a467ede5p-1, 0x1.8cb8a265b2f6bp-45, 0x1.8f4a0a467ede5p-1}},
        {"exact-ob45-g100-a", {-0.6, 0.4, -0.3}, uOb45G100, 8, 2, 1e-3, {-0x1.7829f5d563208p-3, 0x1.d5f79cb0e8845p-2, 0x1.75e6f68b91f78p-3}, {-0x1.4c4ce72a4b666p-2, -0x1.092cd6fd91ff4p-2, 0x1.4c4ce72a4b666p-2}},
        {"exact-ob45-g100-b", {1.0, 0.0, 0.0}, uOb45G100, 8, 2, 1e-3, {0x1.93c05049c10a7p-2, -0x1.9e8p-47, -0x1.92b7944e355d9p-2}, {0x1.2514c356f716p-47, 0x1.1d1d87fd3e47ap-1, -0x1.2514c356f716p-47}},
        {"exact-ob45-g100-c", {0.0, 0.5, 0.0}, uOb45G100, 8, 2, 1e-3, {-0x1.3c912c339c0ecp-40, 0x1.2291e29803155p+0, 0x1.3c9093cc63f14p-40}, {-0x1.9ae844811a34dp-1, -0x1.bfab5fba142ep-40, 0x1.9ae844811a34dp-1}},
        {"exact-tilt5-g100-a", {1.0, 0.0, 0.0}, uTilt5G100, 8, 2, 1e-3, {0x1.4816984b068fcp+0, -0x1.d009cp-40, -0x1.caab025fcdd47p-4}, {0x1.ce3fc96413bf5p-40, 0x1.495260e550422p+0, -0x1.4388518f5b27fp-43}},
        {"exact-tilt5-g100-b", {0.0, 0.5, 0.0}, uTilt5G100, 8, 2, 1e-3, {0x1.380921fb801fcp-37, 0x1.1dcd8535dfee1p+0, -0x1.bbc770d39d8a2p-41}, {-0x1.1cb37581dd05fp+0, 0x1.393fff507254dp-37, 0x1.8e879f6cdc55dp-4}},
        {"exact-tilt5-g100-c", {-0.6, 0.4, -0.3}, uTilt5G100, 8, 2, 1e-3, {-0x1.c6dfd44de1335p-1, 0x1.3919edce4ccc9p-1, 0x1.3bf5d4480ec0fp-4}, {-0x1.37e4ed2e194cfp-1, -0x1.c89012e3e1c99p-1, 0x1.b4986a1df330bp-5}},
        {"exact-tilt5-g100-d", {0.0, 0.0, 0.5}, uTilt5G100, 8, 2, 1e-3, {-0x1.58537d2d7e3b6p-4, -0x1.4793f58p-31, 0x1.0c553bef7bc69p-7}, {0x1.4650ab5c342bbp-31, -0x1.59ec025f31cd6p-4, -0x1.c8c82f5894097p-35}},
        {"remainder-g10-a", {2.0, 0.5, 0.0}, uOb45G10, 5, 0, 1e-3, {0x1.b7d202c84cdbfp-6, 0x1.3be9cea14abdbp-7, -0x1.e9178e5a4983p-7}, {-0x1.bc8788ac7e57bp-8, 0x1.e17e920dc1ddap-6, 0x1.bc8788ac7e57bp-8}},
        {"remainder-g10-b", {0.0, 2.5, 0.0}, uOb45G10, 5, 0, 1e-3, {-0x1.dad1a25d3f08p-54, 0x1.1554554b47c99p-1, -0x1.6ceff2fee81p-55}, {-0x1.863cb22dc7d73p-2, -0x1.9b5f8cc9a79e5p-55, 0x1.863cb22dc7d73p-2}},
        {"remainder-axial-g10-a", {2.0, 0.5, 0.0}, uAxialG10, 5, 0, 1e-3, {0x1.49fcc99a5a0d8p-1, 0x1.46b785a3758c6p-3, 0x1.2437828cb9p-58}, {-0x1.4514460ea4451p-3, 0x1.485557a10a787p-1, 0x0p+0}},
        {"remainder-g1-a", {2.0, 0.5, 0.0}, uZero, 5, 0, 1e-3, {0x1.d02015e29d065p-3, 0x1.c79e4a94c6178p-5, -0x1.1ee1f4a54p-59}, {0x0p+0, 0x0p+0, 0x0p+0}},
        {"golden-g10-a", {0.05, 0.0, 0.0}, uOb45G10, 5, 2, 1e-6, {0x1.1038a7b0c6291p-4, -0x1.c4p-46, -0x1.f30d77693e5fep-5}, {0x1.3e029bb6ef233p-46, 0x1.6f14fcc8bf0b3p-4, -0x1.3e029bb6ef233p-46}},
        {"golden-g10-b", {0.5, 0.5, 0.5}, uOb45G10, 5, 2, 1e-6, {0x1.a271c0b26dee6p-6, 0x1.e71c74ed4101bp-2, 0x1.a271c5e6f8de6p-6}, {-0x1.56b68c52a8047p-2, -0x1.d4c371c8p-29, 0x1.56b68c52a8047p-2}},
        {"golden-g100-a", {0.05, 0.0, 0.0}, uOb45G100, 5, 2, 1e-6, {0x1.0a81a7bd76877p-4, -0x1.2d2p-43, -0x1.0a048f400fdb1p-4}, {0x1.a9d571aafa9d3p-44, 0x1.78885ef76a45dp-4, -0x1.a9d571aafa9d3p-44}},
        {"golden-g100-b", {0.5, 0.5, 0.5}, uOb45G100, 5, 2, 1e-6, {0x1.12c2c8bbdd344p-11, 0x1.e182c3202dc38p-2, 0x1.12c2c62f37344p-11}, {-0x1.547667bb9998dp-2, 0x1.cd781458p-35, 0x1.547667bb9998dp-2}},
    }};

    auto const bits = [](double v) { return std::bit_cast<std::uint64_t>(v); };
    for(Pin const& p : pins)
    {
        pbfast::Settings s;
        s.order = p.order;
        s.obliqueOrder = 8;
        s.exactCellReach = p.reach;
        s.tolerance = p.tolerance;
        Vec3 E{}, B{};
        pbfast::shape_field(p.obs, Vec3{}, p.u, 1.0, 1.0, 1.0, s, E, B);
        std::array<double, 3> const gotE{E.x, E.y, E.z};
        std::array<double, 3> const gotB{B.x, B.y, B.z};
        for(int a = 0; a < 3; ++a)
        {
            EXPECT_EQ(bits(gotE[a]), bits(p.E[a])) << p.name << " E[" << a << "]";
            EXPECT_EQ(bits(gotB[a]), bits(p.B[a])) << p.name << " B[" << a << "]";
        }
    }
}
