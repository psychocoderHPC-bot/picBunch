// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Regression guard for the near-shell evaluator regimes.
//
// The reference values are the independent order-12/refine-3 `indfield`
// arbiter values recorded in the v0.8.0 review (review.md §3.2): a single
// gamma-10 source at the origin, unit cells, in-support observation points.
// They pin two facts that the documentation relies on:
//   * `pbfast` (--near-shell fast) is accurate for predominantly axial motion;
//   * `nbref` (--near-shell exact) is accurate for axial and oblique motion;
//   * the oblique in-support box is now integrated by the singularity-aware
//     Duffy fallback, so the fast path is a 1e-3 path for oblique motion too.
//
// The oblique tests also exercise the a-posteriori gate: the plain octant box
// is still bounded, but the ordered residual estimate sends the beamed pairs to
// the controlled Lorentz-contracted quadrature.

#include "picBunch/detail/FastNear.hpp"
#include "picBunch/detail/NearReference.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>

namespace
{
    using nbref::Vec3;

    constexpr double kGamma10Beta = 9.9498743710661995; // sqrt(100 - 1)
    constexpr double kGamma100Beta = 99.99499987499375; // sqrt(10000 - 1)

    double relativeError(Vec3 const& value, std::array<double, 3> const& reference)
    {
        std::array<double, 3> const v{value.x, value.y, value.z};
        double diff2 = 0.0;
        double ref2 = 0.0;
        for(int a = 0; a < 3; ++a)
        {
            diff2 += (v[a] - reference[a]) * (v[a] - reference[a]);
            ref2 += reference[a] * reference[a];
        }
        return std::sqrt(diff2 / std::max(ref2, 1.0e-300));
    }

    double relativeErrorV(Vec3 const& value, Vec3 const& reference)
    {
        return relativeError(value, {reference.x, reference.y, reference.z});
    }

    Vec3 fastField(Vec3 const& obs, Vec3 const& u, int order)
    {
        picBunch::pbfast::Settings settings;
        settings.order = order;
        settings.obliqueOrder = order;
        Vec3 E, B;
        picBunch::pbfast::shape_field(obs, Vec3{}, u, 1.0, 1.0, 1.0, settings, E, B);
        return E;
    }

    Vec3 referenceField(Vec3 const& obs, Vec3 const& u)
    {
        nbref::Settings settings;
        Vec3 E, B;
        nbref::shape_field(obs, Vec3{}, u, 1.0, 1.0, 1.0, settings, E, B);
        return E;
    }

    Vec3 oblique45(double const beta)
    {
        double const d = beta / std::sqrt(2.0);
        return {d, 0.0, d};
    }
} // namespace

TEST(NearShellAccuracy, FastIsAccurateForAxialGamma10)
{
    Vec3 const u{0.0, 0.0, kGamma10Beta};
    // Arbiter (indfield order 12, refine 3), per-unit-charge E.
    EXPECT_LT(relativeError(fastField({-0.6, 0.4, -0.3}, u, 8), {-0.9346, 0.6158, -0.0228}), 1.0e-3);
    EXPECT_LT(relativeError(fastField({1.0, 0.0, 0.0}, u, 8), {1.2582, 0.0, 0.0}), 1.0e-3);
    EXPECT_LT(relativeError(fastField({0.0, 0.5, 0.0}, u, 8), {0.0, 1.0936, 0.0}), 1.0e-3);
}

TEST(NearShellAccuracy, AdaptiveReferenceIsAccurateForAxialGamma10)
{
    Vec3 const u{0.0, 0.0, kGamma10Beta};
    // The Lorentz-contracted in-support quadrature fixes the old axial failure.
    // Off-axis points converge to the arbiter; the closest node retains a
    // measured few-percent residual, so it is bounded separately.
    EXPECT_LT(relativeError(referenceField({1.0, 0.0, 0.0}, u), {1.2582, 0.0, 0.0}), 1.0e-2);
    EXPECT_LT(relativeError(referenceField({0.0, 0.5, 0.0}, u), {0.0, 1.0936, 0.0}), 1.0e-2);
    EXPECT_LT(relativeError(referenceField({-0.6, 0.4, -0.3}, u), {-0.9346, 0.6158, -0.0228}), 5.0e-2);
}

TEST(NearShellAccuracy, AdaptiveReferenceIsAccurateForObliqueGamma10)
{
    Vec3 const u = oblique45(kGamma10Beta);
    // Arbiter (indfield order 12, refine 3).
    EXPECT_LT(relativeError(referenceField({1.0, 0.0, 0.0}, u), {0.4226, 0.0, -0.3752}), 2.0e-2);
    EXPECT_LT(relativeError(referenceField({0.0, 0.5, 0.0}, u), {0.0, 1.1086, 0.0}), 5.0e-3);
}

//! The octant box alone is still a bounded, not a 1e-3, path.  This guards the
//! gate: if someone removes the residual test the next assertions would pass
//! trivially and this one would start failing.
TEST(NearShellAccuracy, OctantObliqueStaysBoundedButIsNotTheDeliveredPath)
{
    Vec3 const u = oblique45(kGamma10Beta);
    picBunch::pbfast::Settings octant;
    octant.order = 8;
    octant.obliqueOrder = 8;
    // Disable the gate by granting an unbounded budget: the scheme then keeps
    // the octant box and the Duffy refinement without the controlled fallback.
    octant.tolerance = 1.0e9;
    octant.legacyOblique = true;
    Vec3 E, B;
    picBunch::pbfast::shape_field({-0.6, 0.4, -0.3}, Vec3{}, u, 1.0, 1.0, 1.0, octant, E, B);
    double const err = relativeError(E, {-0.2114, 0.4607, 0.1584});
    EXPECT_LT(err, 1.0);
    EXPECT_GT(err, 1.0e-2);
}

//! Duffy vertex transform: the refined box beats the plain octant on the same
//! remainder and does not depend on the controlled fallback.
TEST(NearShellAccuracy, DuffyBoxImprovesObliqueBoxResidual)
{
    Vec3 const u = oblique45(kGamma10Beta);
    Vec3 const cells{1.0, 1.0, 1.0};
    for(Vec3 const obs : {Vec3{-0.6, 0.4, -0.3}, Vec3{1.0, 0.0, 0.0}, Vec3{0.2, -0.3, 0.4}})
    {
        Vec3 Eoct{}, Boct{}, Eduf{}, Bduf{};
        picBunch::pbfast::detail::boxTerm(obs, cells, u, 8, 5, Eoct, Boct);
        picBunch::pbfast::detail::boxTermDuffy(obs, cells, u, 8, Eduf, Bduf);
        double const scale = std::max(picBunch::pbfast::detail::norm3(Eduf), 1.0e-300);
        double const boxResidual = picBunch::pbfast::detail::norm3(Eduf - Eoct) / scale;
        // The octant/refined box difference is the very quantity the gate
        // measures; it must be visible (otherwise the Duffy path is a no-op).
        EXPECT_GT(boxResidual, 1.0e-3);
        // And the refined box must be closer to the reference full field than
        // the plain octant when both use the same remainder.
        picBunch::pbfast::Settings co;
        co.order = 8;
        co.obliqueOrder = 8;
        co.tolerance = 1.0e9; // no controlled fallback
        co.legacyOblique = true; // isolate the plain octant remainder
        Vec3 Echeap, Bcheap;
        picBunch::pbfast::shape_field(obs, Vec3{}, u, 1.0, 1.0, 1.0, co, Echeap, Bcheap);
        double const W = picBunch::pbfast::detail::tsc(obs.x)
            * picBunch::pbfast::detail::tsc(obs.y) * picBunch::pbfast::detail::tsc(obs.z);
        Vec3 const Eref = Echeap + W * (Eduf - Eoct);
        Vec3 const Etruth = referenceField(obs, u);
        EXPECT_LT(relativeErrorV(Eref, Etruth), relativeErrorV(Echeap, Etruth));
    }
}

//! Acceptance case: oblique in-support, 45 degrees, gamma 10 and 100, with the
//! default tolerance the fast path is compared against the high-order
//! NearReference.  The controlled fallback must keep every point <= 1e-3.
TEST(NearShellAccuracy, FastObliqueInSupportMeets1e3Gamma10)
{
    Vec3 const u = oblique45(kGamma10Beta);
    for(Vec3 const obs : {Vec3{-0.6, 0.4, -0.3}, Vec3{1.0, 0.0, 0.0}, Vec3{0.0, 0.5, 0.0}})
    {
        Vec3 const got = fastField(obs, u, 8);
        EXPECT_LT(relativeErrorV(got, referenceField(obs, u)), 1.0e-3) << obs.x << "," << obs.y << "," << obs.z;
    }
}

TEST(NearShellAccuracy, FastObliqueInSupportMeets1e3Gamma100)
{
    Vec3 const u = oblique45(kGamma100Beta);
    for(Vec3 const obs : {Vec3{-0.6, 0.4, -0.3}, Vec3{0.2, -0.3, 0.4}, Vec3{0.0, 0.5, 0.0}})
    {
        Vec3 const got = fastField(obs, u, 8);
        EXPECT_LT(relativeErrorV(got, referenceField(obs, u)), 1.0e-3) << obs.x << "," << obs.y << "," << obs.z;
    }
}

// ---- adaptive oblique order ------------------------------------------------
#include "picBunch/detail/ObliqueOrder.hpp"
#if defined(PICBUNCH_BACKEND_CUDA)
#    include "picBunch/detail/FastNearDevice.hpp"
#endif

TEST(NearShellAccuracy, AutoObliqueOrderRule)
{
    using namespace picBunch::pbfast;
    EXPECT_EQ(autoObliqueOrder(1.0), 4);
    EXPECT_EQ(autoObliqueOrder(10.0), 4);
    EXPECT_EQ(autoObliqueOrder(11.2), 4);
    EXPECT_EQ(autoObliqueOrder(20.0), 6);
    EXPECT_EQ(autoObliqueOrder(41.0), 8);
    EXPECT_EQ(autoObliqueOrder(100.0), 8);
    EXPECT_EQ(normalizeObliqueOrder(1), 4);
    EXPECT_EQ(normalizeObliqueOrder(5), 6);
    EXPECT_EQ(normalizeObliqueOrder(7), 8);
    EXPECT_EQ(normalizeObliqueOrder(99), 12);
}

TEST(NearShellAccuracy, AutoGamma100ObliqueMeets1e2)
{
    // The auto rule selects order 8 at gamma 100.  Order 4 is the fast path
    // and stays bounded; the auto order must be at least as accurate.
    Vec3 const u = oblique45(kGamma100Beta);
    for(Vec3 const obs : {Vec3{-0.6, 0.4, -0.3}, Vec3{0.2, -0.3, 0.4}, Vec3{0.0, 0.5, 0.0}})
    {
        Vec3 const truth = referenceField(obs, u);
        Vec3 const fast = fastField(obs, u, picBunch::pbfast::autoObliqueOrder(100.0));
        EXPECT_LT(relativeErrorV(fast, truth), 1.0e-2) << obs.x << "," << obs.y << "," << obs.z;
    }
}

TEST(NearShellAccuracy, AutoGamma10KeepsOrder4)
{
    // gamma <= 10 maps to order 4, and order 4 already meets the gamma-10
    // accuracy budget at these points.
    EXPECT_EQ(picBunch::pbfast::autoObliqueOrder(10.0), 4);
    Vec3 const u = oblique45(kGamma10Beta);
    for(Vec3 const obs : {Vec3{-0.6, 0.4, -0.3}, Vec3{0.2, -0.3, 0.4}, Vec3{0.0, 0.5, 0.0}})
        EXPECT_LT(relativeErrorV(fastField(obs, u, 4), referenceField(obs, u)), 1.0e-2)
            << obs.x << "," << obs.y << "," << obs.z;
}

#if defined(PICBUNCH_BACKEND_CUDA)
namespace
{
    Vec3 deviceField(Vec3 const& obs, Vec3 const& u, int obliqueOrder)
    {
        picBunch::Bunch source;
        source.resize(1);
        source.x[0] = source.y[0] = source.z[0] = 0.0;
        source.ux[0] = u.x;
        source.uy[0] = u.y;
        source.uz[0] = u.z;
        source.weight[0] = 1.0;
        source.charge[0] = 1.0;
        picBunch::pbfast::DeviceParams params;
        params.order = 5;
        params.obliqueOrder = obliqueOrder;
        for(int i = 0; i < params.order; ++i)
        {
            auto const& r = picBunch::pbfast::detail::rule(params.order);
            params.xi[i] = r.x[static_cast<std::size_t>(i)];
            params.wi[i] = r.w[static_cast<std::size_t>(i)];
        }
        for(int slot = 0; slot < picBunch::pbfast::kObliqueOrderCount; ++slot)
        {
            int const n = picBunch::pbfast::kObliqueOrders[slot];
            auto const& r = picBunch::pbfast::detail::rule(n);
            for(int i = 0; i < n; ++i)
            {
                params.oxi[slot][i] = r.x[static_cast<std::size_t>(i)];
                params.owi[slot][i] = r.w[static_cast<std::size_t>(i)];
            }
        }
        for(int c = 0; c < 6; ++c)
            for(int a = 0; a < 3; ++a)
                params.delta[c][a] = 0.0;
        params.unitE = 1.0;
        params.unitB = 1.0;
        auto const out = picBunch::pbfast::evalFastNearDevice(
            source, source, false, 1, 1, 1, obs.x, obs.y, obs.z, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            params);
        return {out[0], out[1], out[2]};
    }
} // namespace

TEST(NearShellAccuracy, DeviceGamma100ObliqueMatchesHostAutoOrder)
{
    // The device must evaluate exactly the same algebra as the host for the
    // order the shared auto rule selects.  Absolute gamma-100 accuracy of this
    // evaluator is bounded separately: the Duffy box does not converge at
    // gamma 100 (see the implementation note), so this guards consistency and
    // the precomputed multi-order tables rather than a 1e-2 claim.
    if(!picBunch::pbfast::fastNearDeviceAvailable())
        GTEST_SKIP() << "no CUDA device";
    Vec3 const u = oblique45(kGamma100Beta);
    int const order = picBunch::pbfast::autoObliqueOrder(100.0);
    EXPECT_EQ(order, 8);
    for(Vec3 const obs : {Vec3{-0.6, 0.4, -0.3}, Vec3{0.2, -0.3, 0.4}, Vec3{0.0, 0.5, 0.0}})
    {
        picBunch::pbfast::Settings settings;
        settings.order = 5;
        settings.obliqueOrder = order;
        settings.tolerance = 1.0e9; // Duffy only, no controlled fallback
        Vec3 hostE, hostB;
        picBunch::pbfast::shape_field(obs, Vec3{}, u, 1.0, 1.0, 1.0, settings, hostE, hostB);
        Vec3 const dev = deviceField(obs, u, order);
        EXPECT_LT(relativeErrorV(dev, hostE), 1.0e-12)
            << obs.x << "," << obs.y << "," << obs.z;
    }
}

TEST(NearShellAccuracy, DeviceGamma10AutoEqualsExplicitOrder4)
{
    if(!picBunch::pbfast::fastNearDeviceAvailable())
        GTEST_SKIP() << "no CUDA device";
    Vec3 const u = oblique45(kGamma10Beta);
    for(Vec3 const obs : {Vec3{-0.6, 0.4, -0.3}, Vec3{0.0, 0.5, 0.0}})
    {
        Vec3 const a = deviceField(obs, u, 4);
        Vec3 const b = deviceField(obs, u, picBunch::pbfast::autoObliqueOrder(10.0));
        EXPECT_EQ(a.x, b.x);
        EXPECT_EQ(a.y, b.y);
        EXPECT_EQ(a.z, b.z);
    }
}
#endif
