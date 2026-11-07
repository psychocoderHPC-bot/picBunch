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
