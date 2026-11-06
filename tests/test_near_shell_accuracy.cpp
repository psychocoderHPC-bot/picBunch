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
//   * `pbfast`'s oblique octant is bounded but not a 1e-3 path.
//
// The last two assertions are deliberately written as inequalities in the
// failing direction.  They do not celebrate an error; they prevent a future
// change from silently labelling the moving in-support result as exact.

#include "picBunch/detail/FastNear.hpp"
#include "picBunch/detail/NearReference.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>

namespace
{
    using nbref::Vec3;

    constexpr double kGamma10Beta = 9.9498743710661995; // sqrt(100 - 1)

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
    double const d = kGamma10Beta / std::sqrt(2.0);
    Vec3 const u{d, 0.0, d};
    // Arbiter (indfield order 12, refine 3).
    EXPECT_LT(relativeError(referenceField({1.0, 0.0, 0.0}, u), {0.4226, 0.0, -0.3752}), 2.0e-2);
    EXPECT_LT(relativeError(referenceField({0.0, 0.5, 0.0}, u), {0.0, 1.1086, 0.0}), 5.0e-3);
}

TEST(NearShellAccuracy, FastObliqueIsBoundedButNotA1e3Path)
{
    double const d = kGamma10Beta / std::sqrt(2.0);
    Vec3 const u{d, 0.0, d};
    // The review measured up to ~80% in this hard 45 degree case; README
    // documents 8.15e-2 for the measured bunch.  Guard both ends: the result
    // stays finite/bounded, but it is not a 1e-3 accuracy claim.
    double const err = relativeError(fastField({-0.6, 0.4, -0.3}, u, 8), {-0.2114, 0.4607, 0.1584});
    EXPECT_LT(err, 1.0);
    EXPECT_GT(err, 1.0e-2);
}
