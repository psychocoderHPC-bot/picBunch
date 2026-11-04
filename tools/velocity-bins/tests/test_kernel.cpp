// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group 2: point-field kernel.  Expected symmetry/antisymmetry, the exact
// stationary Coulomb branch, and the TSC-convolved stationary limit.

#include "picBunch/Kernel.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>

namespace
{
    std::array<double, 6> field(double Rx, double Ry, double Rz, double ux, double uy, double uz)
    {
        std::array<double, 6> out{};
        picBunch::pointFieldNormalized(Rx, Ry, Rz, ux, uy, uz, out.data());
        return out;
    }
} // namespace

TEST(Kernel, TscShapeSymmetryAndSupport)
{
    for(double x : {0.05, 0.25, 0.49, 0.51, 1.0, 1.49})
        EXPECT_DOUBLE_EQ(picBunch::tscShape(x), picBunch::tscShape(-x));
    EXPECT_NEAR(picBunch::tscShape(0.0), 0.75, 0.0);
    EXPECT_NEAR(picBunch::tscShape(0.5), 0.5, 0.0);
    EXPECT_NEAR(picBunch::tscShape(1.0), 0.125, 0.0);
    EXPECT_DOUBLE_EQ(picBunch::tscShape(1.5), 0.0);
}

TEST(Kernel, PointFieldSymmetryAndAntisymmetry)
{
    // E(-R) = -E(R); B(-R) = -B(R); B(R,-u) = -B(R,u).
    double const R[3] = {1.3, -0.7, 2.1};
    double const u[3] = {0.4, -1.2, 3.0};
    auto const plus = field(R[0], R[1], R[2], u[0], u[1], u[2]);
    auto const minusR = field(-R[0], -R[1], -R[2], u[0], u[1], u[2]);
    auto const minusU = field(R[0], R[1], R[2], -u[0], -u[1], -u[2]);
    for(int c = 0; c < 6; ++c)
    {
        EXPECT_NEAR(minusR[c], -plus[c], 1.0e-15 * (1.0 + std::abs(plus[c]))) << "R sign, component " << c;
        double const expected = (c < 3) ? plus[c] : -plus[c];
        EXPECT_NEAR(minusU[c], expected, 1.0e-15 * (1.0 + std::abs(plus[c]))) << "u sign, component " << c;
    }
}

TEST(Kernel, StationaryLimitIsCoulomb)
{
    double const R[3] = {1.7, -2.3, 0.9};
    auto const out = field(R[0], R[1], R[2], 0.0, 0.0, 0.0);
    double const r2 = R[0] * R[0] + R[1] * R[1] + R[2] * R[2];
    double const invR3 = 1.0 / (r2 * std::sqrt(r2));
    for(int c = 0; c < 3; ++c)
        EXPECT_NEAR(out[c], R[c] * invR3, 1.0e-15) << "component " << c;
    for(int c = 3; c < 6; ++c)
        EXPECT_DOUBLE_EQ(out[c], 0.0);

    // A tiny momentum must converge to the stationary branch.
    auto const tiny = field(R[0], R[1], R[2], 1.0e-8, 0.0, 0.0);
    for(int c = 0; c < 6; ++c)
        EXPECT_NEAR(tiny[c], out[c], 1.0e-7 * (1.0 + std::abs(out[c]))) << "component " << c;
}

TEST(Kernel, StationaryTscConvolutionApproachesCoulomb)
{
    // Independent direct TSC sum: E(node) = sum_s W(s) K_coulomb(node - s)
    // with the particle at the cell centre.  At six cells the TSC smoothing
    // is a small correction, so the sum must approach the point Coulomb field.
    double const u[3] = {0.0, 0.0, 0.0};
    double const R[3] = {0.0, 0.0, 6.0};
    std::array<double, 3> sum{0.0, 0.0, 0.0};
    for(int a = -1; a <= 2; ++a)
        for(int b = -1; b <= 2; ++b)
            for(int c = -1; c <= 2; ++c)
            {
                double const w = picBunch::tscShape(a - 0.5) * picBunch::tscShape(b - 0.5)
                    * picBunch::tscShape(c - 0.5);
                auto const k = field(
                    R[0] - a,
                    R[1] - b,
                    R[2] - c,
                    u[0],
                    u[1],
                    u[2]);
                for(int d = 0; d < 3; ++d)
                    sum[d] += w * k[d];
            }
    auto const point = field(R[0], R[1], R[2], u[0], u[1], u[2]);
    double const scale = 1.0 + std::abs(point[2]);
    for(int d = 0; d < 3; ++d)
        EXPECT_LT(std::abs(sum[d] - point[d]) / scale, 5.0e-2) << "component " << d;
}
