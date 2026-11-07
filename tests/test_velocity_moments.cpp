// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group 15: velocity-moment spectral decomposition (candidate C1).
//
// Two things are pinned here.  First, the truncated multivariate Taylor of the
// normalized point kernel (``picBunch/VelocityMoments.hpp``) is checked against
// finite differences: the value at the centre matches ``pointFieldNormalized``,
// the reconstructed field converges at the order the Taylor should, and the
// first-derivative coefficients match central differences.  Second, the whole
// moment pipeline is checked against the 256-velocity-box binned reference on a
// gamma-10 bunch with a 5% energy spread (worst relative L2).

#include "picBunch/Compute.hpp"
#include "picBunch/Kernel.hpp"
#include "picBunch/VelocityMoments.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

namespace
{
    using picBunch::Bunch;
    using picBunch::ComputeOptions;
    using picBunch::VelocityMomentsOptions;
    using picBunch::VelocityMomentsReport;

    constexpr double kElectronCharge = -1.602176634e-19;

    //! Gamma-mean bunch with a Gaussian relative energy spread, moving along +z,
    //! positions on a small lattice so the deposit does not stack.
    Bunch spreadBunch(std::size_t n, double gammaMean, double relSpread)
    {
        Bunch bunch;
        bunch.resize(n);
        std::mt19937 gen(0x12345678u);
        std::normal_distribution<double> gauss(0.0, 1.0);
        std::size_t const side = static_cast<std::size_t>(std::cbrt(static_cast<double>(n))) + 1u;
        for(std::size_t p = 0; p < n; ++p)
        {
            double const gamma = gammaMean * (1.0 + relSpread * gauss(gen));
            bunch.x[p] = 4.0 + 0.25 * static_cast<double>(p % side);
            bunch.y[p] = 4.0 + 0.25 * static_cast<double>((p / side) % side);
            bunch.z[p] = 4.0 + 0.25 * static_cast<double>((p / (side * side)) % side);
            bunch.ux[p] = 0.0;
            bunch.uy[p] = 0.0;
            bunch.uz[p] = std::sqrt(std::max(1.0e-12, gamma * gamma - 1.0));
            bunch.weight[p] = 1.0;
            bunch.charge[p] = kElectronCharge;
        }
        return bunch;
    }

    ComputeOptions momentOptions()
    {
        ComputeOptions options;
        options.tolerance = 1e-3;
        options.cellSize = {1.0, 1.0, 1.0};
        options.gridSize = {48, 48, 48};
        options.gridOrigin = {0.0, 0.0, 0.0};
        options.fftPadding = 2u;
        return options;
    }

    //! Frobenius relative L2 of ``(a - b)`` against ``b`` over all six components.
    double relativeL2(picBunch::Field const& a, picBunch::Field const& b)
    {
        double num = 0.0;
        double den = 0.0;
        for(int c = 0; c < 6; ++c)
        {
            for(std::size_t i = 0; i < a.data[c].size(); ++i)
            {
                double const d = a.data[c][i] - b.data[c][i];
                num += d * d;
                den += b.data[c][i] * b.data[c][i];
            }
        }
        return std::sqrt(num) / std::sqrt(den);
    }

    //! Fill ``out`` with the order-3 Taylor coefficients of the kernel at (R, u0).
    template<std::size_t Order = 3>
    void fillTaylor(double const R[3], double const u0[3], double* out)
    {
        picBunch::detail::pointFieldTaylor<Order>(R[0], R[1], R[2], u0[0], u0[1], u0[2], out);
    }
} // namespace

// ---------------------------------------------------------------- Taylor kernel

TEST(VelocityMoments, TaylorValueMatchesPointFieldAtCentre)
{
    double const R[3] = {0.37, -0.21, 0.62};
    double const u0[3] = {0.0, 0.0, std::sqrt(100.0 - 1.0)}; // gamma 10 along +z
    std::array<double, 6u * picBunch::detail::TaylorPoly<3>::T> coeffs{};
    fillTaylor<3>(R, u0, coeffs.data());
    double exact[6];
    picBunch::pointFieldNormalized(R[0], R[1], R[2], u0[0], u0[1], u0[2], exact);
    for(int c = 0; c < 6; ++c)
    {
        double const value
            = picBunch::detail::pointFieldTaylorValue<3>(coeffs.data(), c, 0.0, 0.0, 0.0);
        EXPECT_NEAR(value, exact[c], 1e-12) << "component " << c << " value at centre";
    }
}

TEST(VelocityMoments, TaylorReconstructionConverges)
{
    double const R[3] = {0.37, -0.21, 0.62};
    double const u0[3] = {0.0, 0.0, std::sqrt(100.0 - 1.0)};
    std::array<double, 6u * picBunch::detail::TaylorPoly<3>::T> coeffs{};
    fillTaylor<3>(R, u0, coeffs.data());
    // Displace along the transverse (y) axis.  An order-3 Taylor in u is exact
    // through the cubic, so the reconstruction error is O(du^4): halving du must
    // shrink the error by about 2^4 = 16 (we require at least 2^3 = 8).
    double previous = 0.0;
    double ratio = 0.0;
    for(double du : {0.1, 0.05, 0.025})
    {
        double exact[6];
        picBunch::pointFieldNormalized(R[0], R[1], R[2], u0[0], u0[1] + du, u0[2], exact);
        double maxErr = 0.0;
        for(int c = 0; c < 6; ++c)
        {
            double const recon
                = picBunch::detail::pointFieldTaylorValue<3>(coeffs.data(), c, 0.0, du, 0.0);
            double const scale = std::max(1.0, std::abs(exact[c]));
            maxErr = std::max(maxErr, std::abs(recon - exact[c]) / scale);
        }
        if(previous > 0.0)
            ratio = maxErr / previous;
        previous = maxErr;
    }
    EXPECT_LT(previous, 1e-3) << "residual at du = 0.025";
    EXPECT_LT(ratio, 8.0) << "convergence ratio " << ratio;
    EXPECT_LT(ratio, 0.9) << "error should shrink as du shrinks";
}

TEST(VelocityMoments, FirstDerivativesMatchFiniteDifferences)
{
    double const R[3] = {0.37, -0.21, 0.62};
    double const u0[3] = {0.0, 0.0, std::sqrt(100.0 - 1.0)};
    std::array<double, 6u * picBunch::detail::TaylorPoly<3>::T> coeffs{};
    fillTaylor<3>(R, u0, coeffs.data());
    double const h = 1.0e-4;
    std::size_t const axis[3][3] = {{1u, 0u, 0u}, {0u, 1u, 0u}, {0u, 0u, 1u}};
    for(int d = 0; d < 3; ++d)
    {
        double up[3] = {u0[0], u0[1], u0[2]};
        double um[3] = {u0[0], u0[1], u0[2]};
        up[d] += h;
        um[d] -= h;
        double fup[6];
        double fum[6];
        picBunch::pointFieldNormalized(R[0], R[1], R[2], up[0], up[1], up[2], fup);
        picBunch::pointFieldNormalized(R[0], R[1], R[2], um[0], um[1], um[2], fum);
        for(int c = 0; c < 6; ++c)
        {
            double const fd = (fup[c] - fum[c]) / (2.0 * h);
            double const coeff = coeffs[static_cast<std::size_t>(c)
                    * picBunch::detail::TaylorPoly<3>::T
                    + picBunch::detail::TaylorPoly<3>::compactIndex(
                      axis[d][0], axis[d][1], axis[d][2])];
            double const scale = std::max(1.0, std::abs(fd));
            EXPECT_NEAR(coeff, fd, 1.0e-3 * scale) << "component " << c << " axis " << d;
        }
    }
}

// ---------------------------------------------------------------- the pipeline

TEST(VelocityMoments, MomentFieldMatchesTwoHundredFiftySixBoxReference)
{
    Bunch const bunch = spreadBunch(216, 10.0, 0.05);

    // Reference: the 256-velocity-box binned field.
    auto refOptions = momentOptions();
    refOptions.velocityBinning.mode = picBunch::VelocityBinningOptions::Mode::Explicit;
    refOptions.velocityBinning.explicitBins = 256u;
    picBunch::VelocityBinning refBinning;
    picBunch::Field const ref = picBunch::computeField(bunch, refOptions, nullptr, &refBinning);
    ASSERT_EQ(refBinning.count, 256u);

    auto makeMomentOptions = []() {
        ComputeOptions options = momentOptions();
        options.velocityMoments.mode = VelocityMomentsOptions::Mode::Explicit;
        return options;
    };

    // Order 3.
    auto o3 = makeMomentOptions();
    o3.velocityMoments.order = 3u;
    VelocityMomentsReport rep3{};
    picBunch::Field const field3
        = picBunch::computeField(bunch, o3, nullptr, nullptr, &rep3);
    ASSERT_TRUE(rep3.used) << "order-3 moment path should be used";
    EXPECT_EQ(rep3.order, 3);

    // Order 2.
    auto o2 = makeMomentOptions();
    o2.velocityMoments.order = 2u;
    VelocityMomentsReport rep2{};
    picBunch::Field const field2
        = picBunch::computeField(bunch, o2, nullptr, nullptr, &rep2);
    ASSERT_TRUE(rep2.used) << "order-2 moment path should be used";

    double const rel3 = relativeL2(field3, ref);
    double const rel2 = relativeL2(field2, ref);
    std::printf(
        "[velocity-moments] gamma-10 5%% spread: order-3 relL2 = %.3e, "
        "order-2 relL2 = %.3e (vs 256 boxes)\n",
        rel3,
        rel2);
    EXPECT_LT(rel3, 5.0e-2) << "order-3 worst relative L2 vs 256 boxes";
    EXPECT_LT(rel3, rel2) << "order 3 should beat order 2";
}

TEST(VelocityMoments, AutoSelectsOrderThreeAndMatches)
{
    Bunch const bunch = spreadBunch(216, 10.0, 0.05);
    auto refOptions = momentOptions();
    refOptions.velocityBinning.mode = picBunch::VelocityBinningOptions::Mode::Explicit;
    refOptions.velocityBinning.explicitBins = 256u;
    picBunch::Field const ref = picBunch::computeField(bunch, refOptions);

    auto options = momentOptions();
    options.velocityMoments.mode = VelocityMomentsOptions::Mode::Auto;
    VelocityMomentsReport rep{};
    picBunch::Field const field = picBunch::computeField(bunch, options, nullptr, nullptr, &rep);
    ASSERT_TRUE(rep.used) << "auto should use the moment path; probe = " << rep.probe;
    EXPECT_EQ(rep.order, 3);
    std::printf("[velocity-moments] auto probe = %.3e (order %d)\n", rep.probe, rep.order);
    EXPECT_LT(relativeL2(field, ref), 5.0e-2);
}

TEST(VelocityMoments, OffKeepsTheBinnedPath)
{
    Bunch const bunch = spreadBunch(216, 10.0, 0.05);
    auto options = momentOptions();
    // Default mode is Off: the moment path must not engage.
    VelocityMomentsReport rep{};
    picBunch::Field const field = picBunch::computeField(bunch, options, nullptr, nullptr, &rep);
    EXPECT_FALSE(rep.used);
    for(auto const& component : field.data)
        for(double v : component)
            EXPECT_TRUE(std::isfinite(v));
}
