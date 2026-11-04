// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group 14: tolerance-driven velocity binning.
//
// The mesh kernel is exact for one shared u; the mixed-bunch field is the sum
// over velocity boxes.  These tests pin the binning geometry, the policy and
// the one-box identity (a one-box spread bunch must equal a single-velocity
// bunch at the charge-weighted mean u), and that the pipeline no longer
// rejects a momentum spread.

#include "picBunch/Compute.hpp"
#include "picBunch/VelocityBinning.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace
{
    using picBunch::Bunch;
    using picBunch::VelocityBinning;
    using picBunch::VelocityBinningOptions;

    constexpr double kElectronCharge = -1.602176634e-19;

    //! One particle per gamma class, all on the beam axis +z, positions on a
    //! small lattice so the deposit does not stack.
    Bunch axisBunch(std::vector<double> const& gammas)
    {
        Bunch bunch;
        bunch.resize(gammas.size());
        for(std::size_t p = 0; p < gammas.size(); ++p)
        {
            bunch.x[p] = 4.0 + static_cast<double>(p % 3);
            bunch.y[p] = 4.0 + static_cast<double>((p / 3) % 3);
            bunch.z[p] = 4.0 + static_cast<double>((p / 9) % 3);
            bunch.ux[p] = 0.0;
            bunch.uy[p] = 0.0;
            bunch.uz[p] = std::sqrt(gammas[p] * gammas[p] - 1.0);
            bunch.weight[p] = 1.0;
            bunch.charge[p] = kElectronCharge;
        }
        return bunch;
    }

    std::vector<double> gammaLinspace(double lo, double hi, std::size_t count)
    {
        std::vector<double> out(count);
        for(std::size_t i = 0; i < count; ++i)
            out[i] = lo + (hi - lo) * static_cast<double>(i) / static_cast<double>(count - 1);
        return out;
    }

    picBunch::ComputeOptions smallOptions()
    {
        picBunch::ComputeOptions options;
        options.tolerance = 1e-3;
        options.cellSize = {1.0, 1.0, 1.0};
        options.gridSize = {16, 16, 16};
        options.gridOrigin = {0.0, 0.0, 0.0};
        options.fftPadding = 2u;
        return options;
    }
} // namespace

TEST(VelocityBins, SingleVelocityStaysOneBox)
{
    Bunch bunch = axisBunch({10.0});
    VelocityBinning const binning
        = picBunch::deriveVelocityBinning(bunch, 1e-3);
    EXPECT_TRUE(binning.derived);
    EXPECT_FALSE(binning.multiVelocity);
    EXPECT_EQ(binning.count, 1u);
    EXPECT_EQ(binning.representativeU[0][2], std::sqrt(100.0 - 1.0));
    EXPECT_DOUBLE_EQ(binning.modelError, 0.0);
}

TEST(VelocityBins, TwoClassBunchKeepsTheHistoricalCheck)
{
    // Distinct classes further apart than 1e-12 relative count as spread.
    Bunch const bunch = axisBunch({10.0, 10.0 * (1.0 + 1e-9)});
    VelocityBinning const binning = picBunch::deriveVelocityBinning(bunch, 1e-3);
    EXPECT_TRUE(binning.multiVelocity);
}

TEST(VelocityBins, OneBoxUsesChargeWeightedMean)
{
    Bunch const bunch = axisBunch({10.0, 20.0});
    VelocityBinningOptions policy;
    policy.mode = VelocityBinningOptions::Mode::Explicit;
    policy.explicitBins = 1;
    VelocityBinning const binning = picBunch::deriveVelocityBinning(bunch, 1e-3, policy);
    EXPECT_TRUE(binning.multiVelocity);
    EXPECT_EQ(binning.count, 1u);
    double const uLo = std::sqrt(100.0 - 1.0);
    double const uHi = std::sqrt(400.0 - 1.0);
    // Equal weights: the mean is the midrange.
    EXPECT_NEAR(binning.representativeU[0][2], 0.5 * (uLo + uHi), 1e-12);
    EXPECT_GT(binning.modelSum, 0.0);
    EXPECT_NEAR(binning.modelError, binning.modelConstant * binning.modelSum, 0.0);
}

TEST(VelocityBins, ModelShrinksWithBoxes)
{
    Bunch const bunch = axisBunch(gammaLinspace(10.0, 20.0, 32));
    double previous = 0.0;
    for(std::size_t B : {1u, 2u, 4u, 8u, 16u})
    {
        VelocityBinningOptions policy;
        policy.mode = VelocityBinningOptions::Mode::Explicit;
        policy.explicitBins = B;
        VelocityBinning const binning = picBunch::deriveVelocityBinning(bunch, 1e-3, policy);
        EXPECT_EQ(binning.count, B);
        EXPECT_GT(binning.modelSum, 0.0);
        if(B > 1u)
            EXPECT_LT(binning.modelSum, previous);
        previous = binning.modelSum;
    }
}

TEST(VelocityBins, EnergySpreadBinsTheBeamAxis)
{
    Bunch const bunch = axisBunch(gammaLinspace(10.0, 20.0, 8));
    VelocityBinningOptions policy;
    policy.mode = VelocityBinningOptions::Mode::Explicit;
    policy.explicitBins = 4;
    VelocityBinning const binning = picBunch::deriveVelocityBinning(bunch, 1e-3, policy);
    // Only the beam axis (e3) carries a spread, so all four boxes sit there.
    EXPECT_EQ(binning.perAxis[2], 4u);
    EXPECT_EQ(binning.perAxis[0] * binning.perAxis[1] * binning.perAxis[2], 4u);
    // The beam axis is +z for this bunch.
    EXPECT_NEAR(binning.basis[2][2], 1.0, 1e-12);
}

TEST(VelocityBins, AutoMeetsTheTolerance)
{
    Bunch const bunch = axisBunch(gammaLinspace(10.0, 20.0, 32));
    VelocityBinning const binning = picBunch::deriveVelocityBinning(bunch, 0.5);
    EXPECT_TRUE(binning.multiVelocity);
    EXPECT_LE(binning.modelError, 0.5);
    // Power of two.
    EXPECT_EQ((binning.count & (binning.count - 1u)), 0u);
}

TEST(VelocityBins, InfeasibleRequirementNamesTheNeededCount)
{
    Bunch const bunch = axisBunch(gammaLinspace(10.0, 20.0, 32));
    VelocityBinningOptions policy;
    policy.maxBins = 2;
    try
    {
        picBunch::deriveVelocityBinning(bunch, 1e-12, policy);
        FAIL() << "expected the infeasible auto policy to throw";
    }
    catch(std::runtime_error const& e)
    {
        EXPECT_NE(std::string(e.what()).find("requires at least"), std::string::npos)
            << e.what();
    }
}

TEST(VelocityBins, MixedBunchNoLongerThrows)
{
    Bunch const bunch = axisBunch(gammaLinspace(10.0, 20.0, 8));
    auto options = smallOptions();
    picBunch::VelocityBinning binning;
    picBunch::Field const field
        = picBunch::computeField(bunch, options, nullptr, &binning);
    ASSERT_TRUE(binning.derived);
    EXPECT_TRUE(binning.multiVelocity);
    for(auto const& component : field.data)
        for(double v : component)
            EXPECT_TRUE(std::isfinite(v));
}

TEST(VelocityBins, OneBoxMatchesSingleVelocityAtMeanU)
{
    // A one-box spread bunch is exactly the single-velocity bunch at the
    // charge-weighted mean u: same deposit, same kernel.
    Bunch const mixed = axisBunch({10.0, 20.0});
    auto options = smallOptions();
    options.velocityBinning.mode = VelocityBinningOptions::Mode::Explicit;
    options.velocityBinning.explicitBins = 1;
    picBunch::VelocityBinning binning;
    picBunch::Field const mixedField = picBunch::computeField(mixed, options, nullptr, &binning);
    ASSERT_TRUE(binning.derived);

    Bunch shared = mixed;
    for(auto& u : shared.ux)
        u = binning.representativeU[0][0];
    for(auto& u : shared.uy)
        u = binning.representativeU[0][1];
    for(auto& u : shared.uz)
        u = binning.representativeU[0][2];
    picBunch::Field const sharedField = picBunch::computeField(shared, options);

    double maxDiff = 0.0;
    for(int c = 0; c < 6; ++c)
        for(std::size_t i = 0; i < mixedField.data[c].size(); ++i)
            maxDiff = std::max(maxDiff, std::abs(mixedField.data[c][i] - sharedField.data[c][i]));
    EXPECT_DOUBLE_EQ(maxDiff, 0.0);
}

TEST(VelocityBins, BinnedFieldEqualsSumOfBoxClassFields)
{
    // Linearity: the field of the mixed bunch at two boxes equals the sum of
    // the single-velocity fields of the box contents at the box
    // representatives.
    Bunch const both = axisBunch(gammaLinspace(10.0, 20.0, 8));
    auto options = smallOptions();
    options.velocityBinning.mode = VelocityBinningOptions::Mode::Explicit;
    options.velocityBinning.explicitBins = 2;
    picBunch::VelocityBinning binning;
    picBunch::Field const bothField = picBunch::computeField(both, options, nullptr, &binning);
    ASSERT_TRUE(binning.derived);
    ASSERT_EQ(binning.count, 2u);

    picBunch::Field sum{};
    sum.grid = bothField.grid;
    for(auto& component : sum.data)
        component.assign(bothField.grid.size(), 0.0);

    for(std::size_t b = 0; b < binning.count; ++b)
    {
        Bunch box;
        for(std::size_t p = 0; p < both.size(); ++p)
            if(binning.boxOf[p] == b)
                box.resize(box.size() + 1);
        std::size_t k = 0;
        for(std::size_t p = 0; p < both.size(); ++p)
        {
            if(binning.boxOf[p] != b)
                continue;
            box.x[k] = both.x[p];
            box.y[k] = both.y[p];
            box.z[k] = both.z[p];
            box.ux[k] = binning.representativeU[b][0];
            box.uy[k] = binning.representativeU[b][1];
            box.uz[k] = binning.representativeU[b][2];
            box.weight[k] = both.weight[p];
            box.charge[k] = both.charge[p];
            ++k;
        }
        ASSERT_GT(box.size(), 0u);
        picBunch::Field const boxField = picBunch::computeField(box, options);
        for(int c = 0; c < 6; ++c)
            for(std::size_t i = 0; i < sum.data[c].size(); ++i)
                sum.data[c][i] += boxField.data[c][i];
    }

    double maxDiff = 0.0;
    for(int c = 0; c < 6; ++c)
        for(std::size_t i = 0; i < bothField.data[c].size(); ++i)
            maxDiff = std::max(maxDiff, std::abs(bothField.data[c][i] - sum.data[c][i]));
    double scale = 1.0;
    for(auto const& component : bothField.data)
        for(double v : component)
            scale = std::max(scale, std::abs(v));
    // FFT roundoff only: same transforms, same order of operations per node.
    EXPECT_LT(maxDiff / scale, 1e-12);
}
