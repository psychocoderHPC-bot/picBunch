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
#include <array>
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

    //! Monoenergetic bunch at ``gamma`` with a constant polar angle
    //! ``divergence`` (rad) around +z, azimuth sampled evenly.
    Bunch divergentBunch(double gamma, double divergence, std::size_t count)
    {
        Bunch bunch;
        bunch.resize(count);
        double const u = std::sqrt(gamma * gamma - 1.0);
        for(std::size_t p = 0; p < count; ++p)
        {
            double const phi
                = 2.0 * std::acos(-1.0) * static_cast<double>(p) / static_cast<double>(count);
            bunch.x[p] = 4.0 + static_cast<double>(p % 3);
            bunch.y[p] = 4.0 + static_cast<double>((p / 3) % 3);
            bunch.z[p] = 4.0 + static_cast<double>((p / 9) % 3);
            bunch.ux[p] = u * std::sin(divergence) * std::cos(phi);
            bunch.uy[p] = u * std::sin(divergence) * std::sin(phi);
            bunch.uz[p] = u * std::cos(divergence);
            bunch.weight[p] = 1.0;
            bunch.charge[p] = kElectronCharge;
        }
        return bunch;
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

TEST(VelocityBins, CounterstreamingDoesNotReportZeroError)
{
    // Equal and opposite gamma-10 velocities: the one-box mean is zero, so the
    // old estimator skipped the angular term and returned zero error.  The one
    // box must now carry a nonzero error, and auto mode must split the bunch
    // instead of accepting one wrong bin.
    Bunch bunch;
    bunch.resize(2);
    bunch.x = {4.0, 4.0};
    bunch.y = {4.0, 4.0};
    bunch.z = {4.0, 4.0};
    double const u = std::sqrt(100.0 - 1.0);
    bunch.ux = {0.0, 0.0};
    bunch.uy = {0.0, 0.0};
    bunch.uz = {u, -u};
    bunch.weight = {1.0, 1.0};
    bunch.charge = {kElectronCharge, kElectronCharge};

    VelocityBinningOptions one;
    one.mode = VelocityBinningOptions::Mode::Explicit;
    one.explicitBins = 1;
    VelocityBinning const oneBox = picBunch::deriveVelocityBinning(bunch, 1.0e-3, one);
    EXPECT_GT(oneBox.modelSum, 0.0)
        << "a stationary mean for moving members is a full angular mismatch";
    EXPECT_GT(oneBox.modelError, 1.0e-12)
        << "one box must not report zero error (old estimator: 0)";
    // auto at a target that converges quickly: one box is rejected (above), so
    // it must pick more than one bin with a nonzero estimate.  A 1e-12 target
    // would force the O(maxBoxes) infeasible search, so it is not used here.
    VelocityBinning const autoB = picBunch::deriveVelocityBinning(bunch, 0.5);
    EXPECT_GT(autoB.count, 1u);
    // With the mean representative each box holds one member exactly, so the
    // converged estimate can legitimately be zero.
    EXPECT_LE(autoB.modelError, 0.5);
    EXPECT_TRUE(std::isfinite(autoB.modelError));
}

TEST(VelocityBins, NeutralCounterstreamingBinsViaAbsoluteCharge)
{
    // Opposite charges with opposite velocities cancel the net charge, so a
    // signed-charge weight would erase the local error (abs of the signed sum is
    // zero).  The estimator must use the absolute-charge content and still see
    // the one-box mismatch.
    Bunch bunch;
    bunch.resize(2);
    bunch.x = {4.0, 4.0};
    bunch.y = {4.0, 4.0};
    bunch.z = {4.0, 4.0};
    double const u = std::sqrt(100.0 - 1.0);
    bunch.ux = {0.0, 0.0};
    bunch.uy = {0.0, 0.0};
    bunch.uz = {u, -u};
    bunch.weight = {1.0, 1.0};
    bunch.charge = {kElectronCharge, -kElectronCharge};

    VelocityBinningOptions one;
    one.mode = VelocityBinningOptions::Mode::Explicit;
    one.explicitBins = 1;
    VelocityBinning const oneBox = picBunch::deriveVelocityBinning(bunch, 1.0e-3, one);
    EXPECT_GT(oneBox.modelSum, 0.0)
        << "a neutral bunch must not cancel the local error";
    EXPECT_GT(oneBox.modelError, 0.0)
        << "the absolute-charge weight keeps the estimate nonzero";
}

TEST(VelocityBins, MeanRepresentativeIsExactForOneParticlePerBin)
{
    // Two widely separated classes force one particle per explicit box.  The
    // representative is the absolute-charge-weighted mean of its members, so
    // each box is represented by its own particle and the estimate is zero.
    Bunch bunch = axisBunch({10.0, 100.0});
    VelocityBinningOptions policy;
    policy.mode = VelocityBinningOptions::Mode::Explicit;
    policy.explicitBins = 2;
    VelocityBinning const binning = picBunch::deriveVelocityBinning(bunch, 1.0e-3, policy);
    EXPECT_EQ(binning.count, 2u);
    EXPECT_DOUBLE_EQ(binning.modelError, 0.0);
    double const uLo = std::sqrt(100.0 - 1.0);
    double const uHi = std::sqrt(10000.0 - 1.0);
    std::array<double, 2> const reps{binning.representativeU[0][2], binning.representativeU[1][2]};
    EXPECT_TRUE((std::abs(reps[0] - uLo) < 1e-9 && std::abs(reps[1] - uHi) < 1e-9)
                || (std::abs(reps[0] - uHi) < 1e-9 && std::abs(reps[1] - uLo) < 1e-9));
}

TEST(VelocityBins, SparseBinsModelIsFinite)
{
    // Widely separated velocity classes leave most boxes empty.  The model must
    // stay finite and non-negative, counting only the occupied boxes, so a
    // sparse occupation cannot drive the estimate to zero or to a non-finite
    // value.
    Bunch bunch;
    bunch.resize(4);
    bunch.x = {4.0, 4.0, 4.0, 4.0};
    bunch.y = {4.0, 4.0, 4.0, 4.0};
    bunch.z = {4.0, 4.0, 4.0, 4.0};
    bunch.ux = {0.0, 0.0, 0.0, 0.0};
    bunch.uy = {0.0, 0.0, 0.0, 0.0};
    bunch.uz = {0.5, 5.0, 9.5, 14.0};
    bunch.weight = {1.0, 1.0, 1.0, 1.0};
    bunch.charge = {kElectronCharge, kElectronCharge, kElectronCharge, kElectronCharge};
    VelocityBinningOptions policy;
    policy.mode = VelocityBinningOptions::Mode::Explicit;
    policy.explicitBins = 16;
    VelocityBinning const binning = picBunch::deriveVelocityBinning(bunch, 1.0e-3, policy);
    EXPECT_EQ(binning.count, 16u);
    EXPECT_TRUE(std::isfinite(binning.modelSum));
    EXPECT_GE(binning.modelSum, 0.0);
    EXPECT_GE(binning.modelError, 0.0);
}

TEST(VelocityBins, EnergyTermIsNodeBoundedSingleTerm)
{
    // Two collinear energy classes.  The old model charged the gamma range plus
    // a max member deviation (~0.79); the node-bounded single term is
    // (2/gamma_max) * (gamma_max - gamma_min)/gamma_max = 0.05.
    Bunch const bunch = axisBunch({10.0, 20.0});
    VelocityBinningOptions policy;
    policy.mode = VelocityBinningOptions::Mode::Explicit;
    policy.explicitBins = 1;
    VelocityBinning const binning = picBunch::deriveVelocityBinning(bunch, 1e-3, policy);
    // The representative is moving along the same axis, so the angular term is 0.
    EXPECT_NEAR(binning.modelSum, 2.0 * (10.0 / 20.0) / 20.0, 1e-12);
}

TEST(VelocityBins, PureAngularSpreadKeepsTheUnitAngularConstant)
{
    // Constant gamma, two symmetric 5 degree tilts: the one-box representative
    // is the beam axis, every member is 5 degrees away, and the gamma range is
    // zero.  The model must stay gamma*Delta_theta (C_angular = 1), which is the
    // archived angular calibration point.
    Bunch bunch;
    bunch.resize(2);
    bunch.x = {4.0, 4.0};
    bunch.y = {4.0, 4.0};
    bunch.z = {4.0, 4.0};
    double const u = std::sqrt(100.0 - 1.0);
    double const theta = 5.0 * M_PI / 180.0;
    bunch.ux = {u * std::sin(theta), -u * std::sin(theta)};
    bunch.uy = {0.0, 0.0};
    bunch.uz = {u * std::cos(theta), u * std::cos(theta)};
    bunch.weight = {1.0, 1.0};
    bunch.charge = {kElectronCharge, kElectronCharge};

    VelocityBinningOptions policy;
    policy.mode = VelocityBinningOptions::Mode::Explicit;
    policy.explicitBins = 1;
    VelocityBinning const binning = picBunch::deriveVelocityBinning(bunch, 1e-3, policy);
    EXPECT_NEAR(binning.modelSum, 10.0 * theta, 1e-12);
}

TEST(VelocityBins, PureEnergySpreadAutoCountIsTractable)
{
    // A 5% energy spread at gamma 10 (4-sigma range) previously needed more
    // than the 256-box cap.  The node-bounded energy term must keep the auto
    // count inside the cap and still meet the requested tolerance.
    Bunch const bunch = axisBunch(gammaLinspace(8.0, 12.0, 64));
    VelocityBinning const binning = picBunch::deriveVelocityBinning(bunch, 1e-3);
    EXPECT_TRUE(binning.multiVelocity);
    EXPECT_GT(binning.count, 1u);
    EXPECT_LE(binning.count, 128u);
    EXPECT_LE(binning.modelError, 1e-3);
    // The spread is collinear along the beam axis, so all boxes sit there.
    EXPECT_EQ(binning.perAxis[0], 1u);
    EXPECT_EQ(binning.perAxis[1], 1u);
    EXPECT_EQ(binning.perAxis[2], binning.count);
}

TEST(VelocityBins, InfeasibleRequirementPrintsTheCapError)
{
    // The reported "cap" number must be the model error at maxBins, not at the
    // required count (the historical bug printed the latter under the cap label).
    Bunch const bunch = axisBunch(gammaLinspace(8.0, 12.0, 64));
    VelocityBinningOptions explicitTwo;
    explicitTwo.mode = VelocityBinningOptions::Mode::Explicit;
    explicitTwo.explicitBins = 2;
    double const capError
        = picBunch::deriveVelocityBinning(bunch, 1e-12, explicitTwo).modelError;

    VelocityBinningOptions policy;
    policy.maxBins = 2;
    try
    {
        picBunch::deriveVelocityBinning(bunch, 1e-12, policy);
        FAIL() << "expected the infeasible auto policy to throw";
    }
    catch(std::runtime_error const& e)
    {
        std::string const message = e.what();
        std::string const label = "at the 2-box cap: ";
        auto const at = message.find(label);
        ASSERT_NE(at, std::string::npos) << message;
        double const parsed = std::stod(message.substr(at + label.size()));
        EXPECT_NEAR(parsed, capError, 1e-5) << message;
        // The required-count error, printed earlier, is a different and smaller
        // number than the cap error (the model decreases with the box count).
        std::string const reqLabel = " boxes: ";
        auto const reqAt = message.find(reqLabel);
        ASSERT_NE(reqAt, std::string::npos) << message;
        double const requiredError = std::stod(message.substr(reqAt + reqLabel.size()));
        EXPECT_LT(requiredError, parsed) << message;
    }
}

TEST(VelocityBins, MonoenergeticHighGammaUsesOneBoxWithoutFlags)
{
    // The operator hit: gamma 1101 with a few mrad divergence demands millions
    // of boxes from the gamma * Delta_theta term.  It is one velocity class, so
    // the default auto policy must use one box and run without flags.
    Bunch const bunch = divergentBunch(1101.0, 3.0e-3, 1024);
    VelocityBinning const binning = picBunch::deriveVelocityBinning(bunch, 1e-3);
    ASSERT_TRUE(binning.derived);
    EXPECT_TRUE(binning.multiVelocity);
    ASSERT_EQ(binning.count, 1u);
    EXPECT_TRUE(binning.singleVelocityFallback);
    ASSERT_EQ(binning.representativeU.size(), 1u);
    EXPECT_LE(
        binning.singleVelocityRelativeSpread,
        picBunch::kSingleVelocityRelativeSpread);
    EXPECT_LE(binning.singleVelocityMaxAngle, picBunch::kSingleVelocityMaxAngle);
    // The diagnostics stay honest: the one-box estimate is above tolerance.
    EXPECT_GT(binning.modelError, 1e-3);
}

TEST(VelocityBins, ExplicitBinsOverrideSingleVelocityFallback)
{
    Bunch const bunch = divergentBunch(1101.0, 3.0e-3, 1024);
    VelocityBinningOptions policy;
    policy.mode = VelocityBinningOptions::Mode::Explicit;
    policy.explicitBins = 4;
    VelocityBinning const four = picBunch::deriveVelocityBinning(bunch, 1e-3, policy);
    EXPECT_EQ(four.count, 4u);
    EXPECT_FALSE(four.singleVelocityFallback);

    // Even an explicit single box is the user's choice, not the fallback.
    policy.explicitBins = 1;
    VelocityBinning const one = picBunch::deriveVelocityBinning(bunch, 1e-3, policy);
    EXPECT_EQ(one.count, 1u);
    EXPECT_FALSE(one.singleVelocityFallback);
}

TEST(VelocityBins, NonDefaultCapKeepsTheHistoricalFailure)
{
    // An explicit --max-velocity-bins is authoritative even for a bunch that
    // would qualify for the default-cap one-box fallback.
    Bunch const bunch = divergentBunch(1101.0, 3.0e-3, 1024);
    VelocityBinningOptions policy;
    policy.maxBins = 2;
    try
    {
        picBunch::deriveVelocityBinning(bunch, 1e-3, policy);
        FAIL() << "expected the explicit 2-box cap to throw";
    }
    catch(std::runtime_error const& e)
    {
        EXPECT_NE(std::string(e.what()).find("requires at least"), std::string::npos)
            << e.what();
    }
}

TEST(VelocityBins, BroadDistributionDoesNotTakeTheFallback)
{    // A 5% energy spread splits within the default cap and keeps the binned
    // path; it is not essentially single-velocity.
    Bunch const spread = axisBunch(gammaLinspace(8.0, 12.0, 64));
    VelocityBinning const autoBins = picBunch::deriveVelocityBinning(spread, 1e-3);
    EXPECT_GT(autoBins.count, 1u);
    EXPECT_FALSE(autoBins.singleVelocityFallback);
    EXPECT_LE(autoBins.modelError, 1e-3);

    // A 5 degree angular spread is far outside the angular criterion.  A
    // non-default cap is authoritative, so the run still fails with the
    // required count instead of silently collapsing to one box.
    Bunch const angular = divergentBunch(10.0, 5.0 * std::acos(-1.0) / 180.0, 64);
    VelocityBinningOptions policy;
    policy.maxBins = 2;
    try
    {
        picBunch::deriveVelocityBinning(angular, 1e-3, policy);
        FAIL() << "expected the broad angular bunch to throw above a 2-box cap";
    }
    catch(std::runtime_error const& e)
    {
        EXPECT_NE(std::string(e.what()).find("requires at least"), std::string::npos)
            << e.what();
    }
}
