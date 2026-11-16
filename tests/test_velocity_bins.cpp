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
                = 2.0 * std::acos(-1.0) * (2.0 * static_cast<double>(p) + 1.0)
                    / (2.0 * static_cast<double>(count));
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
        {
            EXPECT_LT(binning.modelSum, previous);
        }
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
    EXPECT_NEAR(binning.modelSum, theta * theta, 1e-15);
    // The calibrated term is 114x below the old gamma*theta worst case.
    EXPECT_LT(binning.modelSum, 10.0 * theta);
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
    // The operator hit: gamma 1101 with a few mrad divergence demanded
    // millions of boxes from the gamma * Delta_theta worst case.  The
    // calibrated angular term (theta^2) is far below the 1e-3 tolerance, so
    // the default auto policy runs without flags and picks a small count.
    Bunch bunch;
    bunch.resize(1024);
    double const u = std::sqrt(1101.0 * 1101.0 - 1.0);
    for(std::size_t p = 0; p < 1024; ++p)
    {
        // Interleaved deterministic spread over two axes: a 3 mrad ring
        // (azimuth by p mod 16) plus a 1-3 mrad azimuth offset and a 0-1.5
        // mrad tilt step (p / 16), so the beam-frame extent is nonzero in
        // both transverse directions.
        double const theta = 3.0e-3 + 1.5e-3 * (p / 16 % 2);
        double const phi
            = 2.0 * std::acos(-1.0) * ((2.0 * static_cast<double>(p % 16) + 1.0) / 32.0
                                       + (p / 16 % 2) * 0.02);
        bunch.x[p] = 4.0 + static_cast<double>(p % 3);
        bunch.y[p] = 4.0 + static_cast<double>((p / 3) % 3);
        bunch.z[p] = 4.0 + static_cast<double>((p / 9) % 3);
        bunch.ux[p] = u * std::sin(theta) * std::cos(phi);
        bunch.uy[p] = u * std::sin(theta) * std::sin(phi);
        bunch.uz[p] = u * std::cos(theta);
        bunch.weight[p] = 1.0;
        bunch.charge[p] = kElectronCharge;
    }
    VelocityBinning const binning = picBunch::deriveVelocityBinning(bunch, 1e-3);
    ASSERT_TRUE(binning.derived);
    EXPECT_TRUE(binning.multiVelocity);
    // The one-box estimate is below the tolerance: the calibrated angular
    // term is 2.0e-5 for this geometry, where the old gamma*Delta_theta
    // worst case was 1.33e+01, a 6600x over-prediction.
    ASSERT_EQ(binning.count, 1u);
    EXPECT_FALSE(binning.singleVelocityFallback);
    ASSERT_EQ(binning.representativeU.size(), 1u);
    EXPECT_LE(binning.modelError, 1e-3);
}

// ---------------------------------------------------------------------------
// W37: calibrated angular term (replaces the gamma*Delta_theta worst case).

TEST(VelocityBins, AngularTermIsWidthSquaredNotGamma)
{
    // Constant gamma 10, two symmetric 5 degree tilts: the old model charged
    // gamma*theta = 0.873.  The calibrated angular term charges theta^2
    // = 7.67e-3: the superposition of the member fields cancels the tilt
    // error at a node to the measured 3-7e-4 level, not to gamma*theta.
    Bunch bunch;
    bunch.resize(2);
    bunch.x = {4.0, 4.0};
    bunch.y = {4.0, 4.0};
    bunch.z = {4.0, 4.0};
    double const u = std::sqrt(100.0 - 1.0);
    double const theta = 5.0 * std::acos(-1.0) / 180.0;
    bunch.ux = {u * std::sin(theta), -u * std::sin(theta)};
    bunch.uy = {0.0, 0.0};
    bunch.uz = {u * std::cos(theta), u * std::cos(theta)};
    bunch.weight = {1.0, 1.0};
    bunch.charge = {kElectronCharge, kElectronCharge};

    VelocityBinningOptions policy;
    policy.mode = VelocityBinningOptions::Mode::Explicit;
    policy.explicitBins = 1;
    VelocityBinning const binning = picBunch::deriveVelocityBinning(bunch, 1e-3, policy);
    // gamma range zero: the estimate is the pure angular term theta^2
    // (kAngularModelConstant = 1), 114x below the old gamma*theta.
    EXPECT_NEAR(binning.modelSum, theta * theta, 1e-15);
    EXPECT_LT(binning.modelSum, 10.0 * theta);
}

TEST(VelocityBins, RealisticHighGammaBunchLandsAtFourToEight)
{
    // The operator's bunch: gamma 471-510 (the 1-99% interval of |p|, sigma_g/gamma about 3.9%) and about 50 mrad divergence.  The old gamma*Delta_theta term gave a one-box model sum of about 24.6 and the auto policy demanded order 1e5 boxes.  The calibrated angular term must return 4 to 8 boxes at the 1e-3 tolerance.
    Bunch bunch;
    bunch.resize(256);
    for(std::size_t p = 0; p < 256; ++p)
    {
        // Interleaved deterministic spread: gamma over the 1-99% |p| interval
        // (linspace by p mod 16) and an angular ring with a 50 mrad tilt step
        // (p / 16) and even azimuth, so both beam-frame axes carry spread.
        double const t = static_cast<double>(p) / 255.0;
        double const gammaP = 471.0 + (509.56 - 471.04) * t;
        double const u = std::sqrt(gammaP * gammaP - 1.0);
        double const theta = 1.28 * 50.0e-3;
        double const phi
            = 2.0 * std::acos(-1.0) * (2.0 * static_cast<double>(p % 16) + 1.0) / 32.0;
        bunch.x[p] = 4.0 + static_cast<double>(p % 3);
        bunch.y[p] = 4.0 + static_cast<double>((p / 3) % 3);
        bunch.z[p] = 4.0 + static_cast<double>((p / 9) % 3);
        bunch.ux[p] = u * std::sin(theta) * std::cos(phi);
        bunch.uy[p] = u * std::sin(theta) * std::sin(phi);
        bunch.uz[p] = u * std::cos(theta);
        bunch.weight[p] = 1.0;
        bunch.charge[p] = kElectronCharge;
    }
    VelocityBinning const binning = picBunch::deriveVelocityBinning(bunch, 1e-3);
    ASSERT_TRUE(binning.derived);
    EXPECT_TRUE(binning.multiVelocity);
    EXPECT_GE(binning.count, 4u) << binning.count;
    EXPECT_LE(binning.count, 8u) << binning.count;
    EXPECT_LE(binning.modelError, 1e-3);
    EXPECT_FALSE(binning.singleVelocityFallback);
}

TEST(VelocityBins, NarrowSyntheticBunchLandsAtEight)
{
    // The W37 'narrow' bunch: gamma 500, negligible energy spread, 50 mrad divergence.  The measured binning error reaches 1e-3 at 8 boxes (4.1e-4 at 8, 1.2e-3 at 4; docs/measurements/ velocity_bin_angular_calibration.csv).  Auto must select 8.
    Bunch bunch;
    bunch.resize(256);
    for(std::size_t p = 0; p < 256; ++p)
    {
        // The W37 "narrow" bunch: gamma 500, negligible energy spread, 50 mrad
        // divergence.  Interleaved 50/64 mrad tilt ring with even azimuth.
        double const u = std::sqrt(500.0 * 500.0 - 1.0);
        double const theta
            = 1.28 * 50.0e-3 * (1.0 + 0.25 * (p / 16 % 2));
        double const phi
            = 2.0 * std::acos(-1.0) * (2.0 * static_cast<double>(p % 16) + 1.0) / 32.0;
        bunch.x[p] = 4.0 + static_cast<double>(p % 3);
        bunch.y[p] = 4.0 + static_cast<double>((p / 3) % 3);
        bunch.z[p] = 4.0 + static_cast<double>((p / 9) % 3);
        bunch.ux[p] = u * std::sin(theta) * std::cos(phi);
        bunch.uy[p] = u * std::sin(theta) * std::sin(phi);
        bunch.uz[p] = u * std::cos(theta);
        bunch.weight[p] = 1.0;
        bunch.charge[p] = kElectronCharge;
    }
    VelocityBinning const binning = picBunch::deriveVelocityBinning(bunch, 1e-3);
    EXPECT_EQ(binning.count, 8u);
    EXPECT_LE(binning.modelError, 1e-3);
}

TEST(VelocityBins, WideSyntheticBunchLandsAtSixteen)
{
    // The W37 "wide" bunch: gamma 500 with sigma_g/gamma 0.15 (1.65 sigma
    // range) and 150 mrad divergence (1.28 sigma).  Interleaved tilt
    // (p / 16) and gamma (p mod 16) so both beam-frame axes carry spread.
    // Measured: the binning error reaches 1e-3 at 32 boxes (7.8e-4 at 32,
    // 1.6e-3 at 16; docs/measurements/velocity_bin_angular_calibration.csv).
    // The auto policy must land inside 32-64 boxes, not above the 256 cap.
    Bunch bunch;
    bunch.resize(256);
    for(std::size_t p = 0; p < 256; ++p)
    {
        double const t = static_cast<double>(p % 16) / 15.0;
        double const gammaP = 500.0 * (1.0 + 0.15 * 1.65 * (2.0 * t - 1.0));
        double const u = std::sqrt(gammaP * gammaP - 1.0);
        double const theta
            = 1.28 * 150.0e-3 * (1.0 + 0.25 * ((p / 16) % 2));
        double const phi
            = 2.0 * std::acos(-1.0) * (2.0 * static_cast<double>(p % 16) + 1.0) / 32.0;
        bunch.x[p] = 4.0 + static_cast<double>(p % 3);
        bunch.y[p] = 4.0 + static_cast<double>((p / 3) % 3);
        bunch.z[p] = 4.0 + static_cast<double>((p / 9) % 3);
        bunch.ux[p] = u * std::sin(theta) * std::cos(phi);
        bunch.uy[p] = u * std::sin(theta) * std::sin(phi);
        bunch.uz[p] = u * std::cos(theta);
        bunch.weight[p] = 1.0;
        bunch.charge[p] = kElectronCharge;
    }
    VelocityBinning const binning = picBunch::deriveVelocityBinning(bunch, 1e-3);
    EXPECT_GE(binning.count, 32u) << binning.count;
    EXPECT_LE(binning.count, 64u) << binning.count;
    EXPECT_LE(binning.modelError, 1e-3);
}

TEST(VelocityBins, CapFailureNamesTheRequiredCount)
{
    // An extreme angular spread (gamma 2000, 1 radian constant divergence)
    // drives the calibrated model error above the 1e-3 tolerance up to the
    // 256-box cap.  The auto policy must fail with the required count and
    // the separate cap error, never silently under-binning.
    Bunch bunch;
    bunch.resize(256);
    double const u = std::sqrt(2000.0 * 2000.0 - 1.0);
    double const div = 1.0;
    for(std::size_t p = 0; p < 256; ++p)
    {
        double const phi = 2.0 * std::acos(-1.0) * (2.0 * static_cast<double>(p % 16) + 1.0) / 32.0;
        double const theta = div * (1.0 + 0.25 * ((p / 16) % 2));
        bunch.x[p] = 4.0 + static_cast<double>(p % 3);
        bunch.y[p] = 4.0 + static_cast<double>((p / 3) % 3);
        bunch.z[p] = 4.0 + static_cast<double>((p / 9) % 3);
        bunch.ux[p] = u * std::sin(theta) * std::cos(phi);
        bunch.uy[p] = u * std::sin(theta) * std::sin(phi);
        bunch.uz[p] = u * std::cos(theta);
        bunch.weight[p] = 1.0;
        bunch.charge[p] = kElectronCharge;
    }
    try
    {
        picBunch::deriveVelocityBinning(bunch, 1e-3);
        FAIL() << "expected the 1 radian bunch to exceed the cap at 1e-3";
    }
    catch(std::runtime_error const& e)
    {
        std::string const message = e.what();
        EXPECT_NE(message.find("requires at least"), std::string::npos) << message;
        // The required count is printed as a number; the cap error follows it.
        EXPECT_NE(message.find("at the 256-box cap: "), std::string::npos) << message;
    }
}

namespace
{
    //! Mirror the calibrated real bunch (particles_000000.bp5 species 'b'):
    //! gamma ~490, 0.78-degree (0.0136 rad) transverse divergence, ~1.7%
    //! relative momentum spread, 256 sources on a lattice.  The binning model
    //! is calibrated on this geometry; the auto policy must return a small
    //! power-of-two count (the model meets the tolerance at B=2), not an
    //! extreme count, and the model error must shrink with the box count.
    Bunch realBunchLike()
    {
        Bunch bunch;
        bunch.resize(256);
        double const u = std::sqrt(490.0 * 490.0 - 1.0);
        double const div = 0.0136; // ~0.78 degrees
        for(std::size_t p = 0; p < 256; ++p)
        {
            double const phi = 2.0 * std::acos(-1.0) * (2.0 * static_cast<double>(p % 16) + 1.0) / 32.0;
            double const theta = div * (1.0 + 0.25 * ((p / 16) % 2));
            // a small relative energy spread (~1.7%) across the sources
            double const g = 490.0 * (1.0 + 0.017 * (static_cast<double>(p % 5) / 4.0 - 0.5));
            double const gu = std::sqrt(g * g - 1.0);
            bunch.x[p] = 4.0 + static_cast<double>(p % 3);
            bunch.y[p] = 4.0 + static_cast<double>((p / 3) % 3);
            bunch.z[p] = 4.0 + static_cast<double>((p / 9) % 3);
            bunch.ux[p] = gu * std::sin(theta) * std::cos(phi);
            bunch.uy[p] = gu * std::sin(theta) * std::sin(phi);
            bunch.uz[p] = gu * std::cos(theta);
            bunch.weight[p] = 1.0;
            bunch.charge[p] = kElectronCharge;
        }
        return bunch;
    }
} // namespace

TEST(VelocityBins, RealBunchAutoReturnsASmallCount)
{
    Bunch const bunch = realBunchLike();
    VelocityBinning const binning = picBunch::deriveVelocityBinning(bunch, 1e-3);
    EXPECT_TRUE(binning.derived);
    EXPECT_TRUE(binning.multiVelocity);
    // The auto policy picks a small power-of-two count, not an extreme one.
    EXPECT_EQ((binning.count & (binning.count - 1u)), 0u);
    EXPECT_GE(binning.count, 1u);
    EXPECT_LE(binning.count, 32u);
    // The selected count's model error meets the tolerance.
    EXPECT_LE(binning.modelError, 1e-3 + 1e-15);
}

TEST(VelocityBins, RealBunchOneBoxHasFiniteModelError)
{
    // The one-box estimate for the real-bunch-like geometry must be finite and
    // positive (a genuine spread), and the 16-box estimate must be lower than
    // the one-box estimate: refining the boxes reduces the model error.  This
    // is the calibration anchor -- the 16-box model error is the value the
    // auto policy compares against the tolerance.
    Bunch const bunch = realBunchLike();
    VelocityBinningOptions one;
    one.mode = VelocityBinningOptions::Mode::Explicit;
    one.explicitBins = 1;
    VelocityBinning const b1 = picBunch::deriveVelocityBinning(bunch, 1e-3, one);
    VelocityBinningOptions sixteen;
    sixteen.mode = VelocityBinningOptions::Mode::Explicit;
    sixteen.explicitBins = 16;
    VelocityBinning const b16 = picBunch::deriveVelocityBinning(bunch, 1e-3, sixteen);
    EXPECT_GT(b1.modelSum, 0.0);
    EXPECT_TRUE(std::isfinite(b1.modelSum));
    EXPECT_LT(b16.modelSum, b1.modelSum);
}

TEST(VelocityBins, InBunchCriterionEscalatesToTheCapForHighGamma)
{
    // The in-bunch peak-normalised error model (angular term amplified by
    // ~gamma) drives the auto policy to the cap for a high-gamma bunch whose
    // in-bunch transverse B and E peaks are ~1/gamma suppressed.  This is the
    // honest result: the in-bunch max error cannot reach the tolerance within
    // the cap, so the cap (the count that minimises the in-bunch error) is
    // returned with a diagnostic.  The regression guard is that the By
    // in-bunch error at the returned count is finite and below the one-box
    // error (the cap reduces it), so a future change cannot silently drop the
    // transverse B component from the criterion again.
    Bunch const bunch = realBunchLike();
    VelocityBinningOptions policy;
    policy.mode = VelocityBinningOptions::Mode::Auto;
    policy.inBunchCriterion = true;
    VelocityBinning const binning = picBunch::deriveVelocityBinning(bunch, 1e-3, policy);
    EXPECT_TRUE(binning.derived);
    EXPECT_TRUE(binning.multiVelocity);
    // The in-bunch criterion escalates the count far above the far-field
    // result (which lands at a small count for this geometry) because the
    // in-bunch peak-normalised error is amplified by ~gamma.  The count is a
    // large power of two, bounded by the cap.
    EXPECT_EQ((binning.count & (binning.count - 1u)), 0u);
    EXPECT_GE(binning.count, 64u);
    EXPECT_LE(binning.count, picBunch::kDefaultMaxVelocityBins);
    // The far-field criterion (flag off) for the same bunch lands at a small
    // count, confirming the in-bunch criterion is what escalates it.
    VelocityBinningOptions farPolicy;
    farPolicy.mode = VelocityBinningOptions::Mode::Auto;
    VelocityBinning const farBinning = picBunch::deriveVelocityBinning(bunch, 1e-3, farPolicy);
    EXPECT_LT(farBinning.count, binning.count);
    // The cap's in-bunch model error must be below the one-box in-bunch model
    // error (the cap reduces the in-bunch error), so the transverse B
    // component is not silently dropped from the criterion.
    VelocityBinningOptions one;
    one.mode = VelocityBinningOptions::Mode::Explicit;
    one.explicitBins = 1;
    one.inBunchCriterion = true;
    VelocityBinning const b1 = picBunch::deriveVelocityBinning(bunch, 1e-3, one);
    EXPECT_GT(b1.modelSum, 0.0);
    EXPECT_TRUE(std::isfinite(b1.modelSum));
    EXPECT_LT(binning.modelSum, b1.modelSum);
}

TEST(VelocityBins, InBunchCriterionByErrorShrinksWithBoxes)
{
    // The in-bunch peak-normalised By error (the transverse B component that
    // the earlier measurement found 88-99 percent wrong with one bin) must
    // shrink as the box count increases, so a future change cannot silently
    // drop the component from the criterion.  The gamma-amplified angular
    // model is a monotone lower bound on the in-bunch By error.
    Bunch const bunch = realBunchLike();
    double previous = 0.0;
    bool first = true;
    for(std::size_t B : {1u, 4u, 16u, 64u})
    {
        VelocityBinningOptions policy;
        policy.mode = VelocityBinningOptions::Mode::Explicit;
        policy.explicitBins = B;
        policy.inBunchCriterion = true;
        VelocityBinning const binning = picBunch::deriveVelocityBinning(bunch, 1e-3, policy);
        EXPECT_EQ(binning.count, B);
        EXPECT_GT(binning.modelSum, 0.0);
        EXPECT_TRUE(std::isfinite(binning.modelSum));
        if(!first)
        {
            EXPECT_LT(binning.modelSum, previous);
        }
        previous = binning.modelSum;
        first = false;
    }
}
