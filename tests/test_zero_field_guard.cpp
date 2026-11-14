// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group: zero-field post-condition and the near_shell = "none" regression.
//
// ``near_shell = "none"`` is the plain mesh field of all sources, so a charged
// bunch has a non-zero field at every particle count.  The deposit changes at
// 4096 particles (per-particle TSC -> binned per-cell TSC), and on a wide mesh
// the binned ordering itself changes again (host histogram -> device atomic
// scatter).  Both switch points are pinned here, on both sides: a charged bunch
// must yield a non-zero field, and the binned result must equal the
// deterministic host deposit.  A charged bunch with an all-zero field must
// abort with a message naming the configuration; a neutral bunch must not.

#include "picBunch/Compute.hpp"
#include "picBunch/Types.hpp"
#include "picBunch/detail/FieldGuard.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace
{
    using picBunch::Bunch;
    using picBunch::ComputeOptions;
    using picBunch::Field;
    using picBunch::Shape;
    using picBunch::Vec3;

    constexpr double kElectronCharge = -1.602176634e-19;

    //! Uniform cubic cloud, centred in a ``nodes``^3 grid, one velocity.
    Bunch uniformBunch(std::size_t count, std::size_t nodes, double cell, double uz)
    {
        auto const per = static_cast<std::size_t>(std::ceil(std::cbrt(static_cast<double>(count))));
        Bunch bunch;
        bunch.resize(count);
        double const extent = static_cast<double>(per) * cell;
        double const shift = 0.5 * (static_cast<double>(nodes) * cell - extent);
        for(std::size_t p = 0; p < count; ++p)
        {
            std::size_t const i = p % per;
            std::size_t const j = (p / per) % per;
            std::size_t const k = p / (per * per);
            bunch.x[p] = shift + (static_cast<double>(i) + 0.5) * cell;
            bunch.y[p] = shift + (static_cast<double>(j) + 0.5) * cell;
            bunch.z[p] = shift + (static_cast<double>(k) + 0.5) * cell;
            bunch.ux[p] = 0.0;
            bunch.uy[p] = 0.0;
            bunch.uz[p] = uz;
            bunch.weight[p] = 1.0;
            bunch.charge[p] = kElectronCharge;
        }
        return bunch;
    }

    ComputeOptions optionsFor(std::size_t nodes, double cell)
    {
        ComputeOptions options;
        options.tolerance = 1e-3;
        options.cellSize = Vec3{cell, cell, cell};
        options.gridSize = picBunch::Idx3{
            static_cast<std::uint32_t>(nodes),
            static_cast<std::uint32_t>(nodes),
            static_cast<std::uint32_t>(nodes)};
        options.gridOrigin = Vec3{0.0, 0.0, 0.0};
        // Padding 2 keeps the padded mesh above the 2^18-cell device-ordering
        // threshold at 64^3, so the CUDA build runs the atomic scatter path the
        // regression was reported against.  The host build keeps its own
        // ordering path; both must deliver the same field.
        options.fftPadding = 2u;
        options.shape = Shape::TSC;
        options.nearShell.mode = picBunch::NearShellOptions::Mode::None;
        return options;
    }

    double peakE(Field const& field)
    {
        double peak = 0.0;
        for(int c = 0; c < 3; ++c)
            for(double const value : field.data[c])
                peak = std::max(peak, std::abs(value));
        return peak;
    }

    //! Stands in for a peer rank that holds a non-zero part of the field.
    struct PeerWithFieldContext : picBunch::ParallelContext
    {
        void allreduceSum(double* values, std::size_t count) const override
        {
            for(std::size_t i = 0; i < count; ++i)
                values[i] += 1.0;
        }
    };

    //! Seen-to-be non-zero field on a 4^3 grid, all components zero.
    Field zeroField()
    {
        Field field;
        field.grid.n = picBunch::Idx3{4u, 4u, 4u};
        for(auto& component : field.data)
            component.assign(4u * 4u * 4u, 0.0);
        return field;
    }
} // namespace

TEST(ZeroFieldGuard, AbortsForChargedZeroFieldWithConfiguration)
{
    Field const field = zeroField();
    double const charge = 7.0 * kElectronCharge;
    std::string const configuration = "path=test, near_shell=none, grid=4x4x4 nodes, particles=7";
    try
    {
        picBunch::detail::abortOnEmptyChargedField(field, charge, configuration);
        FAIL() << "a charged all-zero field must abort";
    }
    catch(std::runtime_error const& error)
    {
        std::string const message = error.what();
        EXPECT_NE(message.find("identically zero"), std::string::npos) << message;
        EXPECT_NE(message.find("path=test, near_shell=none"), std::string::npos)
            << "the message must name the configuration: " << message;
    }
}

TEST(ZeroFieldGuard, NeutralBunchIsAllowedToHaveZeroField)
{
    Field const field = zeroField();
    EXPECT_TRUE(picBunch::detail::fieldIsIdenticallyZero(field));
    EXPECT_NO_THROW(picBunch::detail::abortOnEmptyChargedField(field, 0.0, "neutral"));
}

TEST(ZeroFieldGuard, NonZeroFieldPassesForChargedBunch)
{
    Field field = zeroField();
    field.data[2][17] = 1e-30;
    EXPECT_FALSE(picBunch::detail::fieldIsIdenticallyZero(field));
    EXPECT_NO_THROW(picBunch::detail::abortOnEmptyChargedField(field, kElectronCharge, "tiny"));
}

TEST(ZeroFieldGuard, DistributedSlabIsAllowedToBeZeroWhenAPeerHoldsTheField)
{
    // A rank owns one z-slab; a slab far from the bunch is legitimately zero.
    // Only a globally all-zero field may abort.
    Field const field = zeroField();
    PeerWithFieldContext peer;
    EXPECT_NO_THROW(picBunch::detail::abortOnEmptyChargedDistributedField(
        field,
        kElectronCharge,
        "distributed slab",
        peer));
    picBunch::ParallelContext alone;
    EXPECT_THROW(
        picBunch::detail::abortOnEmptyChargedDistributedField(
            field,
            kElectronCharge,
            "distributed slab",
            alone),
        std::runtime_error);
}

TEST(ZeroFieldGuard, NearShellNoneIsNonZeroBelowAndAboveDepositThreshold)
{
    // 3000 < 4096 uses the per-particle deposit, 4200 >= 4096 the binned one.
    for(std::size_t const count : {3000u, 4200u})
    {
        Bunch const bunch = uniformBunch(count, 64u, 1.0, 0.0);
        Field const field = picBunch::computeField(bunch, optionsFor(64u, 1.0));
        EXPECT_GT(peakE(field), 0.0)
            << "near_shell none, " << count << " particles: the field is identically zero";
        EXPECT_FALSE(picBunch::detail::fieldIsIdenticallyZero(field));
    }
}

TEST(ZeroFieldGuard, BinnedDepositMatchesDeterministicDeposit)
{
    // The binned path (device atomic ordering on a wide padded mesh) and the
    // serial host deposit must deliver the same field.  The reported defect made
    // the binned path write nothing at all, which this pins for 5000 particles.
    Bunch const bunch = uniformBunch(5000u, 64u, 1.0, 40.0);
    ComputeOptions binned = optionsFor(64u, 1.0);
    ComputeOptions deterministic = binned;
    deterministic.deterministicDeposit = true;
    Field const fast = picBunch::computeField(bunch, binned);
    Field const reference = picBunch::computeField(bunch, deterministic);
    double const peak = peakE(reference);
    ASSERT_GT(peak, 0.0) << "the deterministic deposit must not be zero";
    double maxDifference = 0.0;
    for(int c = 0; c < 6; ++c)
    {
        ASSERT_EQ(fast.data[c].size(), reference.data[c].size());
        for(std::size_t i = 0; i < fast.data[c].size(); ++i)
            maxDifference = std::max(maxDifference, std::abs(fast.data[c][i] - reference.data[c][i]));
    }
    EXPECT_LE(maxDifference, 1e-9 * peak)
        << "binned deposit differs from the deterministic deposit by " << maxDifference
        << " (peak " << peak << ')';
    EXPECT_GT(peakE(fast), 0.0);
}

TEST(ZeroFieldGuard, NeutralBunchDoesNotReachTheGuardAsAFalsePositive)
{
    // A net-neutral bunch: every second particle carries the opposite charge at
    // the same position.  The absolute charge is non-zero, so the solve runs;
    // the net charge is exactly zero, so the guard must accept the (zero) field.
    Bunch bunch = uniformBunch(1000u, 32u, 1.0, 0.0);
    for(std::size_t p = 0; p < bunch.size(); ++p)
    {
        std::size_t const twin = p - p % 2u;
        bunch.x[p] = bunch.x[twin];
        bunch.y[p] = bunch.y[twin];
        bunch.z[p] = bunch.z[twin];
        bunch.charge[p] = (p % 2u == 0u ? 1.0 : -1.0) * kElectronCharge;
    }
    ASSERT_EQ(picBunch::detail::totalCharge(bunch), 0.0);
    Field field;
    EXPECT_NO_THROW(field = picBunch::computeField(bunch, optionsFor(32u, 1.0)));
    EXPECT_NO_THROW(picBunch::detail::abortOnEmptyChargedField(field, 0.0, "neutral"));
}

TEST(ZeroFieldGuard, ZeroChargeBunchIsRejectedBeforeTheFieldSolve)
{
    // An all-zero-charge bunch already fails in the velocity-bin model, so it
    // never reaches the field guard; the neutral bypass is a second line of
    // defence, not the only one.
    Bunch bunch = uniformBunch(64u, 16u, 1.0, 0.0);
    for(std::size_t p = 0; p < bunch.size(); ++p)
        bunch.charge[p] = 0.0;
    EXPECT_THROW(picBunch::computeField(bunch, optionsFor(16u, 1.0)), std::runtime_error);
}

TEST(ZeroFieldGuard, RelativisticMeshFieldIsNonZeroAndFastTilesStayBounded)
{
    // The deposit regression above pins the binned path against the
    // deterministic reference for a near-stationary bunch.  The same fixed
    // deposit must also deliver a finite, non-zero field for a relativistic
    // bunch (gamma ~ 550): the mesh contribution of a moving bunch is
    // contracted, never zero, and the fast near-shell correction must not
    // raise the peak by more than the guard's 3x limit.
    Bunch const bunch = uniformBunch(5000u, 64u, 1.0, 550.0);
    ComputeOptions const options = optionsFor(64u, 1.0);
    Field const reference = picBunch::computeField(bunch, options);
    double const peak = peakE(reference);
    ASSERT_GT(peak, 0.0)
        << "relativistic bunch (u_y=550): the mesh field is identically zero";
    for(int c = 0; c < 6; ++c)
        for(double const value : reference.data[c])
            ASSERT_TRUE(std::isfinite(value)) << "relativistic field is not finite";

    // near-shell fast with explicit tiles: the correction is bounded by the
    // guard; a deposit defect that writes ~1e63 V/m is caught here.
    for(std::size_t const tile : {2u, 4u})
    {
        ComputeOptions fastOptions = options;
        fastOptions.nearShell.mode = picBunch::NearShellOptions::Mode::FastGpu;
        fastOptions.nearShell.radiusCells = 3.0;
        fastOptions.nearShell.tileNodes = tile;
        fastOptions.nearShell.fastOrder = 5;
        fastOptions.nearShell.fastObliqueOrder = 4;
        Field fastField;
        ASSERT_NO_THROW(fastField = picBunch::computeField(bunch, fastOptions));
        double const fastPeak = peakE(fastField);
        EXPECT_LE(fastPeak, 3.0 * peak)
            << "tile " << tile << ": the fast correction raised the peak from "
            << peak << " to " << fastPeak << " V/m";
    }
}
