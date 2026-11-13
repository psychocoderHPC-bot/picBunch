// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group: near-shell "none" zero-field regression.
//
// ``near_shell = "none"`` is the plain mesh field of all sources: it must be
// non-zero for a charged bunch at every particle count, including counts above
// the 4096 threshold where the binned TSC deposit path engages.  A zero field
// there was reported as a regression in which the source projection was
// skipped.  These tests pin the non-zero result on both sides of the threshold
// and at the reported counts, both single-velocity (one box) and with a
// velocity spread, and pin the hard post-condition that a charged bunch with an
// all-zero field aborts while a genuinely neutral bunch does not.

#include "picBunch/Compute.hpp"
#include "picBunch/VelocityBinning.hpp"
#include "picBunch/detail/ZeroFieldGuard.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace
{
    using picBunch::Bunch;
    using picBunch::ComputeOptions;
    using picBunch::Field;
    using picBunch::VelocityBinningOptions;

    constexpr double kElectronCharge = -1.602176634e-19;

    //! ``count`` particles on a uniform cubic lattice (exactly ``count``, not
    //! rounded to a cube) spanning ``ceil(cbrt(count))`` cells and centred in a
    //! ``grid``-node box at ``cell`` spacing.  The lattice keeps the deposit
    //! well conditioned; the count is what decides which deposit branch runs.
    Bunch uniformBunch(std::size_t count, std::size_t grid, double cell, double uz)
    {
        std::size_t const per = static_cast<std::size_t>(
            std::ceil(std::cbrt(static_cast<double>(count))));
        double const extent = static_cast<double>(per) * cell;
        double const offset = 0.5 * (static_cast<double>(grid) * cell - extent);
        Bunch bunch;
        bunch.resize(count);
        for(std::size_t p = 0; p < count; ++p)
        {
            std::size_t const i = p % per;
            std::size_t const j = (p / per) % per;
            std::size_t const k = p / (per * per);
            bunch.x[p] = 0.5 * cell + static_cast<double>(i) * cell + offset;
            bunch.y[p] = 0.5 * cell + static_cast<double>(j) * cell + offset;
            bunch.z[p] = 0.5 * cell + static_cast<double>(k) * cell + offset;
            bunch.ux[p] = 0.0;
            bunch.uy[p] = 0.0;
            bunch.uz[p] = uz;
            bunch.weight[p] = 1.0;
            bunch.charge[p] = kElectronCharge;
        }
        return bunch;
    }

    //! Same geometry with a small per-particle velocity spread along the beam
    //! axis, so the auto binning policy selects more than one velocity box.
    Bunch spreadBunch(std::size_t count, std::size_t grid, double cell)
    {
        Bunch bunch = uniformBunch(count, grid, cell, 0.0);
        for(std::size_t p = 0; p < count; ++p)
            bunch.uz[p] = 1.0 + 1e-3 * static_cast<double>(p % 8);
        return bunch;
    }

    ComputeOptions optionsFor(std::size_t grid)
    {
        ComputeOptions options;
        options.tolerance = 1e-3;
        options.cellSize = {1.0, 1.0, 1.0};
        options.gridSize = {grid, grid, grid};
        options.gridOrigin = {0.0, 0.0, 0.0};
        options.fftPadding = 2u;
        // near_shell = "none" is the default mode; state it explicitly so the
        // regression cannot be masked by a future default change.
        options.nearShell.mode = picBunch::NearShellOptions::Mode::None;
        return options;
    }

    double maxAbsE(Field const& field)
    {
        double peak = 0.0;
        for(int c = 0; c < 3; ++c)
            for(double const value : field.data[c])
                peak = std::max(peak, std::abs(value));
        return peak;
    }

    Field allZeroField()
    {
        Field field;
        field.grid.n = {8, 8, 8};
        for(auto& component : field.data)
            component.assign(8u * 8u * 8u, 0.0);
        return field;
    }
} // namespace

TEST(NearShellNone, ChargedBunchFieldIsNonZeroBelowThreshold)
{
    // N = 4095 < 4096: the per-particle deposit path.
    Bunch const bunch = uniformBunch(4095u, 32u, 1.0, 0.0);
    Field const field = picBunch::computeField(bunch, optionsFor(32));
    EXPECT_GT(maxAbsE(field), 0.0)
        << "near_shell none, N=4095: the electric field of a charged bunch is identically zero";
    EXPECT_FALSE(picBunch::detail::fieldIsIdenticallyZero(field));
}

TEST(NearShellNone, ChargedBunchFieldIsNonZeroAboveThreshold)
{
    // N = 4097 >= 4096: the binned TSC deposit path.
    Bunch const bunch = uniformBunch(4097u, 32u, 1.0, 0.0);
    Field const field = picBunch::computeField(bunch, optionsFor(32));
    EXPECT_GT(maxAbsE(field), 0.0)
        << "near_shell none, N=4097 (binned deposit): the electric field of a charged bunch "
           "is identically zero";
    EXPECT_FALSE(picBunch::detail::fieldIsIdenticallyZero(field));
}

TEST(NearShellNone, ChargedBunchFieldIsNonZeroAtReportedCount)
{
    // N = 5000, the smallest reported count on the binned path.
    Bunch const bunch = uniformBunch(5000u, 48u, 1.0, 0.0);
    Field const field = picBunch::computeField(bunch, optionsFor(48));
    EXPECT_GT(maxAbsE(field), 0.0)
        << "near_shell none, N=5000 (binned deposit): the electric field of a charged bunch "
           "is identically zero";
    EXPECT_FALSE(picBunch::detail::fieldIsIdenticallyZero(field));
}

TEST(NearShellNone, ChargedBunchFieldIsNonZeroAtLargeN)
{
    // N = 10000 well above the threshold, on a small grid.
    Bunch const bunch = uniformBunch(10000u, 48u, 1.0, 0.0);
    Field const field = picBunch::computeField(bunch, optionsFor(48));
    EXPECT_GT(maxAbsE(field), 0.0)
        << "near_shell none, N=10000 (binned deposit): the electric field of a charged bunch "
           "is identically zero";
    EXPECT_FALSE(picBunch::detail::fieldIsIdenticallyZero(field));
}

TEST(NearShellNone, MultiVelocityBunchFieldIsNonZeroAboveThreshold)
{
    // A velocity-spread bunch above the deposit threshold: the auto policy
    // splits it into several boxes and every box runs the same deposit.  The
    // sum over boxes must still be non-zero.
    Bunch const bunch = spreadBunch(5000u, 48u, 1.0);
    picBunch::VelocityBinning binning;
    Field const field
        = picBunch::computeField(bunch, optionsFor(48), nullptr, &binning);
    ASSERT_GT(binning.count, 0u);
    EXPECT_GT(maxAbsE(field), 0.0)
        << "near_shell none, N=5000 multi-velocity (" << binning.count
        << " boxes): the field of a charged bunch is identically zero";
    EXPECT_FALSE(picBunch::detail::fieldIsIdenticallyZero(field));
}

TEST(ZeroFieldGuard, RejectsAnAllZeroFieldForAChargedBunch)
{
    // The hard post-condition: a non-zero total charge and an identically zero
    // field must abort with a message that names the configuration.
    Field const zeroed = allZeroField();
    ASSERT_TRUE(picBunch::detail::fieldIsIdenticallyZero(zeroed));
    double const totalCharge = -1.602176634e-19 * 5000.0;
    try
    {
        picBunch::detail::enforceNonZeroField(
            zeroed,
            totalCharge,
            "test, near_shell none, N=5000, velocity_boxes=1, grid=8x8x8");
        FAIL() << "the zero-field guard accepted a charged bunch with an all-zero field";
    }
    catch(std::runtime_error const& error)
    {
        std::string const message = error.what();
        EXPECT_NE(message.find("identically zero"), std::string::npos);
        EXPECT_NE(message.find("non-zero total charge"), std::string::npos);
        EXPECT_NE(message.find("near_shell none"), std::string::npos);
        EXPECT_NE(message.find("N=5000"), std::string::npos);
        EXPECT_NE(message.find("grid=8x8x8"), std::string::npos);
    }
}

TEST(ZeroFieldGuard, AcceptsAZeroFieldForANeutralBunch)
{
    // A genuinely neutral bunch has no field; the guard must not reject it.
    Field const zeroed = allZeroField();
    EXPECT_NO_THROW(picBunch::detail::enforceNonZeroField(zeroed, 0.0, "test, neutral"));
}

TEST(ZeroFieldGuard, AcceptsANonZeroFieldForAChargedBunch)
{
    Field field = allZeroField();
    field.data[0][0] = 1.0;
    EXPECT_NO_THROW(picBunch::detail::enforceNonZeroField(field, -1.0, "test, non-zero"));
}
