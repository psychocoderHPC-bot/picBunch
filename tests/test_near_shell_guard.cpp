// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#include "picBunch/detail/NearShellGuard.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

namespace
{
    picBunch::detail::NearShellPeak peak(double e, double b)
    {
        picBunch::detail::NearShellPeak value;
        value.e = e;
        value.b = b;
        return value;
    }
} // namespace

TEST(NearShellGuard, AcceptsCorrectionBelowTheGrowthLimit)
{
    EXPECT_NO_THROW(
        picBunch::detail::enforceNearShellPeak(peak(1.0, 1.0), peak(2.9, 2.9), 3.0, 4u, 5));
}

TEST(NearShellGuard, RejectsAnInflatedCorrection)
{
    try
    {
        picBunch::detail::enforceNearShellPeak(peak(1.0, 1.0), peak(4.0, 1.0), 3.0, 8u, 5);
        FAIL() << "expected the post-condition to reject a 4x correction";
    }
    catch(std::runtime_error const& error)
    {
        std::string const message = error.what();
        EXPECT_NE(message.find("near shell"), std::string::npos);
        EXPECT_NE(message.find("4.0x"), std::string::npos);
        EXPECT_NE(message.find("limit 3.0x"), std::string::npos);
        EXPECT_NE(message.find("radius=3"), std::string::npos);
        EXPECT_NE(message.find("tile=8"), std::string::npos);
        EXPECT_NE(message.find("fast_order=5"), std::string::npos);
    }
}

//! A zero uncorrected peak carries no scale and must not raise a false alarm.
TEST(NearShellGuard, ZeroBaselineIsNotAViolation)
{
    EXPECT_NO_THROW(
        picBunch::detail::enforceNearShellPeak(peak(0.0, 0.0), peak(1.0e12, 1.0e-3), 3.0, 4u, 5));
}
