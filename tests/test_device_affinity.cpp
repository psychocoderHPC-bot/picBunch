// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera

#include "picBunch/DeviceAffinity.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    std::vector<picBunch::DeviceDescriptor> syntheticDevices(int count)
    {
        std::vector<picBunch::DeviceDescriptor> devices;
        for(int index = 0; index < count; ++index)
            devices.push_back(picBunch::DeviceDescriptor{
                index,
                "synthetic-" + std::to_string(index),
                std::size_t{1u} << 30});
        return devices;
    }
} // namespace

TEST(DeviceAffinity, BalancedDistributionWithMoreRanksThanDevices)
{
    std::vector<picBunch::DeviceDescriptor> const devices = syntheticDevices(3);
    std::vector<int> const counts = picBunch::ranksPerDevice(7, 3);
    ASSERT_EQ(counts.size(), 3u);
    EXPECT_EQ(counts[0], 3);
    EXPECT_EQ(counts[1], 2);
    EXPECT_EQ(counts[2], 2);
    for(int count : counts)
        EXPECT_TRUE(count == 7 / 3 || count == (7 + 3 - 1) / 3);

    std::vector<int> first;
    std::vector<int> second;
    for(int rank = 0; rank < 7; ++rank)
    {
        picBunch::RankDeviceMapping const mapping
            = picBunch::assignRankDevice(rank, rank, 7, devices);
        first.push_back(mapping.deviceIndex);
        EXPECT_EQ(mapping.ranksPerDevice, counts[static_cast<std::size_t>(mapping.deviceIndex)]);
        second.push_back(picBunch::assignRankDevice(rank, rank, 7, devices).deviceIndex);
    }
    EXPECT_EQ(first, (std::vector<int>{0, 1, 2, 0, 1, 2, 0}));
    // The rule is a pure function of local rank and device count.
    EXPECT_EQ(first, second);
}

TEST(DeviceAffinity, RefusesWhenRequirementExceedsCapacityShare)
{
    picBunch::RankDeviceMapping mapping
        = picBunch::assignRankDevice(3, 3, 8, syntheticDevices(2));
    mapping.deviceCapacityBytes = std::size_t{1u} << 30;
    mapping.ranksPerDevice = 4;

    picBunch::DeviceWorkload workload;
    workload.paddedNodes = std::size_t{64u} << 20;
    workload.spectrumNodes = std::size_t{32u} << 20;
    workload.outputNodes = std::size_t{16u} << 20;
    workload.particleCount = std::size_t{1u} << 20;
    workload.velocityBins = 1;
    workload.rankCount = 4;
    workload.distributed = false;

    picBunch::DeviceFit const fit = picBunch::evaluateDeviceFit(mapping, workload);
    EXPECT_TRUE(fit.refused);
    EXPECT_TRUE(fit.warning);
    EXPECT_GT(fit.requiredBytes, fit.shareBytes);
    EXPECT_EQ(fit.shareBytes, (std::size_t{1u} << 30) / 4u);
    EXPECT_NE(fit.refusalMessage.find("rank 3"), std::string::npos);
    EXPECT_NE(fit.refusalMessage.find("requires"), std::string::npos);
    EXPECT_NE(fit.refusalMessage.find("device capacity"), std::string::npos);
    EXPECT_NE(fit.refusalMessage.find("ranks per device"), std::string::npos);
    EXPECT_NE(fit.refusalMessage.find("refusing to start"), std::string::npos);
    EXPECT_THROW(picBunch::enforceDeviceFit(fit), std::runtime_error);
    try
    {
        picBunch::enforceDeviceFit(fit);
        FAIL() << "expected the placement check to refuse the oversized run";
    }
    catch(std::runtime_error const& error)
    {
        EXPECT_EQ(std::string{error.what()}, fit.refusalMessage);
    }
}

TEST(DeviceAffinity, SingleRankPerDeviceHasOneToOneMappingAndNoWarning)
{
    std::vector<picBunch::DeviceDescriptor> const devices = syntheticDevices(2);
    std::vector<int> const counts = picBunch::ranksPerDevice(2, 2);
    ASSERT_EQ(counts.size(), 2u);
    EXPECT_EQ(counts[0], 1);
    EXPECT_EQ(counts[1], 1);

    picBunch::RankDeviceMapping const first
        = picBunch::assignRankDevice(0, 0, 2, devices);
    picBunch::RankDeviceMapping const second
        = picBunch::assignRankDevice(1, 1, 2, devices);
    EXPECT_EQ(first.deviceIndex, 0);
    EXPECT_EQ(second.deviceIndex, 1);
    EXPECT_EQ(first.ranksPerDevice, 1);
    EXPECT_EQ(second.ranksPerDevice, 1);
    EXPECT_NE(first.deviceIndex, second.deviceIndex);

    picBunch::DeviceWorkload workload;
    workload.paddedNodes = 1024u;
    workload.spectrumNodes = 512u;
    workload.outputNodes = 1024u;
    workload.particleCount = 1000u;
    workload.velocityBins = 1;
    workload.rankCount = 1;
    workload.distributed = false;
    picBunch::DeviceFit const fit = picBunch::evaluateDeviceFit(first, workload);
    EXPECT_FALSE(fit.warning);
    EXPECT_FALSE(fit.refused);
    EXPECT_TRUE(fit.warningMessage.empty());
    EXPECT_TRUE(fit.refusalMessage.empty());
}

TEST(DeviceAffinity, EstimateUsesModePrecisionAndVelocityBins)
{
    picBunch::DeviceWorkload workload;
    workload.paddedNodes = std::size_t{64u} << 20;
    workload.spectrumNodes = std::size_t{32u} << 20;
    workload.outputNodes = std::size_t{16u} << 20;
    workload.particleCount = std::size_t{8u} << 20;
    workload.velocityBins = 1;
    workload.rankCount = 4;
    workload.distributed = false;
    std::size_t const replicated = picBunch::estimateDeviceRequirementBytes(workload);

    workload.distributed = true;
    std::size_t const distributed = picBunch::estimateDeviceRequirementBytes(workload);
    EXPECT_LT(distributed, replicated);

    workload.distributed = false;
    workload.precisionBytes = 4;
    std::size_t const single = picBunch::estimateDeviceRequirementBytes(workload);
    workload.precisionBytes = 8;
    std::size_t const doublePrecision = picBunch::estimateDeviceRequirementBytes(workload);
    EXPECT_LT(single, doublePrecision);

    // More bins reduce the largest single-bin particle staging (buffers are reused).
    workload.precisionBytes = 8;
    workload.velocityBins = 8;
    std::size_t const manyBins = picBunch::estimateDeviceRequirementBytes(workload);
    EXPECT_LT(manyBins, doublePrecision);
}
