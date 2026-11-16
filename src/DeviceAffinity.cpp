// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera

#include "picBunch/DeviceAffinity.hpp"

#include "picBunch/Mpi.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace picBunch
{
    namespace
    {
        std::size_t ceilDiv(std::size_t value, std::size_t divisor)
        {
            return value / divisor + (value % divisor != 0u ? 1u : 0u);
        }

        std::size_t saturatingMul(std::size_t a, std::size_t b)
        {
            if(a != 0u && b > std::numeric_limits<std::size_t>::max() / a)
                return std::numeric_limits<std::size_t>::max();
            return a * b;
        }

        std::size_t saturatingAdd(std::size_t a, std::size_t b)
        {
            if(b > std::numeric_limits<std::size_t>::max() - a)
                return std::numeric_limits<std::size_t>::max();
            return a + b;
        }

        //! Environment variable that filters devices for the compiled backend.
        char const* deviceOverrideVariable()
        {
#if defined(PICBUNCH_BACKEND_CUDA)
            return "CUDA_VISIBLE_DEVICES";
#elif defined(PICBUNCH_BACKEND_HIP)
            return "HIP_VISIBLE_DEVICES";
#else
            return nullptr;
#endif
        }

        std::optional<std::string> deviceOverrideValue()
        {
            char const* const name = deviceOverrideVariable();
            if(name == nullptr)
                return std::nullopt;
            char const* const value = std::getenv(name);
            if(value == nullptr || value[0] == '\0' || std::strcmp(value, "-1") == 0)
                return std::nullopt;
            return std::string{value};
        }
    } // namespace

    int rankLocalRank()
    {
        return mpiLocalRank();
    }

    int rankLocalSize()
    {
        return mpiLocalSize();
    }

    std::vector<DeviceDescriptor> visibleComputeDevices()
    {
#if defined(PICBUNCH_BACKEND_CUDA)
        auto selector
            = alpaka::onHost::makeDeviceSelector(alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu);
#elif defined(PICBUNCH_BACKEND_HIP)
        auto selector
            = alpaka::onHost::makeDeviceSelector(alpaka::api::hip, alpaka::deviceKind::amdGpu);
#else
        auto selector
            = alpaka::onHost::makeDeviceSelector(alpaka::api::host, alpaka::deviceKind::cpu);
#endif
        std::uint32_t const count = selector.getDeviceCount();
        std::vector<DeviceDescriptor> devices;
        devices.reserve(count);
        for(std::uint32_t index = 0u; index < count; ++index)
        {
            auto const properties = selector.getDeviceProperties(index);
            devices.push_back(DeviceDescriptor{
                static_cast<int>(index),
                properties.getName(),
                static_cast<std::size_t>(properties.globalMemCapacityBytes)});
        }
        if(devices.empty())
            throw std::runtime_error(
                "rank-to-device mapping: no visible compute device for the compiled backend");
        return devices;
    }

    int deviceIndexForRank(int localRank, int deviceCount)
    {
        if(deviceCount <= 0)
            throw std::invalid_argument(
                "rank-to-device mapping: visible device count must be positive");
        if(localRank < 0)
            throw std::invalid_argument("rank-to-device mapping: local rank must be non-negative");
        return localRank % deviceCount;
    }

    std::vector<int> ranksPerDevice(int rankCount, int deviceCount)
    {
        if(rankCount < 0 || deviceCount <= 0)
            return {};
        std::vector<int> counts(static_cast<std::size_t>(deviceCount), rankCount / deviceCount);
        int const remainder = rankCount % deviceCount;
        for(int index = 0; index < remainder; ++index)
            ++counts[static_cast<std::size_t>(index)];
        return counts;
    }

    RankDeviceMapping assignRankDevice(
        int worldRank,
        int localRank,
        int localRankCount,
        std::vector<DeviceDescriptor> const& devices,
        bool overrideRespected,
        std::string overrideVariable,
        std::string overrideValue)
    {
        if(devices.empty())
            throw std::invalid_argument("rank-to-device mapping: no visible compute device");
        int const deviceCount = static_cast<int>(devices.size());
        int const deviceIndex = deviceIndexForRank(localRank, deviceCount);
        std::vector<int> const counts = ranksPerDevice(localRankCount, deviceCount);

        RankDeviceMapping mapping;
        mapping.worldRank = worldRank;
        mapping.localRank = localRank;
        mapping.deviceIndex = deviceIndex;
        mapping.visibleDeviceCount = deviceCount;
        mapping.deviceName = devices[static_cast<std::size_t>(deviceIndex)].name;
        mapping.deviceCapacityBytes
            = devices[static_cast<std::size_t>(deviceIndex)].capacityBytes;
        mapping.ranksPerDevice
            = counts.empty() ? 1 : counts[static_cast<std::size_t>(deviceIndex)];
        mapping.overrideRespected = overrideRespected;
        mapping.overrideVariable = std::move(overrideVariable);
        mapping.overrideValue = std::move(overrideValue);
        return mapping;
    }

    RankDeviceMapping currentRankDeviceMapping()
    {
        static std::mutex mutex;
        static std::optional<RankDeviceMapping> cached;
        std::lock_guard<std::mutex> lock{mutex};
        if(cached)
            return *cached;

        std::vector<DeviceDescriptor> const devices = visibleComputeDevices();
        char const* const variable = deviceOverrideVariable();
        std::optional<std::string> const overrideValue = deviceOverrideValue();
        cached = assignRankDevice(
            mpiRank(),
            rankLocalRank(),
            rankLocalSize(),
            devices,
            overrideValue.has_value(),
            overrideValue ? std::string{variable} : std::string{},
            overrideValue ? *overrideValue : std::string{});
        return *cached;
    }

    std::vector<RankDeviceMapping> gatherRankDeviceMappings(RankDeviceMapping const& local)
    {
        return mpiGatherDeviceMappings(local);
    }

    std::string formatByteCount(std::size_t bytes)
    {
        constexpr double kib = 1024.0;
        constexpr double mib = kib * 1024.0;
        constexpr double gib = mib * 1024.0;
        char text[64];
        double const value = static_cast<double>(bytes);
        if(value >= gib)
            std::snprintf(text, sizeof(text), "%.2f GiB", value / gib);
        else if(value >= mib)
            std::snprintf(text, sizeof(text), "%.2f MiB", value / mib);
        else if(value >= kib)
            std::snprintf(text, sizeof(text), "%.2f KiB", value / kib);
        else
            std::snprintf(text, sizeof(text), "%zu B", bytes);
        return std::string{text};
    }

    std::string formatDeviceMapping(RankDeviceMapping const& mapping)
    {
        std::string text = "rank " + std::to_string(mapping.worldRank) + " (local "
            + std::to_string(mapping.localRank) + ") -> device "
            + std::to_string(mapping.deviceIndex) + " \"" + mapping.deviceName + "\" of "
            + std::to_string(mapping.visibleDeviceCount) + " visible ("
            + formatByteCount(mapping.deviceCapacityBytes) + ", "
            + std::to_string(mapping.ranksPerDevice) + " rank"
            + (mapping.ranksPerDevice == 1 ? "" : "s") + " per device)";
        if(mapping.overrideRespected)
            text += "; " + mapping.overrideVariable + "=" + mapping.overrideValue + " respected";
        return text;
    }

    void logDeviceMapping()
    {
        RankDeviceMapping const local = currentRankDeviceMapping();
        std::vector<RankDeviceMapping> const all = gatherRankDeviceMappings(local);
        if(mpiRank() == 0)
        {
            std::printf(
                "device mapping: %d visible compute device%s\n",
                local.visibleDeviceCount,
                local.visibleDeviceCount == 1 ? "" : "s");
            for(auto const& mapping : all)
                std::printf("  %s\n", formatDeviceMapping(mapping).c_str());
            if(local.overrideRespected)
                std::printf(
                    "  override respected: %s=%s; the automatic index is relative to this "
                    "visible list\n",
                    local.overrideVariable.c_str(),
                    local.overrideValue.c_str());
        }
        char const* const logAll = std::getenv("PICBUNCH_DEVICE_LOG");
        if(logAll != nullptr && std::strcmp(logAll, "all") == 0)
            std::printf("%s\n", formatDeviceMapping(local).c_str());
    }

    void printDeviceMappingSummary()
    {
        RankDeviceMapping const local = currentRankDeviceMapping();
        std::vector<RankDeviceMapping> const all = gatherRankDeviceMappings(local);
        if(mpiRank() != 0)
            return;
        std::string placement;
        for(auto const& mapping : all)
        {
            if(!placement.empty())
                placement += ",";
            placement += std::to_string(mapping.worldRank) + "->"
                + std::to_string(mapping.deviceIndex);
        }
        std::printf(
            "run summary: ranks=%zu visible_devices=%d placement=%s",
            all.size(),
            local.visibleDeviceCount,
            placement.c_str());
        if(local.overrideRespected)
            std::printf(
                " override=%s:%s",
                local.overrideVariable.c_str(),
                local.overrideValue.c_str());
        std::printf("\n");
    }

    std::size_t estimateDeviceRequirementBytes(DeviceWorkload const& workload)
    {
        unsigned const precision = std::max(1u, workload.precisionBytes);
        std::size_t const complexBytes = 2u * static_cast<std::size_t>(precision);
        std::size_t const ranks
            = static_cast<std::size_t>(std::max(1, workload.rankCount));
        std::size_t const bins = std::max<std::size_t>(1u, workload.velocityBins);
        std::size_t const particlesPerBin = ceilDiv(workload.particleCount, bins);

        std::size_t gridBytes = 0u;
        if(workload.distributed)
        {
            // One complex work slab, two complex transpose buffers, two complex
            // spectrum buffers (rho and kernel), and the output staging of this
            // rank's z-slab.
            std::size_t const padded = ceilDiv(workload.paddedNodes, ranks);
            std::size_t const spectrum = ceilDiv(workload.spectrumNodes, ranks);
            std::size_t const output = ceilDiv(workload.outputNodes, ranks);
            gridBytes = saturatingAdd(
                saturatingMul(padded, 5u * complexBytes),
                saturatingAdd(
                    saturatingMul(spectrum, 2u * complexBytes),
                    saturatingMul(output, static_cast<std::size_t>(precision))));
        }
        else
        {
            // One real padded array (rho, kernel and inverse output alias it),
            // two distinct packed half-spectra plus the cached kernel spectrum,
            // and the compact output staging plus six accumulators.
            gridBytes = saturatingAdd(
                saturatingMul(workload.paddedNodes, static_cast<std::size_t>(precision)),
                saturatingAdd(
                    saturatingMul(workload.spectrumNodes, 3u * complexBytes),
                    saturatingMul(workload.outputNodes, 7u * static_cast<std::size_t>(precision))));
        }

        // The binned TSC deposit keeps one 32-byte ordered payload per particle
        // of the current bin and, for boxes within the wide-mesh limit, one
        // uint32 histogram counter per padded cell.  Both are freed before the
        // next bin, so the peak is the largest bin, not bins times the payload.
        std::size_t particleBytes = saturatingMul(particlesPerBin, 32u);
        if(workload.paddedNodes <= (std::size_t{1u} << 26))
            particleBytes = saturatingAdd(particleBytes, saturatingMul(workload.paddedNodes, 4u));

        return saturatingAdd(gridBytes, particleBytes);
    }

    DeviceFit evaluateDeviceFit(
        RankDeviceMapping const& mapping,
        DeviceWorkload const& workload)
    {
        DeviceFit fit;
        fit.requiredBytes = estimateDeviceRequirementBytes(workload);
        fit.capacityBytes = mapping.deviceCapacityBytes;
        fit.ranksPerDevice = std::max(1, mapping.ranksPerDevice);
        fit.deviceIndex = mapping.deviceIndex;
        fit.rank = mapping.worldRank;
        fit.shareBytes = fit.capacityBytes / static_cast<std::size_t>(fit.ranksPerDevice);
        fit.warning = fit.ranksPerDevice > 1;
        fit.refused = fit.capacityBytes > 0u && fit.requiredBytes > fit.shareBytes;

        if(fit.warning)
            fit.warningMessage = std::to_string(fit.ranksPerDevice)
                + " ranks per device on device " + std::to_string(fit.deviceIndex) + " (\""
                + mapping.deviceName + "\"): estimated " + formatByteCount(fit.requiredBytes)
                + " required per rank, " + formatByteCount(fit.shareBytes)
                + " available per rank (" + formatByteCount(fit.capacityBytes)
                + " device capacity)";
        if(fit.refused)
            fit.refusalMessage = "rank " + std::to_string(fit.rank) + " requires "
                + formatByteCount(fit.requiredBytes) + " on device "
                + std::to_string(fit.deviceIndex) + " (\"" + mapping.deviceName
                + "\"), but the " + formatByteCount(fit.capacityBytes)
                + " device capacity divided by " + std::to_string(fit.ranksPerDevice)
                + " ranks per device is " + formatByteCount(fit.shareBytes)
                + "; refusing to start";
        return fit;
    }

    void enforceDeviceFit(DeviceFit const& fit)
    {
        if(fit.refused)
            throw std::runtime_error(fit.refusalMessage);
    }

    DeviceFit enforceRankDeviceFit(DeviceWorkload const& workload)
    {
        DeviceFit const fit = evaluateDeviceFit(currentRankDeviceMapping(), workload);
        static bool warningPrinted = false;
        if(fit.warning && !warningPrinted && mpiRank() == 0)
        {
            warningPrinted = true;
            std::fprintf(stderr, "picBunch: warning: %s\n", fit.warningMessage.c_str());
        }
        enforceDeviceFit(fit);
        return fit;
    }
} // namespace picBunch
