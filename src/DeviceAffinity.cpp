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
#include <string_view>
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

        /*! Test-only per-rank memory budget override.
         *
         * Exists only so tests can force the automatic solve-mode decision on
         * a machine whose real capacity is too large to switch.  The value is
         * a byte count, optionally a comma-separated list indexed by MPI rank.
         * A single value applies to every rank and a list shorter than the
         * rank count repeats its last value.  Unset, empty or unparsable input
         * falls back to the capacity the backend reported, so it is inert then.
         *
         * The override is read once while the mapping is built and stored in
         * ``RankDeviceMapping::deviceBudgetBytes``.  The automatic decision,
         * the ``mpiSumLocal`` node total and the post-resolution in-core guard
         * then all read that one budget field, while the reported
         * ``deviceCapacityBytes`` keeps the true device capacity.
         */
        std::size_t deviceCapacityBytesForRank(RankDeviceMapping const& mapping)
        {
            char const* const raw = std::getenv("PICBUNCH_TEST_MEMORY_BUDGET_BYTES");
            if(raw == nullptr || raw[0] == '\0')
                return mapping.deviceCapacityBytes;
            std::vector<std::size_t> budgets;
            std::string_view text{raw};
            while(true)
            {
                std::size_t const comma = text.find(',');
                std::string_view token = text.substr(0u, comma);
                while(!token.empty()
                      && (token.front() == ' ' || token.front() == '\t'))
                    token.remove_prefix(1u);
                while(!token.empty()
                      && (token.back() == ' ' || token.back() == '\t'))
                    token.remove_suffix(1u);
                if(token.empty())
                    return mapping.deviceCapacityBytes;
                std::size_t value = 0u;
                for(char const digit : token)
                {
                    if(digit < '0' || digit > '9')
                        return mapping.deviceCapacityBytes;
                    std::size_t const next = value * 10u + static_cast<std::size_t>(digit - '0');
                    if(next < value)
                        return mapping.deviceCapacityBytes;
                    value = next;
                }
                budgets.push_back(value);
                if(comma == std::string_view::npos)
                    break;
                text.remove_prefix(comma + 1u);
            }
            if(budgets.empty())
                return mapping.deviceCapacityBytes;
            int const rank = std::max(0, mpiRank());
            std::size_t const index
                = std::min(static_cast<std::size_t>(rank), budgets.size() - 1u);
            return budgets[index];
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
        // Derive the test-only budget once, from the true capacity, so the fit,
        // the node-total sum and the in-core guard all read the same value.
        // Reporting and the gather wire keep ``deviceCapacityBytes``.
        mapping.deviceBudgetBytes = deviceCapacityBytesForRank(mapping);
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
            + formatByteCount(mapping.deviceCapacityBytes) + ")";
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

    bool totalDeviceMemoryFits(std::size_t totalRequiredBytes, std::size_t totalCapacityBytes)
    {
        return totalCapacityBytes == 0u || totalRequiredBytes <= totalCapacityBytes;
    }

    DeviceFit evaluateDeviceFit(
        RankDeviceMapping const& mapping,
        DeviceWorkload const& workload)
    {
        DeviceFit fit;
        fit.requiredBytes = estimateDeviceRequirementBytes(workload);
        // A mapping decoded from a wire that predates ``deviceBudgetBytes`` (or
        // one built without the field) carries a zero budget.  Falling back to
        // the real capacity keeps the fit enforced instead of silently
        // disabling it; a genuinely unknown capacity is still zero and remains
        // unenforced.
        std::size_t const decisionBudget = mapping.deviceBudgetBytes != 0u
            ? mapping.deviceBudgetBytes
            : mapping.deviceCapacityBytes;
        fit.decisionBudgetBytes = decisionBudget;
        fit.shareBytes = decisionBudget;
        fit.ranksPerDevice = 1;
        fit.deviceIndex = mapping.deviceIndex;
        fit.rank = mapping.worldRank;
        return fit;
    }

    void enforceDeviceFit(DeviceFit const& fit)
    {
        if(fit.refused)
            throw std::runtime_error(fit.refusalMessage);
    }

    DeviceFit enforceRankDeviceFit(DeviceWorkload const& workload)
    {
        DeviceFit fit = evaluateDeviceFit(currentRankDeviceMapping(), workload);
        std::uint64_t totalRequired = static_cast<std::uint64_t>(fit.requiredBytes);
        std::uint64_t totalCapacity = static_cast<std::uint64_t>(fit.decisionBudgetBytes);
        mpiSumLocal(totalRequired, totalCapacity);
        fit.totalRequiredBytes = static_cast<std::size_t>(totalRequired);
        fit.totalCapacityBytes = static_cast<std::size_t>(totalCapacity);
        fit.rankCount = std::max(1, rankLocalSize());
        // The aggregate can hide a rank whose own requirement exceeds its own
        // device; combine both bounds.  The local term uses only this rank's
        // values, so it adds no collective.
        bool const aggregateRefused
            = !totalDeviceMemoryFits(fit.totalRequiredBytes, fit.totalCapacityBytes);
        fit.refused = deviceFitRefused(
            fit.requiredBytes,
            fit.decisionBudgetBytes,
            fit.totalRequiredBytes,
            fit.totalCapacityBytes);
        if(fit.refused)
        {
            if(aggregateRefused)
                fit.refusalMessage = "total estimated device memory "
                    + formatByteCount(fit.totalRequiredBytes) + " over "
                    + std::to_string(fit.rankCount) + " local rank"
                    + (fit.rankCount == 1 ? "" : "s")
                    + " exceeds the total device capacity "
                    + formatByteCount(fit.totalCapacityBytes) + "; refusing to start";
            else
                fit.refusalMessage = "this rank's estimated device memory "
                    + formatByteCount(fit.requiredBytes)
                    + " exceeds its device budget "
                    + formatByteCount(fit.decisionBudgetBytes)
                    + " even though the local totals fit; refusing to start";
        }
        enforceDeviceFit(fit);
        return fit;
    }

    DeviceFit enforceLocalDeviceFit(DeviceWorkload const& workload)
    {
        DeviceFit fit = evaluateDeviceFit(currentRankDeviceMapping(), workload);
        fit.totalRequiredBytes = fit.requiredBytes;
        fit.totalCapacityBytes = fit.decisionBudgetBytes;
        fit.rankCount = 1;
        fit.refused
            = !totalDeviceMemoryFits(fit.requiredBytes, fit.decisionBudgetBytes);
        if(fit.refused)
            fit.refusalMessage = "this rank's estimated device memory "
                + formatByteCount(fit.requiredBytes) + " exceeds its device budget "
                + formatByteCount(fit.decisionBudgetBytes) + "; refusing to start";
        enforceDeviceFit(fit);
        return fit;
    }
} // namespace picBunch
