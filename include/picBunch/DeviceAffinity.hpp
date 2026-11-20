// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace picBunch
{
    //! One visible compute device as reported by the compiled backend.
    struct DeviceDescriptor
    {
        int index = 0;
        std::string name;
        std::size_t capacityBytes = 0;
    };

    //! Resolved automatic rank-to-device placement for one process.
    struct RankDeviceMapping
    {
        int worldRank = 0;
        int localRank = 0;
        int deviceIndex = 0;
        int visibleDeviceCount = 0;
        std::string deviceName;
        //! True device capacity.  Reporting and the gather wire use this field.
        std::size_t deviceCapacityBytes = 0;
        /*! Bytes this rank may use on the device.
         *
         * Equal to ``deviceCapacityBytes`` unless the test-only budget
         * override (``PICBUNCH_TEST_MEMORY_BUDGET_BYTES``) is active.  The fit
         * and the node-total guard read this field, never the reported
         * capacity, so an override changes the decision without falsifying
         * the logged or gathered device capacity.
         */
        std::size_t deviceBudgetBytes = 0;
        //! Ranks of the shared-memory domain that landed on this device.
        int ranksPerDevice = 1;
        bool overrideRespected = false;
        std::string overrideVariable;
        std::string overrideValue;
    };

    /*! Local rank within the MPI shared-memory domain.
     *
     * Returns the world rank when MPI is not compiled in or the runtime is not
     * initialised, exactly the fallback the automatic mapping documents.
     */
    int rankLocalRank();

    //! Number of ranks in the MPI shared-memory domain (1 without MPI).
    int rankLocalSize();

    //! Visible devices of the compiled backend.  The CPU backend reports one.
    std::vector<DeviceDescriptor> visibleComputeDevices();

    //! Pure mapping rule: ``localRank % deviceCount``.
    int deviceIndexForRank(int localRank, int deviceCount);

    /*! Balanced rank counts per device.
     *
     * Device ``i`` receives ``rankCount / deviceCount`` ranks, and the first
     * ``rankCount % deviceCount`` devices receive one more.  The result is
     * therefore always floor or ceil of ``rankCount / deviceCount`` and is
     * independent of the process that computes it.
     */
    std::vector<int> ranksPerDevice(int rankCount, int deviceCount);

    //! Pure assignment, used by the runtime and by the unit tests.
    RankDeviceMapping assignRankDevice(
        int worldRank,
        int localRank,
        int localRankCount,
        std::vector<DeviceDescriptor> const& devices,
        bool overrideRespected = false,
        std::string overrideVariable = {},
        std::string overrideValue = {});

    //! Resolve and cache this process's mapping, including the chosen device index.
    RankDeviceMapping currentRankDeviceMapping();

    //! Gather every rank's mapping on rank 0 (MPI) or return the local one.
    std::vector<RankDeviceMapping> gatherRankDeviceMappings(
        RankDeviceMapping const& local);

    //! One human-readable mapping line, including the override note.
    std::string formatDeviceMapping(RankDeviceMapping const& mapping);

    //! Print the startup mapping once, from rank 0 (every rank with PICBUNCH_DEVICE_LOG=all).
    void logDeviceMapping();

    //! Print the resolved mapping in the run summary, from rank 0.
    void printDeviceMappingSummary();

    /*! Inputs to the per-rank device-memory estimate.
     *
     * The model follows the buffers that the field pipeline actually keeps
     * resident: the padded real array, the packed half-spectra and the kernel
     * cache, the output staging plus six accumulators, and the one-velocity-bin
     * particle staging.  Buffers are reused across velocity bins, so only the
     * largest bin's particle staging is counted; the bin count divides the
     * particle staging, not the grid buffers.
     */
    struct DeviceWorkload
    {
        std::size_t paddedNodes = 0;
        std::size_t spectrumNodes = 0;
        std::size_t outputNodes = 0;
        std::size_t particleCount = 0;
        std::size_t velocityBins = 1;
        int rankCount = 1;
        bool distributed = false;
        //! Bytes per real value (8 for the double-precision pipeline).
        unsigned precisionBytes = sizeof(double);
    };

    //! Estimated peak device bytes for one rank.
    std::size_t estimateDeviceRequirementBytes(DeviceWorkload const& workload);

    //! True when the summed requirement fits the summed capacity.  A zero
    //! capacity means the backend did not report a limit and is not enforced.
    bool totalDeviceMemoryFits(std::size_t totalRequiredBytes, std::size_t totalCapacityBytes);

    //! Result of comparing the summed estimate with the summed device capacity.
    struct DeviceFit
    {
        std::size_t requiredBytes = 0;
        std::size_t capacityBytes = 0;
        std::size_t shareBytes = 0;
        int ranksPerDevice = 1;
        int deviceIndex = 0;
        int rank = 0;
        std::size_t totalRequiredBytes = 0;
        std::size_t totalCapacityBytes = 0;
        int rankCount = 1;
        bool warning = false;
        bool refused = false;
        std::string warningMessage;
        std::string refusalMessage;
    };

    DeviceFit evaluateDeviceFit(
        RankDeviceMapping const& mapping,
        DeviceWorkload const& workload);

    //! Throw ``std::runtime_error`` with the refusal text when the estimate does not fit.
    void enforceDeviceFit(DeviceFit const& fit);

    //! Evaluate, sum over the shared-memory domain, and refuse when the total exceeds capacity.
    DeviceFit enforceRankDeviceFit(DeviceWorkload const& workload);

    //! Human-readable byte count (GiB/MiB) used by the warning and refusal text.
    std::string formatByteCount(std::size_t bytes);
} // namespace picBunch
