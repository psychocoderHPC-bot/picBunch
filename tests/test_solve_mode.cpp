// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera

#include "picBunch/Config.hpp"
#include "picBunch/DeviceAffinity.hpp"
#include "picBunch/Mpi.hpp"
#include "picBunch/SolveMode.hpp"
#include "picBunch/Types.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(PICBUNCH_ENABLE_MPI)
#    include <mpi.h>
#endif

namespace
{
    // The unit-test binary does not run the application's ``MpiGuard``.  The
    // other MPI tests initialise lazily as well; the process-exit finalizer in
    // test_mpi_options.cpp finalizes once.
    void ensureMpiInitialised()
    {
#if defined(PICBUNCH_ENABLE_MPI)
        int initialised = 0;
        MPI_Initialized(&initialised);
        if(!initialised)
            MPI_Init(nullptr, nullptr);
#endif
    }

    picBunch::DeviceFit makeFit(
        std::size_t requiredBytes,
        std::size_t shareBytes,
        bool refused = false)
    {
        picBunch::DeviceFit fit;
        fit.requiredBytes = requiredBytes;
        fit.capacityBytes = shareBytes;
        fit.shareBytes = shareBytes;
        fit.refused = refused;
        return fit;
    }

    std::string const validConfig = R"(
[picongpu]
delta_t = 1.0e-15
cell_size = [1.0e-7, 1.0e-7, 1.0e-7]
base_density = 1.0e24
typical_particles_per_cell = 2
step = 0
species = "electrons"

[domain]
cells = [16, 16, 16]

[placement]
position = [0.5, 0.5, 0.5]
offset_unit = "fraction"

[accuracy]
tolerance = 1.0e-3
)";

    class ScopedEnv
    {
    public:
        ScopedEnv(char const* name, char const* value)
            : m_name(name)
            , m_hadValue(std::getenv(name) != nullptr)
            , m_oldValue(m_hadValue ? std::getenv(name) : "")
        {
            setenv(name, value, 1);
        }
        ~ScopedEnv()
        {
            if(m_hadValue)
                setenv(m_name.c_str(), m_oldValue.c_str(), 1);
            else
                unsetenv(m_name.c_str());
        }
        ScopedEnv(ScopedEnv const&) = delete;
        ScopedEnv& operator=(ScopedEnv const&) = delete;

    private:
        std::string m_name;
        bool m_hadValue;
        std::string m_oldValue;
    };

    std::string writeConfig(std::string const& name, std::string const& text)
    {
        auto const directory
            = std::filesystem::temp_directory_path() / "picbunch-solve-mode-test";
        std::filesystem::create_directories(directory);
        auto const path = directory / name;
        std::ofstream output(path);
        output << text;
        return path.string();
    }
} // namespace

TEST(SolveMode, ParseAcceptsTheFourSpellings)
{
    ASSERT_TRUE(picBunch::parseSolveMode("auto").has_value());
    EXPECT_EQ(*picBunch::parseSolveMode("auto"), picBunch::SolveMode::Auto);
    ASSERT_TRUE(picBunch::parseSolveMode("chunked").has_value());
    EXPECT_EQ(*picBunch::parseSolveMode("chunked"), picBunch::SolveMode::Chunked);
    for(char const* text : {"none", "in-core"})
    {
        ASSERT_TRUE(picBunch::parseSolveMode(text).has_value());
        EXPECT_EQ(*picBunch::parseSolveMode(text), picBunch::SolveMode::InCore);
    }
}

TEST(SolveMode, ParseRejectsGarbage)
{
    for(char const* text : {"", "AUTO", "core", "auto ", "chunk", "in_core", "incore"})
        EXPECT_FALSE(picBunch::parseSolveMode(text).has_value());
}

TEST(SolveMode, AutoFitsKeepsInCore)
{
    auto const decision
        = picBunch::resolveSolveMode(picBunch::SolveMode::Auto, makeFit(100u, 1000u), 0u, false, 1);
    EXPECT_EQ(decision.mode, picBunch::SolveMode::InCore);
    EXPECT_FALSE(decision.switched);
    EXPECT_FALSE(decision.refused);
    EXPECT_EQ(decision.reason, "in-core-fits");
    EXPECT_EQ(decision.requiredBytes, 100u);
    EXPECT_EQ(decision.budgetBytes, 1000u);
}

TEST(SolveMode, AutoSwitchesToChunkedWhenInCoreDoesNotFit)
{
    auto const decision = picBunch::resolveSolveMode(
        picBunch::SolveMode::Auto,
        makeFit(2000u, 1000u, true),
        900u,
        true,
        1);
    EXPECT_EQ(decision.mode, picBunch::SolveMode::Chunked);
    EXPECT_TRUE(decision.switched);
    EXPECT_FALSE(decision.refused);
    EXPECT_EQ(decision.reason, "in-core-does-not-fit;chunked-fits");
    EXPECT_EQ(decision.requiredBytes, 900u);
    EXPECT_EQ(decision.budgetBytes, 1000u);
    EXPECT_NE(
        picBunch::formatSolveModeDecision(decision).find("auto -> chunked"),
        std::string::npos);
}

TEST(SolveMode, AutoRefusesWhenChunkedIsUnavailable)
{
    auto const decision = picBunch::resolveSolveMode(
        picBunch::SolveMode::Auto,
        makeFit(2000u, 1000u, true),
        900u,
        false,
        1);
    EXPECT_TRUE(decision.refused);
    EXPECT_FALSE(decision.switched);
    EXPECT_EQ(decision.reason, "in-core-does-not-fit;chunked-unavailable");
    EXPECT_EQ(decision.requiredBytes, 900u);
    EXPECT_EQ(decision.budgetBytes, 1000u);
    EXPECT_THROW(picBunch::enforceSolveMode(decision), picBunch::SolveModeRefusal);
    std::string const message = picBunch::solveModeRefusalMessage(decision);
    EXPECT_NE(message.find("required"), std::string::npos);
    EXPECT_NE(message.find("capacity"), std::string::npos);
    EXPECT_NE(message.find("--fft-mode distributed"), std::string::npos);
    EXPECT_NE(message.find("--fft-padding 1"), std::string::npos);
}

TEST(SolveMode, AutoRefusesWhenChunkedExceedsBudget)
{
    auto const decision = picBunch::resolveSolveMode(
        picBunch::SolveMode::Auto,
        makeFit(3000u, 1000u, true),
        1500u,
        true,
        1);
    EXPECT_TRUE(decision.refused);
    EXPECT_EQ(decision.reason, "in-core-does-not-fit;chunked-exceeds-budget");
    EXPECT_EQ(decision.requiredBytes, 1500u);
    EXPECT_EQ(decision.budgetBytes, 1000u);
}

TEST(SolveMode, ChunkedBoundaryEqualBudgetFits)
{
    auto const decision = picBunch::resolveSolveMode(
        picBunch::SolveMode::Auto,
        makeFit(2000u, 1000u, true),
        1000u,
        true,
        1);
    EXPECT_EQ(decision.mode, picBunch::SolveMode::Chunked);
    EXPECT_TRUE(decision.switched);
    EXPECT_FALSE(decision.refused);
}

TEST(SolveMode, UnknownCapacityKeepsHistoricalInCore)
{
    auto const decision
        = picBunch::resolveSolveMode(picBunch::SolveMode::Auto, makeFit(5000u, 0u, true), 10u, true, 1);
    EXPECT_EQ(decision.mode, picBunch::SolveMode::InCore);
    EXPECT_FALSE(decision.switched);
    EXPECT_FALSE(decision.refused);
    EXPECT_EQ(decision.reason, "capacity-unknown");
    EXPECT_EQ(decision.budgetBytes, 0u);
    EXPECT_NE(
        picBunch::formatSolveModeDecision(decision).find("capacity unknown"),
        std::string::npos);
}

TEST(SolveMode, ExplicitChunkedIsNeverGatedByTheInCoreFit)
{
    auto const decision = picBunch::resolveSolveMode(
        picBunch::SolveMode::Chunked,
        makeFit(9000u, 10u, true),
        50u,
        false,
        1);
    EXPECT_EQ(decision.mode, picBunch::SolveMode::Chunked);
    EXPECT_FALSE(decision.switched);
    EXPECT_FALSE(decision.refused);
    EXPECT_EQ(decision.reason, "explicit-chunked");
}

TEST(SolveMode, ExplicitInCoreIsAHardOverride)
{
    auto const decision = picBunch::resolveSolveMode(
        picBunch::SolveMode::InCore,
        makeFit(9000u, 10u, true),
        50u,
        true,
        1);
    EXPECT_EQ(decision.mode, picBunch::SolveMode::InCore);
    EXPECT_FALSE(decision.switched);
    EXPECT_FALSE(decision.refused);
    EXPECT_EQ(decision.reason, "explicit-in-core");
}

TEST(SolveMode, EstimateMatchesTheSingleVirtualRankFormula)
{
    // N=1: nz=Mz and nx=Mx, so the bracket is
    // Mx*My*Mz + 2*Mx*My*Mz + 2*max(Mx*Mz, Mz*Mx).
    std::size_t const Mx = 8u;
    std::size_t const My = 6u;
    std::size_t const Mz = 4u;
    std::size_t const particles = 123u;
    std::size_t const volume = Mx * My * Mz;
    std::size_t const expected = 16u * (3u * volume + 2u * Mx * Mz) + particles;
    EXPECT_EQ(
        picBunch::estimateChunkedRequirementBytes(Mx, My, Mz, 1u, particles),
        expected);
    EXPECT_EQ(
        picBunch::estimateChunkedRequirementBytes(Mx, My, Mz, 0u, particles),
        expected);
}

TEST(SolveMode, EstimateUsesTheCeilSplitForVirtualRanks)
{
    // N > 1: nz = ceil(Mz/N) and nx = ceil(Mx/N) are the largest contiguous
    // split counts, so the bracket is
    //   Mx*My*nz + 2*nx*My*Mz + 2*max(Mx*nz, Mz*nx).
    std::size_t const Mx = 13u;
    std::size_t const My = 11u;
    std::size_t const Mz = 15u;
    std::size_t const particles = 321u;
    for(std::size_t ranks : {2u, 3u, 4u, 5u, 7u})
    {
        std::size_t const nz = (Mz + ranks - 1u) / ranks;
        std::size_t const nx = (Mx + ranks - 1u) / ranks;
        std::size_t const expected
            = 16u
                * (Mx * My * nz + 2u * nx * My * Mz
                   + 2u * std::max<std::size_t>(Mx * nz, Mz * nx))
            + particles;
        SCOPED_TRACE("virtual ranks " + std::to_string(ranks));
        EXPECT_EQ(
            picBunch::estimateChunkedRequirementBytes(Mx, My, Mz, ranks, particles),
            expected);
    }
}

TEST(SolveMode, EstimateDoesNotGrowWithVirtualRanks)
{
    std::size_t const Mx = 128u;
    std::size_t const My = 96u;
    std::size_t const Mz = 64u;
    std::size_t const particles = 4096u;
    std::size_t previous
        = picBunch::estimateChunkedRequirementBytes(Mx, My, Mz, 1u, particles);
    for(std::size_t ranks = 2u; ranks <= 16u; ++ranks)
    {
        std::size_t const current
            = picBunch::estimateChunkedRequirementBytes(Mx, My, Mz, ranks, particles);
        EXPECT_LE(current, previous);
        previous = current;
    }
}

TEST(SolveMode, ConfigParsesOptionalSolveTable)
{
    auto const config = picBunch::loadConfig(
        writeConfig("solve.toml", validConfig + "\n[solve]\nsolve_mode = \"chunked\"\n"));
    EXPECT_EQ(config.solve.solveMode, picBunch::SolveMode::Chunked);
    EXPECT_TRUE(config.warnings.empty());
}

TEST(SolveMode, ConfigDefaultsToAuto)
{
    auto const config = picBunch::loadConfig(writeConfig("solve-default.toml", validConfig));
    EXPECT_EQ(config.solve.solveMode, picBunch::SolveMode::Auto);
}

TEST(SolveMode, ConfigRejectsAnInvalidSolveMode)
{
    std::string const text = validConfig + "\n[solve]\nsolve_mode = \"bogus\"\n";
    try
    {
        (void)picBunch::loadConfig(writeConfig("solve-bad.toml", text));
        FAIL() << "loadConfig accepted an invalid solve_mode";
    }
    catch(std::runtime_error const& error)
    {
        EXPECT_NE(std::string(error.what()).find("solve.solve_mode"), std::string::npos);
    }
}

TEST(SolveMode, WireValidatorAcceptsOnlyLegalModes)
{
    for(picBunch::SolveMode mode : {picBunch::SolveMode::Auto,
                                     picBunch::SolveMode::InCore,
                                     picBunch::SolveMode::Chunked})
    {
        auto const decoded = picBunch::solveModeFromWire(static_cast<double>(mode));
        ASSERT_TRUE(decoded.has_value());
        EXPECT_EQ(*decoded, mode);
    }
    // -1 is the refusal sentinel, not a mode; 4 is past the last enumerator
    // (``Auto`` is 3 after it was moved to the end of the enum).
    for(double invalid : {-1.0, 4.0, 0.5, 1.0e300})
        EXPECT_FALSE(picBunch::solveModeFromWire(invalid).has_value());
}

TEST(SolveMode, ChunkedResidentBytesAddsTheSixComponentSlab)
{
    std::size_t const solverBytes = 1000u;
    std::size_t const slabPlanes = 2u;
    std::size_t const planeElems = 10u;
    std::size_t const slabBytes = 6u * slabPlanes * planeElems * sizeof(double);
    EXPECT_EQ(
        picBunch::chunkedResidentBytes(solverBytes, slabPlanes, planeElems),
        solverBytes + slabBytes);
}

TEST(SolveMode, ChunkedResidentBytesCountsTheWholeVolumeForOneRank)
{
    // parSize == 1: one slab is the whole output volume, nz planes of nx*ny.
    std::size_t const nx = 8u;
    std::size_t const ny = 6u;
    std::size_t const nz = 5u;
    std::size_t const solverBytes = 4096u;
    std::size_t const expectedSlab = 6u * nz * nx * ny * sizeof(double);
    EXPECT_EQ(
        picBunch::chunkedResidentBytes(solverBytes, nz, nx * ny),
        solverBytes + expectedSlab);
}

TEST(SolveMode, ChunkedGateRejectsWhenOnlyTheSolverBytesFit)
{
    // The old gate compared the solver requirement alone.  These numbers fit
    // that comparison but exceed the same budget once the output slab is
    // counted, which is the regression the resident term fixes.
    std::size_t const solverBytes = 512u;
    std::size_t const slabPlanes = 3u;
    std::size_t const planeElems = 8u;
    std::size_t const slabBytes = 6u * slabPlanes * planeElems * sizeof(double);
    std::size_t const budget = solverBytes + slabBytes - 1u;
    EXPECT_LE(solverBytes, budget);
    EXPECT_GT(
        picBunch::chunkedResidentBytes(solverBytes, slabPlanes, planeElems),
        budget);
}

TEST(SolveMode, InCoreFitsDecisionPrintsRequiredAndBudget)
{
    // The runtime decision budget is shareBytes = min(capacity, override), so
    // the end-to-end counterpart is only a genuinely reduced budget while the
    // override stays below the real capacity.  This unit case passes a
    // synthetic fit and only pins the formatting numbers.
    auto const decision = picBunch::resolveSolveMode(
        picBunch::SolveMode::Auto,
        makeFit(std::size_t{3} << 30, std::size_t{5} << 30),
        0u,
        false,
        1);
    EXPECT_EQ(
        picBunch::formatSolveModeDecision(decision),
        "picBunch: solve mode: auto -> in-core (required 3.000 GiB, "
        "capacity 5.000 GiB over 1 rank, reason in-core-fits)");
}

TEST(SolveMode, EvaluateDeviceFitReadsTheBudgetAndKeepsTheReportedCapacity)
{
    // The test-only budget override is derived once at the mapping source.  The
    // fit must read it, while the reported and gathered device capacity keeps
    // the true device value.
    constexpr std::size_t deviceCapacity = std::size_t{64} << 30;
    constexpr std::size_t budget = std::size_t{8} << 30;
    std::vector<picBunch::DeviceDescriptor> const devices{
        picBunch::DeviceDescriptor{0, "synthetic", deviceCapacity}};
    ScopedEnv const budgetOverride(
        "PICBUNCH_TEST_MEMORY_BUDGET_BYTES", std::to_string(budget).c_str());

    picBunch::RankDeviceMapping const mapping
        = picBunch::assignRankDevice(0, 0, 1, devices);
    EXPECT_EQ(mapping.deviceCapacityBytes, deviceCapacity);
    EXPECT_EQ(mapping.deviceBudgetBytes, budget);

    picBunch::DeviceWorkload workload;
    workload.paddedNodes = 1024u;
    workload.spectrumNodes = 512u;
    workload.outputNodes = 1024u;
    workload.particleCount = 100u;
    picBunch::DeviceFit const fit = picBunch::evaluateDeviceFit(mapping, workload);
    EXPECT_EQ(fit.capacityBytes, budget);
    EXPECT_EQ(fit.shareBytes, budget);
    // The estimate from these small synthetic dimensions stays below 8 GiB, so
    // the budget, not the real capacity, is the value under test.
    EXPECT_LT(fit.requiredBytes, budget);
}

TEST(SolveModeAggregate, CriterionFromTheReviewNumbers)
{
    // Review case: four ranks on one 64 GiB device, each rank sharing 16 GiB
    // of it.  Per-rank in-core is 18 GiB and the chunked peak is 9 GiB.  The
    // criteria are node totals, so 72 GiB in-core must not fit 64 GiB and the
    // 36 GiB chunked aggregate must.  Run under ``mpirun -n 4``.
    ensureMpiInitialised();
    if(picBunch::mpiSize() != 4)
        GTEST_SKIP() << "requires mpirun -n 4";

    constexpr std::size_t gib = std::size_t{1} << 30;
    std::vector<picBunch::DeviceDescriptor> const devices{
        picBunch::DeviceDescriptor{0, "synthetic", 64u * gib}};
    ScopedEnv const budgetOverride(
        "PICBUNCH_TEST_MEMORY_BUDGET_BYTES", std::to_string(16u * gib).c_str());
    picBunch::RankDeviceMapping const mapping
        = picBunch::assignRankDevice(0, 0, 1, devices);
    EXPECT_EQ(mapping.deviceCapacityBytes, 64u * gib);
    EXPECT_EQ(mapping.deviceBudgetBytes, 16u * gib);

    std::uint64_t totalRequired = 18u * gib;
    std::uint64_t totalCapacity = mapping.deviceBudgetBytes;
    picBunch::mpiSumLocal(totalRequired, totalCapacity);
    std::uint64_t totalChunked = 9u * gib;
    std::uint64_t capacityScratch = 0u;
    picBunch::mpiSumLocal(totalChunked, capacityScratch);
    EXPECT_EQ(totalRequired, 72u * gib);
    EXPECT_EQ(totalCapacity, 64u * gib);
    EXPECT_EQ(totalChunked, 36u * gib);

    picBunch::DeviceFit fit;
    fit.requiredBytes = static_cast<std::size_t>(totalRequired);
    fit.capacityBytes = static_cast<std::size_t>(totalCapacity);
    fit.shareBytes = static_cast<std::size_t>(totalCapacity);
    fit.rankCount = 4;
    fit.refused = !picBunch::totalDeviceMemoryFits(fit.requiredBytes, fit.shareBytes);
    EXPECT_TRUE(fit.refused);
    picBunch::SolveModeDecision const decision = picBunch::resolveSolveMode(
        picBunch::SolveMode::Auto,
        fit,
        static_cast<std::size_t>(totalChunked),
        true,
        4);
    EXPECT_EQ(decision.mode, picBunch::SolveMode::Chunked);
    EXPECT_TRUE(decision.switched);
    EXPECT_FALSE(decision.refused);
    EXPECT_EQ(decision.reason, "in-core-does-not-fit;chunked-fits");
}

TEST(SolveMode, SlabPlanesForRankSplitsDivisibleVolume)
{
    std::size_t const outNz = 12u;
    std::size_t const parSize = 4u;
    std::size_t total = 0u;
    for(std::size_t rank = 0u; rank < parSize; ++rank)
    {
        EXPECT_EQ(picBunch::slabPlanesForRank(rank, parSize, outNz), 3u);
        total += picBunch::slabPlanesForRank(rank, parSize, outNz);
    }
    EXPECT_EQ(total, outNz);
}

TEST(SolveMode, SlabPlanesForRankSplitsNonDivisibleVolume)
{
    std::size_t const outNz = 10u;
    {
        std::size_t const parSize = 3u;
        std::size_t const expected[] = {3u, 3u, 4u};
        std::size_t total = 0u;
        for(std::size_t rank = 0u; rank < parSize; ++rank)
        {
            EXPECT_EQ(
                picBunch::slabPlanesForRank(rank, parSize, outNz),
                expected[rank]);
            total += picBunch::slabPlanesForRank(rank, parSize, outNz);
        }
        EXPECT_EQ(total, outNz);
    }
    {
        std::size_t const parSize = 4u;
        std::size_t const expected[] = {2u, 3u, 2u, 3u};
        std::size_t total = 0u;
        for(std::size_t rank = 0u; rank < parSize; ++rank)
        {
            EXPECT_EQ(
                picBunch::slabPlanesForRank(rank, parSize, outNz),
                expected[rank]);
            total += picBunch::slabPlanesForRank(rank, parSize, outNz);
        }
        EXPECT_EQ(total, outNz);
    }
}

TEST(SolveMode, SlabPlanesForRankHandlesMoreRanksThanPlanes)
{
    std::size_t const outNz = 2u;
    std::size_t const parSize = 5u;
    std::size_t const expected[] = {0u, 0u, 1u, 0u, 1u};
    std::size_t total = 0u;
    for(std::size_t rank = 0u; rank < parSize; ++rank)
    {
        EXPECT_EQ(
            picBunch::slabPlanesForRank(rank, parSize, outNz),
            expected[rank]);
        total += picBunch::slabPlanesForRank(rank, parSize, outNz);
    }
    EXPECT_EQ(total, outNz);
}

TEST(SolveMode, SlabPlanesForRankOwnsTheWholeVolumeOnOneRank)
{
    EXPECT_EQ(picBunch::slabPlanesForRank(0u, 1u, 7u), 7u);
}
