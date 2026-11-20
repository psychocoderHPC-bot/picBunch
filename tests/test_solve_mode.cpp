// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera

#include "picBunch/Config.hpp"
#include "picBunch/SolveMode.hpp"
#include "picBunch/Types.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace
{
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
    EXPECT_NE(message.find("budget"), std::string::npos);
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
    // -1 is the refusal sentinel, not a mode; 3 is past the last enumerator.
    for(double invalid : {-1.0, 3.0, 0.5, 1.0e300})
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
