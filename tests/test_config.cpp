// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#include "picBunch/Config.hpp"
#include "picBunch/Placement.hpp"
#include "picBunch/Types.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace
{
    struct ConfigTest
    {
        ConfigTest()
        {
            directory = std::filesystem::temp_directory_path() / "picbunch-config-test";
            std::filesystem::remove_all(directory);
            std::filesystem::create_directories(directory);
        }
        ~ConfigTest()
        {
            std::filesystem::remove_all(directory);
        }
        std::string write(std::string const& name, std::string const& text)
        {
            auto const path = directory / name;
            std::ofstream output(path);
            output << text;
            return path.string();
        }
        std::filesystem::path directory;
    };

    std::string const fullConfig = R"(
[picongpu]
delta_t = 1.0e-15
cell_size = [1.0e-7, 1.0e-7, 1.0e-7]
base_mass = 9.1093837139e-31
base_charge = -1.602176634e-19
base_density = 1.0e24
typical_particles_per_cell = 2
step = 42
precision = "float64"
species = "electrons"

[domain]
cells = [32, 24, 16]
origin = [1.0e-6, 2.0e-6, 0.0]

[placement]
position = [0.5, 0.25, 0.75]
offset_unit = "fraction"
patch_count = 4

[accuracy]
tolerance = 1.0e-4
near_shell = "exact"
near_shell_radius = 4.0
near_shell_tile = 8
)";
} // namespace

TEST(Config, ParsesEveryRequiredKey)
{
    ConfigTest fixture;
    auto const config = picBunch::loadConfig(fixture.write("full.toml", fullConfig));
    EXPECT_DOUBLE_EQ(config.picongpu.deltaT, 1.0e-15);
    EXPECT_DOUBLE_EQ(config.picongpu.cellSize[0], 1.0e-7);
    EXPECT_DOUBLE_EQ(config.picongpu.baseMass, 9.1093837139e-31);
    EXPECT_DOUBLE_EQ(config.picongpu.baseCharge, -1.602176634e-19);
    EXPECT_DOUBLE_EQ(config.picongpu.baseDensity, 1.0e24);
    EXPECT_DOUBLE_EQ(config.picongpu.typicalParticlesPerCell, 2.0);
    // n_mpp = 1e24 * 1e-21 / 2 = 500.
    EXPECT_DOUBLE_EQ(config.picongpu.unitTime, 1.0e-15);
    EXPECT_DOUBLE_EQ(config.picongpu.unitSpeed, 2.99792458e8);
    EXPECT_DOUBLE_EQ(config.picongpu.unitLength, 2.99792458e-7);
    EXPECT_DOUBLE_EQ(config.picongpu.unitMass, 9.1093837139e-31 * 500.0);
    EXPECT_DOUBLE_EQ(config.picongpu.unitCharge, 1.602176634e-19 * 500.0);
    EXPECT_DOUBLE_EQ(config.picongpu.dt, 1.0e-15);
    EXPECT_EQ(config.picongpu.step, 42u);
    EXPECT_TRUE(config.picongpu.float64);
    EXPECT_EQ(config.picongpu.species, (std::vector<std::string>{"electrons"}));
    EXPECT_TRUE(config.picongpu.outputSpecies.empty());
    EXPECT_TRUE(config.picongpu.emptySpecies.empty());
    EXPECT_EQ(config.domain.cells, (picBunch::Idx3{32, 24, 16}));
    EXPECT_DOUBLE_EQ(config.domain.cellSize[0], 1.0e-7);
    EXPECT_DOUBLE_EQ(config.domain.origin[0], 1.0e-6);
    EXPECT_TRUE(config.placement.fraction);
    EXPECT_DOUBLE_EQ(config.placement.offset[2], 0.75);
    EXPECT_EQ(
        static_cast<int>(config.placement.patch),
        static_cast<int>(picBunch::Config::Placement::Patch::BoundingBox));
    EXPECT_EQ(config.placement.patchCount, 4u);
    EXPECT_FALSE(config.strict);
    EXPECT_DOUBLE_EQ(config.accuracy.tolerance, 1.0e-4);
    EXPECT_EQ(config.accuracy.nearShell, "exact");
    EXPECT_DOUBLE_EQ(config.accuracy.nearShellRadius, 4.0);
    EXPECT_EQ(config.accuracy.nearShellTile, 8u);
    EXPECT_TRUE(config.warnings.empty());
}

TEST(Config, DerivationMatchesPIConGPUDeckA)
{
    // Deck A from the validation note: dt=0.8e-16, dx=0.1772e-6, rho0=1e25,
    // electrons, TYPICAL_PARTICLES_PER_CELL=2.  The expected values are the
    // closed-form Units struct evaluated with the 2022 CODATA constants.
    ConfigTest fixture;
    std::string const text = R"(
[picongpu]
delta_t = 0.8e-16
cell_size = [0.1772e-6, 0.1772e-6, 0.1772e-6]
base_density = 1.0e25
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
    auto const config = picBunch::loadConfig(fixture.write("deck-a.toml", text));
    double const n = 1.0e25 * std::pow(0.1772e-6, 3) / 2.0;
    EXPECT_DOUBLE_EQ(config.picongpu.unitTime, 0.8e-16);
    EXPECT_DOUBLE_EQ(config.picongpu.unitSpeed, 2.99792458e8);
    EXPECT_DOUBLE_EQ(config.picongpu.unitLength, 2.398339664e-8);
    EXPECT_DOUBLE_EQ(config.picongpu.unitMass, 9.1093837139e-31 * n);
    EXPECT_DOUBLE_EQ(config.picongpu.unitCharge, 1.602176634e-19 * n);
    EXPECT_DOUBLE_EQ(
        config.picongpu.unitEnergy,
        config.picongpu.unitMass * std::pow(config.picongpu.unitLength, 2)
            / std::pow(config.picongpu.unitTime, 2));
    EXPECT_DOUBLE_EQ(
        config.picongpu.unitEField,
        1.0
            / (std::pow(config.picongpu.unitTime, 2) / config.picongpu.unitMass
               / config.picongpu.unitLength * config.picongpu.unitCharge));
    EXPECT_DOUBLE_EQ(
        config.picongpu.unitBField,
        config.picongpu.unitMass / (config.picongpu.unitTime * config.picongpu.unitCharge));
    EXPECT_NEAR(config.picongpu.unitEField, 2.1306362829337473e13, 1e-12 * 2.1306362829337473e13);
    EXPECT_NEAR(config.picongpu.unitBField, 7.107037639131492e4, 1e-12 * 7.107037639131492e4);
}

TEST(Config, DefaultsAreFloat32AndPlainMesh)
{
    ConfigTest fixture;
    std::string text = fullConfig;
    auto erase = [&](std::string const& needle) {
        auto const position = text.find(needle);
        if(position != std::string::npos)
            text.erase(position, needle.size());
    };
    erase("precision = \"float64\"\n");
    erase("near_shell = \"exact\"\n");
    erase("near_shell_tile = 8\n");
    auto const config = picBunch::loadConfig(fixture.write("defaults.toml", text));
    EXPECT_FALSE(config.picongpu.float64);
    EXPECT_EQ(config.accuracy.nearShell, "none");
    EXPECT_EQ(config.accuracy.nearShellTile, 0u);
    EXPECT_DOUBLE_EQ(config.picongpu.baseMass, picBunch::kDeckElectronMass);
    EXPECT_DOUBLE_EQ(config.picongpu.baseCharge, picBunch::kDeckElectronCharge);
    EXPECT_DOUBLE_EQ(config.picongpu.speedOfLight, picBunch::kDeckSpeedOfLight);
    EXPECT_DOUBLE_EQ(config.picongpu.typicalParticlesPerCell, 2.0);
}

TEST(Config, OptionalSpeedOfLightChangesDerivedLength)
{
    ConfigTest fixture;
    std::string text = fullConfig;
    text.insert(
        text.find("step = 42"),
        "speed_of_light = 1.0e8\n");
    auto const config = picBunch::loadConfig(fixture.write("speed.toml", text));
    EXPECT_DOUBLE_EQ(config.picongpu.speedOfLight, 1.0e8);
    EXPECT_DOUBLE_EQ(config.picongpu.unitSpeed, 1.0e8);
    EXPECT_DOUBLE_EQ(config.picongpu.unitLength, 1.0e-15 * 1.0e8);
}

TEST(Config, MissingKeyNamesTheKey)
{
    ConfigTest fixture;
    std::string text = fullConfig;
    std::string const needle = "base_density = 1.0e24\n";
    auto const position = text.find(needle);
    ASSERT_NE(position, std::string::npos);
    text.erase(position, needle.size());
    try
    {
        (void)picBunch::loadConfig(fixture.write("missing.toml", text));
        FAIL() << "expected a missing-key error";
    }
    catch(std::exception const& error)
    {
        EXPECT_NE(std::string(error.what()).find("base_density"), std::string::npos);
    }
}

TEST(Config, MissingDeltaTIsRejected)
{
    ConfigTest fixture;
    std::string text = fullConfig;
    std::string const needle = "delta_t = 1.0e-15\n";
    auto const position = text.find(needle);
    ASSERT_NE(position, std::string::npos);
    text.erase(position, needle.size());
    try
    {
        (void)picBunch::loadConfig(fixture.write("missing-dt.toml", text));
        FAIL() << "expected a missing-key error";
    }
    catch(std::exception const& error)
    {
        EXPECT_NE(std::string(error.what()).find("delta_t"), std::string::npos);
    }
}

TEST(Config, LegacyUnitKeysWarnAndOverride)
{
    ConfigTest fixture;
    std::string text = fullConfig;
    text.insert(
        text.find("step = 42"),
        "unit_length = 1.0e-7\nunit_e = 1.0e12\nunit_b = 1.0e3\n");
    auto const config = picBunch::loadConfig(fixture.write("legacy.toml", text));
    EXPECT_DOUBLE_EQ(config.picongpu.unitLength, 1.0e-7);
    EXPECT_DOUBLE_EQ(config.picongpu.unitEField, 1.0e12);
    EXPECT_DOUBLE_EQ(config.picongpu.unitBField, 1.0e3);
    std::size_t warned = 0;
    for(auto const& warning : config.warnings)
        if(warning.find("deprecated") != std::string::npos
           && warning.find("overrides the derived unit") != std::string::npos)
            ++warned;
    EXPECT_EQ(warned, 3u);
}

TEST(Config, LegacyDomainCellSizeIsAcceptedWithWarning)
{
    ConfigTest fixture;
    std::string text = fullConfig;
    std::string const needle = "cell_size = [1.0e-7, 1.0e-7, 1.0e-7]\n";
    auto const position = text.find(needle);
    ASSERT_NE(position, std::string::npos);
    text.erase(position, needle.size());
    std::string const domainNeedle = "[domain]\ncells = [32, 24, 16]\n";
    ASSERT_NE(text.find(domainNeedle), std::string::npos);
    text.replace(
        text.find(domainNeedle),
        domainNeedle.size(),
        "[domain]\ncells = [32, 24, 16]\ncell_size = [1.0e-7, 1.0e-7, 1.0e-7]\n");
    auto const config = picBunch::loadConfig(fixture.write("legacy-cell.toml", text));
    EXPECT_DOUBLE_EQ(config.picongpu.cellSize[0], 1.0e-7);
    EXPECT_DOUBLE_EQ(config.domain.cellSize[0], 1.0e-7);
    bool warned = false;
    for(auto const& warning : config.warnings)
        if(warning.find("domain.cell_size") != std::string::npos
           && warning.find("deprecated") != std::string::npos)
            warned = true;
    EXPECT_TRUE(warned);
}

TEST(Config, ConflictingCellSizesAreRejected)
{
    ConfigTest fixture;
    std::string text = fullConfig;
    std::string const needle = "cells = [32, 24, 16]\n";
    ASSERT_NE(text.find(needle), std::string::npos);
    text.replace(
        text.find(needle),
        needle.size(),
        "cells = [32, 24, 16]\ncell_size = [2.0e-7, 1.0e-7, 1.0e-7]\n");
    EXPECT_THROW(
        (void)picBunch::loadConfig(fixture.write("cell-conflict.toml", text)),
        std::runtime_error);
}

TEST(Config, MalformedTomlFails)
{
    ConfigTest fixture;
    try
    {
        (void)picBunch::loadConfig(fixture.write("malformed.toml", "[picongpu\ndelta_t = 1"));
        FAIL() << "expected a parse error";
    }
    catch(std::exception const& error)
    {
        EXPECT_NE(std::string(error.what()).find("malformed TOML"), std::string::npos);
    }
}

TEST(Config, UnknownKeyIsWarned)
{
    ConfigTest fixture;
    std::string text = fullConfig + "\nunknown_top_level = 1\n";
    auto const config = picBunch::loadConfig(fixture.write("unknown.toml", text));
    bool warned = false;
    for(auto const& warning : config.warnings)
        if(warning.find("unknown_top_level") != std::string::npos)
            warned = true;
    EXPECT_TRUE(warned);
}

TEST(Config, StrictRejectsUnknownKey)
{
    ConfigTest fixture;
    std::string text = "strict = true\n" + fullConfig + "\nunknown_top_level = 1\n";
    try
    {
        (void)picBunch::loadConfig(fixture.write("strict-unknown.toml", text));
        FAIL() << "expected an unknown-key error in strict mode";
    }
    catch(std::exception const& error)
    {
        std::string const message = error.what();
        EXPECT_NE(message.find("strict"), std::string::npos);
        EXPECT_NE(message.find("unknown_top_level"), std::string::npos);
    }
}

TEST(Config, StrictAcceptsKnownKeys)
{
    ConfigTest fixture;
    std::string text = "strict = true\n" + fullConfig;
    auto const config = picBunch::loadConfig(fixture.write("strict-ok.toml", text));
    EXPECT_TRUE(config.strict);
    EXPECT_TRUE(config.warnings.empty());
}

TEST(Config, StrictAcceptsNearShellFastOrder)
{
    ConfigTest fixture;
    // The key lives in the [accuracy] table, which is the last table in
    // fullConfig, so appending inherits that table.
    std::string text
        = "strict = true\n" + fullConfig + "near_shell_fast_order = 8\n";
    auto const config
        = picBunch::loadConfig(fixture.write("strict-fast-order.toml", text));
    EXPECT_TRUE(config.strict);
    EXPECT_EQ(config.accuracy.nearShellFastOrder, 8);
    EXPECT_TRUE(config.warnings.empty());
}

TEST(Config, ParsesNearShellObliqueOrder)
{
    ConfigTest fixture;
    std::string autoText
        = "strict = true\n" + fullConfig + "near_shell_oblique_order = \"auto\"\n";
    auto const autoConfig
        = picBunch::loadConfig(fixture.write("oblique-auto.toml", autoText));
    EXPECT_EQ(autoConfig.accuracy.nearShellObliqueOrder, 0);
    EXPECT_TRUE(autoConfig.warnings.empty());

    std::string explicitText
        = "strict = true\n" + fullConfig + "near_shell_oblique_order = 8\n";
    auto const explicitConfig
        = picBunch::loadConfig(fixture.write("oblique-8.toml", explicitText));
    EXPECT_EQ(explicitConfig.accuracy.nearShellObliqueOrder, 8);
    EXPECT_TRUE(explicitConfig.warnings.empty());
}

TEST(Config, RejectsInvalidNearShellObliqueOrder)
{
    ConfigTest fixture;
    std::string low
        = "strict = true\n" + fullConfig + "near_shell_oblique_order = 2\n";
    EXPECT_THROW(
        (void)picBunch::loadConfig(fixture.write("oblique-low.toml", low)),
        std::runtime_error);
    std::string bad
        = "strict = true\n" + fullConfig + "near_shell_oblique_order = \"fast\"\n";
    EXPECT_THROW(
        (void)picBunch::loadConfig(fixture.write("oblique-bad.toml", bad)),
        std::runtime_error);
}

TEST(Config, ParsesDomainPatch)
{
    ConfigTest fixture;
    std::string text = fullConfig;
    text.replace(text.find("offset_unit = \"fraction\""), 24, "offset_unit = \"fraction\"\npatch = \"domain\"");
    auto const config = picBunch::loadConfig(fixture.write("patch.toml", text));
    EXPECT_EQ(
        static_cast<int>(config.placement.patch),
        static_cast<int>(picBunch::Config::Placement::Patch::Domain));
}

TEST(Config, RejectsUnknownPatch)
{
    ConfigTest fixture;
    std::string text = fullConfig;
    text.replace(text.find("offset_unit = \"fraction\""), 24, "offset_unit = \"fraction\"\npatch = \"cube\"");
    EXPECT_THROW(
        (void)picBunch::loadConfig(fixture.write("patch-bad.toml", text)),
        std::runtime_error);
}

TEST(Config, PrecisionSwitchRejectsUnknownValue)
{
    ConfigTest fixture;
    std::string text = fullConfig;
    text.replace(text.find("\"float64\""), 9, "\"float16\"");
    EXPECT_THROW(
        (void)picBunch::loadConfig(fixture.write("precision.toml", text)),
        std::runtime_error);
}

namespace
{
    void replaceOnce(std::string& text, std::string const& needle, std::string const& replacement)
    {
        auto const at = text.find(needle);
        ASSERT_NE(at, std::string::npos);
        text.replace(at, needle.size(), replacement);
    }
} // namespace

TEST(Config, ParsesPositionVectorInMetres)
{
    ConfigTest fixture;
    std::string text = fullConfig;
    replaceOnce(text, "position = [0.5, 0.25, 0.75]", "position = [1.0e-6, 2.0e-6, 3.0e-6]");
    replaceOnce(text, "offset_unit = \"fraction\"", "offset_unit = \"metres\"");
    auto const config = picBunch::loadConfig(fixture.write("position-metres.toml", text));
    EXPECT_FALSE(config.placement.fraction);
    EXPECT_DOUBLE_EQ(config.placement.offset[0], 1.0e-6);
    EXPECT_DOUBLE_EQ(config.placement.offset[1], 2.0e-6);
    EXPECT_DOUBLE_EQ(config.placement.offset[2], 3.0e-6);
    EXPECT_TRUE(config.warnings.empty());
}

TEST(Config, ParsesPositionVectorAsFraction)
{
    ConfigTest fixture;
    auto const config = picBunch::loadConfig(fixture.write("position-fraction.toml", fullConfig));
    EXPECT_TRUE(config.placement.fraction);
    EXPECT_DOUBLE_EQ(config.placement.offset[0], 0.5);
    EXPECT_DOUBLE_EQ(config.placement.offset[1], 0.25);
    EXPECT_DOUBLE_EQ(config.placement.offset[2], 0.75);
    EXPECT_TRUE(config.warnings.empty());
}

TEST(Config, MissingPositionNamesTheKey)
{
    ConfigTest fixture;
    std::string text = fullConfig;
    replaceOnce(text, "position = [0.5, 0.25, 0.75]\n", "");
    try
    {
        (void)picBunch::loadConfig(fixture.write("missing-position.toml", text));
        FAIL() << "expected a missing-position error";
    }
    catch(std::exception const& error)
    {
        EXPECT_NE(std::string(error.what()).find("placement.position"), std::string::npos);
    }
}

TEST(Config, RejectsPositionFractionOutsideRange)
{
    ConfigTest fixture;
    std::string text = fullConfig;
    replaceOnce(text, "position = [0.5, 0.25, 0.75]", "position = [0.5, 1.5, 0.75]");
    EXPECT_THROW(
        (void)picBunch::loadConfig(fixture.write("position-range.toml", text)),
        std::runtime_error);
}

TEST(Config, PositionDrivesTargetCentreFromDomainOrigin)
{
    ConfigTest fixture;
    std::string text = fullConfig;
    replaceOnce(text, "position = [0.5, 0.25, 0.75]", "position = [2.0e-6, 1.0e-6, 1.0e-6]");
    replaceOnce(text, "offset_unit = \"fraction\"", "offset_unit = \"metres\"");
    auto const config = picBunch::loadConfig(fixture.write("position-shift.toml", text));

    picBunch::Bunch bunch;
    bunch.resize(3);
    double const target[3] = {3.0e-6, 3.0e-6, 1.0e-6};
    double const delta[3] = {1.0e-8, 1.0e-8, 1.0e-8};
    for(std::size_t p = 0; p < 3; ++p)
    {
        double const sign = static_cast<double>(p) - 1.0;
        bunch.x[p] = target[0] + sign * delta[0];
        bunch.y[p] = target[1] + sign * delta[1];
        bunch.z[p] = target[2] + sign * delta[2];
        bunch.ux[p] = bunch.uy[p] = bunch.uz[p] = 0.0;
        bunch.weight[p] = 1.0;
        bunch.charge[p] = picBunch::kDeckElectronCharge;
        bunch.restMass[p] = picBunch::kDeckElectronMass;
    }
    auto const placed = picBunch::placeBunch(bunch, config);
    for(int a = 0; a < 3; ++a)
    {
        EXPECT_NEAR(placed.targetCentre[a], config.domain.origin[a] + config.placement.offset[a], 1e-18);
        EXPECT_NEAR(placed.centroid[a], target[a], 1e-18);
        EXPECT_NEAR(placed.shift[a], 0.0, 1e-18);
    }
}

TEST(Config, SpeciesListIsOrderedAndOutputNamesOptional)
{
    ConfigTest fixture;
    std::string text = fullConfig;
    replaceOnce(text, "species = \"electrons\"", "species = [\"b\", \"ions\"]\noutput_species = [\"e\", \"i\"]");
    auto const config = picBunch::loadConfig(fixture.write("species-list.toml", text));
    EXPECT_EQ(config.picongpu.species, (std::vector<std::string>{"b", "ions"}));
    EXPECT_EQ(config.picongpu.outputSpecies, (std::vector<std::string>{"e", "i"}));
    EXPECT_TRUE(config.picongpu.emptySpecies.empty());
}

TEST(Config, EmptySpeciesParsesAndIsNeverAGeneratedName)
{
    ConfigTest fixture;
    std::string text = fullConfig;
    replaceOnce(text, "species = \"electrons\"", "species = [\"b\"]\nempty_species = [\"e\", \"protons\"]");
    auto const config = picBunch::loadConfig(fixture.write("empty-species.toml", text));
    EXPECT_EQ(config.picongpu.species, (std::vector<std::string>{"b"}));
    EXPECT_EQ(config.picongpu.emptySpecies, (std::vector<std::string>{"e", "protons"}));
    EXPECT_TRUE(config.warnings.empty());
}

TEST(Config, RejectsSpeciesThatIsBothFilledAndEmpty)
{
    ConfigTest fixture;
    std::string text = fullConfig;
    replaceOnce(text, "species = \"electrons\"", "species = [\"b\"]\nempty_species = [\"b\"]");
    try
    {
        (void)picBunch::loadConfig(fixture.write("overlap.toml", text));
        FAIL() << "expected a validation error";
    }
    catch(std::runtime_error const& error)
    {
        std::string const message = error.what();
        EXPECT_NE(message.find("empty_species"), std::string::npos) << message;
        EXPECT_NE(message.find("'b'"), std::string::npos) << message;
    }
}

TEST(Config, RejectsOutputSpeciesLengthMismatch)
{
    ConfigTest fixture;
    std::string text = fullConfig;
    replaceOnce(text, "species = \"electrons\"", "species = [\"b\"]\noutput_species = [\"e\", \"f\"]");
    try
    {
        (void)picBunch::loadConfig(fixture.write("output-mismatch.toml", text));
        FAIL() << "expected a validation error";
    }
    catch(std::runtime_error const& error)
    {
        EXPECT_NE(std::string(error.what()).find("output_species"), std::string::npos)
            << error.what();
    }
}

TEST(Config, RejectsDuplicateSpecies)
{
    ConfigTest fixture;
    std::string text = fullConfig;
    replaceOnce(text, "species = \"electrons\"", "species = [\"b\", \"b\"]");
    EXPECT_THROW(
        (void)picBunch::loadConfig(fixture.write("duplicate.toml", text)),
        std::runtime_error);
}

TEST(Config, PlacementCenterNoneMakesPositionOptional)
{
    ConfigTest fixture;
    std::string text = fullConfig;
    replaceOnce(text, "position = [0.5, 0.25, 0.75]\n", "");
    replaceOnce(text, "offset_unit = \"fraction\"\n", "");
    replaceOnce(text, "patch_count = 4", "patch_count = 4\ncenter = \"none\"");
    auto const config = picBunch::loadConfig(fixture.write("center-none.toml", text));
    EXPECT_EQ(
        static_cast<int>(config.placement.centre),
        static_cast<int>(picBunch::Config::Placement::Centre::None));
    // position/offset_unit are optional in this mode; nothing is reported.
    EXPECT_TRUE(config.warnings.empty());
}

TEST(Config, PlacementCenterNoneWarnsThatPositionIsIgnored)
{
    ConfigTest fixture;
    std::string text = fullConfig;
    replaceOnce(text, "patch_count = 4", "patch_count = 4\ncenter = \"none\"");
    auto const config = picBunch::loadConfig(fixture.write("center-none-pos.toml", text));
    ASSERT_EQ(config.warnings.size(), 1u);
    EXPECT_NE(config.warnings[0].find("placement.position"), std::string::npos);
    EXPECT_NE(config.warnings[0].find("ignored"), std::string::npos);
}

TEST(Config, PlacementCenterDefaultIsCentroid)
{
    ConfigTest fixture;
    auto const config = picBunch::loadConfig(fixture.write("center-default.toml", fullConfig));
    EXPECT_EQ(
        static_cast<int>(config.placement.centre),
        static_cast<int>(picBunch::Config::Placement::Centre::Centroid));
}

TEST(Config, RejectsUnknownPlacementCenter)
{
    ConfigTest fixture;
    std::string text = fullConfig;
    replaceOnce(text, "patch_count = 4", "patch_count = 4\ncenter = \"middle\"");
    EXPECT_THROW(
        (void)picBunch::loadConfig(fixture.write("center-bad.toml", text)),
        std::runtime_error);
}

TEST(Config, StrictAcceptsSpeciesListAndCenterKeys)
{
    ConfigTest fixture;
    std::string text = std::string("strict = true\n") + fullConfig;
    replaceOnce(
        text,
        "species = \"electrons\"",
        "species = [\"b\", \"ions\"]\noutput_species = [\"e\", \"i\"]\nempty_species = [\"protons\"]");
    replaceOnce(text, "patch_count = 4", "patch_count = 4\ncenter = \"centroid\"");
    EXPECT_NO_THROW((void)picBunch::loadConfig(fixture.write("strict-new-keys.toml", text)));
}

TEST(Config, CheckpointBackendDefaultsToAuto)
{
    ConfigTest fixture;
    auto const config = picBunch::loadConfig(fixture.write("default.toml", fullConfig));
    EXPECT_EQ(config.checkpoint.backend, "auto");
    EXPECT_TRUE(config.warnings.empty());
}

TEST(Config, CheckpointBackendParsesAndIsAKnownTable)
{
    ConfigTest fixture;
    auto const config = picBunch::loadConfig(fixture.write(
        "adios2.toml", fullConfig + "\n[checkpoint]\nbackend = \"ADIOS2\"\n"));
    EXPECT_EQ(config.checkpoint.backend, "adios2");
    EXPECT_TRUE(config.warnings.empty());
}

TEST(Config, CheckpointBackendRejectsUnknownValue)
{
    ConfigTest fixture;
    EXPECT_THROW(
        (void)picBunch::loadConfig(fixture.write(
            "bad.toml", fullConfig + "\n[checkpoint]\nbackend = \"raw\"\n")),
        std::runtime_error);
}
