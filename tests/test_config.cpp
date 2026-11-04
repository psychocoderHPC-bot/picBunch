// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#include "picBunch/Config.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

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
unit_length = 1.0e-7
unit_e = 1.0e12
unit_b = 1.0e3
dt = 1.0e-15
step = 42
precision = "float64"
species = "electrons"

[domain]
cells = [32, 24, 16]
cell_size = [1.0e-7, 1.0e-7, 1.0e-7]
origin = [1.0e-6, 2.0e-6, 0.0]

[placement]
left = 0.5
front = 0.25
top = 0.75
offset_unit = "fraction"

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
    EXPECT_DOUBLE_EQ(config.picongpu.unitLength, 1.0e-7);
    EXPECT_DOUBLE_EQ(config.picongpu.unitEField, 1.0e12);
    EXPECT_DOUBLE_EQ(config.picongpu.unitBField, 1.0e3);
    EXPECT_DOUBLE_EQ(config.picongpu.dt, 1.0e-15);
    EXPECT_EQ(config.picongpu.step, 42u);
    EXPECT_TRUE(config.picongpu.float64);
    EXPECT_EQ(config.picongpu.species, "electrons");
    EXPECT_EQ(config.domain.cells, (std::array<std::size_t, 3>{32, 24, 16}));
    EXPECT_DOUBLE_EQ(config.domain.cellSize[0], 1.0e-7);
    EXPECT_DOUBLE_EQ(config.domain.origin[0], 1.0e-6);
    EXPECT_TRUE(config.placement.fraction);
    EXPECT_DOUBLE_EQ(config.placement.offset[2], 0.75);
    EXPECT_DOUBLE_EQ(config.accuracy.tolerance, 1.0e-4);
    EXPECT_EQ(config.accuracy.nearShell, "exact");
    EXPECT_DOUBLE_EQ(config.accuracy.nearShellRadius, 4.0);
    EXPECT_EQ(config.accuracy.nearShellTile, 8u);
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
    auto const config = picBunch::loadConfig(fixture.write("defaults.toml", text));
    EXPECT_FALSE(config.picongpu.float64);
    EXPECT_EQ(config.accuracy.nearShell, "none");
}

TEST(Config, MissingKeyNamesTheKey)
{
    ConfigTest fixture;
    std::string text = fullConfig;
    std::string const needle = "unit_b = 1.0e3\n";
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
        EXPECT_NE(std::string(error.what()).find("unit_b"), std::string::npos);
    }
}

TEST(Config, MalformedTomlFails)
{
    ConfigTest fixture;
    try
    {
        (void)picBunch::loadConfig(fixture.write("malformed.toml", "[picongpu\nunit_length = 1"));
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

TEST(Config, PrecisionSwitchRejectsUnknownValue)
{
    ConfigTest fixture;
    std::string text = fullConfig;
    text.replace(text.find("\"float64\""), 9, "\"float16\"");
    EXPECT_THROW(
        (void)picBunch::loadConfig(fixture.write("precision.toml", text)),
        std::runtime_error);
}
