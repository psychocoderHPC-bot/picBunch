// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#include "picBunch/Config.hpp"

#include <toml++/toml.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace picBunch
{
    namespace
    {
        [[noreturn]] void fail(std::string const& key, std::string const& why)
        {
            throw std::runtime_error("configuration key '" + key + "': " + why);
        }

        toml::node const& require(toml::table const& table, std::string const& key)
        {
            auto const* node = table.get(key);
            if(node == nullptr)
                fail(key, "required key is missing");
            return *node;
        }

        toml::table const& requireTable(toml::table const& table, std::string const& key)
        {
            auto const* node = require(table, key).as_table();
            if(node == nullptr)
                fail(key, "expected a table");
            return *node;
        }

        double asDouble(toml::node const& node, std::string const& key)
        {
            auto const value = node.value<double>();
            if(!value || !std::isfinite(*value))
                fail(key, "expected a finite number");
            return *value;
        }

        std::int64_t asInteger(toml::node const& node, std::string const& key)
        {
            auto const value = node.value<std::int64_t>();
            if(!value)
                fail(key, "expected an integer");
            return *value;
        }

        std::string asString(toml::node const& node, std::string const& key)
        {
            auto const value = node.value<std::string>();
            if(!value || value->empty())
                fail(key, "expected a non-empty string");
            return *value;
        }

        bool asBool(toml::node const& node, std::string const& key)
        {
            auto const value = node.value<bool>();
            if(!value)
                fail(key, "expected a boolean");
            return *value;
        }

        std::array<double, 3> asTriple(toml::node const& node, std::string const& key)
        {
            auto const* array = node.as_array();
            if(array == nullptr || array->size() != 3u)
                fail(key, "expected an array of three numbers");
            std::array<double, 3> result{};
            for(std::size_t a = 0; a < 3; ++a)
                result[a] = asDouble((*array)[a], key);
            return result;
        }

        std::array<std::size_t, 3> asCells(toml::node const& node, std::string const& key)
        {
            auto const* array = node.as_array();
            if(array == nullptr || array->size() != 3u)
                fail(key, "expected an array of three positive integers");
            std::array<std::size_t, 3> result{};
            for(std::size_t a = 0; a < 3; ++a)
            {
                auto const value = asInteger((*array)[a], key);
                if(value <= 0)
                    fail(key, "every entry must be positive");
                result[a] = static_cast<std::size_t>(value);
            }
            return result;
        }

        void warnUnknown(
            toml::table const& table,
            std::string const& prefix,
            std::set<std::string> const& known,
            std::vector<std::string>& warnings)
        {
            for(auto const& [key, node] : table)
            {
                std::string const name = prefix.empty() ? std::string(key.str()) : prefix + "." + std::string(key.str());
                if(known.find(name) == known.end())
                {
                    warnings.push_back("unknown configuration key '" + name + "' was ignored");
                    continue;
                }
                if(auto const* child = node.as_table())
                    (void)child;
            }
        }
    } // namespace

    Config loadConfig(std::string const& path)
    {
        toml::table document;
        try
        {
            document = toml::parse_file(path);
        }
        catch(toml::parse_error const& error)
        {
            std::ostringstream message;
            message << "malformed TOML configuration '" << path << "': " << error.description()
                    << " (" << error.source().begin << ")";
            throw std::runtime_error(message.str());
        }

        Config config;
        if(auto const* node = document.get("strict"))
            config.strict = asBool(*node, "strict");
        auto const& picongpu = requireTable(document, "picongpu");
        auto const& domain = requireTable(document, "domain");
        auto const& placement = requireTable(document, "placement");
        auto const& accuracy = requireTable(document, "accuracy");

        config.picongpu.unitLength = asDouble(require(picongpu, "unit_length"), "picongpu.unit_length");
        config.picongpu.unitEField = asDouble(require(picongpu, "unit_e"), "picongpu.unit_e");
        config.picongpu.unitBField = asDouble(require(picongpu, "unit_b"), "picongpu.unit_b");
        config.picongpu.dt = asDouble(require(picongpu, "dt"), "picongpu.dt");
        auto const step = asInteger(require(picongpu, "step"), "picongpu.step");
        config.picongpu.species = asString(require(picongpu, "species"), "picongpu.species");

        if(auto const* node = picongpu.get("precision"))
        {
            std::string const precision = asString(*node, "picongpu.precision");
            if(precision == "float32")
                config.picongpu.float64 = false;
            else if(precision == "float64")
                config.picongpu.float64 = true;
            else
                fail("picongpu.precision", "expected \"float32\" or \"float64\"");
        }
        if(auto const* node = picongpu.get("unit_charge"))
            config.picongpu.unitCharge = asDouble(*node, "picongpu.unit_charge");
        if(auto const* node = picongpu.get("unit_mass"))
            config.picongpu.unitMass = asDouble(*node, "picongpu.unit_mass");
        if(auto const* node = picongpu.get("unit_speed"))
            config.picongpu.unitSpeed = asDouble(*node, "picongpu.unit_speed");

        config.domain.cells = asCells(require(domain, "cells"), "domain.cells");
        config.domain.cellSize = asTriple(require(domain, "cell_size"), "domain.cell_size");
        if(auto const* node = domain.get("origin"))
            config.domain.origin = asTriple(*node, "domain.origin");

        config.placement.offset[0] = asDouble(require(placement, "left"), "placement.left");
        config.placement.offset[1] = asDouble(require(placement, "front"), "placement.front");
        config.placement.offset[2] = asDouble(require(placement, "top"), "placement.top");
        std::string const unit = asString(require(placement, "offset_unit"), "placement.offset_unit");
        if(unit == "fraction")
            config.placement.fraction = true;
        else if(unit == "metres")
            config.placement.fraction = false;
        else
            fail("placement.offset_unit", "expected \"fraction\" or \"metres\"");
        if(auto const* node = placement.get("patch"))
        {
            std::string const patch = asString(*node, "placement.patch");
            if(patch == "bbox")
                config.placement.patch = Config::Placement::Patch::BoundingBox;
            else if(patch == "domain")
                config.placement.patch = Config::Placement::Patch::Domain;
            else
                fail("placement.patch", "expected \"bbox\" or \"domain\"");
        }
        if(auto const* node = placement.get("patch_count"))
        {
            auto const count = asInteger(*node, "placement.patch_count");
            if(count <= 0)
                fail("placement.patch_count", "must be positive");
            config.placement.patchCount = static_cast<std::size_t>(count);
        }

        config.accuracy.tolerance = asDouble(require(accuracy, "tolerance"), "accuracy.tolerance");
        if(auto const* node = accuracy.get("near_shell"))
            config.accuracy.nearShell = asString(*node, "accuracy.near_shell");
        if(auto const* node = accuracy.get("near_shell_radius"))
            config.accuracy.nearShellRadius = asDouble(*node, "accuracy.near_shell_radius");
        if(auto const* node = accuracy.get("near_shell_tile"))
        {
            auto const tile = asInteger(*node, "accuracy.near_shell_tile");
            if(tile <= 0)
                fail("accuracy.near_shell_tile", "must be positive");
            config.accuracy.nearShellTile = static_cast<std::size_t>(tile);
        }
        if(auto const* node = accuracy.get("near_shell_fast_order"))
        {
            auto const order = asInteger(*node, "accuracy.near_shell_fast_order");
            if(order < 2 || order > 32)
                fail("accuracy.near_shell_fast_order", "must be between 2 and 32");
            config.accuracy.nearShellFastOrder = static_cast<int>(order);
        }

        // Validation.
        auto positive = [](double value, std::string const& key) {
            if(!(value > 0.0) || !std::isfinite(value))
                fail(key, "must be finite and positive");
        };
        positive(config.picongpu.unitLength, "picongpu.unit_length");
        positive(config.picongpu.unitEField, "picongpu.unit_e");
        positive(config.picongpu.unitBField, "picongpu.unit_b");
        positive(config.picongpu.unitCharge, "picongpu.unit_charge");
        positive(config.picongpu.unitMass, "picongpu.unit_mass");
        positive(config.picongpu.unitSpeed, "picongpu.unit_speed");
        if(config.picongpu.dt < 0.0)
            fail("picongpu.dt", "must be non-negative");
        if(step < 0 || step > 999999)
            fail("picongpu.step", "must be in [0, 999999]");
        config.picongpu.step = static_cast<std::uint64_t>(step);
        for(std::size_t a = 0; a < 3; ++a)
        {
            positive(config.domain.cellSize[a], "domain.cell_size");
            if(!std::isfinite(config.domain.origin[a]))
                fail("domain.origin", "must be finite");
            if(!std::isfinite(config.placement.offset[a]))
                fail("placement.offset", "must be finite");
        }
        if(config.placement.fraction)
            for(std::size_t a = 0; a < 3; ++a)
                if(config.placement.offset[a] < 0.0 || config.placement.offset[a] > 1.0)
                    fail("placement.offset", "fractional offsets must be in [0, 1]");
        positive(config.accuracy.tolerance, "accuracy.tolerance");
        if(config.accuracy.nearShell != "none" && config.accuracy.nearShell != "exact"
           && config.accuracy.nearShell != "fast")
            fail("accuracy.near_shell", "expected \"none\", \"exact\" or \"fast\"");
        if(config.accuracy.nearShellRadius < 0.0 || !std::isfinite(config.accuracy.nearShellRadius))
            fail("accuracy.near_shell_radius", "must be finite and non-negative");
        if(config.accuracy.nearShellTile == 0u)
            fail("accuracy.near_shell_tile", "must be positive");

        // Unknown keys are warnings.  The list is deliberately explicit so a
        // typo is visible instead of silently changing nothing.
        warnUnknown(
            picongpu,
            "picongpu",
            {"picongpu.unit_length",
             "picongpu.unit_e",
             "picongpu.unit_b",
             "picongpu.unit_charge",
             "picongpu.unit_mass",
             "picongpu.unit_speed",
             "picongpu.dt",
             "picongpu.step",
             "picongpu.precision",
             "picongpu.species"},
            config.warnings);
        warnUnknown(
            domain,
            "domain",
            {"domain.cells", "domain.cell_size", "domain.origin"},
            config.warnings);
        warnUnknown(
            placement,
            "placement",
            {"placement.left",
             "placement.front",
             "placement.top",
             "placement.offset_unit",
             "placement.patch",
             "placement.patch_count"},
            config.warnings);
        warnUnknown(
            accuracy,
            "accuracy",
            {"accuracy.tolerance",
             "accuracy.near_shell",
             "accuracy.near_shell_radius",
             "accuracy.near_shell_tile"},
            config.warnings);
        for(auto const& [key, node] : document)
        {
            std::string const name(key.str());
            if(name != "strict" && name != "picongpu" && name != "domain" && name != "placement"
               && name != "accuracy")
                config.warnings.push_back("unknown configuration key '" + name + "' was ignored");
        }
        if(config.strict && !config.warnings.empty())
        {
            std::string joined;
            for(std::size_t i = 0; i < config.warnings.size(); ++i)
            {
                if(i != 0)
                    joined += "; ";
                joined += config.warnings[i];
            }
            throw std::runtime_error(
                "configuration key 'strict': unknown keys are not allowed in strict mode: "
                + joined);
        }
        return config;
    }
} // namespace picBunch
