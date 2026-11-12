// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#include "picBunch/Config.hpp"

#include <toml++/toml.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
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

        Vec3 asTriple(toml::node const& node, std::string const& key)
        {
            auto const* array = node.as_array();
            if(array == nullptr || array->size() != 3u)
                fail(key, "expected an array of three numbers");
            Vec3 result{};
            for(std::size_t a = 0; a < 3; ++a)
                result[a] = asDouble((*array)[a], key);
            return result;
        }

        Idx3 asCells(toml::node const& node, std::string const& key)
        {
            auto const* array = node.as_array();
            if(array == nullptr || array->size() != 3u)
                fail(key, "expected an array of three positive integers");
            Idx3 result{};
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

    void derivePicOnGPUUnits(Config::PicOnGPU& picongpu)
    {
        // Exactly include/picongpu/unitless/simulation.unitless (struct Units).
        double const numParticlesPerMacroParticle
            = picongpu.baseDensity * picongpu.cellSize[0] * picongpu.cellSize[1]
              * picongpu.cellSize[2] / picongpu.typicalParticlesPerCell;
        picongpu.unitTime = picongpu.deltaT;
        picongpu.unitSpeed = picongpu.speedOfLight;
        picongpu.unitLength = picongpu.unitTime * picongpu.unitSpeed;
        picongpu.unitMass = picongpu.baseMass * numParticlesPerMacroParticle;
        picongpu.unitCharge = -1.0 * picongpu.baseCharge * numParticlesPerMacroParticle;
        picongpu.unitEnergy = picongpu.unitMass * picongpu.unitLength * picongpu.unitLength
                              / (picongpu.unitTime * picongpu.unitTime);
        picongpu.unitEField = 1.0
                              / (picongpu.unitTime * picongpu.unitTime / picongpu.unitMass
                                 / picongpu.unitLength * picongpu.unitCharge);
        picongpu.unitBField = picongpu.unitMass / (picongpu.unitTime * picongpu.unitCharge);
        picongpu.dt = picongpu.deltaT;
    }

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

        // Independent deck inputs; the SI unit factors are derived below.
        if(auto const* node = picongpu.get("delta_t"))
            config.picongpu.deltaT = asDouble(*node, "picongpu.delta_t");
        else if(auto const* node = picongpu.get("dt"))
        {
            config.picongpu.deltaT = asDouble(*node, "picongpu.dt");
            config.warnings.push_back(
                "configuration key 'picongpu.dt' is deprecated; use 'picongpu.delta_t'");
        }
        else
            fail("picongpu.delta_t", "required key is missing");

        if(auto const* node = picongpu.get("cell_size"))
            config.picongpu.cellSize = asTriple(*node, "picongpu.cell_size");
        if(auto const* node = picongpu.get("base_mass"))
            config.picongpu.baseMass = asDouble(*node, "picongpu.base_mass");
        if(auto const* node = picongpu.get("base_charge"))
            config.picongpu.baseCharge = asDouble(*node, "picongpu.base_charge");
        config.picongpu.baseDensity
            = asDouble(require(picongpu, "base_density"), "picongpu.base_density");
        if(auto const* node = picongpu.get("typical_particles_per_cell"))
        {
            config.picongpu.typicalParticlesPerCell
                = asDouble(*node, "picongpu.typical_particles_per_cell");
        }
        if(auto const* node = picongpu.get("speed_of_light"))
            config.picongpu.speedOfLight = asDouble(*node, "picongpu.speed_of_light");

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

        config.domain.cells = asCells(require(domain, "cells"), "domain.cells");
        // The physical cell size may live in [picongpu] (new schema) or in
        // [domain] (legacy).  Both are the same deck value; if both are given
        // they must agree, otherwise the unit derivation and the geometry
        // would silently describe different decks.
        bool const picCellGiven = picongpu.get("cell_size") != nullptr;
        std::optional<Vec3> domainCellSize;
        if(auto const* node = domain.get("cell_size"))
            domainCellSize = asTriple(*node, "domain.cell_size");
        if(!picCellGiven)
        {
            if(!domainCellSize)
                fail(
                    "picongpu.cell_size",
                    "required key is missing (also absent as domain.cell_size)");
            config.picongpu.cellSize = *domainCellSize;
            config.warnings.push_back(
                "configuration key 'domain.cell_size' is deprecated; use 'picongpu.cell_size'");
        }
        config.domain.cellSize = domainCellSize ? *domainCellSize : config.picongpu.cellSize;
        if(domainCellSize)
        {
            for(std::size_t a = 0; a < 3; ++a)
            {
                double const reference = config.picongpu.cellSize[a];
                double const other = (*domainCellSize)[a];
                double const scale = std::max(std::abs(reference), std::abs(other));
                if(scale > 0.0 && std::abs(reference - other) > 1e-12 * scale)
                    fail(
                        "picongpu.cell_size",
                        "must equal domain.cell_size; the unit derivation and the grid geometry "
                        "describe different decks");
            }
        }
        if(auto const* node = domain.get("origin"))
            config.domain.origin = asTriple(*node, "domain.origin");

        // placement.position = [x, y, z], ordered x, y, z (PIConGPU
        // notation): the offset from the domain origin to the bunch centroid,
        // in the unit selected by placement.offset_unit.
        if(auto const* positionNode = placement.get("position"))
            config.placement.offset = asTriple(*positionNode, "placement.position");
        else
            fail("placement.position", "required key is missing");
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
            if(tile < 0)
                fail("accuracy.near_shell_tile", "must be zero (automatic) or positive");
            config.accuracy.nearShellTile = static_cast<std::size_t>(tile);
        }
        if(auto const* node = accuracy.get("near_shell_fast_order"))
        {
            auto const order = asInteger(*node, "accuracy.near_shell_fast_order");
            if(order < 2 || order > 32)
                fail("accuracy.near_shell_fast_order", "must be between 2 and 32");
            config.accuracy.nearShellFastOrder = static_cast<int>(order);
        }
        if(auto const* node = accuracy.get("near_shell_oblique_order"))
        {
            if(auto const text = node->value<std::string>())
            {
                if(*text != "auto")
                    fail("accuracy.near_shell_oblique_order", "expected \"auto\" or an integer");
                config.accuracy.nearShellObliqueOrder = 0;
            }
            else
            {
                auto const order = asInteger(*node, "accuracy.near_shell_oblique_order");
                if(order < 4 || order > 12)
                    fail("accuracy.near_shell_oblique_order", "must be between 4 and 12");
                config.accuracy.nearShellObliqueOrder = static_cast<int>(order);
            }
        }
        if(auto const* node = accuracy.get("near_shell_exact_cells"))
        {
            auto const cells = asInteger(*node, "accuracy.near_shell_exact_cells");
            if(cells < -2 || cells > 2)
                fail("accuracy.near_shell_exact_cells", "must be between -2 and 2");
            config.accuracy.nearShellExactCells = static_cast<int>(cells);
        }

        // Validation.
        auto positive = [](double value, std::string const& key) {
            if(!(value > 0.0) || !std::isfinite(value))
                fail(key, "must be finite and positive");
        };
        // Validate the independent deck inputs, then derive the unit system.
        positive(config.picongpu.deltaT, "picongpu.delta_t");
        for(std::size_t a = 0; a < 3; ++a)
            positive(config.picongpu.cellSize[a], "picongpu.cell_size");
        positive(config.picongpu.baseMass, "picongpu.base_mass");
        if(!std::isfinite(config.picongpu.baseCharge) || config.picongpu.baseCharge == 0.0)
            fail("picongpu.base_charge", "must be finite and non-zero");
        positive(config.picongpu.baseDensity, "picongpu.base_density");
        positive(config.picongpu.typicalParticlesPerCell, "picongpu.typical_particles_per_cell");
        positive(config.picongpu.speedOfLight, "picongpu.speed_of_light");
        derivePicOnGPUUnits(config.picongpu);

        // Deprecated manual unit keys are explicit overrides: they win over the
        // derived value for that factor only and always warn.  The default path
        // is derivation, so a config without them is fully determined by the
        // deck inputs above.
        auto overrideUnit = [&](char const* key, double& target) {
            if(auto const* node = picongpu.get(key))
            {
                target = asDouble(*node, std::string("picongpu.") + key);
                config.warnings.push_back(
                    std::string("configuration key 'picongpu.") + key
                    + "' is deprecated and overrides the derived unit; remove it so the "
                      "value is derived from the deck inputs");
            }
        };
        overrideUnit("unit_length", config.picongpu.unitLength);
        overrideUnit("unit_e", config.picongpu.unitEField);
        overrideUnit("unit_b", config.picongpu.unitBField);
        overrideUnit("unit_charge", config.picongpu.unitCharge);
        overrideUnit("unit_mass", config.picongpu.unitMass);
        overrideUnit("unit_speed", config.picongpu.unitSpeed);

        positive(config.picongpu.unitLength, "picongpu.unit_length");
        positive(config.picongpu.unitEField, "picongpu.unit_e");
        positive(config.picongpu.unitBField, "picongpu.unit_b");
        positive(config.picongpu.unitCharge, "picongpu.unit_charge");
        positive(config.picongpu.unitMass, "picongpu.unit_mass");
        positive(config.picongpu.unitSpeed, "picongpu.unit_speed");
        positive(config.picongpu.unitTime, "picongpu.delta_t");
        if(!(config.picongpu.unitEnergy > 0.0) || !std::isfinite(config.picongpu.unitEnergy))
            fail("picongpu.unit_energy", "derived energy unit must be finite and positive");
        config.picongpu.dt = config.picongpu.deltaT;
        if(step < 0 || step > 999999)
            fail("picongpu.step", "must be in [0, 999999]");
        config.picongpu.step = static_cast<std::uint64_t>(step);
        for(std::size_t a = 0; a < 3; ++a)
        {
            positive(config.domain.cellSize[a], "domain.cell_size");
            if(!std::isfinite(config.domain.origin[a]))
                fail("domain.origin", "must be finite");
            if(!std::isfinite(config.placement.offset[a]))
                fail("placement.position", "must be finite");
        }
        if(config.placement.fraction)
            for(std::size_t a = 0; a < 3; ++a)
                if(config.placement.offset[a] < 0.0 || config.placement.offset[a] > 1.0)
                    fail("placement.position", "fractional offsets must be in [0, 1]");
        positive(config.accuracy.tolerance, "accuracy.tolerance");
        if(config.accuracy.nearShell != "none" && config.accuracy.nearShell != "exact"
           && config.accuracy.nearShell != "fast" && config.accuracy.nearShell != "fast-gpu")
            fail("accuracy.near_shell", "expected \"none\", \"exact\", \"fast\" or \"fast-gpu\"");
        if(config.accuracy.nearShellRadius < 0.0 || !std::isfinite(config.accuracy.nearShellRadius))
            fail("accuracy.near_shell_radius", "must be finite and non-negative");
        // Zero is valid here: it selects the automatic tile rule.

        // Unknown keys are warnings.  The list is deliberately explicit so a
        // typo is visible instead of silently changing nothing.
        warnUnknown(
            picongpu,
            "picongpu",
            {"picongpu.delta_t",
             "picongpu.cell_size",
             "picongpu.base_mass",
             "picongpu.base_charge",
             "picongpu.base_density",
             "picongpu.typical_particles_per_cell",
             "picongpu.speed_of_light",
             "picongpu.dt",
             "picongpu.unit_length",
             "picongpu.unit_e",
             "picongpu.unit_b",
             "picongpu.unit_charge",
             "picongpu.unit_mass",
             "picongpu.unit_speed",
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
            {"placement.position",
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
             "accuracy.near_shell_tile",
             "accuracy.near_shell_fast_order",
             "accuracy.near_shell_oblique_order",
             "accuracy.near_shell_exact_cells"},
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
