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
#include <string_view>
#include <vector>

namespace picBunch
{
    namespace
    {
        [[noreturn]] void fail(std::string_view key, std::string_view why)
        {
            throw std::runtime_error(
                "configuration key '" + std::string(key) + "': " + std::string(why));
        }

        toml::node const& require(toml::table const& table, std::string_view key)
        {
            auto const* node = table.get(key);
            if(node == nullptr)
                fail(key, "required key is missing");
            return *node;
        }

        toml::table const& requireTable(toml::table const& table, std::string_view key)
        {
            auto const* node = require(table, key).as_table();
            if(node == nullptr)
                fail(key, "expected a table");
            return *node;
        }

        // Optional table: returns a pointer to the table, or nullptr when the
        // key is absent (no failure).
        toml::table const* requireTableOptional(toml::table const& table, std::string_view key)
        {
            auto const* node = table.get(key);
            if(node == nullptr)
                return nullptr;
            auto const* sub = node->as_table();
            if(sub == nullptr)
                fail(key, "expected a table");
            return sub;
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

        //! A string or an array of non-empty strings, in declaration order.
        std::vector<std::string> asStringList(toml::node const& node, std::string const& key)
        {
            std::vector<std::string> values;
            if(node.is_string())
            {
                values.push_back(asString(node, key));
                return values;
            }
            auto const* array = node.as_array();
            if(array == nullptr)
                fail(key, "expected a string or an array of strings");
            values.reserve(array->size());
            for(std::size_t i = 0; i < array->size(); ++i)
                values.push_back(asString((*array)[i], key));
            return values;
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
        // ``species`` accepts the historical scalar string and an ordered
        // array.  A scalar is the one-entry list, so every existing TOML keeps
        // its meaning.
        config.picongpu.species
            = asStringList(require(picongpu, "species"), "picongpu.species");
        if(config.picongpu.species.empty())
            fail("picongpu.species", "at least one species is required");
        {
            std::set<std::string> seen;
            for(auto const& name : config.picongpu.species)
                if(!seen.insert(name).second)
                    fail(
                        "picongpu.species",
                        "duplicate species '" + name + "'; each entry is read and written once");
        }
        if(auto const* node = picongpu.get("output_species"))
        {
            config.picongpu.outputSpecies = asStringList(*node, "picongpu.output_species");
            if(config.picongpu.outputSpecies.size() != config.picongpu.species.size())
                fail(
                    "picongpu.output_species",
                    "must have one entry per picongpu.species entry ("
                        + std::to_string(config.picongpu.species.size()) + " expected, "
                        + std::to_string(config.picongpu.outputSpecies.size()) + " given)");
            std::set<std::string> seen;
            for(auto const& name : config.picongpu.outputSpecies)
                if(!seen.insert(name).second)
                    fail(
                        "picongpu.output_species",
                        "duplicate output species '" + name + "'; group names must be unique");
        }
        if(auto const* node = picongpu.get("empty_species"))
            config.picongpu.emptySpecies = asStringList(*node, "picongpu.empty_species");

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

        if(auto const* node = picongpu.get("absorber"))
        {
            std::string const absorber = asString(*node, "picongpu.absorber");
            if(absorber == "none" || absorber == "exponential" || absorber == "pml")
                config.picongpu.absorber = absorber;
            else
                fail(
                    "picongpu.absorber",
                    "expected \"none\", \"exponential\" or \"pml\"; "
                    "\"pml\" writes the convolutional PML slab records a "
                    "--fieldAbsorber pml restart deck requires");
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

        // placement.center selects the placement mode.  "centroid" is the
        // historical behaviour: position/offset_unit place the charge-weighted
        // centroid.  "none" keeps the absolute input coordinates and ignores
        // position/offset_unit, so both keys become optional.
        if(auto const* node = placement.get("center"))
        {
            std::string const centre = asString(*node, "placement.center");
            if(centre == "centroid")
                config.placement.centre = Config::Placement::Centre::Centroid;
            else if(centre == "none")
                config.placement.centre = Config::Placement::Centre::None;
            else
                fail("placement.center", "expected \"centroid\" or \"none\"");
        }
        bool const keepAbsolute
            = config.placement.centre == Config::Placement::Centre::None;
        // placement.position = [x, y, z], ordered x, y, z (PIConGPU
        // notation): the offset from the domain origin to the bunch centroid,
        // in the unit selected by placement.offset_unit.
        if(auto const* positionNode = placement.get("position"))
        {
            config.placement.offset = asTriple(*positionNode, "placement.position");
            if(keepAbsolute)
            {
                bool nonZero = false;
                for(std::size_t a = 0; a < 3; ++a)
                    if(config.placement.offset[a] != 0.0)
                        nonZero = true;
                if(nonZero)
                    config.warnings.push_back(
                        "configuration key 'placement.position' is ignored when "
                        "placement.center = \"none\"; the input coordinates are kept");
            }
        }
        else if(!keepAbsolute)
            fail("placement.position", "required key is missing");
        if(auto const* unitNode = placement.get("offset_unit"))
        {
            std::string const unit = asString(*unitNode, "placement.offset_unit");
            if(unit == "fraction")
                config.placement.fraction = true;
            else if(unit == "metres")
                config.placement.fraction = false;
            else
                fail("placement.offset_unit", "expected \"fraction\" or \"metres\"");
        }
        else if(!keepAbsolute)
            fail("placement.offset_unit", "required key is missing");
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
        // Checkpoint output backend.  The table is optional; ``backend`` defaults
        // to "auto".  Only "auto", "adios2" and "hdf5" are accepted.
        if(auto const* checkpointTable = requireTableOptional(document, "checkpoint"))
        {
            if(auto const* node = checkpointTable->get("backend"))
            {
                config.checkpoint.backend = asString(*node, "checkpoint.backend");
                std::string lower = config.checkpoint.backend;
                for(char& ch : lower)
                    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
                if(lower != "auto" && lower != "adios2" && lower != "hdf5")
                    fail(
                        "checkpoint.backend",
                        "must be \"auto\", \"adios2\" or \"hdf5\" (got \"" + config.checkpoint.backend
                            + "\")");
                config.checkpoint.backend = lower;
            }
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
        if(config.placement.fraction && config.placement.centre == Config::Placement::Centre::Centroid)
            for(std::size_t a = 0; a < 3; ++a)
                if(config.placement.offset[a] < 0.0 || config.placement.offset[a] > 1.0)
                    fail("placement.position", "fractional offsets must be in [0, 1]");
        // A name cannot be both filled and empty: the generated species is
        // read from the input, the empty one never is.
        {
            std::set<std::string> generated(config.picongpu.species.begin(), config.picongpu.species.end());
            if(!config.picongpu.outputSpecies.empty())
                generated.insert(
                    config.picongpu.outputSpecies.begin(), config.picongpu.outputSpecies.end());
            std::set<std::string> emptySeen;
            for(auto const& name : config.picongpu.emptySpecies)
            {
                if(!emptySeen.insert(name).second)
                    fail(
                        "picongpu.empty_species",
                        "duplicate species '" + name + "'");
                if(generated.count(name) != 0u)
                    fail(
                        "picongpu.empty_species",
                        "'" + name
                            + "' is also a generated species; a species is either read from "
                              "the input or written empty, never both");
            }
        }
        positive(config.accuracy.tolerance, "accuracy.tolerance");
        if(config.accuracy.nearShell != "none" && config.accuracy.nearShell != "exact"
           && config.accuracy.nearShell != "fast" && config.accuracy.nearShell != "fast-gpu")
            fail("accuracy.near_shell", "expected \"none\", \"exact\", \"fast\" or \"fast-gpu\"");
        if(config.accuracy.nearShellRadius < 0.0 || !std::isfinite(config.accuracy.nearShellRadius))
            fail("accuracy.near_shell_radius", "must be finite and non-negative");
        // Zero is valid here: it selects the automatic tile rule.  The fast
        // path additionally rejects an explicit tile that leaves the measured
        // envelope.  The tile half-width must not exceed the shell radius, so
        // the tile edge must not exceed the shell diameter (2 * radius).  The
        // validated tile 4 at the default radius 3 therefore stays valid,
        // while tile 8 at radius 3 is rejected.
        std::size_t const tile = config.accuracy.nearShellTile;
        bool const fastNearShell
            = config.accuracy.nearShell == "fast" || config.accuracy.nearShell == "fast-gpu";
        if(fastNearShell && tile != 0u)
        {
            if(config.accuracy.nearShellRadius > 0.0
               && static_cast<double>(tile) > 2.0 * config.accuracy.nearShellRadius)
                fail(
                    "accuracy.near_shell_tile",
                    "explicit fast-path tile " + std::to_string(tile)
                        + " nodes exceeds twice near_shell_radius "
                        + std::to_string(config.accuracy.nearShellRadius)
                        + " (the shell diameter, "
                        + std::to_string(2.0 * config.accuracy.nearShellRadius)
                        + " nodes); the tile half-width must not exceed the radius. Use "
                          "near_shell_tile = 0 (automatic) or raise the radius");
            if(tile > kMaxExplicitNearShellTile)
                fail(
                    "accuracy.near_shell_tile",
                    "explicit fast-path tile " + std::to_string(tile)
                        + " nodes exceeds the documented maximum "
                        + std::to_string(kMaxExplicitNearShellTile)
                        + " nodes; the validated campaigns cover tiles 2 and 4 and tile 8 "
                          "is the first measured spike case. Use near_shell_tile = 0 "
                          "(automatic)");
        }

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
             "picongpu.absorber",
             "picongpu.species",
             "picongpu.output_species",
             "picongpu.empty_species"},
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
             "placement.center",
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
               && name != "accuracy" && name != "checkpoint")
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
