// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace picBunch
{
    /** Parsed and validated TOML configuration of the placement interface.
     *
     * Every physical value is explicitly supplied by the operator.  The parser
     * never guesses a unit or a conversion factor.  Unknown keys are collected
     * in ``warnings`` instead of being rejected, so a configuration from a
     * newer version still runs and the operator sees what was ignored.
     */
    struct Config
    {
        struct PicOnGPU
        {
            double unitLength = 0.0; //!< metres
            double unitEField = 0.0; //!< V/m
            double unitBField = 0.0; //!< T
            double unitCharge = 1.602176634e-19; //!< C, optional
            double unitMass = 9.1093837015e-31; //!< kg, optional
            double unitSpeed = 299792458.0; //!< m/s, optional
            double dt = 0.0; //!< stored dt in ``unit_length``/``unit_speed`` time units
            std::uint64_t step = 0;
            bool float64 = false;
            std::string species;
        } picongpu;

        struct Domain
        {
            std::array<std::size_t, 3> cells{0, 0, 0};
            std::array<double, 3> cellSize{0.0, 0.0, 0.0}; //!< metres
            std::array<double, 3> origin{0.0, 0.0, 0.0}; //!< metres
        } domain;

        struct Placement
        {
            //! Offsets from the PIConGPU domain origin along +x (left), +y
            //! (front), +z (top); metres or a fraction of the domain extent.
            std::array<double, 3> offset{0.0, 0.0, 0.0};
            bool fraction = true;
        } placement;

        struct Accuracy
        {
            double tolerance = 1e-3;
            std::string nearShell = "none";
            double nearShellRadius = 0.0; //!< PIC cells; 0 derives from tolerance
            std::size_t nearShellTile = 4;
        } accuracy;

        std::vector<std::string> warnings;
    };

    /** Parse and validate a TOML configuration file.
     *
     * @throws std::runtime_error for a malformed file, an invalid value, or a
     *         missing required key; the message names the TOML key.
     */
    Config loadConfig(std::string const& path);
} // namespace picBunch
