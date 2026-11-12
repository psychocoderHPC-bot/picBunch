// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/Types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace picBunch
{
    //! 2022 CODATA speed of light in vacuum (m/s), as used by the deck.
    inline constexpr double kDeckSpeedOfLight = 2.99792458e8;
    //! 2022 CODATA electron rest mass (kg).
    inline constexpr double kDeckElectronMass = 9.1093837139e-31;
    //! 2022 CODATA electron charge (C, signed).
    inline constexpr double kDeckElectronCharge = -1.602176634e-19;

    /** Parsed and validated TOML configuration of the placement interface.
     *
     * The ``[picongpu]`` table carries the same independent deck inputs that
     * PIConGPU reads from ``simulation.param``, ``speciesConstants.param`` and
     * ``physicalConstants.param``.  The SI unit factors are derived exactly as
     * ``include/picongpu/unitless/simulation.unitless`` (``struct Units``), so
     * the operator repeats only the deck setup, never a transcribed factor.
     *
     * Unknown keys are collected in ``warnings`` instead of being rejected, so
     * a configuration from a newer version still runs and the operator sees
     * what was ignored.  The deprecated manual ``unit_*`` keys are accepted as
     * explicit overrides (see ``derivePicOnGPUUnits``) and warn.
     */
    struct Config
    {
        struct PicOnGPU
        {
            // Independent PIConGPU deck inputs.
            double deltaT = 0.0; //!< SI::DELTA_T_SI, seconds
            Vec3 cellSize{0.0, 0.0, 0.0}; //!< CELL_*_SI, metres
            double baseMass = kDeckElectronMass; //!< SI::BASE_MASS_SI, kg
            double baseCharge = kDeckElectronCharge; //!< SI::BASE_CHARGE_SI, C (signed)
            double baseDensity = 0.0; //!< SI::BASE_DENSITY_SI, m^-3
            double typicalParticlesPerCell = 2.0; //!< TYPICAL_PARTICLES_PER_CELL
            double speedOfLight = kDeckSpeedOfLight; //!< SI::SPEED_OF_LIGHT_SI, m/s

            // Derived exactly as PIConGPU's ``struct Units``.
            double unitTime = 0.0; //!< s
            double unitSpeed = kDeckSpeedOfLight; //!< m/s
            double unitLength = 0.0; //!< m
            double unitMass = 0.0; //!< kg
            double unitCharge = 1.602176634e-19; //!< C
            double unitEnergy = 0.0; //!< J
            double unitEField = 0.0; //!< V/m
            double unitBField = 0.0; //!< T

            double dt = 0.0; //!< stored dt, equal to ``deltaT`` (seconds)
            std::uint64_t step = 0;
            bool float64 = false;
            //! Ordered input species names read from ``--input``.  The scalar
            //! TOML form (a single string) parses to a one-entry list.
            std::vector<std::string> species;
            //! Optional explicit output group name per ``species`` entry,
            //! positionally aligned.  Empty means "same as the input name".
            std::vector<std::string> outputSpecies;
            //! Deck species that must exist in the checkpoint with zero
            //! particles.  They are never read from the input file.
            std::vector<std::string> emptySpecies;
        } picongpu;

        struct Domain
        {
            Idx3 cells{0, 0, 0};
            Vec3 cellSize{0.0, 0.0, 0.0}; //!< metres
            Vec3 origin{0.0, 0.0, 0.0}; //!< metres
        } domain;

        struct Placement
        {
            //! One particle patch per generated species.
            enum class Patch
            {
                BoundingBox, //!< cover the shifted bunch bounding box only
                Domain //!< cover the full PIConGPU domain
            };

            //! How the bunch is positioned in the domain.
            enum class Centre
            {
                Centroid, //!< shift the charge-weighted centroid to ``position``
                None //!< keep the absolute input coordinates, no shift
            };

            //! ``Centroid`` (default, historical behaviour) or ``None``.
            Centre centre = Centre::Centroid;

            //! Offset from the PIConGPU domain origin to the bunch centroid,
            //! ordered x, y, z, in metres or as a fraction of the domain
            //! extent.  Parsed from ``placement.position``; the deprecated
            //! ``left``/``front``/``top`` scalars map to x/y/z.
            Vec3 offset{0.0, 0.0, 0.0};
            bool fraction = true;
            //! ``bbox`` (default) or ``domain``.  A single domain-covering patch
            //! is the safe choice for a multi-rank restart; the bounding-box
            //! patch is validated only for single-rank restarts.
            Patch patch = Patch::BoundingBox;
            //! Number of particle patches written for the generated species.
            //! One patch per rank-overlap region; 1 keeps the single-rank layout.
            std::size_t patchCount = 1;
        } placement;

        struct Accuracy
        {
            double tolerance = 1e-3;
            std::string nearShell = "none";
            double nearShellRadius = 0.0; //!< PIC cells; 0 derives from tolerance
            //! 0 selects the automatic near-set pair-count rule.
            std::size_t nearShellTile = 0;
            int nearShellFastOrder = 5; //!< Gauss order of the fast remainder
            //! Oblique singular-box order: 0 = auto, else an allowed even order.
            int nearShellObliqueOrder = 0;
            //! Exact routing cell reach for the fast path.  -2 selects the
            //! shipped automatic policy (observer cell at the default 1e-3
            //! tolerance, all in-support cells at tolerance <= 1e-4), -1 forces
            //! legacy quadrature, 0/1/2 are the explicit reaches.
            int nearShellExactCells = -2;
        } accuracy;

        //! Reject unknown keys instead of warning about them.
        bool strict = false;
        std::vector<std::string> warnings;
    };

    /** Derive the PIConGPU SI unit system from the independent deck inputs.
     *
     * Mirrors ``include/picongpu/unitless/simulation.unitless``:
     * @code
     * n_mpp       = baseDensity * dx*dy*dz / typicalParticlesPerCell
     * unitTime    = deltaT
     * unitSpeed   = speedOfLight
     * unitLength  = unitTime * unitSpeed
     * unitMass    = baseMass * n_mpp
     * unitCharge  = -baseCharge * n_mpp
     * unitEnergy  = unitMass * unitLength^2 / unitTime^2
     * unitEField  = 1 / (unitTime^2 / unitMass / unitLength * unitCharge)
     * unitBField  = unitMass / (unitTime * unitCharge)
     * @endcode
     * The inputs must already be validated.  ``dt`` is set to ``deltaT``.
     */
    void derivePicOnGPUUnits(Config::PicOnGPU& picongpu);

    /** Parse and validate a TOML configuration file.
     *
     * @throws std::runtime_error for a malformed file, an invalid value, or a
     *         missing required key; the message names the TOML key.
     */
    Config loadConfig(std::string const& path);
} // namespace picBunch
