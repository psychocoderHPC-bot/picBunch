// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace picBunch
{
    //! Number of nodes per physical PIC cell along one axis (``h`` in the algorithm spec).
    //!
    //! v0.1 keeps the mesh at cell resolution.  The tolerance-driven refinement policy is a
    //! later change; all code reads this constant from one place.
    inline constexpr double kRefinement = 1.0;
    //! Interpolation order of the assignment / read-out shape. 2 = quadratic B-spline (TSC).
    inline constexpr int kInterpolationOrder = 2;

    /** Uniform Cartesian output grid.
     *
     * ``n`` nodes per axis, node ``(i,j,k)`` sits at
     * ``origin + (i,j,k) * spacing``.  ``spacing`` is the physical node spacing, i.e.
     * ``kRefinement`` PIC cells.  ``fftPadding`` is the zero-padding factor of the
     * free-space convolution: the FFT grid has ``M = fftPadding * n`` nodes per axis.
     * It is normally selected from ``ComputeOptions::tolerance`` (never below 2);
     * a value of 1 disables zero padding and is a test/diagnostic override only.
     */
    struct Grid
    {
        std::array<std::size_t, 3> n{48, 48, 48};
        std::array<double, 3> spacing{1.0, 1.0, 1.0}; //!< physical node spacing
        std::array<double, 3> origin{0.0, 0.0, 0.0}; //!< physical position of node (0,0,0)
        unsigned fftPadding{2};

        //! Total number of output nodes.
        [[nodiscard]] std::size_t size() const
        {
            return n[0] * n[1] * n[2];
        }
        //! Number of nodes per axis of the zero-padded FFT box.
        [[nodiscard]] std::array<std::size_t, 3> paddedSize() const
        {
            return {n[0] * fftPadding, n[1] * fftPadding, n[2] * fftPadding};
        }
    };

    /** Macro-particle bunch, SI units for positions and charge.
     *
     * ``ux/uy/uz`` are dimensionless momenta ``p/(m c)``.  ``charge`` is the
     * charge of one macro particle in Coulomb, ``weight`` its statistical
     * weight.  The source weight deposited on the grid is ``weight*charge``.
     */
    struct Bunch
    {
        std::vector<double> x, y, z;
        std::vector<double> ux, uy, uz;
        std::vector<double> weight;
        std::vector<double> charge;

        [[nodiscard]] std::size_t size() const
        {
            return x.size();
        }
        void resize(std::size_t n);

        /** min/max corner of the particle positions. */
        [[nodiscard]] std::array<std::array<double, 3>, 2> bounds() const;
    };

    /** Six staggered field components on one grid.
     *
     * ``data[c]`` is stored with x fastest, then y, then z, on the *output*
     * grid (``grid.n``).  The field of component ``c`` lives at the staggered
     * position ``node + kYeeOffset[c]`` (in PIC-cell units).
     */
    struct Field
    {
        Grid grid;
        std::array<std::vector<double>, 6> data;
    };

    //! Wall-clock time per pipeline stage, milliseconds.
    struct StageTimes
    {
        double readerMs = 0.0;
        double depositMs = 0.0;
        double kernelMs = 0.0;
        double fftMs = 0.0;
        double readoutMs = 0.0;
        double writerMs = 0.0;
    };

    /** User-facing knobs of the field computation.
     *
     * The user sets an accuracy target, the physical cell size, and the empty
     * shell margin.  Bin counts, interpolation orders and FFT sizes are never
     * user input: ``tolerance`` selects the only tunable discretisation choice,
     * the FFT zero padding, and the remaining values are constants of the
     * implementation.
     */
    struct ComputeOptions
    {
        /*! Relative accuracy target of the delivered free-space convolution.
         *
         * The target refers to the discrete free-space TSC convolution, not to
         * the continuum Coulomb field near the source.  It selects the FFT
         * padding (see ``makeGrid``): padding 2 for targets that are at or above
         * the measured padding-2 error bound, padding 3 for tighter targets.
         * A target below the measured double-precision floor is rejected.
         */
        double tolerance = 1e-3;
        //! Physical PIC cell size (m). Node spacing is ``kRefinement * cellSize``.
        std::array<double, 3> cellSize{1.0, 1.0, 1.0};
        /*! Diagnostic override of the FFT zero-padding factor; 0 selects the
         * tolerance policy.  A non-zero value (including 1 for the wraparound
         * test) bypasses the policy and must be >= 1.
         */
        unsigned fftPadding = 0;
        //! Empty shell margin around the particle bounding box, in PIC cells.
        double marginCells = 4.0;
        /*! Explicit output-grid overrides (metres / node counts).

         * Unset values keep the automatic derivation from the bunch bounding box.
         * When set they are used verbatim: an explicit ``--grid`` or ``--origin``
         * never blends with the derived value.  The CLI supplies ``gridOrigin``
         * from ``--origin`` when given, otherwise from the input mesh
         * ``gridGlobalOffset``.  Grid node counts must be even and positive so the
         * R2C transform keeps its even last extent.
         */
        std::optional<std::array<std::size_t, 3>> gridSize;
        std::optional<std::array<double, 3>> gridOrigin;
    };

    //! openPMD record names and component order: E then B.
    inline constexpr std::array<char const*, 6> kComponentNames{
        "E_x",
        "E_y",
        "E_z",
        "B_x",
        "B_y",
        "B_z"};

    //! Yee offsets in units of the PIC cell, same order as kComponentNames.
    inline constexpr double kYeeOffset[6][3] = {
        {0.5, 0.0, 0.0},
        {0.0, 0.5, 0.0},
        {0.0, 0.0, 0.5},
        {0.0, 0.5, 0.5},
        {0.5, 0.0, 0.5},
        {0.5, 0.5, 0.0}};

    //! 1/(4 pi eps0) in N m^2/C^2.
    inline constexpr double kCoulombConstant = 8.9875517873681764e9;
    //! speed of light in m/s.
    inline constexpr double kSpeedOfLight = 299792458.0;
    //! elementary charge in C (magnitude).
    inline constexpr double kElementaryCharge = 1.602176634e-19;
    //! electron rest mass in kg.
    inline constexpr double kElectronMass = 9.1093837015e-31;

    /** Geometry and time metadata of an openPMD mesh iteration.
     *
     * All physical vectors are returned in ``x, y, z`` order regardless of the
     * axis-label order stored in the file.
     */
    struct MeshGeometry
    {
        std::array<std::size_t, 3> shape{0, 0, 0};       //!< nodes per x, y, z
        std::array<double, 3> gridSpacing{1.0, 1.0, 1.0}; //!< physical spacing per axis
        std::array<double, 3> gridGlobalOffset{0.0, 0.0, 0.0}; //!< physical offset per axis
        std::array<double, 3> gridUnitSI{1.0, 1.0, 1.0}; //!< raw gridUnitSI per axis
        std::array<std::string, 3> axisLabels{"x", "y", "z"};
        std::string geometry{"cartesian"};
        std::string dataOrder{"C"};
        double time = 0.0;
        double timeUnitSI = 1.0;
        std::uint64_t iteration = 0;
    };

    //! One particle iteration read from an openPMD series.
    struct BunchData
    {
        Bunch bunch;
        double time = 0.0;
        double timeUnitSI = 1.0;
        std::uint64_t iteration = 0;
    };

} // namespace picBunch
