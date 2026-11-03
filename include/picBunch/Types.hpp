// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace picBunch
{
    /** Uniform Cartesian output grid.
     *
     * ``n`` nodes per axis, node ``(i,j,k)`` sits at
     * ``origin + (i,j,k) * spacing``.  ``fftPadding`` is the zero-padding
     * factor of the free-space convolution: the FFT grid has
     * ``M = fftPadding * n`` nodes per axis.  It must be >= 2.
     */
    struct Grid
    {
        std::array<std::size_t, 3> n{48, 48, 48};
        std::array<double, 3> spacing{1.0, 1.0, 1.0}; //!< physical grid spacing
        std::array<double, 3> origin{0.0, 0.0, 0.0}; //!< physical position of node (0,0,0)
        unsigned fftPadding{2};

        [[nodiscard]] std::size_t size() const
        {
            return n[0] * n[1] * n[2];
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
     * position ``node + kYeeOffset[c]``.
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

    //! openPMD record names and component order: E then B.
    inline constexpr std::array<char const*, 6> kComponentNames{
        "E_x",
        "E_y",
        "E_z",
        "B_x",
        "B_y",
        "B_z"};

    //! Yee offsets in units of the grid spacing, same order as kComponentNames.
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

} // namespace picBunch
