// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/Types.hpp"

#include <array>
#include <vector>

namespace picBunch
{
    /** Asymptotic moving-multipole far field of a compact bunch.
     *
     * The charge and the bounding-box centre are lab-frame quantities.  The
     * dipole and traceless quadrupole ``D_ij = sum q (3 x_i x_j - |x|^2
     * delta_ij)`` are accumulated in the bunch rest frame (longitudinal
     * separations stretched by ``gamma``), because for a rigidly moving
     * distribution the field is the Lorentz transform of the rest-frame static
     * field.  The monopole is the exact Lienard point field of the total charge
     * at the mean velocity; the rest-frame dipole and quadrupole are evaluated
     * at the Lorentz-stretched observation point and their field is boosted
     * back (parallel component unchanged, perpendicular component times
     * ``gamma``).  This is an asymptotic moving multipole, not an exact Lienard
     * multipole sum, so it is exact only in the limit of a point source (it
     * reproduces ``pointFieldNormalized`` for one particle in every direction).
     * For an extended bunch the error grows with the bunch size over the
     * observation distance; the component transverse to the motion of a bunch
     * with extent along the beam axis is the least accurate (the beamed field
     * is very sensitive to the source position along the motion).  For a
     * stationary bunch (``meanVelocity = 0``) it reduces exactly to the
     * static monopole + dipole + quadrupole field.  The magnetic field uses the
     * mean velocity, ``B = v x E / c^2``.  This is the tail used where the FFT
     * mesh is not computed, i.e. outside the exact compute box of a global
     * output domain.  It replaces the former exact-zero fill.
     *
     * ``switchDistance`` is the radius beyond which the expansion has been
     * measured to stay below ``tolerance * fieldPeak`` for the supported bunch
     * shapes.  It is a measured rule, not a proof: see README and
     * tests/test_margin.cpp.
     */
    struct FarTail
    {
        std::array<double, 3> centre{0.0, 0.0, 0.0};
        double charge = 0.0;
        std::array<double, 3> dipole{0.0, 0.0, 0.0};
        std::array<double, 9> quadrupole{};
        std::array<double, 3> meanVelocity{0.0, 0.0, 0.0};
        double supportRadius = 0.0;
        double switchDistance = 0.0;
        double gammaMax = 1.0;
        //! Optional spatial groups.  When non-empty, evaluation sums the group
        //! multipoles instead of the single global expansion.  Each group is a
        //! FarTail about its own centre and has no nested groups.
        std::vector<FarTail> groups;
    };

    /** Moments and expansion switch radius of a bunch.
     *
     * ``switchFactor`` multiplies the support radius ``R`` (largest distance
     * from the bounding-box centre).  The default 6 is the measured safe switch
     * for the 20-cell, 4096-particle recipe.
     */
    FarTail deriveFarTail(
        Bunch const& bunch,
        double switchFactor = 6.0,
        std::array<double, 3> const& cellSize = {0.0, 0.0, 0.0},
        int groupCells = 0);

    //! Static multipole electric field of the tail at a physical position.
    std::array<double, 3> evaluateFarTailE(
        FarTail const& tail,
        std::array<double, 3> const& position);

    //! Uniform-velocity magnetic field of the tail, ``v x E / c^2``.
    std::array<double, 3> evaluateFarTailB(
        FarTail const& tail,
        std::array<double, 3> const& position);

} // namespace picBunch
