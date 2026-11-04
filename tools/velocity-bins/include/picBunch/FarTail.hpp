// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/Types.hpp"

#include <array>

namespace picBunch
{
    /** Monopole + dipole + quadrupole far field of a compact bunch.
     *
     * The moments are taken about the bounding-box centre in the lab frame.
     * ``quadrupole`` is the traceless tensor
     * ``D_ij = sum q (3 x_i x_j - |x|^2 delta_ij)`` (row-major).  The electric
     * field is the static multipole field multiplied by ``gammaMax`` for the
     * boosted transverse component; the magnetic field uses the charge-weighted
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
    };

    /** Moments and expansion switch radius of a bunch.
     *
     * ``switchFactor`` multiplies the support radius ``R`` (largest distance
     * from the bounding-box centre).  The default 6 is the measured safe switch
     * for the 20-cell, 4096-particle recipe.
     */
    FarTail deriveFarTail(Bunch const& bunch, double switchFactor = 6.0);

    //! Static multipole electric field of the tail at a physical position.
    std::array<double, 3> evaluateFarTailE(
        FarTail const& tail,
        std::array<double, 3> const& position);

    //! Uniform-velocity magnetic field of the tail, ``v x E / c^2``.
    std::array<double, 3> evaluateFarTailB(
        FarTail const& tail,
        std::array<double, 3> const& position);

} // namespace picBunch
