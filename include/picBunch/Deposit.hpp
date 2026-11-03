// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/Types.hpp"

#include <vector>

namespace picBunch
{
    /** TSC deposit of ``weight*charge`` onto the unpadded output grid.
     *
     * Returns an ``n[0]*n[1]*n[2]`` array (x fastest) in which a single
     * particle deposits its full charge, i.e. the sum equals
     * ``sum(weight*charge)`` up to roundoff.  Nodes farther than 1.5 cells
     * from a particle receive exactly zero.  Particles outside the grid are
     * silently ignored, exactly like the device deposit.
     */
    std::vector<double> depositTSC(Bunch const& bunch, Grid const& grid);

    /** True if every particle lies inside the grid volume (node box). */
    bool allParticlesInside(Bunch const& bunch, Grid const& grid);

} // namespace picBunch
