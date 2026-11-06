// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/Types.hpp"

#include <cstddef>
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

    /** Host TSC deposit onto the padded ``Mx*My*Mz`` box with a source halo.
     *
     * Same recipe as the device deposit used by ``computeField``: the TSC
     * support of a particle whose centre lies in the physical node box
     * ``[0, n)`` is kept whole.  Logical indices outside ``0..n-1`` wrap
     * circularly into the padded extent ``M = fftPadding*n``: negative support
     * lands at ``M-1, M-2, ...`` and support beyond ``n-1`` lands in the upper
     * padding ``n, n+1, ...``.  The physical grid ``n`` and the FFT layout are
     * unchanged.  ``fftPadding`` must be at least 2 and every axis must satisfy
     * ``M >= n + 2`` so the two halo sides cannot alias onto each other.
     *
     * The TSC shape is sampled in node units, so ``refinement`` does not scale
     * the weights: ``grid.spacing`` already contains the number of nodes per PIC
     * cell (``kRefinement``).  The argument is validated and kept for
     * API/testing compatibility.  Kept on the host so unit tests can compare the
     * accelerator deposit node for node.
     */
    std::vector<double> depositTSCPaddedHost(Bunch const& bunch, Grid const& grid, double refinement);

    /** True if every particle lies inside the grid volume (node box). */
    bool allParticlesInside(Bunch const& bunch, Grid const& grid);

} // namespace picBunch
