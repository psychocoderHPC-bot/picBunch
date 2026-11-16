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
     * circularly into the padded extent ``M = paddedSize()``: negative support
     * lands at ``M-1, M-2, ...`` and support beyond ``n-1`` lands in the upper
     * halo ``n, n+1, ...``.  The physical grid ``n`` and the FFT layout are
     * unchanged.  ``validateDepositPadding`` compares the halo nodes actually
     * available (``M - n``, from the FFT padding factor plus ``depositHalo``)
     * with the stencil requirement; a particle whose support would leave that
     * halo is rejected instead of wrapped onto the opposite side.
     *
     * The TSC shape is sampled in node units: ``grid.spacing`` already contains
     * the number of nodes per PIC cell (``kRefinement``), so no additional
     * refinement factor enters the weights.  Kept on the host so unit tests can
     * compare the accelerator deposit node for node.
     */
    std::vector<double> depositTSCPaddedHost(Bunch const& bunch, Grid const& grid);

    /** Number of padding nodes per axis the deposit stencil requires.
     *
     * A particle inside the physical node box reaches at most
     * ``floor(shapeSupport)`` nodes past either end.  The padded region above
     * ``n`` serves both halos: negative support maps in from its top,
     * positive support from its bottom.  ``M - n`` must therefore be at least
     * twice the reach: 2 for CIC/TSC and 4 for PQS.
     */
    std::size_t requiredDepositPadding(Shape shape);

    /** Refuse a grid whose padded extent cannot hold the deposit halo.
     *
     * The check uses ``M = grid.paddedSize()`` and the actual halo
     * ``M - n`` on every axis, whether that halo comes from the FFT padding
     * factor, an explicit ``Grid::depositHalo``, or both.  It must satisfy
     * ``M - n >= requiredDepositPadding(shape)``.  Otherwise the index
     * wrap that maps negative support into the upper halo would fold support
     * that belongs in the halo onto the opposite side of the grid.  Throws
     * ``std::invalid_argument`` naming the axis, the required halo nodes and
     * the halo nodes actually available.
     */
    void validateDepositPadding(Grid const& grid);

    /** True if every particle lies inside the grid volume (node box). */
    bool allParticlesInside(Bunch const& bunch, Grid const& grid);

} // namespace picBunch
