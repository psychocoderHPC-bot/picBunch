// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/Types.hpp"

namespace picBunch
{
    /** Derive the output grid that covers the bunch bounding box plus a shell margin.
     *
     * The node spacing is ``kRefinement * options.cellSize`` and ``n`` is rounded up
     * so the box covers ``[min - margin, max + margin]`` on every axis with at least
     * one extra node.  ``n`` is forced even so the R2C last extent is even regardless
     * of the padding factor.  The padding factor is taken from ``options`` and raised
     * by the tolerance heuristic below.
     */
    Grid makeGrid(Bunch const& bunch, ComputeOptions const& options);

    /** Full field pipeline: TSC deposit -> moving point kernel -> free-space FFT
     *  convolution -> Yee-staggered read-out.
     *
     * Selects the device of the compiled backend (CUDA, HIP, or host CPU), runs
     * every precision-critical stage in double precision, and returns the six
     * components in SI units (V/m and T) on ``makeGrid(bunch, options)``.
     *
     * ``tolerance`` is the only accuracy knob.  v0.1 uses it to pick the FFT
     * padding; the refinement / near-shell policy replaces that heuristic later.
     *
     * ``stageTimes``, when non-null, receives a wall-clock breakdown.  Every
     * boundary waits for the device, so the numbers are comparable across backends.
     */
    Field computeField(Bunch const& bunch, ComputeOptions const& options = {}, StageTimes* stageTimes = nullptr);

} // namespace picBunch
