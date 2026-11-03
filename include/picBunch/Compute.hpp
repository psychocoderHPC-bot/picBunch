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
     * of the padding factor.  The padding factor comes from the tolerance policy
     * (padding 2 for ``tolerance >= 1e-12``, padding 3 below that down to the
     * measured double-precision floor ``1e-14``) unless ``options.fftPadding`` is
     * set explicitly as a diagnostic override.  A tolerance below the floor is
     * rejected with ``std::runtime_error``.
     */
    Grid makeGrid(Bunch const& bunch, ComputeOptions const& options);

    /** Full field pipeline: TSC deposit -> moving point kernel -> free-space FFT
     *  convolution -> Yee-staggered read-out.
     *
     * Selects the device of the compiled backend (CUDA, HIP, or host CPU), runs
     * every precision-critical stage in double precision, and returns the six
     * components in SI units (V/m and T) on ``makeGrid(bunch, options)``.
     *
     * ``tolerance`` is the only accuracy knob exposed to users.  It selects the
     * FFT zero padding, the only discretisation choice not fixed by the input
     * geometry.  It does not change the TSC deposit order or ``kRefinement``, so
     * near-source accuracy stays a stated limitation of v0.1.  The contract is
     * written down in ``docs/algorithm.md``.
     *
     * ``stageTimes``, when non-null, receives a wall-clock breakdown.  Every
     * boundary waits for the device, so the numbers are comparable across backends.
     */
    Field computeField(Bunch const& bunch, ComputeOptions const& options = {}, StageTimes* stageTimes = nullptr);

} // namespace picBunch
