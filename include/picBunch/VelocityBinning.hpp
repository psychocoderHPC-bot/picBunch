// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/Types.hpp"

namespace picBunch
{
    /** Derive the velocity binning of ``bunch``.
     *
     * A single-velocity bunch (all ``u`` equal within ``1e-12`` relative)
     * keeps the historical one-box path with the shared ``u`` as
     * representative.  Otherwise the boxes follow the policy: ``Explicit``
     * uses ``explicitBins`` verbatim; ``Auto`` takes the smallest power of two
     * whose best axis split estimates ``modelError <= tolerance``.  When no
     * count up to ``maxBins`` satisfies the model, the search continues
     * model-only and throws ``std::runtime_error`` naming the required count.
     */
    VelocityBinning deriveVelocityBinning(
        Bunch const& bunch,
        double tolerance,
        VelocityBinningOptions const& policy = {});

} // namespace picBunch
