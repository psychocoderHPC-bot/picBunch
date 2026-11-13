// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/Types.hpp"

namespace picBunch
{
    /*! Largest charge-weighted relative spread of ``|u|`` that still counts as
     *  one velocity class for the one-box auto fallback.
     *
     * A bunch with this little momentum-magnitude spread and the angular limit
     * below is a cold, nearly monoenergetic beam: its particles share one
     * velocity to the accuracy the binned model can meaningfully resolve, so
     * one box is the physical model even though the ``gamma * Delta_theta``
     * worst-case term asks for millions of boxes.
     */
    inline constexpr double kSingleVelocityRelativeSpread = 2.0e-2;

    /*! Largest particle angle (rad) to the charge-weighted mean direction that
     *  still counts as one velocity class for the one-box auto fallback.
     *
     * 50 mrad (about 2.9 degrees) admits the 1-degree-divergence beams that hit
     * the cap while rejecting the 5-degree angular calibration case.
     */
    inline constexpr double kSingleVelocityMaxAngle = 5.0e-2;

    /** Derive the velocity binning of ``bunch``.
     *
     * A single-velocity bunch (all ``u`` equal within ``1e-12`` relative)
     * keeps the historical one-box path with the shared ``u`` as
     * representative.  Otherwise the boxes follow the policy: ``Explicit``
     * uses ``explicitBins`` verbatim; ``Auto`` takes the smallest power of two
     * whose best axis split estimates ``modelError <= tolerance``.  When no
     * count up to ``maxBins`` satisfies the model, an ``Auto`` bunch under the
     * default cap still uses one box if it is essentially single-velocity
     * (charge-weighted relative ``|u|`` spread at most
     * ``kSingleVelocityRelativeSpread`` and every particle within
     * ``kSingleVelocityMaxAngle`` of the charge-weighted mean direction), and
     * the returned ``modelError`` honestly reports the one-box estimate.  An
     * explicit ``--velocity-bins`` keeps its fixed count; a non-default
     * ``--max-velocity-bins`` disables the fallback, so the search continues
     * model-only and throws ``std::runtime_error`` naming the required count.
     */
    VelocityBinning deriveVelocityBinning(
        Bunch const& bunch,
        double tolerance,
        VelocityBinningOptions const& policy = {});

} // namespace picBunch
