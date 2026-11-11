// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/Config.hpp"
#include "picBunch/Types.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace picBunch
{
    //! Charge-weighted centroid ``sum(q w r) / sum(q w)`` in SI metres.
    Vec3 chargeWeightedCentroid(Bunch const& bunch);

    //! Arithmetic centre of the particle bounding box in SI metres.
    Vec3 boundingBoxCentre(Bunch const& bunch);

    struct PlacedBunch
    {
        Bunch bunch; //!< positions shifted to the requested placement
        Vec3 centroid{}; //!< charge-weighted centroid before the shift
        Vec3 boundingCentre{}; //!< bounding-box centre before the shift
        Vec3 targetCentre{}; //!< requested centre in absolute SI coordinates
        Vec3 shift{}; //!< targetCentre - centroid, SI metres
        //! Patch cell offset relative to the PIConGPU domain origin (0-based).
        IVec3 patchOffset{};
        //! Patch extent in cells; the patch covers ``[offset, offset + extent)``.
        UVec3 patchExtent{};
        //! One patch per rank-overlap region (always contains at least one).
        std::vector<ParticlePatch> patches;
    };

    /** Shift a bunch so its charge-weighted centroid lands on the requested
     * position and derive the single particle patch that contains it.
     *
     * The requested offset is measured from ``domain.origin`` along the
     * configured axes.  ``placement.offset_unit`` selects metres or a fraction
     * of the domain extent.  The function rejects a result whose bounding box
     * leaves the PIConGPU domain, because such a patch cannot be restarted.
     */
    PlacedBunch placeBunch(Bunch const& bunch, Config const& config);
} // namespace picBunch
