// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/Types.hpp"

#include <cstddef>

namespace picBunch
{
    struct VelocityMomentsReport;
    /*! Total field with the exact near shell applied.
     *
     * Computes the ordinary padded-FFT mesh field first, then applies the
     * per-tile subtraction/addition described in ``NearShellOptions``.  The
     * returned field has the same grid and units (SI V/m and T) as
     * ``computeField``.
     */
    Field computeFieldNearExact(
        Bunch const& bunch,
        ComputeOptions const& options,
        NearShellOptions const& nearShell,
        StageTimes* stageTimes = nullptr,
        VelocityBinning* outBinning = nullptr,
        VelocityMomentsReport* outReport = nullptr);

    /*! Resolve the shell radius policy (exported for tests and reports). */
    double resolveNearShellRadius(ComputeOptions const& options, NearShellOptions const& nearShell);

} // namespace picBunch
