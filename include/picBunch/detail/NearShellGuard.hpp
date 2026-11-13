// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/Types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace picBunch
{
    namespace detail
    {
        /*! Largest accepted growth of the peak field caused by the fast
         * near-shell correction.
         *
         * The stable near-shell configurations (fast, R = 3, tile 2 or 4) keep
         * the peak at the uncorrected value to within a few percent.  The
         * measured failure modes are 13x (R = 6, tile 4), 35x (tile 8, R = 3)
         * and about 17000x (tile 16).  A factor of 3 is therefore far below the
         * smallest measured failure and far above the stable growth.
         */
        inline constexpr double kNearShellPeakGrowthLimit = 3.0;

        //! Peak absolute value of the electric (c < 3) and magnetic components.
        struct NearShellPeak
        {
            double e = 0.0;
            double b = 0.0;
        };

        inline NearShellPeak nearShellPeak(Field const& field)
        {
            NearShellPeak peak;
            for(int c = 0; c < 6; ++c)
            {
                double& target = (c < 3) ? peak.e : peak.b;
                for(double const value : field.data[static_cast<std::size_t>(c)])
                    target = std::max(target, std::abs(value));
            }
            return peak;
        }

        /*! Abort when the corrected electric peak exceeds the uncorrected
         * electric peak by more than ``factor``.
         *
         * The measured failure modes are electric-field spikes (peak |Ex|), so
         * the post-condition tracks the electric peak.  B is derived from the
         * same correction and follows E/c for a physical field; comparing a
         * near-zero B baseline is numerically ill-conditioned and would flag
         * harmless round-off.  A zero uncorrected peak is not a violation: it
         * carries no scale.  The message names the radius, tile and order so a
         * spike is a diagnosable failure instead of a silently written field.
         */
        inline void enforceNearShellPeak(
            NearShellPeak const& uncorrected,
            NearShellPeak const& corrected,
            double radiusCells,
            std::size_t tileNodes,
            int fastOrder,
            double factor = kNearShellPeakGrowthLimit)
        {
            double const base = uncorrected.e;
            double const value = corrected.e;
            if(!(base > 0.0) || !(value > factor * base))
                return;

            std::ostringstream message;
            message << "near shell: the fast correction raised the peak |E| from "
                    << std::scientific << std::setprecision(3) << base << " to " << value
                    << " V/m (" << std::fixed << std::setprecision(1) << value / base
                    << "x, limit " << factor << "x) at radius=" << radiusCells
                    << " cells, tile=" << tileNodes << " nodes, fast_order=" << fastOrder
                    << "; the near-shell boundary treatment is under investigation. "
                       "Refusing to write an unphysical field. Use near_shell_tile = 0 "
                       "(automatic) or a smaller radius.";
            throw std::runtime_error(message.str());
        }
    } // namespace detail
} // namespace picBunch
