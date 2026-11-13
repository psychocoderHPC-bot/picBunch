// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/Types.hpp"

#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace picBunch
{
    namespace detail
    {
        //! True when every stored component value is exactly zero.
        inline bool fieldIsIdenticallyZero(Field const& field)
        {
            for(int c = 0; c < 6; ++c)
                for(double const value : field.data[static_cast<std::size_t>(c)])
                    if(value != 0.0)
                        return false;
            return true;
        }

        /*! Abort when a charged bunch produced an identically zero field.
         *
         * The mesh field of a bunch with non-zero total charge is never
         * identically zero: the point kernel is non-vanishing on its support
         * and every source writes its charge onto that support, in every
         * near-shell mode and for every velocity-box count.  An all-zero field
         * therefore means the source projection was skipped or silently
         * produced no charge, not that the bunch has no field.  A bunch with
         * exactly zero total charge is the only legitimate zero field, so the
         * guard returns without a verdict for it and never rejects a genuinely
         * neutral input.
         *
         * ``totalCharge`` is the charge sum of the full bunch (weight times
         * charge), and ``context`` names the configuration so a failure is
         * diagnosable instead of a silently written zero field.
         */
        inline void enforceNonZeroField(
            Field const& field,
            double totalCharge,
            std::string const& context)
        {
            if(totalCharge == 0.0)
                return;
            if(!fieldIsIdenticallyZero(field))
                return;

            std::ostringstream message;
            message << "computeField (" << context << "): the field is identically zero "
                       "although the bunch carries a non-zero total charge ("
                    << std::scientific << std::setprecision(6) << totalCharge
                    << " C); the source projection was skipped or produced no charge. "
                       "This is never a valid field; aborting.";
            throw std::runtime_error(message.str());
        }
    } // namespace detail
} // namespace picBunch
