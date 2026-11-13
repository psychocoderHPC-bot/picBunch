// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Post-condition of the field solve: a bunch that carries net charge always has
// a non-zero mesh field.
//
// The TSC deposit spreads every source over its 3x3x3 support and the free-space
// kernel does not vanish on the padded mesh, so an identically zero field can
// only mean that the source projection was skipped, dropped every particle, or
// wrote outside the buffer the FFT consumes.  That is a defect in the solve,
// never a physical result, and it must not reach a checkpoint or a user.
//
// A neutral bunch (total charge exactly zero) is the one legitimate zero field
// and is never reported.

#pragma once

#include "picBunch/Parallel.hpp"
#include "picBunch/Types.hpp"

#include <string>

namespace picBunch::detail
{
    /*! Net charge of the bunch, ``sum_p weight[p] * charge[p]``.
     *
     * In double, in the same order as the deposit uses the sources.
     */
    [[nodiscard]] double totalCharge(Bunch const& bunch);

    //! True when every value of every component is exactly zero.
    [[nodiscard]] bool fieldIsIdenticallyZero(Field const& field);

    /*! Throw when ``field`` is identically zero although ``charge`` is not zero.
     *
     * ``configuration`` names the run (near-shell mode, shape, grid, particle
     * count, velocity boxes, FFT padding, replicated/distributed) and is carried
     * into the message verbatim so the failing setup can be reproduced from the
     * error alone.  Returns without throwing for a neutral bunch and for a
     * non-zero field.
     */
    void abortOnEmptyChargedField(
        Field const& field,
        double charge,
        std::string const& configuration);

    /*! Distributed variant: one rank's z-slab may legitimately be zero.
     *
     * "This rank holds a non-zero value" is reduced over all ranks, so only a
     * globally all-zero field is reported and every rank throws the same text.
     */
    void abortOnEmptyChargedDistributedField(
        Field const& field,
        double charge,
        std::string const& configuration,
        ParallelContext const& par);
} // namespace picBunch::detail
