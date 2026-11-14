// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include <cstddef>
#include <cstdint>

namespace picBunch
{
    /*! Host-side collective hook for the rank-parallel stages.
     *
     * The compute library stays free of MPI headers: the application supplies a
     * concrete context backed by MPI, tests and single-process runs leave the
     * pointer null.  Only host code calls these methods; no device kernel sees
     * this type.
     */
    struct ParallelContext
    {
        int rank = 0;
        int size = 1;

        virtual ~ParallelContext() = default;

        //! In-place sum over all ranks of ``count`` doubles.
        virtual void allreduceSum(double* /*values*/, std::size_t /*count*/) const {}
        //! In-place sum over all ranks of one unsigned 64-bit value.
        virtual void allreduceSum(std::uint64_t& /*value*/) const {}
        //! In-place element-wise maximum over all ranks of ``count`` doubles.
        virtual void allreduceMax(double* /*values*/, std::size_t /*count*/) const {}

        /*! All-to-all exchange of double values with per-rank counts and displacements.
         *
         * Counts and displacements are in doubles, not bytes.  The distributed
         * FFT uses this for its slab/pencil transposes.  A single-rank context
         * copies the local block so callers need no special case.
         */
        virtual void alltoallv(
            double* /*send*/,
            int const* /*sendCounts*/,
            int const* /*sendDispls*/,
            double* /*recv*/,
            int const* /*recvCounts*/,
            int const* /*recvDispls*/) const
        {
        }

        /*! All-gather variable-sized double blocks so that every rank holds the
         * concatenation of every rank's ``send`` block.
         *
         * ``recv`` must be sized to the sum of all ``recvCounts``.  The default
         * (single-rank) context copies the local block.  The MPI backend maps
         * this to ``MPI_Allgatherv``.
         */
        virtual void allgatherv(
            double const* /*send*/,
            int /*sendCount*/,
            double* /*recv*/,
            int const* /*recvCounts*/,
            int const* /*recvDispls*/) const
        {
        }

        [[nodiscard]] bool distributed() const
        {
            return size > 1;
        }
    };
} // namespace picBunch
