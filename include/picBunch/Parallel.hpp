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

        [[nodiscard]] bool distributed() const
        {
            return size > 1;
        }
    };
} // namespace picBunch
