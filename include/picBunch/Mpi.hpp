// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/Parallel.hpp"
#include "picBunch/Types.hpp"

#include <cstddef>
#include <string>

namespace picBunch
{
    /*! Optional MPI runtime.
     *
     * Built only with ``PICBUNCH_ENABLE_MPI``.  Without that option every
     * function below behaves as a single rank: ``mpiRank()`` is 0,
     * ``mpiSize()`` is 1, broadcasts and barriers are no-ops.  The field
     * computation always runs on rank 0; the other ranks receive the computed
     * bunch-box field by broadcast and only participate in the chunked write.
     */

    //! Initialise MPI once for the process.  No-op when MPI is not compiled in.
    class MpiGuard
    {
    public:
        MpiGuard(int& argc, char**& argv);
        ~MpiGuard();
        MpiGuard(MpiGuard const&) = delete;
        MpiGuard& operator=(MpiGuard const&) = delete;

    private:
        bool m_ownsRuntime = false;
    };

    /*! MPI-backed collective context for the rank-parallel compute stages.
     *
     * Construct after ``MpiGuard``.  Reports rank/size of ``MPI_COMM_WORLD`` and
     * reduces in place.  Without MPI it is a single-rank context.
     */
    class MpiParallelContext final : public ParallelContext
    {
    public:
        MpiParallelContext();
        void allreduceSum(double* values, std::size_t count) const override;
        void allreduceSum(std::uint64_t& value) const override;
    };

    //! Broadcast every particle array and the per-particle metadata from root.
    void mpiBroadcastBunch(Bunch& bunch, int root);

    /*! Broadcast the resolved compute options (grid, margins, near-shell
     * policy) from root.  Pointers are never sent; the receiver keeps its own
     * ``parallel`` pointer, which the caller sets after this call.
     */
    void mpiBroadcastComputeOptions(ComputeOptions& options, int root);

    //! True when this build has MPI and the runtime is initialised.
    bool mpiActive();
    int mpiRank();
    int mpiSize();

    void mpiBarrier();

    //! Broadcast ``count`` doubles from ``root``; no-op for a single rank.
    void mpiBroadcast(double* values, std::size_t count, int root);
    void mpiBroadcastString(std::string& value, int root);

    /*! Broadcast the compute grid metadata and all six component arrays.
     *
     * On non-root ranks ``field`` is resized and filled from rank 0.  This is
     * the whole bunch-box field, not the global domain, so its size stays small.
     */
    void mpiBroadcastField(Field& field, int root);
} // namespace picBunch
