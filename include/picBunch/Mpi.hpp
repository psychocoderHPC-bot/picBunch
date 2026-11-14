// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/Parallel.hpp"
#include "picBunch/Types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace picBunch
{
    /*! Optional MPI runtime.
     *
     * Built only with ``PICBUNCH_ENABLE_MPI``.  Without that option every
     * function below behaves as a single rank: ``mpiRank()`` is 0,
     * ``mpiSize()`` is 1, broadcasts and barriers are no-ops.  Rank 0 reads the
     * input, resolves the geometry and broadcasts the bunch and the resolved
     * options; every rank then computes its component or output-tile share of
     * the field.  The result is either replicated (component-parallel mesh) or
     * left as one z-slab per rank (distributed FFT and chunked writer).  All
     * ranks participate in the collective stages and the disjoint writes.
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
        void allreduceMax(double* values, std::size_t count) const override;
        void alltoallv(
            double* send,
            int const* sendCounts,
            int const* sendDispls,
            double* recv,
            int const* recvCounts,
            int const* recvDispls) const override;
        void allgatherv(
            double const* send,
            int sendCount,
            double* recv,
            int const* recvCounts,
            int const* recvDispls) const override;
    };

    //! Broadcast every particle array and the per-particle metadata from root.
    void mpiBroadcastBunch(Bunch& bunch, int root);

    /*! Broadcast the resolved compute options (grid, margins, near-shell
     * policy, velocity binning) from root.  Pointers are never sent; the
     * receiver keeps its own ``parallel`` pointer, which the caller sets after
     * this call.
     *
     * Every effective physics option travels in a versioned fixed-width wire
     * format.  After the transfer each rank recomputes
     * ``computeOptionsFingerprint`` and the rank set verifies it against root;
     * a mismatch throws on every rank instead of letting components run with
     * different physics.
     */
    void mpiBroadcastComputeOptions(ComputeOptions& options, int root);

    /*! Deterministic fingerprint of every broadcast physics option.
     *
     * Covers the near-shell policy (including ``fastOrder`` and
     * ``fastObliqueOrder``) and all ``velocityBinning`` fields, together with
     * the grid, margin, FFT and determinism options.  The value is identical on
     * every rank after a correct broadcast and is independent of MPI.
     */
    std::uint64_t computeOptionsFingerprint(ComputeOptions const& options);

    /*! Broadcast the outcome of root-only setup (reader, config, geometry).
     *
     * On root ``message`` carries the error text (empty on success); on every
     * rank the return value is 0 on success and nonzero on failure, with
     * ``message`` holding root's text.  The call is collective and must be made
     * by all ranks before any data broadcast.
     */
    int mpiBroadcastRootSetupStatus(std::string& message, int root);

    //! Abort the MPI job when MPI is initialised with more than one rank.
    //!
    //! Serial builds and single-rank runs are a no-op so the caller can keep
    //! returning a nonzero status.  Used by the top-level failure path after a
    //! collective has started, where a local ``return`` would strand peers.
    void mpiAbort(int errorCode);

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
