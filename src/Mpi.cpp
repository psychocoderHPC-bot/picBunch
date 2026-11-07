// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera

#include "picBunch/Mpi.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(PICBUNCH_ENABLE_MPI)
#    include <mpi.h>
#endif

namespace picBunch
{
    namespace
    {
        bool initialized()
        {
#if defined(PICBUNCH_ENABLE_MPI)
            int flag = 0;
            MPI_Initialized(&flag);
            return flag != 0;
#else
            return false;
#endif
        }

        //! Version of the compute-options wire layout and fingerprint.
        constexpr std::uint64_t kComputeOptionsWireVersion = 4u;

        /*! Deterministic FNV-1a hasher over the logical option fields.
         *
         * Only fixed-width values are fed in, so no struct padding or endianness
         * dependent byte range leaks into the digest.  All ranks run the same
         * binary, so double bit patterns are comparable.
         */
        class OptionsHasher
        {
        public:
            void bytes(void const* data, std::size_t size)
            {
                auto const* begin = static_cast<unsigned char const*>(data);
                for(std::size_t i = 0; i < size; ++i)
                {
                    m_hash ^= static_cast<std::uint64_t>(begin[i]);
                    m_hash *= 1099511628211ull;
                }
            }

            void u64(std::uint64_t value)
            {
                bytes(&value, sizeof(value));
            }

            void real(double value)
            {
                std::uint64_t bits = 0;
                static_assert(sizeof(bits) == sizeof(value));
                std::memcpy(&bits, &value, sizeof(bits));
                u64(bits);
            }

            [[nodiscard]] std::uint64_t value() const
            {
                return m_hash;
            }

        private:
            std::uint64_t m_hash = 1469598103934665603ull;
        };

        void hashOptionalSizes(
            OptionsHasher& hasher,
            std::optional<std::array<std::size_t, 3>> const& value)
        {
            hasher.u64(value ? 1u : 0u);
            if(value)
                for(auto const entry : *value)
                    hasher.u64(static_cast<std::uint64_t>(entry));
        }

        void hashOptionalReals(
            OptionsHasher& hasher,
            std::optional<std::array<double, 3>> const& value)
        {
            hasher.u64(value ? 1u : 0u);
            if(value)
                for(auto const entry : *value)
                    hasher.real(entry);
        }
    } // namespace


    MpiParallelContext::MpiParallelContext()
    {
        rank = mpiRank();
        size = mpiSize();
    }

    void MpiParallelContext::allreduceSum(double* values, std::size_t count) const
    {
#if defined(PICBUNCH_ENABLE_MPI)
        if(initialized() && mpiSize() > 1 && count > 0)
            MPI_Allreduce(
                MPI_IN_PLACE,
                values,
                static_cast<int>(count),
                MPI_DOUBLE,
                MPI_SUM,
                MPI_COMM_WORLD);
#else
        (void)values;
        (void)count;
#endif
    }

    void MpiParallelContext::allreduceSum(std::uint64_t& value) const
    {
#if defined(PICBUNCH_ENABLE_MPI)
        if(initialized() && mpiSize() > 1)
            MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
#else
        (void)value;
#endif
    }

    void MpiParallelContext::allreduceMax(double* values, std::size_t count) const
    {
#if defined(PICBUNCH_ENABLE_MPI)
        if(initialized() && mpiSize() > 1 && count > 0)
            MPI_Allreduce(
                MPI_IN_PLACE,
                values,
                static_cast<int>(count),
                MPI_DOUBLE,
                MPI_MAX,
                MPI_COMM_WORLD);
#else
        (void)values;
        (void)count;
#endif
    }

    void MpiParallelContext::alltoallv(
        double* send,
        int const* sendCounts,
        int const* sendDispls,
        double* recv,
        int const* recvCounts,
        int const* recvDispls) const
    {
#if defined(PICBUNCH_ENABLE_MPI)
        if(initialized() && mpiSize() > 1)
        {
            MPI_Alltoallv(
                send,
                sendCounts,
                sendDispls,
                MPI_DOUBLE,
                recv,
                recvCounts,
                recvDispls,
                MPI_DOUBLE,
                MPI_COMM_WORLD);
            return;
        }
#endif
        // Single rank: the only block is rank 0's own contribution.
        if(sendCounts != nullptr && recvCounts != nullptr && sendCounts[0] > 0)
            std::copy_n(send + sendDispls[0], sendCounts[0], recv + recvDispls[0]);
    }

    void mpiBroadcastBunch(Bunch& bunch, int root)
    {
#if defined(PICBUNCH_ENABLE_MPI)
        if(!initialized() || mpiSize() <= 1)
            return;
        std::uint64_t count = (mpiRank() == root) ? bunch.size() : 0;
        MPI_Bcast(&count, 1, MPI_UINT64_T, root, MPI_COMM_WORLD);
        if(mpiRank() != root)
            bunch.resize(static_cast<std::size_t>(count));
        auto broadcastVector = [&](std::vector<double>& values, bool always) {
            std::uint64_t present = values.empty() ? 0u : 1u;
            MPI_Bcast(&present, 1, MPI_UINT64_T, root, MPI_COMM_WORLD);
            if(present == 0u && !always)
                return;
            if(mpiRank() != root)
                values.assign(static_cast<std::size_t>(count), 0.0);
            MPI_Bcast(values.data(), static_cast<int>(values.size()), MPI_DOUBLE, root, MPI_COMM_WORLD);
        };
        broadcastVector(bunch.x, true);
        broadcastVector(bunch.y, true);
        broadcastVector(bunch.z, true);
        broadcastVector(bunch.ux, true);
        broadcastVector(bunch.uy, true);
        broadcastVector(bunch.uz, true);
        broadcastVector(bunch.weight, true);
        broadcastVector(bunch.charge, true);
        broadcastVector(bunch.restMass, false);
#else
        (void)bunch;
        (void)root;
#endif
    }

    std::uint64_t computeOptionsFingerprint(ComputeOptions const& options)
    {
        OptionsHasher hasher;
        hasher.u64(kComputeOptionsWireVersion);
        // Physics scalars.
        hasher.real(options.tolerance);
        hasher.real(options.marginCells);
        hasher.real(options.nearShell.radiusCells);
        hasher.real(options.velocityBinning.modelConstant);
        // Integer / enum policy fields.  Every value is cast through a fixed
        // width so the digest cannot depend on the in-memory enum size.
        hasher.u64(static_cast<std::uint64_t>(options.fftPadding));
        hasher.u64(options.deterministicDeposit ? 1u : 0u);
        hasher.u64(static_cast<std::uint64_t>(options.nearShell.mode));
        hasher.u64(static_cast<std::uint64_t>(options.nearShell.tileNodes));
        hasher.u64(static_cast<std::uint64_t>(options.nearShell.fastOrder));
        hasher.u64(static_cast<std::uint64_t>(options.nearShell.fastObliqueOrder));
        hasher.u64(static_cast<std::uint64_t>(options.velocityBinning.mode));
        hasher.u64(static_cast<std::uint64_t>(options.velocityBinning.explicitBins));
        hasher.u64(static_cast<std::uint64_t>(options.velocityBinning.maxBins));
        hasher.u64(static_cast<std::uint64_t>(options.fftMode));
        hasher.u64(static_cast<std::uint64_t>(options.distributedFftMinNodes));
        hasher.u64(static_cast<std::uint64_t>(options.shape));
        hasher.u64(static_cast<std::uint64_t>(options.velocityMoments.mode));
        hasher.u64(static_cast<std::uint64_t>(options.velocityMoments.order));
        for(auto const entry : options.cellSize)
            hasher.real(entry);
        // Optional grid overrides: presence plus values.
        hashOptionalSizes(hasher, options.meshGridSize);
        hashOptionalReals(hasher, options.meshGridOrigin);
        hashOptionalSizes(hasher, options.gridSize);
        hashOptionalReals(hasher, options.gridOrigin);
        hashOptionalSizes(hasher, options.globalGridSize);
        hashOptionalReals(hasher, options.globalGridOrigin);
        hashOptionalReals(hasher, options.marginCellsPerAxis);
        return hasher.value();
    }

    void mpiBroadcastComputeOptions(ComputeOptions& options, int root)
    {
#if defined(PICBUNCH_ENABLE_MPI)
        if(!initialized() || mpiSize() <= 1)
            return;
        double scalars[4] = {
            options.tolerance,
            options.marginCells,
            options.nearShell.radiusCells,
            options.velocityBinning.modelConstant};
        std::uint64_t integers[14] = {
            static_cast<std::uint64_t>(options.fftPadding),
            options.deterministicDeposit ? 1u : 0u,
            static_cast<std::uint64_t>(options.nearShell.mode),
            static_cast<std::uint64_t>(options.nearShell.tileNodes),
            static_cast<std::uint64_t>(options.nearShell.fastOrder),
            static_cast<std::uint64_t>(options.nearShell.fastObliqueOrder),
            static_cast<std::uint64_t>(options.velocityBinning.mode),
            static_cast<std::uint64_t>(options.velocityBinning.explicitBins),
            static_cast<std::uint64_t>(options.velocityBinning.maxBins),
            static_cast<std::uint64_t>(options.fftMode),
            static_cast<std::uint64_t>(options.distributedFftMinNodes),
            static_cast<std::uint64_t>(options.shape),
            static_cast<std::uint64_t>(options.velocityMoments.mode),
            static_cast<std::uint64_t>(options.velocityMoments.order)};
        double cell[3] = {options.cellSize[0], options.cellSize[1], options.cellSize[2]};
        std::uint64_t version = kComputeOptionsWireVersion;
        MPI_Bcast(&version, 1, MPI_UINT64_T, root, MPI_COMM_WORLD);
        if(version != kComputeOptionsWireVersion)
            throw std::runtime_error("MPI compute-options wire version mismatch");
        MPI_Bcast(scalars, 4, MPI_DOUBLE, root, MPI_COMM_WORLD);
        MPI_Bcast(integers, 14, MPI_UINT64_T, root, MPI_COMM_WORLD);
        MPI_Bcast(cell, 3, MPI_DOUBLE, root, MPI_COMM_WORLD);
        // Optional arrays: one presence flag, then three values each.
        auto broadcastOptional = [&](std::optional<std::array<std::size_t, 3>>& value) {
            std::uint64_t present = (mpiRank() == root && value) ? 1u : 0u;
            MPI_Bcast(&present, 1, MPI_UINT64_T, root, MPI_COMM_WORLD);
            if(present == 0u)
            {
                value.reset();
                return;
            }
            std::uint64_t data[3] = {};
            if(mpiRank() == root)
                for(int a = 0; a < 3; ++a)
                    data[a] = (*value)[static_cast<std::size_t>(a)];
            MPI_Bcast(data, 3, MPI_UINT64_T, root, MPI_COMM_WORLD);
            value = std::array<std::size_t, 3>{
                static_cast<std::size_t>(data[0]),
                static_cast<std::size_t>(data[1]),
                static_cast<std::size_t>(data[2])};
        };
        auto broadcastOptionalDouble = [&](std::optional<std::array<double, 3>>& value) {
            std::uint64_t present = (mpiRank() == root && value) ? 1u : 0u;
            MPI_Bcast(&present, 1, MPI_UINT64_T, root, MPI_COMM_WORLD);
            if(present == 0u)
            {
                value.reset();
                return;
            }
            double data[3] = {};
            if(mpiRank() == root)
                for(int a = 0; a < 3; ++a)
                    data[a] = (*value)[static_cast<std::size_t>(a)];
            MPI_Bcast(data, 3, MPI_DOUBLE, root, MPI_COMM_WORLD);
            value = std::array<double, 3>{data[0], data[1], data[2]};
        };
        broadcastOptional(options.meshGridSize);
        broadcastOptionalDouble(options.meshGridOrigin);
        broadcastOptional(options.gridSize);
        broadcastOptionalDouble(options.gridOrigin);
        broadcastOptional(options.globalGridSize);
        broadcastOptionalDouble(options.globalGridOrigin);
        broadcastOptionalDouble(options.marginCellsPerAxis);
        if(mpiRank() != root)
        {
            options.tolerance = scalars[0];
            options.marginCells = scalars[1];
            options.nearShell.radiusCells = scalars[2];
            options.velocityBinning.modelConstant = scalars[3];
            options.fftPadding = static_cast<unsigned>(integers[0]);
            options.deterministicDeposit = integers[1] != 0u;
            options.nearShell.mode = static_cast<NearShellOptions::Mode>(integers[2]);
            options.nearShell.tileNodes = static_cast<std::size_t>(integers[3]);
            options.nearShell.fastOrder = static_cast<int>(integers[4]);
            options.nearShell.fastObliqueOrder = static_cast<int>(integers[5]);
            options.velocityBinning.mode = static_cast<VelocityBinningOptions::Mode>(integers[6]);
            options.velocityBinning.explicitBins = static_cast<std::size_t>(integers[7]);
            options.velocityBinning.maxBins = static_cast<std::size_t>(integers[8]);
            options.fftMode = static_cast<FftMode>(integers[9]);
            options.distributedFftMinNodes = static_cast<std::size_t>(integers[10]);
            options.shape = static_cast<Shape>(integers[11]);
            options.velocityMoments.mode
                = static_cast<VelocityMomentsOptions::Mode>(integers[12]);
            options.velocityMoments.order = static_cast<std::size_t>(integers[13]);
            options.cellSize = {cell[0], cell[1], cell[2]};
        }
        // Verify that every rank ended up with root's effective options.
        std::uint64_t reference = (mpiRank() == root) ? computeOptionsFingerprint(options) : 0u;
        MPI_Bcast(&reference, 1, MPI_UINT64_T, root, MPI_COMM_WORLD);
        std::uint64_t const local = computeOptionsFingerprint(options);
        int const localMismatch = (local == reference) ? 0 : 1;
        int mismatch = 0;
        MPI_Allreduce(&localMismatch, &mismatch, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if(mismatch != 0)
        {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            char detail[160];
            std::snprintf(
                detail,
                sizeof(detail),
                "rank %d local fingerprint %016llx differs from root %016llx",
                rank,
                static_cast<unsigned long long>(local),
                static_cast<unsigned long long>(reference));
            throw std::runtime_error(
                std::string("MPI compute-options broadcast produced different physics "
                            "options on different ranks: ")
                + detail);
        }
#else
        (void)options;
        (void)root;
#endif
    }

    int mpiBroadcastRootSetupStatus(std::string& message, int root)
    {
#if defined(PICBUNCH_ENABLE_MPI)
        if(initialized() && mpiSize() > 1)
        {
            int status = 0;
            int length = 0;
            if(mpiRank() == root)
            {
                status = message.empty() ? 0 : 1;
                length = static_cast<int>(message.size());
            }
            MPI_Bcast(&status, 1, MPI_INT, root, MPI_COMM_WORLD);
            MPI_Bcast(&length, 1, MPI_INT, root, MPI_COMM_WORLD);
            if(mpiRank() != root)
                message.resize(length > 0 ? static_cast<std::size_t>(length) : 0u);
            if(length > 0)
                MPI_Bcast(message.data(), length, MPI_CHAR, root, MPI_COMM_WORLD);
            return status;
        }
#else
        (void)root;
#endif
        return message.empty() ? 0 : 1;
    }

    void mpiAbort(int errorCode)
    {
#if defined(PICBUNCH_ENABLE_MPI)
        if(initialized())
        {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if(!finalized && mpiSize() > 1)
                MPI_Abort(MPI_COMM_WORLD, errorCode);
        }
#else
        (void)errorCode;
#endif
    }

    MpiGuard::MpiGuard(int& argc, char**& argv)
    {
#if defined(PICBUNCH_ENABLE_MPI)
        int flag = 0;
        MPI_Initialized(&flag);
        if(!flag)
        {
            int provided = 0;
            if(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS)
                throw std::runtime_error("MPI_Init_thread failed");
            m_ownsRuntime = true;
        }
        // Silence unused-parameter warnings in the non-MPI build.
        (void)argc;
        (void)argv;
#else
        (void)argc;
        (void)argv;
#endif
    }

    MpiGuard::~MpiGuard()
    {
#if defined(PICBUNCH_ENABLE_MPI)
        int flag = 0;
        MPI_Finalized(&flag);
        if(m_ownsRuntime && !flag)
            MPI_Finalize();
#endif
    }

    bool mpiActive()
    {
        return initialized();
    }

    int mpiRank()
    {
#if defined(PICBUNCH_ENABLE_MPI)
        if(initialized())
        {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            return rank;
        }
#endif
        return 0;
    }

    int mpiSize()
    {
#if defined(PICBUNCH_ENABLE_MPI)
        if(initialized())
        {
            int size = 1;
            MPI_Comm_size(MPI_COMM_WORLD, &size);
            return size;
        }
#endif
        return 1;
    }

    void mpiBarrier()
    {
#if defined(PICBUNCH_ENABLE_MPI)
        if(initialized())
            MPI_Barrier(MPI_COMM_WORLD);
#endif
    }

    void mpiBroadcast(double* values, std::size_t count, int root)
    {
#if defined(PICBUNCH_ENABLE_MPI)
        if(initialized() && mpiSize() > 1)
            MPI_Bcast(values, static_cast<int>(count), MPI_DOUBLE, root, MPI_COMM_WORLD);
#else
        (void)values;
        (void)count;
        (void)root;
#endif
    }

    void mpiBroadcastString(std::string& value, int root)
    {
#if defined(PICBUNCH_ENABLE_MPI)
        if(initialized() && mpiSize() > 1)
        {
            int length = (mpiRank() == root) ? static_cast<int>(value.size()) : 0;
            MPI_Bcast(&length, 1, MPI_INT, root, MPI_COMM_WORLD);
            if(mpiRank() != root)
                value.resize(static_cast<std::size_t>(length));
            MPI_Bcast(value.data(), length, MPI_CHAR, root, MPI_COMM_WORLD);
        }
#else
        (void)value;
        (void)root;
#endif
    }

    void mpiBroadcastField(Field& field, int root)
    {
        // A broadcast field is always the full grid on every rank.
        field.zSlab = false;
        field.slabOffset = 0;
        field.slabCount = 0;
#if defined(PICBUNCH_ENABLE_MPI)
        if(!initialized() || mpiSize() <= 1)
            return;
        std::uint64_t meta[6] = {
            field.grid.n[0],
            field.grid.n[1],
            field.grid.n[2],
            field.grid.fftPadding,
            0,
            0};
        double spacing[3] = {
            field.grid.spacing[0],
            field.grid.spacing[1],
            field.grid.spacing[2]};
        double origin[3] = {field.grid.origin[0], field.grid.origin[1], field.grid.origin[2]};
        MPI_Bcast(meta, 6, MPI_UINT64_T, root, MPI_COMM_WORLD);
        MPI_Bcast(spacing, 3, MPI_DOUBLE, root, MPI_COMM_WORLD);
        MPI_Bcast(origin, 3, MPI_DOUBLE, root, MPI_COMM_WORLD);
        field.grid.n = {
            static_cast<std::size_t>(meta[0]),
            static_cast<std::size_t>(meta[1]),
            static_cast<std::size_t>(meta[2])};
        field.grid.fftPadding = static_cast<unsigned>(meta[3]);
        field.grid.spacing = {spacing[0], spacing[1], spacing[2]};
        field.grid.origin = {origin[0], origin[1], origin[2]};
        for(auto& component : field.data)
        {
            component.resize(field.grid.size());
            MPI_Bcast(component.data(), static_cast<int>(component.size()), MPI_DOUBLE, root, MPI_COMM_WORLD);
        }
#else
        (void)field;
        (void)root;
#endif
    }
} // namespace picBunch
