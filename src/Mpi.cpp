// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera

#include "picBunch/Mpi.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
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

    void mpiBroadcastComputeOptions(ComputeOptions& options, int root)
    {
#if defined(PICBUNCH_ENABLE_MPI)
        if(!initialized() || mpiSize() <= 1)
            return;
        double scalars[8] = {
            options.tolerance,
            options.marginCells,
            options.nearShell.radiusCells,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0};
        std::uint64_t integers[6] = {
            options.fftPadding,
            options.deterministicDeposit ? 1u : 0u,
            static_cast<std::uint64_t>(options.nearShell.mode),
            static_cast<std::uint64_t>(options.nearShell.tileNodes),
            static_cast<std::uint64_t>(options.fftMode),
            static_cast<std::uint64_t>(options.distributedFftMinNodes)};
        double cell[3] = {options.cellSize[0], options.cellSize[1], options.cellSize[2]};
        MPI_Bcast(scalars, 8, MPI_DOUBLE, root, MPI_COMM_WORLD);
        MPI_Bcast(integers, 6, MPI_UINT64_T, root, MPI_COMM_WORLD);
        MPI_Bcast(cell, 3, MPI_DOUBLE, root, MPI_COMM_WORLD);
        // Optional arrays: one presence flag, then three values each.
        auto broadcastOptional = [&](std::optional<std::array<std::size_t, 3>>& value) {
            std::uint64_t present = (mpiRank() == root && value) ? 1u : 0u;
            MPI_Bcast(&present, 1, MPI_UINT64_T, root, MPI_COMM_WORLD);
            if(present == 0u)
            {
                if(mpiRank() != root)
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
                if(mpiRank() != root)
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
        if(mpiRank() != root)
        {
            options.tolerance = scalars[0];
            options.marginCells = scalars[1];
            options.nearShell.radiusCells = scalars[2];
            options.fftPadding = static_cast<unsigned>(integers[0]);
            options.deterministicDeposit = integers[1] != 0u;
            options.nearShell.mode = static_cast<NearShellOptions::Mode>(integers[2]);
            options.nearShell.tileNodes = static_cast<std::size_t>(integers[3]);
            options.fftMode = static_cast<FftMode>(integers[4]);
            options.distributedFftMinNodes = static_cast<std::size_t>(integers[5]);
            options.cellSize = {cell[0], cell[1], cell[2]};
        }
#else
        (void)options;
        (void)root;
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
