// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera

#include "picBunch/Mpi.hpp"

#include <cstdint>
#include <stdexcept>

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
