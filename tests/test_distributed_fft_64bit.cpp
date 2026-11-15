// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group DistributedFft64Bit: regression for the 32-bit index overflow in the
// distributed FFT path.
//
// Measured trigger (rosi_v100, V100, CUDA, commit 69378dcd): a per-rank
// padded slab above 2^28 complex nodes (4 GiB with Complex<double>) corrupted
// the field.  With a 100-particle bunch at 1.772e-7 m cell size, padding 2:
//
//   grid            np  padded/rank    max|E|
//   336x512x352      2  242,221,056    5.66921e+07 V/m   correct
//   336x576x352      2  272,498,688    1.02146e+25 V/m   corrupt
//   336x640x352      4  151,388,160    5.66921e+07 V/m   correct
//   336x1152x352     4  272,498,688    1.02146e+25 V/m   corrupt
//   336x640x352      2  302,776,320    7.53219e+25 V/m   corrupt
//
// The 1-rank replicated run of the same bunch on 64x128x64 gives
// 5.66921e+07 V/m, so the below-limit slab is the in-test reference.
//
// The end-to-end boundary case needs more than 4 GiB per rank (about 13 GiB of
// unified memory per rank for the work, spectrum and kernel slabs), so it is
// opt-in:
//
//   PICBUNCH_LARGE_FFT_TESTS=1 PICBUNCH_LARGE_FFT_GPUS=2,3 mpirun -n 2 picBunchUnitTests --gtest_filter=DistributedFft64Bit.*
//   PICBUNCH_LARGE_FFT_TESTS=1 PICBUNCH_LARGE_FFT_GPUS=2,3 mpirun -n 4 picBunchUnitTests --gtest_filter=DistributedFft64Bit.*
//
// PICBUNCH_LARGE_FFT_GPUS remaps CUDA_VISIBLE_DEVICES per rank (rank modulo the
// list) before the first CUDA call, so four ranks run two per card instead of
// four on card 0.  The guard test below is cheap and always runs.

#include "picBunch/Compute.hpp"
#include "picBunch/Mpi.hpp"
#include "picBunch/Parallel.hpp"

#include <alpaka/math/Complex.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#if defined(PICBUNCH_ENABLE_MPI)
#    include <mpi.h>
#endif

namespace
{
#if defined(PICBUNCH_ENABLE_MPI)
    void ensureMpiInitialised()
    {
        int initialised = 0;
        MPI_Initialized(&initialised);
        if(!initialised)
            MPI_Init(nullptr, nullptr);
    }

    // gtest_main returns from main without MPI_Finalize; OpenMPI then reports
    // an improper exit even when every test passed.
    struct MpiFinalizer
    {
        ~MpiFinalizer()
        {
            int initialised = 0;
            int finalised = 0;
            MPI_Initialized(&initialised);
            MPI_Finalized(&finalised);
            if(initialised && !finalised)
                MPI_Finalize();
        }
    };
#endif

    //! Map this rank to its own GPU when the operator supplied a device list.
    void pinRankToGpu(int rank)
    {
        char const* list = std::getenv("PICBUNCH_LARGE_FFT_GPUS");
        if(list == nullptr || *list == '\0')
            return;
        std::vector<std::string> devices;
        std::string current;
        for(char const* c = list; *c != '\0'; ++c)
        {
            if(*c == ',')
            {
                if(!current.empty())
                    devices.push_back(current);
                current.clear();
            }
            else
                current.push_back(*c);
        }
        if(!current.empty())
            devices.push_back(current);
        if(devices.empty())
            return;
        std::string const selected = devices[static_cast<std::size_t>(rank) % devices.size()];
        ::setenv("CUDA_VISIBLE_DEVICES", selected.c_str(), 1);
    }

    //! Deterministic pseudo-random positions in a box centred in the grid.
    picBunch::Bunch makeCentredBunch(
        std::array<std::size_t, 3> const& cells,
        std::size_t count,
        double boxCells)
    {
        picBunch::Bunch bunch;
        bunch.resize(count);
        auto next = [state = std::uint64_t{0x9e3779b97f4a7c15u}]() mutable {
            state ^= state << 13u;
            state ^= state >> 7u;
            state ^= state << 17u;
            return static_cast<double>(state >> 11u) * (1.0 / 9007199254740992.0);
        };
        double const gamma = 490.0;
        // The read-out coordinates are metres; ``cells`` is a node count, so a
        // cell-centred box is scaled by the cell size.
        double const cell = 1.772e-7;
        for(std::size_t p = 0; p < count; ++p)
        {
            double const lo[3] = {
                (static_cast<double>(cells[0]) - boxCells) * 0.5,
                (static_cast<double>(cells[1]) - boxCells) * 0.5,
                (static_cast<double>(cells[2]) - boxCells) * 0.5};
            bunch.x[p] = (lo[0] + boxCells * next()) * cell;
            bunch.y[p] = (lo[1] + boxCells * next()) * cell;
            bunch.z[p] = (lo[2] + boxCells * next()) * cell;
            bunch.ux[p] = 0.0;
            bunch.uy[p] = gamma;
            bunch.uz[p] = 0.0;
            bunch.weight[p] = 1.0;
            bunch.charge[p] = -picBunch::kElementaryCharge;
            bunch.restMass[p] = 9.1093837139e-31;
        }
        return bunch;
    }

    picBunch::ComputeOptions makeOptions(std::array<std::size_t, 3> const& cells)
    {
        picBunch::ComputeOptions options;
        options.cellSize = {1.772e-7, 1.772e-7, 1.772e-7};
        options.gridSize = {cells[0], cells[1], cells[2]};
        options.gridOrigin = {0.0, 0.0, 0.0};
        options.fftMode = picBunch::FftMode::Distributed;
        // padding 2, as in the reproducer
        options.tolerance = 1.0e-3;
        options.velocityBinning.mode = picBunch::VelocityBinningOptions::Mode::Explicit;
        options.velocityBinning.explicitBins = 1;
        return options;
    }

    //! Largest |field| of any component on any rank.
    double fieldPeak(picBunch::Field const& field, picBunch::ParallelContext const& par)
    {
        double local = 0.0;
        for(auto const& component : field.data)
            for(double const value : component)
                local = std::max(local, std::abs(value));
        par.allreduceMax(&local, 1u);
        return local;
    }
} // namespace

TEST(DistributedFft64Bit, AboveFourGiBPerRankMatchesBelowLimit)
{
#if defined(PICBUNCH_ENABLE_MPI)
    ensureMpiInitialised();
    static MpiFinalizer finalizer;
#endif
    if(std::getenv("PICBUNCH_LARGE_FFT_TESTS") == nullptr)
        GTEST_SKIP() << "set PICBUNCH_LARGE_FFT_TESTS=1 to run the >4 GiB per-rank boundary case";
    if(std::getenv("PICBUNCH_LARGE_FFT_GPUS") == nullptr)
        GTEST_SKIP()
            << "set PICBUNCH_LARGE_FFT_GPUS to the device list (e.g. 2,3) so every rank "
               "gets its own GPU; without it all ranks would share device 0 and run out of memory";
    picBunch::MpiParallelContext parallel;
    if(parallel.size < 2)
        GTEST_SKIP() << "run under 'mpirun -n 2' or 'mpirun -n 4'";
    ASSERT_TRUE(parallel.size == 2 || parallel.size == 4)
        << "the measured boundary matrix covers 2 and 4 ranks";
    pinRankToGpu(parallel.rank);

    // Per-rank padded nodes = 8*nx*ny*nz / ranks (padding 2).  The below case
    // stays under 2^28 per rank, the above case is in (2^28, 2^29).
    std::array<std::size_t, 3> below{336u, 512u, 352u};
    std::array<std::size_t, 3> above{336u, 576u, 352u};
    if(parallel.size == 4)
    {
        below = {336u, 640u, 352u};
        above = {336u, 1152u, 352u};
    }
    auto const perRank = [&](std::array<std::size_t, 3> const& cells) {
        return 8u * cells[0] * cells[1] * cells[2]
            / static_cast<std::size_t>(parallel.size);
    };
    std::size_t const boundary = std::size_t{1u} << 28;
    ASSERT_LT(perRank(below), boundary);
    ASSERT_GT(perRank(above), boundary);
    // ..and the above case really is past the 4 GiB byte-offset boundary.
    ASSERT_GT(perRank(above) * sizeof(alpaka::math::Complex<double>), std::size_t{1u} << 32);

    picBunch::ComputeOptions belowOptions = makeOptions(below);
    belowOptions.parallel = &parallel;
    picBunch::Field const belowField
        = picBunch::computeField(makeCentredBunch(below, 100u, 20.0), belowOptions);
    double const belowPeak = fieldPeak(belowField, parallel);
    ASSERT_TRUE(std::isfinite(belowPeak));
    ASSERT_GT(belowPeak, 0.0);

    picBunch::ComputeOptions aboveOptions = makeOptions(above);
    aboveOptions.parallel = &parallel;
    picBunch::Field const aboveField
        = picBunch::computeField(makeCentredBunch(above, 100u, 20.0), aboveOptions);
    double const abovePeak = fieldPeak(aboveField, parallel);

    // The 1-rank replicated run of the same bunch on 64x128x64 gives
    // 5.66921e+07 V/m; the smaller distributed box reproduces it.  The buggy
    // path produced 1.02e+25 V/m, eighteen orders larger.
    if(parallel.rank == 0)
        std::printf(
            "DistributedFft64Bit: ranks=%d per-rank nodes below=%zu above=%zu; "
            "max|E| below=%.17g above=%.17g V/m\n",
            parallel.size,
            perRank(below),
            perRank(above),
            belowPeak,
            abovePeak);
    EXPECT_NEAR(abovePeak, belowPeak, 1.0e-4 * belowPeak)
        << "per-rank " << perRank(above) << " padded nodes (above 4 GiB): max|E| "
        << abovePeak << " V/m vs " << belowPeak << " V/m below the limit (rank "
        << parallel.rank << " of " << parallel.size << ")";
}
