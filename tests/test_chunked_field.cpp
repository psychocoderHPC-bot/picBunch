// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group: Chunked.  The out-of-core 3-D solve must reproduce the in-core
// transformed field of a known analytic source (a single charge, then a
// low-order multipole) to FFT round-off, independently of the chunk size.

#include "picBunch/Compute.hpp"
#include "picBunch/Types.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include <unistd.h>

#if !defined(PICBUNCH_BACKEND_CUDA) && !defined(PICBUNCH_BACKEND_HIP)

namespace
{
    double const kCell = 1.0e-6;

    picBunch::Bunch singleCharge()
    {
        picBunch::Bunch bunch;
        bunch.resize(1);
        bunch.x[0] = 6.0 * kCell;
        bunch.y[0] = 5.0 * kCell;
        bunch.z[0] = 7.0 * kCell;
        bunch.ux[0] = 0.02;
        bunch.uy[0] = 0.0;
        bunch.uz[0] = 0.0;
        bunch.weight[0] = 1.0;
        bunch.charge[0] = picBunch::kElementaryCharge;
        return bunch;
    }

    //! Eight alternating charges: a dipole plus a quadrupole, non-symmetric.
    picBunch::Bunch multipole()
    {
        picBunch::Bunch bunch;
        double const sign[8] = {1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125};
        for(int i = 0; i < 8; ++i)
        {
            std::size_t const p = bunch.size();
            bunch.resize(p + 1);
            double const sx = (i & 1) ? 1.0 : -1.0;
            double const sy = (i & 2) ? 1.0 : -1.0;
            double const sz = (i & 4) ? 1.0 : -1.0;
            bunch.x[p] = (6.0 + 2.0 * sx) * kCell;
            bunch.y[p] = (5.0 + 1.5 * sy) * kCell;
            bunch.z[p] = (7.0 + 2.5 * sz) * kCell;
            bunch.ux[p] = 0.01;
            bunch.uy[p] = 0.02;
            bunch.uz[p] = 0.0;
            bunch.weight[p] = sign[i];
            bunch.charge[p] = picBunch::kElementaryCharge;
        }
        return bunch;
    }

    picBunch::ComputeOptions options(std::size_t chunkCells)
    {
        picBunch::ComputeOptions options;
        options.tolerance = 1.0e-3;
        options.fftPadding = 2;
        options.cellSize = {kCell, kCell, kCell};
        options.gridSize = picBunch::Idx3{12, 10, 14};
        options.gridOrigin = picBunch::Vec3{0.0, 0.0, 0.0};
        options.deterministicDeposit = true;
        if(chunkCells != 0u)
        {
            options.solveMode = picBunch::SolveMode::Chunked;
            options.chunkCells = chunkCells;
        }
        return options;
    }

    std::filesystem::path stagingPath()
    {
        return std::filesystem::temp_directory_path()
            / ("picbunch-chunked-" + std::to_string(static_cast<long long>(getpid())));
    }

    //! Largest relative difference over the six components.
    double relativeDifference(picBunch::Field const& a, picBunch::Field const& b)
    {
        double peak = 0.0;
        double largest = 0.0;
        for(std::size_t c = 0; c < 6; ++c)
        {
            EXPECT_EQ(a.data[c].size(), b.data[c].size());
            for(std::size_t l = 0; l < a.data[c].size(); ++l)
            {
                largest = std::max(largest, std::abs(a.data[c][l] - b.data[c][l]));
                peak = std::max(peak, std::max(std::abs(a.data[c][l]), std::abs(b.data[c][l])));
            }
        }
        return largest / std::max(peak, 1.0e-300);
    }
} // namespace

TEST(Chunked, SingleChargeMatchesInCore)
{
    auto const bunch = singleCharge();
    picBunch::Field const inCore = picBunch::computeField(bunch, options(0u));
    auto chunkedOptions = options(4u * 24u * 20u);
    chunkedOptions.stagingDirectory = stagingPath().string();
    picBunch::Field const chunked = picBunch::computeField(bunch, chunkedOptions);
    EXPECT_LT(relativeDifference(inCore, chunked), 1.0e-12);
    std::filesystem::remove_all(stagingPath());
}

TEST(Chunked, MultipoleMatchesInCore)
{
    auto const bunch = multipole();
    picBunch::Field const inCore = picBunch::computeField(bunch, options(0u));
    auto chunkedOptions = options(3u * 24u * 20u);
    chunkedOptions.stagingDirectory = stagingPath().string();
    picBunch::Field const chunked = picBunch::computeField(bunch, chunkedOptions);
    EXPECT_LT(relativeDifference(inCore, chunked), 1.0e-12);
    std::filesystem::remove_all(stagingPath());
}

TEST(Chunked, ChunkSizeDoesNotChangeResult)
{
    auto const bunch = multipole();
    auto smallOptions = options(2u * 24u * 20u);
    smallOptions.stagingDirectory = stagingPath().string();
    picBunch::Field const small = picBunch::computeField(bunch, smallOptions);
    auto largeOptions = options(14u * 24u * 20u);
    largeOptions.stagingDirectory = stagingPath().string();
    picBunch::Field const large = picBunch::computeField(bunch, largeOptions);
    EXPECT_LT(relativeDifference(small, large), 1.0e-13);
    std::filesystem::remove_all(stagingPath());
}

TEST(Chunked, SinkStreamsTheSameField)
{
    auto const bunch = multipole();
    picBunch::Field const inCore = picBunch::computeField(bunch, options(0u));

    picBunch::Field streamed;
    streamed.grid = inCore.grid;
    std::size_t const plane = inCore.grid.n[0] * inCore.grid.n[1];
    for(auto& component : streamed.data)
        component.assign(inCore.grid.size(), 0.0);
    std::vector<std::vector<unsigned char>> covered(
        6u,
        std::vector<unsigned char>(inCore.grid.n[2], 0u));

    auto chunkedOptions = options(2u * 24u * 20u);
    chunkedOptions.stagingDirectory = stagingPath().string();
    chunkedOptions.chunkSink = [&](picBunch::FieldChunk const& chunk) {
        ASSERT_GE(chunk.component, 0);
        ASSERT_LT(chunk.component, 6);
        ASSERT_EQ(chunk.data.size(), plane * chunk.zCount);
        for(std::size_t k = 0; k < chunk.zCount; ++k)
            ++covered[static_cast<std::size_t>(chunk.component)][chunk.zOffset + k];
        std::copy(
            chunk.data.begin(),
            chunk.data.end(),
            streamed.data[static_cast<std::size_t>(chunk.component)].begin()
                + static_cast<std::ptrdiff_t>(chunk.zOffset * plane));
    };
    picBunch::Field const returned = picBunch::computeField(bunch, chunkedOptions);
    EXPECT_TRUE(returned.data[0].empty());
    // Every component covers every plane exactly once.
    for(auto const& mask : covered)
        for(auto value : mask)
            EXPECT_EQ(static_cast<int>(value), 1);
    EXPECT_LT(relativeDifference(inCore, streamed), 1.0e-12);
    std::filesystem::remove_all(stagingPath());
}

#endif // host backend
