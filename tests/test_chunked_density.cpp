// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group: ChunkedDensity.  The chunked path deposits slab by slab with the same
// TSC shape, index wrap and node indexing as the in-core host deposit.  This
// test compares the concatenated slabs bit for bit with the in-core padded
// density before any transform, and checks the weight of a particle that sits
// exactly on a virtual-chunk boundary.

#include "picBunch/Compute.hpp"
#include "picBunch/Deposit.hpp"
#include "picBunch/DistributedFft.hpp"
#include "picBunch/Types.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

namespace
{
    picBunch::Grid smallGrid()
    {
        picBunch::Grid grid;
        grid.n = {64, 96, 64};
        grid.spacing = {1.0e-6, 1.0e-6, 1.0e-6};
        grid.origin = {0.0, 0.0, 0.0};
        grid.fftPadding = 2;
        grid.shape = picBunch::Shape::TSC;
        return grid;
    }

    picBunch::Bunch cloud()
    {
        picBunch::Bunch bunch;
        // A deterministic cloud plus one particle exactly on the z=32 chunk
        // boundary of a four-chunk split of Mz=128.
        std::uint64_t state = 0x9e3779b97f4a7c15ull;
        auto next = [&]() {
            state = state * 6364136223846793005ull + 1442695040888963407ull;
            return static_cast<double>((state >> 11) & 0xffffffffu) / 4294967296.0;
        };
        for(int i = 0; i < 64; ++i)
        {
            std::size_t const p = bunch.size();
            bunch.resize(p + 1);
            bunch.x[p] = (0.5 + 40.0 * next()) * 1.0e-6;
            bunch.y[p] = (0.5 + 60.0 * next()) * 1.0e-6;
            bunch.z[p] = (0.5 + 40.0 * next()) * 1.0e-6;
            bunch.weight[p] = 0.5 + next();
            bunch.charge[p] = picBunch::kElementaryCharge;
        }
        {
            std::size_t const p = bunch.size();
            bunch.resize(p + 1);
            bunch.x[p] = 30.25e-6;
            bunch.y[p] = 40.50e-6;
            bunch.z[p] = 32.0e-6; // exact boundary of chunk 0/1
            bunch.weight[p] = 1.25;
            bunch.charge[p] = picBunch::kElementaryCharge;
        }
        return bunch;
    }

    std::vector<double> chunkedDensity(
        picBunch::Bunch const& bunch,
        picBunch::Grid const& grid,
        int ranks)
    {
        std::uint32_t const Mx = static_cast<std::uint32_t>(grid.n[0] * grid.fftPadding);
        std::uint32_t const My = static_cast<std::uint32_t>(grid.n[1] * grid.fftPadding);
        std::uint32_t const Mz = static_cast<std::uint32_t>(grid.n[2] * grid.fftPadding);
        std::vector<std::size_t> particles(bunch.size());
        std::iota(particles.begin(), particles.end(), std::size_t{0});
        std::vector<double> full(static_cast<std::size_t>(Mx) * My * Mz, 0.0);
        std::size_t const plane = static_cast<std::size_t>(Mx) * My;
        for(int r = 0; r < ranks; ++r)
        {
            auto const z = picBunch::detail::slabRange(Mz, ranks, r);
            std::vector<double> slab(plane * z.second, 0.0);
            picBunch::detail::depositPaddedSlabHost(
                bunch,
                particles,
                grid,
                Mx,
                My,
                Mz,
                z.first,
                z.second,
                slab.data(),
                1u,
                grid.shape);
            std::copy(
                slab.begin(),
                slab.end(),
                full.begin() + static_cast<std::ptrdiff_t>(plane * z.first));
        }
        return full;
    }
} // namespace

TEST(ChunkedDensity, SlabsEqualInCoreDensityBitForBit)
{
    auto const grid = smallGrid();
    auto const bunch = cloud();
    auto const reference = picBunch::depositTSCPaddedHost(bunch, grid);
    auto const chunked = chunkedDensity(bunch, grid, 4);
    ASSERT_EQ(reference.size(), chunked.size());
    std::size_t different = 0;
    for(std::size_t i = 0; i < reference.size(); ++i)
        if(reference[i] != chunked[i])
            ++different;
    EXPECT_EQ(different, 0u);
}

TEST(ChunkedDensity, BoundaryParticleWeightAppearsExactlyOnce)
{
    auto const grid = smallGrid();
    picBunch::Bunch bunch;
    bunch.resize(1);
    bunch.x[0] = 30.25e-6;
    bunch.y[0] = 40.50e-6;
    bunch.z[0] = 32.0e-6; // exact boundary between chunk 0 and chunk 1
    bunch.weight[0] = 1.25;
    bunch.charge[0] = picBunch::kElementaryCharge;
    double const analytic = bunch.weight[0] * bunch.charge[0];

    auto const reference = picBunch::depositTSCPaddedHost(bunch, grid);
    double const total = std::accumulate(reference.begin(), reference.end(), 0.0);
    std::printf(
        "chunked density boundary particle: deposited sum=%.17g analytic=%.17g "
        "relative=%.3e\n",
        total,
        analytic,
        std::abs(total - analytic) / std::abs(analytic));
    EXPECT_NEAR(total, analytic, 1e-14 * std::abs(analytic));

    // The support of a particle at z=32 reaches z=31 (chunk 0) and z=32,33
    // (chunk 1).  Both chunks must receive mass, and their sum is the full
    // weight exactly once.
    auto const chunked = chunkedDensity(bunch, grid, 4);
    std::uint32_t const Mx = static_cast<std::uint32_t>(grid.n[0] * grid.fftPadding);
    std::uint32_t const My = static_cast<std::uint32_t>(grid.n[1] * grid.fftPadding);
    std::size_t const plane = static_cast<std::size_t>(Mx) * My;
    double const chunk0 = std::accumulate(
        chunked.begin(),
        chunked.begin() + static_cast<std::ptrdiff_t>(plane * 32),
        0.0);
    double const chunk1 = std::accumulate(
        chunked.begin() + static_cast<std::ptrdiff_t>(plane * 32),
        chunked.begin() + static_cast<std::ptrdiff_t>(plane * 64),
        0.0);
    EXPECT_GT(std::abs(chunk0), 0.0);
    EXPECT_GT(std::abs(chunk1), 0.0);
    EXPECT_NEAR(chunk0 + chunk1, analytic, 1e-14 * std::abs(analytic));
    // Every node of the full volume is deposited once: the concatenation equals
    // the in-core density bit for bit.
    for(std::size_t i = 0; i < reference.size(); ++i)
        EXPECT_EQ(reference[i], chunked[i]);
}
