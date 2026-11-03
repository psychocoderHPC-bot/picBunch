// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group 1: TSC deposit.  Partition of unity, compact support (+-1.5 cells),
// and a single particle of weight 1 depositing exactly its charge.

#include "picBunch/Deposit.hpp"
#include "picBunch/Kernel.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>

namespace
{
    double total(picBunch::Bunch const& bunch, picBunch::Grid const& grid)
    {
        auto const rho = picBunch::depositTSC(bunch, grid);
        return std::accumulate(rho.begin(), rho.end(), 0.0);
    }
} // namespace

TEST(Deposit, TscPartitionOfUnityAndSupport)
{
    // Partition of unity for every particle offset within a cell.
    for(double frac : {0.0, 0.1, 0.25, 0.5, 0.75, 0.9})
    {
        double sum = 0.0;
        double firstMoment = 0.0;
        for(int m = -2; m <= 3; ++m)
        {
            double const w = picBunch::tscShape(static_cast<double>(m) - frac);
            sum += w;
            firstMoment += (static_cast<double>(m) - frac) * w;
        }
        EXPECT_NEAR(sum, 1.0, 1.0e-14) << "frac=" << frac;
        EXPECT_NEAR(firstMoment, 0.0, 1.0e-14) << "frac=" << frac;
    }

    // Symmetry and compact support: exactly zero at and beyond 1.5 cells.
    for(double x : {0.0, 0.2, 0.5, 1.0, 1.4999})
        EXPECT_DOUBLE_EQ(picBunch::tscShape(x), picBunch::tscShape(-x));
    EXPECT_DOUBLE_EQ(picBunch::tscShape(1.5), 0.0);
    EXPECT_DOUBLE_EQ(picBunch::tscShape(-1.5), 0.0);
    EXPECT_DOUBLE_EQ(picBunch::tscShape(2.0), 0.0);
}

TEST(Deposit, SingleParticleWeightOne)
{
    picBunch::Grid grid;
    grid.n = {16, 16, 16};
    grid.spacing = {1.0, 1.0, 1.0};
    grid.origin = {0.0, 0.0, 0.0};

    picBunch::Bunch bunch;
    bunch.resize(1);
    bunch.x[0] = 7.3;
    bunch.y[0] = 8.7;
    bunch.z[0] = 6.1;
    bunch.weight[0] = 1.0;
    bunch.charge[0] = 1.0;

    auto const rho = picBunch::depositTSC(bunch, grid);
    EXPECT_EQ(rho.size(), grid.size());
    EXPECT_NEAR(std::accumulate(rho.begin(), rho.end(), 0.0), 1.0, 1.0e-14);

    // Every nonzero node is within +-1.5 cells and there are at most 27 of them.
    std::size_t nonzero = 0;
    double const pos[3] = {bunch.x[0], bunch.y[0], bunch.z[0]};
    for(std::size_t k = 0; k < grid.n[2]; ++k)
        for(std::size_t j = 0; j < grid.n[1]; ++j)
            for(std::size_t i = 0; i < grid.n[0]; ++i)
            {
                std::size_t const flat = i + grid.n[0] * (j + grid.n[1] * k);
                if(rho[flat] == 0.0)
                    continue;
                ++nonzero;
                double const coord[3] = {
                    grid.origin[0] + static_cast<double>(i) * grid.spacing[0],
                    grid.origin[1] + static_cast<double>(j) * grid.spacing[1],
                    grid.origin[2] + static_cast<double>(k) * grid.spacing[2]};
                for(int a = 0; a < 3; ++a)
                    EXPECT_LE(std::abs(coord[a] - pos[a]), 1.5) << "axis " << a;
            }
    EXPECT_LE(nonzero, 27u);
    EXPECT_GT(nonzero, 0u);
}

TEST(Deposit, WeightAndChargeScaleLinearly)
{
    picBunch::Grid grid;
    grid.n = {12, 12, 12};
    grid.spacing = {1.0, 1.0, 1.0};
    grid.origin = {0.0, 0.0, 0.0};

    picBunch::Bunch bunch;
    bunch.resize(1);
    bunch.x[0] = 5.4;
    bunch.y[0] = 6.2;
    bunch.z[0] = 4.9;
    bunch.weight[0] = 2.5;
    bunch.charge[0] = -3.0;

    EXPECT_NEAR(total(bunch, grid), -7.5, 1.0e-13);
}
