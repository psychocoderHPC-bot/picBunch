// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group 1: TSC deposit.  Partition of unity, compact support (+-1.5 cells),
// and a single particle of weight 1 depositing exactly its charge.

#include "picBunch/Compute.hpp"
#include "picBunch/Deposit.hpp"
#include "picBunch/Kernel.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
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

TEST(Deposit, PaddedDepositConservesCharge)
{
    // The padded host deposit must deposit exactly the particle charge: the TSC
    // shape is sampled in node units and the grid spacing already carries the
    // node spacing.  Before the fix the weights were evaluated as
    // tscShape(refinement*(ia-t)) with a support shrunk by 1/refinement while
    // the grid spacing already contained the refinement, so the sum dropped
    // (e.g. to 0.421875) instead of 1.  The assertion below fails on that code
    // and on any clipped boundary support.
    picBunch::Grid grid;
    grid.n = {16, 16, 16};
    grid.spacing = {1.0, 1.0, 1.0};
    grid.origin = {0.0, 0.0, 0.0};
    grid.fftPadding = 2;

    std::array<std::array<double, 3>, 3> const positions{{
        {8.0, 8.0, 8.0},   // on a node
        {8.5, 8.5, 8.5},   // cell centre
        {8.3, 7.7, 8.9},   // sub-cell
    }};
    char const* const label[3] = {"node", "cell-centre", "sub-cell"};

    for(std::size_t p = 0; p < positions.size(); ++p)
    {
        picBunch::Bunch bunch;
        bunch.resize(1);
        bunch.x[0] = positions[p][0];
        bunch.y[0] = positions[p][1];
        bunch.z[0] = positions[p][2];
        bunch.weight[0] = 1.0;
        bunch.charge[0] = 1.0;

        auto const rho = picBunch::depositTSCPaddedHost(bunch, grid);
        double const sum = std::accumulate(rho.begin(), rho.end(), 0.0);
        EXPECT_NEAR(sum, 1.0, 1.0e-12) << "position=" << label[p];
    }
}

TEST(Deposit, PaddedSourceHaloConservesChargeAtBoundaries)
{
    // Regression for P1-2: a particle whose TSC support crosses the physical
    // 0..n-1 boundary must still deposit its full charge.  The support outside
    // the physical box belongs in the padded FFT extent (negative logical nodes
    // wrap to M-1.., nodes >= n land in the upper padding).  The pre-fix code
    // clipped at the physical boundary, so a particle at node (0,0,0)
    // deposited only 0.669921875.
    picBunch::Grid grid;
    grid.n = {8, 8, 8};
    grid.spacing = {1.0, 1.0, 1.0};
    grid.origin = {0.0, 0.0, 0.0};
    grid.fftPadding = 2;

    // Each coordinate is either a face node (0 or 7) or an interior node (3).
    // The 27 combinations cover every corner, edge, face and the interior.
    for(std::size_t kk = 0; kk < 3; ++kk)
        for(std::size_t jj = 0; jj < 3; ++jj)
            for(std::size_t ii = 0; ii < 3; ++ii)
            {
                double const coord[3] = {ii == 0 ? 0.0 : (ii == 1 ? 3.0 : 7.0),
                                         jj == 0 ? 0.0 : (jj == 1 ? 3.0 : 7.0),
                                         kk == 0 ? 0.0 : (kk == 1 ? 3.0 : 7.0)};
                picBunch::Bunch bunch;
                bunch.resize(1);
                bunch.x[0] = coord[0];
                bunch.y[0] = coord[1];
                bunch.z[0] = coord[2];
                bunch.weight[0] = 1.0;
                bunch.charge[0] = 1.0;

                auto const rho = picBunch::depositTSCPaddedHost(bunch, grid);
                EXPECT_EQ(rho.size(), 16u * 16u * 16u);
                double const sum = std::accumulate(rho.begin(), rho.end(), 0.0);
                EXPECT_NEAR(sum, 1.0, 1.0e-14)
                    << "node (" << coord[0] << "," << coord[1] << "," << coord[2] << ")";
            }
}

TEST(Deposit, PaddedSourceHaloWrapsNegativeSupport)
{
    // A particle exactly on node zero has support at logical node -1.  With the
    // halo that weight must appear in the upper padded plane, not be dropped.
    picBunch::Grid grid;
    grid.n = {8, 8, 8};
    grid.spacing = {1.0, 1.0, 1.0};
    grid.origin = {0.0, 0.0, 0.0};
    grid.fftPadding = 2;

    picBunch::Bunch bunch;
    bunch.resize(1);
    bunch.x[0] = 0.0;
    bunch.y[0] = 3.0;
    bunch.z[0] = 3.0;
    bunch.weight[0] = 1.0;
    bunch.charge[0] = 1.0;

    auto const rho = picBunch::depositTSCPaddedHost(bunch, grid);
    std::size_t const M = 8u * 2u;
    // Logical (ia,ib,ic) maps to (ia+M)%M etc.  The -1 support is at x=M-1.
    std::size_t const idx = static_cast<std::size_t>(M - 1u)
        + M * (static_cast<std::size_t>(3u) + M * static_cast<std::size_t>(3u));
    // tscShape(-1) * tscShape(0) * tscShape(0) = 0.125 * 0.75 * 0.75.
    EXPECT_NEAR(rho[idx], 0.0703125, 1.0e-14);
}

TEST(Deposit, ShapeChargeConservationAndSupport)
{
    for(picBunch::Shape shape : {picBunch::Shape::CIC, picBunch::Shape::TSC, picBunch::Shape::PQS})
    {
        picBunch::Bunch bunch;
        bunch.resize(1);
        bunch.x = {0.0};
        bunch.y = {0.0};
        bunch.z = {0.0};
        bunch.ux = {0.0};
        bunch.uy = {0.0};
        bunch.uz = {0.0};
        bunch.weight = {1.0};
        bunch.charge = {1.0};

        picBunch::Grid grid;
        grid.n = {16, 16, 16};
        grid.spacing = {1.0, 1.0, 1.0};
        grid.origin = {-8.0, -8.0, -8.0};
        grid.fftPadding = 2u;
        grid.shape = shape;

        auto const rho = picBunch::depositTSC(bunch, grid);
        double sum = 0.0;
        double maxRadius = 0.0;
        for(std::size_t k = 0; k < 16u; ++k)
            for(std::size_t j = 0; j < 16u; ++j)
                for(std::size_t i = 0; i < 16u; ++i)
                {
                    double const value = rho[i + 16u * (j + 16u * k)];
                    sum += value;
                    if(value != 0.0)
                    {
                        double const dx = static_cast<double>(i) - 8.0;
                        double const dy = static_cast<double>(j) - 8.0;
                        double const dz = static_cast<double>(k) - 8.0;
                        maxRadius = std::max(
                            maxRadius, std::sqrt(dx * dx + dy * dy + dz * dz));
                    }
                }
        EXPECT_NEAR(sum, 1.0, 1.0e-12) << "shape " << static_cast<int>(shape);
        EXPECT_LE(maxRadius, picBunch::shapeSupport(shape) * std::sqrt(3.0) + 1.0e-12)
            << "shape " << static_cast<int>(shape);
    }
}

TEST(Deposit, NearShellRejectsNonTscShape)
{
    picBunch::Bunch bunch;
    bunch.resize(1);
    bunch.x = {0.0};
    bunch.y = {0.0};
    bunch.z = {0.0};
    bunch.ux = {0.0};
    bunch.uy = {0.0};
    bunch.uz = {0.0};
    bunch.weight = {1.0};
    bunch.charge = {-picBunch::kElementaryCharge};

    picBunch::ComputeOptions options;
    options.cellSize = {1.0, 1.0, 1.0};
    options.gridSize = {8, 8, 8};
    options.gridOrigin = {-4.0, -4.0, -4.0};
    options.shape = picBunch::Shape::PQS;
    options.nearShell.mode = picBunch::NearShellOptions::Mode::Fast;
    options.nearShell.radiusCells = 2.0;
    EXPECT_THROW(picBunch::computeField(bunch, options), std::runtime_error);
}

TEST(Deposit, ShapesProduceDifferentStencils)
{
    picBunch::Bunch bunch;
    bunch.resize(1);
    bunch.x = {0.3};
    bunch.y = {0.4};
    bunch.z = {0.5};
    bunch.ux = {0.0};
    bunch.uy = {0.0};
    bunch.uz = {0.0};
    bunch.weight = {1.0};
    bunch.charge = {1.0};
    picBunch::Grid grid;
    grid.n = {16, 16, 16};
    grid.spacing = {1.0, 1.0, 1.0};
    grid.origin = {-8.0, -8.0, -8.0};
    grid.shape = picBunch::Shape::TSC;
    auto const tsc = picBunch::depositTSC(bunch, grid);
    grid.shape = picBunch::Shape::PQS;
    auto const pqs = picBunch::depositTSC(bunch, grid);
    double maxDiff = 0.0;
    for(std::size_t l = 0; l < tsc.size(); ++l)
        maxDiff = std::max(maxDiff, std::abs(tsc[l] - pqs[l]));
    EXPECT_GT(maxDiff, 1.0e-9) << "the deposit must use the selected shape";
}
