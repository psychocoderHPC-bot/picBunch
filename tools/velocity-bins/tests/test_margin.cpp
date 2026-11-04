// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Empirical validation of the tolerance-derived tail margin and of the
// multipole far tail that replaces the global-domain zero fill.

#include "picBunch/Compute.hpp"
#include "picBunch/FarTail.hpp"
#include "picBunch/Types.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

namespace
{
    picBunch::Bunch denseBunch(std::size_t n, double halfWidth)
    {
        picBunch::Bunch bunch;
        std::mt19937 rng(20261102u);
        std::uniform_real_distribution<double> uniform(-halfWidth, halfWidth);
        bunch.resize(n);
        for(std::size_t p = 0; p < n; ++p)
        {
            bunch.x[p] = uniform(rng);
            bunch.y[p] = uniform(rng);
            bunch.z[p] = uniform(rng);
            bunch.ux[p] = bunch.uy[p] = bunch.uz[p] = 0.0;
            bunch.weight[p] = 1.0;
            bunch.charge[p] = -picBunch::kElementaryCharge;
        }
        return bunch;
    }

    double peakE(picBunch::Field const& field)
    {
        double peak = 0.0;
        for(int c = 0; c < 3; ++c)
            for(double value : field.data[c])
                peak = std::max(peak, std::abs(value));
        return peak;
    }

    picBunch::ComputeOptions marginOptions(double margin)
    {
        picBunch::ComputeOptions options;
        options.tolerance = 1.0e-3;
        options.cellSize = {1.0e-6, 1.0e-6, 1.0e-6};
        options.deterministicDeposit = true;
        options.marginCellsPerAxis = std::array<double, 3>{margin, margin, margin};
        return options;
    }
} // namespace

TEST(Margin, QuadrupoleBoundUsesActualMoment)
{
    // Two opposite charges on x give a pure dipole, no quadrupole.  Add a
    // quadrupole by placing equal charges on the y axis: Q = 0, p = 0, D != 0.
    picBunch::Bunch bunch;
    bunch.resize(4);
    bunch.x = {0.5, -0.5, 0.0, 0.0};
    bunch.y = {0.0, 0.0, 0.5, -0.5};
    bunch.z = {0.0, 0.0, 0.0, 0.0};
    bunch.ux = {0.0, 0.0, 0.0, 0.0};
    bunch.uy = {0.0, 0.0, 0.0, 0.0};
    bunch.uz = {0.0, 0.0, 0.0, 0.0};
    bunch.weight = {1.0, 1.0, 1.0, 1.0};
    double const e = picBunch::kElementaryCharge;
    bunch.charge = {e, e, -e, -e};

    double const peak = picBunch::kCoulombConstant * e;
    auto const margin = picBunch::deriveFieldTailMargin(
        bunch, peak, 1.0e-3, std::array<double, 3>{1.0, 1.0, 1.0});

    EXPECT_NEAR(margin.totalCharge, 0.0, 1.0e-30);
    EXPECT_NEAR(margin.dipoleMagnitude, 0.0, 1.0e-30);
    EXPECT_GT(margin.quadrupoleNorm, 0.0);
    EXPECT_LE(margin.bound, 1.0e-3 * peak);
    EXPECT_GT(margin.marginCells[0], 0.0);
}

TEST(Margin, FarTailMatchesDirectSum)
{
    double const width = 1.7e-5; // ~17 cells, the documented 20-cell box
    auto const bunch = denseBunch(4096, width);
    auto const tail = picBunch::deriveFarTail(bunch);

    // Peak from a small mesh solve; the tail check uses the same target.
    auto const options = marginOptions(4.0);
    auto const field = picBunch::computeField(bunch, options);
    double const peak = peakE(field);
    double const target = 1.0e-3 * peak;

    auto directE = [&](std::array<double, 3> const& point) {
        std::array<double, 3> result{0.0, 0.0, 0.0};
        for(std::size_t p = 0; p < bunch.size(); ++p)
        {
            std::array<double, 3> const dr{
                point[0] - bunch.x[p], point[1] - bunch.y[p], point[2] - bunch.z[p]};
            double const r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            double const r = std::sqrt(r2);
            double const scale = picBunch::kCoulombConstant * bunch.weight[p]
                * bunch.charge[p] / (r2 * r);
            for(int a = 0; a < 3; ++a)
                result[a] += scale * dr[a];
        }
        return result;
    };

    std::array<std::array<double, 3>, 7> const directions{{
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
        {1.0, 1.0, 0.0},
        {1.0, -1.0, 0.0},
        {1.0, 1.0, 1.0},
        {1.0, -1.0, 1.0},
    }};
    double worst = 0.0;
    for(int factor : {4, 6, 8})
    {
        for(auto direction : directions)
        {
            double const norm = std::sqrt(direction[0] * direction[0]
                + direction[1] * direction[1] + direction[2] * direction[2]);
            std::array<double, 3> point{};
            for(int a = 0; a < 3; ++a)
                point[a] = tail.centre[a]
                    + direction[a] / norm * static_cast<double>(factor) * tail.supportRadius;
            auto const exact = directE(point);
            auto const multipole = picBunch::evaluateFarTailE(tail, point);
            for(int a = 0; a < 3; ++a)
                worst = std::max(worst, std::abs(exact[a] - multipole[a]));
        }
    }
    std::printf(
        "far tail: R=%g cells, switch=%g cells, peak=%g V/m, worst |E_tail-E_direct|=%g V/m, "
        "target=%g V/m\n",
        tail.supportRadius / 1.0e-6, tail.switchDistance / 1.0e-6, peak, worst, target);
    EXPECT_LE(worst, target);
}

TEST(Margin, MarginBoxMatchesLargerReference)
{
    // Dense bunch in a 4-cell cube: the monopole margin stays small enough for
    // a reference box three times larger than the derived margin.
    double const width = 1.0e-6;
    auto const bunch = denseBunch(4096, width);
    auto const base = picBunch::computeField(bunch, marginOptions(4.0));
    double const peak = peakE(base);
    double const target = 1.0e-3 * peak;
    auto const margin = picBunch::deriveFieldTailMargin(
        bunch, peak, 1.0e-3, std::array<double, 3>{1.0e-6, 1.0e-6, 1.0e-6});
    double const m = margin.marginCells[0];

    auto const small = picBunch::computeField(bunch, marginOptions(m));
    auto const reference = picBunch::computeField(bunch, marginOptions(3.0 * m));

    // Both grids are centred on the bounding-box centre and have even extents,
    // so the small grid is an aligned subset of the reference grid.
    std::array<long long, 3> offset{};
    for(int a = 0; a < 3; ++a)
        offset[a] = static_cast<long long>(
            (reference.grid.origin[a] - small.grid.origin[a]) / small.grid.spacing[a]);
    // offset is negative when the reference box starts further out.

    std::array<double, 6> worst{};
    std::size_t const nx = small.grid.n[0];
    std::size_t const ny = small.grid.n[1];
    std::size_t const nz = small.grid.n[2];
    for(int c = 0; c < 6; ++c)
        for(std::size_t k = 0; k < nz; ++k)
            for(std::size_t j = 0; j < ny; ++j)
                for(std::size_t i = 0; i < nx; ++i)
                {
                    bool const boundary = i < 2 || i + 2 >= nx || j < 2 || j + 2 >= ny
                        || k < 2 || k + 2 >= nz;
                    if(!boundary)
                        continue;
                    std::size_t const rx = static_cast<std::size_t>(i - offset[0]);
                    std::size_t const ry = static_cast<std::size_t>(j - offset[1]);
                    std::size_t const rz = static_cast<std::size_t>(k - offset[2]);
                    std::size_t const rs = rx
                        + reference.grid.n[0]
                            * (ry + reference.grid.n[1] * rz);
                    std::size_t const ls = i + nx * (j + ny * k);
                    worst[c] = std::max(
                        worst[c], std::abs(small.data[c][ls] - reference.data[c][rs]));
                }

    std::printf(
        "margin box: margin=%g cells, small n=%zu^3, reference n=%zu^3, target=%g V/m\n",
        m, small.grid.n[0], reference.grid.n[0], target);
    for(int c = 0; c < 6; ++c)
    {
        std::printf("  component %d worst |small-reference| = %g V/m\n", c, worst[c]);
        EXPECT_LE(worst[c], target);
    }
}
