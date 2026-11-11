// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group 6: padding control.  A no-padding run (fftPadding = 1) must reproduce
// the wraparound artefact, and the padded result must equal a direct, non-cyclic
// convolution evaluated independently in this file.  That direct reference uses
// its own TSC shape and its own Coulomb field, so a wrong 1/r, a wrong sign, a
// wrong deposit weight or an unnoticed periodic image all fail the test.  A
// converged padding-2 result alone would only prove self-consistency.
//
// Bunch: 27 charges on an asymmetric 3x3x3 lattice spanning about 20 cells,
// all at rest.  With fftPadding = 1 the FFT box equals the output box, so
// source-observer lags beyond half the box wrap onto the wrong side; with
// fftPadding = 2 every lag fits inside the box.

#include "picBunch/Compute.hpp"
#include "picBunch/Types.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace
{
    picBunch::Bunch latticeBunch()
    {
        picBunch::Bunch bunch;
        std::array<double, 3> const xs{10.0, 20.3, 30.0};
        std::array<double, 3> const ys{11.0, 20.7, 30.0};
        std::array<double, 3> const zs{9.0, 19.1, 29.0};
        for(std::size_t ix = 0; ix < 3; ++ix)
            for(std::size_t iy = 0; iy < 3; ++iy)
                for(std::size_t iz = 0; iz < 3; ++iz)
                {
                    std::size_t const p = bunch.size();
                    bunch.resize(p + 1);
                    bunch.x[p] = xs[ix];
                    bunch.y[p] = ys[iy];
                    bunch.z[p] = zs[iz];
                    bunch.ux[p] = 0.0;
                    bunch.uy[p] = 0.0;
                    bunch.uz[p] = 0.0;
                    // Non-uniform weights so periodic images do not cancel by
                    // lattice symmetry.
                    bunch.weight[p] = 0.5 + 0.25 * static_cast<double>((ix + 2 * iy + 3 * iz) % 4);
                    bunch.charge[p] = -picBunch::kElementaryCharge;
                }
        return bunch;
    }

    //! Independent quadratic B-spline assignment, argument in node units.
    double independentTsc(double x)
    {
        double const a = std::abs(x);
        if(a < 0.5)
            return 0.75 - a * a;
        if(a < 1.5)
        {
            double const t = 1.5 - a;
            return 0.5 * t * t;
        }
        return 0.0;
    }

    //! Independent Coulomb field of a charge at rest: E = k q R / |R|^3.
    std::array<double, 3> independentElectricField(
        std::array<double, 3> const& R,
        double q)
    {
        double const r2 = R[0] * R[0] + R[1] * R[1] + R[2] * R[2];
        double const r3 = r2 * std::sqrt(r2);
        double const f = (r3 > 0.0) ? picBunch::kCoulombConstant * q / r3 : 0.0;
        return {f * R[0], f * R[1], f * R[2]};
    }

    //! Yee offsets, written out again rather than imported from Types.hpp.
    std::array<std::array<double, 3>, 6> const kOffsets{{
        {0.5, 0.0, 0.0},
        {0.0, 0.5, 0.0},
        {0.0, 0.0, 0.5},
        {0.0, 0.5, 0.5},
        {0.5, 0.0, 0.5},
        {0.5, 0.5, 0.0},
    }};

    /*! Direct non-cyclic sum of the TSC-deposited source with the Coulomb kernel.
     *
     * The FFT result is the discrete convolution of ``rho`` with the sampled
     * point kernel.  This reference evaluates the same convolution as a plain
     * sum over all source nodes, with no transform and no periodic wrap.
     */
    std::array<double, 6> directReference(
        picBunch::Bunch const& bunch,
        picBunch::Grid const& grid,
        std::array<std::size_t, 3> const& node)
    {
        std::array<double, 6> result{};
        auto const& n = grid.n;
        for(std::size_t c = 0; c < 6; ++c)
        {
            std::array<double, 3> const observer{
                grid.origin[0] + (static_cast<double>(node[0]) + kOffsets[c][0]) * grid.spacing[0],
                grid.origin[1] + (static_cast<double>(node[1]) + kOffsets[c][1]) * grid.spacing[1],
                grid.origin[2] + (static_cast<double>(node[2]) + kOffsets[c][2]) * grid.spacing[2]};
            double sum = 0.0;
            for(std::size_t k = 0; k < n[2]; ++k)
                for(std::size_t j = 0; j < n[1]; ++j)
                    for(std::size_t i = 0; i < n[0]; ++i)
                    {
                        std::array<double, 3> const source{
                            grid.origin[0] + static_cast<double>(i) * grid.spacing[0],
                            grid.origin[1] + static_cast<double>(j) * grid.spacing[1],
                            grid.origin[2] + static_cast<double>(k) * grid.spacing[2]};
                        // Deposit this node's charge from every particle.
                        double rho = 0.0;
                        for(std::size_t p = 0; p < bunch.size(); ++p)
                        {
                            double const tx
                                = (bunch.x[p] - grid.origin[0]) / grid.spacing[0] - static_cast<double>(i);
                            double const ty
                                = (bunch.y[p] - grid.origin[1]) / grid.spacing[1] - static_cast<double>(j);
                            double const tz
                                = (bunch.z[p] - grid.origin[2]) / grid.spacing[2] - static_cast<double>(k);
                            double const weight = independentTsc(tx) * independentTsc(ty)
                                * independentTsc(tz);
                            if(weight != 0.0)
                                rho += weight * bunch.weight[p] * bunch.charge[p];
                        }
                        if(rho == 0.0)
                            continue;
                        std::array<double, 3> const R{
                            observer[0] - source[0],
                            observer[1] - source[1],
                            observer[2] - source[2]};
                        auto const E = independentElectricField(R, rho);
                        if(c < 3)
                            sum += E[c];
                        // All particles are at rest, so B is identically zero.
                    }
            result[c] = sum;
        }
        return result;
    }

    double maxDifference(picBunch::Field const& a, picBunch::Field const& b)
    {
        double result = 0.0;
        for(int c = 0; c < 6; ++c)
            for(std::size_t l = 0; l < a.data[c].size(); ++l)
                result = std::max(result, std::abs(a.data[c][l] - b.data[c][l]));
        return result;
    }

    //! Largest difference away from the outermost ``skip`` nodes per axis.
    double maxInteriorDifference(picBunch::Field const& a, picBunch::Field const& b, std::size_t skip)
    {
        double result = 0.0;
        auto const& n = a.grid.n;
        for(int c = 0; c < 6; ++c)
            for(std::size_t k = skip; k + skip < n[2]; ++k)
                for(std::size_t j = skip; j + skip < n[1]; ++j)
                    for(std::size_t i = skip; i + skip < n[0]; ++i)
                    {
                        std::size_t const l = i + n[0] * (j + n[1] * k);
                        result = std::max(result, std::abs(a.data[c][l] - b.data[c][l]));
                    }
        return result;
    }

    double maxField(picBunch::Field const& field)
    {
        double result = 0.0;
        for(auto const& component : field.data)
            for(double value : component)
                result = std::max(result, std::abs(value));
        return result;
    }
} // namespace

TEST(Padding, NoPaddingReproducesWraparoundAndPaddingMatchesDirectSum)
{
    picBunch::Bunch const bunch = latticeBunch();
    picBunch::ComputeOptions options;
    options.tolerance = 1e-3;
    options.cellSize = {1.0, 1.0, 1.0};
    options.marginCells = 6.0;

    options.fftPadding = 1;
    picBunch::Field const unpadded = picBunch::computeField(bunch, options);
    options.fftPadding = 2;
    picBunch::Field const padded = picBunch::computeField(bunch, options);
    options.fftPadding = 3;
    picBunch::Field const paddedMore = picBunch::computeField(bunch, options);

    ASSERT_EQ(unpadded.grid.n, padded.grid.n);
    ASSERT_EQ(unpadded.grid.n, paddedMore.grid.n);
    ASSERT_EQ(unpadded.grid.fftPadding, 1u);
    ASSERT_EQ(padded.grid.fftPadding, 2u);
    ASSERT_EQ(paddedMore.grid.fftPadding, 3u);

    // Corners and face centres are where a periodic image is closest, so the
    // padding-1 artefact is largest there.
    auto const& n = padded.grid.n;
    std::vector<std::array<std::size_t, 3>> const probes{
        {0, 0, 0},
        {n[0] - 1, 0, 0},
        {0, n[1] - 1, 0},
        {0, 0, n[2] - 1},
        {n[0] - 1, n[1] - 1, n[2] - 1},
        {n[0] / 2, 0, n[2] / 2},
        {n[0] - 1, n[1] / 2, n[2] / 2}};

    double const scale = maxField(padded);
    ASSERT_GT(scale, 0.0);
    double const wrapError = maxDifference(unpadded, padded) / scale;
    double const wrapInterior = maxInteriorDifference(unpadded, padded, 2u) / scale;
    double const paddingConverged = maxDifference(padded, paddedMore) / scale;

    double directError = 0.0;
    double unpaddedDirectError = 0.0;
    for(auto const& node : probes)
    {
        std::size_t const flat = node[0] + n[0] * (node[1] + n[1] * node[2]);
        auto const reference = directReference(bunch, padded.grid, node);
        for(std::size_t c = 0; c < 6; ++c)
        {
            directError = std::max(directError, std::abs(padded.data[c][flat] - reference[c]) / scale);
            unpaddedDirectError = std::max(
                unpaddedDirectError, std::abs(unpadded.data[c][flat] - reference[c]) / scale);
        }
    }

    std::printf(
        "padding control: max|pad1-pad2|/max|field|=%.6e, interior=%.6e, "
        "max|pad2-pad3|/max|field|=%.6e, "
        "probe max|pad2-direct|/max|field|=%.6e, max|pad1-direct|/max|field|=%.6e\n",
        wrapError,
        wrapInterior,
        paddingConverged,
        directError,
        unpaddedDirectError);

    // Padded result equals the direct free-space convolution, including the
    // sign of E and the 1/r field law, at points where an image would land.
    EXPECT_LT(directError, 1.0e-10) << "padding=2 must match the direct non-cyclic sum";

    // The no-padding run must visibly disagree with that same reference: this is
    // the wraparound the padding removes.
    EXPECT_GT(unpaddedDirectError, 1.0e-3) << "padding=1 must differ from the direct sum";
    EXPECT_GT(wrapError, 1.0e-3) << "padding=1 must show a wraparound artefact";
    EXPECT_LT(wrapError, 2.0e-1) << "wraparound artefact must be a bounded correction";
    EXPECT_GT(wrapInterior, 1.0e-3) << "the interior must also show the artefact";

    // Padding 2 is already converged: adding more padding changes nothing
    // beyond FFT roundoff.
    EXPECT_LT(paddingConverged, 1.0e-9) << "padding=2 must be converged";
}

TEST(Padding, ToleranceSelectsPaddingAndRejectsUnreachableTargets)
{
    picBunch::Bunch const bunch = latticeBunch();
    picBunch::ComputeOptions options;
    options.cellSize = {1.0, 1.0, 1.0};
    options.marginCells = 6.0;

    // Default target: padding 2.
    options.tolerance = 1e-3;
    EXPECT_EQ(picBunch::makeGrid(bunch, options).fftPadding, 2u);

    // Tighter than the measured padding-2 bound: padding 3.
    options.tolerance = 1e-13;
    EXPECT_EQ(picBunch::makeGrid(bunch, options).fftPadding, 3u);

    // Below the measured double-precision floor: refuse, do not under-deliver.
    options.tolerance = 1e-15;
    EXPECT_THROW(picBunch::makeGrid(bunch, options), std::runtime_error);

    // The diagnostic override still works and bypasses the policy.
    options.tolerance = 1e-3;
    options.fftPadding = 1;
    EXPECT_EQ(picBunch::makeGrid(bunch, options).fftPadding, 1u);
}

// The landscape production case runs at 384^3.  Padding 1 must yield a valid
// grid there (the FFT box equals the output box), while padding 2 makes the
// padded box 768^3, which exceeds the replicated 2^28-node cap in makeGrid and
// must still raise the documented infeasibility error.
TEST(Padding, ExplicitPaddingOneSupportsLargeGridAndPaddingTwoHitsTheGuard)
{
    picBunch::Bunch const bunch = latticeBunch();
    picBunch::ComputeOptions options;
    options.cellSize = {1.0, 1.0, 1.0};
    options.marginCells = 6.0;
    options.gridOrigin = {0.0, 0.0, 0.0};
    options.gridSize = {384u, 384u, 384u};

    options.fftPadding = 1;
    picBunch::Grid const grid = picBunch::makeGrid(bunch, options);
    EXPECT_EQ(grid.n, (picBunch::Idx3{384u, 384u, 384u}));
    EXPECT_EQ(grid.fftPadding, 1u);

    options.fftPadding = 2;
    EXPECT_THROW(picBunch::makeGrid(bunch, options), std::runtime_error)
        << "padding 2 at 384^3 must hit the padded-node guard";

    // The default tolerance policy also selects padding 2 here, so the same
    // guard applies unless the user asks for --fft-padding 1 explicitly.
    options.fftPadding = 0;
    options.tolerance = 1e-3;
    EXPECT_THROW(picBunch::makeGrid(bunch, options), std::runtime_error);
}
