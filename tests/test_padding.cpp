// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group 6: padding control.  The padded result must equal a direct, non-cyclic
// convolution evaluated independently in this file.  That direct reference uses
// its own TSC shape and its own Coulomb field, so a wrong 1/r, a wrong sign, a
// wrong deposit weight or an unnoticed periodic image all fail the test.  A
// converged padding-2 result alone would only prove self-consistency.
//
// A no-padding grid (fftPadding = 1) leaves no deposit halo and is rejected.
// The guard is tested here and in Deposit.PaddedBoundaryParticleLandsInFullSupport.
//
// Bunch: 27 charges on an asymmetric 3x3x3 lattice spanning about 20 cells,
// all at rest.  With fftPadding = 2 every source-observer lag fits inside the
// box and the positive/negative deposit halos stay disjoint.

#include "picBunch/Compute.hpp"
#include "picBunch/Types.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
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

    double maxField(picBunch::Field const& field)
    {
        double result = 0.0;
        for(auto const& component : field.data)
            for(double value : component)
                result = std::max(result, std::abs(value));
        return result;
    }
} // namespace

TEST(Padding, PaddingTwoMatchesDirectSumAndPaddingThreeIsConverged)
{
    picBunch::Bunch const bunch = latticeBunch();
    picBunch::ComputeOptions options;
    options.tolerance = 1e-3;
    options.cellSize = {1.0, 1.0, 1.0};
    options.marginCells = 6.0;

    options.fftPadding = 2;
    picBunch::Field const padded = picBunch::computeField(bunch, options);
    options.fftPadding = 3;
    picBunch::Field const paddedMore = picBunch::computeField(bunch, options);

    ASSERT_EQ(padded.grid.n, paddedMore.grid.n);
    ASSERT_EQ(padded.grid.fftPadding, 2u);
    ASSERT_EQ(paddedMore.grid.fftPadding, 3u);

    // Corners and face centres are where a periodic image would be closest.
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
    double const paddingConverged = maxDifference(padded, paddedMore) / scale;

    double directError = 0.0;
    for(auto const& node : probes)
    {
        std::size_t const flat = node[0] + n[0] * (node[1] + n[1] * node[2]);
        auto const reference = directReference(bunch, padded.grid, node);
        for(std::size_t c = 0; c < 6; ++c)
            directError = std::max(
                directError, std::abs(padded.data[c][flat] - reference[c]) / scale);
    }

    std::printf(
        "padding control: max|pad2-pad3|/max|field|=%.6e, "
        "probe max|pad2-direct|/max|field|=%.6e\n",
        paddingConverged,
        directError);

    // Padded result equals the direct free-space convolution, including the
    // sign of E and the 1/r field law, at points where an image would land.
    EXPECT_LT(directError, 1.0e-10) << "padding=2 must match the direct non-cyclic sum";

    // Padding 2 is already converged: adding more padding changes nothing
    // beyond FFT roundoff.
    EXPECT_LT(paddingConverged, 1.0e-9) << "padding=2 must be converged";
}

TEST(Padding, InsufficientPaddingIsRejectedWithAxisAndRequirement)
{
    picBunch::Bunch const bunch = latticeBunch();
    picBunch::ComputeOptions options;
    options.cellSize = {1.0, 1.0, 1.0};
    options.marginCells = 6.0;

    // fftPadding = 1 leaves no halo at all.
    options.fftPadding = 1;
    try
    {
        (void)picBunch::makeGrid(bunch, options);
        FAIL() << "padding 1 must be rejected";
    }
    catch(std::invalid_argument const& error)
    {
        std::string const message = error.what();
        EXPECT_NE(message.find("axis 0"), std::string::npos) << message;
        EXPECT_NE(message.find("(x)"), std::string::npos) << message;
        EXPECT_NE(message.find("requires at least 2"), std::string::npos) << message;
        EXPECT_NE(message.find("fftPadding=1"), std::string::npos) << message;
        EXPECT_NE(message.find("provides 0"), std::string::npos) << message;
    }

    // PQS needs four halo nodes.  With an 8x8x2 grid at padding 2 only the z
    // axis is short, so the message must name axis 2 (z), not axis 0.
    picBunch::Bunch single;
    single.resize(1);
    single.x[0] = 3.3;
    single.y[0] = 4.1;
    single.z[0] = 1.0;
    single.weight[0] = 1.0;
    single.charge[0] = 1.0;
    options.gridSize = picBunch::Idx3{8u, 8u, 2u};
    options.gridOrigin = picBunch::Vec3{0.0, 0.0, 0.0};
    options.fftPadding = 2;
    options.shape = picBunch::Shape::PQS;
    try
    {
        (void)picBunch::makeGrid(single, options);
        FAIL() << "PQS with a 2-node padding-2 axis must be rejected";
    }
    catch(std::invalid_argument const& error)
    {
        std::string const message = error.what();
        EXPECT_NE(message.find("axis 2"), std::string::npos) << message;
        EXPECT_NE(message.find("(z)"), std::string::npos) << message;
        EXPECT_NE(message.find("requires at least 4"), std::string::npos) << message;
        EXPECT_NE(message.find("fftPadding=2"), std::string::npos) << message;
    }
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

    // An explicit factor below the deposit stencil's halo requirement is
    // rejected instead of wrapping charge onto the opposite side.
    options.tolerance = 1e-3;
    options.fftPadding = 1;
    EXPECT_THROW(picBunch::makeGrid(bunch, options), std::invalid_argument);
}

// The landscape production case runs at 384^3.  Padding 1 leaves no halo and
// is rejected before it can deposit.  Padding 2 makes the padded box 768^3,
// which exceeds the replicated 2^28-node cap in makeGrid and must still raise
// the documented infeasibility error.
TEST(Padding, PaddingOneIsRejectedAndPaddingTwoHitsTheSizeGuard)
{
    picBunch::Bunch const bunch = latticeBunch();
    picBunch::ComputeOptions options;
    options.cellSize = {1.0, 1.0, 1.0};
    options.marginCells = 6.0;
    options.gridOrigin = {0.0, 0.0, 0.0};
    options.gridSize = {384u, 384u, 384u};

    options.fftPadding = 1;
    EXPECT_THROW(picBunch::makeGrid(bunch, options), std::invalid_argument)
        << "padding 1 has no deposit halo and must be rejected";

    options.fftPadding = 2;
    EXPECT_THROW(picBunch::makeGrid(bunch, options), std::runtime_error)
        << "padding 2 at 384^3 must hit the padded-node guard";

    // The default tolerance policy also selects padding 2 here.
    options.fftPadding = 0;
    options.tolerance = 1e-3;
    EXPECT_THROW(picBunch::makeGrid(bunch, options), std::runtime_error);
}

TEST(Padding, InteriorParticleFieldUnchangedBitForBit)
{
    // The guard is pure validation; it adds no arithmetic to the deposit.  A
    // legitimate interior particle must therefore reproduce the exact field of
    // the pre-change build.  The hash is over the raw IEEE-754 bytes of all six
    // components in component order.
    picBunch::Bunch bunch;
    bunch.resize(1);
    bunch.x[0] = 5.3;
    bunch.y[0] = 6.1;
    bunch.z[0] = 5.7;
    bunch.ux[0] = 0.0;
    bunch.uy[0] = 0.0;
    bunch.uz[0] = 0.0;
    bunch.weight[0] = 1.0;
    bunch.charge[0] = 1.0;

    picBunch::ComputeOptions options;
    options.tolerance = 1e-3;
    options.cellSize = {1.0, 1.0, 1.0};
    options.gridSize = picBunch::Idx3{12u, 12u, 12u};
    options.gridOrigin = picBunch::Vec3{0.0, 0.0, 0.0};
    options.fftPadding = 2;
    options.deterministicDeposit = true;

    auto hashField = [](picBunch::Field const& field) {
        std::uint64_t hash = 1469598103934665603ull;
        auto mix = [&](void const* data, std::size_t bytes) {
            auto const* bytesPtr = static_cast<unsigned char const*>(data);
            for(std::size_t i = 0; i < bytes; ++i)
            {
                hash ^= bytesPtr[i];
                hash *= 1099511628211ull;
            }
        };
        for(int c = 0; c < 6; ++c)
            mix(field.data[c].data(), field.data[c].size() * sizeof(double));
        return hash;
    };

    picBunch::Field const field = picBunch::computeField(bunch, options);
    std::uint64_t const hash = hashField(field);
#if !defined(PICBUNCH_BACKEND_CUDA) && !defined(PICBUNCH_BACKEND_HIP)
    // Captured from main before the padded-boundary guard (host/FFTW backend).
    EXPECT_EQ(hash, 0x16315e43730d10daull)
        << "an interior particle must keep the exact pre-change field";
#else
    // cuFFT and rocFFT round differently from FFTW, so a host bit pattern is
    // not portable.  The device deposit still has to stay deterministic.
    picBunch::Field const again = picBunch::computeField(bunch, options);
    EXPECT_EQ(hash, hashField(again)) << "repeated runs must stay bit-identical";
#endif
}
