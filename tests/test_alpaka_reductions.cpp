// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group: alpaka host reductions.
//
// The velocity-binning assignment/representatives and the far-tail multipoles
// used to be plain particle loops; their per-particle arithmetic now runs in
// alpaka host kernels.  The velocity-moment centre stays a serial loop because
// it is an ordered sum; its radius is an order-independent maximum folded by
// the alpaka host reduction.  The accumulations deliberately keep the
// historical particle order, so every result must be bit-identical to
// straightforward serial references and must not change with the executor's
// thread count.

#include "picBunch/FarTail.hpp"
#include "picBunch/VelocityBinning.hpp"
#include "picBunch/VelocityMoments.hpp"
#include "picBunch/detail/HostReduction.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

#if defined(_OPENMP)
#    include <omp.h>
#endif

namespace
{
    using picBunch::Bunch;

    constexpr double kElectronCharge = -1.602176634e-19;

    Bunch mixedBunch(std::size_t n, unsigned seed = 0x51a7u)
    {
        Bunch bunch;
        bunch.resize(n);
        std::mt19937 gen(seed);
        std::normal_distribution<double> gauss(0.0, 1.0);
        for(std::size_t p = 0; p < n; ++p)
        {
            double const gamma = 12.0 * (1.0 + 0.06 * gauss(gen));
            bunch.x[p] = 4.0 + 0.3 * gauss(gen);
            bunch.y[p] = 4.0 + 0.3 * gauss(gen);
            bunch.z[p] = 4.0 + 0.3 * gauss(gen);
            bunch.ux[p] = 0.15 * gamma * gauss(gen);
            bunch.uy[p] = 0.15 * gamma * gauss(gen);
            bunch.uz[p] = std::sqrt(std::max(1.0e-12, gamma * gamma - 1.0));
            bunch.weight[p] = 1.0;
            // Mixed signs keep the absolute-charge weighting active.
            bunch.charge[p] = ((p % 5u) == 0u ? -1.0 : 1.0) * kElectronCharge;
        }
        return bunch;
    }

    std::array<double, 3> serialCentre(Bunch const& bunch)
    {
        double sum[3] = {0.0, 0.0, 0.0};
        double w = 0.0;
        for(std::size_t p = 0; p < bunch.size(); ++p)
        {
            double const ap = std::abs(bunch.weight[p] * bunch.charge[p]);
            sum[0] += ap * bunch.ux[p];
            sum[1] += ap * bunch.uy[p];
            sum[2] += ap * bunch.uz[p];
            w += ap;
        }
        return {sum[0] / w, sum[1] / w, sum[2] / w};
    }

    double serialRadius(Bunch const& bunch, std::array<double, 3> const& u0)
    {
        double r = 0.0;
        for(std::size_t p = 0; p < bunch.size(); ++p)
        {
            double const d[3] = {
                bunch.ux[p] - u0[0],
                bunch.uy[p] - u0[1],
                bunch.uz[p] - u0[2]};
            r = std::max(r, std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]));
        }
        return r;
    }

    picBunch::FarTail serialFarTail(Bunch const& bunch, double switchFactor)
    {
        auto const bounds = bunch.bounds();
        std::array<double, 3> centre{};
        for(int a = 0; a < 3; ++a)
            centre[a] = 0.5 * (bounds[0][a] + bounds[1][a]);
        picBunch::FarTail tail;
        tail.centre = centre;
        double absoluteCharge = 0.0;
        std::array<double, 3> velocitySumCharge{0.0, 0.0, 0.0};
        std::array<double, 3> velocitySumAbs{0.0, 0.0, 0.0};
        for(std::size_t p = 0; p < bunch.size(); ++p)
        {
            double const q = bunch.weight[p] * bunch.charge[p];
            std::array<double, 3> const rel{
                bunch.x[p] - centre[0],
                bunch.y[p] - centre[1],
                bunch.z[p] - centre[2]};
            tail.charge += q;
            absoluteCharge += std::abs(q);
            double const r2 = rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2];
            tail.supportRadius = std::max(tail.supportRadius, std::sqrt(r2));
            double const u2 = bunch.ux[p] * bunch.ux[p] + bunch.uy[p] * bunch.uy[p]
                + bunch.uz[p] * bunch.uz[p];
            double const gamma = std::sqrt(1.0 + u2);
            tail.gammaMax = std::max(tail.gammaMax, gamma);
            double const invGamma = 1.0 / gamma;
            for(int a = 0; a < 3; ++a)
            {
                double const u = (a == 0) ? bunch.ux[p] : (a == 1) ? bunch.uy[p] : bunch.uz[p];
                double const v = picBunch::kSpeedOfLight * u * invGamma;
                velocitySumCharge[a] += q * v;
                velocitySumAbs[a] += std::abs(q) * v;
            }
        }
        bool const neutral
            = (absoluteCharge == 0.0)
            || (std::abs(tail.charge) <= 1.0e-12 * absoluteCharge);
        std::array<double, 3> const& velocitySum = neutral ? velocitySumAbs : velocitySumCharge;
        double const velocityWeight = neutral ? absoluteCharge : tail.charge;
        if(velocityWeight != 0.0)
            for(int a = 0; a < 3; ++a)
                tail.meanVelocity[a] = velocitySum[a] / velocityWeight;

        double const meanSpeed2 = tail.meanVelocity[0] * tail.meanVelocity[0]
            + tail.meanVelocity[1] * tail.meanVelocity[1] + tail.meanVelocity[2] * tail.meanVelocity[2];
        double const meanGamma2
            = 1.0 / std::max(1.0 - meanSpeed2 / (picBunch::kSpeedOfLight * picBunch::kSpeedOfLight), 1.0e-300);
        double const meanGamma = std::sqrt(meanGamma2);
        double const meanSpeed = std::sqrt(meanSpeed2);
        std::array<double, 3> const nhat{
            (meanSpeed > 0.0) ? tail.meanVelocity[0] / meanSpeed : 0.0,
            (meanSpeed > 0.0) ? tail.meanVelocity[1] / meanSpeed : 0.0,
            (meanSpeed > 0.0) ? tail.meanVelocity[2] / meanSpeed : 0.0};
        for(std::size_t p = 0; p < bunch.size(); ++p)
        {
            double const q = bunch.weight[p] * bunch.charge[p];
            std::array<double, 3> const rel{
                bunch.x[p] - centre[0],
                bunch.y[p] - centre[1],
                bunch.z[p] - centre[2]};
            double const rn = rel[0] * nhat[0] + rel[1] * nhat[1] + rel[2] * nhat[2];
            std::array<double, 3> rest{};
            for(int a = 0; a < 3; ++a)
                rest[a] = rel[a] + (meanGamma - 1.0) * rn * nhat[a];
            double const r2 = rest[0] * rest[0] + rest[1] * rest[1] + rest[2] * rest[2];
            for(int a = 0; a < 3; ++a)
                tail.dipole[a] += q * rest[a];
            for(int i = 0; i < 3; ++i)
                for(int j = 0; j < 3; ++j)
                    tail.quadrupole[i * 3 + j]
                        += q * (3.0 * rest[i] * rest[j] - (i == j ? r2 : 0.0));
        }
        tail.switchDistance = switchFactor * tail.supportRadius;
        return tail;
    }

    void expectFarTailEqual(picBunch::FarTail const& a, picBunch::FarTail const& b)
    {
        for(int i = 0; i < 3; ++i)
        {
            EXPECT_DOUBLE_EQ(a.centre[i], b.centre[i]);
            EXPECT_DOUBLE_EQ(a.dipole[i], b.dipole[i]);
            EXPECT_DOUBLE_EQ(a.meanVelocity[i], b.meanVelocity[i]);
        }
        for(int i = 0; i < 9; ++i)
            EXPECT_DOUBLE_EQ(a.quadrupole[i], b.quadrupole[i]);
        EXPECT_DOUBLE_EQ(a.charge, b.charge);
        EXPECT_DOUBLE_EQ(a.supportRadius, b.supportRadius);
        EXPECT_DOUBLE_EQ(a.switchDistance, b.switchDistance);
        EXPECT_DOUBLE_EQ(a.gammaMax, b.gammaMax);
    }

    using ReductionResults = std::tuple<
        std::array<double, 3>,
        double,
        picBunch::FarTail,
        picBunch::VelocityBinning>;

    void expectReductionResultsEqual(ReductionResults const& reference, ReductionResults const& candidate)
    {
        for(int a = 0; a < 3; ++a)
            EXPECT_DOUBLE_EQ(std::get<0>(reference)[a], std::get<0>(candidate)[a]) << "axis " << a;
        EXPECT_DOUBLE_EQ(std::get<1>(reference), std::get<1>(candidate));
        expectFarTailEqual(std::get<2>(reference), std::get<2>(candidate));
        auto const& a = std::get<3>(reference);
        auto const& b = std::get<3>(candidate);
        ASSERT_EQ(a.count, b.count);
        EXPECT_EQ(a.perAxis, b.perAxis);
        EXPECT_EQ(a.boxOf, b.boxOf);
        EXPECT_DOUBLE_EQ(a.modelError, b.modelError);
        for(std::size_t i = 0; i < a.representativeU.size(); ++i)
            for(int c = 0; c < 3; ++c)
                EXPECT_DOUBLE_EQ(a.representativeU[i][c], b.representativeU[i][c]);
    }
} // namespace

TEST(AlpakaReductions, MomentsMatchSerialReference)
{
    Bunch const bunch = mixedBunch(2000u);
    auto const centre = picBunch::velocityMomentsCentre(bunch);
    auto const reference = serialCentre(bunch);
    for(int a = 0; a < 3; ++a)
        EXPECT_DOUBLE_EQ(centre[a], reference[a]) << "axis " << a;
    EXPECT_DOUBLE_EQ(picBunch::velocityMomentsRadius(bunch, centre), serialRadius(bunch, reference));
}

TEST(AlpakaReductions, BoundsMatchSerialReference)
{
    Bunch const bunch = mixedBunch(5000u, 0x2b17u);
    std::array<std::array<double, 3>, 2> reference{};
    reference[0].fill(std::numeric_limits<double>::max());
    reference[1].fill(-std::numeric_limits<double>::max());
    for(std::size_t p = 0; p < bunch.size(); ++p)
    {
        double const v[3] = {bunch.x[p], bunch.y[p], bunch.z[p]};
        for(int a = 0; a < 3; ++a)
        {
            reference[0][a] = std::min(reference[0][a], v[a]);
            reference[1][a] = std::max(reference[1][a], v[a]);
        }
    }

    auto const bits = [](double v) { return std::bit_cast<std::uint64_t>(v); };
    auto const bounds = bunch.bounds();
    for(int a = 0; a < 3; ++a)
    {
        EXPECT_EQ(bits(bounds[0][a]), bits(reference[0][a])) << "min axis " << a;
        EXPECT_EQ(bits(bounds[1][a]), bits(reference[1][a])) << "max axis " << a;
    }

    // The min/max fold is order insensitive, so the executor choice and a
    // repeated call must reproduce the same bits.
    using picBunch::detail::HostReductionExecutor;
    auto& executorOverride = picBunch::detail::hostReductionExecutorOverride();
    executorOverride = HostReductionExecutor::Serial;
    auto const serial = bunch.bounds();
    executorOverride = HostReductionExecutor::Parallel;
    auto const parallel = bunch.bounds();
    executorOverride = HostReductionExecutor::Default;
    auto const repeated = bunch.bounds();
    for(int a = 0; a < 3; ++a)
    {
        EXPECT_EQ(bits(serial[0][a]), bits(parallel[0][a]));
        EXPECT_EQ(bits(serial[1][a]), bits(parallel[1][a]));
        EXPECT_EQ(bits(serial[0][a]), bits(repeated[0][a]));
        EXPECT_EQ(bits(serial[1][a]), bits(repeated[1][a]));
    }
}

TEST(AlpakaReductions, FarTailMatchesSerialReference)
{
    Bunch const bunch = mixedBunch(4000u);
    auto const tail = picBunch::deriveFarTail(bunch, 6.0);
    expectFarTailEqual(tail, serialFarTail(bunch, 6.0));
    EXPECT_TRUE(tail.groups.empty());
}

TEST(AlpakaReductions, BinningMatchesSerialReference)
{
    Bunch const bunch = mixedBunch(3000u, 0x9e11u);
    picBunch::VelocityBinningOptions options;
    options.mode = picBunch::VelocityBinningOptions::Mode::Explicit;
    options.explicitBins = 16u;
    auto const binning = picBunch::deriveVelocityBinning(bunch, 1.0e-3, options);
    ASSERT_EQ(binning.boxOf.size(), bunch.size());
    ASSERT_EQ(binning.representativeU.size(), binning.count);

    // Independent serial reference: project on the reported basis, reuse the
    // public boxIndex, and accumulate the box means in particle order.
    std::vector<std::size_t> expectedBox(bunch.size());
    std::vector<double> absCharge(bunch.size());
    for(std::size_t p = 0; p < bunch.size(); ++p)
    {
        double const u[3] = {bunch.ux[p], bunch.uy[p], bunch.uz[p]};
        double const b1 = u[0] * binning.basis[0][0] + u[1] * binning.basis[0][1] + u[2] * binning.basis[0][2];
        double const b2 = u[0] * binning.basis[1][0] + u[1] * binning.basis[1][1] + u[2] * binning.basis[1][2];
        double const b3 = u[0] * binning.basis[2][0] + u[1] * binning.basis[2][1] + u[2] * binning.basis[2][2];
        expectedBox[p] = binning.boxIndex(b1, b2, b3);
        absCharge[p] = std::abs(bunch.weight[p] * bunch.charge[p]);
        EXPECT_EQ(binning.boxOf[p], expectedBox[p]) << "particle " << p;
    }
    std::vector<std::array<double, 3>> sumU(binning.count);
    std::vector<double> sumW(binning.count, 0.0);
    for(std::size_t p = 0; p < bunch.size(); ++p)
    {
        std::size_t const b = expectedBox[p];
        sumU[b][0] += absCharge[p] * bunch.ux[p];
        sumU[b][1] += absCharge[p] * bunch.uy[p];
        sumU[b][2] += absCharge[p] * bunch.uz[p];
        sumW[b] += absCharge[p];
    }
    for(std::size_t b = 0; b < binning.count; ++b)
    {
        if(sumW[b] == 0.0)
            continue;
        for(int a = 0; a < 3; ++a)
            EXPECT_DOUBLE_EQ(binning.representativeU[b][a], sumU[b][a] / sumW[b]) << "box " << b;
    }
}

TEST(AlpakaReductions, ResultsAreThreadCountIndependent)
{
    Bunch const bunch = mixedBunch(1500u, 0x77u);
    picBunch::VelocityBinningOptions options;
    options.mode = picBunch::VelocityBinningOptions::Mode::Explicit;
    options.explicitBins = 8u;

    auto const run = [&] {
        auto const centre = picBunch::velocityMomentsCentre(bunch);
        return ReductionResults{
            centre,
            picBunch::velocityMomentsRadius(bunch, centre),
            picBunch::deriveFarTail(bunch, 6.0),
            picBunch::deriveVelocityBinning(bunch, 1.0e-3, options)};
    };

    // Same production kernels under the explicit serial and the parallel
    // executor: the per-particle results and the ordered accumulations must be
    // bit-identical, so the executor choice cannot change the physics.
    using picBunch::detail::HostReductionExecutor;
    auto& executorOverride = picBunch::detail::hostReductionExecutorOverride();
    executorOverride = HostReductionExecutor::Serial;
    auto const serialExecutor = run();
    executorOverride = HostReductionExecutor::Parallel;
    auto const parallelExecutor = run();
    executorOverride = HostReductionExecutor::Default;
    expectReductionResultsEqual(serialExecutor, parallelExecutor);

    // The production default must also not follow the OpenMP thread count.
#if defined(_OPENMP)
    int const savedThreads = omp_get_max_threads();
    omp_set_num_threads(1);
    auto const single = run();
    omp_set_num_threads(std::max(2, savedThreads));
    auto const many = run();
    omp_set_num_threads(savedThreads);
#else
    auto const single = run();
    auto const many = run();
#endif
    expectReductionResultsEqual(single, many);
}
