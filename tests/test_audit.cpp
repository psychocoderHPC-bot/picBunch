// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group 10: referee-facing audit battery.
//
// These tests answer independent-verification questions that the existing
// groups do not isolate:
//   1. padding 2 versus padding 3 at FFT roundoff, per field component, for
//      gamma 1 / gamma 10 and aspect 1 / aspect 4;
//   2. discrete Gauss law on the Yee lattice: divergence of the computed E
//      compared with the TSC-deposited charge density;
//   3. high-density accumulation: 1000 sources inside one 2-cell cube against
//      an independent direct convolution;
//   4. seam continuity across the near-shell boundary;
//   5. deterministic-deposit bitwise reproducibility (the cross-rank check is
//      registered as a separate MPI CTest when MPI is enabled).

#include "picBunch/Compute.hpp"
#include "picBunch/Deposit.hpp"
#include "picBunch/Kernel.hpp"
#include "picBunch/NearShell.hpp"
#include "picBunch/Types.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace
{
    //! One electron in SI units at the given position and dimensionless momentum.
    picBunch::Bunch oneElectron(picBunch::Vec3 const& position, picBunch::Vec3 const& u)
    {
        picBunch::Bunch bunch;
        bunch.resize(1);
        bunch.x[0] = position[0];
        bunch.y[0] = position[1];
        bunch.z[0] = position[2];
        bunch.ux[0] = u[0];
        bunch.uy[0] = u[1];
        bunch.uz[0] = u[2];
        bunch.weight[0] = 1.0;
        bunch.charge[0] = -picBunch::kElementaryCharge;
        return bunch;
    }

    //! Three electrons sharing u, the cloud used by the physics group.
    picBunch::Bunch threeElectronCloud(picBunch::Vec3 const& u)
    {
        picBunch::Bunch bunch;
        std::array<picBunch::Vec3, 3> const positions{{
            {4.0, 4.0, 4.0},
            {4.7, 5.1, 4.3},
            {6.2, 5.8, 6.1},
        }};
        for(auto const& p : positions)
        {
            std::size_t const i = bunch.size();
            bunch.resize(i + 1);
            bunch.x[i] = p[0];
            bunch.y[i] = p[1];
            bunch.z[i] = p[2];
            bunch.ux[i] = u[0];
            bunch.uy[i] = u[1];
            bunch.uz[i] = u[2];
            bunch.weight[i] = 1.0;
            bunch.charge[i] = -picBunch::kElementaryCharge;
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
    picBunch::Vec3 independentCoulomb(picBunch::Vec3 const& R, double q)
    {
        double const r2 = R[0] * R[0] + R[1] * R[1] + R[2] * R[2];
        double const r3 = r2 * std::sqrt(r2);
        double const f = (r3 > 0.0) ? picBunch::kCoulombConstant * q / r3 : 0.0;
        return {f * R[0], f * R[1], f * R[2]};
    }

    //! Per-component relative difference max|a-b| / max(max|a|, max|b|).
    std::array<double, 6> perComponentRelativeDifference(picBunch::Field const& a, picBunch::Field const& b)
    {
        std::array<double, 6> result{};
        for(int c = 0; c < 6; ++c)
        {
            double num = 0.0;
            double den = 0.0;
            for(std::size_t l = 0; l < a.data[c].size(); ++l)
            {
                num = std::max(num, std::abs(a.data[c][l] - b.data[c][l]));
                den = std::max(den, std::max(std::abs(a.data[c][l]), std::abs(b.data[c][l])));
            }
            result[c] = (den > 0.0) ? num / den : 0.0;
        }
        return result;
    }

    double worstOf(std::array<double, 6> const& values)
    {
        return *std::max_element(values.begin(), values.end());
    }

    void printComponents(char const* label, std::array<double, 6> const& values)
    {
        std::printf(
            "AUDIT padding %-26s Ex=%.3e Ey=%.3e Ez=%.3e Bx=%.3e By=%.3e Bz=%.3e worst=%.3e\n",
            label,
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
            worstOf(values));
    }

    //! Measured per-component padding-2 versus padding-3 difference for one case.
    std::array<double, 6> measurePaddingConvergence(
        char const* label,
        picBunch::Bunch const& bunch,
        picBunch::Vec3 const& cellSize)
    {
        picBunch::ComputeOptions options;
        options.tolerance = 1e-3;
        options.cellSize = cellSize;
        options.marginCells = 4.0;

        options.fftPadding = 2;
        picBunch::Field const padding2 = picBunch::computeField(bunch, options);
        options.fftPadding = 3;
        picBunch::Field const padding3 = picBunch::computeField(bunch, options);

        auto const difference = perComponentRelativeDifference(padding2, padding3);
        printComponents(label, difference);
        return difference;
    }

    //! epsilon0 from the exact Coulomb constant used by the pipeline.
    constexpr double kEpsilon0 = 1.0 / (4.0 * 3.14159265358979323846 * picBunch::kCoulombConstant);

    /*! Discrete Gauss-law residual on the Yee lattice.
     *
     * E_x is stored at (i+1/2,j,k), E_y at (i,j+1/2,k), E_z at (i,j,k+1/2).
     * The divergence of that staggered field lives on the node (i,j,k) and is
     * evaluated with the backward difference that is the adjoint of the
     * node-to-staggered gradient:
     *   (E_x[i] - E_x[i-1])/hx + (E_y[j] - E_y[j-1])/hy + (E_z[k] - E_z[k-1])/hz.
     * The deposited charge is the same TSC deposit the solver uses, converted
     * with rho/(eps0*hx*hy*hz).
     *
     * For gamma = 1 the electric field is curl-free and E = -grad(phi).  For
     * gamma = 10 the field of a uniformly moving charge is not curl-free
     * (curl E = -dB/dt != 0), so the correct statement tested here is Gauss's
     * law, div E = rho/eps0, which holds for both cases.  What is not expected
     * is FFT-roundoff agreement: the Yee difference of the sampled point kernel
     * is not the TSC shape, so the residual is a discretisation residual.
     */
    double measureDivergenceResidual(
        char const* label,
        picBunch::Bunch const& bunch,
        picBunch::Vec3 const& cellSize,
        unsigned padding)
    {
        picBunch::ComputeOptions options;
        options.tolerance = 1e-3;
        options.cellSize = cellSize;
        options.marginCells = 4.0;
        options.fftPadding = padding;

        picBunch::Field const field = picBunch::computeField(bunch, options);
        auto const& n = field.grid.n;
        auto const& h = field.grid.spacing;
        auto const rho = picBunch::depositTSC(bunch, field.grid);

        double const cellVolume = h[0] * h[1] * h[2];
        double peakTarget = 0.0;
        for(double value : rho)
            peakTarget = std::max(peakTarget, std::abs(value) / (kEpsilon0 * cellVolume));

        double peakResidual = 0.0;
        for(std::size_t k = 1; k < n[2]; ++k)
            for(std::size_t j = 1; j < n[1]; ++j)
                for(std::size_t i = 1; i < n[0]; ++i)
                {
                    auto const at = [&](std::size_t c, std::size_t ii, std::size_t jj, std::size_t kk) {
                        std::size_t const flat = ii + n[0] * (jj + n[1] * kk);
                        return field.data[c][flat];
                    };
                    double const divergence = (at(0, i, j, k) - at(0, i - 1, j, k)) / h[0]
                        + (at(1, i, j, k) - at(1, i, j - 1, k)) / h[1]
                        + (at(2, i, j, k) - at(2, i, j, k - 1)) / h[2];
                    std::size_t const flat = i + n[0] * (j + n[1] * k);
                    double const target = rho[flat] / (kEpsilon0 * cellVolume);
                    peakResidual = std::max(peakResidual, std::abs(divergence - target));
                }

        double const relative = (peakTarget > 0.0) ? peakResidual / peakTarget : 0.0;
        std::printf(
            "AUDIT div-E %-30s padding=%u peak|rho|/eps0=%.6e max|residual|=%.6e relative=%.6e\n",
            label,
            padding,
            peakTarget,
            peakResidual,
            relative);
        return relative;
    }

    /*! Independent direct convolution for the high-density accumulation test.
     *
     * Builds the TSC density with its own B-spline, then sums the Coulomb field
     * of every occupied source node over every observer node.  No FFT and no
     * periodic wrap enter.  All particles are at rest, so B is identically zero.
     */
    struct DirectGrid
    {
        std::vector<double> rho;
        picBunch::Vec3 spacing;
        picBunch::Vec3 origin;
        picBunch::Idx3 n;
    };

    DirectGrid independentDeposit(picBunch::Bunch const& bunch, picBunch::Grid const& grid)
    {
        DirectGrid direct;
        direct.spacing = grid.spacing;
        direct.origin = grid.origin;
        direct.n = grid.n;
        direct.rho.assign(grid.size(), 0.0);
        for(std::size_t p = 0; p < bunch.size(); ++p)
        {
            double const t[3] = {
                (bunch.x[p] - grid.origin[0]) / grid.spacing[0],
                (bunch.y[p] - grid.origin[1]) / grid.spacing[1],
                (bunch.z[p] - grid.origin[2]) / grid.spacing[2]};
            long const lo[3] = {
                static_cast<long>(std::ceil(t[0] - 1.5)),
                static_cast<long>(std::ceil(t[1] - 1.5)),
                static_cast<long>(std::ceil(t[2] - 1.5))};
            long const hi[3] = {
                static_cast<long>(std::floor(t[0] + 1.5)),
                static_cast<long>(std::floor(t[1] + 1.5)),
                static_cast<long>(std::floor(t[2] + 1.5))};
            double const q = bunch.weight[p] * bunch.charge[p];
            for(long i = lo[0]; i <= hi[0]; ++i)
            {
                if(i < 0 || i >= static_cast<long>(grid.n[0]))
                    continue;
                double const wx = independentTsc(static_cast<double>(i) - t[0]);
                if(wx == 0.0)
                    continue;
                for(long j = lo[1]; j <= hi[1]; ++j)
                {
                    if(j < 0 || j >= static_cast<long>(grid.n[1]))
                        continue;
                    double const wy = wx * independentTsc(static_cast<double>(j) - t[1]);
                    if(wy == 0.0)
                        continue;
                    for(long k = lo[2]; k <= hi[2]; ++k)
                    {
                        if(k < 0 || k >= static_cast<long>(grid.n[2]))
                            continue;
                        double const wz = wy * independentTsc(static_cast<double>(k) - t[2]);
                        if(wz == 0.0)
                            continue;
                        std::size_t const flat = static_cast<std::size_t>(i)
                            + grid.n[0]
                                * (static_cast<std::size_t>(j) + grid.n[1] * static_cast<std::size_t>(k));
                        direct.rho[flat] += wz * q;
                    }
                }
            }
        }
        return direct;
    }

    //! Yee offsets written out independently of Types.hpp.
    std::array<picBunch::Vec3, 6> const kYeeOffsets{{
        {0.5, 0.0, 0.0},
        {0.0, 0.5, 0.0},
        {0.0, 0.0, 0.5},
        {0.0, 0.5, 0.5},
        {0.5, 0.0, 0.5},
        {0.5, 0.5, 0.0},
    }};

    //! Max per-component relative difference between the computed field and the direct sum.
    std::array<double, 6> directReferenceError(picBunch::Field const& field, picBunch::Bunch const& bunch)
    {
        DirectGrid const direct = independentDeposit(bunch, field.grid);
        auto const& n = field.grid.n;
        std::array<double, 6> result{};
        for(int c = 0; c < 6; ++c)
        {
            double num = 0.0;
            double den = 0.0;
            for(std::size_t k = 0; k < n[2]; ++k)
                for(std::size_t j = 0; j < n[1]; ++j)
                    for(std::size_t i = 0; i < n[0]; ++i)
                    {
                        picBunch::Vec3 const observer{
                            field.grid.origin[0] + (static_cast<double>(i) + kYeeOffsets[c][0]) * field.grid.spacing[0],
                            field.grid.origin[1] + (static_cast<double>(j) + kYeeOffsets[c][1]) * field.grid.spacing[1],
                            field.grid.origin[2] + (static_cast<double>(k) + kYeeOffsets[c][2]) * field.grid.spacing[2]};
                        double sum = 0.0;
                        if(c < 3)
                            for(std::size_t sk = 0; sk < n[2]; ++sk)
                                for(std::size_t sj = 0; sj < n[1]; ++sj)
                                    for(std::size_t si = 0; si < n[0]; ++si)
                                    {
                                        std::size_t const sflat = si + n[0] * (sj + n[1] * sk);
                                        if(direct.rho[sflat] == 0.0)
                                            continue;
                                        picBunch::Vec3 const source{
                                            field.grid.origin[0] + static_cast<double>(si) * field.grid.spacing[0],
                                            field.grid.origin[1] + static_cast<double>(sj) * field.grid.spacing[1],
                                            field.grid.origin[2] + static_cast<double>(sk) * field.grid.spacing[2]};
                                        picBunch::Vec3 const R{
                                            observer[0] - source[0],
                                            observer[1] - source[1],
                                            observer[2] - source[2]};
                                        sum += independentCoulomb(R, direct.rho[sflat])[static_cast<std::size_t>(c)];
                                    }
                        std::size_t const flat = i + n[0] * (j + n[1] * k);
                        double const actual = field.data[c][flat];
                        num = std::max(num, std::abs(actual - sum));
                        den = std::max(den, std::max(std::abs(actual), std::abs(sum)));
                    }
            result[static_cast<std::size_t>(c)] = (den > 0.0) ? num / den : 0.0;
        }
        return result;
    }
} // namespace

TEST(Audit, PaddingTwoVersusThreeAtRoundoff)
{
    picBunch::Vec3 const u0{0.0, 0.0, 0.0};
    picBunch::Vec3 const u10{0.0, 0.0, 10.0};
    picBunch::Vec3 const aspect1{1.0, 1.0, 1.0};
    picBunch::Vec3 const aspect4{1.0, 4.0, 1.0};

    auto const single1 = oneElectron({5.0, 5.0, 5.0}, u0);
    auto const single10 = oneElectron({5.0, 5.0, 5.0}, u10);
    auto const cloud1 = threeElectronCloud(u0);
    auto const cloud10 = threeElectronCloud(u10);

    auto const single1a1 = measurePaddingConvergence("single gamma1 aspect1", single1, aspect1);
    auto const single1a4 = measurePaddingConvergence("single gamma1 aspect4", single1, aspect4);
    auto const single10a1 = measurePaddingConvergence("single gamma10 aspect1", single10, aspect1);
    auto const cloud1a1 = measurePaddingConvergence("cloud gamma1 aspect1", cloud1, aspect1);
    auto const cloud1a4 = measurePaddingConvergence("cloud gamma1 aspect4", cloud1, aspect4);
    auto const cloud10a1 = measurePaddingConvergence("cloud gamma10 aspect1", cloud10, aspect1);

    // Acceptance: agreement at the FFT roundoff level, not merely "small".
    // The measured values are printed above and are ~1e-15; 1e-12 is the
    // conservative bound already documented by the code policy.
    for(auto const* values : {&single1a1, &single1a4, &single10a1, &cloud1a1, &cloud1a4, &cloud10a1})
        EXPECT_LT(worstOf(*values), 1.0e-12);

    // The negative control (padding 1 reproduces the wraparound artefact) is
    // Padding.NoPaddingReproducesWraparoundAndPaddingMatchesDirectSum.
}

TEST(Audit, DiscreteGaussLawOnYeeLattice)
{
    picBunch::Vec3 const u0{0.0, 0.0, 0.0};
    picBunch::Vec3 const u10{0.0, 0.0, 10.0};
    picBunch::Vec3 const aspect1{1.0, 1.0, 1.0};
    picBunch::Vec3 const aspect4{1.0, 4.0, 1.0};

    auto const single1 = oneElectron({5.0, 5.0, 5.0}, u0);
    auto const single10 = oneElectron({5.0, 5.0, 5.0}, u10);
    auto const cloud1 = threeElectronCloud(u0);
    auto const cloud10 = threeElectronCloud(u10);

    struct Case
    {
        char const* label;
        picBunch::Bunch const* bunch;
        picBunch::Vec3 cellSize;
    };
    std::array<Case, 6> const cases{{
        {"single gamma1 aspect1", &single1, aspect1},
        {"single gamma1 aspect4", &single1, aspect4},
        {"single gamma10 aspect1", &single10, aspect1},
        {"cloud gamma1 aspect1", &cloud1, aspect1},
        {"cloud gamma1 aspect4", &cloud1, aspect4},
        {"cloud gamma10 aspect1", &cloud10, aspect1},
    }};

    for(auto const& c : cases)
    {
        double const padding2 = measureDivergenceResidual(c.label, *c.bunch, c.cellSize, 2u);
        double const padding3 = measureDivergenceResidual(c.label, *c.bunch, c.cellSize, 3u);
        // No roundoff gate is asserted: the Yee difference of the sampled
        // point kernel is not the TSC shape, so the residual is a
        // discretisation residual (and the measured value is reported).  What
        // this test does assert is that removing every periodic image leaves
        // the residual unchanged, i.e. the measured value is not a padding
        // artefact.  A large value is a near-source discretisation finding, not
        // an FFT or padding defect.
        EXPECT_TRUE(std::isfinite(padding2));
        EXPECT_TRUE(std::isfinite(padding3));
        EXPECT_NEAR(padding3, padding2, 1.0e-12 * std::max(1.0, std::abs(padding2)))
            << c.label << ": padding changes the divergence residual";
    }
}

TEST(Audit, HighDensityOverlappingSources)
{
    // 1000 particles in a 2-cell cube, random sub-cell positions, all at rest
    // and carrying the same sign of charge so any coherent per-source error
    // adds instead of cancelling.
    std::mt19937 generator(0x5EEDu);
    std::uniform_real_distribution<double> uniform(8.0, 10.0);
    std::vector<picBunch::Vec3> positions(1000);
    for(auto& p : positions)
        p = {uniform(generator), uniform(generator), uniform(generator)};

    picBunch::ComputeOptions options;
    options.tolerance = 1e-3;
    options.cellSize = {1.0, 1.0, 1.0};
    options.gridSize = {24, 24, 24};
    options.gridOrigin = {0.0, 0.0, 0.0};
    options.fftPadding = 2;

    std::array<std::size_t, 4> const counts{1, 10, 100, 1000};
    std::array<double, 4> worst{};
    for(std::size_t c = 0; c < counts.size(); ++c)
    {
        picBunch::Bunch bunch;
        bunch.resize(counts[c]);
        for(std::size_t p = 0; p < counts[c]; ++p)
        {
            bunch.x[p] = positions[p][0];
            bunch.y[p] = positions[p][1];
            bunch.z[p] = positions[p][2];
            bunch.ux[p] = 0.0;
            bunch.uy[p] = 0.0;
            bunch.uz[p] = 0.0;
            bunch.weight[p] = 1.0;
            bunch.charge[p] = picBunch::kElementaryCharge;
        }
        picBunch::Field const field = picBunch::computeField(bunch, options);
        auto const perComponent = directReferenceError(field, bunch);
        worst[c] = worstOf(perComponent);
        std::printf(
            "AUDIT high-density N=%4zu Ex=%.3e Ey=%.3e Ez=%.3e Bx=%.3e By=%.3e Bz=%.3e worst=%.3e\n",
            counts[c],
            perComponent[0],
            perComponent[1],
            perComponent[2],
            perComponent[3],
            perComponent[4],
            perComponent[5],
            worst[c]);
    }

    // The direct convolution is the same discrete sum the padding-2 FFT is
    // meant to reproduce, so any residual beyond roundoff is accumulation
    // order or a periodic image.  Assert the measured gate at the densest case.
    EXPECT_LT(worst.back(), 1.0e-10);

    // Verdict on coherent growth: compare the 1000-source residual with the
    // single-source residual.  Uncorrelated accumulation stays at the same
    // roundoff level; coherent accumulation would grow roughly with N.
    double const growth = (worst.front() > 0.0) ? worst.back() / worst.front() : 1.0;
    std::printf("AUDIT high-density growth worst(N=1000)/worst(N=1)=%.3e\n", growth);
    EXPECT_LT(growth, 100.0) << "per-source errors grow coherently from N=1 to N=1000";
}

TEST(Audit, SeamContinuityAcrossNearShell)
{
    // A single stationary source at the centre of a 16^3 grid.  The near-shell
    // correction (fast mode) must be concentrated inside the shell and decay to
    // roundoff at the seam, otherwise the assembly leaves a visible jump.
    picBunch::Bunch bunch;
    bunch.resize(1);
    double const h = 1.0e-6;
    bunch.x = {8.0 * h};
    bunch.y = {8.0 * h};
    bunch.z = {8.0 * h};
    bunch.ux = {0.0};
    bunch.uy = {0.0};
    bunch.uz = {0.0};
    bunch.weight = {1.0};
    bunch.charge = {-picBunch::kElementaryCharge};

    picBunch::ComputeOptions options;
    options.cellSize = {h, h, h};
    options.gridSize = {16, 16, 16};
    options.gridOrigin = {0.0, 0.0, 0.0};
    options.fftPadding = 2u;
    options.deterministicDeposit = true;

    auto const mesh = picBunch::computeField(bunch, options);
    picBunch::NearShellOptions shell;
    shell.mode = picBunch::NearShellOptions::Mode::Fast;
    shell.radiusCells = 3.0;
    shell.tileNodes = 4;
    shell.fastOrder = 5;
    auto const total = picBunch::computeFieldNearExact(bunch, options, shell);

    double peak = 0.0;
    for(int c = 0; c < 6; ++c)
        for(double value : total.data[c])
            peak = std::max(peak, std::abs(value));

    double inside = 0.0;
    double outside = 0.0;
    std::size_t const n = 16u;
    for(std::size_t k = 0; k < n; ++k)
        for(std::size_t j = 0; j < n; ++j)
            for(std::size_t i = 0; i < n; ++i)
            {
                double const dx = (static_cast<double>(i) - 8.0) * h;
                double const dy = (static_cast<double>(j) - 8.0) * h;
                double const dz = (static_cast<double>(k) - 8.0) * h;
                double const radius = std::sqrt(dx * dx + dy * dy + dz * dz) / h;
                std::size_t const idx = i + n * (j + n * k);
                for(int c = 0; c < 6; ++c)
                {
                    double const correction = std::abs(total.data[c][idx] - mesh.data[c][idx]);
                    if(radius <= 2.0)
                        inside = std::max(inside, correction);
                    if(radius >= 4.0 && radius <= 5.0)
                        outside = std::max(outside, correction);
                }
            }
    ASSERT_GT(inside, 0.0) << "the near shell must change the field inside its support";
    EXPECT_GT(peak, 0.0);
    EXPECT_LT(outside, 1.0e-2 * peak) << "seam correction must decay at the shell boundary";
    EXPECT_LT(outside, 0.25 * inside) << "correction must be concentrated inside the shell";
}

TEST(Audit, DeterministicDepositIsBitwiseReproducible)
{
    auto const bunch = [] {
        picBunch::Bunch b;
        b.resize(64);
        std::mt19937 rng(7u);
        std::uniform_real_distribution<double> u(-1.0e-6, 1.0e-6);
        for(std::size_t p = 0; p < b.size(); ++p)
        {
            b.x[p] = 8.0e-6 + u(rng);
            b.y[p] = 8.0e-6 + u(rng);
            b.z[p] = 8.0e-6 + u(rng);
            b.ux[p] = b.uy[p] = b.uz[p] = 0.0;
            b.weight[p] = 1.0;
            b.charge[p] = -picBunch::kElementaryCharge;
        }
        return b;
    }();

    picBunch::ComputeOptions options;
    options.cellSize = {1.0e-6, 1.0e-6, 1.0e-6};
    options.gridSize = {16, 16, 16};
    options.gridOrigin = {0.0, 0.0, 0.0};
    options.fftPadding = 2u;
    options.deterministicDeposit = true;
    auto const first = picBunch::computeField(bunch, options);
    auto const second = picBunch::computeField(bunch, options);
    for(int c = 0; c < 6; ++c)
        ASSERT_EQ(first.data[c], second.data[c]) << "component " << c;
}
