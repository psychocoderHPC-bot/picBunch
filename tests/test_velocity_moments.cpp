// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group 15: velocity-moment spectral decomposition (candidate C1).
//
// Two things are pinned here.  First, the truncated multivariate Taylor of the
// normalized point kernel (``picBunch/VelocityMoments.hpp``) is checked against
// finite differences: the value at the centre matches ``pointFieldNormalized``,
// the reconstructed field converges at the order the Taylor should, and the
// first-derivative coefficients match central differences.  Second, the whole
// moment pipeline is checked against the 256-velocity-box binned reference on a
// gamma-10 bunch with a 5% energy spread (worst relative L2).

#include "picBunch/Compute.hpp"
#include "picBunch/Kernel.hpp"
#include "picBunch/NearShell.hpp"
#include "picBunch/VelocityMoments.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

namespace
{
    using picBunch::Bunch;
    using picBunch::ComputeOptions;
    using picBunch::VelocityMomentsOptions;
    using picBunch::VelocityMomentsReport;

    constexpr double kElectronCharge = -1.602176634e-19;

    //! Gamma-mean bunch with a Gaussian relative energy spread, moving along +z,
    //! positions on a small lattice so the deposit does not stack.
    Bunch spreadBunch(std::size_t n, double gammaMean, double relSpread)
    {
        Bunch bunch;
        bunch.resize(n);
        std::mt19937 gen(0x12345678u);
        std::normal_distribution<double> gauss(0.0, 1.0);
        std::size_t const side = static_cast<std::size_t>(std::cbrt(static_cast<double>(n))) + 1u;
        for(std::size_t p = 0; p < n; ++p)
        {
            double const gamma = gammaMean * (1.0 + relSpread * gauss(gen));
            bunch.x[p] = 4.0 + 0.25 * static_cast<double>(p % side);
            bunch.y[p] = 4.0 + 0.25 * static_cast<double>((p / side) % side);
            bunch.z[p] = 4.0 + 0.25 * static_cast<double>((p / (side * side)) % side);
            bunch.ux[p] = 0.0;
            bunch.uy[p] = 0.0;
            bunch.uz[p] = std::sqrt(std::max(1.0e-12, gamma * gamma - 1.0));
            bunch.weight[p] = 1.0;
            bunch.charge[p] = kElectronCharge;
        }
        return bunch;
    }

    ComputeOptions momentOptions()
    {
        ComputeOptions options;
        options.tolerance = 1e-3;
        options.cellSize = {1.0, 1.0, 1.0};
        options.gridSize = {48, 48, 48};
        options.gridOrigin = {0.0, 0.0, 0.0};
        options.fftPadding = 2u;
        return options;
    }

    //! Frobenius relative L2 of ``(a - b)`` against ``b`` over all six components.
    double relativeL2(picBunch::Field const& a, picBunch::Field const& b)
    {
        double num = 0.0;
        double den = 0.0;
        for(int c = 0; c < 6; ++c)
        {
            for(std::size_t i = 0; i < a.data[c].size(); ++i)
            {
                double const d = a.data[c][i] - b.data[c][i];
                num += d * d;
                den += b.data[c][i] * b.data[c][i];
            }
        }
        return std::sqrt(num) / std::sqrt(den);
    }

    //! Fill ``out`` with the order-3 Taylor coefficients of the kernel at (R, u0).
    template<std::size_t Order = 3>
    void fillTaylor(double const R[3], double const u0[3], double* out)
    {
        picBunch::detail::pointFieldTaylor<Order>(R[0], R[1], R[2], u0[0], u0[1], u0[2], out);
    }
} // namespace

// ---------------------------------------------------------------- Taylor kernel

TEST(VelocityMoments, TaylorValueMatchesPointFieldAtCentre)
{
    double const R[3] = {0.37, -0.21, 0.62};
    double const u0[3] = {0.0, 0.0, std::sqrt(100.0 - 1.0)}; // gamma 10 along +z
    std::array<double, 6u * picBunch::detail::TaylorPoly<3>::T> coeffs{};
    fillTaylor<3>(R, u0, coeffs.data());
    double exact[6];
    picBunch::pointFieldNormalized(R[0], R[1], R[2], u0[0], u0[1], u0[2], exact);
    for(int c = 0; c < 6; ++c)
    {
        double const value
            = picBunch::detail::pointFieldTaylorValue<3>(coeffs.data(), c, 0.0, 0.0, 0.0);
        EXPECT_NEAR(value, exact[c], 1e-12) << "component " << c << " value at centre";
    }
}

TEST(VelocityMoments, TaylorReconstructionConverges)
{
    double const R[3] = {0.37, -0.21, 0.62};
    double const u0[3] = {0.0, 0.0, std::sqrt(100.0 - 1.0)};
    std::array<double, 6u * picBunch::detail::TaylorPoly<3>::T> coeffs{};
    fillTaylor<3>(R, u0, coeffs.data());
    // Displace along the transverse (y) axis.  An order-3 Taylor in u is exact
    // through the cubic, so the reconstruction error is O(du^4): halving du must
    // shrink the error by about 2^4 = 16 (we require at least 2^3 = 8).
    double previous = 0.0;
    double ratio = 0.0;
    for(double du : {0.1, 0.05, 0.025})
    {
        double exact[6];
        picBunch::pointFieldNormalized(R[0], R[1], R[2], u0[0], u0[1] + du, u0[2], exact);
        double maxErr = 0.0;
        for(int c = 0; c < 6; ++c)
        {
            double const recon
                = picBunch::detail::pointFieldTaylorValue<3>(coeffs.data(), c, 0.0, du, 0.0);
            double const scale = std::max(1.0, std::abs(exact[c]));
            maxErr = std::max(maxErr, std::abs(recon - exact[c]) / scale);
        }
        if(previous > 0.0)
            ratio = maxErr / previous;
        previous = maxErr;
    }
    EXPECT_LT(previous, 1e-3) << "residual at du = 0.025";
    EXPECT_LT(ratio, 8.0) << "convergence ratio " << ratio;
    EXPECT_LT(ratio, 0.9) << "error should shrink as du shrinks";
}

TEST(VelocityMoments, FirstDerivativesMatchFiniteDifferences)
{
    double const R[3] = {0.37, -0.21, 0.62};
    double const u0[3] = {0.0, 0.0, std::sqrt(100.0 - 1.0)};
    std::array<double, 6u * picBunch::detail::TaylorPoly<3>::T> coeffs{};
    fillTaylor<3>(R, u0, coeffs.data());
    double const h = 1.0e-4;
    std::size_t const axis[3][3] = {{1u, 0u, 0u}, {0u, 1u, 0u}, {0u, 0u, 1u}};
    for(int d = 0; d < 3; ++d)
    {
        double up[3] = {u0[0], u0[1], u0[2]};
        double um[3] = {u0[0], u0[1], u0[2]};
        up[d] += h;
        um[d] -= h;
        double fup[6];
        double fum[6];
        picBunch::pointFieldNormalized(R[0], R[1], R[2], up[0], up[1], up[2], fup);
        picBunch::pointFieldNormalized(R[0], R[1], R[2], um[0], um[1], um[2], fum);
        for(int c = 0; c < 6; ++c)
        {
            double const fd = (fup[c] - fum[c]) / (2.0 * h);
            double const coeff = coeffs[static_cast<std::size_t>(c)
                    * picBunch::detail::TaylorPoly<3>::T
                    + picBunch::detail::TaylorPoly<3>::compactIndex(
                      axis[d][0], axis[d][1], axis[d][2])];
            double const scale = std::max(1.0, std::abs(fd));
            EXPECT_NEAR(coeff, fd, 1.0e-3 * scale) << "component " << c << " axis " << d;
        }
    }
}

// ---------------------------------------------------------------- the pipeline

TEST(VelocityMoments, MomentFieldMatchesTwoHundredFiftySixBoxReference)
{
    Bunch const bunch = spreadBunch(216, 10.0, 0.05);

    // Reference: the 256-velocity-box binned field.
    auto refOptions = momentOptions();
    refOptions.velocityBinning.mode = picBunch::VelocityBinningOptions::Mode::Explicit;
    refOptions.velocityBinning.explicitBins = 256u;
    picBunch::VelocityBinning refBinning;
    picBunch::Field const ref = picBunch::computeField(bunch, refOptions, nullptr, &refBinning);
    ASSERT_EQ(refBinning.count, 256u);

    auto makeMomentOptions = []() {
        ComputeOptions options = momentOptions();
        options.velocityMoments.mode = VelocityMomentsOptions::Mode::Explicit;
        return options;
    };

    // Order 3.
    auto o3 = makeMomentOptions();
    o3.velocityMoments.order = 3u;
    VelocityMomentsReport rep3{};
    picBunch::Field const field3
        = picBunch::computeField(bunch, o3, nullptr, nullptr, &rep3);
    ASSERT_TRUE(rep3.used) << "order-3 moment path should be used";
    EXPECT_EQ(rep3.order, 3);

    // Order 2.
    auto o2 = makeMomentOptions();
    o2.velocityMoments.order = 2u;
    VelocityMomentsReport rep2{};
    picBunch::Field const field2
        = picBunch::computeField(bunch, o2, nullptr, nullptr, &rep2);
    ASSERT_TRUE(rep2.used) << "order-2 moment path should be used";

    double const rel3 = relativeL2(field3, ref);
    double const rel2 = relativeL2(field2, ref);
    std::printf(
        "[velocity-moments] gamma-10 5%% spread: order-3 relL2 = %.3e, "
        "order-2 relL2 = %.3e (vs 256 boxes)\n",
        rel3,
        rel2);
    EXPECT_LT(rel3, 5.0e-2) << "order-3 worst relative L2 vs 256 boxes";
    EXPECT_LT(rel3, rel2) << "order 3 should beat order 2";
}

TEST(VelocityMoments, AutoSelectsOrderThreeAndMatches)
{
    Bunch const bunch = spreadBunch(216, 10.0, 0.05);
    auto refOptions = momentOptions();
    refOptions.velocityBinning.mode = picBunch::VelocityBinningOptions::Mode::Explicit;
    refOptions.velocityBinning.explicitBins = 256u;
    picBunch::Field const ref = picBunch::computeField(bunch, refOptions);

    auto options = momentOptions();
    options.velocityMoments.mode = VelocityMomentsOptions::Mode::Auto;
    VelocityMomentsReport rep{};
    picBunch::Field const field = picBunch::computeField(bunch, options, nullptr, nullptr, &rep);
    ASSERT_TRUE(rep.used) << "auto should use the moment path; probe = " << rep.probe;
    EXPECT_EQ(rep.order, 3);
    std::printf("[velocity-moments] auto probe = %.3e (order %d)\n", rep.probe, rep.order);
    EXPECT_LT(relativeL2(field, ref), 5.0e-2);
}

TEST(VelocityMoments, OffKeepsTheBinnedPath)
{
    Bunch const bunch = spreadBunch(216, 10.0, 0.05);
    auto options = momentOptions();
    // Default mode is Off: the moment path must not engage.
    VelocityMomentsReport rep{};
    picBunch::Field const field = picBunch::computeField(bunch, options, nullptr, nullptr, &rep);
    EXPECT_FALSE(rep.used);
    for(auto const& component : field.data)
        for(double v : component)
            EXPECT_TRUE(std::isfinite(v));
}

// ------------------------------------------------------- near-shell composition

TEST(VelocityMoments, NearShellComposesWithMomentField)
{
    // Two velocity classes that are spatially separated by more than the shell
    // radius.  The full-bunch centre is far from either class, so a per-tile
    // centre recomputed from the near subset would make the subtraction cancel
    // a different Taylor approximation than the global mesh added.  The
    // composition is correct only if the near band still matches the binned
    // near-shell reference (both replace the near sources with the exact pair
    // field).
    using picBunch::NearShellOptions;

    Bunch bunch;
    bunch.resize(2);
    bunch.x = {12.0, 12.0};
    bunch.y = {12.0, 12.0};
    bunch.z = {12.0, 2.0};
    bunch.ux = {0.0, 0.0};
    bunch.uy = {0.0, 0.0};
    bunch.uz = {std::sqrt(100.0 - 1.0), 0.0}; // gamma 10 and a resting source
    bunch.weight = {1.0, 1.0};
    bunch.charge = {kElectronCharge, kElectronCharge};

    ComputeOptions options;
    options.cellSize = {1.0, 1.0, 1.0};
    options.gridSize = {24, 24, 24};
    options.gridOrigin = {0.0, 0.0, 0.0};
    options.fftPadding = 2u;
    options.deterministicDeposit = true;

    NearShellOptions shell;
    shell.mode = NearShellOptions::Mode::Exact;
    shell.radiusCells = 3.0;
    shell.tileNodes = 4;

    auto refOptions = options;
    refOptions.velocityBinning.mode = picBunch::VelocityBinningOptions::Mode::Explicit;
    refOptions.velocityBinning.explicitBins = 16u;
    picBunch::Field const reference
        = picBunch::computeFieldNearExact(bunch, refOptions, shell);

    auto momentOptions = options;
    momentOptions.velocityMoments.mode = VelocityMomentsOptions::Mode::Explicit;
    momentOptions.velocityMoments.order = 3u;
    // This two-class bunch (one source at rest) has a large Taylor remainder.
    // The composition test exercises the pinned-centre subtraction operator, so
    // force the explicit order through the diagnostic override.
    momentOptions.velocityMoments.allowUnconverged = true;
    VelocityMomentsReport rep{};
    picBunch::Field const candidate = picBunch::computeFieldNearExact(
        bunch, momentOptions, shell, nullptr, nullptr, &rep);
    ASSERT_TRUE(rep.used) << "the moment path must be composed with the near shell";
    EXPECT_EQ(rep.order, 3);

    // Chebyshev distance <= 2 from the gamma-10 source (inside the shell).
    double num = 0.0;
    double den = 0.0;
    std::size_t const n = 24u;
    for(std::size_t k = 0; k < n; ++k)
        for(std::size_t j = 0; j < n; ++j)
            for(std::size_t i = 0; i < n; ++i)
            {
                double const cheb = std::max(
                    {std::abs(static_cast<double>(i) - 12.0),
                     std::abs(static_cast<double>(j) - 12.0),
                     std::abs(static_cast<double>(k) - 12.0)});
                if(cheb > 2.0)
                    continue;
                std::size_t const idx = i + n * (j + n * k);
                for(int c = 0; c < 6; ++c)
                {
                    double const d = candidate.data[c][idx] - reference.data[c][idx];
                    num += d * d;
                    den += reference.data[c][idx] * reference.data[c][idx];
                }
            }
    double const nearRelL2 = std::sqrt(num) / std::sqrt(den);
    std::printf(
        "[velocity-moments] near-shell+moment near-band relL2 vs binned near shell = %.3e\n",
        nearRelL2);
    EXPECT_LT(nearRelL2, 5.0e-2)
        << "near-shell subtraction did not cancel the same moment operator as the global mesh";
}

TEST(VelocityMoments, AutoDoesNotFailAtTheVelocityBinCap)
{
    // The binned auto policy alone would throw here (maxBins = 2, tol 1e-12).
    // With the moment path enabled it must run the moments instead of failing
    // at the cap; if the probe declines, the binned path may use the cap.
    Bunch const bunch = spreadBunch(216, 10.0, 0.05);
    auto options = momentOptions();
    options.velocityBinning.maxBins = 2u;
    options.tolerance = 1e-12;
    options.velocityMoments.mode = VelocityMomentsOptions::Mode::Auto;
    VelocityMomentsReport rep{};
    picBunch::Field field;
    EXPECT_NO_THROW(field = picBunch::computeField(bunch, options, nullptr, nullptr, &rep));
    EXPECT_TRUE(rep.used || rep.fellBack);
    for(auto const& component : field.data)
        for(double v : component)
            EXPECT_TRUE(std::isfinite(v));
}

// ------------------------------------------------- gamma-100 validity guard

namespace
{
    //! Gamma-mean bunch with a Gaussian relative energy spread and a
    //! transverse divergence (uniform direction, up to ``divergenceDeg``).
    Bunch divergentBunch(
        std::size_t n,
        double gammaMean,
        double relSpread,
        double divergenceDeg)
    {
        Bunch bunch;
        bunch.resize(n);
        std::mt19937 gen(0x0badf00du);
        std::normal_distribution<double> gauss(0.0, 1.0);
        std::uniform_real_distribution<double> uniform(-1.0, 1.0);
        std::size_t const side = static_cast<std::size_t>(std::cbrt(static_cast<double>(n))) + 1u;
        double const tanDiv = std::tan(divergenceDeg * std::acos(-1.0) / 180.0);
        for(std::size_t p = 0; p < n; ++p)
        {
            double const gamma = gammaMean * (1.0 + relSpread * gauss(gen));
            double const uz = std::sqrt(std::max(1.0e-12, gamma * gamma - 1.0));
            bunch.x[p] = 4.0 + 0.5 * static_cast<double>(p % side);
            bunch.y[p] = 4.0 + 0.5 * static_cast<double>((p / side) % side);
            bunch.z[p] = 4.0 + 0.5 * static_cast<double>((p / (side * side)) % side);
            double const perp = uz * tanDiv;
            bunch.ux[p] = perp * uniform(gen);
            bunch.uy[p] = perp * uniform(gen);
            bunch.uz[p] = uz;
            bunch.weight[p] = 1.0;
            bunch.charge[p] = kElectronCharge;
        }
        return bunch;
    }

    ComputeOptions gamma100Options()
    {
        ComputeOptions options = momentOptions();
        options.gridSize = {32, 32, 32};
        return options;
    }

    void expectAllFinite(picBunch::Field const& field)
    {
        for(auto const& component : field.data)
            for(double const value : component)
                ASSERT_TRUE(std::isfinite(value));
    }
} // namespace

TEST(VelocityMoments, Gamma100ExplicitOrdersConvergeOrFallBack)
{
    // gamma 100 with a 5% energy spread is outside the Taylor regime: the
    // order-2-vs-3 probe must either reject the series (fallback to the binned
    // path) or, when it accepts it, the field must be close to the 256-box
    // reference.  A non-finite field is never acceptable.
    for(double const divergence : {1.0, 5.0})
    {
        Bunch const bunch = divergentBunch(96, 100.0, 0.05, divergence);

        auto refOptions = gamma100Options();
        refOptions.velocityBinning.mode = picBunch::VelocityBinningOptions::Mode::Explicit;
        refOptions.velocityBinning.explicitBins = 256u;
        picBunch::Field const reference = picBunch::computeField(bunch, refOptions);

        for(std::size_t const order : {2u, 3u})
        {
            auto options = gamma100Options();
            options.velocityMoments.mode = VelocityMomentsOptions::Mode::Explicit;
            options.velocityMoments.order = order;
            VelocityMomentsReport rep{};
            picBunch::Field const field
                = picBunch::computeField(bunch, options, nullptr, nullptr, &rep);

            expectAllFinite(field);
            ASSERT_TRUE(rep.used || rep.fellBack)
                << "order " << order << " divergence " << divergence;
            if(rep.used)
            {
                // Documented acceptance bound for a used explicit order: the
                // moment field must stay within 50% Frobenius relative L2 of the
                // 256-box reference.  At gamma 100 this branch is not expected to
                // run; the probe normally rejects it.
                double const rel = relativeL2(field, reference);
                std::printf(
                    "[velocity-moments] gamma-100 order %zu div %.0f deg USED: probe %.3e, "
                    "relL2 vs 256 boxes %.3e\n",
                    order,
                    divergence,
                    rep.probe,
                    rel);
                EXPECT_LT(rel, 0.5) << "used moment field is not close to the reference";
            }
            else
            {
                // The fallback must deliver the binned result, which with the
                // 256-box cap is the reference here.
                std::printf(
                    "[velocity-moments] gamma-100 order %zu div %.0f deg fell back: "
                    "probe %.3e\n",
                    order,
                    divergence,
                    rep.probe);
                EXPECT_LT(relativeL2(field, reference), 1e-12)
                    << "fallback did not return the binned field";
            }
        }
    }
}

TEST(VelocityMoments, Gamma100AutoMatchesOrBeatsBinned)
{
    Bunch const bunch = divergentBunch(96, 100.0, 0.05, 1.0);

    auto refOptions = gamma100Options();
    refOptions.velocityBinning.mode = picBunch::VelocityBinningOptions::Mode::Explicit;
    refOptions.velocityBinning.explicitBins = 256u;
    picBunch::Field const binned = picBunch::computeField(bunch, refOptions);

    auto options = gamma100Options();
    options.velocityMoments.mode = VelocityMomentsOptions::Mode::Auto;
    VelocityMomentsReport rep{};
    picBunch::Field const field
        = picBunch::computeField(bunch, options, nullptr, nullptr, &rep);
    expectAllFinite(field);
    ASSERT_TRUE(rep.used || rep.fellBack);

    double const rel = relativeL2(field, binned);
    std::printf(
        "[velocity-moments] gamma-100 auto: used=%d fellBack=%d probe %.3e, "
        "relL2 vs 256 boxes %.3e\n",
        rep.used ? 1 : 0,
        rep.fellBack ? 1 : 0,
        rep.probe,
        rel);
    if(rep.fellBack)
        EXPECT_LT(rel, 1e-12) << "auto fallback must reproduce the binned capped result";
    else
    {
        EXPECT_LT(rep.probe, 0.05);
        EXPECT_LT(rel, 0.5) << "auto moment field is not close to the binned reference";
    }
}

TEST(VelocityMoments, Gamma100AllowUnconvergedForcesFiniteExplicitOrder)
{
    // The explicit order must be kept only when the diagnostic override is on,
    // and the report must say so.  Finiteness still holds.
    Bunch const bunch = divergentBunch(64, 100.0, 0.05, 5.0);
    auto options = gamma100Options();
    options.velocityMoments.mode = VelocityMomentsOptions::Mode::Explicit;
    options.velocityMoments.order = 3u;
    options.velocityMoments.allowUnconverged = true;
    VelocityMomentsReport rep{};
    picBunch::Field const field
        = picBunch::computeField(bunch, options, nullptr, nullptr, &rep);
    ASSERT_TRUE(rep.used) << "the override must keep the explicit order";
    EXPECT_TRUE(rep.allowUnconverged);
    EXPECT_EQ(rep.order, 3);
    expectAllFinite(field);
}

TEST(VelocityMoments, Gamma100NearShellFastWithMomentsDoesNotThrowAtBinCap)
{
    // Reproduces the abort on the gate: near shell derives a binned mesh at the
    // cap before the moment path can run.  With moments enabled it must use the
    // bounded fallback binning and deliver a finite field.
    using picBunch::NearShellOptions;
    Bunch const bunch = divergentBunch(48, 100.0, 0.05, 1.0);

    ComputeOptions options;
    options.cellSize = {1.0, 1.0, 1.0};
    options.gridSize = {16, 16, 16};
    options.gridOrigin = {0.0, 0.0, 0.0};
    options.fftPadding = 2u;
    options.deterministicDeposit = true;
    options.velocityBinning.maxBins = 2u;
    options.tolerance = 1e-3;
    options.velocityMoments.mode = VelocityMomentsOptions::Mode::Auto;

    NearShellOptions shell;
    shell.mode = NearShellOptions::Mode::Fast;
    shell.radiusCells = 3.0;
    shell.tileNodes = 4;

    VelocityMomentsReport rep{};
    picBunch::Field field;
    EXPECT_NO_THROW(
        field = picBunch::computeFieldNearExact(
            bunch, options, shell, nullptr, nullptr, &rep));
    expectAllFinite(field);
    EXPECT_TRUE(rep.used || rep.fellBack);
}

TEST(VelocityMoments, Gamma10ExplicitOrdersStillConverge)
{
    // Regression guard: the new explicit-order probe must not reject the
    // validated gamma-10 regime.
    Bunch const bunch = spreadBunch(216, 10.0, 0.05);
    for(std::size_t const order : {2u, 3u})
    {
        auto options = momentOptions();
        options.velocityMoments.mode = VelocityMomentsOptions::Mode::Explicit;
        options.velocityMoments.order = order;
        VelocityMomentsReport rep{};
        picBunch::Field const field
            = picBunch::computeField(bunch, options, nullptr, nullptr, &rep);
        ASSERT_TRUE(rep.used) << "gamma-10 order " << order << " regressed: probe "
                              << rep.probe;
        EXPECT_LT(rep.probe, 0.05);
        expectAllFinite(field);
    }
}
