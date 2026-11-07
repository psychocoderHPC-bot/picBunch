// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Empirical validation of the tolerance-derived tail margin and of the
// multipole far tail that replaces the global-domain zero fill.

#include "picBunch/Compute.hpp"
#include "picBunch/FarTail.hpp"
#include "picBunch/Kernel.hpp"
#include "picBunch/Kernel.hpp"
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

    using Vec3 = std::array<double, 3>;

    double vnorm(Vec3 const& v)
    {
        return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }
    double relErr(Vec3 const& a, Vec3 const& b)
    {
        return vnorm({a[0] - b[0], a[1] - b[1], a[2] - b[2]}) / (vnorm(b) + 1e-300);
    }
    //! Electric field of a point charge ``Q`` (dimensionless momentum ``u``) at
    //! displacement ``R``, i.e. the exact ``Q * pointFieldNormalized``.
    Vec3 ePoint(Vec3 const& R, double Q, Vec3 const& u)
    {
        double out[6];
        picBunch::pointFieldNormalized(R[0], R[1], R[2], u[0], u[1], u[2], out);
        return {picBunch::kCoulombConstant * Q * out[0],
                picBunch::kCoulombConstant * Q * out[1],
                picBunch::kCoulombConstant * Q * out[2]};
    }
    //! Magnetic field of the same point charge.
    Vec3 bPoint(Vec3 const& R, double Q, Vec3 const& u)
    {
        double out[6];
        picBunch::pointFieldNormalized(R[0], R[1], R[2], u[0], u[1], u[2], out);
        double const s = picBunch::kCoulombConstant * Q / picBunch::kSpeedOfLight;
        return {s * out[3], s * out[4], s * out[5]};
    }
    //! The pre-fix static multipole formula (monopole + dipole + quadrupole,
    //! ``gammaMax = 1``): the reference the moving tail must reduce to at
    //! ``meanVelocity = 0``.
    Vec3 staticMultipole(picBunch::FarTail const& t, Vec3 const& pos)
    {
        Vec3 const r{pos[0] - t.centre[0], pos[1] - t.centre[1], pos[2] - t.centre[2]};
        double const d2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        double const d = std::sqrt(d2);
        Vec3 const n{r[0] / d, r[1] / d, r[2] / d};
        double const pDotN = t.dipole[0] * n[0] + t.dipole[1] * n[1] + t.dipole[2] * n[2];
        Vec3 dN{};
        for(int i = 0; i < 3; ++i)
            dN[i] = t.quadrupole[i * 3] * n[0] + t.quadrupole[i * 3 + 1] * n[1]
                + t.quadrupole[i * 3 + 2] * n[2];
        double const nDotDn = n[0] * dN[0] + n[1] * dN[1] + n[2] * dN[2];
        double const mono = picBunch::kCoulombConstant * t.charge / d2;
        double const ds = picBunch::kCoulombConstant * 2.0 / (d2 * d);
        double const qs = picBunch::kCoulombConstant * 0.5 / (d2 * d2);
        Vec3 f{};
        for(int a = 0; a < 3; ++a)
        {
            double const dipoleTerm = 1.5 * pDotN * n[a] - 0.5 * t.dipole[a];
            double const quadrupoleTerm = 5.0 * nDotDn * n[a] - 2.0 * dN[a];
            f[a] = mono * n[a] + ds * dipoleTerm + qs * quadrupoleTerm;
        }
        return f;
    }
    //! One macro particle at the origin with charge ``q`` and momentum ``u``.
    picBunch::Bunch pointBunch(double q, Vec3 const& u)
    {
        picBunch::Bunch b;
        b.resize(1);
        b.x = {0.0};
        b.y = {0.0};
        b.z = {0.0};
        b.ux = {u[0]};
        b.uy = {u[1]};
        b.uz = {u[2]};
        b.weight = {1.0};
        b.charge = {q};
        return b;
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

TEST(Margin, FarTailStaticReductionExact)
{
    // A stationary bunch (meanVelocity = 0) must reduce EXACTLY to the static
    // monopole + dipole + quadrupole field (the pre-fix formula, gammaMax = 1).
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
    auto const tail = picBunch::deriveFarTail(bunch);
    EXPECT_DOUBLE_EQ(vnorm(tail.meanVelocity), 0.0);
    Vec3 dirs[3] = {{0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 1.0}};
    for(auto const& d : dirs)
        for(double const rad : {3.0, 5.0, 8.0})
        {
            double const n = vnorm(d);
            Vec3 const P{tail.centre[0] + d[0] / n * rad,
                         tail.centre[1] + d[1] / n * rad,
                         tail.centre[2] + d[2] / n * rad};
            EXPECT_NEAR(
                relErr(picBunch::evaluateFarTailE(tail, P), staticMultipole(tail, P)), 0.0,
                1.0e-14)
                << "beta=0 must reduce exactly to the static multipole field";
        }
}

TEST(Margin, MovingFarTailMatchesPointKernel)
{
    // A one-particle bunch is a pure monopole: the moving far tail must equal
    // the exact ``Q * pointFieldNormalized`` Lienard point field in every
    // direction.  The old isotropic gammaMax scaling was 1000x too high on axis
    // at gamma=10, so this fails on the pre-fix code.
    double const e = picBunch::kElementaryCharge;
    Vec3 dirs[3] = {{0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}; // axial, transverse, oblique
    for(double const gamma : {1.0, 10.0, 100.0})
    {
        double const u = std::sqrt(gamma * gamma - 1.0);
        auto const tail = picBunch::deriveFarTail(pointBunch(-e, {0.0, 0.0, u}));
        for(auto const& d : dirs)
        {
            double const n = vnorm(d);
            Vec3 const Rv{d[0] / n, d[1] / n, d[2] / n}; // unit distance from centre
            EXPECT_NEAR(
                relErr(picBunch::evaluateFarTailE(tail, Rv), ePoint(Rv, -e, {0.0, 0.0, u})),
                0.0, 1.0e-8)
                << "gamma=" << gamma << " direction not reproduced by the point kernel";
        }
    }
}

TEST(Margin, FarTailMovingMagneticField)
{
    // B = v x E / c^2; for a one-particle bunch it also equals the exact point B.
    double const e = picBunch::kElementaryCharge;
    double const u = std::sqrt(100.0 - 1.0);
    auto const tail = picBunch::deriveFarTail(pointBunch(-e, {0.0, 0.0, u}));
    Vec3 const Rv{0.4, 0.0, 1.0};
    Vec3 const E = picBunch::evaluateFarTailE(tail, Rv);
    Vec3 const B = picBunch::evaluateFarTailB(tail, Rv);
    double const i2 = 1.0 / (picBunch::kSpeedOfLight * picBunch::kSpeedOfLight);
    Vec3 const Bexp{(tail.meanVelocity[1] * E[2] - tail.meanVelocity[2] * E[1]) * i2,
                    (tail.meanVelocity[2] * E[0] - tail.meanVelocity[0] * E[2]) * i2,
                    (tail.meanVelocity[0] * E[1] - tail.meanVelocity[1] * E[0]) * i2};
    EXPECT_NEAR(vnorm({B[0] - Bexp[0], B[1] - Bexp[1], B[2] - Bexp[2]}), 0.0, 1e-300)
        << "B must equal meanVelocity x E / c^2";
    EXPECT_NEAR(relErr(B, bPoint(Rv, -e, {0.0, 0.0, u})), 0.0, 1.0e-8)
        << "B must equal the exact point-kernel B";
}

TEST(Margin, FarTailMovingCloudApproximatesDirectSum)
{
    // A compact cloud moving uniformly: the far tail must approximate the direct
    // sum of the per-particle Lienard fields.  The axial field is the decisive
    // case (the old code was 1000x off there).  The component transverse to the
    // motion of a bunch with beam-axis extent is the least accurate part of the
    // asymptotic point multipole, so it is checked with a looser bound plus a
    // convergence check rather than a tight tolerance.
    double const e = picBunch::kElementaryCharge;
    double const u = std::sqrt(100.0 - 1.0); // gamma 10
    double const w = 1.0e-6;
    picBunch::Bunch bunch;
    bunch.resize(27);
    int k = 0;
    for(int ix = -1; ix <= 1; ++ix)
        for(int iy = -1; iy <= 1; ++iy)
            for(int iz = -1; iz <= 1; ++iz)
            {
                bunch.x[k] = ix * w * 0.9;
                bunch.y[k] = iy * w * 0.9;
                bunch.z[k] = iz * w * 0.9;
                bunch.ux[k] = 0.0;
                bunch.uy[k] = 0.0;
                bunch.uz[k] = u;
                bunch.weight[k] = 1.0;
                bunch.charge[k] = -e;
                ++k;
            }
    bunch.resize(k);
    auto const tail = picBunch::deriveFarTail(bunch);
    auto directAt = [&](Vec3 const& P) {
        Vec3 s{0.0, 0.0, 0.0};
        for(std::size_t p = 0; p < bunch.size(); ++p)
        {
            Vec3 const Rv{P[0] - bunch.x[p], P[1] - bunch.y[p], P[2] - bunch.z[p]};
            Vec3 const pe = ePoint(Rv, -e, {0.0, 0.0, u});
            s = {s[0] + pe[0], s[1] + pe[1], s[2] + pe[2]};
        }
        return s;
    };
    Vec3 dirs[3] = {{0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}; // axial, transverse, oblique
    double const tol[3] = {0.02, 0.30, 0.10};
    for(int i = 0; i < 3; ++i)
    {
        double const n = vnorm(dirs[i]);
        auto at = [&](double f) {
            Vec3 const P{tail.centre[0] + dirs[i][0] / n * f * tail.supportRadius,
                         tail.centre[1] + dirs[i][1] / n * f * tail.supportRadius,
                         tail.centre[2] + dirs[i][2] / n * f * tail.supportRadius};
            return relErr(picBunch::evaluateFarTailE(tail, P), directAt(P));
        };
        EXPECT_TRUE(std::isfinite(at(12.0))) << "direction " << i;
        EXPECT_LT(at(12.0), tol[i]) << "direction " << i << " (axial=0, transverse=1)";
        EXPECT_LT(at(12.0), at(8.0)) << "must converge, direction " << i;
    }
}

TEST(Margin, NeutralCounterstreamingUsesAbsoluteChargeMeanVelocity)
{
    // Net charge zero with counterstreaming members.  The mean velocity must use
    // the absolute-charge weights decided from the TOTAL charge after the loop;
    // the old running-partial-sum logic produced a different (charge-weighted,
    // and here ill-defined) direction.
    double const e = picBunch::kElementaryCharge;
    double const c = picBunch::kSpeedOfLight;
    picBunch::Bunch bunch;
    bunch.resize(3);
    bunch.x = {0.0, 0.0, 0.0};
    bunch.y = {0.0, 0.0, 0.0};
    bunch.z = {0.0, 0.0, 0.0};
    // p0: +2e u=+3 ; p1: -e u=-3 ; p2: -e u=+1  -> net charge 0, counterstreaming
    bunch.ux = {3.0, -3.0, 1.0};
    bunch.uy = {0.0, 0.0, 0.0};
    bunch.uz = {0.0, 0.0, 0.0};
    bunch.weight = {1.0, 1.0, 1.0};
    bunch.charge = {2.0 * e, -e, -e};
    auto const tail = picBunch::deriveFarTail(bunch);
    EXPECT_NEAR(tail.charge, 0.0, 1.0e-12 * 4.0 * e) << "bunch must be neutral";
    double const v0 = c * 3.0 / std::sqrt(10.0);
    double const v1 = -c * 3.0 / std::sqrt(10.0);
    double const v2 = c * 1.0 / std::sqrt(2.0);
    double const expected = (2.0 * v0 + v1 + v2) / 4.0; // absolute-charge weighted
    EXPECT_NEAR(tail.meanVelocity[0], expected, 1.0e-9 * c);
    EXPECT_NEAR(tail.meanVelocity[1], 0.0, 1.0e-12 * c);
    EXPECT_NEAR(tail.meanVelocity[2], 0.0, 1.0e-12 * c);
    // the old running-partial-sum value was c*0.888; the correct one is c*0.414
    EXPECT_GT(std::abs(tail.meanVelocity[0] - c * 0.888290), 1.0e-2 * c)
        << "must use the absolute-charge weights, not the running partial sum";
}

TEST(Margin, MovingDipolarCloudBound)
{
    // A neutral, dipole-dominated bunch moving at gamma 10 exercises the
    // higher-multipole transform, which the one-particle and symmetric-cube
    // tests cannot see (both have zero dipole and quadrupole).  The leading
    // moving multipole is exact in the principal alignment; away from it the
    // documented bound is used, so this test pins both.
    double const gamma = 10.0;
    double const u = std::sqrt(gamma * gamma - 1.0);
    picBunch::Bunch bunch;
    bunch.resize(2);
    bunch.x = {-0.5, 0.5};
    bunch.y = {0.0, 0.0};
    bunch.z = {0.0, 0.0};
    bunch.ux = {0.0, 0.0};
    bunch.uy = {0.0, 0.0};
    bunch.uz = {u, u};
    bunch.weight = {1.0, 1.0};
    bunch.charge = {picBunch::kElementaryCharge, -picBunch::kElementaryCharge};

    auto const tail = picBunch::deriveFarTail(bunch);
    ASSERT_NEAR(tail.meanVelocity[2], picBunch::kSpeedOfLight * u / gamma, 1.0);
    ASSERT_NEAR(tail.charge, 0.0, 1.0e-30);

    auto direct = [&](std::array<double, 3> const& point) {
        std::array<double, 3> result{0.0, 0.0, 0.0};
        for(std::size_t p = 0; p < bunch.size(); ++p)
        {
            double const R[3] = {
                point[0] - bunch.x[p],
                point[1] - bunch.y[p],
                point[2] - bunch.z[p]};
            double field[6];
            picBunch::pointFieldNormalized(
                R[0], R[1], R[2], bunch.ux[p], bunch.uy[p], bunch.uz[p], field);
            double const scale = picBunch::kCoulombConstant * bunch.weight[p] * bunch.charge[p];
            for(int a = 0; a < 3; ++a)
                result[a] += scale * field[a];
        }
        return result;
    };

    // Principal alignment: dipole along x, observation along x, motion along z.
    std::array<double, 3> const along{12.0, 0.0, 0.0};
    auto const exact = direct(along);
    auto const approx = picBunch::evaluateFarTailE(tail, along);
    double num = 0.0;
    double den = 0.0;
    for(int a = 0; a < 3; ++a)
    {
        num += (approx[a] - exact[a]) * (approx[a] - exact[a]);
        den += exact[a] * exact[a];
    }
    EXPECT_LT(std::sqrt(num / std::max(den, 1.0e-300)), 1.0e-2)
        << "the principal moving-dipole alignment must be accurate";

    // Off-alignment the leading multipole has a finite residual.  Bound it by
    // the margin's conservative gamma-scaled static scale rather than claiming
    // a point-value accuracy the expansion does not have.
    double const scale = picBunch::kCoulombConstant * std::abs(bunch.charge[0]) / (12.0 * 12.0);
    std::array<std::array<double, 3>, 3> const directions{{
        {0.0, 12.0, 0.0},
        {0.0, 0.0, 12.0},
        {8.0, 8.0, 8.0},
    }};
    double worst = 0.0;
    for(auto direction : directions)
    {
        auto const e = direct(direction);
        auto const a = picBunch::evaluateFarTailE(tail, direction);
        double d2 = 0.0;
        for(int c = 0; c < 3; ++c)
            d2 += (a[c] - e[c]) * (a[c] - e[c]);
        worst = std::max(worst, std::sqrt(d2));
    }
    EXPECT_LT(worst / scale, 2.0)
        << "off-alignment moving multipole must stay inside the conservative bound";
}

TEST(Margin, GroupedMultipolesBeatTheGlobalExpansion)
{
    // Two separated clusters create strong higher multipoles.  A single global
    // expansion is limited by its quadrupole truncation; spatial supercell
    // groups reduce the tail error.
    picBunch::Bunch bunch;
    int const n = 400;
    bunch.resize(n);
    std::mt19937 rng(5u);
    std::uniform_real_distribution<double> uniform(-0.5, 0.5);
    for(int i = 0; i < n; ++i)
    {
        double const side = (i < n / 2) ? -1.0 : 1.0;
        bunch.x[i] = side * 6.0 + uniform(rng);
        bunch.y[i] = uniform(rng);
        bunch.z[i] = uniform(rng);
        bunch.ux[i] = 0.0;
        bunch.uy[i] = 0.0;
        bunch.uz[i] = std::sqrt(99.0);
        bunch.weight[i] = 1.0;
        bunch.charge[i] = -picBunch::kElementaryCharge;
    }
    auto direct = [&](std::array<double, 3> const& point) {
        std::array<double, 3> result{0.0, 0.0, 0.0};
        for(std::size_t p = 0; p < bunch.size(); ++p)
        {
            double const R[3] = {
                point[0] - bunch.x[p],
                point[1] - bunch.y[p],
                point[2] - bunch.z[p]};
            double field[6];
            picBunch::pointFieldNormalized(
                R[0], R[1], R[2], bunch.ux[p], bunch.uy[p], bunch.uz[p], field);
            double const scale = picBunch::kCoulombConstant * bunch.weight[p] * bunch.charge[p];
            for(int a = 0; a < 3; ++a)
                result[a] += scale * field[a];
        }
        return result;
    };
    auto relative = [](std::array<double, 3> const& value, std::array<double, 3> const& ref) {
        double num = 0.0;
        double den = 0.0;
        for(int a = 0; a < 3; ++a)
        {
            num += (value[a] - ref[a]) * (value[a] - ref[a]);
            den += ref[a] * ref[a];
        }
        return std::sqrt(num / std::max(den, 1.0e-300));
    };
    auto worstFor = [&](picBunch::FarTail const& tail) {
        double worst = 0.0;
        for(int factor : {6, 8, 12})
            for(int dir = 0; dir < 3; ++dir)
                for(int sign : {-1, 1})
                {
                    std::array<double, 3> point = tail.centre;
                    point[static_cast<std::size_t>(dir)]
                        += sign * factor * tail.supportRadius;
                    worst = std::max(
                        worst, relative(picBunch::evaluateFarTailE(tail, point), direct(point)));
                }
        return worst;
    };

    auto const global = picBunch::deriveFarTail(bunch);
    auto const grouped
        = picBunch::deriveFarTail(bunch, 6.0, {1.0, 1.0, 1.0}, 4);
    ASSERT_FALSE(grouped.groups.empty());
    double const globalWorst = worstFor(global);
    double const groupedWorst = worstFor(grouped);
    EXPECT_LT(groupedWorst, globalWorst) << "grouped multipoles must improve the tail";
    EXPECT_LT(groupedWorst, 5.0e-4);
}
