// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Regression guard for the near-shell evaluator regimes.
//
// Oblique in-support pairs are evaluated by the exact Lorentz-contracted
// support-box field plus a d-space ray integral of the TSC remainder.  The
// arbiter values below are the independent `indfield` (order 20, refine 2)
// results; `nbref` is retained only as the axial/exact-path reference because
// it is not a gamma-100 truth.

#include "picBunch/detail/FastNear.hpp"
#include "picBunch/detail/FastNearDevice.hpp"
#include "picBunch/detail/NearReference.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace
{
    using nbref::Vec3;

    constexpr double kGamma10Beta = 9.9498743710661995; // sqrt(100 - 1)
    constexpr double kGamma100Beta = 99.99499987499375; // sqrt(10000 - 1)

    double relativeError(Vec3 const& value, std::array<double, 3> const& reference)
    {
        std::array<double, 3> const v{value.x, value.y, value.z};
        double diff2 = 0.0;
        double ref2 = 0.0;
        for(int a = 0; a < 3; ++a)
        {
            diff2 += (v[a] - reference[a]) * (v[a] - reference[a]);
            ref2 += reference[a] * reference[a];
        }
        return std::sqrt(diff2 / std::max(ref2, 1.0e-300));
    }

    double relativeErrorV(Vec3 const& value, Vec3 const& reference)
    {
        return relativeError(value, {reference.x, reference.y, reference.z});
    }

    Vec3 fastField(Vec3 const& obs, Vec3 const& u, int order)
    {
        picBunch::pbfast::Settings settings;
        settings.order = order;
        settings.obliqueOrder = order;
        Vec3 E, B;
        picBunch::pbfast::shape_field(obs, Vec3{}, u, 1.0, 1.0, 1.0, settings, E, B);
        return E;
    }

    Vec3 referenceField(Vec3 const& obs, Vec3 const& u)
    {
        nbref::Settings settings;
        Vec3 E, B;
        nbref::shape_field(obs, Vec3{}, u, 1.0, 1.0, 1.0, settings, E, B);
        return E;
    }

    Vec3 oblique45(double const beta)
    {
        double const d = beta / std::sqrt(2.0);
        return {d, 0.0, d};
    }

    Vec3 tilt5(double const beta, double const degrees)
    {
        double const t = degrees * 3.141592653589793238462643383279502884 / 180.0;
        return {beta * std::sin(t), 0.0, beta * std::cos(t)};
    }
} // namespace

TEST(NearShellAccuracy, FastIsAccurateForAxialGamma10)
{
    Vec3 const u{0.0, 0.0, kGamma10Beta};
    EXPECT_LT(relativeError(fastField({-0.6, 0.4, -0.3}, u, 8), {-0.9346, 0.6158, -0.0228}), 1.0e-3);
    EXPECT_LT(relativeError(fastField({1.0, 0.0, 0.0}, u, 8), {1.2582, 0.0, 0.0}), 1.0e-3);
    EXPECT_LT(relativeError(fastField({0.0, 0.5, 0.0}, u, 8), {0.0, 1.0936, 0.0}), 1.0e-3);
}

TEST(NearShellAccuracy, AdaptiveReferenceIsAccurateForAxialGamma10)
{
    Vec3 const u{0.0, 0.0, kGamma10Beta};
    EXPECT_LT(relativeError(referenceField({1.0, 0.0, 0.0}, u), {1.2582, 0.0, 0.0}), 1.0e-2);
    EXPECT_LT(relativeError(referenceField({0.0, 0.5, 0.0}, u), {0.0, 1.0936, 0.0}), 1.0e-2);
    EXPECT_LT(relativeError(referenceField({-0.6, 0.4, -0.3}, u), {-0.9346, 0.6158, -0.0228}), 5.0e-2);
}

TEST(NearShellAccuracy, AdaptiveReferenceIsAccurateForObliqueGamma10)
{
    Vec3 const u = oblique45(kGamma10Beta);
    EXPECT_LT(relativeError(referenceField({1.0, 0.0, 0.0}, u), {0.4226, 0.0, -0.3752}), 2.0e-2);
    EXPECT_LT(relativeError(referenceField({0.0, 0.5, 0.0}, u), {0.0, 1.1086, 0.0}), 5.0e-3);
}

//! Independent indfield arbiter (order 20, refine 2), oblique 45 degrees.
TEST(NearShellAccuracy, FastObliqueGamma10MeetsArbiter)
{
    Vec3 const u = oblique45(kGamma10Beta);
    EXPECT_LT(relativeError(fastField({-0.6, 0.4, -0.3}, u, 8),
                            {-0.210369983, 0.460777915, 0.157158018}), 1.0e-2);
    EXPECT_LT(relativeError(fastField({1.0, 0.0, 0.0}, u, 8),
                            {0.422217066, 0.0, -0.374819481}), 1.0e-2);
    EXPECT_LT(relativeError(fastField({0.0, 0.5, 0.0}, u, 8),
                            {0.0, 1.108450387, 0.0}), 1.0e-2);
}

TEST(NearShellAccuracy, FastObliqueGamma100MeetsArbiter)
{
    Vec3 const u = oblique45(kGamma100Beta);
    EXPECT_LT(relativeError(fastField({-0.6, 0.4, -0.3}, u, 8),
                            {-0.18358776, 0.45896317, 0.18266562}), 1.0e-2);
    EXPECT_LT(relativeError(fastField({1.0, 0.0, 0.0}, u, 8),
                            {0.394222992, 0.0, -0.393359907}), 1.0e-2);
    EXPECT_LT(relativeError(fastField({0.0, 0.5, 0.0}, u, 8),
                            {0.0, 1.135118460, 0.0}), 1.0e-2);
}

//! The requested gamma-100 5-degree probes, E only, against indfield.
TEST(NearShellAccuracy, Gamma100FiveDegreeProbesMeetArbiter)
{
    Vec3 const u5 = tilt5(kGamma100Beta, 5.0);
    EXPECT_LT(relativeError(fastField({1.0, 0.0, 0.0}, u5, 8),
                            {1.281766386, 0.0, -0.111995445}), 1.0e-2);
    EXPECT_LT(relativeError(fastField({0.0, 0.5, 0.0}, u5, 8),
                            {0.0, 1.116464807, 0.0}), 1.0e-2);
    EXPECT_LT(relativeError(fastField({-0.6, 0.4, -0.3}, u5, 8),
                            {-0.887934812, 0.611520131, 0.077096379}), 1.0e-2);
    EXPECT_LT(relativeError(fastField({0.0, 0.0, 0.5}, u5, 8),
                            {-0.084033311, 0.0, 0.008186130}), 1.0e-2);
}

//! Closed-form constant-density polyhedron field of the contracted support box.
TEST(NearShellAccuracy, ExactObliqueBoxMatchesClosedForm)
{
    using picBunch::pbfast::detail::exactObliqueBox;
    Vec3 const u = tilt5(kGamma100Beta, 5.0);
    Vec3 const cells{1.0, 1.0, 1.0};
    EXPECT_LT(relativeErrorV(exactObliqueBox({1.0, 0.0, 0.0}, cells, u),
                             {6.56301472843, 0.0, -0.574059966302}), 1.0e-10);
    EXPECT_LT(relativeErrorV(exactObliqueBox({0.0, 0.5, 0.0}, cells, u),
                             {0.0, 3.1858879439, 0.0}), 1.0e-10);
    EXPECT_LT(relativeErrorV(exactObliqueBox({-0.6, 0.4, -0.3}, cells, u),
                             {-3.7397710657, 2.40940209431, 0.326850093195}), 1.0e-10);
    EXPECT_LT(relativeErrorV(exactObliqueBox({0.0, 0.0, 0.5}, cells, u),
                             {-5.34610492868e-05, 0.0, 0.000464413840951}), 1.0e-10);
}

//! The oblique order is a compatibility knob only: the ray path must not move.
TEST(NearShellAccuracy, ObliquePathIndependentOfOrder)
{
    Vec3 const u = oblique45(kGamma100Beta);
    for(Vec3 const obs : {Vec3{-0.6, 0.4, -0.3}, Vec3{1.0, 0.0, 0.0}, Vec3{0.0, 0.5, 0.0}})
    {
        Vec3 const a = fastField(obs, u, 4);
        Vec3 const b = fastField(obs, u, 12);
        EXPECT_LT(relativeErrorV(a, b), 1.0e-12);
    }
}

// ---- independent indfield golden data at near distances -------------------
namespace
{
    struct GoldenRow
    {
        double gamma = 0.0;
        std::string config;
        double d = 0.0;
        std::array<double, 3> R0{};
        std::array<double, 3> u{};
        std::array<double, 3> E{};
        std::array<double, 3> B{};
        int converged = 0;
        //! Reference convergence delta (relative), for the ray rows.
        double error = 0.0;
    };

    std::vector<GoldenRow> loadNearDistanceGolden()
    {
        std::vector<GoldenRow> rows;
        std::ifstream f(std::string(PICBUNCH_TEST_DATA_DIR) + "/near_distance_golden.csv");
        if(!f)
            return rows;
        std::string line;
        std::getline(f, line); // header
        while(std::getline(f, line))
        {
            if(line.empty())
                continue;
            std::stringstream ss(line);
            std::string cell;
            std::vector<std::string> tok;
            while(std::getline(ss, cell, ','))
                tok.push_back(cell);
            if(tok.size() < 16)
                continue;
            GoldenRow r;
            // gamma,config,d,x,y,z,ux,uy,uz,Ex,Ey,Ez,Bx,By,Bz,converged,...
            r.gamma = std::stod(tok[0]);
            r.config = tok[1];
            r.d = std::stod(tok[2]);
            for(int a = 0; a < 3; ++a)
            {
                r.R0[a] = std::stod(tok[3 + a]);
                r.u[a] = std::stod(tok[6 + a]);
                r.E[a] = std::stod(tok[9 + a]);
                r.B[a] = std::stod(tok[12 + a]);
            }
            r.converged = std::stoi(tok[15]);
            if(tok.size() > 16)
                r.error = std::stod(tok[16]);
            rows.push_back(r);
        }
        return rows;
    }

    double relativeErrorA(std::array<double, 3> const& v, std::array<double, 3> const& ref)
    {
        return relativeError(Vec3{v[0], v[1], v[2]}, ref);
    }
} // namespace

//! Near-distance (d=0.05..2 cells) point checks against the independent
//! indfield reference, not NearReference.  A zero convergence flag in the
//! golden data is a hard failure: the arbiter must have converged.
TEST(NearShellAccuracy, NearDistanceMatchesIndfieldGolden)
{
    auto const rows = loadNearDistanceGolden();
    ASSERT_FALSE(rows.empty()) << "missing tests/data/near_distance_golden.csv";
    int inSupport = 0;
    int outSupport = 0;
    for(GoldenRow const& r : rows)
    {
        ASSERT_EQ(r.converged, 1)
            << "indfield reference did not converge for gamma=" << r.gamma
            << " d=" << r.d;
        picBunch::pbfast::Settings settings;
        settings.order = 5;
        settings.obliqueOrder = 8;
        // The exact evaluator must not depend on this knob.
        settings.tolerance = 1.0e-6;
        Vec3 E, B;
        picBunch::pbfast::shape_field(
            Vec3{r.R0[0], r.R0[1], r.R0[2]}, Vec3{}, Vec3{r.u[0], r.u[1], r.u[2]},
            1.0, 1.0, 1.0, settings, E, B);
        double const errE = relativeErrorA({E.x, E.y, E.z}, r.E);
        double const errB = relativeErrorA({B.x, B.y, B.z}, r.B);
        double const err = std::max(errE, errB);
        double const s0 = r.R0[0];
        double const s1 = r.R0[1];
        double const s2 = r.R0[2];
        bool const inside = std::abs(s0) < 1.5 && std::abs(s1) < 1.5 && std::abs(s2) < 1.5;
        // The indfield golden is converged to 1e-4, but its own
        // order-estimator underestimates the gamma-10 oblique residual by up
        // to ~4e-4 at d<=0.25; the in-support bound is reference-limited.
        double const bound = inside ? 2.0e-3 : 1.0e-2;
        if(inside)
            ++inSupport;
        else
            ++outSupport;
        EXPECT_LT(err, bound) << "gamma=" << r.gamma << " d=" << r.d
                              << " inside=" << inside << " errE=" << errE
                              << " errB=" << errB;
    }
    EXPECT_GT(inSupport, 0);
    EXPECT_GT(outSupport, 0);
}

//! gamma-100 oblique in-support truth from the direct d-space ray quadrature
//! (tools/near-shell/ray_reference.cpp), independent of the s-space moment
//! method under test.  The `oblique-ray` rows store the relative delta between
//! the mandated 128x256x10 rule and a 256x512x10 convergence run.  A
//! non-converged reference or a larger than 1e-3 agreement gap is a hard
//! failure.
TEST(NearShellAccuracy, Gamma100ObliqueRayReference)
{
    auto const rows = loadNearDistanceGolden();
    int checked = 0;
    for(GoldenRow const& r : rows)
    {
        if(r.config != "oblique-ray")
            continue;
        ASSERT_EQ(r.converged, 1) << "ray reference did not converge at d=" << r.d;
        ASSERT_LE(r.error, 1.0e-3)
            << "ray reference convergence delta too large at d=" << r.d
            << " delta=" << r.error;
        picBunch::pbfast::Settings settings;
        settings.order = 5;
        settings.obliqueOrder = 8;
        settings.tolerance = 1.0e-3;
        Vec3 E, B;
        picBunch::pbfast::shape_field(
            Vec3{r.R0[0], r.R0[1], r.R0[2]}, Vec3{}, Vec3{r.u[0], r.u[1], r.u[2]},
            1.0, 1.0, 1.0, settings, E, B);
        double const errE = relativeErrorA({E.x, E.y, E.z}, r.E);
        double const errB = relativeErrorA({B.x, B.y, B.z}, r.B);
        double const err = std::max(errE, errB);
        EXPECT_LE(err, 1.0e-3) << "gamma=" << r.gamma << " d=" << r.d
                               << " errE=" << errE << " errB=" << errB;
        ++checked;
    }
    EXPECT_EQ(checked, 4) << "missing gamma-100 oblique-ray golden rows";
}

//! A direct ComputeOptions fastOrder above the 32-slot device table must not
//! silently under-fill.  The host remainder caps at the table size, and an
//! out-of-support pair above the cap routes to the exact evaluator, so the
//! two requests stay within round-off instead of reading past the table.
TEST(NearShellAccuracy, FastOrderAboveCapStaysBounded)
{
    static_assert(
        picBunch::pbfast::kQuadratureOrderCap <= picBunch::pbfast::DeviceParams::kMaxOrder,
        "host quadrature cap exceeds the device xi/wi table");
    Vec3 const u = oblique45(kGamma10Beta);
    picBunch::pbfast::Settings capped;
    capped.order = 32;
    capped.obliqueOrder = 8;
    picBunch::pbfast::Settings oversized;
    oversized.order = 64;
    oversized.obliqueOrder = 8;
    for(Vec3 const obs : {Vec3{2.0, 0.5, 0.0}, Vec3{0.0, 2.5, 0.0}})
    {
        Vec3 Ea, Ba, Eb, Bb;
        picBunch::pbfast::shape_field(obs, Vec3{}, u, 1.0, 1.0, 1.0, capped, Ea, Ba);
        picBunch::pbfast::shape_field(obs, Vec3{}, u, 1.0, 1.0, 1.0, oversized, Eb, Bb);
        EXPECT_LT(relativeErrorV(Ea, Eb), 1.0e-9);
        EXPECT_LT(relativeErrorV(Ba, Bb), 1.0e-9);
    }
}

// ---- adaptive oblique order ------------------------------------------------
#include "picBunch/detail/ObliqueOrder.hpp"
#if defined(PICBUNCH_BACKEND_CUDA)
#    include "picBunch/detail/FastNearDevice.hpp"
#endif

TEST(NearShellAccuracy, AutoObliqueOrderRule)
{
    using namespace picBunch::pbfast;
    EXPECT_EQ(autoObliqueOrder(1.0), 4);
    EXPECT_EQ(autoObliqueOrder(10.0), 4);
    EXPECT_EQ(autoObliqueOrder(11.2), 4);
    EXPECT_EQ(autoObliqueOrder(20.0), 6);
    EXPECT_EQ(autoObliqueOrder(41.0), 8);
    EXPECT_EQ(autoObliqueOrder(100.0), 8);
    EXPECT_EQ(normalizeObliqueOrder(1), 4);
    EXPECT_EQ(normalizeObliqueOrder(5), 6);
    EXPECT_EQ(normalizeObliqueOrder(7), 8);
    EXPECT_EQ(normalizeObliqueOrder(99), 12);
}

TEST(NearShellAccuracy, AutoGamma100ObliqueMeets1e2)
{
    Vec3 const u = oblique45(kGamma100Beta);
    for(Vec3 const obs : {Vec3{-0.6, 0.4, -0.3}, Vec3{0.2, -0.3, 0.4}, Vec3{0.0, 0.5, 0.0}})
    {
        Vec3 const fast = fastField(obs, u, picBunch::pbfast::autoObliqueOrder(100.0));
        Vec3 ref;
        if(obs.x < -0.5 && obs.z < 0.0)
            ref = {-0.18358776, 0.45896317, 0.18266562};
        else if(obs.x > 0.1)
            ref = {-0.19206878, -0.55226776, 0.19297211};
        else
            ref = {0.0, 1.135118460, 0.0};
        EXPECT_LT(relativeErrorV(fast, ref), 1.0e-2) << obs.x << "," << obs.y << "," << obs.z;
    }
}

TEST(NearShellAccuracy, AutoGamma10KeepsOrder4)
{
    EXPECT_EQ(picBunch::pbfast::autoObliqueOrder(10.0), 4);
    Vec3 const u = oblique45(kGamma10Beta);
    EXPECT_LT(relativeErrorV(fastField({-0.6, 0.4, -0.3}, u, 4),
                             {-0.210369983, 0.460777915, 0.157158018}), 1.0e-2);
    EXPECT_LT(relativeErrorV(fastField({0.0, 0.5, 0.0}, u, 4),
                             {0.0, 1.108450387, 0.0}), 1.0e-2);
}

//! gamma-10 control: the auto order and the explicit fast order coincide.
TEST(NearShellAccuracy, Gamma10AutoEqualsExplicitFastOrder)
{
    Vec3 const u = oblique45(kGamma10Beta);
    for(Vec3 const obs : {Vec3{-0.6, 0.4, -0.3}, Vec3{0.0, 0.5, 0.0}})
    {
        Vec3 const a = fastField(obs, u, 4);
        Vec3 const b = fastField(obs, u, picBunch::pbfast::autoObliqueOrder(10.0));
        EXPECT_LT(relativeErrorV(a, b), 1.0e-12);
    }
}

#if defined(PICBUNCH_BACKEND_CUDA)
namespace
{
    Vec3 deviceField(Vec3 const& obs, Vec3 const& u, int obliqueOrder)
    {
        picBunch::Bunch source;
        source.resize(1);
        source.x[0] = source.y[0] = source.z[0] = 0.0;
        source.ux[0] = u.x;
        source.uy[0] = u.y;
        source.uz[0] = u.z;
        source.weight[0] = 1.0;
        source.charge[0] = 1.0;
        picBunch::pbfast::DeviceParams params;
        params.order = 5;
        params.obliqueOrder = obliqueOrder;
        for(int i = 0; i < params.order; ++i)
        {
            auto const& r = picBunch::pbfast::detail::rule(params.order);
            params.xi[i] = r.x[static_cast<std::size_t>(i)];
            params.wi[i] = r.w[static_cast<std::size_t>(i)];
        }
        for(int slot = 0; slot < picBunch::pbfast::kObliqueOrderCount; ++slot)
        {
            int const n = picBunch::pbfast::kObliqueOrders[slot];
            auto const& r = picBunch::pbfast::detail::rule(n);
            for(int i = 0; i < n; ++i)
            {
                params.oxi[slot][i] = r.x[static_cast<std::size_t>(i)];
                params.owi[slot][i] = r.w[static_cast<std::size_t>(i)];
            }
        }
        for(int c = 0; c < 6; ++c)
            for(int a = 0; a < 3; ++a)
                params.delta[c][a] = 0.0;
        params.unitE = 1.0;
        params.unitB = 1.0;
        auto const out = picBunch::pbfast::evalFastNearDevice(
            source, source, false, 1, 1, 1, obs.x, obs.y, obs.z, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            params);
        return {out[0], out[1], out[2]};
    }
} // namespace

TEST(NearShellAccuracy, DeviceGamma100ObliqueMatchesHost)
{
    if(!picBunch::pbfast::fastNearDeviceAvailable())
        GTEST_SKIP() << "no CUDA device";
    Vec3 const u = oblique45(kGamma100Beta);
    for(Vec3 const obs : {Vec3{-0.6, 0.4, -0.3}, Vec3{0.2, -0.3, 0.4}, Vec3{0.0, 0.5, 0.0}})
    {
        Vec3 const hostE = fastField(obs, u, picBunch::pbfast::autoObliqueOrder(100.0));
        Vec3 const dev = deviceField(obs, u, picBunch::pbfast::autoObliqueOrder(100.0));
        EXPECT_LT(relativeErrorV(dev, hostE), 1.0e-10) << obs.x << "," << obs.y << "," << obs.z;
    }
}

TEST(NearShellAccuracy, DeviceGamma100FiveDegreeMatchesReference)
{
    if(!picBunch::pbfast::fastNearDeviceAvailable())
        GTEST_SKIP() << "no CUDA device";
    Vec3 const u = tilt5(kGamma100Beta, 5.0);
    struct Case
    {
        Vec3 obs;
        std::array<double, 3> ref;
    };
    Case const cases[] = {
        {{1.0, 0.0, 0.0}, {1.281766386, 0.0, -0.111995445}},
        {{0.0, 0.5, 0.0}, {0.0, 1.116464807, 0.0}},
        {{-0.6, 0.4, -0.3}, {-0.887934812, 0.611520131, 0.077096379}},
        {{0.0, 0.0, 0.5}, {-0.084033311, 0.0, 0.008186130}},
    };
    for(auto const& c : cases)
    {
        Vec3 const dev = deviceField(c.obs, u, picBunch::pbfast::autoObliqueOrder(100.0));
        EXPECT_LT(relativeError(dev, c.ref), 1.0e-2);
    }
}

TEST(NearShellAccuracy, DeviceNearDistanceExactMatchesHost)
{
    if(!picBunch::pbfast::fastNearDeviceAvailable())
        GTEST_SKIP() << "no CUDA device";
    std::array<double, 3> const dir{0.6, 0.0, 0.8};
    struct Config
    {
        char const* name;
        double gamma;
        std::array<double, 3> u;
    };
    std::array<Config, 5> const configs = {{
        {"g1", 1.0, {0.0, 0.0, 0.0}},
        {"g10-axial", 10.0, {0.0, 0.0, std::sqrt(99.0)}},
        {"g10-oblique", 10.0,
         {std::sqrt(99.0) / std::sqrt(2.0), 0.0, std::sqrt(99.0) / std::sqrt(2.0)}},
        {"g100-axial", 100.0, {0.0, 0.0, std::sqrt(9999.0)}},
        {"g100-oblique", 100.0,
         {std::sqrt(9999.0) / std::sqrt(2.0), 0.0, std::sqrt(9999.0) / std::sqrt(2.0)}},
    }};
    for(Config const& c : configs)
        for(double d : {0.05, 0.25, 1.0})
        {
            Vec3 const obs{d * dir[0], d * dir[1], d * dir[2]};
            Vec3 const u{c.u[0], c.u[1], c.u[2]};
            Vec3 const hostE = fastField(obs, u, 8);
            Vec3 const dev = deviceField(obs, u, 8);
            // The exact s-space evaluator is bit-stable through gamma 10.  At
            // gamma 100 its own input-to-output condition number is ~1e8 (a
            // 1e-15 relative input nudge moves the field by ~2e-7), so the
            // device/host bound there is the conditioning floor, not 1e-10.
            double const bound = c.gamma >= 100.0 ? 1.0e-6 : 1.0e-10;
            EXPECT_LT(relativeErrorV(dev, hostE), bound)
                << c.name << " d=" << d;
        }
}

TEST(NearShellAccuracy, DeviceGamma10AutoEqualsExplicitOrder4)
{
    if(!picBunch::pbfast::fastNearDeviceAvailable())
        GTEST_SKIP() << "no CUDA device";
    Vec3 const u = oblique45(kGamma10Beta);
    for(Vec3 const obs : {Vec3{-0.6, 0.4, -0.3}, Vec3{0.0, 0.5, 0.0}})
    {
        Vec3 const a = deviceField(obs, u, 4);
        Vec3 const b = deviceField(obs, u, picBunch::pbfast::autoObliqueOrder(10.0));
        EXPECT_EQ(a.x, b.x);
        EXPECT_EQ(a.y, b.y);
        EXPECT_EQ(a.z, b.z);
    }
}
#endif
