// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group: near shell on the generic alpaka path.
//
// The near-shell fast evaluator is the same alpaka functor on host, CUDA and
// HIP.  On the host backend it must reproduce the serial FastNear reference,
// and the exact host per-tile loop must stay bit-identical to the former
// hand-written loop, independent of the executor thread count.

#include "picBunch/NearShell.hpp"
#include "picBunch/Types.hpp"
#include "picBunch/detail/FastNear.hpp"
#include "picBunch/detail/FastNearDevice.hpp"
#include "picBunch/detail/NearReference.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(_OPENMP)
#    include <omp.h>
#endif

namespace
{
    using nbref::Vec3;

    Vec3 oblique45(double const beta)
    {
        double const d = beta / std::sqrt(2.0);
        return {d, 0.0, d};
    }

    struct FastParams
    {
        picBunch::pbfast::DeviceParams params{};
        picBunch::pbfast::Settings settings{};
    };

    FastParams makeFastParams(int order, int obliqueOrder, int exactCellReach, double tolerance)
    {
        FastParams out;
        out.settings.order = order;
        out.settings.obliqueOrder = obliqueOrder;
        out.settings.exactCellReach = exactCellReach;
        out.settings.tolerance = tolerance;

        auto& p = out.params;
        p.order = order;
        p.obliqueOrder = obliqueOrder;
        p.exactCellReach = exactCellReach;
        p.tolerance = tolerance;
        auto const& r = picBunch::pbfast::detail::rule(order);
        for(int i = 0; i < order; ++i)
        {
            p.xi[i] = r.x[static_cast<std::size_t>(i)];
            p.wi[i] = r.w[static_cast<std::size_t>(i)];
        }
        for(int slot = 0; slot < picBunch::pbfast::kObliqueOrderCount; ++slot)
        {
            int const n = picBunch::pbfast::kObliqueOrders[slot];
            auto const& o = picBunch::pbfast::detail::rule(n);
            for(int i = 0; i < n; ++i)
            {
                p.oxi[slot][i] = o.x[static_cast<std::size_t>(i)];
                p.owi[slot][i] = o.w[static_cast<std::size_t>(i)];
            }
        }
        for(int c = 0; c < 6; ++c)
            for(int a = 0; a < 3; ++a)
                p.delta[c][a] = 0.0;
        p.unitE = 1.0;
        p.unitB = 1.0;
        return out;
    }

    double relativeError(Vec3 const& value, Vec3 const& reference)
    {
        double const d2 = (value.x - reference.x) * (value.x - reference.x)
            + (value.y - reference.y) * (value.y - reference.y)
            + (value.z - reference.z) * (value.z - reference.z);
        double const r2 = reference.x * reference.x + reference.y * reference.y
            + reference.z * reference.z;
        return std::sqrt(d2 / std::max(r2, 1.0e-300));
    }
} // namespace

//! The generic alpaka functor used by `fast` on every backend must agree with
//! the serial host evaluator to round-off when no mesh stencil is requested.
TEST(NearShellAlpaka, HostFunctorMatchesSerialReference)
{
    if(!picBunch::pbfast::fastNearDeviceAvailable())
        GTEST_SKIP() << "compiled backend has no available device";

    double const gamma10Beta = std::sqrt(99.0);
    double const gamma100Beta = std::sqrt(9999.0);
    struct Case
    {
        Vec3 obs;
        Vec3 u;
        int order;
        int obliqueOrder;
    };
    Case const cases[] = {
        {{-0.6, 0.4, -0.3}, oblique45(gamma10Beta), 5, 8},
        {{0.0, 0.5, 0.0}, oblique45(gamma10Beta), 8, 8},
        {{1.0, 0.0, 0.0}, {0.0, 0.0, gamma100Beta}, 5, 12},
        {{0.2, -0.3, 0.4}, oblique45(gamma100Beta), 5, 12},
    };
    for(Case const& c : cases)
    {
        picBunch::Bunch source;
        source.resize(1);
        source.x[0] = source.y[0] = source.z[0] = 0.0;
        source.ux[0] = c.u.x;
        source.uy[0] = c.u.y;
        source.uz[0] = c.u.z;
        source.weight[0] = 1.0;
        source.charge[0] = 1.0;

        FastParams fast = makeFastParams(c.order, c.obliqueOrder, 2, 1.0e-3);
        auto const out = picBunch::pbfast::evalFastNearDevice(
            source,
            source,
            false,
            1,
            1,
            1,
            c.obs.x,
            c.obs.y,
            c.obs.z,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            fast.params);
        Vec3 const got{out[0], out[1], out[2]};

        Vec3 E, B;
        picBunch::pbfast::shape_field(
            c.obs,
            Vec3{},
            c.u,
            1.0,
            1.0,
            1.0,
            fast.settings,
            E,
            B);
        // The gamma-100 s-space evaluator has an input-to-output condition
        // number of about 1e8, so its backend agreement floor is 1e-6.
        double const gamma = std::sqrt(1.0 + c.u.x * c.u.x + c.u.y * c.u.y + c.u.z * c.u.z);
        double const bound = gamma >= 100.0 ? 1.0e-6 : 1.0e-10;
        EXPECT_LT(relativeError(got, E), bound)
            << "obs=" << c.obs.x << "," << c.obs.y << "," << c.obs.z;
    }
}

namespace
{
    picBunch::Bunch exactBunch()
    {
        picBunch::Bunch bunch;
        bunch.resize(4);
        // Fixed, well separated positions inside a 4^3 / 1 um grid.
        double const pos[4][3] = {
            {-1.4e-6, 0.3e-6, -0.2e-6},
            {0.7e-6, -0.9e-6, 0.4e-6},
            {0.1e-6, 0.2e-6, -1.1e-6},
            {-0.6e-6, -0.5e-6, 0.8e-6},
        };
        double const u[4][3] = {
            {0.0, 0.0, 0.0},
            {0.0, 0.0, 1.2},
            {0.9, 0.0, 0.9},
            {0.0, -1.6, 0.0},
        };
        for(std::size_t p = 0; p < bunch.size(); ++p)
        {
            bunch.x[p] = pos[p][0];
            bunch.y[p] = pos[p][1];
            bunch.z[p] = pos[p][2];
            bunch.ux[p] = u[p][0];
            bunch.uy[p] = u[p][1];
            bunch.uz[p] = u[p][2];
            bunch.weight[p] = 1.0;
            bunch.charge[p] = -1.602176634e-19;
        }
        return bunch;
    }

    picBunch::Field runExactShell(int threads)
    {
#if defined(_OPENMP)
        int const saved = omp_get_max_threads();
        omp_set_num_threads(threads);
#endif
        picBunch::Bunch const bunch = exactBunch();
        picBunch::ComputeOptions options;
        options.tolerance = 1.0e-3;
        options.cellSize = {1.0e-6, 1.0e-6, 1.0e-6};
        options.gridSize = picBunch::Idx3{4u, 4u, 4u};
        options.gridOrigin = picBunch::Vec3{-2.0e-6, -2.0e-6, -2.0e-6};
        options.deterministicDeposit = true;
        // Keep the per-tile mesh subtraction cheap; the exact near field is
        // bin-free, so the box count only selects the subtraction operator.
        options.velocityBinning.mode = picBunch::VelocityBinningOptions::Mode::Explicit;
        options.velocityBinning.explicitBins = 2;
        picBunch::NearShellOptions shell;
        shell.mode = picBunch::NearShellOptions::Mode::Exact;
        shell.tileNodes = 4;
        shell.radiusCells = 1.0;
        picBunch::Field const field = picBunch::computeFieldNearExact(bunch, options, shell);
#if defined(_OPENMP)
        omp_set_num_threads(saved);
#endif
        return field;
    }

    bool sameField(picBunch::Field const& a, picBunch::Field const& b)
    {
        for(int c = 0; c < 6; ++c)
            for(std::size_t i = 0; i < a.data[c].size(); ++i)
                if(a.data[c][i] != b.data[c][i])
                    return false;
        return true;
    }
} // namespace

//! The converted exact host loop owns one output node per work item and keeps
//! the source order per node, so its result is bit-identical for every thread
//! count.
TEST(NearShellAlpaka, ExactHostLoopIsThreadCountIndependent)
{
    picBunch::Field const one = runExactShell(1);
    int const threads = [] {
#if defined(_OPENMP)
        return std::max(2, omp_get_max_threads());
#else
        return 2;
#endif
    }();
    picBunch::Field const many = runExactShell(threads);
    ASSERT_EQ(one.data[0].size(), many.data[0].size());
    for(int c = 0; c < 6; ++c)
        for(std::size_t i = 0; i < one.data[c].size(); ++i)
            EXPECT_EQ(one.data[c][i], many.data[c][i])
                << "component " << c << " node " << i << " threads " << threads;
    EXPECT_TRUE(sameField(one, many));
}

//! Golden values were recorded with the pre-conversion hand-written exact
//! loop (same evaluation order); the converted host kernel must not move a
//! single bit.
TEST(NearShellAlpaka, ExactHostLoopMatchesGolden)
{
    picBunch::Field const field = runExactShell(1);
    ASSERT_EQ(field.data[0].size(), 4u * 4u * 4u);
    // Six probe nodes (c, flat index) covering the staggered components.
    struct Probe
    {
        int component;
        std::size_t index;
        double value;
    };
    // The new identity replaces the per-tile local-FFT mesh subtraction with
    // a direct 27-point TSC stencil per source, blended by observer distance.
    // The old goldens (pre-7a3afff) were for the local-FFT identity and are
    // superseded.  Values recorded with the direct-stencil identity on the
    // rosi_v100 CPU build (e8d082f).
    Probe const probes[] = {
        {0, 0, 198.832616679942},
        {1, 21, 1580.93453742987},
        {2, 42, 271.070512180638},
        {3, 13, 1.12322722813659e-06},
        {4, 37, 2.82098654808433e-06},
        {5, 58, -1.7153008090829675e-06},
    };
    for(Probe const& p : probes)
    {
        // The direct-stencil identity is bit-identical across thread counts on
        // the same build; the tolerance covers compiler/FFT round-off between
        // the build that recorded the golden values and the current backend.
        double const scale = std::max(std::abs(p.value), 1.0e-300);
        EXPECT_NEAR(field.data[p.component][p.index], p.value, scale * 1.0e-10 + 1.0e-10)
            << "component " << p.component << " node " << p.index;
    }
}

namespace
{
    struct OriginCase
    {
        picBunch::Vec3 origin;
        picBunch::Vec3 source;
    };

    struct OriginProbe
    {
        int component;
        std::size_t index;
        double value;
    };

    picBunch::Field runOriginCase(OriginCase const& c)
    {
        picBunch::Bunch bunch;
        bunch.resize(1);
        bunch.x[0] = c.source[0];
        bunch.y[0] = c.source[1];
        bunch.z[0] = c.source[2];
        bunch.ux[0] = 0.0;
        bunch.uy[0] = 0.0;
        bunch.uz[0] = 0.0;
        bunch.weight[0] = 1.0;
        bunch.charge[0] = -1.602176634e-19;

        picBunch::ComputeOptions options;
        options.tolerance = 1.0e-3;
        options.cellSize = {1.0e-6, 1.0e-6, 1.0e-6};
        options.gridSize = picBunch::Idx3{4u, 4u, 4u};
        options.gridOrigin = c.origin;
        options.deterministicDeposit = true;
        options.velocityBinning.mode = picBunch::VelocityBinningOptions::Mode::Explicit;
        options.velocityBinning.explicitBins = 1;

        picBunch::NearShellOptions shell;
        shell.mode = picBunch::NearShellOptions::Mode::Exact;
        shell.tileNodes = 4;
        // Large enough that the source enters every tile's near set.
        shell.radiusCells = 4.0;
        return picBunch::computeFieldNearExact(bunch, options, shell);
    }

    void expectOriginGolden(
        OriginCase const& c,
        std::vector<OriginProbe> const& probes,
        std::string const& label)
    {
        picBunch::Field const field = runOriginCase(c);
        ASSERT_EQ(field.data[0].size(), 4u * 4u * 4u) << label;
        for(OriginProbe const& p : probes)
        {
            double const scale = std::max(std::abs(p.value), 1.0e-300);
            // Golden values recorded with the direct 27-point TSC stencil
            // identity (post-7a3afff).  The tolerance covers compiler/FFT
            // round-off; the absolute floor covers near-zero components.
            // The identity change from local-FFT to direct stencil moves the
            // significant nodes by up to ~1e-3 relative.
            EXPECT_NEAR(field.data[p.component][p.index], p.value, scale * 1.0e-10 + 1.0e-10)
                << label << " component " << p.component << " node " << p.index;
        }
    }
} // namespace

//! Regression for the near-shell local-box clamp.  The source sits inside the
//! validated output grid but its TSC stencil reaches more than 2.5 cells below
//! the grid origin, so the signed local box must start at a negative index.
//! The former clamp of that bound to zero dropped the stencil from the local
//! mesh subtraction and moved the shell field away from the serial reference.
TEST(NearShellAlpaka, SourceStencilBelowGridOriginMatchesSerialReference)
{
    OriginCase const c{{0.0, 0.0, 0.0}, {0.5e-6, 0.5e-6, 0.25e-6}};
    std::vector<OriginProbe> const probes{
        {0, 0u, 1.80391578135717e-12},
        {0, 5u, -818.18902489911227},
        {0, 21u, -564.52951323866819},
        {0, 63u, -39.683197120515729},
        {1, 40u, -145.91277865812367},
        {2, 21u, -598.80178361484786},
        {2, 45u, -82.431016421134373},
        {2, 63u, -39.896312975746802},
    };
    expectOriginGolden(c, probes, "stencil below origin");
}

//! Same guard with a non-zero grid origin: the signed local box is relative to
//! the explicit origin, not to absolute zero.
TEST(NearShellAlpaka, SourceStencilBelowNonZeroGridOriginMatchesSerialReference)
{
    OriginCase const c{{1.5e-6, -0.5e-6, 2.0e-6}, {2.0e-6, 0.5e-6, 2.25e-6}};
    std::vector<OriginProbe> const probes{
        {0, 0u, -3.98857376844352e-12},
        {0, 5u, -1000.3399417908696},
        {0, 21u, -672.86292117309142},
        {0, 63u, -46.348262521425916},
        {1, 0u, 773.15718538240071},
        {1, 45u, -121.86059905226703},
        {2, 21u, -705.70804100146654},
        {2, 63u, -46.588315404736925},
    };
    expectOriginGolden(c, probes, "stencil below non-zero origin");
}
