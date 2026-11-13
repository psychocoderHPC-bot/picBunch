// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Exact per-pair near shell.  The total field is assembled as
//
//   F = F_mesh(all) - F_mesh(near) + F_exact(near)
//
// The subtraction is not a direct sum: each output tile runs its own local,
// zero-padded FFT containing every source within the shell radius of that
// tile.  Because the mesh convolution is linear and the local box contains
// those sources, the local result equals their contribution to the global
// mesh field (padding 2 removes the periodic image).  F_exact is the vendored
// FP64 TSC-convolution reference, evaluated per (source, node) pair.

#include "picBunch/NearShell.hpp"

#include "picBunch/Compute.hpp"
#include "picBunch/VelocityMoments.hpp"
#include "picBunch/detail/FastNear.hpp"
#include "picBunch/detail/FastNearDevice.hpp"
#include "picBunch/detail/NearReference.hpp"
#include "picBunch/detail/NearShellGuard.hpp"
#include "picBunch/detail/NearTileEstimate.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace picBunch
{
    namespace
    {
        nbref::Settings exactSettings()
        {
            // Pin of the validated C++ reference (tools/neural-bunch/reference/
            // cpp/neural_reference.hpp).  Do not tune per case: the convergence
            // study that produced 1.6e-14 agreement used these values.
            nbref::Settings s;
            s.order = 4;            // starting low Gauss order (4 vs 6 pair)
            s.tol = 1e-7;           // order-pair convergence tolerance
            s.max_level = 5;        // maximum per-axis grading level
            s.max_order = 16;       // maximum Gauss order
            s.sphere_n = 10;        // in-support angular order, start
            s.sphere_max_n = 32;    // in-support angular order, maximum
            s.sphere_radial_order = 4;
            return s;
        }

        Bunch selectSources(Bunch const& bunch, std::vector<std::uint32_t> const& index)
        {
            Bunch out;
            out.resize(index.size());
            for(std::size_t k = 0; k < index.size(); ++k)
            {
                std::size_t const p = index[k];
                out.x[k] = bunch.x[p];
                out.y[k] = bunch.y[p];
                out.z[k] = bunch.z[p];
                out.ux[k] = bunch.ux[p];
                out.uy[k] = bunch.uy[p];
                out.uz[k] = bunch.uz[p];
                out.weight[k] = bunch.weight[p];
                out.charge[k] = bunch.charge[p];
                out.restMass[k] = bunch.restMass.empty() ? 0.0 : bunch.restMass[p];
            }
            return out;
        }

        //! Spread the low 10 bits of one coordinate over every third bit.
        inline std::uint32_t mortonExpand(std::uint32_t x)
        {
            x &= 0x3ffu;
            x = (x | (x << 16u)) & 0x030000ffu;
            x = (x | (x << 8u)) & 0x0300f00fu;
            x = (x | (x << 4u)) & 0x030c30c3u;
            x = (x | (x << 2u)) & 0x09249249u;
            return x;
        }

        //! 30-bit Morton code of a 10-bit-per-axis cell coordinate.
        inline std::uint32_t mortonCode(std::uint32_t x, std::uint32_t y, std::uint32_t z)
        {
            return mortonExpand(x) | (mortonExpand(y) << 1u) | (mortonExpand(z) << 2u);
        }

        /*! Host queue for the per-tile exact near-field kernel.
         *
         * The near shell works on host ``std::vector`` field storage even when
         * the library targets CUDA or HIP, so the kernel runs on the host API
         * and the data never leaves the host.  A dedicated blocking queue keeps
         * the launches independent of the field solver's device.
         */
        inline auto& nearShellHostQueue()
        {
            static auto selector
                = alpaka::onHost::makeDeviceSelector(alpaka::api::host, alpaka::deviceKind::cpu);
            static auto device = selector.makeDevice(0);
            static auto queue = device.makeQueue();
            return queue;
        }

        template<typename TKernel, typename... TArgs>
        void runNearShellHostKernel(std::size_t count, TKernel const& kernel, TArgs... args)
        {
            if(count == 0u)
                return;
            std::uint32_t const frame = 64u;
            auto const blocks = static_cast<std::uint32_t>((count + frame - 1u) / frame);
            auto& queue = nearShellHostQueue();
            queue.enqueue(alpaka::onHost::FrameSpec{blocks, frame}, kernel, args...);
            alpaka::onHost::wait(queue);
        }

        //! Dense 1-D span over one field component, local-mesh component or
        //! particle property.
        //!
        //! ``Field`` keeps its six components in six separate ``std::vector``
        //! allocations, so the correction and local terms are six independent
        //! dense x-fastest 1-D views here.  A single 2-D ``{6, nodes}`` view
        //! is only possible for the G2 device accumulator, whose storage is
        //! one contiguous ``6*nodes`` block.  This host path is scalar (one
        //! index per thread via ``makeIdxMap`` with a scalar ``IdxRange``), so
        //! only element alignment is required; no vector-load alignment is
        //! assumed from the allocation.
        //! Local cell indices live in the distributed global cell space, so
        //! the span extents and the local origin use 64-bit indices.
        using SpanIndex = alpaka::Vec<std::size_t, 1u>;
        using LocalCellVec = alpaka::Vec<std::size_t, 3u>;
        //! Signed local-cell offset.  The subtraction box may start below the
        //! global grid origin when a near source's TSC stencil reaches there;
        //! a signed type lets the box include that part instead of dropping it.
        using LocalOffsetVec = alpaka::Vec<std::int64_t, 3u>;
        using Span1D = alpaka::MdSpan<double, SpanIndex, SpanIndex, alpaka::Alignment<>>;
        using Span1DConst = alpaka::MdSpan<double const, SpanIndex, SpanIndex, alpaka::Alignment<>>;
        static_assert(alpaka::concepts::IMdSpan<Span1D>);
        static_assert(alpaka::concepts::IMdSpan<Span1DConst>);
        //! The span extent type must agree with the local index vector type;
        //! mixing 32-bit extents with 64-bit indices would narrow silently.
        static_assert(std::is_same_v<
                      typename std::remove_cvref_t<decltype(std::declval<Span1D const&>().getExtents())>::value_type,
                      LocalCellVec::value_type>);
        static_assert(std::is_same_v<
                      typename std::remove_cvref_t<decltype(std::declval<Span1DConst const&>().getExtents())>::value_type,
                      LocalCellVec::value_type>);

        //! POD arguments of the exact per-tile host kernel.
        struct ExactNearHostArgs
        {
            Span1D correction[6]{};
            Span1DConst local[6]{};
            Span1DConst x{};
            Span1DConst y{};
            Span1DConst z{};
            Span1DConst ux{};
            Span1DConst uy{};
            Span1DConst uz{};
            Span1DConst weight{};
            Span1DConst charge{};
            std::size_t nsrc{};
            std::size_t nx{};
            std::size_t ny{};
            std::size_t nz{};
            std::size_t i0{};
            std::size_t j0{};
            std::size_t k0{};
            std::size_t n0{};
            std::size_t n1{};
            LocalOffsetVec localLo{};
            std::size_t nLocalX{};
            std::size_t nLocalY{};
            //! 6, or 3 when the tile is stationary and B is exactly zero.
            std::size_t activeComponents{6u};
            alpaka::Vec<double, 3u> origin{};
            alpaka::Vec<double, 3u> spacing{};
            alpaka::Vec<double, 3u> cells{};
            bool useFast{};
            pbfast::Settings fastSettings{};
            nbref::Settings settings{};
        };

        /*! Exact per-pair near-field evaluation for one output tile.
         *
         * One work item owns one (component, node) slot and accumulates the
         * sources in their stored order.  The summation order is therefore the
         * historical one and independent of the executor's thread count, so the
         * result stays bit-identical to the former hand-written OpenMP loop.
         * The reference evaluators are host-only, hence ALPAKA_FN_HOST.
         */
        struct ExactNearHostKernel
        {
            template<typename TAcc>
            ALPAKA_FN_HOST void operator()(TAcc const& acc, ExactNearHostArgs args) const
            {
                std::size_t const nodes = args.nx * args.ny * args.nz;
                for(auto [idx] : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{args.activeComponents * nodes}))
                {
                    int const c = static_cast<int>(idx / nodes);
                    std::size_t const node = idx - static_cast<std::size_t>(c) * nodes;
                    std::size_t const ii = node % args.nx;
                    std::size_t const jj = (node / args.nx) % args.ny;
                    std::size_t const kk = node / (args.nx * args.ny);

                    std::size_t const i = args.i0 + ii;
                    std::size_t const j = args.j0 + jj;
                    std::size_t const k = args.k0 + kk;
                    std::size_t const gidx = i + args.n0 * (j + args.n1 * k);
                    std::size_t const li = static_cast<std::size_t>(
                        static_cast<std::int64_t>(i) - args.localLo[0]);
                    std::size_t const lj = static_cast<std::size_t>(
                        static_cast<std::int64_t>(j) - args.localLo[1]);
                    std::size_t const lk = static_cast<std::size_t>(
                        static_cast<std::int64_t>(k) - args.localLo[2]);
                    std::size_t const lidx
                        = li + args.nLocalX * (lj + args.nLocalY * lk);
                    double value = -args.local[c][lidx];

                    double const x_obs = args.origin[0] + static_cast<double>(i) * args.spacing[0]
                        + kYeeOffset[c][0] * args.cells[0];
                    double const y_obs = args.origin[1] + static_cast<double>(j) * args.spacing[1]
                        + kYeeOffset[c][1] * args.cells[1];
                    double const z_obs = args.origin[2] + static_cast<double>(k) * args.spacing[2]
                        + kYeeOffset[c][2] * args.cells[2];
                    double const unit = (c < 3) ? kCoulombConstant
                                                : kCoulombConstant / kSpeedOfLight;
                    for(std::size_t q = 0; q < args.nsrc; ++q)
                    {
                        // beta = 0 gives B = 0 exactly; skip those sources in a
                        // mixed tile's B sums.
                        if(c >= 3 && args.ux[q] == 0.0 && args.uy[q] == 0.0
                           && args.uz[q] == 0.0)
                            continue;
                        nbref::Vec3 E, B;
                        if(args.useFast)
                            pbfast::shape_field(
                                {x_obs, y_obs, z_obs},
                                {args.x[q], args.y[q], args.z[q]},
                                {args.ux[q], args.uy[q], args.uz[q]},
                                args.cells[0],
                                args.cells[1],
                                args.cells[2],
                                args.fastSettings,
                                E,
                                B);
                        else
                            nbref::shape_field(
                                {x_obs, y_obs, z_obs},
                                {args.x[q], args.y[q], args.z[q]},
                                {args.ux[q], args.uy[q], args.uz[q]},
                                args.cells[0],
                                args.cells[1],
                                args.cells[2],
                                args.settings,
                                E,
                                B);
                        double const component = (c < 3) ? E[c] : B[c - 3];
                        value += unit * (args.weight[q] * args.charge[q]) * component;
                    }
                    args.correction[c][gidx] = value;
                }
            }
        };

        /*! Resolve the output tile edge, applying the automatic pair-count rule.
         *
         * An explicit positive ``tileNodes`` always wins.  Otherwise the
         * estimator compares the pair count at tiles 2 and 4 and picks 2 only
         * when its work reduction clearly exceeds tile 4's measured per-pair
         * throughput advantage.  This is the measured rule: a cloud whose near
         * set shrinks with the tile uses tile 2, while a cloud already covered
         * by one tile-4 block (or a small grid where the two pair counts are
         * close) keeps tile 4.
         */
        std::size_t resolveNearShellTile(
            Bunch const& bunch,
            Grid const& grid,
            NearShellOptions const& nearShell,
            Vec3 const& cellSize,
            double radius)
        {
            if(nearShell.tileNodes > 0u)
            {
                if(nearShell.resolvedTileNodes)
                    *nearShell.resolvedTileNodes = nearShell.tileNodes;
                return nearShell.tileNodes;
            }
            detail::NearTileEstimate const estimate
                = detail::estimateNearTile(bunch, grid, cellSize, radius);
            if(nearShell.resolvedTileNodes)
                *nearShell.resolvedTileNodes = estimate.tile;
            std::fprintf(
                stderr,
                "near shell: automatic tile=%zu (%s; estimated pairs: tile 2 = %.3g, "
                "tile 4 = %.3g)\n",
                estimate.tile,
                estimate.tile2Wins ? "tile 2 pair saving exceeds the tile 4 throughput gain"
                                   : "pair counts near equal, keeping tile 4",
                estimate.pairs2,
                estimate.pairs4);
            return estimate.tile;
        }
    } // namespace

    double resolveNearShellRadius(ComputeOptions const& options, NearShellOptions const& nearShell)
    {
        if(nearShell.radiusCells > 0.0)
            return nearShell.radiusCells;
        double const tol = options.tolerance;
        double radius = 3.0;
        if(tol < 1.0e-3)
            radius += std::ceil(std::log10(1.0e-3 / tol));
        return std::min(6.0, std::max(3.0, radius));
    }

    Field computeFieldNearExact(
        Bunch const& bunch,
        ComputeOptions const& options,
        NearShellOptions const& nearShell,
        StageTimes* stageTimes,
        VelocityBinning* outBinning,
        VelocityMomentsReport* outReport)
    {
        if(bunch.size() == 0)
            throw std::runtime_error("computeFieldNearExact: empty bunch");

        // The exact near term evaluates every (source, node) pair with the
        // source's own velocity: bin-free.  The mesh subtraction cancels the
        // global mesh bin for bin only if both use the same box geometry and
        // representatives, so the binning is derived once from the full bunch
        // and fixed for every per-tile local mesh.
        VelocityBinning binning;
        bool const momentsRequestedForBinning
            = options.velocityMoments.mode != VelocityMomentsOptions::Mode::Off;
        if(!options.fixedBinning)
        {
            try
            {
                binning = deriveVelocityBinning(
                    bunch, options.tolerance, options.velocityBinning);
            }
            catch(std::runtime_error const& error)
            {
                // Only the model's bin-cap failure may be downgraded.  Any other
                // setup error (invalid input, empty bunch, bad geometry) must
                // still propagate, also when the moment path is enabled.
                if(!momentsRequestedForBinning
                   || std::string(error.what()).find("velocity-bin model requires")
                          == std::string::npos)
                    throw;
                // The moment path does not consume the velocity boxes, but the
                // near-shell mesh subtraction (and the fallback if its probe
                // declines) still needs the same fixed geometry as the global
                // mesh.  Use the largest count the policy allows instead of
                // aborting at the model cap; the moment centre stays pinned to
                // the full-bunch u0, so the subtraction operator is unchanged.
                std::fprintf(
                    stderr,
                    "picBunch: near shell: velocity-bin auto policy exceeded the %zu-box "
                    "cap (%s); using a %zu-box fallback binning for the mesh and trying "
                    "the velocity-moment path\n",
                    options.velocityBinning.maxBins,
                    error.what(),
                    options.velocityBinning.maxBins);
                VelocityBinningOptions capped = options.velocityBinning;
                capped.mode = VelocityBinningOptions::Mode::Explicit;
                capped.explicitBins = capped.maxBins;
                binning = deriveVelocityBinning(bunch, options.tolerance, capped);
            }
        }
        if(outBinning)
            *outBinning = options.fixedBinning ? *options.fixedBinning : binning;

        ComputeOptions plain = options;
        plain.nearShell = NearShellOptions{};
        // The per-tile subtraction and the returned field are full-grid host
        // arrays; the near shell therefore keeps the replicated mesh path even
        // when the outer call supplied a distributed context.
        plain.fftMode = FftMode::Replicated;
        // The base mesh field is itself component-parallel when a distributed
        // context is supplied, and computeField already reduced the six arrays
        // onto every rank.  The near-shell tiles below are then partitioned so
        // each rank touches only its own output nodes.
        if(!options.fixedBinning)
            plain.fixedBinning = binning;
        // Pin the Taylor centre of the velocity-moment path to the full-bunch
        // one.  The per-tile subtraction is the same linear operator applied to
        // the near subset; a centre recomputed per tile would cancel a
        // different approximation and leave a residual inside the shell.
        bool const momentsRequested
            = options.velocityMoments.mode != VelocityMomentsOptions::Mode::Off;
        if(momentsRequested)
            plain.velocityMomentsCentre = velocityMomentsCentre(bunch);
        VelocityMomentsReport baseReport;
        Field field = computeField(
            bunch,
            plain,
            stageTimes,
            nullptr,
            momentsRequested ? &baseReport : nullptr);
        if(outReport)
            *outReport = baseReport;

        // The tiled subtraction must use exactly the velocity model the base
        // mesh used.  When the moment path engaged, run the local mesh at the
        // resolved fixed order; when it fell back (auto probe, size or memory),
        // run the local mesh on the binned path as well.
        ComputeOptions tileOptions = plain;
        if(momentsRequested)
        {
            if(baseReport.used)
            {
                tileOptions.velocityMoments.mode = VelocityMomentsOptions::Mode::Explicit;
                tileOptions.velocityMoments.order
                    = static_cast<std::size_t>(baseReport.order);
            }
            else
            {
                tileOptions.velocityMoments.mode = VelocityMomentsOptions::Mode::Off;
            }
        }

        // Each rank owns a disjoint set of output tiles and writes only the
        // correction there.  A zero correction buffer plus one all-reduce keeps
        // the untouched tiles and the base field out of the summation.
        Field correction;
        correction.grid = field.grid;
        for(auto& component : correction.data)
            component.assign(field.grid.size(), 0.0);

        ParallelContext const* par = options.parallel;
        int const parSize = (par && par->distributed()) ? par->size : 1;
        int const parRank = (par && par->distributed()) ? par->rank : 0;

        Grid const& grid = field.grid;
        double const radius = resolveNearShellRadius(options, nearShell);
        double const cells[3] = {options.cellSize[0], options.cellSize[1], options.cellSize[2]};
        std::size_t const tile
            = resolveNearShellTile(bunch, grid, nearShell, options.cellSize, radius);
        nbref::Settings const settings = exactSettings();
        std::uint64_t pairs = 0;
        std::size_t tileIndex = 0;
        auto const nearStart = std::chrono::steady_clock::now();
        // Resolve the oblique-box policy once.  0 (or a negative value) means
        // `auto`; an explicit value is mapped to an allowed even order in
        // [4, 12].  Under `auto` the per-tile resolution below may raise
        // `tileObliqueOrder` from the largest oblique, support-overlapping
        // source gamma.
        bool const autoObliqueOrderPolicy = nearShell.fastObliqueOrder <= 0;
        int const explicitObliqueOrder = autoObliqueOrderPolicy
            ? pbfast::kObliqueOrderMin
            : pbfast::normalizeObliqueOrder(nearShell.fastObliqueOrder);
        // The device xi/wi table holds kQuadratureOrderCap (32) nodes per
        // axis.  Clamp a programmatically set fastOrder to the table capacity
        // so host and device use the same rule and no slot is left
        // uninitialised.  The CLI and TOML parsers already reject values
        // outside [2, 32]; this guard covers direct ComputeOptions writes.
        int const requestedFastOrder
            = std::clamp(nearShell.fastOrder, 2, pbfast::kQuadratureOrderCap);
        pbfast::Settings fastSettings{requestedFastOrder, explicitObliqueOrder, 5};
        fastSettings.tolerance = options.tolerance;
        bool useFast = nearShell.mode == NearShellOptions::Mode::Fast;
        bool useFastGpu = nearShell.mode == NearShellOptions::Mode::FastGpu;
        // The fast evaluator is the same alpaka functor for every compiled
        // backend (host, CUDA, HIP); only the device and the FrameSpec differ.
        // `fast` uses the compiled backend's device, while `fast-gpu` is an
        // explicit request that falls back to the host evaluator.
        if(nearShell.mode == NearShellOptions::Mode::Fast
           && pbfast::fastNearDeviceAvailable())
        {
            useFastGpu = true;
            useFast = false;
        }
        if(useFastGpu && !pbfast::fastNearDeviceAvailable())
        {
            std::fprintf(
                stderr,
                "near shell: no device for the fast evaluator is available; "
                "falling back to the host fast evaluator\n");
            useFastGpu = false;
            useFast = true;
        }
        pbfast::DeviceParams deviceParams;
        deviceParams.order = fastSettings.order;
        deviceParams.obliqueOrder = explicitObliqueOrder;
        deviceParams.tolerance = options.tolerance;
        deviceParams.exactCellReach = nearShell.exactCellReach;
        {
            auto const& xi = pbfast::detail::rule(fastSettings.order).x;
            auto const& wi = pbfast::detail::rule(fastSettings.order).w;
            for(int i = 0; i < fastSettings.order; ++i)
            {
                deviceParams.xi[i] = xi[static_cast<std::size_t>(i)];
                deviceParams.wi[i] = wi[static_cast<std::size_t>(i)];
            }
            // Precompute every allowed oblique rule once.  The kernel selects
            // the slot matching the tile's resolved order, so no thread ever
            // regenerates a Gauss table.
            for(int slotIndex = 0; slotIndex < pbfast::kObliqueOrderCount; ++slotIndex)
            {
                int const order = pbfast::kObliqueOrders[slotIndex];
                auto const& oxi = pbfast::detail::rule(order).x;
                auto const& owi = pbfast::detail::rule(order).w;
                for(int i = 0; i < order; ++i)
                {
                    deviceParams.oxi[slotIndex][i] = oxi[static_cast<std::size_t>(i)];
                    deviceParams.owi[slotIndex][i] = owi[static_cast<std::size_t>(i)];
                }
            }
            for(int c = 0; c < 6; ++c)
                for(int a = 0; a < 3; ++a)
                    deviceParams.delta[c][a] = kYeeOffset[c][a];
            deviceParams.unitE = kCoulombConstant;
            deviceParams.unitB = kCoulombConstant / kSpeedOfLight;
        }

        auto const n = grid.n;
        auto const& origin = grid.origin;
        auto const& spacing = grid.spacing;

        // Spatial (Morton) order of the sources.  The per-tile lists below
        // then contain near-contiguous runs of the source arrays instead of
        // indices scattered over the whole bunch, so a tile's inner loop reads
        // a compact block from cache and the device can coalesce the loads.
        // The permutation changes the floating-point summation order of the
        // near correction only; the base mesh is still built from the caller's
        // bunch.  Sources are reordered once per near-shell call.
        Bunch nearBunch;
        if(bunch.size() > 1u && bunch.size() <= 0xffffffffull)
        {
            double lo[3] = {bunch.x[0], bunch.y[0], bunch.z[0]};
            double hi[3] = {bunch.x[0], bunch.y[0], bunch.z[0]};
            for(std::size_t p = 1; p < bunch.size(); ++p)
            {
                double const pos[3] = {bunch.x[p], bunch.y[p], bunch.z[p]};
                for(int a = 0; a < 3; ++a)
                {
                    lo[a] = std::min(lo[a], pos[a]);
                    hi[a] = std::max(hi[a], pos[a]);
                }
            }
            std::vector<std::uint64_t> keys(bunch.size());
            for(std::size_t p = 0; p < bunch.size(); ++p)
            {
                double const pos[3] = {bunch.x[p], bunch.y[p], bunch.z[p]};
                std::uint32_t code[3];
                for(int a = 0; a < 3; ++a)
                {
                    double const extent = hi[a] - lo[a];
                    double const t = extent > 0.0 ? (pos[a] - lo[a]) / extent : 0.0;
                    code[a] = static_cast<std::uint32_t>(
                        std::clamp(t * 1023.0 + 0.5, 0.0, 1023.0));
                }
                keys[p] = (static_cast<std::uint64_t>(mortonCode(code[0], code[1], code[2]))
                           << 32u)
                    | static_cast<std::uint64_t>(p);
            }
            std::sort(keys.begin(), keys.end());
            std::vector<std::uint32_t> order(bunch.size());
            for(std::size_t i = 0; i < keys.size(); ++i)
                order[i] = static_cast<std::uint32_t>(keys[i]);
            nearBunch = selectSources(bunch, order);
        }
        else
            nearBunch = bunch;

        // Per-tile near-source lists.  Every source is visited once and
        // inserted into each tile block whose node box lies within the shell
        // radius.  This costs O(sources * neighbouring tiles) instead of the
        // former O(tiles * bucket sources): on a large grid an empty tile now
        // costs nothing, which removes the grid^3 per-tile scan that dominated
        // compact clouds at 128^3.  Sources are appended in spatial order, so
        // each list is already sorted and no per-tile sort is needed.
        std::size_t const tileCount[3] = {
            (n[0] + tile - 1u) / tile,
            (n[1] + tile - 1u) / tile,
            (n[2] + tile - 1u) / tile};
        std::vector<std::vector<std::uint32_t>> tileSources(
            tileCount[0] * tileCount[1] * tileCount[2]);
        for(std::size_t p = 0; p < nearBunch.size(); ++p)
        {
            double const pos[3] = {nearBunch.x[p], nearBunch.y[p], nearBunch.z[p]};
            long qLo[3];
            long qHi[3];
            for(int a = 0; a < 3; ++a)
            {
                double const reach = radius * cells[a];
                double const sp = spacing[a];
                double const tj = sp * static_cast<double>(tile);
                double const u = pos[a] - origin[a];
                long const lo = static_cast<long>(std::floor(
                    (u - reach - static_cast<double>(tile - 1u) * sp) / tj));
                long const hi = static_cast<long>(std::floor((u + reach) / tj));
                // One guard tile each side absorbs floating-point rounding at
                // the range boundary; the exact distance test below decides.
                qLo[a] = std::clamp(lo - 1L, 0L, static_cast<long>(tileCount[a]) - 1L);
                qHi[a] = std::clamp(hi + 1L, 0L, static_cast<long>(tileCount[a]) - 1L);
            }
            for(long qk = qLo[2]; qk <= qHi[2]; ++qk)
                for(long qj = qLo[1]; qj <= qHi[1]; ++qj)
                    for(long qi = qLo[0]; qi <= qHi[0]; ++qi)
                    {
                        std::size_t const i0 = static_cast<std::size_t>(qi) * tile;
                        std::size_t const j0 = static_cast<std::size_t>(qj) * tile;
                        std::size_t const k0 = static_cast<std::size_t>(qk) * tile;
                        std::size_t const i1 = std::min(n[0], i0 + tile);
                        std::size_t const j1 = std::min(n[1], j0 + tile);
                        std::size_t const k1 = std::min(n[2], k0 + tile);
                        double const low[3] = {
                            origin[0] + static_cast<double>(i0) * spacing[0],
                            origin[1] + static_cast<double>(j0) * spacing[1],
                            origin[2] + static_cast<double>(k0) * spacing[2]};
                        double const high[3] = {
                            origin[0] + static_cast<double>(i1 - 1u) * spacing[0],
                            origin[1] + static_cast<double>(j1 - 1u) * spacing[1],
                            origin[2] + static_cast<double>(k1 - 1u) * spacing[2]};
                        double dist2 = 0.0;
                        for(int a = 0; a < 3; ++a)
                        {
                            // Observation points extend half a PIC cell past
                            // the node box; expand the membership box so the
                            // per-pair radial blend can see every source.
                            double const pad = 0.5 * cells[a];
                            double const d = std::max(
                                {low[a] - pad - pos[a], 0.0, pos[a] - (high[a] + pad)});
                            double const dn = d / cells[a];
                            dist2 += dn * dn;
                        }
                        if(dist2 <= radius * radius)
                            tileSources[static_cast<std::size_t>(qi)
                                        + tileCount[0]
                                            * (static_cast<std::size_t>(qj)
                                               + tileCount[1] * static_cast<std::size_t>(qk))]
                                .push_back(static_cast<std::uint32_t>(p));
                    }
        }

        // The 27-point stencil path needs no per-tile FFT.  All of its tiles
        // are collected and evaluated in one device launch, which removes the
        // per-tile allocator and stream-synchronisation cost that dominated
        // large grids (and lets small tiles be used without paying it).
        bool const deviceStencil = useFastGpu
            && (std::getenv("PICBUNCH_NEAR_SHELL_LOCAL_FFT") == nullptr);
        if(deviceStencil && nearBunch.size() > 0xffffffffull)
            throw std::runtime_error(
                "near shell: the batched device path holds source indices in 32 bits");
        std::vector<std::uint32_t> batchIndices;
        std::vector<pbfast::BatchedTile> batchTiles;
        std::vector<pbfast::DeviceParams> batchParams;
        std::vector<double> batchRepU;
        std::uint64_t batchOutSize = 0;
        if(deviceStencil)
        {
            // Binned representative velocity per full-bunch source, resolved
            // once instead of once per tile.
            VelocityBinning const& meshBin
                = options.fixedBinning ? *options.fixedBinning : binning;
            batchRepU.resize(3u * nearBunch.size());
            for(std::size_t p = 0; p < nearBunch.size(); ++p)
            {
                double const coords[3] = {
                    nearBunch.ux[p] * meshBin.basis[0][0]
                        + nearBunch.uy[p] * meshBin.basis[0][1]
                        + nearBunch.uz[p] * meshBin.basis[0][2],
                    nearBunch.ux[p] * meshBin.basis[1][0]
                        + nearBunch.uy[p] * meshBin.basis[1][1]
                        + nearBunch.uz[p] * meshBin.basis[1][2],
                    nearBunch.ux[p] * meshBin.basis[2][0]
                        + nearBunch.uy[p] * meshBin.basis[2][1]
                        + nearBunch.uz[p] * meshBin.basis[2][2]};
                std::size_t const b = meshBin.boxIndex(coords[0], coords[1], coords[2]);
                batchRepU[3u * p] = meshBin.representativeU[b][0];
                batchRepU[3u * p + 1u] = meshBin.representativeU[b][1];
                batchRepU[3u * p + 2u] = meshBin.representativeU[b][2];
            }
        }

        // Stable in-support reference expects cells (dx,dy,dz); picBunch's
        // node spacing is kRefinement * cellSize with kRefinement == 1.
        for(std::size_t k0 = 0; k0 < n[2]; k0 += tile)
        {
            std::size_t const k1 = std::min(n[2], k0 + tile);
            for(std::size_t j0 = 0; j0 < n[1]; j0 += tile)
            {
                std::size_t const j1 = std::min(n[1], j0 + tile);
                for(std::size_t i0 = 0; i0 < n[0]; i0 += tile)
                {
                    std::size_t const thisTile = tileIndex++;
                    if(parSize > 1 && (thisTile % static_cast<std::size_t>(parSize))
                           != static_cast<std::size_t>(parRank))
                        continue; // another rank owns this output tile
                    std::size_t const i1 = std::min(n[0], i0 + tile);

                    // Node bounding box of this tile.
                    double low[3] = {
                        origin[0] + static_cast<double>(i0) * spacing[0],
                        origin[1] + static_cast<double>(j0) * spacing[1],
                        origin[2] + static_cast<double>(k0) * spacing[2]};
                    double high[3] = {
                        origin[0] + static_cast<double>(i1 - 1u) * spacing[0],
                        origin[1] + static_cast<double>(j1 - 1u) * spacing[1],
                        origin[2] + static_cast<double>(k1 - 1u) * spacing[2]};

                    // Near set from the per-source tile lists.  The lists
                    // are already sorted in source index order.
                    auto const& nearIndex = tileSources[
                        static_cast<std::size_t>(i0 / tile)
                        + tileCount[0]
                            * (static_cast<std::size_t>(j0 / tile)
                               + tileCount[1] * static_cast<std::size_t>(k0 / tile))];
                    if(nearIndex.empty())
                        continue;

                    Bunch const near = selectSources(nearBunch, nearIndex);

                    // One coherent oblique-box order per output tile.  Under
                    // `auto`, raise it only for oblique sources whose TSC
                    // support can reach this tile and whose gamma puts the
                    // contracted peak below the order-4 resolution.  Axial
                    // sources are skipped here (their box is exact and does
                    // not use the Duffy order), so the gamma-10 and axial
                    // behaviour is untouched.  The scan is O(near.size()) and
                    // runs once per tile.
                    // Remainder order responds to --tolerance and to the
                    // contracted d/gamma tube of the tile's fastest source.
                    // The host and device dispatches use the same rule; pairs
                    // whose requirement exceeds the cap are routed to the
                    // exact evaluator per (source, node) pair.
                    double tileMaxGamma = 1.0;
                    double tileMaxU2 = 0.0;
                    for(std::size_t q = 0; q < near.size(); ++q)
                    {
                        double const uq2 = near.ux[q] * near.ux[q]
                            + near.uy[q] * near.uy[q] + near.uz[q] * near.uz[q];
                        tileMaxU2 = std::max(tileMaxU2, uq2);
                        if(uq2 > 0.0)
                            tileMaxGamma
                                = std::max(tileMaxGamma, std::sqrt(1.0 + uq2));
                    }
                    // beta = 0 gives B = beta x E = 0 exactly.  A stationary
                    // tile therefore needs only the three electric components;
                    // the device kernel skips the three B slots (left at zero)
                    // and the pair counter reports the reduced component work.
                    int const activeComponents = (tileMaxU2 == 0.0) ? 3 : 6;
                    int const exactReach = pbfast::resolveExactCellReach(
                        nearShell.exactCellReach, options.tolerance, tileMaxGamma);
                    int const tileRemainderOrder = std::min(
                        pbfast::kQuadratureOrderCap,
                        std::max(
                            requestedFastOrder,
                            pbfast::quadratureOrderFor(
                                tileMaxGamma, options.tolerance, requestedFastOrder)));

                    int tileObliqueOrder = explicitObliqueOrder;
                    if(autoObliqueOrderPolicy)
                    {
                        double maxObliqueGamma = 0.0;
                        for(std::size_t q = 0; q < near.size(); ++q)
                        {
                            double const ux = near.ux[q];
                            double const uy = near.uy[q];
                            double const uz = near.uz[q];
                            double const uu2 = ux * ux + uy * uy + uz * uz;
                            if(uu2 == 0.0)
                                continue;
                            double const invLen = 1.0 / std::sqrt(uu2);
                            bool const axial = std::abs(std::abs(ux * invLen) - 1.0) < 1e-12
                                || std::abs(std::abs(uy * invLen) - 1.0) < 1e-12
                                || std::abs(std::abs(uz * invLen) - 1.0) < 1e-12;
                            if(axial)
                                continue;
                            double const pos[3] = {near.x[q], near.y[q], near.z[q]};
                            bool overlaps = true;
                            for(int a = 0; a < 3; ++a)
                            {
                                double const reach = 1.5 * cells[a];
                                if(pos[a] < low[a] - reach || pos[a] > high[a] + reach)
                                {
                                    overlaps = false;
                                    break;
                                }
                            }
                            if(!overlaps)
                                continue;
                            maxObliqueGamma = std::max(maxObliqueGamma, std::sqrt(1.0 + uu2));
                        }
                        tileObliqueOrder = pbfast::autoObliqueOrder(maxObliqueGamma);
                    }
                    // The generic fast evaluator reads DeviceParams on every
                    // backend; the host fast fallback reads tileFastSettings.
                    deviceParams.obliqueOrder = tileObliqueOrder;
                    // Explicitly clamp to the 32-slot table.  tileRemainderOrder
                    // is already bounded by kQuadratureOrderCap, and the
                    // static_assert in FastNearDevice.hpp keeps that cap equal
                    // to the device table size; this makes the fill loop
                    // self-evidently in range.
                    int const deviceOrder
                        = std::min(tileRemainderOrder, pbfast::DeviceParams::kMaxOrder);
                    deviceParams.order = deviceOrder;
                    deviceParams.exactCellReach = exactReach;
                    deviceParams.activeComponents = activeComponents;
                    {
                        auto const& tx = pbfast::detail::rule(deviceOrder).x;
                        auto const& tw = pbfast::detail::rule(deviceOrder).w;
                        for(int i = 0; i < deviceOrder; ++i)
                        {
                            deviceParams.xi[i] = tx[static_cast<std::size_t>(i)];
                            deviceParams.wi[i] = tw[static_cast<std::size_t>(i)];
                        }
                    }
                    // Shipped policy: the observer-containing cell is exact,
                    // the rest stays on the base order-5 remainder.  A tight
                    // global tolerance (<= 1e-4) selects the full-support exact
                    // mode, and a gamma-100 tile is always exact.
                    pbfast::Settings tileFastSettings = fastSettings;
                    tileFastSettings.obliqueOrder = tileObliqueOrder;
                    tileFastSettings.order = tileRemainderOrder;
                    tileFastSettings.exactCellReach = exactReach;

                    // The local FFT subtraction is only needed on the legacy
                    // path.  The direct 27-point TSC stencil performs the same
                    // discrete convolution per (source, node) pair on the
                    // device, so the per-tile FFT (and its plan setup) is
                    // skipped by default.
                    Field local;
                    std::int64_t localLo[3] = {0, 0, 0};
                    std::size_t nLocalX = 0;
                    std::size_t nLocalY = 0;
                    if(!deviceStencil)
                    {
                        // Local node box: the tile plus the full TSC support of
                        // every near source.  It must contain every source used
                        // in the subtraction and every observer in the tile.
                        // The bounds are derived in signed 64-bit space: the
                        // TSC stencil of a near source close to the lower grid
                        // origin starts at a negative global index.  Clamping
                        // that to zero would silently drop the part of the
                        // deposit below the origin.  Instead the local box
                        // starts at the negative index and gets its own origin;
                        // the exact kernel maps global indices with the same
                        // signed offset, so the subtraction stays complete.
                        std::int64_t localHi[3] = {
                            static_cast<std::int64_t>(i1 - 1u),
                            static_cast<std::int64_t>(j1 - 1u),
                            static_cast<std::int64_t>(k1 - 1u)};
                        for(std::size_t q = 0; q < near.size(); ++q)
                        {
                            double const pos[3] = {near.x[q], near.y[q], near.z[q]};
                            for(int a = 0; a < 3; ++a)
                            {
                                double const t = (pos[a] - origin[a]) / spacing[a];
                                if(!(std::isfinite(t)))
                                    throw std::runtime_error(
                                        "near shell: non-finite source coordinate while "
                                        "deriving the local subtraction box");
                                // Full TSC stencil plus one guard cell, in the
                                // signed global cell index space.
                                std::int64_t const lo
                                    = static_cast<std::int64_t>(std::floor(t - 1.5)) - 1;
                                std::int64_t const hi
                                    = static_cast<std::int64_t>(std::ceil(t + 1.5)) + 1;
                                localLo[a] = std::min(localLo[a], lo);
                                localHi[a] = std::max(localHi[a], hi);
                            }
                        }
                        Idx3 localN{};
                        for(int a = 0; a < 3; ++a)
                        {
                            std::int64_t const len64 = localHi[a] - localLo[a] + 1;
                            if(len64 <= 0)
                                throw std::runtime_error(
                                    "near shell: empty local subtraction box");
                            std::size_t len = static_cast<std::size_t>(len64);
                            if(len % 2u)
                                ++len;
                            localN[a] = len;
                        }
                        Vec3 localOrigin{};
                        for(int a = 0; a < 3; ++a)
                            localOrigin[a] = origin[a] + static_cast<double>(localLo[a]) * spacing[a];

                    ComputeOptions localOptions = tileOptions;
                    // The local subtraction is computed by this rank alone for
                    // its own tile; never split it across ranks.
                    localOptions.parallel = nullptr;
                    localOptions.gridSize = localN;
                    localOptions.gridOrigin = localOrigin;
                    localOptions.meshGridSize = std::nullopt;
                    localOptions.meshGridOrigin = std::nullopt;
                    localOptions.globalGridSize = std::nullopt;
                    localOptions.globalGridOrigin = std::nullopt;
                    localOptions.fftPadding = 2u;
                    VelocityMomentsReport localReport;
                    local = computeField(near, localOptions, nullptr, nullptr, &localReport);
                    if(localOptions.velocityMoments.mode
                           != VelocityMomentsOptions::Mode::Off
                       && !localReport.used)
                        throw std::runtime_error(
                            "near shell: the velocity-moment subtraction did not engage on a "
                            "tile; refusing to deliver a field whose near shell cancels a "
                            "different velocity model than the global mesh");
                    nLocalX = local.grid.n[0];
                    nLocalY = local.grid.n[1];
                    }

                    // Subtract F_mesh(near) and add F_exact(near) inside the tile.
                    std::size_t const nx = i1 - i0;
                    std::size_t const ny = j1 - j0;
                    std::size_t const nz = k1 - k0;

                    // Diagnostic: count the exact (source, node, component)
                    // pairs with the same predicate the evaluators use.  Only
                    // runs when the caller asks for it, never in production.
                    if(nearShell.exactPairCount)
                    {
                        std::size_t exactLocal = 0;
                        nbref::Vec3 const cellVec{cells[0], cells[1], cells[2]};
                        for(int c = 0; c < 6; ++c)
                            for(std::size_t kk = 0; kk < nz; ++kk)
                                for(std::size_t jj = 0; jj < ny; ++jj)
                                    for(std::size_t ii = 0; ii < nx; ++ii)
                                    {
                                        nbref::Vec3 const xObs{
                                            origin[0] + static_cast<double>(i0 + ii) * spacing[0]
                                                + kYeeOffset[c][0] * cells[0],
                                            origin[1] + static_cast<double>(j0 + jj) * spacing[1]
                                                + kYeeOffset[c][1] * cells[1],
                                            origin[2] + static_cast<double>(k0 + kk) * spacing[2]
                                                + kYeeOffset[c][2] * cells[2]};
                                        for(std::size_t q = 0; q < near.size(); ++q)
                                        {
                                            nbref::Vec3 const R0{
                                                xObs.x - near.x[q],
                                                xObs.y - near.y[q],
                                                xObs.z - near.z[q]};
                                            nbref::Vec3 const u{
                                                near.ux[q], near.uy[q], near.uz[q]};
                                            if(pbfast::detail::routesExact(
                                                   R0, cellVec, u, tileFastSettings))
                                                ++exactLocal;
                                        }
                                    }
                        *nearShell.exactPairCount += exactLocal;
                    }

                    if(useFastGpu)
                    {
                        if(deviceStencil)
                        {
                            // Record the tile; the whole batch runs in one
                            // device launch after the tile loop.
                            pbfast::BatchedTile bt;
                            bt.i0 = static_cast<std::uint32_t>(i0);
                            bt.j0 = static_cast<std::uint32_t>(j0);
                            bt.k0 = static_cast<std::uint32_t>(k0);
                            bt.nx = static_cast<std::uint32_t>(nx);
                            bt.ny = static_cast<std::uint32_t>(ny);
                            bt.nz = static_cast<std::uint32_t>(nz);
                            bt.srcBegin = static_cast<std::uint32_t>(batchIndices.size());
                            bt.srcCount = static_cast<std::uint32_t>(near.size());
                            bt.outOffset = batchOutSize;
                            for(std::size_t q = 0; q < near.size(); ++q)
                                batchIndices.push_back(
                                    static_cast<std::uint32_t>(nearIndex[q]));
                            batchTiles.push_back(bt);
                            batchParams.push_back(deviceParams);
                            batchOutSize += static_cast<std::uint64_t>(6u * nx * ny * nz);
                            pairs += static_cast<std::uint64_t>(
                                static_cast<std::uint64_t>(activeComponents) * nx * ny * nz
                                * near.size());
                        }
                        else
                        {
                            // Legacy path: per-tile local FFT plus device eval.
                            // Same generic alpaka functor as CUDA/HIP; on a CPU
                            // build the host device runs it with the host
                            // FrameSpec.
                            Bunch meshB = near;
                            std::vector<double> const dev = pbfast::evalFastNearDevice(
                                near,
                                meshB,
                                false,
                                nx,
                                ny,
                                nz,
                                origin[0] + static_cast<double>(i0) * spacing[0],
                                origin[1] + static_cast<double>(j0) * spacing[1],
                                origin[2] + static_cast<double>(k0) * spacing[2],
                                spacing[0],
                                spacing[1],
                                spacing[2],
                                cells[0],
                                cells[1],
                                cells[2],
                                deviceParams);
                            for(int c = 0; c < 6; ++c)
                                for(std::size_t kk = 0; kk < nz; ++kk)
                                    for(std::size_t jj = 0; jj < ny; ++jj)
                                        for(std::size_t ii = 0; ii < nx; ++ii)
                                        {
                                            std::size_t const i = i0 + ii;
                                            std::size_t const j = j0 + jj;
                                            std::size_t const k = k0 + kk;
                                            std::size_t const gidx = i + n[0] * (j + n[1] * k);
                                            std::size_t const li = static_cast<std::size_t>(
                                                static_cast<std::int64_t>(i) - localLo[0]);
                                            std::size_t const lj = static_cast<std::size_t>(
                                                static_cast<std::int64_t>(j) - localLo[1]);
                                            std::size_t const lk = static_cast<std::size_t>(
                                                static_cast<std::int64_t>(k) - localLo[2]);
                                            std::size_t const lidx
                                                = li + nLocalX * (lj + nLocalY * lk);
                                            std::size_t const didx = ii + nx * (jj + ny * kk);
                                            correction.data[c][gidx]
                                                = -local.data[c][lidx]
                                                + dev[static_cast<std::size_t>(c) * nx * ny * nz
                                                      + didx];
                                        }
                            pairs += static_cast<std::uint64_t>(6u * nx * ny * nz * near.size());
                        }
                    }
                    else
                    {
                        // The host reference loop is an alpaka host kernel now.
                        // One work item owns one (component, node) slot and
                        // keeps the source order, so the field is bit-identical
                        // to the former hand-written loop for every thread count.
                        ExactNearHostArgs exactArgs;
                        // Build the span views once per tile.  Each field and
                        // local-mesh component is a separate dense x-fastest
                        // allocation, so it gets its own 1-D view; the particle
                        // properties are dense 1-D views of length nsrc.  This
                        // scalar host path needs only element alignment.
                        auto const fieldExtents = SpanIndex{correction.data[0].size()};
                        auto const localExtents = SpanIndex{local.data[0].size()};
                        auto const particleExtents = SpanIndex{near.size()};
                        for(int c = 0; c < 6; ++c)
                        {
                            exactArgs.correction[c]
                                = alpaka::makeMdSpan(correction.data[c].data(), fieldExtents);
                            exactArgs.local[c] = Span1DConst{
                                alpaka::makeMdSpan(local.data[c].data(), localExtents)};
                        }
                        exactArgs.x = alpaka::makeMdSpan(near.x.data(), particleExtents);
                        exactArgs.y = alpaka::makeMdSpan(near.y.data(), particleExtents);
                        exactArgs.z = alpaka::makeMdSpan(near.z.data(), particleExtents);
                        exactArgs.ux = alpaka::makeMdSpan(near.ux.data(), particleExtents);
                        exactArgs.uy = alpaka::makeMdSpan(near.uy.data(), particleExtents);
                        exactArgs.uz = alpaka::makeMdSpan(near.uz.data(), particleExtents);
                        exactArgs.weight = alpaka::makeMdSpan(near.weight.data(), particleExtents);
                        exactArgs.charge = alpaka::makeMdSpan(near.charge.data(), particleExtents);
                        exactArgs.nsrc = near.size();
                        exactArgs.nx = nx;
                        exactArgs.ny = ny;
                        exactArgs.nz = nz;
                        exactArgs.i0 = i0;
                        exactArgs.j0 = j0;
                        exactArgs.k0 = k0;
                        exactArgs.n0 = n[0];
                        exactArgs.n1 = n[1];
                        exactArgs.localLo = LocalOffsetVec{localLo[0], localLo[1], localLo[2]};
                        exactArgs.origin
                            = alpaka::Vec<double, 3u>{origin[0], origin[1], origin[2]};
                        exactArgs.spacing
                            = alpaka::Vec<double, 3u>{spacing[0], spacing[1], spacing[2]};
                        exactArgs.cells = alpaka::Vec<double, 3u>{cells[0], cells[1], cells[2]};
                        exactArgs.nLocalX = nLocalX;
                        exactArgs.nLocalY = nLocalY;
                        exactArgs.activeComponents = static_cast<std::size_t>(activeComponents);
                        exactArgs.useFast = useFast;
                        exactArgs.fastSettings = tileFastSettings;
                        exactArgs.settings = settings;
                        runNearShellHostKernel(
                            static_cast<std::size_t>(activeComponents) * nx * ny * nz,
                            ExactNearHostKernel{},
                            exactArgs);
                        pairs += static_cast<std::uint64_t>(
                            static_cast<std::uint64_t>(activeComponents) * nx * ny * nz
                            * near.size());
                    }
                }
            }
        }
        if(deviceStencil && !batchTiles.empty())
        {
            std::vector<double> const dev = pbfast::evalFastNearDeviceBatched(
                nearBunch,
                batchIndices,
                batchTiles,
                batchParams,
                batchRepU,
                true,
                n[0],
                n[1],
                n[2],
                origin[0],
                origin[1],
                origin[2],
                spacing[0],
                spacing[1],
                spacing[2],
                cells[0],
                cells[1],
                cells[2]);
            for(std::size_t t = 0; t < batchTiles.size(); ++t)
            {
                pbfast::BatchedTile const& bt = batchTiles[t];
                std::size_t const nodes
                    = static_cast<std::size_t>(bt.nx) * bt.ny * bt.nz;
                for(int c = 0; c < 6; ++c)
                    for(std::uint32_t kk = 0; kk < bt.nz; ++kk)
                        for(std::uint32_t jj = 0; jj < bt.ny; ++jj)
                            for(std::uint32_t ii = 0; ii < bt.nx; ++ii)
                            {
                                std::size_t const i = bt.i0 + ii;
                                std::size_t const j = bt.j0 + jj;
                                std::size_t const k = bt.k0 + kk;
                                std::size_t const gidx = i + n[0] * (j + n[1] * k);
                                std::size_t const didx
                                    = static_cast<std::size_t>(ii)
                                    + static_cast<std::size_t>(bt.nx)
                                        * (static_cast<std::size_t>(jj)
                                           + static_cast<std::size_t>(bt.ny) * kk);
                                correction.data[c][gidx]
                                    = dev[static_cast<std::size_t>(bt.outOffset)
                                          + static_cast<std::size_t>(c) * nodes + didx];
                            }
            }
        }
        if(parSize > 1)
        {
            for(int c = 0; c < 6; ++c)
                par->allreduceSum(correction.data[c].data(), correction.data[c].size());
            par->allreduceSum(pairs);
        }
        bool const fastMode = nearShell.mode == NearShellOptions::Mode::Fast
            || nearShell.mode == NearShellOptions::Mode::FastGpu;
        detail::NearShellPeak const uncorrectedPeak
            = fastMode ? detail::nearShellPeak(field) : detail::NearShellPeak{};
        for(int c = 0; c < 6; ++c)
            for(std::size_t i = 0; i < field.data[c].size(); ++i)
                field.data[c][i] += correction.data[c][i];
        if(fastMode)
            detail::enforceNearShellPeak(
                uncorrectedPeak,
                detail::nearShellPeak(field),
                radius,
                tile,
                requestedFastOrder);
        if(parRank == 0)
            std::fprintf(
                stderr,
                "near shell wall [rank 0]: %.3f ms\n",
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - nearStart)
                    .count());
        if(nearShell.exactPairs)
            *nearShell.exactPairs = static_cast<std::size_t>(pairs);
        return field;
    }

} // namespace picBunch
