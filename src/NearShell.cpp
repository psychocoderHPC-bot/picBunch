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
#if defined(PICBUNCH_BACKEND_CUDA)
#    include "picBunch/detail/FastNearDevice.hpp"
#endif
#include "picBunch/detail/NearReference.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
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

        Bunch selectSources(Bunch const& bunch, std::vector<std::size_t> const& index)
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
        std::size_t const tile = std::max<std::size_t>(1u, nearShell.tileNodes);
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
        pbfast::Settings fastSettings{nearShell.fastOrder, explicitObliqueOrder, 5};
        fastSettings.tolerance = options.tolerance;
        bool useFast = nearShell.mode == NearShellOptions::Mode::Fast;
        bool useFastGpu = false;
#if defined(PICBUNCH_BACKEND_CUDA)
        useFastGpu = nearShell.mode == NearShellOptions::Mode::FastGpu;
        // The fast evaluator is the same algebra on host and device.  When a
        // CUDA device is compiled in and available, `fast` uses it; `fast-gpu`
        // remains an explicit request and falls back to the host otherwise.
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
                "near shell: fast-gpu requested but no CUDA device is available; "
                "falling back to the host fast evaluator\n");
            useFastGpu = false;
            useFast = true;
        }
#else
        if(nearShell.mode == NearShellOptions::Mode::FastGpu)
        {
            std::fprintf(
                stderr,
                "near shell: fast-gpu is not compiled in; "
                "falling back to the host fast evaluator\n");
            useFast = true;
        }
#endif
#if defined(PICBUNCH_BACKEND_CUDA)
        pbfast::DeviceParams deviceParams;
        deviceParams.order = fastSettings.order;
        deviceParams.obliqueOrder = explicitObliqueOrder;
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
#endif

        auto const n = grid.n;
        auto const& origin = grid.origin;
        auto const& spacing = grid.spacing;

        // Uniform cell list over the sources.  One cell is `radius` PIC cells
        // wide per axis, so a tile only visits the (up to) 3x3x3 neighbouring
        // buckets instead of scanning all sources for every tile.  Sources are
        // pushed in index order and the per-tile list is sorted afterwards, so
        // the near set is identical to the former O(tiles x N) scan.
        double cl[3];
        double smin[3] = {bunch.x[0], bunch.y[0], bunch.z[0]};
        double smax[3] = {bunch.x[0], bunch.y[0], bunch.z[0]};
        for(int a = 0; a < 3; ++a)
            cl[a] = std::max(radius * cells[a], 1e-300);
        for(std::size_t p = 0; p < bunch.size(); ++p)
        {
            double const pos[3] = {bunch.x[p], bunch.y[p], bunch.z[p]};
            for(int a = 0; a < 3; ++a)
            {
                smin[a] = std::min(smin[a], pos[a]);
                smax[a] = std::max(smax[a], pos[a]);
            }
        }
        long cn[3];
        for(int a = 0; a < 3; ++a)
            cn[a] = std::max<long>(1, static_cast<long>(std::ceil((smax[a] - smin[a]) / cl[a])) + 1);
        std::vector<std::vector<std::size_t>> buckets(
            static_cast<std::size_t>(cn[0] * cn[1] * cn[2]));
        for(std::size_t p = 0; p < bunch.size(); ++p)
        {
            double const pos[3] = {bunch.x[p], bunch.y[p], bunch.z[p]};
            long ci[3];
            for(int a = 0; a < 3; ++a)
            {
                ci[a] = static_cast<long>(std::floor((pos[a] - smin[a]) / cl[a]));
                ci[a] = std::clamp(ci[a], 0L, cn[a] - 1L);
            }
            buckets[static_cast<std::size_t>(ci[0] + cn[0] * (ci[1] + cn[1] * ci[2]))].push_back(p);
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

                    // Near set: every source within `radius` PIC cells of the
                    // tile box, measured per axis in that axis's own cell unit.
                    // All of them are then evaluated exactly, so the tile's
                    // effective radius is `radius` plus at most half a tile.
                    std::vector<std::size_t> nearIndex;
                    long cLo[3];
                    long cHi[3];
                    for(int a = 0; a < 3; ++a)
                    {
                        double const reach = radius * cells[a];
                        cLo[a] = static_cast<long>(std::floor((low[a] - reach - smin[a]) / cl[a]));
                        cHi[a] = static_cast<long>(std::floor((high[a] + reach - smin[a]) / cl[a]));
                        cLo[a] = std::clamp(cLo[a], 0L, cn[a] - 1L);
                        cHi[a] = std::clamp(cHi[a], 0L, cn[a] - 1L);
                    }
                    for(long ck = cLo[2]; ck <= cHi[2]; ++ck)
                        for(long cj = cLo[1]; cj <= cHi[1]; ++cj)
                            for(long ci = cLo[0]; ci <= cHi[0]; ++ci)
                            {
                                auto const& bucket = buckets[static_cast<std::size_t>(
                                    ci + cn[0] * (cj + cn[1] * ck))];
                                for(std::size_t const p : bucket)
                                {
                                    double const pos[3] = {bunch.x[p], bunch.y[p], bunch.z[p]};
                                    bool inside = true;
                                    for(int a = 0; a < 3; ++a)
                                    {
                                        double const reach = radius * cells[a];
                                        if(pos[a] < low[a] - reach || pos[a] > high[a] + reach)
                                        {
                                            inside = false;
                                            break;
                                        }
                                    }
                                    if(inside)
                                        nearIndex.push_back(p);
                                }
                            }
                    std::sort(nearIndex.begin(), nearIndex.end());
                    if(nearIndex.empty())
                        continue;

                    Bunch const near = selectSources(bunch, nearIndex);

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
                    for(std::size_t q = 0; q < near.size(); ++q)
                    {
                        double const uq2 = near.ux[q] * near.ux[q]
                            + near.uy[q] * near.uy[q] + near.uz[q] * near.uz[q];
                        if(uq2 > 0.0)
                            tileMaxGamma
                                = std::max(tileMaxGamma, std::sqrt(1.0 + uq2));
                    }
                    int const tileRemainderOrder = std::min(
                        pbfast::kQuadratureOrderCap,
                        std::max(
                            nearShell.fastOrder,
                            pbfast::quadratureOrderFor(
                                tileMaxGamma, options.tolerance, nearShell.fastOrder)));

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
#if defined(PICBUNCH_BACKEND_CUDA)
                    // Only the CUDA device evaluator consumes DeviceParams; the
                    // host fast path reads tileFastSettings below.  Keep the
                    // assignment behind the same guard as the declaration so a
                    // CPU-only build compiles.
                    deviceParams.obliqueOrder = tileObliqueOrder;
                    deviceParams.order = tileRemainderOrder;
                    {
                        auto const& tx = pbfast::detail::rule(tileRemainderOrder).x;
                        auto const& tw = pbfast::detail::rule(tileRemainderOrder).w;
                        for(int i = 0; i < tileRemainderOrder; ++i)
                        {
                            deviceParams.xi[i] = tx[static_cast<std::size_t>(i)];
                            deviceParams.wi[i] = tw[static_cast<std::size_t>(i)];
                        }
                    }
#endif
                    pbfast::Settings tileFastSettings = fastSettings;
                    tileFastSettings.obliqueOrder = tileObliqueOrder;
                    tileFastSettings.order = tileRemainderOrder;

                    // The local FFT subtraction is only needed on the legacy
                    // path.  The direct 27-point TSC stencil performs the same
                    // discrete convolution per (source, node) pair on the
                    // device, so the per-tile FFT (and its plan setup) is
                    // skipped by default.
                    bool const deviceStencil = useFastGpu
                        && (std::getenv("PICBUNCH_NEAR_SHELL_LOCAL_FFT") == nullptr);
                    Field local;
                    long localLo[3] = {0, 0, 0};
                    std::size_t nLocalX = 0;
                    std::size_t nLocalY = 0;
                    if(!deviceStencil)
                    {
                        // Local node box: the tile plus the full TSC support of
                        // every near source.  It must contain every source used
                        // in the subtraction and every observer in the tile.
                        long localHi[3] = {
                            static_cast<long>(i1 - 1u),
                            static_cast<long>(j1 - 1u),
                            static_cast<long>(k1 - 1u)};
                        for(std::size_t q = 0; q < near.size(); ++q)
                        {
                            double const pos[3] = {near.x[q], near.y[q], near.z[q]};
                            for(int a = 0; a < 3; ++a)
                            {
                                double const t = (pos[a] - origin[a]) / spacing[a];
                                long const lo = static_cast<long>(std::floor(t - 1.5)) - 1;
                                long const hi = static_cast<long>(std::ceil(t + 1.5)) + 1;
                                localLo[a] = std::min(localLo[a], lo);
                                localHi[a] = std::max(localHi[a], hi);
                            }
                        }
                        std::array<std::size_t, 3> localN{};
                        for(int a = 0; a < 3; ++a)
                        {
                            std::size_t len = static_cast<std::size_t>(localHi[a] - localLo[a] + 1);
                            if(len % 2u)
                                ++len;
                            localN[a] = len;
                        }
                        std::array<double, 3> localOrigin{};
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
                    if(useFastGpu)
                    {
#if defined(PICBUNCH_BACKEND_CUDA)
                        Bunch meshB = near;
                        if(deviceStencil)
                        {
                            VelocityBinning const& meshBin
                                = options.fixedBinning ? *options.fixedBinning : binning;
                            for(std::size_t q = 0; q < meshB.size(); ++q)
                            {
                                double const coords[3] = {
                                    meshB.ux[q] * meshBin.basis[0][0]
                                        + meshB.uy[q] * meshBin.basis[0][1]
                                        + meshB.uz[q] * meshBin.basis[0][2],
                                    meshB.ux[q] * meshBin.basis[1][0]
                                        + meshB.uy[q] * meshBin.basis[1][1]
                                        + meshB.uz[q] * meshBin.basis[1][2],
                                    meshB.ux[q] * meshBin.basis[2][0]
                                        + meshB.uy[q] * meshBin.basis[2][1]
                                        + meshB.uz[q] * meshBin.basis[2][2]};
                                std::size_t const b = meshBin.boxIndex(coords[0], coords[1], coords[2]);
                                meshB.ux[q] = meshBin.representativeU[b][0];
                                meshB.uy[q] = meshBin.representativeU[b][1];
                                meshB.uz[q] = meshBin.representativeU[b][2];
                            }
                        }
                        std::vector<double> const dev = pbfast::evalFastNearDevice(
                            near,
                            meshB,
                            deviceStencil,
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
                        {
#pragma omp parallel for collapse(3)
                            for(std::size_t kk = 0; kk < nz; ++kk)
                                for(std::size_t jj = 0; jj < ny; ++jj)
                                    for(std::size_t ii = 0; ii < nx; ++ii)
                                    {
                                        std::size_t const i = i0 + ii;
                                        std::size_t const j = j0 + jj;
                                        std::size_t const k = k0 + kk;
                                        std::size_t const gidx = i + n[0] * (j + n[1] * k);
                                        std::size_t const li = i - static_cast<std::size_t>(localLo[0]);
                                        std::size_t const lj = j - static_cast<std::size_t>(localLo[1]);
                                        std::size_t const lk = k - static_cast<std::size_t>(localLo[2]);
                                        std::size_t const lidx = li + nLocalX * (lj + nLocalY * lk);
                                        std::size_t const didx = ii + nx * (jj + ny * kk);
                                        correction.data[c][gidx]
                                            = deviceStencil
                                            ? dev[static_cast<std::size_t>(c) * nx * ny * nz + didx]
                                            : -local.data[c][lidx]
                                                + dev[static_cast<std::size_t>(c) * nx * ny * nz + didx];
                                    }
                        }
#endif
                        pairs += static_cast<std::uint64_t>(6u * nx * ny * nz * near.size());
                    }
                    else
                    {
                        for(int c = 0; c < 6; ++c)
                        {
                            double const unit = (c < 3) ? kCoulombConstant
                                                        : kCoulombConstant / kSpeedOfLight;
#pragma omp parallel for collapse(3)
                            for(std::size_t kk = 0; kk < nz; ++kk)
                                for(std::size_t jj = 0; jj < ny; ++jj)
                                    for(std::size_t ii = 0; ii < nx; ++ii)
                                    {
                                        std::size_t const i = i0 + ii;
                                        std::size_t const j = j0 + jj;
                                        std::size_t const k = k0 + kk;
                                        std::size_t const gidx = i
                                            + n[0] * (j + n[1] * k);
                                        std::size_t const li = i - static_cast<std::size_t>(localLo[0]);
                                        std::size_t const lj = j - static_cast<std::size_t>(localLo[1]);
                                        std::size_t const lk = k - static_cast<std::size_t>(localLo[2]);
                                        std::size_t const lidx = li + nLocalX * (lj + nLocalY * lk);
                                        double value = -local.data[c][lidx];

                                        double const x_obs = origin[0] + static_cast<double>(i) * spacing[0]
                                            + kYeeOffset[c][0] * cells[0];
                                        double const y_obs = origin[1] + static_cast<double>(j) * spacing[1]
                                            + kYeeOffset[c][1] * cells[1];
                                        double const z_obs = origin[2] + static_cast<double>(k) * spacing[2]
                                            + kYeeOffset[c][2] * cells[2];
                                        for(std::size_t q = 0; q < near.size(); ++q)
                                        {
                                            nbref::Vec3 E, B;
                                            if(useFast)
                                            {
                                                pbfast::shape_field(
                                                    {x_obs, y_obs, z_obs},
                                                    {near.x[q], near.y[q], near.z[q]},
                                                    {near.ux[q], near.uy[q], near.uz[q]},
                                                    cells[0],
                                                    cells[1],
                                                    cells[2],
                                                    tileFastSettings,
                                                    E,
                                                    B);
                                            }
                                            else
                                            {
                                                nbref::shape_field(
                                                    {x_obs, y_obs, z_obs},
                                                    {near.x[q], near.y[q], near.z[q]},
                                                    {near.ux[q], near.uy[q], near.uz[q]},
                                                    cells[0],
                                                    cells[1],
                                                    cells[2],
                                                    settings,
                                                    E,
                                                    B);
                                            }
                                            double const component = (c < 3) ? E[c] : B[c - 3];
                                            value += unit * (near.weight[q] * near.charge[q]) * component;
                                        }
                                        correction.data[c][gidx] = value;
                                    }
                            pairs += static_cast<std::uint64_t>(nx * ny * nz * near.size());
                        }
                    }
                }
            }
        }
        if(parSize > 1)
        {
            for(int c = 0; c < 6; ++c)
                par->allreduceSum(correction.data[c].data(), correction.data[c].size());
            par->allreduceSum(pairs);
        }
        for(int c = 0; c < 6; ++c)
            for(std::size_t i = 0; i < field.data[c].size(); ++i)
                field.data[c][i] += correction.data[c][i];
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
