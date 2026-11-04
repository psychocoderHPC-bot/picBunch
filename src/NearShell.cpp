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
#include "picBunch/detail/FastNear.hpp"
#include "picBunch/detail/NearReference.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <array>
#include <cmath>
#include <cstddef>
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
        VelocityBinning* outBinning)
    {
        if(bunch.size() == 0)
            throw std::runtime_error("computeFieldNearExact: empty bunch");

        // The exact near term evaluates every (source, node) pair with the
        // source's own velocity: bin-free.  The mesh subtraction cancels the
        // global mesh bin for bin only if both use the same box geometry and
        // representatives, so the binning is derived once from the full bunch
        // and fixed for every per-tile local mesh.
        VelocityBinning binning;
        if(!options.fixedBinning)
            binning = deriveVelocityBinning(bunch, options.tolerance, options.velocityBinning);
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
        Field field = computeField(bunch, plain, stageTimes);

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
        pbfast::Settings const fastSettings{nearShell.fastOrder, nearShell.fastObliqueOrder, 5};
        bool const useFast = nearShell.mode == NearShellOptions::Mode::Fast;

        auto const n = grid.n;
        auto const& origin = grid.origin;
        auto const& spacing = grid.spacing;

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
                    for(std::size_t p = 0; p < bunch.size(); ++p)
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
                    if(nearIndex.empty())
                        continue;

                    Bunch const near = selectSources(bunch, nearIndex);

                    // Local node box: the tile plus the full TSC support of
                    // every near source.  It must contain every source used in
                    // the subtraction and every observer in the tile.
                    long localLo[3] = {
                        static_cast<long>(i0),
                        static_cast<long>(j0),
                        static_cast<long>(k0)};
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

                    ComputeOptions localOptions = plain;
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
                    Field const local = computeField(near, localOptions);

                    // Subtract F_mesh(near) and add F_exact(near) inside the tile.
                    std::size_t const nx = i1 - i0;
                    std::size_t const ny = j1 - j0;
                    std::size_t const nz = k1 - k0;
                    std::size_t const nLocalX = local.grid.n[0];
                    std::size_t const nLocalY = local.grid.n[1];
                    std::size_t const nLocalZ = local.grid.n[2];
                    (void)nLocalZ;
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
                                                fastSettings,
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
