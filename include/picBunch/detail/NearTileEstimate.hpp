// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/Types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace picBunch::detail
{
    //! Number of sources used by the automatic tile estimate.
    inline constexpr std::size_t kNearTileSampleLimit = 4096u;
    //! Measured tile 4 / tile 2 pair throughput on the V100 dense campaign.
    inline constexpr double kAutoTileThroughputRatio = 1.25;
    //! Extra margin so the automatic rule only switches when tile 2 wins
    //! clearly; a 5% loss against the best fixed tile is not acceptable.
    inline constexpr double kAutoTileSafetyMargin = 1.05;

    /*! Cheap estimate of the near-shell work for one tile edge.
     *
     * Mirrors the per-source membership test that builds the near set in
     * ``NearShell.cpp``: a source belongs to every tile whose clipped node box
     * is within the shell radius.  Sampling up to ``kNearTileSampleLimit``
     * sources and scaling by the inverse sample fraction predicts the
     * (source, node) pair count without allocating any tile list.  The factor
     * six matches the device evaluator's six field components; the host kernel
     * uses one, so the ratio between two tile edges is the same either way.
     *
     * Source positions are absolute, as in the near-set builder, so the tile
     * index must be measured from ``grid.origin``.  Without that translation a
     * non-zero origin shifts the sampled source out of its tile range and the
     * estimate collapses to zero.
     */
    inline double estimateNearTilePairs(
        Bunch const& bunch,
        Grid const& grid,
        Vec3 const& cellSize,
        double radius,
        std::size_t tile)
    {
        std::size_t const total = bunch.size();
        std::size_t const samples = std::min(total, kNearTileSampleLimit);
        if(samples == 0u || tile == 0u)
            return 0.0;
        std::size_t const tileCount[3] = {
            (grid.n[0] + tile - 1u) / tile,
            (grid.n[1] + tile - 1u) / tile,
            (grid.n[2] + tile - 1u) / tile};
        double sampledVolume = 0.0;
        for(std::size_t s = 0; s < samples; ++s)
        {
            // Evenly strided sample: deterministic and independent of any
            // container ordering.
            std::size_t const p = (s * total) / samples;
            double const pos[3] = {bunch.x[p], bunch.y[p], bunch.z[p]};
            long qLo[3];
            long qHi[3];
            for(int a = 0; a < 3; ++a)
            {
                double const reach = radius * cellSize[a];
                double const sp = grid.spacing[a];
                double const tj = sp * static_cast<double>(tile);
                // Tile indices are grid coordinates: subtract the grid origin
                // exactly as the near-set builder does for its cell list.
                double const u = pos[a] - grid.origin[a];
                long const lo = static_cast<long>(std::floor(
                    (u - reach - static_cast<double>(tile - 1u) * sp) / tj));
                long const hi = static_cast<long>(std::floor((u + reach) / tj));
                // One guard tile each side, exactly as in the near-set build.
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
                        std::size_t const i1 = std::min(grid.n[0], i0 + tile);
                        std::size_t const j1 = std::min(grid.n[1], j0 + tile);
                        std::size_t const k1 = std::min(grid.n[2], k0 + tile);
                        double const low[3] = {
                            grid.origin[0] + static_cast<double>(i0) * grid.spacing[0],
                            grid.origin[1] + static_cast<double>(j0) * grid.spacing[1],
                            grid.origin[2] + static_cast<double>(k0) * grid.spacing[2]};
                        double const high[3] = {
                            grid.origin[0] + static_cast<double>(i1 - 1u) * grid.spacing[0],
                            grid.origin[1] + static_cast<double>(j1 - 1u) * grid.spacing[1],
                            grid.origin[2] + static_cast<double>(k1 - 1u) * grid.spacing[2]};
                        double dist2 = 0.0;
                        for(int a = 0; a < 3; ++a)
                        {
                            double const pad = 0.5 * cellSize[a];
                            double const d = std::max(
                                {low[a] - pad - pos[a], 0.0, pos[a] - (high[a] + pad)});
                            double const dn = d / cellSize[a];
                            dist2 += dn * dn;
                        }
                        if(dist2 <= radius * radius)
                            sampledVolume += static_cast<double>(
                                (i1 - i0) * (j1 - j0) * (k1 - k0));
                    }
        }
        double const scale = static_cast<double>(total) / static_cast<double>(samples);
        return 6.0 * scale * sampledVolume;
    }

    //! Result of the automatic tile rule for one bunch and grid.
    struct NearTileEstimate
    {
        double pairs2 = 0.0;
        double pairs4 = 0.0;
        bool tile2Wins = false;
        std::size_t tile = 4u;
    };

    //! Compare the tile-2 and tile-4 pair estimates with the measured margin.
    inline NearTileEstimate estimateNearTile(
        Bunch const& bunch,
        Grid const& grid,
        Vec3 const& cellSize,
        double radius)
    {
        NearTileEstimate out;
        out.pairs2 = estimateNearTilePairs(bunch, grid, cellSize, radius, 2u);
        out.pairs4 = estimateNearTilePairs(bunch, grid, cellSize, radius, 4u);
        out.tile2Wins = out.pairs4 > kAutoTileThroughputRatio * kAutoTileSafetyMargin * out.pairs2;
        out.tile = out.tile2Wins ? 2u : 4u;
        return out;
    }
} // namespace picBunch::detail
