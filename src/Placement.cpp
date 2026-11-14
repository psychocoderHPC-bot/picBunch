// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#include "picBunch/Placement.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace picBunch
{
    namespace
    {
        Vec3 position(Bunch const& bunch, std::size_t p)
        {
            return {bunch.x[p], bunch.y[p], bunch.z[p]};
        }
    } // namespace

    Vec3 chargeWeightedCentroid(Bunch const& bunch)
    {
        if(bunch.size() == 0u)
            throw std::runtime_error("chargeWeightedCentroid: empty bunch");
        double weightSum = 0.0;
        Vec3 sum{0.0, 0.0, 0.0};
        for(std::size_t p = 0; p < bunch.size(); ++p)
        {
            double const w = bunch.weight[p] * bunch.charge[p];
            auto const r = position(bunch, p);
            weightSum += w;
            for(int a = 0; a < 3; ++a)
                sum[a] += w * r[a];
        }
        if(std::abs(weightSum) <= std::numeric_limits<double>::min())
            throw std::runtime_error(
                "chargeWeightedCentroid: the total charge-weighted weight is zero; "
                "the charge-weighted centroid is undefined");
        Vec3 result{};
        for(int a = 0; a < 3; ++a)
            result[a] = sum[a] / weightSum;
        return result;
    }

    Vec3 boundingBoxCentre(Bunch const& bunch)
    {
        auto const bounds = bunch.bounds();
        Vec3 result{};
        for(int a = 0; a < 3; ++a)
            result[a] = 0.5 * (bounds[0][a] + bounds[1][a]);
        return result;
    }

    PlacedBunch placeBunch(Bunch const& bunch, Config const& config)
    {
        PlacedBunch result;
        bool const keepAbsolute
            = config.placement.centre == Config::Placement::Centre::None;
        // ``centre = "none"`` keeps the absolute input coordinates: there is no
        // centroid shift, so the requested target is the input centroid and the
        // shift is exactly zero.  An empty bunch has no centroid at all; it is
        // only reachable through the checkpoint-only ``empty_species`` path,
        // which never calls this function.
        if(bunch.size() > 0u)
        {
            result.centroid = chargeWeightedCentroid(bunch);
            result.boundingCentre = boundingBoxCentre(bunch);
        }

        for(int a = 0; a < 3; ++a)
        {
            if(keepAbsolute)
            {
                result.targetCentre[a] = result.centroid[a];
                result.shift[a] = 0.0;
                continue;
            }
            double const extent = static_cast<double>(config.domain.cells[a]) * config.domain.cellSize[a];
            double const requested = config.placement.fraction
                ? config.placement.offset[a] * extent
                : config.placement.offset[a];
            result.targetCentre[a] = config.domain.origin[a] + requested;
            result.shift[a] = result.targetCentre[a] - result.centroid[a];
        }

        result.bunch = bunch;
        for(std::size_t p = 0; p < result.bunch.size(); ++p)
        {
            result.bunch.x[p] += result.shift[0];
            result.bunch.y[p] += result.shift[1];
            result.bunch.z[p] += result.shift[2];
        }

        auto const bounds = result.bunch.bounds();
        char const* const axisNames[3] = {"x", "y", "z"};
        for(int a = 0; a < 3; ++a)
        {
            double const lower = config.domain.origin[a];
            double const upper = lower + static_cast<double>(config.domain.cells[a]) * config.domain.cellSize[a];
            if(bounds[0][a] < lower || bounds[1][a] >= upper)
            {
                // Never clip: report the largest distance outside the
                // [lower, upper) domain and abort before writing any patch.
                double const outside = std::max({lower - bounds[0][a], bounds[1][a] - upper, 0.0});
                std::ostringstream message;
                message << "placement leaves the PIConGPU domain on axis " << axisNames[a]
                        << ": shifted bunch spans [" << bounds[0][a] << ", " << bounds[1][a]
                        << "] m, domain is [" << lower << ", " << upper << ") m; "
                        << outside << " m (" << outside / config.domain.cellSize[a]
                        << " cells) lie outside the domain; reduce the placement offset";
                throw std::runtime_error(message.str());
            }
            if(config.placement.patch == Config::Placement::Patch::Domain)
            {
                // One domain-covering patch.  positionOffset still indexes the
                // global cells, so the particle cell indices are unchanged.
                result.patchOffset[a] = 0;
                result.patchExtent[a] = static_cast<std::uint64_t>(config.domain.cells[a]);
                continue;
            }
            // Snap a value that is within one part in 1e9 of a cell boundary
            // to that boundary.  Adding a shift and dividing again can leave
            // 3.0 as 2.9999999999999996, which would otherwise put a particle
            // in the wrong cell and split it from the patch.
            double const lowCell = std::floor(
                (bounds[0][a] - lower) / config.domain.cellSize[a] + 1e-9);
            double const highCell = std::floor(
                (bounds[1][a] - lower) / config.domain.cellSize[a] + 1e-9);
            auto const offset = static_cast<std::int64_t>(lowCell);
            auto const extent = static_cast<std::uint64_t>(highCell - lowCell) + 1u;
            if(offset < 0 || extent == 0u
               || static_cast<std::uint64_t>(offset) + extent
                   > static_cast<std::uint64_t>(config.domain.cells[a]))
                throw std::runtime_error(
                    "placement produces an invalid particle patch on axis "
                    + std::string(axisNames[a]));
            result.patchOffset[a] = offset;
            result.patchExtent[a] = extent;
        }

        // One particle patch per requested region.  The base region (bbox or
        // full domain) is partitioned along z into adjacent, non-overlapping
        // slabs that cover it exactly once, matching a PIConGPU checkpoint's
        // per-rank patch table.  Particle ownership is disjoint because the
        // writer assigns each particle to the one patch that contains its cell
        // and writes cumulative numParticlesOffset.
        std::size_t const requested = std::max<std::size_t>(1u, config.placement.patchCount);
        std::uint64_t const zOffset
            = static_cast<std::uint64_t>(std::max<std::int64_t>(0, result.patchOffset[2]));
        std::uint64_t const zExtent = result.patchExtent[2];
        std::size_t count = std::min<std::size_t>(requested, static_cast<std::size_t>(zExtent));
        if(count == 0u)
            count = 1u;
        for(std::size_t p = 0; p < count; ++p)
        {
            ParticlePatch patch;
            for(int a = 0; a < 3; ++a)
            {
                patch.offset[a] = result.patchOffset[a];
                patch.extent[a] = result.patchExtent[a];
            }
            // Integer slab boundaries, last slab absorbs the remainder.  The
            // half-open ranges [lo, hi) tile [zOffset, zOffset + zExtent)
            // exactly once: no gap and no double-covered cell.
            std::uint64_t const lo = zOffset + (zExtent * p) / count;
            std::uint64_t const hi = zOffset + (zExtent * (p + 1u)) / count;
            patch.offset[2] = static_cast<std::int64_t>(lo);
            patch.extent[2] = hi - lo;
            result.patches.push_back(patch);
        }
        return result;
    }
} // namespace picBunch
