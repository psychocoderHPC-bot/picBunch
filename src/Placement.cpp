// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#include "picBunch/Placement.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace picBunch
{
    namespace
    {
        std::array<double, 3> position(Bunch const& bunch, std::size_t p)
        {
            return {bunch.x[p], bunch.y[p], bunch.z[p]};
        }
    } // namespace

    std::array<double, 3> chargeWeightedCentroid(Bunch const& bunch)
    {
        if(bunch.size() == 0u)
            throw std::runtime_error("chargeWeightedCentroid: empty bunch");
        double weightSum = 0.0;
        std::array<double, 3> sum{0.0, 0.0, 0.0};
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
        std::array<double, 3> result{};
        for(int a = 0; a < 3; ++a)
            result[a] = sum[a] / weightSum;
        return result;
    }

    std::array<double, 3> boundingBoxCentre(Bunch const& bunch)
    {
        auto const bounds = bunch.bounds();
        std::array<double, 3> result{};
        for(int a = 0; a < 3; ++a)
            result[a] = 0.5 * (bounds[0][a] + bounds[1][a]);
        return result;
    }

    PlacedBunch placeBunch(Bunch const& bunch, Config const& config)
    {
        PlacedBunch result;
        result.centroid = chargeWeightedCentroid(bunch);
        result.boundingCentre = boundingBoxCentre(bunch);

        for(int a = 0; a < 3; ++a)
        {
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
        for(int a = 0; a < 3; ++a)
        {
            double const lower = config.domain.origin[a];
            double const upper = lower + static_cast<double>(config.domain.cells[a]) * config.domain.cellSize[a];
            // Snap a value that is within one part in 1e9 of a cell boundary
            // to that boundary.  Adding a shift and dividing again can leave
            // 3.0 as 2.9999999999999996, which would otherwise put a particle
            // in the wrong cell and split it from the patch.
            double const lowCell = std::floor(
                (bounds[0][a] - lower) / config.domain.cellSize[a] + 1e-9);
            double const highCell = std::floor(
                (bounds[1][a] - lower) / config.domain.cellSize[a] + 1e-9);
            if(bounds[0][a] < lower || bounds[1][a] >= upper)
                throw std::runtime_error(
                    "placement leaves the PIConGPU domain on axis "
                    + std::to_string(a) + ": shifted bunch spans ["
                    + std::to_string(bounds[0][a]) + ", " + std::to_string(bounds[1][a])
                    + "] m, domain is [" + std::to_string(lower) + ", "
                    + std::to_string(upper) + ") m; reduce the placement offset");
            auto const offset = static_cast<std::int64_t>(lowCell);
            auto const extent = static_cast<std::uint64_t>(highCell - lowCell) + 1u;
            if(offset < 0 || extent == 0u
               || static_cast<std::uint64_t>(offset) + extent
                   > static_cast<std::uint64_t>(config.domain.cells[a]))
                throw std::runtime_error(
                    "placement produces an invalid particle patch on axis "
                    + std::to_string(a));
            result.patchOffset[a] = offset;
            result.patchExtent[a] = extent;
        }
        return result;
    }
} // namespace picBunch
