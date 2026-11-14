// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group GridIndependence: the computed field must be grid-independent when the
// bunch is well inside the compute box.  Two stationary electrons are placed
// at the centre of a cubic grid; the max|E| at the grid centre must agree
// between a small and a large grid to within roundoff, and the field at the
// grid corner must remain much smaller than the centre field (the field must
// decay away from the bunch, not grow).
//
// The test is meaningful under ``mpirun -n 1`` and ``-n 4``.  A
// normalisation defect that scales with grid size or rank count shows up as
// a mismatch between the small and large grid results.

#include "picBunch/Compute.hpp"
#include "picBunch/Mpi.hpp"
#include "picBunch/Parallel.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace
{
    //! Two stationary electrons at the grid centre.
    picBunch::Bunch makeBunch(double cell)
    {
        picBunch::Bunch bunch;
        bunch.resize(2);
        // Place the electrons one cell above and below the centre.
        bunch.x = {0.0, 0.0};
        bunch.y = {0.0, 0.0};
        bunch.z = {-0.5 * cell, 0.5 * cell};
        bunch.ux = {0.0, 0.0};
        bunch.uy = {0.0, 0.0};
        bunch.uz = {0.0, 0.0};
        bunch.weight = {1.0, 1.0};
        bunch.charge = {-picBunch::kElementaryCharge, -picBunch::kElementaryCharge};
        bunch.restMass = {9.1093837139e-31, 9.1093837139e-31};
        return bunch;
    }

    //! Build a centred cubic grid of the given size.
    picBunch::ComputeOptions makeOptions(
        std::size_t n, double cell, picBunch::FftMode fftMode, double fftPadding)
    {
        picBunch::ComputeOptions options;
        options.cellSize = {cell, cell, cell};
        options.gridSize = {n, n, n};
        options.gridOrigin = {-0.5 * static_cast<double>(n) * cell,
                              -0.5 * static_cast<double>(n) * cell,
                              -0.5 * static_cast<double>(n) * cell};
        options.tolerance = 1.0e-3;
        options.fftMode = fftMode;
        options.fftPadding = static_cast<unsigned>(fftPadding);
        return options;
    }

    //! Compute the field, gather the full replicated form, and return
    //! (centreMax, cornerMax, globalMax) over the three E components.
    struct FieldStats
    {
        double centreMax = 0.0;
        double cornerMax = 0.0;
        double globalMax = 0.0;
    };

    FieldStats computeStats(
        picBunch::Bunch const& bunch,
        picBunch::ComputeOptions const& options,
        picBunch::ParallelContext const* par)
    {
        picBunch::ComputeOptions local = options;
        local.parallel = par;
        picBunch::Field const field = picBunch::computeField(bunch, local);
        std::size_t const nx = field.grid.n[0];
        std::size_t const ny = field.grid.n[1];
        std::size_t const nz = field.grid.n[2];
        std::size_t const plane = nx * ny;

        // Gather the full field on every rank.
        std::vector<std::vector<double>> full(6);
        for(int c = 0; c < 6; ++c)
            full[c].assign(nz * plane, 0.0);
        for(int c = 0; c < 6; ++c)
        {
            if(!field.zSlab)
            {
                std::copy(field.data[c].begin(), field.data[c].end(), full[c].begin());
            }
            else
            {
                std::vector<int> counts(par->size), displs(par->size);
                std::size_t const base = nz / static_cast<std::size_t>(par->size);
                std::size_t const rem = nz % static_cast<std::size_t>(par->size);
                for(int r = 0; r < par->size; ++r)
                {
                    std::size_t const rb
                        = static_cast<std::size_t>(r) * base
                          + std::min(static_cast<std::size_t>(r), rem);
                    std::size_t const rc = base + (static_cast<std::size_t>(r) < rem ? 1u : 0u);
                    counts[r] = static_cast<int>(rc * plane);
                    displs[r] = static_cast<int>(rb * plane);
                }
                par->allgatherv(
                    field.data[c].data(), counts[par->rank], full[c].data(),
                    counts.data(), displs.data());
            }
        }

        auto const idx3 = [&](std::size_t i, std::size_t j, std::size_t k) -> std::size_t {
            return (k * ny + j) * nx + i;
        };

        // max|E| at the centre node (nx/2, ny/2, nz/2).
        std::size_t const ci = nx / 2, cj = ny / 2, ck = nz / 2;
        double centreMax = 0.0;
        for(int c = 0; c < 3; ++c)
            centreMax = std::max(centreMax, std::abs(full[c][idx3(ci, cj, ck)]));

        // max|E| at the corner node (0, 0, 0).
        double cornerMax = 0.0;
        for(int c = 0; c < 3; ++c)
            cornerMax = std::max(cornerMax, std::abs(full[c][idx3(0, 0, 0)]));

        // Global max|E| over all nodes.
        double globalMax = 0.0;
        for(int c = 0; c < 3; ++c)
            for(std::size_t l = 0; l < full[c].size(); ++l)
                globalMax = std::max(globalMax, std::abs(full[c][l]));

        return {centreMax, cornerMax, globalMax};
    }
} // namespace

TEST(GridIndependence, CentreFieldIsGridIndependent)
{
    double const cell = 1.0e-6;
    picBunch::Bunch const bunch = makeBunch(cell);
    picBunch::MpiParallelContext parallel;
    picBunch::ParallelContext const* par = &parallel;

    // Small and large grids, both centred on the bunch.
    auto const small = computeStats(bunch, makeOptions(64, cell, picBunch::FftMode::Distributed, 1.0), par);
    auto const large = computeStats(bunch, makeOptions(128, cell, picBunch::FftMode::Distributed, 1.0), par);

    // The centre field must agree to relative roundoff.
    double const scale = std::max(1.0, small.centreMax);
    EXPECT_LT(std::abs(small.centreMax - large.centreMax), 1.0e-10 * scale)
        << "centre field differs between 64^3 and 128^3 grids: "
        << small.centreMax << " vs " << large.centreMax
        << " (rank " << parallel.rank << " of " << parallel.size << ")";

    // The field must decay away from the bunch: the corner must be much
    // smaller than the centre.
    EXPECT_LT(large.cornerMax, 0.01 * large.centreMax)
        << "corner field " << large.cornerMax << " is not much smaller than "
           "centre field " << large.centreMax
        << " (rank " << parallel.rank << " of " << parallel.size << ")";
}
