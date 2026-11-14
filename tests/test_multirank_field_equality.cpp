// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group MultiRankField: the distributed z-slab transform must deliver the same
// field as the replicated one, on every rank, for a rank count that does not
// divide the grid.  A normalisation defect that scales with the rank count
// (for example a missing or duplicated 1/N factor in the slab transform or the
// read-out redistribution) shows up here immediately: the 4-rank result would
// be larger or smaller than the 1-rank result by a factor depending on the
// rank count, while the two must agree to roundoff.
//
// The test is meaningful under ``mpirun -n 1``, ``-n 2`` and ``-n 3`` (the
// CTest entries run it at all three).  A single-process launch without MPI
// still exercises the single-rank path of the distributed transform (which
// must equal the replicated path) so a regression is caught even without a
// multi-rank launch.

#include "picBunch/Compute.hpp"
#include "picBunch/Mpi.hpp"
#include "picBunch/Parallel.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

namespace
{
    //! A small non-cubic grid whose z extent (30) is not divisible by 2 or 3,
    //! so the balanced slab split is exercised.  The bunch is two stationary
    //! electrons well inside the grid, giving a non-trivial, smooth field.
    picBunch::Bunch makeBunch()
    {
        picBunch::Bunch bunch;
        bunch.resize(2);
        bunch.x = {4.0e-6, 6.0e-6};
        bunch.y = {4.0e-6, 5.0e-6};
        bunch.z = {4.0e-6, 5.0e-6};
        bunch.ux = {0.0, 0.0};
        bunch.uy = {0.0, 0.0};
        bunch.uz = {0.0, 0.0};
        bunch.weight = {1.0, 1.0};
        bunch.charge = {-picBunch::kElementaryCharge, -picBunch::kElementaryCharge};
        bunch.restMass = {9.1093837139e-31, 9.1093837139e-31};
        return bunch;
    }

    picBunch::ComputeOptions makeOptions(picBunch::FftMode mode)
    {
        picBunch::ComputeOptions options;
        options.cellSize = {1.0e-6, 1.0e-6, 1.0e-6};
        options.gridSize = {20, 25, 30};
        options.gridOrigin = {0.0, 0.0, 0.0};
        options.fftMode = mode;
        options.tolerance = 1.0e-3;
        return options;
    }

    //! Compute the field with the given FFT mode and return the six components
    //! as a single flat vector (all ranks, full grid, replicated form).
    std::vector<double> computeFlat(
        picBunch::Bunch const& bunch,
        picBunch::ComputeOptions const& options,
        picBunch::ParallelContext const* par)
    {
        picBunch::ComputeOptions local = options;
        local.parallel = par;
        picBunch::Field field = picBunch::computeField(bunch, local);
        // The distributed transform returns a z-slab per rank; gather the full
        // replicated field so the comparison is rank-independent.  The single-
        // rank path already returns the full field.
        std::size_t const nx = field.grid.n[0];
        std::size_t const ny = field.grid.n[1];
        std::size_t const nz = field.grid.n[2];
        std::size_t const plane = nx * ny;
        std::vector<double> out(6u * nz * plane);
        for(int c = 0; c < 6; ++c)
        {
            if(!field.zSlab)
            {
                std::copy(field.data[c].begin(), field.data[c].end(),
                          out.begin() + static_cast<std::size_t>(c) * nz * plane);
            }
            else
            {
                std::vector<double> full(nz * plane, 0.0);
                std::size_t const local = field.slabCount * plane;
                std::vector<int> counts(1), displs(1);
                counts[0] = static_cast<int>(local);
                displs[0] = static_cast<int>(field.slabOffset * plane);
                par->allgatherv(field.data[c].data(), static_cast<int>(local), full.data(),
                                counts.data(), displs.data());
                std::copy(full.begin(), full.end(),
                          out.begin() + static_cast<std::size_t>(c) * nz * plane);
            }
        }
        return out;
    }
} // namespace

TEST(MultiRankField, DistributedMatchesReplicatedOnEveryRank)
{
    picBunch::Bunch const bunch = makeBunch();
    picBunch::MpiParallelContext parallel;
    picBunch::ParallelContext const* par = &parallel;

    // The single-rank replicated result is the reference.
    std::vector<double> const reference = computeFlat(bunch, makeOptions(picBunch::FftMode::Replicated), nullptr);
    // The distributed result at the current rank count must match it to
    // roundoff.  A normalisation defect scales with the rank count.
    std::vector<double> const distributed = computeFlat(bunch, makeOptions(picBunch::FftMode::Distributed), par);
    ASSERT_EQ(reference.size(), distributed.size());
    double const scale = std::max(1.0, reference[0]);
    double maxDiff = 0.0;
    for(std::size_t l = 0; l < reference.size(); ++l)
        maxDiff = std::max(maxDiff, std::abs(reference[l] - distributed[l]));
    EXPECT_LT(maxDiff, 1.0e-12 * scale)
        << "distributed vs replicated max difference " << maxDiff
        << " (rank " << parallel.rank << " of " << parallel.size << ")";
}
