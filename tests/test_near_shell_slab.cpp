// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Regression guard for the slab-local near-shell correction buffer.
//
// The near-shell assembly used to allocate the correction for the full output
// grid on every rank, which made the fast-gpu path unrunnable at large grids
// and high rank counts even though the base field was already distributed as
// one z-slab per rank.  The correction now covers only the rank's own slab
// (rounded out to whole tile rows).  This test checks two properties:
//
//   1. the correction buffer is smaller than the full grid for a multi-rank
//      launch, and never bigger than the field slab plus one tile row on each
//      side;
//   2. the distributed field equals the replicated (single-rank) field to
//      roundoff, so the result is independent of the rank count.
//
// The CTest entries run the group at one, two and four ranks.

#include "picBunch/Compute.hpp"
#include "picBunch/Mpi.hpp"
#include "picBunch/Parallel.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace
{
    constexpr std::size_t kTile = 2u;

    //! Two stationary and two moving electrons inside a small grid.  The
    //! moving pair keeps the magnetic correction non-zero.
    picBunch::Bunch makeBunch()
    {
        picBunch::Bunch bunch;
        bunch.resize(4);
        bunch.x = {6.0e-6, 9.0e-6, 11.0e-6, 7.0e-6};
        bunch.y = {7.0e-6, 8.0e-6, 6.0e-6, 12.0e-6};
        bunch.z = {8.0e-6, 10.0e-6, 9.0e-6, 7.0e-6};
        bunch.ux = {0.0, 0.0, 1.5, -0.8};
        bunch.uy = {0.0, 0.0, -0.7, 0.4};
        bunch.uz = {0.0, 0.0, 0.9, 1.2};
        bunch.weight = {1.0, 1.0, 1.0, 1.0};
        bunch.charge = {
            -picBunch::kElementaryCharge,
            -picBunch::kElementaryCharge,
            -picBunch::kElementaryCharge,
            -picBunch::kElementaryCharge};
        bunch.restMass = {9.1093837139e-31, 9.1093837139e-31, 9.1093837139e-31, 9.1093837139e-31};
        return bunch;
    }

    picBunch::ComputeOptions makeOptions(
        picBunch::NearShellOptions::Mode mode,
        picBunch::FftMode fftMode,
        std::size_t* correctionNodes = nullptr)
    {
        picBunch::ComputeOptions options;
        options.cellSize = {1.0e-6, 1.0e-6, 1.0e-6};
        options.gridSize = {20, 25, 30};
        options.gridOrigin = {0.0, 0.0, 0.0};
        options.fftMode = fftMode;
        options.tolerance = 1.0e-3;
        options.nearShell.mode = mode;
        options.nearShell.radiusCells = 3.0;
        options.nearShell.tileNodes = kTile;
        options.nearShell.resolvedCorrectionNodes = correctionNodes;
        return options;
    }

    /*! Gather the field into the replicated full-grid form.
     *
     * The per-rank output slabs are disjoint, so a global sum reconstructs the
     * full field.  Replicated fields already hold every node on every rank.
     */
    std::vector<double> computeFlat(
        picBunch::Bunch const& bunch,
        picBunch::ComputeOptions options,
        picBunch::ParallelContext const* par,
        picBunch::Field* outField = nullptr)
    {
        options.parallel = par;
        picBunch::Field field = picBunch::computeField(bunch, options);
        std::size_t const nx = field.grid.n[0];
        std::size_t const ny = field.grid.n[1];
        std::size_t const nz = field.grid.n[2];
        std::size_t const plane = nx * ny;
        std::vector<double> out(6u * nz * plane, 0.0);
        for(int c = 0; c < 6; ++c)
        {
            auto const begin = out.begin() + static_cast<std::ptrdiff_t>(c) * nz * plane;
            if(!field.zSlab)
                std::copy(field.data[c].begin(), field.data[c].end(), begin);
            else
                std::copy(
                    field.data[c].begin(),
                    field.data[c].end(),
                    begin + static_cast<std::ptrdiff_t>(field.slabOffset * plane));
        }
        if(field.zSlab && par != nullptr)
            par->allreduceSum(out.data(), out.size());
        if(outField)
            *outField = std::move(field);
        return out;
    }

    double maxAbs(std::vector<double> const& values)
    {
        double peak = 0.0;
        for(double const value : values)
            peak = std::max(peak, std::abs(value));
        return peak;
    }

    double maxAbsDiff(std::vector<double> const& a, std::vector<double> const& b)
    {
        double diff = 0.0;
        for(std::size_t l = 0; l < a.size(); ++l)
            diff = std::max(diff, std::abs(a[l] - b[l]));
        return diff;
    }
} // namespace

TEST(NearShellSlab, CorrectionBufferIsSlabSized)
{
    picBunch::Bunch const bunch = makeBunch();
    picBunch::MpiParallelContext parallel;
    picBunch::ParallelContext const* par = &parallel;

    std::size_t correctionNodes = 0;
    picBunch::ComputeOptions options
        = makeOptions(picBunch::NearShellOptions::Mode::FastGpu, picBunch::FftMode::Distributed, &correctionNodes);
    picBunch::Field field;
    computeFlat(bunch, options, par, &field);

    std::size_t const plane = field.grid.n[0] * field.grid.n[1];
    std::size_t const full = plane * field.grid.n[2];
    ASSERT_EQ(correctionNodes, field.data[0].size());
    if(par->size > 1)
    {
        EXPECT_TRUE(field.zSlab) << "multi-rank near shell must keep the distributed field";
        // No full-domain correction buffer on any rank.
        EXPECT_LT(correctionNodes, full)
            << "correction is still replicated: " << correctionNodes << " of " << full
            << " nodes (rank " << parallel.rank << " of " << parallel.size << ")";
        // Slab plus at most one tile row of overlap on each side.
        EXPECT_LE(correctionNodes, plane * (field.slabCount + 2u * kTile))
            << "correction exceeds the slab plus tile-row overlap (rank " << parallel.rank
            << " of " << parallel.size << ")";
    }
    else
        EXPECT_EQ(correctionNodes, full);
}

TEST(NearShellSlab, DistributedMatchesReplicated)
{
    picBunch::Bunch const bunch = makeBunch();
    picBunch::MpiParallelContext parallel;
    picBunch::ParallelContext const* par = &parallel;

    // The replicated near-shell field is the rank-count-independent reference:
    // every rank computes it independently without communication.
    std::vector<double> const reference = computeFlat(
        bunch,
        makeOptions(picBunch::NearShellOptions::Mode::FastGpu, picBunch::FftMode::Replicated),
        nullptr);
    std::vector<double> const distributed = computeFlat(
        bunch,
        makeOptions(picBunch::NearShellOptions::Mode::FastGpu, picBunch::FftMode::Distributed),
        par);

    ASSERT_EQ(reference.size(), distributed.size());
    double const scale = std::max(1.0, maxAbs(reference));
    EXPECT_LT(maxAbsDiff(reference, distributed), 1.0e-12 * scale)
        << "slab-corrected near shell differs from the replicated reference (rank "
        << parallel.rank << " of " << parallel.size << ")";
}

TEST(NearShellSlab, FarFieldMatchesNoNearShell)
{
    // Outside the shell the correction is exactly zero, so the fast-gpu field
    // must equal the plain mesh there.  This is the near-shell contract and it
    // must not depend on the rank count.
    picBunch::Bunch const bunch = makeBunch();
    picBunch::MpiParallelContext parallel;
    picBunch::ParallelContext const* par = &parallel;

    std::vector<double> const mesh = computeFlat(
        bunch,
        makeOptions(picBunch::NearShellOptions::Mode::None, picBunch::FftMode::Distributed),
        par);
    std::vector<double> const fast = computeFlat(
        bunch,
        makeOptions(picBunch::NearShellOptions::Mode::FastGpu, picBunch::FftMode::Distributed),
        par);

    ASSERT_EQ(mesh.size(), fast.size());
    std::size_t const nx = 20;
    std::size_t const ny = 25;
    std::size_t const nz = 30;
    std::size_t const plane = nx * ny;
    double const radius = 3.0;
    double const scale = std::max(1.0, maxAbs(mesh));
    for(std::size_t c = 0; c < 6; ++c)
        for(std::size_t k = 0; k < nz; ++k)
            for(std::size_t j = 0; j < ny; ++j)
                for(std::size_t i = 0; i < nx; ++i)
                {
                    double minDistance = 1.0e300;
                    for(std::size_t q = 0; q < bunch.size(); ++q)
                    {
                        double const dx = static_cast<double>(i) - bunch.x[q] / 1.0e-6;
                        double const dy = static_cast<double>(j) - bunch.y[q] / 1.0e-6;
                        double const dz = static_cast<double>(k) - bunch.z[q] / 1.0e-6;
                        minDistance = std::min(minDistance, std::sqrt(dx * dx + dy * dy + dz * dz));
                    }
                    if(minDistance <= radius + 1.0)
                        continue;
                    std::size_t const index = c * nz * plane + k * plane + j * nx + i;
                    EXPECT_LE(std::abs(mesh[index] - fast[index]), 1.0e-12 * scale)
                        << "rank " << parallel.rank << " of " << parallel.size << " node ("
                        << i << ", " << j << ", " << k << "), component " << c;
                }
}
