// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Field probe used to characterise the grid dependence of the computed field
// and to mirror the acceptance criteria of the grid-independence regression
// test (tests/test_grid_independence.cpp).  It reads a contiguous particle
// range of an openPMD series, computes the field at one or more grid sizes,
// and reports on rank 0:
//
//   * max|E| in a 5x5x5 node box around the grid centre (the bunch centroid
//     sits at the grid centre by construction),
//   * max|E| in a 3x3x3 node box around the grid corner node,
//   * the global max|E|,
//   * the field energy  0.5 * eps0 * sum |E|^2 dV  (SI joules).
//
// The grid origin is chosen so that the bunch bounding-box centre sits at the
// grid centre; the same rule is used by the regression test.  The probe never
// writes an output file.
//
// usage: fieldprobe --input <openPMD file> [--species b] [--count 1000]
//                   --cell <h> --grids 128,256,512,640 --mode none|fast-gpu
//                   [--tolerance 1e-3] [--near-radius 3] [--near-tile 4]

#include "picBunch/Compute.hpp"
#include "picBunch/Mpi.hpp"
#include "picBunch/OpenPMD.hpp"
#include "picBunch/Types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace pb = picBunch;

namespace
{
    double parseDouble(char const* s)
    {
        return std::atof(s);
    }

    std::vector<double> parseList(char const* s)
    {
        std::vector<double> out;
        std::string t(s);
        std::size_t pos = 0;
        while(pos < t.size())
        {
            std::size_t const comma = t.find(',', pos);
            std::string const part
                = (comma == std::string::npos)
                    ? t.substr(pos)
                    : t.substr(pos, comma - pos);
            out.push_back(std::atof(part.c_str()));
            if(comma == std::string::npos)
                break;
            pos = comma + 1;
        }
        return out;
    }

    pb::Bunch readBroadcast(
        std::string const& input,
        std::string const& species,
        std::size_t count,
        int rank)
    {
        pb::BunchSelection selection;
        if(!species.empty())
            selection.speciesName = species;
        std::size_t const total = (rank == 0) ? pb::readParticleCount(input, selection) : 0u;
        std::size_t const n = std::min(count, total);
        pb::BunchData data;
        if(rank == 0)
            data = pb::readBunchRange(input, selection, 0, n);
        pb::mpiBroadcastBunch(data.bunch, 0);
        return data.bunch;
    }

    //! Bounding-box centre of the bunch (the grid-centre anchor).
    std::array<double, 3> centreOf(pb::Bunch const& bunch)
    {
        auto const b = bunch.bounds();
        std::array<double, 3> c{};
        for(int a = 0; a < 3; ++a)
            c[a] = 0.5 * (b[0][a] + b[1][a]);
        return c;
    }

    //! Bounding-box half-extent of the bunch.
    std::array<double, 3> halfExtent(pb::Bunch const& bunch)
    {
        auto const b = bunch.bounds();
        std::array<double, 3> h{};
        for(int a = 0; a < 3; ++a)
            h[a] = 0.5 * (b[1][a] - b[0][a]);
        return h;
    }

    pb::ComputeOptions makeOptions(
        pb::Bunch const& bunch,
        std::size_t baseN,
        double cell,
        char const* mode,
        double tolerance,
        double nearRadius,
        std::size_t nearTile,
        char const* fftMode,
        int rank,
        int size,
        unsigned fftPadding)
    {
        std::array<double, 3> const centre = centreOf(bunch);
        std::array<double, 3> const half = halfExtent(bunch);
        // The grid must contain every particle.  The bounding-box half-extent
        // gives the minimum half-grid length needed on each axis.
        std::size_t const minNx = static_cast<std::size_t>(
            std::ceil(2.0 * half[0] / cell)) + 2u;
        std::size_t const minNy = static_cast<std::size_t>(
            std::ceil(2.0 * half[1] / cell)) + 2u;
        std::size_t const minNz = static_cast<std::size_t>(
            std::ceil(2.0 * half[2] / cell)) + 2u;
        std::size_t nx = std::max(baseN, minNx);
        std::size_t ny = std::max(2u * baseN, minNy);
        std::size_t nz = std::max(baseN, minNz);
        // Make even for R2C.
        if(nx % 2u) ++nx;
        if(ny % 2u) ++ny;
        if(nz % 2u) ++nz;
        // For the distributed z-slab transform, every rank owns a contiguous
        // z-slab.  The slab boundaries fall at multiples of the per-rank slab
        // size, so the full grid z extent must be a multiple of the rank count.
        if(size > 1 && nz % static_cast<std::size_t>(size) != 0u)
        {
            nz += static_cast<std::size_t>(size) - nz % static_cast<std::size_t>(size);
        }
        pb::ComputeOptions options;
        options.cellSize = {cell, cell, cell};
        options.gridSize = {nx, ny, nz};
        options.gridOrigin
            = {centre[0] - 0.5 * static_cast<double>(nx) * cell,
                centre[1] - 0.5 * static_cast<double>(ny) * cell,
                centre[2] - 0.5 * static_cast<double>(nz) * cell};
        options.tolerance = tolerance;
        options.fftPadding = fftPadding;
        if(std::strcmp(fftMode, "replicated") == 0)
            options.fftMode = pb::FftMode::Replicated;
        else if(std::strcmp(fftMode, "distributed") == 0)
            options.fftMode = pb::FftMode::Distributed;
        else
            options.fftMode = pb::FftMode::Auto;
        options.nearShell.radiusCells = nearRadius;
        options.nearShell.tileNodes = nearTile;
        if(std::strcmp(mode, "none") == 0)
            options.nearShell.mode = pb::NearShellOptions::Mode::None;
        else if(std::strcmp(mode, "fast") == 0)
            options.nearShell.mode = pb::NearShellOptions::Mode::Fast;
        else
            options.nearShell.mode = pb::NearShellOptions::Mode::FastGpu;
        return options;
    }

    //! Gather the full replicated field on every rank.  The distributed
    //! transform returns one z-slab per rank in the same balanced contiguous
    //! layout the near-shell gather uses.
    void gatherFull(pb::Field const& field, pb::MpiParallelContext const& par,
                    std::vector<std::vector<double>>& full)
    {
        std::size_t const pnx = field.grid.n[0];
        std::size_t const pny = field.grid.n[1];
        std::size_t const pnz = field.grid.n[2];
        std::size_t const plane = pnx * pny;
        full.resize(6);
        for(int c = 0; c < 6; ++c)
            full[c].assign(pnz * plane, 0.0);
        if(!field.zSlab)
        {
            for(int c = 0; c < 6; ++c)
                std::copy(field.data[c].begin(), field.data[c].end(), full[c].begin());
            return;
        }
        int const rank = par.rank;
        int const size = par.size;
        std::vector<int> counts(size), displs(size);
        std::size_t const base = pnz / static_cast<std::size_t>(size);
        std::size_t const rem = pnz % static_cast<std::size_t>(size);
        for(int r = 0; r < size; ++r)
        {
            std::size_t const rb
                = static_cast<std::size_t>(r) * base + std::min(static_cast<std::size_t>(r), rem);
            std::size_t const rc = base + (static_cast<std::size_t>(r) < rem ? 1u : 0u);
            counts[r] = static_cast<int>(rc * plane);
            displs[r] = static_cast<int>(rb * plane);
        }
        for(int c = 0; c < 6; ++c)
        {
            par.allgatherv(field.data[c].data(), counts[rank], full[c].data(),
                           counts.data(), displs.data());
        }
    }

    //! max|E_c| over the 3 components at nodes inside the axis-aligned box
    //! centred on `centre` with per-axis half-width `half`.
    template<typename TFull>
    double boxMax(
        TFull const& full,
        std::size_t pnx,
        std::size_t pny,
        std::size_t pnz,
        std::array<std::size_t, 3> const& centre,
        std::size_t half)
    {
        double m = 0.0;
        std::size_t const k0 = (centre[2] >= half) ? centre[2] - half : 0;
        std::size_t const k1 = std::min(pnz, centre[2] + half + 1);
        std::size_t const j0 = (centre[1] >= half) ? centre[1] - half : 0;
        std::size_t const j1 = std::min(pny, centre[1] + half + 1);
        std::size_t const i0 = (centre[0] >= half) ? centre[0] - half : 0;
        std::size_t const i1 = std::min(pnx, centre[0] + half + 1);
        for(std::size_t k = k0; k < k1; ++k)
            for(std::size_t j = j0; j < j1; ++j)
                for(std::size_t i = i0; i < i1; ++i)
                {
                    std::size_t const l = i + pnx * (j + pny * k);
                    for(int c = 0; c < 3; ++c)
                        m = std::max(m, std::abs(full[c][l]));
                }
        return m;
    }
}

int main(int argc, char** argv)
{
    pb::MpiGuard guard(argc, argv);
    pb::MpiParallelContext par;
    int const rank = pb::mpiRank();
    int const size = pb::mpiSize();

    std::string input;
    std::string species;
    std::size_t count = 1000;
    double cell = 1.772e-7;
    std::vector<std::size_t> grids;
    std::string mode = "fast-gpu";
    double tolerance = 1.0e-3;
    double nearRadius = 3.0;
    std::size_t nearTile = 4;
    std::string fftMode = "auto";
    unsigned fftPadding = 0;
    for(int i = 1; i < argc; ++i)
    {
        std::string const o = argv[i];
        auto next = [&]() -> char const* {
            if(i + 1 >= argc)
                throw std::runtime_error("missing value for " + o);
            return argv[++i];
        };
        if(o == "--input")
            input = next();
        else if(o == "--species")
            species = next();
        else if(o == "--count")
            count = static_cast<std::size_t>(std::atof(next()));
        else if(o == "--cell")
            cell = parseDouble(next());
        else if(o == "--grids")
            for(double v : parseList(next()))
                grids.push_back(static_cast<std::size_t>(v));
        else if(o == "--mode")
            mode = next();
        else if(o == "--tolerance")
            tolerance = parseDouble(next());
        else if(o == "--near-radius")
            nearRadius = parseDouble(next());
        else if(o == "--near-tile")
            nearTile = static_cast<std::size_t>(std::atof(next()));
        else if(o == "--fft-mode")
            fftMode = next();
        else if(o == "--fft-padding")
            fftPadding = static_cast<unsigned>(std::atof(next()));
        else
            throw std::runtime_error("unknown argument " + o);
    }
    if(input.empty() || grids.empty())
        throw std::runtime_error("fieldprobe: --input and --grids are required");

    pb::Bunch const bunch = readBroadcast(input, species, count, rank);

    std::array<double, 3> const centre = centreOf(bunch);
    if(rank == 0)
        std::fprintf(
            stderr,
            "[probe] %zu particles, cell=%.9g m, mode=%s, centroid=(%.9g, %.9g, %.9g)\n",
            bunch.size(), cell, mode.c_str(), centre[0], centre[1], centre[2]);

    double const dV = cell * cell * cell;
    double const eps0 = 8.8541878128e-12;
    std::size_t const centroidHalf = 2;
    std::size_t const cornerHalf = 1;

    for(std::size_t const baseN : grids)
    {
        pb::ComputeOptions options
            = makeOptions(
                bunch, baseN, cell, mode.c_str(), tolerance, nearRadius, nearTile,
                fftMode.c_str(), rank, size, fftPadding);
        options.parallel = &par;
        pb::Field field = pb::computeField(bunch, options);

        std::size_t const pnx = field.grid.n[0];
        std::size_t const pny = field.grid.n[1];
        std::size_t const pnz = field.grid.n[2];
        std::vector<std::vector<double>> full;
        gatherFull(field, par, full);

        std::array<std::size_t, 3> centIdx{pnx / 2, pny / 2, pnz / 2};
        std::array<std::size_t, 3> cornerIdx{0, 0, 0};

        double eCentroid = boxMax(full, pnx, pny, pnz, centIdx, centroidHalf);
        double eCorner = boxMax(full, pnx, pny, pnz, cornerIdx, cornerHalf);
        double globalMax = 0.0;
        double energy = 0.0;
        for(int c = 0; c < 3; ++c)
            for(double v : full[c])
            {
                globalMax = std::max(globalMax, std::abs(v));
                energy += v * v;
            }
        par.allreduceMax(&globalMax, 1);
        par.allreduceMax(&eCentroid, 1);
        par.allreduceMax(&eCorner, 1);
        par.allreduceSum(&energy, 1);
        energy *= 0.5 * eps0 * dV;

        if(rank == 0)
        {
            double const ratio = (eCentroid > 0.0) ? eCorner / eCentroid : 0.0;
            std::printf(
                "grid %zux%zux%zu mode %s ranks %d | centroid %.6e corner %.6e ratio %.4g globalMax %.6e energy %.6e J\n",
                pnx, pny, pnz, mode.c_str(), size,
                eCentroid, eCorner, ratio, globalMax, energy);
            std::fflush(stdout);
        }
    }
    return 0;
}
