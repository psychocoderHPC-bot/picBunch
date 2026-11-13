// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Vendored into the precision campaign from the picBunch reproduction
// run (pr24-20261102) so a fresh reference can be rebuilt without an
// external checkout.
// Standalone producer of the PIConGPU referenceOnly field for one bunch.
//
// It calls exactly the same header-only FP64 routine that
// ReferenceTile.cpp::tileFieldParallel uses (ReferenceKernel.hpp), but without
// the PIConGPU runtime, so the reference field of a production-size bunch can
// be produced on a host without building or running a simulation.
//
// This exists because referenceOnly evaluates the exact shape integral for
// every (tile, source, output) triple: 512 tiles * 1e4 sources * 1536 outputs
// at 64x64x32. That is a multi-CPU-day job inside a simulation, but the same
// arithmetic can be run offline, at a chosen grid and resolution, and written
// to an openPMD-like file for the picBunch cross-validation.
//
// Build:
//   g++ -std=c++20 -O3 -march=native -fopenmp exact_field_driver.cpp -o exact_field_driver
// Usage:
//   exact_field_driver <sources.bin> <nx> <ny> <nz> <dx> <dy> <dz> <tol> <out.bin> [maxLevel] [order]
// sources.bin is a flat double[N*7]: x,y,z in cells, ux,uy,uz, weight.
// out.bin is double[6*nz*ny*nx], component-major, each component (z,y,x).
#include "picongpu/particles/neuralBunch/ReferenceKernel.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace nbr = picongpu::particles::neuralBunch::reference;

int main(int argc, char** argv)
{
    bool const samples = std::strcmp(argv[1], "--samples") == 0;
    char const* const sourcesPath = samples ? argv[2] : argv[1];
    if(argc < (samples ? 11 : 10))
    {
        std::fprintf(
            stderr,
            "usage: %s sources.bin nx ny nz dx dy dz tol out.bin [maxLevel] [order]\n"
            "       %s --samples sources.bin points.bin out.bin tol dx dy dz [maxLevel] [order]\n",
            argv[0],
            argv[0]);
        return 2;
    }
    int const nx = samples ? 0 : std::atoi(argv[2]);
    int const ny = samples ? 0 : std::atoi(argv[3]);
    int const nz = samples ? 0 : std::atoi(argv[4]);
    double const dx = std::atof(argv[samples ? 6 : 5]);
    double const dy = std::atof(argv[samples ? 7 : 6]);
    double const dz = std::atof(argv[samples ? 8 : 7]);
    double const tol = std::atof(argv[samples ? 5 : 8]);
    int const maxLevel = argc > (samples ? 9 : 10) ? std::atoi(argv[samples ? 9 : 10]) : 5;
    int const order = argc > (samples ? 10 : 11) ? std::atoi(argv[samples ? 10 : 11]) : 4;
    if(!samples && (nx % 8 != 0 || ny % 8 != 0 || nz % 4 != 0))
    {
        std::fprintf(stderr, "grid must be a multiple of the 8x8x4 supercell\n");
        return 2;
    }

    std::FILE* in = std::fopen(sourcesPath, "rb");
    if(!in)
    {
        std::perror("sources.bin");
        return 3;
    }
    std::fseek(in, 0, SEEK_END);
    long const bytes = std::ftell(in);
    std::fseek(in, 0, SEEK_SET);
    std::size_t const count = static_cast<std::size_t>(bytes) / (7 * sizeof(double));
    std::vector<double> raw(count * 7);
    if(std::fread(raw.data(), sizeof(double), raw.size(), in) != raw.size())
    {
        std::fprintf(stderr, "short read\n");
        return 3;
    }
    std::fclose(in);

    std::vector<nbr::Source> sources(count);
    for(std::size_t i = 0; i < count; ++i)
    {
        sources[i].x = {raw[7 * i + 0] * dx, raw[7 * i + 1] * dy, raw[7 * i + 2] * dz};
        sources[i].u = {raw[7 * i + 3], raw[7 * i + 4], raw[7 * i + 5]};
        sources[i].w = raw[7 * i + 6];
    }


    // Sample-point mode: points.bin is double[N*4] = (x,y,z in cells of the
    // given cell size, component 0..5). One exact value per point is written
    // to out.bin. This is the feasible way to get truth at a handful of
    // selected outputs of a dense bunch, where the full grid is O(N*neighbours).
    if(samples)
    {
        int const maxLevel = argc > 9 ? std::atoi(argv[9]) : 5;
        int const order = argc > 10 ? std::atoi(argv[10]) : 4;
        std::FILE* pf = std::fopen(argv[3], "rb");
        if(!pf)
        {
            std::perror("points.bin");
            return 3;
        }
        std::fseek(pf, 0, SEEK_END);
        long const pbytes = std::ftell(pf);
        std::fseek(pf, 0, SEEK_SET);
        std::size_t const npoints = static_cast<std::size_t>(pbytes) / (4 * sizeof(double));
        std::vector<double> pts(npoints * 4);
        if(std::fread(pts.data(), sizeof(double), pts.size(), pf) != pts.size())
            return 3;
        std::fclose(pf);
        nbr::Settings settings;
        settings.tol = tol;
        settings.max_level = maxLevel;
        settings.order = order;
        std::vector<double> values(npoints, 0.0);
        auto const tp0 = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
        for(std::size_t ip = 0; ip < npoints; ++ip)
        {
            nbr::Vec3 const target{pts[4 * ip] * dx, pts[4 * ip + 1] * dy, pts[4 * ip + 2] * dz};
            int const comp = static_cast<int>(pts[4 * ip + 3]);
            nbr::Vec3 sum{};
            for(auto const& src : sources)
            {
                nbr::Vec3 E, B;
                nbr::shape_field(target, src.x, src.u, dx, dy, dz, settings, E, B);
                double const value = comp < 3 ? E[comp] : B[comp - 3];
                double& acc = comp == 0 ? sum.x : comp == 1 ? sum.y : comp == 2 ? sum.z
                              : comp == 3 ? sum.x : comp == 4 ? sum.y : sum.z;
                acc += -src.w * dx * dx * value;
            }
            values[ip] = comp == 0 || comp == 1 || comp == 2 ? (comp == 0 ? sum.x : comp == 1 ? sum.y : sum.z)
                                                             : (comp == 3 ? sum.x : comp == 4 ? sum.y : sum.z);
        }
        auto const tp1 = std::chrono::steady_clock::now();
        std::FILE* of = std::fopen(argv[4], "wb");
        if(!of)
            return 4;
        std::fwrite(values.data(), sizeof(double), values.size(), of);
        std::fclose(of);
        std::fprintf(
            stderr,
            "exact_field_driver(points): points=%zu sources=%zu tol=%g seconds=%.3f\n",
            npoints,
            count,
            tol,
            std::chrono::duration<double>(tp1 - tp0).count());
        return 0;
    }

    std::vector<double> field(static_cast<std::size_t>(6) * nz * ny * nx, 0.0);
    nbr::Settings settings;
    settings.tol = tol;
    settings.max_level = maxLevel;
    settings.order = order;

    std::size_t notConverged = 0;
    double maxError = 0.0;
    auto const t0 = std::chrono::steady_clock::now();
    for(int tz = 0; tz < nz / 4; ++tz)
        for(int ty = 0; ty < ny / 8; ++ty)
            for(int tx = 0; tx < nx / 8; ++tx)
            {
                nbr::Vec3 const origin{
                    tx * 8.0 * dx,
                    ty * 8.0 * dy,
                    tz * 4.0 * dz};
                nbr::TileReport report;
                auto const values = nbr::tile_field(origin, sources, dx, dy, dz, settings, &report);
                notConverged += report.not_converged;
                if(report.max_error > maxError)
                    maxError = report.max_error;
                for(int k = 0; k < 4; ++k)
                    for(int j = 0; j < 8; ++j)
                        for(int i = 0; i < 8; ++i)
                        {
                            int const zi = tz * 4 + k;
                            int const yj = ty * 8 + j;
                            int const xi = tx * 8 + i;
                            std::size_t const dst
                                = (static_cast<std::size_t>(zi) * ny + yj) * nx + xi;
                            for(int c = 0; c < 6; ++c)
                                field[static_cast<std::size_t>(c) * nz * ny * nx + dst]
                                    = values[static_cast<std::size_t>(c) + 6u * (i + 8u * (j + 8u * k))];
                        }
            }
    auto const t1 = std::chrono::steady_clock::now();
    double const seconds = std::chrono::duration<double>(t1 - t0).count();

    std::FILE* out = std::fopen(argv[9], "wb");
    if(!out)
    {
        std::perror(argv[9]);
        return 4;
    }
    std::fwrite(field.data(), sizeof(double), field.size(), out);
    std::fclose(out);

    std::fprintf(
        stderr,
        "exact_field_driver: sources=%zu grid=%dx%dx%d cells=(%g,%g,%g) tol=%g level=%d order=%d "
        "seconds=%.3f not_converged=%zu max_error=%.3e\n",
        count,
        nx,
        ny,
        nz,
        dx,
        dy,
        dz,
        tol,
        maxLevel,
        order,
        seconds,
        notConverged,
        maxError);
    return 0;
}
