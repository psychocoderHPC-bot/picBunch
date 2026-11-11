// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Diagnostic: compare the distributed slab FFT with a replicated 3-D c2c plan.
// Run with: mpiexec -n 2 ./picBunchDfftCheck

#include "picBunch/Compute.hpp"
#include "picBunch/DistributedFft.hpp"
#include "picBunch/Mpi.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/fft.hpp>
#include <alpaka/math/Complex.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

int main(int argc, char** argv)
{
    using namespace picBunch;
    MpiGuard guard{argc, argv};
    MpiParallelContext parallel;
    int const rank = mpiRank();
    int const size = mpiSize();

#if defined(PICBUNCH_BACKEND_CUDA)
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu);
#elif defined(PICBUNCH_BACKEND_HIP)
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::hip, alpaka::deviceKind::amdGpu);
#else
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::host, alpaka::deviceKind::cpu);
#endif
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    using Complex = alpaka::math::Complex<double>;
    using Extents3 = alpaka::Vec<std::uint32_t, 3u>;

    std::uint32_t const Mx = 8, My = 6, Mz = 4;

    //! Read a field entry from a full field or from the rank's local z-slab.
    auto fieldValue = [](Field const& f, int c, std::size_t i, std::size_t j, std::size_t k) {
        std::size_t const nx = f.grid.n[0];
        std::size_t const ny = f.grid.n[1];
        if(!f.zSlab)
            return f.data[c][i + nx * (j + ny * k)];
        if(k < f.slabOffset || k >= f.slabOffset + f.slabCount)
            return 0.0;
        std::size_t const lk = k - f.slabOffset;
        return f.data[c][i + nx * (j + ny * lk)];
    };

    // End-to-end convolution: replicated vs distributed on a one-particle bunch.
    {
        Bunch bunch;
        bunch.x = {0.0};
        bunch.y = {0.0};
        bunch.z = {0.0};
        bunch.ux = {0.0};
        bunch.uy = {0.0};
        bunch.uz = {10.0};
        bunch.weight = {1.0};
        bunch.charge = {-1.602176634e-19};
        ComputeOptions options;
        options.gridSize = picBunch::Idx3{8, 8, 8};
        options.gridOrigin = picBunch::Vec3{-4e-6, -4e-6, -4e-6};
        options.cellSize = picBunch::Vec3{1e-6, 1e-6, 1e-6};
        options.tolerance = 1e-3;
        options.deterministicDeposit = true;
        options.parallel = &parallel;
        options.fftMode = FftMode::Replicated;
        Field rep = computeField(bunch, options);
        options.fftMode = FftMode::Distributed;
        Field dist = computeField(bunch, options);
        double maxDiff = 0.0;
        double maxRef = 0.0;
        for(int c = 0; c < 6; ++c)
            for(std::size_t k = dist.slabOffset; k < dist.slabOffset + dist.slabCount; ++k)
                for(std::size_t j = 0; j < rep.grid.n[1]; ++j)
                    for(std::size_t i = 0; i < rep.grid.n[0]; ++i)
                    {
                        double const reference = fieldValue(rep, c, i, j, k);
                        maxDiff = std::max(
                            maxDiff,
                            std::abs(reference - fieldValue(dist, c, i, j, k)));
                        maxRef = std::max(maxRef, std::abs(reference));
                    }
        parallel.allreduceMax(&maxDiff, 1);
        parallel.allreduceMax(&maxRef, 1);
        if(rank == 0)
            std::printf(
                "convolution max|rep-dist|=%.6e max|ref|=%.6e rel=%.6e\n",
                maxDiff,
                maxRef,
                maxDiff / std::max(1e-300, maxRef));
        for(int c = 0; c < 6; ++c)
        {
            double mr = 0.0, md = 0.0;
            for(std::size_t k = dist.slabOffset;
                k < dist.slabOffset + dist.slabCount; ++k)
                for(std::size_t j = 0; j < rep.grid.n[1]; ++j)
                    for(std::size_t i = 0; i < rep.grid.n[0]; ++i)
                    {
                        mr = std::max(mr, std::abs(fieldValue(rep, c, i, j, k)));
                        md = std::max(md, std::abs(fieldValue(dist, c, i, j, k)));
                    }
            parallel.allreduceMax(&mr, 1);
            parallel.allreduceMax(&md, 1);
            if(rank == 0)
                std::printf("  c=%d rep=%.6e dist=%.6e\n", c, mr, md);
        }
    }

    detail::DistributedFft3D dfft{device, queue, Mx, My, Mz, &parallel};
    auto* work = dfft.work();
    auto autoReal = [&](std::uint32_t x, std::uint32_t y, std::uint32_t z) {
        return std::sin(0.3 * x + 0.2 * y + 0.1 * z) + 0.5 * std::cos(0.17 * x * y - 0.11 * z);
    };
    for(std::uint32_t zl = 0; zl < dfft.zCount(); ++zl)
        for(std::uint32_t y = 0; y < My; ++y)
            for(std::uint32_t x = 0; x < Mx; ++x)
                work[x + Mx * (y + My * zl)] = Complex{autoReal(x, y, dfft.zOffset() + zl), 0.0};
    dfft.forwardWorkToSpectrum();

    std::size_t const full = std::size_t(Mx) * My * Mz;
    std::vector<Complex> gathered(full, Complex{0.0, 0.0});
    for(std::uint32_t z = 0; z < Mz; ++z)
        for(std::uint32_t y = 0; y < My; ++y)
            for(std::uint32_t xl = 0; xl < dfft.xCount(); ++xl)
                gathered[std::size_t(dfft.xOffset() + xl) + std::size_t(Mx) * (y + std::size_t(My) * z)]
                    = dfft.spectrum()[xl + std::size_t(dfft.xCount()) * (y + std::size_t(My) * z)];
    parallel.allreduceSum(reinterpret_cast<double*>(gathered.data()), 2 * gathered.size());

    // Replicated reference transform.
    std::vector<Complex> reference(full);
    std::vector<Complex> in(full), out(full);
    for(std::uint32_t z = 0; z < Mz; ++z)
        for(std::uint32_t y = 0; y < My; ++y)
            for(std::uint32_t x = 0; x < Mx; ++x)
                in[std::size_t(x) + std::size_t(Mx) * (y + std::size_t(My) * z)]
                    = Complex{autoReal(x, y, z), 0.0};
    auto plan = alpaka::fft::onHost::PlanBuilder<Complex, Extents3>{Extents3{Mz, My, Mx}}
                    .c2c()
                    .outOfPlace()
                    .build(device);
    auto inBuf = alpaka::onHost::allocUnified<Complex>(device, full);
    auto outBuf = alpaka::onHost::allocUnified<Complex>(device, full);
    for(std::size_t l = 0; l < full; ++l)
        inBuf.data()[l] = in[l];
    alpaka::fft::onHost::executeForward(queue, plan, inBuf, outBuf);
    alpaka::onHost::wait(queue);
    double maxDiff = 0.0;
    double maxRef = 0.0;
    for(std::size_t l = 0; l < full; ++l)
    {
        auto const d = gathered[l] - outBuf.data()[l];
        maxDiff = std::max(maxDiff, std::hypot(d.real(), d.imag()));
        maxRef = std::max(maxRef, std::hypot(outBuf.data()[l].real(), outBuf.data()[l].imag()));
    }
    if(rank == 0)
        std::printf(
            "ranks=%d spectrum max|dist-rep|=%.6e max|ref|=%.6e rel=%.6e\n",
            size,
            maxDiff,
            maxRef,
            maxDiff / std::max(1.0, maxRef));

    // Round trip.
    for(std::uint32_t z = 0; z < Mz; ++z)
        for(std::uint32_t y = 0; y < My; ++y)
            for(std::uint32_t xl = 0; xl < dfft.xCount(); ++xl)
                dfft.kernelSpectrum()[xl + std::size_t(dfft.xCount()) * (y + std::size_t(My) * z)]
                    = gathered[std::size_t(dfft.xOffset() + xl) + std::size_t(Mx) * (y + std::size_t(My) * z)];
    dfft.inverseToWork();
    double const norm = 1.0 / double(full);
    double round = 0.0;
    for(std::uint32_t zl = 0; zl < dfft.zCount(); ++zl)
        for(std::uint32_t y = 0; y < My; ++y)
            for(std::uint32_t x = 0; x < Mx; ++x)
                round = std::max(
                    round,
                    std::abs(dfft.work()[x + Mx * (y + My * zl)].real() * norm - autoReal(x, y, dfft.zOffset() + zl)));
    if(rank == 0)
        std::printf("roundtrip max|err|=%.6e\n", round);

    // Physical validation: distributed path against the independent quadrature values.
    {
        auto runCase = [&](Bunch const& bunch, std::vector<std::array<double, 10>> const& rows) {
            ComputeOptions options;
            options.tolerance = 1e-3;
            options.cellSize = {1.0, 1.0, 1.0};
            options.marginCells = 4.0;
            options.deterministicDeposit = true;
            options.parallel = &parallel;
            double const scale = kCoulombConstant * kElementaryCharge;
            auto worstOf = [&](Field const& field) {
                double worst = 0.0;
                for(auto const& row : rows)
                {
                    std::size_t const i = std::size_t(row[0]);
                    std::size_t const j = std::size_t(row[1]);
                    std::size_t const k = std::size_t(row[2]);
                    if(field.zSlab
                       && (k < field.slabOffset || k >= field.slabOffset + field.slabCount))
                        continue; // another rank owns this row
                    for(int c = 0; c < 6; ++c)
                        worst = std::max(
                            worst,
                            std::abs(fieldValue(field, c, i, j, k) - row[3 + c]) / scale);
                }
                parallel.allreduceMax(&worst, 1);
                return worst;
            };
            options.fftMode = FftMode::Replicated;
            Field repField = computeField(bunch, options);
            options.fftMode = FftMode::Distributed;
            Field field = computeField(bunch, options);
            double maxDiff = 0.0;
            for(int c = 0; c < 6; ++c)
                for(std::size_t k = field.slabOffset; k < field.slabOffset + field.slabCount; ++k)
                    for(std::size_t j = 0; j < repField.grid.n[1]; ++j)
                        for(std::size_t i = 0; i < repField.grid.n[0]; ++i)
                            maxDiff = std::max(
                                maxDiff,
                                std::abs(
                                    fieldValue(repField, c, i, j, k)
                                    - fieldValue(field, c, i, j, k)));
            parallel.allreduceMax(&maxDiff, 1);
            if(rank == 0)
                std::printf("  physics rep-vs-dist maxabs=%.6e\n", maxDiff);
            return std::pair<double, double>{worstOf(repField), worstOf(field)};
        };
        Bunch one;
        one.resize(1);
        one.x[0] = 5.0;
        one.y[0] = 5.0;
        one.z[0] = 5.0;
        one.ux[0] = one.uy[0] = 0.0;
        one.uz[0] = 10.0;
        one.weight[0] = 1.0;
        one.charge[0] = -kElementaryCharge;
        std::vector<std::array<double, 10>> oneRows{
            {4, 4, 9, -6.06397185445433445e-14, -6.06397185445428901e-14,
             -4.83305196711727292e-13, 1.49564992178825176e-22, -1.49564992178824000e-22, 0.0, 0.0},
            {9, 4, 5, -1.02519670526764164e-10, -1.09195348836882690e-11,
             -6.75390531206454471e-12, 1.14360184508520855e-20, -1.15885229998238335e-19, 0.0, 0.0}};
        auto const oneWorst = runCase(one, oneRows);

        Bunch three;
        three.resize(3);
        std::array<picBunch::Vec3, 3> const positions{{
            {4.0, 4.0, 4.0}, {4.7, 5.1, 4.3}, {6.2, 5.8, 6.1}}};
        for(std::size_t p = 0; p < 3; ++p)
        {
            three.x[p] = positions[p][0];
            three.y[p] = positions[p][1];
            three.z[p] = positions[p][2];
            three.ux[p] = three.uy[p] = 0.0;
            three.uz[p] = 5.0;
            three.weight[p] = 1.0;
            three.charge[p] = -kElementaryCharge;
        }
        std::vector<std::array<double, 10>> threeRows{
            {10, 10, 10, -6.97936454106874156e-12, -7.21632727373796700e-12,
             -5.25249825806437189e-12, 1.74965105667648229e-20, -1.69763341601991386e-20, 0.0, 0.0},
            {2, 10, 6, 1.51384154525387339e-10, -1.85535170020026201e-10,
             -2.28129662407870640e-11, 4.80484933738683279e-19, 4.20136310563715075e-19, 0.0, 0.0},
            {9, 1, 9, -1.16742752122980636e-11, 1.24458661076109207e-11,
             -8.02589217242825820e-12, -2.67239478050564750e-20, -2.61404222491720268e-20, 0.0, 0.0}};
        auto const threeWorst = runCase(three, threeRows);
        if(rank == 0)
            std::printf(
                "physics: single replicated=%.6e distributed=%.6e of qk; "
                "three replicated=%.6e distributed=%.6e of qk\n",
                oneWorst.first,
                oneWorst.second,
                threeWorst.first,
                threeWorst.second);
    }

    return (maxDiff / std::max(1.0, maxRef) < 1e-12 && round < 1e-12) ? 0 : 1;
}
