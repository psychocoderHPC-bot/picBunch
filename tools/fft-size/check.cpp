// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Characterise the FFT-extent dependence of the replicated convolution path.
//
// For every padded extent M the tool reproduces, line for line, the buffer
// allocation and plan construction of picBunch::computeField
// (src/Compute.cpp): 3-D `allocUnified` buffers, R2C/C2R plans built without
// explicit strides (i.e. with the vendor's default contiguous layout).
//
// Three checks per size:
//   A3: vendor R2C on the 3-D allocated buffer, compared element by element
//       with an independent FFTW reference transform of the same flat data.
//   A1: the same transform on a 1-D (therefore contiguous) buffer, same plan.
//   B:  the full pipeline (TSC deposit, kernel build, forward, multiply,
//       backward, read-out) against a direct sum of the free-space point
//       field over the deposited grid nodes.  A single particle at the box
//       centre (symmetric) and a three-particle cloud (the task's
//       "small cloud") are both evaluated; the cloud also reports the
//       worst-node value for comparison against the near-shell evidence.
//
// Exit code 0 when every reported relative error is below the round-off
// floor; the per-size table is always printed.

#include "picBunch/Kernel.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/fft.hpp>
#include <alpaka/math/Complex.hpp>

#include <fftw3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace
{
    using Complex = alpaka::math::Complex<double>;
    using Extents3 = alpaka::Vec<std::uint32_t, 3u>;

    double const kRoundOffFloor = 1.0e-12;

    // Stage skip flags for state-contamination bisection (--noA3 / --noA1).
    bool gSkipA3 = false;
    bool gSkipA1 = false;
    // Stage-B lifecycle: gBins >= 2 runs the production bin loop (one
    // memset+deposit+FFT pass per cloud, fields summed); gBuf1D allocates the
    // stage-B buffers flat 1-D (the fix) instead of 3-D allocUnified.
    int gBins = 1;
    bool gBuf1D = false;

    std::vector<std::uint32_t> parseSizes(char const* arg)
    {
        std::vector<std::uint32_t> sizes;
        char const* p = arg;
        while(*p != '\0')
        {
            char* end = nullptr;
            unsigned long v = std::strtoul(p, &end, 10);
            if(end == p || v < 2u || v > 512u)
                throw std::invalid_argument(std::string("bad size in '") + arg + "'");
            sizes.push_back(static_cast<std::uint32_t>(v));
            p = (*end == ',') ? end + 1 : end;
        }
        if(sizes.empty())
            throw std::invalid_argument("no sizes given");
        return sizes;
    }

    //! Deterministic, spectrally rich pattern indexed by the flat node order
    //! (x fastest) that the whole pipeline uses.
    double patternAt(std::uint32_t i, std::uint32_t j, std::uint32_t k)
    {
        double const x = static_cast<double>(i);
        double const y = static_cast<double>(j);
        double const z = static_cast<double>(k);
        return std::sin(0.7 * x + 0.3 * y) * std::cos(0.5 * z) + 0.5 * std::cos(0.11 * x - 0.9 * y + 0.31 * z);
    }

    //! FFTW reference: forward R2C of the flat (x fastest) real array.
    std::vector<Complex> fftwReferenceR2C(std::vector<double> const& in, std::uint32_t M)
    {
        std::uint32_t const CP = M / 2u + 1u;
        // Portable complex buffer: std::vector<fftw_complex> is invalid when
        // fftw_complex is the classic double[2] typedef (nvcc rejects it).
        std::vector<std::array<double, 2>> out(static_cast<std::size_t>(M) * M * CP);
        std::array<int, 3> n{static_cast<int>(M), static_cast<int>(M), static_cast<int>(M)};
        std::array<int, 3> inembed{static_cast<int>(M), static_cast<int>(M), static_cast<int>(M)};
        std::array<int, 3> onembed{static_cast<int>(M), static_cast<int>(M), static_cast<int>(CP)};
        // The plan-constructor ABI differs between FFTW generations: classic
        // 3.3 takes a `sign` argument; the newer FFTW-compatible builds
        // (e.g. the MLIR-derived 3.6 on hal) omit it and transform forward.
        // CMake probes the installed header and defines PBFFTW3_PLAN_TAKES_SIGN.
        fftw_plan plan = nullptr;
#ifdef PBFFTW3_PLAN_TAKES_SIGN
        plan = fftw_plan_many_dft_r2c(
            3,
            n.data(),
            1,
            const_cast<double*>(in.data()),
            inembed.data(),
            1,
            static_cast<int>(M * M * M),
            reinterpret_cast<fftw_complex*>(out.data()),
            onembed.data(),
            1,
            static_cast<int>(M * M * CP),
            FFTW_FORWARD,
            FFTW_ESTIMATE | FFTW_UNALIGNED);
#else
        plan = fftw_plan_many_dft_r2c(
            3,
            n.data(),
            1,
            const_cast<double*>(in.data()),
            inembed.data(),
            1,
            static_cast<int>(M * M * M),
            reinterpret_cast<fftw_complex*>(out.data()),
            onembed.data(),
            1,
            static_cast<int>(M * M * CP),
            FFTW_ESTIMATE | FFTW_UNALIGNED);
#endif
        fftw_execute(plan);
        fftw_destroy_plan(plan);
        std::vector<Complex> spectrum(static_cast<std::size_t>(M) * M * CP);
        for(std::size_t l = 0; l < spectrum.size(); ++l)
            spectrum[l] = Complex{out[l][0], out[l][1]};
        return spectrum;
    }

    //! FFTW reference: inverse, UNNORMALISED C2R of the packed half-spectrum
    //! (same convention as the vendor's executeBackward, verified in stage A3).
    std::vector<double> fftwReferenceC2R(std::vector<Complex> const& in, std::uint32_t M)
    {
        std::uint32_t const CP = M / 2u + 1u;
        std::size_t const specCount = static_cast<std::size_t>(M) * M * CP;
        std::vector<std::array<double, 2>> inC(specCount);
        for(std::size_t l = 0; l < specCount; ++l)
        {
            inC[l][0] = in[l].real();
            inC[l][1] = in[l].imag();
        }
        std::vector<double> out(static_cast<std::size_t>(M) * M * M);
        std::array<int, 3> n{static_cast<int>(M), static_cast<int>(M), static_cast<int>(M)};
        std::array<int, 3> inembed{static_cast<int>(M), static_cast<int>(M), static_cast<int>(CP)};
        std::array<int, 3> onembed{static_cast<int>(M), static_cast<int>(M), static_cast<int>(M)};
        fftw_plan plan = nullptr;
#ifdef PBFFTW3_PLAN_TAKES_SIGN
        plan = fftw_plan_many_dft_c2r(
            3,
            n.data(),
            1,
            reinterpret_cast<fftw_complex*>(inC.data()),
            inembed.data(),
            1,
            static_cast<int>(M * M * CP),
            out.data(),
            onembed.data(),
            1,
            static_cast<int>(M * M * M),
            FFTW_BACKWARD,
            FFTW_ESTIMATE | FFTW_UNALIGNED);
#else
        plan = fftw_plan_many_dft_c2r(
            3,
            n.data(),
            1,
            reinterpret_cast<fftw_complex*>(inC.data()),
            inembed.data(),
            1,
            static_cast<int>(M * M * CP),
            out.data(),
            onembed.data(),
            1,
            static_cast<int>(M * M * M),
            FFTW_ESTIMATE | FFTW_UNALIGNED);
#endif
        fftw_execute(plan);
        fftw_destroy_plan(plan);
        return out;
    }

    //! Max |a - b| and max |b| over two flat complex arrays.
    void maxAbsDelta(
        Complex const* a,
        Complex const* b,
        std::size_t count,
        double* maxDelta,
        double* maxRef)
    {
        double d = 0.0;
        double r = 0.0;
        for(std::size_t l = 0; l < count; ++l)
        {
            double const dr = a[l].real() - b[l].real();
            double const di = a[l].imag() - b[l].imag();
            d = std::max(d, std::sqrt(dr * dr + di * di));
            r = std::max(r, std::sqrt(b[l].real() * b[l].real() + b[l].imag() * b[l].imag()));
        }
        *maxDelta = d;
        *maxRef = r;
    }

    struct DepositNodes
    {
        std::vector<std::array<std::int64_t, 3>> node;
        std::vector<double> weight;
    };

    //! TSC deposit of the particle list into the unpadded grid, exactly the
    //! stencil of src/Compute.cpp depositHost (node units, charge included).
    DepositNodes depositNodes(
        std::vector<std::array<double, 3>> const& pos,
        std::vector<double> const& charge,
        std::array<std::int64_t, 3> const& origin,
        std::uint32_t n)
    {
        DepositNodes out;
        for(std::size_t p = 0; p < pos.size(); ++p)
        {
            std::int64_t lo[3];
            std::int64_t hi[3];
            double t[3];
            for(int a = 0; a < 3; ++a)
            {
                t[a] = pos[p][a] - static_cast<double>(origin[a]);
                lo[a] = static_cast<std::int64_t>(std::ceil(t[a] - 1.5));
                hi[a] = static_cast<std::int64_t>(std::floor(t[a] + 1.5));
            }
            for(std::int64_t ia = lo[0]; ia <= hi[0]; ++ia)
                if(ia >= 0 && ia < static_cast<std::int64_t>(n))
                    for(std::int64_t ib = lo[1]; ib <= hi[1]; ++ib)
                        if(ib >= 0 && ib < static_cast<std::int64_t>(n))
                            for(std::int64_t ic = lo[2]; ic <= hi[2]; ++ic)
                                if(ic >= 0 && ic < static_cast<std::int64_t>(n))
                                {
                                    double const w
                                        = picBunch::tscShape(static_cast<double>(ia) - t[0])
                                        * picBunch::tscShape(static_cast<double>(ib) - t[1])
                                        * picBunch::tscShape(static_cast<double>(ic) - t[2]);
                                    if(w == 0.0)
                                        continue;
                                    out.node.push_back({ia, ib, ic});
                                    out.weight.push_back(w * charge[p]);
                                }
        }
        return out;
    }

    //! The kernel build of KernelBuildKernel (src/Compute.cpp) for component
    //! 0 (E_x), velocity zero, cell 1, Yee offset 0, evaluated host-side.
    void buildKernel(std::vector<double>& kernel, std::uint32_t M)
    {
        kernel.assign(static_cast<std::size_t>(M) * M * M, 0.0);
        for(std::uint32_t k = 0; k < M; ++k)
            for(std::uint32_t j = 0; j < M; ++j)
                for(std::uint32_t i = 0; i < M; ++i)
                {
                    int const di = (i <= (M - 1u) / 2u) ? static_cast<int>(i) : static_cast<int>(i) - static_cast<int>(M);
                    int const dj = (j <= (M - 1u) / 2u) ? static_cast<int>(j) : static_cast<int>(j) - static_cast<int>(M);
                    int const dk = (k <= (M - 1u) / 2u) ? static_cast<int>(k) : static_cast<int>(k) - static_cast<int>(M);
                    double field[6];
                    picBunch::pointFieldNormalized(
                        static_cast<double>(di),
                        static_cast<double>(dj),
                        static_cast<double>(dk),
                        0.0,
                        0.0,
                        0.0,
                        field);
                    double const value = field[0];
                    kernel[static_cast<std::size_t>(i)
                            + static_cast<std::size_t>(M) * (static_cast<std::size_t>(j)
                                + static_cast<std::size_t>(M) * static_cast<std::size_t>(k))]
                        = std::isfinite(value) ? value : 0.0;
                }
    }

    //! Direct sum: free-space point field of the deposited nodes, evaluated
    //! on the unpadded grid (node units).  ``periodic`` adds the |k| <= 2
    //! periodic images of every source node, i.e. the true circular
    //! convolution reference.
    void directSum(
        std::vector<double>& field,
        DepositNodes const& dep,
        std::array<std::int64_t, 3> const& origin,
        std::uint32_t n,
        std::uint32_t M,
        bool periodic)
    {
        field.assign(static_cast<std::size_t>(n) * n * n, 0.0);
#pragma omp parallel for collapse(3)
        for(std::uint32_t k = 0; k < n; ++k)
            for(std::uint32_t j = 0; j < n; ++j)
                for(std::uint32_t i = 0; i < n; ++i)
                {
                    double ox = static_cast<double>(i + origin[0]);
                    double oy = static_cast<double>(j + origin[1]);
                    double oz = static_cast<double>(k + origin[2]);
                    double sum = 0.0;
                    for(std::size_t q = 0; q < dep.node.size(); ++q)
                    {
                        double sx = static_cast<double>(dep.node[q][0] + origin[0]);
                        double sy = static_cast<double>(dep.node[q][1] + origin[1]);
                        double sz = static_cast<double>(dep.node[q][2] + origin[2]);
                        for(int kk = periodic ? -2 : 0; kk <= (periodic ? 2 : 0); ++kk)
                            for(int jj = periodic ? -2 : 0; jj <= (periodic ? 2 : 0); ++jj)
                                for(int ii = periodic ? -2 : 0; ii <= (periodic ? 2 : 0); ++ii)
                                {
                                    if(!periodic && (ii != 0 || jj != 0 || kk != 0))
                                        continue;
                                    double Rx = ox - sx - static_cast<double>(ii) * static_cast<double>(M);
                                    double Ry = oy - sy - static_cast<double>(jj) * static_cast<double>(M);
                                    double Rz = oz - sz - static_cast<double>(kk) * static_cast<double>(M);
                                    double field6[6];
                                    picBunch::pointFieldNormalized(Rx, Ry, Rz, 0.0, 0.0, 0.0, field6);
                                    sum += dep.weight[q] * field6[0];
                                }
                    }
                    field[static_cast<std::size_t>(i) + static_cast<std::size_t>(n) * (static_cast<std::size_t>(j)
                        + static_cast<std::size_t>(n) * static_cast<std::size_t>(k))]
                        = sum;
                }
    }

    struct SizeReport
    {
        std::uint32_t M = 0;
        std::uint32_t n = 0;
        // Buffer metadata of the 3-D allocation, in bytes.
        std::size_t pitchX = 0;
        std::size_t pitchY = 0;
        std::size_t pitchZ = 0;
        bool contiguous = false;
        // Stage A3: vendor R2C on the 3-D buffer vs FFTW.
        double a3SpectrumRel = 0.0;
        double a3BackwardRel = 0.0;
        // Stage A1: vendor R2C/C2R on a 1-D contiguous buffer vs FFTW.
        double a1SpectrumRel = 0.0;
        double a1BackwardRel = 0.0;
        // Stage B: pipeline vs direct sum (component E_x, scale applied).
        double bPointFreeRel = 0.0;
        double bPointCircRel = 0.0;
        double bCloudFreeRel = 0.0;
        double bCloudCircRel = 0.0;
        double bMaxRef = 0.0;
        double fftCloudValue = 0.0;
        double refCloudValue = 0.0;
        std::uint32_t worstNode[3] = {0, 0, 0};
    };

    template<typename T_Device, typename T_Queue>
    SizeReport runSize(std::uint32_t M, T_Device& device, T_Queue& queue)
    {
        SizeReport rep;
        rep.M = M;
        rep.n = M / 2u;
        std::uint32_t const n = rep.n;
        std::uint32_t const CP = M / 2u + 1u;
        std::size_t const realCount = static_cast<std::size_t>(M) * M * M;
        std::size_t const specCount = static_cast<std::size_t>(M) * M * CP;

        Extents3 const realExt{M, M, M};
        Extents3 const specExt{M, M, CP};

        // ------------------------------------------------ 3-D buffers, as in
        // src/Compute.cpp lines 809-813.
        auto rhoReal3 = alpaka::onHost::allocUnified<double>(device, realExt);
        auto specReal3 = alpaka::onHost::allocUnified<Complex>(device, specExt);
        auto kernelReal3 = alpaka::onHost::allocUnified<double>(device, realExt);
        auto kernelSpec3 = alpaka::onHost::allocUnified<Complex>(device, specExt);
        auto outReal3 = alpaka::onHost::allocUnified<double>(device, realExt);

        auto const pitches = rhoReal3.getPitches();
        rep.pitchX = static_cast<std::size_t>(pitches[2]);
        rep.pitchY = static_cast<std::size_t>(pitches[1]);
        rep.pitchZ = static_cast<std::size_t>(pitches[0]);
        rep.contiguous = (rep.pitchX == sizeof(double) && rep.pitchY == static_cast<std::size_t>(M) * sizeof(double)
            && rep.pitchZ == static_cast<std::size_t>(M) * M * sizeof(double));

        // Deterministic fill through the flat interface the production code
        // uses everywhere (deposit, kernels, read-out).
        std::vector<double> flat(realCount);
        for(std::size_t k = 0; k < M; ++k)
            for(std::size_t j = 0; j < M; ++j)
                for(std::size_t i = 0; i < M; ++i)
                {
                    flat[i + M * (j + M * k)] = patternAt(static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(j), static_cast<std::uint32_t>(k));
                    rhoReal3.data()[i + M * (j + M * k)] = flat[i + M * (j + M * k)];
                }
        std::vector<Complex> reference = fftwReferenceR2C(flat, M);

        // Plans exactly as in src/Compute.cpp lines 942-945.
        auto r2cPlan = alpaka::fft::onHost::PlanBuilder<double, Extents3>{realExt}.r2c().outOfPlace().build(device);
        auto c2rPlan = alpaka::fft::onHost::PlanBuilder<double, Extents3>{realExt}.c2r().outOfPlace().build(device);

        // ------------------------------------------------ Stage A3.
        if(!gSkipA3)
        {
            alpaka::fft::onHost::executeForward(queue, r2cPlan, rhoReal3, specReal3);
            alpaka::onHost::wait(queue);
            {
                double d = 0.0;
                double r = 0.0;
                maxAbsDelta(specReal3.data(), reference.data(), specCount, &d, &r);
                rep.a3SpectrumRel = (r > 0.0) ? d / r : d;
            }
            // C2R of the reference spectrum must return M^3 * input.
            for(std::size_t l = 0; l < specCount; ++l)
                kernelSpec3.data()[l] = reference[l];
            alpaka::fft::onHost::executeBackward(queue, c2rPlan, kernelSpec3, outReal3);
            alpaka::onHost::wait(queue);
            {
                double d = 0.0;
                for(std::size_t l = 0; l < realCount; ++l)
                    d = std::max(d, std::abs(outReal3.data()[l] - static_cast<double>(realCount) * flat[l]));
                rep.a3BackwardRel = d / (static_cast<double>(realCount) * *std::max_element(flat.begin(), flat.end()));
            }
        }

        // ------------------------------------------------ Stage A1: the same
        // plan on a 1-D (contiguous) buffer.
        if(!gSkipA1)
        {
            auto rhoReal1 = alpaka::onHost::allocUnified<double>(device, std::size_t{realCount});
            auto specReal1 = alpaka::onHost::allocUnified<Complex>(device, std::size_t{specCount});
            auto kernelSpec1 = alpaka::onHost::allocUnified<Complex>(device, std::size_t{specCount});
            auto outReal1 = alpaka::onHost::allocUnified<double>(device, std::size_t{realCount});
            for(std::size_t l = 0; l < realCount; ++l)
                rhoReal1.data()[l] = flat[l];
            alpaka::fft::onHost::executeForward(queue, r2cPlan, rhoReal1, specReal1);
            alpaka::onHost::wait(queue);
            {
                double d = 0.0;
                double r = 0.0;
                maxAbsDelta(specReal1.data(), reference.data(), specCount, &d, &r);
                rep.a1SpectrumRel = (r > 0.0) ? d / r : d;
            }
            for(std::size_t l = 0; l < specCount; ++l)
                kernelSpec1.data()[l] = reference[l];
            alpaka::fft::onHost::executeBackward(queue, c2rPlan, kernelSpec1, outReal1);
            alpaka::onHost::wait(queue);
            {
                double d = 0.0;
                for(std::size_t l = 0; l < realCount; ++l)
                    d = std::max(d, std::abs(outReal1.data()[l] - static_cast<double>(realCount) * flat[l]));
                rep.a1BackwardRel = d / (static_cast<double>(realCount) * *std::max_element(flat.begin(), flat.end()));
            }
        }

        // ------------------------------------------------ Stage B.
        std::array<std::int64_t, 3> origin{};
        for(int a = 0; a < 3; ++a)
            origin[a] = -static_cast<std::int64_t>((n - 1u) / 2u);

        std::vector<double> kernel;
        buildKernel(kernel, M);

        double const scale = 1.0 / static_cast<double>(realCount);

        auto compare = [&](std::vector<double> const& field, std::vector<double> const& ref, bool recordCloud, double* rel) {
            double d = 0.0;
            double r = 0.0;
            std::uint32_t wi = 0, wj = 0, wk = 0;
            for(std::uint32_t k = 0; k < n; ++k)
                for(std::uint32_t j = 0; j < n; ++j)
                    for(std::uint32_t i = 0; i < n; ++i)
                    {
                        std::size_t const l
                            = static_cast<std::size_t>(i)
                                + static_cast<std::size_t>(n) * (static_cast<std::size_t>(j) + static_cast<std::size_t>(n) * static_cast<std::size_t>(k));
                        double const delta = std::abs(field[l] - ref[l]);
                        r = std::max(r, std::abs(ref[l]));
                        if(delta > d)
                        {
                            d = delta;
                            wi = i;
                            wj = j;
                            wk = k;
                        }
                    }
            if(recordCloud)
            {
                rep.worstNode[0] = wi;
                rep.worstNode[1] = wj;
                rep.worstNode[2] = wk;
            }
            *rel = (r > 0.0) ? d / r : d;
        };

        // Stage B on an explicit buffer set.  The buffer lifetime mirrors the
        // production one (src/Compute.cpp:809-813, 1223-1328): allocated once
        // per run, cleared with alpaka's memset and scatter-deposited per
        // velocity bin, reused by every FFT in between.
        auto runStageB = [&](auto& rhoB, auto& specB, auto& kernB, auto& kernSpecB, auto& outB) {
            auto runPipeline = [&](std::vector<std::array<double, 3>> const& pos, std::vector<double> const& charge) {
                alpaka::onHost::memset(queue, rhoB, std::uint8_t{0});
                alpaka::onHost::wait(queue);
                auto const dep = depositNodes(pos, charge, origin, n);
                for(std::size_t q = 0; q < dep.node.size(); ++q)
                {
                    auto const& nd = dep.node[q];
                    std::size_t const idx
                        = static_cast<std::size_t>(nd[0])
                            + static_cast<std::size_t>(M)
                                  * (static_cast<std::size_t>(nd[1]) + static_cast<std::size_t>(M) * static_cast<std::size_t>(nd[2]));
                    rhoB.data()[idx] += dep.weight[q];
                }
                alpaka::onHost::memset(queue, kernB, std::uint8_t{0});
                alpaka::onHost::wait(queue);
                for(std::size_t l = 0; l < realCount; ++l)
                    kernB.data()[l] = kernel[l];
                alpaka::fft::onHost::executeForward(queue, r2cPlan, rhoB, specB);
                alpaka::onHost::wait(queue);
                alpaka::fft::onHost::executeForward(queue, r2cPlan, kernB, kernSpecB);
                alpaka::onHost::wait(queue);
                // MultiplySpectrumKernel, flat over the packed half-spectrum.
                for(std::size_t l = 0; l < specCount; ++l)
                {
                    double const ar = kernSpecB.data()[l].real();
                    double const ai = kernSpecB.data()[l].imag();
                    double const br = specB.data()[l].real();
                    double const bi = specB.data()[l].imag();
                    kernSpecB.data()[l] = Complex{ar * br - ai * bi, ar * bi + ai * br};
                }
                alpaka::fft::onHost::executeBackward(queue, c2rPlan, kernSpecB, outB);
                alpaka::onHost::wait(queue);
                std::vector<double> field(static_cast<std::size_t>(n) * n * n);
                for(std::uint32_t k = 0; k < n; ++k)
                    for(std::uint32_t j = 0; j < n; ++j)
                        for(std::uint32_t i = 0; i < n; ++i)
                        {
                            std::size_t const in
                                = static_cast<std::size_t>(i)
                                    + static_cast<std::size_t>(M)
                                          * (static_cast<std::size_t>(j)
                                             + static_cast<std::size_t>(M) * static_cast<std::size_t>(k));
                            field[static_cast<std::size_t>(i)
                                    + static_cast<std::size_t>(n)
                                          * (static_cast<std::size_t>(j) + static_cast<std::size_t>(n) * static_cast<std::size_t>(k))]
                                = outB.data()[in] * scale;
                        }
                return std::pair{field, dep};
            };

            if(gBins >= 2)
            {
                // Multi-bin lifecycle: gBins different clouds, each run
                // through one full memset+deposit+FFT pass, fields summed as
                // in the production bin loop.  Reference: the sum of the
                // per-bin direct free-space (and circular) sums.
                std::vector<std::array<double, 3>> const base{{0.0, 0.0, 0.0}, {1.4, 0.6, -0.8}, {-0.7, 1.2, 0.9}};
                std::vector<double> const charge{1.0, 1.0, 1.0};
                std::size_t const nBox = static_cast<std::size_t>(n) * n * n;
                std::vector<double> acc(nBox, 0.0);
                std::vector<double> refFree(nBox, 0.0);
                std::vector<double> refCirc(nBox, 0.0);
                for(int b = 0; b < gBins; ++b)
                {
                    double const shift = 0.2 * static_cast<double>(b);
                    std::vector<std::array<double, 3>> pos = base;
                    for(auto& p : pos)
                    {
                        p[0] += shift;
                        p[1] -= 0.75 * shift;
                        p[2] += 0.5 * shift;
                    }
                    auto const [field, dep] = runPipeline(pos, charge);
                    for(std::size_t l = 0; l < nBox; ++l)
                        acc[l] += field[l];
                    std::vector<double> rf;
                    std::vector<double> rc;
                    directSum(rf, dep, origin, n, M, false);
                    directSum(rc, dep, origin, n, M, true);
                    for(std::size_t l = 0; l < nBox; ++l)
                    {
                        refFree[l] += rf[l];
                        refCirc[l] += rc[l];
                    }
                }
                compare(acc, refFree, true, &rep.bCloudFreeRel);
                compare(acc, refCirc, false, &rep.bCloudCircRel);
                rep.bMaxRef = 0.0;
                for(auto const v : refFree)
                    rep.bMaxRef = std::max(rep.bMaxRef, std::abs(v));
                rep.fftCloudValue = acc[static_cast<std::size_t>(rep.worstNode[0])
                    + static_cast<std::size_t>(n) * (static_cast<std::size_t>(rep.worstNode[1])
                        + static_cast<std::size_t>(n) * static_cast<std::size_t>(rep.worstNode[2]))];
                rep.refCloudValue = refFree[static_cast<std::size_t>(rep.worstNode[0])
                    + static_cast<std::size_t>(n) * (static_cast<std::size_t>(rep.worstNode[1])
                        + static_cast<std::size_t>(n) * static_cast<std::size_t>(rep.worstNode[2]))];
                return;
            }

            // Single particle at the box centre (symmetric: image background
            // is at the round-off level for every M).
            {
                std::vector<std::array<double, 3>> const pos{{0.0, 0.0, 0.0}};
                std::vector<double> const charge{1.0};
                auto const [field, dep] = runPipeline(pos, charge);
                std::vector<double> refFree;
                std::vector<double> refCirc;
                directSum(refFree, dep, origin, n, M, false);
                directSum(refCirc, dep, origin, n, M, true);
                compare(field, refFree, false, &rep.bPointFreeRel);
                compare(field, refCirc, false, &rep.bPointCircRel);
            }
            // Three-particle cloud.
            {
                std::vector<std::array<double, 3>> const pos{{0.0, 0.0, 0.0}, {1.4, 0.6, -0.8}, {-0.7, 1.2, 0.9}};
                std::vector<double> const charge{1.0, 1.0, 1.0};
                auto const [field, dep] = runPipeline(pos, charge);
                std::vector<double> refFree;
                std::vector<double> refCirc;
                directSum(refFree, dep, origin, n, M, false);
                directSum(refCirc, dep, origin, n, M, true);
                compare(field, refFree, true, &rep.bCloudFreeRel);
                compare(field, refCirc, false, &rep.bCloudCircRel);
                rep.bMaxRef = 0.0;
                for(auto const v : refFree)
                    rep.bMaxRef = std::max(rep.bMaxRef, std::abs(v));
                rep.fftCloudValue = field[static_cast<std::size_t>(rep.worstNode[0])
                    + static_cast<std::size_t>(n) * (static_cast<std::size_t>(rep.worstNode[1])
                        + static_cast<std::size_t>(n) * static_cast<std::size_t>(rep.worstNode[2]))];
                rep.refCloudValue = refFree[static_cast<std::size_t>(rep.worstNode[0])
                    + static_cast<std::size_t>(n) * (static_cast<std::size_t>(rep.worstNode[1])
                        + static_cast<std::size_t>(n) * static_cast<std::size_t>(rep.worstNode[2]))];
            }
        };

        if(gBuf1D)
        {
            // The fix under test: flat 1-D allocation, no row padding.
            auto rhoB = alpaka::onHost::allocUnified<double>(device, std::size_t{realCount});
            auto specB = alpaka::onHost::allocUnified<Complex>(device, std::size_t{specCount});
            auto kernB = alpaka::onHost::allocUnified<double>(device, std::size_t{realCount});
            auto kernSpecB = alpaka::onHost::allocUnified<Complex>(device, std::size_t{specCount});
            auto outB = alpaka::onHost::allocUnified<double>(device, std::size_t{realCount});
            runStageB(rhoB, specB, kernB, kernSpecB, outB);
        }
        else
        {
            runStageB(rhoReal3, specReal3, kernelReal3, kernelSpec3, outReal3);
        }
        return rep;
    }

    //! --diag M: re-run the stage-B pipeline for one size with the FFTW
    //! reference as a host oracle at every block, to localise which step is
    //! size dependent:
    //!   0 flat write -> flat read-back identity on the 3-D buffers
    //!   1 vendor R2C(rho) vs FFTW(rho)
    //!   2 vendor R2C(kernel) vs FFTW(kernel)
    //!   3 pointwise product vs product of the FFTW spectra
    //!   4 vendor C2R(product) vs FFTW C2R(product)
    //!   5 n-box read-out vs the direct free-space sum
    template<typename T_Device, typename T_Queue>
    int diagSize(std::uint32_t M, T_Device& device, T_Queue& queue)
    {
        std::uint32_t const n = M / 2u;
        std::uint32_t const CP = M / 2u + 1u;
        std::size_t const realCount = static_cast<std::size_t>(M) * M * M;
        std::size_t const specCount = static_cast<std::size_t>(M) * M * CP;
        Extents3 const realExt{M, M, M};
        Extents3 const specExt{M, M, CP};

        auto rhoReal3 = alpaka::onHost::allocUnified<double>(device, realExt);
        auto specReal3 = alpaka::onHost::allocUnified<Complex>(device, specExt);
        auto kernelReal3 = alpaka::onHost::allocUnified<double>(device, realExt);
        auto kernelSpec3 = alpaka::onHost::allocUnified<Complex>(device, specExt);
        auto outReal3 = alpaka::onHost::allocUnified<double>(device, realExt);

        auto const pitches = rhoReal3.getPitches();
        std::printf(
            "M=%u n=%u  pitches(bytes) X=%zu Y=%zu Z=%zu  contiguous=%s\n",
            M,
            n,
            static_cast<std::size_t>(pitches[2]),
            static_cast<std::size_t>(pitches[1]),
            static_cast<std::size_t>(pitches[0]),
            (pitches[2] == static_cast<std::size_t>(sizeof(double))
                && pitches[1] == static_cast<std::size_t>(M) * sizeof(double)
                && pitches[0] == static_cast<std::size_t>(M) * M * sizeof(double))
                ? "yes"
                : "NO");

        auto r2cPlan = alpaka::fft::onHost::PlanBuilder<double, Extents3>{realExt}.r2c().outOfPlace().build(device);
        auto c2rPlan = alpaka::fft::onHost::PlanBuilder<double, Extents3>{realExt}.c2r().outOfPlace().build(device);

        // Memset probe: the runSize flow relies on alpaka's memset to clear a
        // pattern fill before the pipeline; verify it actually zeroes.
        for(std::size_t l = 0; l < realCount; ++l)
            rhoReal3.data()[l] = 1.0;
        alpaka::onHost::memset(queue, rhoReal3, std::uint8_t{0});
        alpaka::onHost::wait(queue);
        std::size_t nonzeroAfterMemset = 0;
        double maxAfterMemset = 0.0;
        for(std::size_t l = 0; l < realCount; ++l)
            if(rhoReal3.data()[l] != 0.0)
            {
                ++nonzeroAfterMemset;
                maxAfterMemset = std::max(maxAfterMemset, std::abs(rhoReal3.data()[l]));
            }
        std::printf("  memset probe: nonzero after memset = %zu / %zu (max %.3e)\n",
            nonzeroAfterMemset,
            realCount,
            maxAfterMemset);

        std::array<std::int64_t, 3> origin{};
        for(int a = 0; a < 3; ++a)
            origin[a] = -static_cast<std::int64_t>((n - 1u) / 2u);
        std::vector<std::array<double, 3>> const pos{{0.0, 0.0, 0.0}, {1.4, 0.6, -0.8}, {-0.7, 1.2, 0.9}};
        std::vector<double> const charge{1.0, 1.0, 1.0};
        auto const dep = depositNodes(pos, charge, origin, n);
        std::vector<double> rhoExp(realCount, 0.0);
        for(std::size_t q = 0; q < dep.node.size(); ++q)
        {
            auto const& nd = dep.node[q];
            std::size_t const idx
                = static_cast<std::size_t>(nd[0])
                    + static_cast<std::size_t>(M)
                          * (static_cast<std::size_t>(nd[1]) + static_cast<std::size_t>(M) * static_cast<std::size_t>(nd[2]));
            rhoExp[idx] += dep.weight[q];
        }
        std::vector<double> kernExp;
        buildKernel(kernExp, M);

        // Writes, exactly as runPipeline does them.
        alpaka::onHost::memset(queue, rhoReal3, std::uint8_t{0});
        alpaka::onHost::wait(queue);
        for(std::size_t l = 0; l < realCount; ++l)
            rhoReal3.data()[l] = rhoExp[l];
        alpaka::onHost::memset(queue, kernelReal3, std::uint8_t{0});
        alpaka::onHost::wait(queue);
        for(std::size_t l = 0; l < realCount; ++l)
            kernelReal3.data()[l] = kernExp[l];

        auto report = [](char const* tag, double d, double r) {
            std::printf("  %-30s max|delta|=%.6e  max|ref|=%.6e  rel=%.6e\n", tag, d, r, (r > 0.0) ? d / r : d);
        };

        // Block 0: flat write -> flat read-back identity.
        {
            double d = 0.0;
            double r = 0.0;
            for(std::size_t l = 0; l < realCount; ++l)
            {
                d = std::max(d, std::abs(rhoReal3.data()[l] - rhoExp[l]));
                r = std::max(r, std::abs(rhoExp[l]));
            }
            report("0 rho write/readback", d, r);
            d = 0.0;
            r = 0.0;
            for(std::size_t l = 0; l < realCount; ++l)
            {
                d = std::max(d, std::abs(kernelReal3.data()[l] - kernExp[l]));
                r = std::max(r, std::abs(kernExp[l]));
            }
            report("0 kernel write/readback", d, r);
        }

        std::vector<double> rhoBack(realCount);
        std::vector<double> kernBack(realCount);
        for(std::size_t l = 0; l < realCount; ++l)
        {
            rhoBack[l] = rhoReal3.data()[l];
            kernBack[l] = kernelReal3.data()[l];
        }
        std::vector<Complex> const rhoRef = fftwReferenceR2C(rhoBack, M);
        std::vector<Complex> const kernRef = fftwReferenceR2C(kernBack, M);

        // Block 1/2: vendor R2C vs FFTW of the read-back flat data.
        alpaka::fft::onHost::executeForward(queue, r2cPlan, rhoReal3, specReal3);
        alpaka::onHost::wait(queue);
        {
            double d = 0.0;
            double r = 0.0;
            maxAbsDelta(specReal3.data(), rhoRef.data(), specCount, &d, &r);
            report("1 R2C(rho) vs FFTW", d, r);
        }
        alpaka::fft::onHost::executeForward(queue, r2cPlan, kernelReal3, kernelSpec3);
        alpaka::onHost::wait(queue);
        {
            double d = 0.0;
            double r = 0.0;
            maxAbsDelta(kernelSpec3.data(), kernRef.data(), specCount, &d, &r);
            report("2 R2C(kernel) vs FFTW", d, r);
        }

        // Block 3: pointwise product.
        std::vector<Complex> prodRef(specCount);
        for(std::size_t l = 0; l < specCount; ++l)
        {
            double const ar = kernRef[l].real();
            double const ai = kernRef[l].imag();
            double const br = rhoRef[l].real();
            double const bi = rhoRef[l].imag();
            prodRef[l] = Complex{ar * br - ai * bi, ar * bi + ai * br};
        }
        for(std::size_t l = 0; l < specCount; ++l)
        {
            double const ar = kernelSpec3.data()[l].real();
            double const ai = kernelSpec3.data()[l].imag();
            double const br = specReal3.data()[l].real();
            double const bi = specReal3.data()[l].imag();
            kernelSpec3.data()[l] = Complex{ar * br - ai * bi, ar * bi + ai * br};
        }
        {
            double d = 0.0;
            double r = 0.0;
            maxAbsDelta(kernelSpec3.data(), prodRef.data(), specCount, &d, &r);
            report("3 product vs FFTW product", d, r);
        }

        // Block 4: vendor C2R vs FFTW C2R of the reference product.
        alpaka::fft::onHost::executeBackward(queue, c2rPlan, kernelSpec3, outReal3);
        alpaka::onHost::wait(queue);
        std::vector<double> const outRef = fftwReferenceC2R(prodRef, M);
        {
            double d = 0.0;
            double r = 0.0;
            for(std::size_t l = 0; l < realCount; ++l)
            {
                d = std::max(d, std::abs(outReal3.data()[l] - outRef[l]));
                r = std::max(r, std::abs(outRef[l]));
            }
            report("4 C2R(product) vs FFTW", d, r);
        }

        // Block 5: n-box read-out vs the direct free-space sum.
        std::vector<double> refFree;
        directSum(refFree, dep, origin, n, M, false);
        double const scale = 1.0 / static_cast<double>(realCount);
        {
            double d = 0.0;
            double r = 0.0;
            for(std::uint32_t k = 0; k < n; ++k)
                for(std::uint32_t j = 0; j < n; ++j)
                    for(std::uint32_t i = 0; i < n; ++i)
                    {
                        std::size_t const in
                            = static_cast<std::size_t>(i)
                                + static_cast<std::size_t>(M)
                                      * (static_cast<std::size_t>(j) + static_cast<std::size_t>(M) * static_cast<std::size_t>(k));
                        double const got = outReal3.data()[in] * scale;
                        double const want
                            = refFree[static_cast<std::size_t>(i)
                                + static_cast<std::size_t>(n) * (static_cast<std::size_t>(j)
                                    + static_cast<std::size_t>(n) * static_cast<std::size_t>(k))];
                        d = std::max(d, std::abs(got - want));
                        r = std::max(r, std::abs(want));
                    }
            report("5 n-box vs free-space sum", d, r);
        }
        return 0;
    }
} // namespace

int main(int argc, char** argv)
{
    bool diag = false;
    std::string sizeArg;
    for(int a = 1; a < argc; ++a)
    {
        std::string const s(argv[a]);
        if(s == "--diag")
            diag = true;
        else if(s == "--noA3")
            gSkipA3 = true;
        else if(s == "--noA1")
            gSkipA1 = true;
        else if(s == "--bins" && a + 1 < argc)
        {
            ++a;
            gBins = std::max(1, std::stoi(argv[a]));
        }
        else if(s == "--buf" && a + 1 < argc)
        {
            ++a;
            if(std::string(argv[a]) == "1d")
                gBuf1D = true;
            else if(std::string(argv[a]) == "3d")
                gBuf1D = false;
            else
                throw std::invalid_argument("--buf must be 1d or 3d");
        }
        else
            sizeArg = s;
    }
    char const* const kDefaultSizes = "16,20,24,27,32,36,40,45,48,54,60,64,72,75,80,96,100,120,128";
    std::vector<std::uint32_t> const sizes
        = diag ? std::vector<std::uint32_t>{} : parseSizes(sizeArg.empty() ? kDefaultSizes : sizeArg.c_str());

    auto selector = alpaka::onHost::makeDeviceSelector(
#if defined(PICBUNCH_BACKEND_CUDA)
        alpaka::api::cuda,
        alpaka::deviceKind::nvidiaGpu
#elif defined(PICBUNCH_BACKEND_HIP)
        alpaka::api::hip,
        alpaka::deviceKind::amdGpu
#else
        alpaka::api::host,
        alpaka::deviceKind::cpu
#endif
    );
    if(!selector.isAvailable())
    {
        std::fprintf(stderr, "fft-size-check: no device for the compiled backend\n");
        return 2;
    }
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();

    if(diag)
    {
        if(sizeArg.empty())
        {
            std::fprintf(stderr, "fft-size-check: --diag needs a size, e.g. --diag 24\n");
            return 2;
        }
        return diagSize(static_cast<std::uint32_t>(std::stoul(sizeArg)), device, queue);
    }

    std::printf(
        "M   n   contiguous  pitchX  pitchY  pitchZ  |  A3 spec   A3 back   |  A1 spec   A1 back   |  B pt free B pt circ  B cloud free B cloud circ\n");
    bool allGood = true;
    for(auto const M : sizes)
    {
        SizeReport const rep = runSize(M, device, queue);
        std::printf(
            "%3u %3u  %s  %7zu %7zu %7zu  | %9.2e %9.2e  | %9.2e %9.2e  | %8.2e %8.2e  %9.2e %9.2e\n",
            rep.M,
            rep.n,
            rep.contiguous ? "yes" : "NO ",
            rep.pitchX,
            rep.pitchY,
            rep.pitchZ,
            rep.a3SpectrumRel,
            rep.a3BackwardRel,
            rep.a1SpectrumRel,
            rep.a1BackwardRel,
            rep.bPointFreeRel,
            rep.bPointCircRel,
            rep.bCloudFreeRel,
            rep.bCloudCircRel);
        std::printf(
            "      worst node (cloud, free-space ref): (%u,%u,%u)  fft %.6e  ref %.6e  max|ref| %.3e\n",
            rep.worstNode[0],
            rep.worstNode[1],
            rep.worstNode[2],
            rep.fftCloudValue,
            rep.refCloudValue,
            rep.bMaxRef);
        allGood = allGood
            && rep.a3SpectrumRel < kRoundOffFloor && rep.a3BackwardRel < kRoundOffFloor
            && rep.a1SpectrumRel < kRoundOffFloor && rep.a1BackwardRel < kRoundOffFloor
            && rep.bPointFreeRel < kRoundOffFloor && rep.bPointCircRel < kRoundOffFloor
            && rep.bCloudFreeRel < kRoundOffFloor && rep.bCloudCircRel < kRoundOffFloor;
    }
    std::printf(allGood ? "ALL SIZES WITHIN ROUND-OFF FLOOR\n" : "SIZE DEPENDENT FAILURE PRESENT\n");
    return allGood ? 0 : 1;
}
