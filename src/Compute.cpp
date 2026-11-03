// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Compute core of picBunch: TSC deposit, TSC-convolved moving point kernel,
// free-space convolution through the alpakaVendor FFT, Yee read-out.
//
// All precision-critical stages are double.  The FFT is unnormalized, so the
// caller divides by M^3; the mesh density is charge-per-node, so the physical
// field additionally carries h^3 (h = nodes per cell).  The delivered reference
// implementation forgot h^3 because it only ran h = 1; here scale = h^3 / M^3.

#include "picBunch/Compute.hpp"

#include "picBunch/Deposit.hpp"
#include "picBunch/Kernel.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/fft.hpp>
#include <alpaka/math/Complex.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace picBunch
{
    namespace
    {
        using Extents3 = alpaka::Vec<std::uint32_t, 3u>;
        using Complex = alpaka::math::Complex<double>;

        template<typename T_Vec>
        T_Vec ceilDivVec(T_Vec const& a, T_Vec const& b)
        {
            T_Vec r{};
            for(std::uint32_t i = 0; i < T_Vec::dim(); ++i)
                r[i] = static_cast<decltype(a[0])>((a[i] + b[i] - 1u) / b[i]);
            return r;
        }

        //! TSC assignment of weight*charge onto the n^3 sub-box of the M^3 padded grid.
        struct TscDepositKernel
        {
            template<typename TAcc>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                double const* px,
                double const* py,
                double const* pz,
                double const* pq,
                std::uint32_t numParticles,
                double* rho,
                std::uint32_t Mx,
                std::uint32_t My,
                std::uint32_t nx,
                std::uint32_t ny,
                std::uint32_t nz,
                double ox,
                double oy,
                double oz,
                double sx,
                double sy,
                double sz,
                double refinement) const
            {
                double const support = 1.5 / refinement;
                for(auto [p] : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{numParticles}))
                {
                    double const pos[3] = {px[p], py[p], pz[p]};
                    double const org[3] = {ox, oy, oz};
                    double const sp[3] = {sx, sy, sz};
                    double t[3];
                    long lo[3];
                    long hi[3];
                    for(int a = 0; a < 3; ++a)
                    {
                        t[a] = (pos[a] - org[a]) / sp[a];
                        lo[a] = static_cast<long>(::ceil(t[a] - support));
                        hi[a] = static_cast<long>(::floor(t[a] + support));
                    }
                    double const q = pq[p];
                    for(long ia = lo[0]; ia <= hi[0]; ++ia)
                    {
                        if(ia < 0 || ia >= static_cast<long>(nx))
                            continue;
                        double const wa = tscShape(refinement * (static_cast<double>(ia) - t[0]));
                        if(wa == 0.0)
                            continue;
                        for(long ib = lo[1]; ib <= hi[1]; ++ib)
                        {
                            if(ib < 0 || ib >= static_cast<long>(ny))
                                continue;
                            double const wb = wa * tscShape(refinement * (static_cast<double>(ib) - t[1]));
                            if(wb == 0.0)
                                continue;
                            for(long ic = lo[2]; ic <= hi[2]; ++ic)
                            {
                                if(ic < 0 || ic >= static_cast<long>(nz))
                                    continue;
                                double const wc = wb * tscShape(refinement * (static_cast<double>(ic) - t[2]));
                                if(wc == 0.0)
                                    continue;
                                auto const idx = static_cast<std::size_t>(ia)
                                    + static_cast<std::size_t>(Mx)
                                        * (static_cast<std::size_t>(ib)
                                           + static_cast<std::size_t>(My) * static_cast<std::size_t>(ic));
                                alpaka::onAcc::atomicAdd(acc, &rho[idx], wc * q);
                            }
                        }
                    }
                }
            }
        };

        //! One component of the TSC-convolved moving point kernel on the padded box.
        struct KernelBuildKernel
        {
            template<typename TAcc>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                double* kernel,
                std::uint32_t Mx,
                std::uint32_t My,
                std::uint32_t Mz,
                int component,
                double dx,
                double dy,
                double dz,
                double cellx,
                double celly,
                double cellz,
                double refinement,
                double ux,
                double uy,
                double uz) const
            {
                for(auto const& idx : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{Extents3{Mx, My, Mz}}))
                {
                    auto const i = static_cast<std::uint32_t>(idx[0]);
                    auto const j = static_cast<std::uint32_t>(idx[1]);
                    auto const k = static_cast<std::uint32_t>(idx[2]);
                    int const di = (i < Mx / 2u) ? static_cast<int>(i) : static_cast<int>(i) - static_cast<int>(Mx);
                    int const dj = (j < My / 2u) ? static_cast<int>(j) : static_cast<int>(j) - static_cast<int>(My);
                    int const dk = (k < Mz / 2u) ? static_cast<int>(k) : static_cast<int>(k) - static_cast<int>(Mz);
                    // R = (d*h + yee_delta) * cell
                    double const Rx = (static_cast<double>(di) * refinement + dx) * cellx;
                    double const Ry = (static_cast<double>(dj) * refinement + dy) * celly;
                    double const Rz = (static_cast<double>(dk) * refinement + dz) * cellz;
                    double field[6];
                    pointFieldNormalized(Rx, Ry, Rz, ux, uy, uz, field);
                    double value = field[component];
                    if(!::isfinite(value))
                        value = 0.0;
                    auto const flat = static_cast<std::size_t>(i)
                        + static_cast<std::size_t>(Mx)
                            * (static_cast<std::size_t>(j)
                               + static_cast<std::size_t>(My) * static_cast<std::size_t>(k));
                    kernel[flat] = value;
                }
            }
        };

        //! Multiply two packed half-spectra.
        //!
        //! The product of two half-spectra of real arrays is the half-spectrum of a real
        //! array, so the packed planes are already consistent for C2R.  Do not zero the
        //! imaginary part of the last-axis DC/Nyquist planes: in 3-D R2C those planes are
        //! a complex 2-D transform, not a real 1-D transform, and zeroing them injects a
        //! constant field error that grows with the kernel's dynamic range.
        struct MultiplySpectrumKernel
        {
            template<typename TAcc>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                double* a,
                double const* b,
                std::uint32_t count) const
            {
                for(auto [l] : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{count}))
                {
                    std::size_t const off = 2u * static_cast<std::size_t>(l);
                    double const ar = a[off];
                    double const ai = a[off + 1u];
                    double const br = b[off];
                    double const bi = b[off + 1u];
                    double re = ar * br - ai * bi;
                    double im = ar * bi + ai * br;
                    a[off] = re;
                    a[off + 1u] = im;
                }
            }
        };

        using Clock = std::chrono::steady_clock;

        double elapsedMs(Clock::time_point const& start)
        {
            return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        }
    } // namespace

    Grid makeGrid(Bunch const& bunch, ComputeOptions const& options)
    {
        if(bunch.size() == 0)
            throw std::runtime_error("makeGrid: empty bunch");
        if(options.tolerance <= 0.0)
            throw std::invalid_argument("makeGrid: tolerance must be positive");
        if(options.marginCells < 0.0)
            throw std::invalid_argument("makeGrid: marginCells must not be negative");
        for(double c : options.cellSize)
            if(!(c > 0.0) || !std::isfinite(c))
                throw std::invalid_argument("makeGrid: every cell size must be finite and positive");

        auto const b = bunch.bounds();
        Grid grid;
        // v0.1 tolerance heuristic: tighten the padding for a tight accuracy target.
        unsigned padding = std::max(1u, options.fftPadding);
        if(options.tolerance < 1e-6)
            padding = std::max(padding, 3u);
        grid.fftPadding = padding;

        for(int a = 0; a < 3; ++a)
        {
            double const spacing = kRefinement * options.cellSize[a];
            double const margin = options.marginCells * options.cellSize[a];
            double const low = b[0][a] - margin;
            double const high = b[1][a] + margin;
            std::size_t nodes = static_cast<std::size_t>(std::ceil((high - low) / spacing)) + 1u;
            // even node count keeps M = padding*nodes even for the R2C transform
            if(nodes % 2u)
                ++nodes;
            nodes = std::max<std::size_t>(nodes, 8u);
            grid.n[a] = nodes;
            grid.spacing[a] = spacing;
            grid.origin[a] = low;
        }
        return grid;
    }

    Field computeField(Bunch const& bunch, ComputeOptions const& options, StageTimes* stageTimes)
    {
        // v0.1 is one velocity class: the kernel is exact for a single shared u.
        if(bunch.size() > 1)
        {
            double const scaleU = std::max({1.0, std::abs(bunch.ux[0]), std::abs(bunch.uy[0]), std::abs(bunch.uz[0])});
            for(std::size_t p = 1; p < bunch.size(); ++p)
                if(std::abs(bunch.ux[p] - bunch.ux[0]) > 1e-12 * scaleU
                   || std::abs(bunch.uy[p] - bunch.uy[0]) > 1e-12 * scaleU
                   || std::abs(bunch.uz[p] - bunch.uz[0]) > 1e-12 * scaleU)
                    throw std::runtime_error(
                        "computeField: v0.1 supports a single shared velocity; the bunch has a momentum spread");
        }

        Grid const grid = makeGrid(bunch, options);
        Field field;
        field.grid = grid;
        for(auto& component : field.data)
            component.assign(grid.size(), 0.0);

        StageTimes localTimes;
        StageTimes& times = stageTimes ? *stageTimes : localTimes;
        times = StageTimes{};

#if defined(PICBUNCH_BACKEND_CUDA)
        auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu);
#elif defined(PICBUNCH_BACKEND_HIP)
        auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::hip, alpaka::deviceKind::amdGpu);
#else
        auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::host, alpaka::deviceKind::cpu);
#endif
        if(!selector.isAvailable())
            throw std::runtime_error("computeField: no device available for the compiled backend");
        auto device = selector.makeDevice(0);
        auto queue = device.makeQueue();

        std::uint32_t const Mx = static_cast<std::uint32_t>(grid.n[0] * grid.fftPadding);
        std::uint32_t const My = static_cast<std::uint32_t>(grid.n[1] * grid.fftPadding);
        std::uint32_t const Mz = static_cast<std::uint32_t>(grid.n[2] * grid.fftPadding);
        if(Mx % 2u != 0u)
            throw std::runtime_error("computeField: padded x extent must be even for R2C");
        std::uint32_t const CP = Mx / 2u + 1u;
        // Flat layout is x fastest, then y, then z.  The plan's last dimension is x,
        // so the plan extents are (z, y, x) and the packed spectrum is (z, y, CP).
        Extents3 const realExt{Mz, My, Mx};
        Extents3 const specExt{Mz, My, CP};

        auto const cell = std::array<double, 3>{
            grid.spacing[0] / kRefinement,
            grid.spacing[1] / kRefinement,
            grid.spacing[2] / kRefinement};

        auto rhoReal = alpaka::onHost::allocUnified<double>(device, realExt);
        auto rhoSpec = alpaka::onHost::allocUnified<Complex>(device, specExt);
        auto kernelReal = alpaka::onHost::allocUnified<double>(device, realExt);
        auto kernelSpec = alpaka::onHost::allocUnified<Complex>(device, specExt);
        auto outReal = alpaka::onHost::allocUnified<double>(device, realExt);

        // ------------------------------------------------------------- deposit
        auto const depositStart = Clock::now();
        alpaka::onHost::memset(queue, rhoReal, std::uint8_t{0});
        std::size_t const particleCount = bunch.size();
        if(particleCount > 0)
        {
            if(particleCount > 0xffffffffu)
                throw std::runtime_error("computeField: particle count exceeds uint32 range");
            auto const numParticles = static_cast<std::uint32_t>(particleCount);
            auto posX = alpaka::onHost::allocUnified<double>(device, numParticles);
            auto posY = alpaka::onHost::allocUnified<double>(device, numParticles);
            auto posZ = alpaka::onHost::allocUnified<double>(device, numParticles);
            auto charge = alpaka::onHost::allocUnified<double>(device, numParticles);
            for(std::uint32_t p = 0; p < numParticles; ++p)
            {
                posX.data()[p] = bunch.x[p];
                posY.data()[p] = bunch.y[p];
                posZ.data()[p] = bunch.z[p];
                charge.data()[p] = bunch.weight[p] * bunch.charge[p];
            }
            std::uint32_t const frame = 64u;
            queue.enqueue(
                alpaka::onHost::FrameSpec{(numParticles + frame - 1u) / frame, frame},
                TscDepositKernel{},
                posX.data(),
                posY.data(),
                posZ.data(),
                charge.data(),
                numParticles,
                rhoReal.data(),
                Mx,
                My,
                static_cast<std::uint32_t>(grid.n[0]),
                static_cast<std::uint32_t>(grid.n[1]),
                static_cast<std::uint32_t>(grid.n[2]),
                grid.origin[0],
                grid.origin[1],
                grid.origin[2],
                grid.spacing[0],
                grid.spacing[1],
                grid.spacing[2],
                kRefinement);
        }
        alpaka::onHost::wait(queue);
        times.depositMs += elapsedMs(depositStart);

        // ---------------------------------------------------------------- FFT
        auto r2cPlan
            = alpaka::fft::onHost::PlanBuilder<double, Extents3>{realExt}.r2c().outOfPlace().build(device);
        auto c2rPlan
            = alpaka::fft::onHost::PlanBuilder<double, Extents3>{realExt}.c2r().outOfPlace().build(device);

        auto const fftStart = Clock::now();
        alpaka::fft::onHost::executeForward(queue, r2cPlan, rhoReal, rhoSpec);
        alpaka::onHost::wait(queue);
        times.fftMs += elapsedMs(fftStart);

        double const scale = (kRefinement * kRefinement * kRefinement)
            / (static_cast<double>(Mx) * static_cast<double>(My) * static_cast<double>(Mz));
        double const eFactor = kCoulombConstant;
        double const bFactor = kCoulombConstant / kSpeedOfLight;

        std::uint32_t const spectrumCount = Mz * My * CP;

        for(int c = 0; c < 6; ++c)
        {
            // ------------------------------------------------------------ kernel
            auto const kernelStart = Clock::now();
            queue.enqueue(
                alpaka::onHost::FrameSpec{
                    ceilDivVec(Extents3{Mx, My, Mz}, Extents3::fill(4u)),
                    Extents3::fill(4u)},
                KernelBuildKernel{},
                kernelReal.data(),
                Mx,
                My,
                Mz,
                c,
                kYeeOffset[c][0],
                kYeeOffset[c][1],
                kYeeOffset[c][2],
                cell[0],
                cell[1],
                cell[2],
                kRefinement,
                bunch.size() > 0 ? bunch.ux[0] : 0.0,
                bunch.size() > 0 ? bunch.uy[0] : 0.0,
                bunch.size() > 0 ? bunch.uz[0] : 0.0);
            alpaka::onHost::wait(queue);
            times.kernelMs += elapsedMs(kernelStart);

            // --------------------------------------------------------- spectrum
            auto const spectrumStart = Clock::now();
            alpaka::fft::onHost::executeForward(queue, r2cPlan, kernelReal, kernelSpec);
            alpaka::onHost::wait(queue);
            // product of two packed half-spectra; Complex<double> is two adjacent doubles
            static_assert(sizeof(Complex) == 2u * sizeof(double), "unexpected Complex layout");
            queue.enqueue(
                alpaka::onHost::FrameSpec{(spectrumCount + 63u) / 64u, 64u},
                MultiplySpectrumKernel{},
                reinterpret_cast<double*>(kernelSpec.data()),
                reinterpret_cast<double const*>(rhoSpec.data()),
                spectrumCount);
            alpaka::onHost::wait(queue);
            alpaka::fft::onHost::executeBackward(queue, c2rPlan, kernelSpec, outReal);
            alpaka::onHost::wait(queue);
            times.fftMs += elapsedMs(spectrumStart);

            // ---------------------------------------------------------- read-out
            auto const readoutStart = Clock::now();
            double const factor = scale * (c < 3 ? eFactor : bFactor);
            for(std::uint32_t k = 0; k < static_cast<std::uint32_t>(grid.n[2]); ++k)
                for(std::uint32_t j = 0; j < static_cast<std::uint32_t>(grid.n[1]); ++j)
                    for(std::uint32_t i = 0; i < static_cast<std::uint32_t>(grid.n[0]); ++i)
                    {
                        auto const in = static_cast<std::size_t>(i)
                            + static_cast<std::size_t>(Mx)
                                * (static_cast<std::size_t>(j)
                                   + static_cast<std::size_t>(My) * static_cast<std::size_t>(k));
                        auto const out = static_cast<std::size_t>(i)
                            + static_cast<std::size_t>(grid.n[0])
                                * (static_cast<std::size_t>(j)
                                   + static_cast<std::size_t>(grid.n[1]) * static_cast<std::size_t>(k));
                        field.data[c][out] = outReal.data()[in] * factor;
                    }
            times.readoutMs += elapsedMs(readoutStart);
        }

        return field;
    }

} // namespace picBunch
