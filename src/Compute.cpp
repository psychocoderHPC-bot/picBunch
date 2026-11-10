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
#include "picBunch/NearShell.hpp"
#include "picBunch/DistributedFft.hpp"
#include "picBunch/Readout.hpp"
#include "picBunch/detail/SpectrumNorm.hpp"
#include "picBunch/VelocityMoments.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/fft.hpp>
#if defined(ALPAKAV_DEP_ROCFFT) && ALPAKAV_DEP_ROCFFT
#    include <rocfft/rocfft.h>
#endif
#include <alpaka/math/Complex.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace picBunch
{
    namespace
    {
        using Extents3 = alpaka::Vec<std::uint32_t, 3u>;
        using Complex = alpaka::math::Complex<double>;

        //! Extents type stored in an alpaka MdSpan.  Extracting it through a
        //! class-template specialization avoids decltype of getExtents() in an
        //! unevaluated nested context, which nvcc 13.4 cannot parse.
        template<typename T>
        struct MdSpanExtents;

        template<typename T_Value, typename T_Extents, typename T_Pitches, typename T_Align>
        struct MdSpanExtents<alpaka::MdSpan<T_Value, T_Extents, T_Pitches, T_Align>>
        {
            using type = T_Extents;
        };

        template<typename T_Vec>
        T_Vec ceilDivVec(T_Vec const& a, T_Vec const& b)
        {
            T_Vec r{};
            for(std::uint32_t i = 0; i < T_Vec::dim(); ++i)
                r[i] = static_cast<decltype(a[0])>((a[i] + b[i] - 1u) / b[i]);
            return r;
        }

        //! Wrap a logical source index into the padded extent; the source halo lives there.
        ALPAKA_FN_ACC inline long wrapPaddedIndex(long i, std::uint32_t m)
        {
            long const mm = static_cast<long>(m);
            long r = i % mm;
            return r < 0 ? r + mm : r;
        }

        //! TSC assignment of weight*charge onto the padded grid with a source halo.
        template<Shape S>
        struct TscDepositKernel
        {
            template<typename TAcc>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                alpaka::concepts::IMdSpan auto const& px,
                alpaka::concepts::IMdSpan auto const& py,
                alpaka::concepts::IMdSpan auto const& pz,
                alpaka::concepts::IMdSpan auto const& pq,
                std::uint32_t numParticles,
                alpaka::concepts::IMdSpan auto rho,
                std::uint32_t zOffset,
                std::uint32_t zExtent,
                std::uint32_t Mx,
                std::uint32_t My,
                std::uint32_t Mz,
                alpaka::Vec<double, 3u> origin,
                alpaka::Vec<double, 3u> spacing) const
            {
                double const support = shapeSupport<S>();
                for(auto [p] : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{numParticles}))
                {
                    auto const t = (alpaka::Vec{px[p], py[p], pz[p]} - origin) / spacing;
                    alpaka::Vec<long, 3u> lo;
                    alpaka::Vec<long, 3u> hi;
                    for(int a = 0; a < 3; ++a)
                    {
                        lo[a] = static_cast<long>(alpaka::math::ceil(t[a] - support));
                        hi[a] = static_cast<long>(alpaka::math::floor(t[a] + support));
                    }
                    double const q = pq[p];
                    for(long ia = lo[0]; ia <= hi[0]; ++ia)
                    {
                        double const wa = shapeWeight<S>(static_cast<double>(ia) - t[0]);
                        if(wa == 0.0)
                            continue;
                        long const ma = wrapPaddedIndex(ia, Mx);
                        for(long ib = lo[1]; ib <= hi[1]; ++ib)
                        {
                            double const wb = wa * shapeWeight<S>(static_cast<double>(ib) - t[1]);
                            if(wb == 0.0)
                                continue;
                            long const mb = wrapPaddedIndex(ib, My);
                            for(long ic = lo[2]; ic <= hi[2]; ++ic)
                            {
                                double const wc = wb * shapeWeight<S>(static_cast<double>(ic) - t[2]);
                                if(wc == 0.0)
                                    continue;
                                long const mc = wrapPaddedIndex(ic, Mz);
                                if(mc < static_cast<long>(zOffset)
                                   || mc >= static_cast<long>(zOffset + zExtent))
                                    continue;
                                auto const idx = alpaka::Vec{
                                    static_cast<std::uint32_t>(mc - static_cast<long>(zOffset)),
                                    static_cast<std::uint32_t>(mb),
                                    static_cast<std::uint32_t>(ma)};
                                // rho is a double box in the replicated path and a
                                // Complex slab in the distributed path; the real
                                // part is always the first double of the element.
                                alpaka::onAcc::atomicAdd(
                                    acc,
                                    reinterpret_cast<double*>(&rho[idx]),
                                    wc * q);
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
                alpaka::concepts::IMdSpan auto kernel,
                std::uint32_t zOffset,
                std::uint32_t zExtent,
                std::uint32_t Mx,
                std::uint32_t My,
                std::uint32_t Mz,
                int component,
                alpaka::Vec<double, 3u> cell,
                alpaka::Vec<double, 3u> cellOrigin,
                double refinement,
                alpaka::Vec<double, 3u> uRep) const
            {
                for(auto const& idx : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{Extents3{Mx, My, zExtent}}))
                {
                    auto const i = static_cast<std::uint32_t>(idx[0]);
                    auto const j = static_cast<std::uint32_t>(idx[1]);
                    auto const kLocal = static_cast<std::uint32_t>(idx[2]);
                    auto const k = kLocal + zOffset;
                    // Centred index of a node on the (zero-padded) FFT box.  ``i <= (M-1)/2``
                    // is the parity-correct split: for even M it is identical to ``i < M/2``
                    // (bins 0..M/2-1 positive, M/2..M-1 negative, Nyquist bin at M/2); for odd
                    // M (no Nyquist plane) it is 0..(M-1)/2 positive and (M+1)/2..M-1 negative.
                    // A plain ``i < M/2`` mis-centres the kernel on an odd extent.
                    int const di = (i <= (Mx - 1u) / 2u) ? static_cast<int>(i) : static_cast<int>(i) - static_cast<int>(Mx);
                    int const dj = (j <= (My - 1u) / 2u) ? static_cast<int>(j) : static_cast<int>(j) - static_cast<int>(My);
                    int const dk = (k <= (Mz - 1u) / 2u) ? static_cast<int>(k) : static_cast<int>(k) - static_cast<int>(Mz);
                    // R = (d*h + yee_delta) * cell
                    auto const d = alpaka::Vec{
                        static_cast<double>(di),
                        static_cast<double>(dj),
                        static_cast<double>(dk)};
                    auto const R = (d * refinement + cell) * cellOrigin;
                    double field[6];
                    pointFieldNormalized(R[0], R[1], R[2], uRep[0], uRep[1], uRep[2], field);
                    double value = field[component];
                    if(!alpaka::math::isfinite(value))
                        value = 0.0;
                    auto const nodeIdx = alpaka::Vec{
                        static_cast<std::uint32_t>(kLocal),
                        static_cast<std::uint32_t>(j),
                        static_cast<std::uint32_t>(i)};
                    kernel[nodeIdx] = value;
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
                alpaka::concepts::IMdSpan auto a,
                alpaka::concepts::IMdSpan auto const& b,
                std::uint32_t count) const
            {
                for(auto [l] : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{count}))
                {
                    // Packed half-spectrum: re at [l,0], im at [l,1].
                    double const ar = a[alpaka::Vec{l, 0u}];
                    double const ai = a[alpaka::Vec{l, 1u}];
                    double const br = b[alpaka::Vec{l, 0u}];
                    double const bi = b[alpaka::Vec{l, 1u}];
                    double const re = ar * br - ai * bi;
                    double const im = ar * bi + ai * br;
                    a[alpaka::Vec{l, 0u}] = re;
                    a[alpaka::Vec{l, 1u}] = im;
                }
            }
        };

        //! TSC deposit of the velocity moment ``q_p (u_p - u0)^a`` onto the padded box.
        //! Same stencil as the charge deposit, but the per-particle coefficient is the
        //! signed charge times the velocity moment (power form, no ``1/a!``).
        template<Shape S>
        struct MomentDepositKernel
        {
            template<typename TAcc>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                alpaka::concepts::IMdSpan auto const& px,
                alpaka::concepts::IMdSpan auto const& py,
                alpaka::concepts::IMdSpan auto const& pz,
                alpaka::concepts::IMdSpan auto const& pux,
                alpaka::concepts::IMdSpan auto const& puy,
                alpaka::concepts::IMdSpan auto const& puz,
                alpaka::concepts::IMdSpan auto const& pq,
                std::uint32_t numParticles,
                alpaka::concepts::IMdSpan auto rho,
                std::uint32_t zOffset,
                std::uint32_t zExtent,
                std::uint32_t Mx,
                std::uint32_t My,
                std::uint32_t Mz,
                alpaka::Vec<double, 3u> origin,
                alpaka::Vec<double, 3u> spacing,
                alpaka::Vec<double, 3u> u0,
                int a1,
                int a2,
                int a3) const
            {
                double const support = shapeSupport<S>();
                auto const powSmall = [](double x, int e) {
                    double r = 1.0;
                    for(int i = 0; i < e; ++i)
                        r *= x;
                    return r;
                };
                for(auto [p] : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{numParticles}))
                {
                    auto const t = (alpaka::Vec{px[p], py[p], pz[p]} - origin) / spacing;
                    alpaka::Vec<long, 3u> lo;
                    alpaka::Vec<long, 3u> hi;
                    for(int a = 0; a < 3; ++a)
                    {
                        lo[a] = static_cast<long>(alpaka::math::ceil(t[a] - support));
                        hi[a] = static_cast<long>(alpaka::math::floor(t[a] + support));
                    }
                    double const coeff = pq[p] * powSmall(pux[p] - u0[0], a1)
                        * powSmall(puy[p] - u0[1], a2) * powSmall(puz[p] - u0[2], a3);
                    if(coeff == 0.0)
                        continue;
                    for(long ia = lo[0]; ia <= hi[0]; ++ia)
                    {
                        double const wa = shapeWeight<S>(static_cast<double>(ia) - t[0]);
                        if(wa == 0.0)
                            continue;
                        long const ma = wrapPaddedIndex(ia, Mx);
                        for(long ib = lo[1]; ib <= hi[1]; ++ib)
                        {
                            double const wb = wa * shapeWeight<S>(static_cast<double>(ib) - t[1]);
                            if(wb == 0.0)
                                continue;
                            long const mb = wrapPaddedIndex(ib, My);
                            for(long ic = lo[2]; ic <= hi[2]; ++ic)
                            {
                                double const wc = wb * shapeWeight<S>(static_cast<double>(ic) - t[2]);
                                if(wc == 0.0)
                                    continue;
                                long const mc = wrapPaddedIndex(ic, Mz);
                                if(mc < static_cast<long>(zOffset)
                                   || mc >= static_cast<long>(zOffset + zExtent))
                                    continue;
                                auto const idx = alpaka::Vec{
                                    static_cast<std::uint32_t>(mc - static_cast<long>(zOffset)),
                                    static_cast<std::uint32_t>(mb),
                                    static_cast<std::uint32_t>(ma)};
                                alpaka::onAcc::atomicAdd(acc, &rho[idx], wc * coeff);
                            }
                        }
                    }
                }
            }
        };

        //! Velocity-moment kernel for one output component: writes the full
        //! ``Order``-order Taylor of component ``component`` (all multi-indices,
        //! compact order) at the node's Yee-offset displacement ``R``.  The buffer is
        //! laid out ``[multiIndex][node]`` so each coefficient is a contiguous
        //! padded field ready for one R2C transform.
        template<std::size_t Order>
        struct MomentKernelBuildKernel
        {
            template<typename TAcc>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                alpaka::concepts::IMdSpan auto kernelPlanes,
                std::uint32_t zOffset,
                std::uint32_t zExtent,
                std::uint32_t Mx,
                std::uint32_t My,
                std::uint32_t Mz,
                int component,
                alpaka::Vec<double, 3u> cell,
                alpaka::Vec<double, 3u> cellOrigin,
                double refinement,
                alpaka::Vec<double, 3u> u0) const
            {
                using TP = detail::TaylorPoly<Order>;
                constexpr std::size_t T = TP::T;
                for(auto const& idx : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{Extents3{Mx, My, zExtent}}))
                {
                    auto const i = static_cast<std::uint32_t>(idx[0]);
                    auto const j = static_cast<std::uint32_t>(idx[1]);
                    auto const kLocal = static_cast<std::uint32_t>(idx[2]);
                    auto const k = kLocal + zOffset;
                    int const di = (i <= (Mx - 1u) / 2u) ? static_cast<int>(i)
                                                         : static_cast<int>(i) - static_cast<int>(Mx);
                    int const dj = (j <= (My - 1u) / 2u) ? static_cast<int>(j)
                                                         : static_cast<int>(j) - static_cast<int>(My);
                    int const dk = (k <= (Mz - 1u) / 2u) ? static_cast<int>(k)
                                                         : static_cast<int>(k) - static_cast<int>(Mz);
                    auto const d = alpaka::Vec{
                        static_cast<double>(di),
                        static_cast<double>(dj),
                        static_cast<double>(dk)};
                    auto const R = (d * refinement + cell) * cellOrigin;
                    double coeffs[6 * T];
                    detail::pointFieldTaylor<Order>(R[0], R[1], R[2], u0[0], u0[1], u0[2], coeffs);
                    double const* comp = coeffs + static_cast<std::size_t>(component) * T;
                    for(std::size_t a = 0; a < T; ++a)
                    {
                        double const v = comp[a];
                        auto const nodeIdx = alpaka::Vec{
                            static_cast<std::uint32_t>(a),
                            static_cast<std::uint32_t>(kLocal),
                            static_cast<std::uint32_t>(j),
                            static_cast<std::uint32_t>(i)};
                        kernelPlanes[nodeIdx] = alpaka::math::isfinite(v) ? v : 0.0;
                    }
                }
            }
        };

        //! Accumulate the complex product ``a * b`` into ``out``: ``out[i] += a[i] * b[i]``.
        //! Complex is two adjacent doubles; the half-spectrum packing is untouched.
        struct SpectrumAccumulateKernel
        {
            template<typename TAcc>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                alpaka::concepts::IMdSpan auto out,
                alpaka::concepts::IMdSpan auto a,
                alpaka::concepts::IMdSpan auto const& b,
                std::uint32_t count) const
            {
                for(auto [l] : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{count}))
                {
                    // Packed half-spectrum: re at [l,0], im at [l,1].
                    double const ar = a[alpaka::Vec{l, 0u}];
                    double const ai = a[alpaka::Vec{l, 1u}];
                    double const br = b[alpaka::Vec{l, 0u}];
                    double const bi = b[alpaka::Vec{l, 1u}];
                    double const orr = out[alpaka::Vec{l, 0u}];
                    double const ori = out[alpaka::Vec{l, 1u}];
                    out[alpaka::Vec{l, 0u}] = orr + ar * br - ai * bi;
                    out[alpaka::Vec{l, 1u}] = ori + ar * bi + ai * br;
                }
            }
        };

        //! Scalar ``acc + in * factor`` used by the read-out ``onHost::transform``.
        //! The operator is generic so alpaka can lower it onto SIMD packs (the
        //! scalar-functor wrapper is intentionally dropped).  It still evaluates
        //! ``acc + in * factor`` per lane; the read-out tests pin the resulting
        //! <= 1 ulp / <= 2e-15 difference against the double reference.
        struct ReadoutScaleAdd
        {
            double factor;

            ALPAKA_FN_ACC auto operator()(auto const& acc, auto const& in) const
            {
                return acc + in * factor;
            }
        };

        //! Gather one local read-out slab into a compact double buffer and scale
        //! it: ``dst[l] = Re(work[Vec{k,j,i}]) * factor`` with x-fastest ``l``.
        //! The complex layout needs the index mapping, so this stays a small kernel
        //! with ``makeIdxMap`` rather than an ``onHost::transform`` over Complex.
        //! Each element is independent, so the result is order-independent and
        //! bit-identical to the former in-place scale followed by a host read.
        struct SlabGatherScaleKernel
        {
            template<typename TAcc>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                alpaka::concepts::IMdSpan auto const& work,
                alpaka::concepts::IMdSpan auto dst,
                std::uint32_t nx,
                std::uint32_t ny,
                std::uint32_t nz,
                double factor) const
            {
                std::size_t const count
                    = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny)
                    * static_cast<std::size_t>(nz);
                for(auto [l] : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{count}))
                {
                    std::size_t const i = l % nx;
                    std::size_t const rem = l / nx;
                    std::size_t const j = rem % ny;
                    std::size_t const k = rem / ny;
                    auto const idx = alpaka::Vec{
                        static_cast<std::uint32_t>(k),
                        static_cast<std::uint32_t>(j),
                        static_cast<std::uint32_t>(i)};
                    // size_t linear index into 32-bit extents; count <= kMaxPaddedNodes (2^28).
                    dst[l] = work[idx].real() * factor;
                }
            }
        };

        //! One multi-index ``(a1,a2,a3)`` with its compact order-3 packed index and total degree.
        struct VelocityMomentMultiIndex
        {
            int a1;
            int a2;
            int a3;
            std::size_t idx; //!< compactIndex into the order-3 coefficient layout
            int tot;         //!< a1 + a2 + a3
        };

        //! Every multi-index with ``a1+a2+a3 <= order``, in order-3 packed order.
        std::vector<VelocityMomentMultiIndex> velocityMomentMultiIndices(std::size_t order)
        {
            std::vector<VelocityMomentMultiIndex> out;
            for(int a3 = 0; a3 <= static_cast<int>(order); ++a3)
                for(int a2 = 0; a2 <= static_cast<int>(order) - a3; ++a2)
                    for(int a1 = 0; a1 <= static_cast<int>(order) - a3 - a2; ++a1)
                        out.push_back(
                            {a1,
                             a2,
                             a3,
                             detail::TaylorPoly<3>::compactIndex(
                                 static_cast<std::size_t>(a1),
                                 static_cast<std::size_t>(a2),
                                 static_cast<std::size_t>(a3)),
                             a1 + a2 + a3});
            return out;
        }

        using Clock = std::chrono::steady_clock;

        double elapsedMs(Clock::time_point const& start)
        {
            return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        }

        /*! Measured relative error of the padding-2 result.
         *
         * The support of the free-space convolution is ``n`` output nodes; padding
         * 2 makes the FFT box ``2n`` wide, so every source-observer lag is shorter
         * than half the box and no periodic image can reach an observer.  The
         * remaining difference to padding 3 is FFT/roundoff only.  Measured on
         * rosi_v100 (CUDA, V100) as ``max|pad2-pad3| / max|field|`` for the toy
         * lattice used by the padding test: 8.37e-16; a 500-particle
         * ``cellSize = 1e-6 m`` cloud on an 18^3 grid gives 4.15e-16.  This value
         * is a conservative bound over both, rounded up by roughly three orders
         * of magnitude to cover geometries and backends that were not measured.
         */
        //! Largest absolute eigenvalue of a symmetric 3x3 matrix (row-major).
        double quadrupoleSpectralNorm(std::array<double, 9> const& a)
        {
            double const p1 = a[1] * a[1] + a[2] * a[2] + a[5] * a[5];
            if(p1 == 0.0)
                return std::max({std::abs(a[0]), std::abs(a[4]), std::abs(a[8])});
            double const q = (a[0] + a[4] + a[8]) / 3.0;
            double const p2 = (a[0] - q) * (a[0] - q) + (a[4] - q) * (a[4] - q)
                + (a[8] - q) * (a[8] - q) + 2.0 * p1;
            double const p = std::sqrt(p2 / 6.0);
            std::array<double, 9> b{};
            for(int i = 0; i < 9; ++i)
                b[i] = a[i] / p;
            b[0] -= q / p;
            b[4] -= q / p;
            b[8] -= q / p;
            double const det = b[0] * (b[4] * b[8] - b[5] * b[7])
                - b[1] * (b[3] * b[8] - b[5] * b[6])
                + b[2] * (b[3] * b[7] - b[4] * b[6]);
            double const r = std::max(-1.0, std::min(1.0, det / 2.0));
            double const phi = std::acos(r) / 3.0;
            double const e1 = q + 2.0 * p * std::cos(phi);
            double const e3 = q + 2.0 * p * std::cos(phi + 2.0943951023931953);
            double const e2 = 3.0 * q - e1 - e3;
            return std::max({std::abs(e1), std::abs(e2), std::abs(e3)});
        }

        constexpr double kPadding2ErrorBound = 1.0e-12;

        /*! Lowest tolerance the double-precision FFT pipeline can honour.
         *
         * Padding 3 is converged to FFT roundoff (measured 8.37e-16 relative),
         * but accumulation and transform roundoff in a single double run cannot
         * be guaranteed below this floor.  Requests below it are rejected instead
         * of being silently under-delivered.
         */
        constexpr double kMinTolerance = 1.0e-14;

        /*! Hard limits used to turn an infeasible input into a clear error.
         *
         * The v0.1 double conversion made a 1e-6 m cell look like 1e-12 m and
         * the node count ran into ``std::vector``'s max_size, surfacing as a
         * ``bad_alloc``-style message.  A grid beyond these bounds cannot be
         * allocated by the pipeline, so reject it before any allocation.
         */
        constexpr std::size_t kMaxNodesPerAxis = 1u << 20;
        constexpr std::size_t kMaxPaddedNodes = 1u << 28;

        /*! Tolerance -> FFT padding.  Padding 1 is never selected: it is a
         * diagnostic mode whose periodic-image artefact (measured 4.0e-2 for the
         * padding-test geometry) no tolerance can bound.
         */
        unsigned selectFftPadding(double tolerance)
        {
            if(!(tolerance > 0.0) || !std::isfinite(tolerance))
                throw std::invalid_argument("makeGrid: tolerance must be finite and positive");
            if(tolerance < kMinTolerance)
                throw std::runtime_error(
                    "makeGrid: tolerance " + std::to_string(tolerance)
                    + " is below the measured double-precision floor "
                    + std::to_string(kMinTolerance)
                    + "; raise --tolerance (the FFT padding cannot deliver a tighter target)");
            return (tolerance < kPadding2ErrorBound) ? 3u : 2u;
        }
    } // namespace

    std::array<double, 3> resolveCellSize(
        std::optional<std::array<double, 3>> const& explicitCellSize,
        std::optional<std::array<double, 3>> const& meshCellSize)
    {
        auto validate = [](std::array<double, 3> const& value, char const* source) {
            for(double c : value)
                if(!(c > 0.0) || !std::isfinite(c))
                    throw std::invalid_argument(
                        std::string("resolveCellSize: ") + source + " cell size must be finite and positive");
        };
        if(explicitCellSize)
        {
            validate(*explicitCellSize, "--cell-size");
            return *explicitCellSize;
        }
        if(meshCellSize)
        {
            validate(*meshCellSize, "mesh-derived");
            return *meshCellSize;
        }
        throw std::runtime_error(
            "no cell size available: the input has no usable mesh geometry and --cell-size was not given; "
            "pass --cell-size <dx>[,<dy>,<dz>] in metres");
    }

    std::optional<std::array<double, 3>> physicalCellSize(MeshGeometry const& geometry)
    {
        // readMeshGeometry returns physical SI spacing and offset.  Do not
        // multiply by geometry.gridUnitSI here: the reader already did it, and
        // the CLI did it again in v0.1 (physical 1e-6 m became 1e-12 m).
        for(double spacing : geometry.gridSpacing)
            if(!(spacing > 0.0) || !std::isfinite(spacing))
                return std::nullopt;
        return geometry.gridSpacing;
    }

    std::optional<std::array<double, 3>> physicalGridOrigin(MeshGeometry const& geometry)
    {
        // Same ownership rule as the spacing: readMeshGeometry already multiplied
        // gridGlobalOffset by gridUnitSI, so the CLI must not apply it again.
        for(double offset : geometry.gridGlobalOffset)
            if(!std::isfinite(offset))
                return std::nullopt;
        return geometry.gridGlobalOffset;
    }

    std::optional<std::array<std::size_t, 3>> physicalGridExtent(MeshGeometry const& geometry)
    {
        // readMeshGeometry already mapped the dataset extent from the stored axis
        // order to x,y,z.  A node count has no unit conversion; the reader owns
        // the gridUnitSI handling of spacing and offset.
        for(std::size_t nodes : geometry.shape)
            if(nodes == 0u)
                return std::nullopt;
        return geometry.shape;
    }

    FieldTailMargin deriveFieldTailMargin(
        Bunch const& bunch,
        double fieldPeak,
        double tolerance,
        std::array<double, 3> const& cellSize,
        double maxDistance)
    {
        if(bunch.size() == 0u)
            throw std::runtime_error("deriveFieldTailMargin: empty bunch");
        if(!(tolerance > 0.0) || !std::isfinite(tolerance))
            throw std::invalid_argument("deriveFieldTailMargin: tolerance must be finite and positive");
        if(!(fieldPeak >= 0.0) || !std::isfinite(fieldPeak))
            throw std::invalid_argument("deriveFieldTailMargin: field peak must be finite and non-negative");
        if(!(maxDistance > 0.0) || !std::isfinite(maxDistance))
            throw std::invalid_argument("deriveFieldTailMargin: maxDistance must be finite and positive");
        for(double h : cellSize)
            if(!(h > 0.0) || !std::isfinite(h))
                throw std::invalid_argument("deriveFieldTailMargin: cell sizes must be finite and positive");

        // Source moments about the bounding-box centre.  Using the box half
        // extents for the per-axis conversion is tighter than the circumscribed
        // radius and still bounds every point outside the expanded box.
        auto const bounds = bunch.bounds();
        std::array<double, 3> centre{};
        std::array<double, 3> halfExtent{};
        for(int a = 0; a < 3; ++a)
        {
            centre[a] = 0.5 * (bounds[0][a] + bounds[1][a]);
            halfExtent[a] = 0.5 * (bounds[1][a] - bounds[0][a]);
        }

        double totalCharge = 0.0;
        double absoluteCharge = 0.0;
        std::array<double, 3> dipole{0.0, 0.0, 0.0};
        std::array<double, 9> quadrupole{};
        double sourceRadius = 0.0;
        double maxGamma = 1.0;
        for(std::size_t p = 0; p < bunch.size(); ++p)
        {
            double const u2 = bunch.ux[p] * bunch.ux[p]
                + bunch.uy[p] * bunch.uy[p] + bunch.uz[p] * bunch.uz[p];
            maxGamma = std::max(maxGamma, std::sqrt(1.0 + u2));
            double const q = bunch.weight[p] * bunch.charge[p];
            totalCharge += q;
            absoluteCharge += std::abs(q);
            std::array<double, 3> const displacement{
                bunch.x[p] - centre[0],
                bunch.y[p] - centre[1],
                bunch.z[p] - centre[2]};
            for(int a = 0; a < 3; ++a)
                dipole[a] += q * displacement[a];
            double const r2 = displacement[0] * displacement[0]
                + displacement[1] * displacement[1]
                + displacement[2] * displacement[2];
            for(int i = 0; i < 3; ++i)
                for(int j = 0; j < 3; ++j)
                    quadrupole[i * 3 + j]
                        += q * (3.0 * displacement[i] * displacement[j]
                                - (i == j ? r2 : 0.0));
            sourceRadius = std::max(sourceRadius, std::sqrt(r2));
        }
        double const dipoleMagnitude = std::sqrt(
            dipole[0] * dipole[0] + dipole[1] * dipole[1] + dipole[2] * dipole[2]);
        double const quadrupoleNorm = quadrupoleSpectralNorm(quadrupole);

        FieldTailMargin result;
        result.totalCharge = totalCharge;
        result.absoluteCharge = absoluteCharge;
        result.dipoleMagnitude = dipoleMagnitude;
        result.dipoleMoment = dipole;
        result.quadrupoleMoment = quadrupole;
        result.quadrupoleNorm = quadrupoleNorm;
        result.sourceRadius = sourceRadius;
        result.maxGamma = maxGamma;
        result.fieldPeak = fieldPeak;
        result.tolerance = tolerance;
        result.distance = sourceRadius;
        result.bound = 0.0;

        double const target = tolerance * fieldPeak;
        if(absoluteCharge == 0.0 || target == 0.0)
            return result; // no source or no field: zero margin is exact

        auto const tailBound = [&](double d) {
            if(!(d > sourceRadius))
                return std::numeric_limits<double>::infinity();
            // Monopole, dipole and the actual quadrupole.  The angular maximum
            // of the quadrupole field is (k/2)(5|S| + 2|D n|)/d^4 with
            // |S|, |D n| <= ||D||_2, hence the factor 3.5.
            return maxGamma * kCoulombConstant
                * (std::abs(totalCharge) / (d * d) + 2.0 * dipoleMagnitude / (d * d * d)
                   + 3.5 * quadrupoleNorm / (d * d * d * d));
        };

        double low = sourceRadius;
        if(low <= 0.0)
            low = 1.0e-6 * std::min({cellSize[0], cellSize[1], cellSize[2]});
        else
            low += 1.0e-12 * std::max(low, std::min({cellSize[0], cellSize[1], cellSize[2]}));
        if(tailBound(low) <= target)
        {
            result.distance = low;
            result.bound = tailBound(low);
        }
        else
        {
            double high = std::max(2.0 * low, sourceRadius + std::min({cellSize[0], cellSize[1], cellSize[2]}));
            while(tailBound(high) > target && high < maxDistance)
                high = std::min(2.0 * high, maxDistance);
            if(tailBound(high) > target)
                throw std::runtime_error(
                    "deriveFieldTailMargin: a neglected-field bound below tolerance requires a distance"
                    " larger than " + std::to_string(maxDistance) + " m; enlarge the compute box or"
                    " acknowledge the truncation explicitly");
            for(int iteration = 0; iteration < 200; ++iteration)
            {
                double const mid = 0.5 * (low + high);
                if(tailBound(mid) > target)
                    low = mid;
                else
                    high = mid;
            }
            result.distance = high;
            result.bound = tailBound(high);
        }

        for(int a = 0; a < 3; ++a)
        {
            double const cells = (result.distance - halfExtent[a]) / cellSize[a];
            result.marginCells[a] = std::max(0.0, std::ceil(cells));
        }
        return result;
    }

    Grid makeGrid(Bunch const& bunch, ComputeOptions const& options)
    {
        if(bunch.size() == 0)
            throw std::runtime_error("makeGrid: empty bunch");
        if(options.marginCells < 0.0)
            throw std::invalid_argument("makeGrid: marginCells must not be negative");
        if(options.marginCellsPerAxis)
            for(double margin : *options.marginCellsPerAxis)
                if(!(margin >= 0.0) || !std::isfinite(margin))
                    throw std::invalid_argument(
                        "makeGrid: every marginCellsPerAxis value must be finite and non-negative");
        for(double c : options.cellSize)
            if(!(c > 0.0) || !std::isfinite(c))
                throw std::invalid_argument("makeGrid: every cell size must be finite and positive");

        auto const b = bunch.bounds();
        Grid grid;
        // The tolerance policy picks the FFT padding; a non-zero fftPadding is an
        // explicit diagnostic override (1 is the wraparound control).
        unsigned const padding = (options.fftPadding != 0u) ? options.fftPadding
                                                           : selectFftPadding(options.tolerance);
        if(padding == 0u)
            throw std::invalid_argument("makeGrid: fftPadding must be >= 1");
        grid.fftPadding = padding;
        grid.shape = options.shape;

        for(int a = 0; a < 3; ++a)
        {
            double const spacing = kRefinement * options.cellSize[a];
            double const marginCellCount = options.marginCellsPerAxis
                ? (*options.marginCellsPerAxis)[a]
                : options.marginCells;
            double const margin = marginCellCount * options.cellSize[a];
            double const low = b[0][a] - margin;
            double const high = b[1][a] + margin;
            double const span = (high - low) / spacing;
            if(!std::isfinite(span) || span > static_cast<double>(kMaxNodesPerAxis))
                throw std::runtime_error(
                    "makeGrid: the physical cell size produces " + std::to_string(span)
                    + " nodes on one axis, which is infeasible (limit "
                    + std::to_string(kMaxNodesPerAxis)
                    + "); check the input gridSpacing/gridUnitSI or pass --cell-size");
            std::size_t nodes = static_cast<std::size_t>(std::ceil(span)) + 1u;
            // The auto-derived grid keeps an even node count for backwards
            // compatibility (earlier tests expect even n).  Odd extents are now
            // supported by the R2C path (parity-correct centring in
            // KernelBuildKernel and the relaxed --grid check); this rounding only
            // constrains the derived grid, never an explicit one.
            if(nodes % 2u)
                ++nodes;
            nodes = std::max<std::size_t>(nodes, 8u);
            grid.n[a] = nodes;
            grid.spacing[a] = spacing;
            grid.origin[a] = low;
        }
        // An explicit CLI --grid overrides the mesh extent, which in turn
        // overrides the bunch-bounding-box derivation.  Values are used
        // verbatim and never blended.
        if(options.gridSize)
        {
            for(int a = 0; a < 3; ++a)
            {
                std::size_t const n = (*options.gridSize)[a];
                if(n == 0u)
                    throw std::invalid_argument(
                        "makeGrid: --grid values must be positive; got "
                        + std::to_string((*options.gridSize)[0]) + ","
                        + std::to_string((*options.gridSize)[1]) + ","
                        + std::to_string((*options.gridSize)[2]));
                grid.n[a] = n;
            }
        }
        else if(options.meshGridSize)
        {
            // The mesh extent is the node count of its dataset, already mapped
            // to x,y,z by readMeshGeometry.  Copy it verbatim.
            for(int a = 0; a < 3; ++a)
            {
                if((*options.meshGridSize)[a] == 0u)
                    throw std::invalid_argument("makeGrid: the input mesh extent must be positive");
                grid.n[a] = (*options.meshGridSize)[a];
            }
        }
        // An explicit --origin overrides the mesh gridGlobalOffset, which in
        // turn overrides the derived bunch-box minimum.
        std::optional<std::array<double, 3>> const& originOverride
            = options.gridOrigin ? options.gridOrigin : options.meshGridOrigin;
        if(originOverride)
        {
            for(int a = 0; a < 3; ++a)
            {
                if(!std::isfinite((*originOverride)[a]))
                    throw std::invalid_argument("makeGrid: grid origin values must be finite");
                grid.origin[a] = (*originOverride)[a];
            }
        }

        // A global output domain does not enlarge the computation.  It only
        // says where the small compute box sits inside the written domain.  Snap
        // the compute box onto global nodes so the read-out maps one-to-one and
        // no interpolation is needed; clip it to the global extent.
        if(options.globalGridSize)
        {
            std::array<std::size_t, 3> const globalN = *options.globalGridSize;
            std::array<double, 3> const globalOrigin
                = options.globalGridOrigin.value_or(std::array<double, 3>{0.0, 0.0, 0.0});
            for(int a = 0; a < 3; ++a)
            {
                if(globalN[a] == 0u || globalN[a] % 2u != 0u)
                    throw std::invalid_argument(
                        "makeGrid: --global-grid values must be even and positive");
                if(!std::isfinite(globalOrigin[a]))
                    throw std::invalid_argument("makeGrid: --global-origin values must be finite");
            }
            for(int a = 0; a < 3; ++a)
            {
                double const spacing = grid.spacing[a];
                // Smallest global node at or below the compute-box origin.
                long long low = std::max<long long>(
                    0,
                    static_cast<long long>(
                        std::floor((grid.origin[a] - globalOrigin[a]) / spacing)));
                long long high = std::min<long long>(
                    static_cast<long long>(globalN[a]),
                    static_cast<long long>(std::ceil(
                        (grid.origin[a] + static_cast<double>(grid.n[a]) * spacing
                         - globalOrigin[a])
                        / spacing)));
                if(low >= high)
                    throw std::runtime_error(
                        "makeGrid: the compute box does not overlap the global output domain on axis "
                        + std::to_string(a));
                // Keep the R2C last extent even.  Prefer growing, then shrinking
                // the low side, so a particle at the upper edge is never dropped.
                if((high - low) % 2 != 0)
                {
                    if(high < static_cast<long long>(globalN[a]))
                        ++high;
                    else
                        --low;
                }
                grid.origin[a] = globalOrigin[a] + static_cast<double>(low) * spacing;
                grid.n[a] = static_cast<std::size_t>(high - low);
            }
        }

        // The TSC deposit silently skips nodes outside the output grid, so a
        // grid that does not contain every particle would drop charge without
        // a trace.  Reject that instead.  The physical extent of the node set
        // is [origin, origin + n*spacing], matching the mesh dataset extent.
        {
            std::array<double, 3> lower{};
            std::array<double, 3> upper{};
            for(int a = 0; a < 3; ++a)
            {
                lower[a] = grid.origin[a];
                upper[a] = grid.origin[a] + static_cast<double>(grid.n[a]) * grid.spacing[a];
            }
            std::size_t outside = 0u;
            for(std::size_t p = 0; p < bunch.size(); ++p)
            {
                double const position[3] = {bunch.x[p], bunch.y[p], bunch.z[p]};
                for(int a = 0; a < 3; ++a)
                {
                    if(position[a] < lower[a] || position[a] > upper[a])
                    {
                        ++outside;
                        break;
                    }
                }
            }
            if(outside != 0u)
                throw std::runtime_error(
                    "makeGrid: " + std::to_string(outside) + " of " + std::to_string(bunch.size())
                    + " particles lie outside the output grid bounds "
                    + "[" + std::to_string(lower[0]) + ", " + std::to_string(upper[0]) + "] x "
                    + "[" + std::to_string(lower[1]) + ", " + std::to_string(upper[1]) + "] x "
                    + "[" + std::to_string(lower[2]) + ", " + std::to_string(upper[2]) + "]"
                    + "; the grid comes from the input mesh or --grid/--origin/--cell-size,"
                    + " and out-of-grid particles would be dropped");
        }

        // Reject an infeasible grid before the first allocation.  Both the
        // per-axis and the padded total are checked; the product is formed in
        // 64-bit with an explicit overflow guard.  The replicated path keeps
        // the historical 2^28-node cap; a distributed context may exceed it by
        // up to the aggregate per-rank budget, because the slab decomposition
        // keeps only M^3/size nodes per rank.
        bool const canDistribute = options.parallel && options.parallel->distributed()
            && options.fftMode != FftMode::Replicated;
        std::uint64_t maxPadded = static_cast<std::uint64_t>(kMaxPaddedNodes);
        if(canDistribute)
            maxPadded *= static_cast<std::uint64_t>(options.parallel->size) * 8u;
        std::uint64_t padded = 1u;
        for(int a = 0; a < 3; ++a)
        {
            if(grid.n[a] > kMaxNodesPerAxis)
                throw std::runtime_error(
                    "makeGrid: output grid " + std::to_string(grid.n[0]) + "x"
                    + std::to_string(grid.n[1]) + "x" + std::to_string(grid.n[2])
                    + " is infeasible (per-axis limit " + std::to_string(kMaxNodesPerAxis)
                    + "); check the input gridSpacing/gridUnitSI or pass --cell-size");
            std::uint64_t const axisPadded = static_cast<std::uint64_t>(grid.n[a]) * padding;
            if(axisPadded != 0u && padded > maxPadded / axisPadded)
                throw std::runtime_error(
                    "makeGrid: padded FFT grid for output grid " + std::to_string(grid.n[0]) + "x"
                    + std::to_string(grid.n[1]) + "x" + std::to_string(grid.n[2])
                    + " with padding " + std::to_string(padding)
                    + " is infeasible (limit " + std::to_string(maxPadded)
                    + " nodes); check the input gridSpacing/gridUnitSI or pass --cell-size");
            padded *= axisPadded;
        }
        return grid;
    }

    Field computeField(
        Bunch const& bunch,
        ComputeOptions const& options,
        StageTimes* stageTimes,
        VelocityBinning* outBinning,
        VelocityMomentsReport* outReport)
    {
        if(outReport)
            *outReport = VelocityMomentsReport{};
        if(options.nearShell.mode != NearShellOptions::Mode::None
           && options.shape != Shape::TSC)
            throw std::runtime_error(
                "the near shell currently supports the TSC shape only; use "
                "--near-shell none or --shape tsc");

        // Both near-shell backends share the tiled assembly
        // F = F_mesh(all) - F_mesh(near) + F_exact(near); only the per-pair
        // evaluator differs (adaptive reference vs closed form).
        if(options.nearShell.mode != NearShellOptions::Mode::None)
            return computeFieldNearExact(
                bunch, options, options.nearShell, stageTimes, outBinning, outReport);

        // A multi-velocity bunch is the sum over velocity classes: the kernel
        // is linear in charge, so each velocity box contributes its own
        // deposit, kernel spectrum and convolution, accumulated into the same
        // output.  The box geometry depends only on the u-space extent, so a
        // fixed binning (the near shell passes the full-bunch one down) stays
        // bin-consistent on any sub-bunch and the near shell is bin-free.
        VelocityBinning derivedBinning;
        VelocityBinning const* binningPointer
            = options.fixedBinning ? &*options.fixedBinning : nullptr;
        if(!binningPointer)
        {
            try
            {
                derivedBinning = deriveVelocityBinning(
                    bunch, options.tolerance, options.velocityBinning);
            }
            catch(std::runtime_error const& error)
            {
                // An enabled moment path does not use the velocity boxes.  Let
                // it proceed instead of failing at the bin-count cap; only when
                // its a-posteriori probe declines does the binned path below run,
                // and then with the largest count the policy allows.
                if(options.velocityMoments.mode == VelocityMomentsOptions::Mode::Off)
                    throw;
                std::fprintf(
                    stderr,
                    "picBunch: velocity-bin auto policy exceeded the %zu-box cap (%s); "
                    "trying the velocity-moment path\n",
                    options.velocityBinning.maxBins,
                    error.what());
                VelocityBinningOptions capped = options.velocityBinning;
                capped.mode = VelocityBinningOptions::Mode::Explicit;
                capped.explicitBins = capped.maxBins;
                derivedBinning = deriveVelocityBinning(bunch, options.tolerance, capped);
            }
            binningPointer = &derivedBinning;
        }
        if(outBinning)
            *outBinning = *binningPointer;
        VelocityBinning const& binning = *binningPointer;
        std::size_t const binCount = binning.count;

        // Assignment of the actual bunch to the boxes.  A fixed binning was
        // derived for the full bunch, so a sub-bunch (the near shell tile
        // meshes) re-derives its own membership from the shared geometry.
        std::vector<std::size_t> const boxOf = [&] {
            if(!options.fixedBinning)
                return binning.boxOf;
            std::vector<std::size_t> assignment(bunch.size());
            for(std::size_t p = 0; p < bunch.size(); ++p)
            {
                double const coords[3] = {
                    bunch.ux[p] * binning.basis[0][0] + bunch.uy[p] * binning.basis[0][1]
                        + bunch.uz[p] * binning.basis[0][2],
                    bunch.ux[p] * binning.basis[1][0] + bunch.uy[p] * binning.basis[1][1]
                        + bunch.uz[p] * binning.basis[1][2],
                    bunch.ux[p] * binning.basis[2][0] + bunch.uy[p] * binning.basis[2][1]
                        + bunch.uz[p] * binning.basis[2][2]};
                assignment[p] = binning.boxIndex(coords[0], coords[1], coords[2]);
            }
            return assignment;
        }();
        std::vector<std::vector<std::size_t>> members(binCount);
        for(std::size_t p = 0; p < bunch.size(); ++p)
            members[boxOf[p]].push_back(p);

        Grid const grid = makeGrid(bunch, options);
        Field field;
        field.grid = grid;

        StageTimes localTimes;
        StageTimes& times = stageTimes ? *stageTimes : localTimes;
        times = StageTimes{};
        // The region from entry to the first deposit is currently untimed:
        // velocity binning, grid setup, allocations, and the r2c/c2r plan and
        // workspace build.  Record it once, at the first deposit of the path
        // that actually runs, as StageTimes::planMs.
        auto const callStart = Clock::now();
        bool planCaptured = false;

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
        // Mx may be odd: the R2C half-spectrum then has (Mx+1)/2 planes and no Nyquist
        // plane; KernelBuildKernel's parity-correct centred index handles that.  The
        // integer ``Mx/2 + 1`` already equals (Mx+1)/2 for odd Mx and Mx/2+1 for even.
        std::uint32_t const CP = Mx / 2u + 1u;
        // Flat layout is x fastest, then y, then z.  The plan's last dimension is x,
        // so the plan extents are (z, y, x) and the packed spectrum is (z, y, CP).
        Extents3 const realExt{Mz, My, Mx};
        Extents3 const specExt{Mz, My, CP};

        auto const cell = std::array<double, 3>{
            grid.spacing[0] / kRefinement,
            grid.spacing[1] / kRefinement,
            grid.spacing[2] / kRefinement};

        std::uint64_t const paddedNodes = static_cast<std::uint64_t>(Mx) * My * Mz;
        std::uint32_t const spectrumCount = Mz * My * CP;
        // Packed half-spectrum view: Complex is two adjacent doubles.  The last
        // MdSpan dimension is the fastest one, so the {count, 2} view with index
        // {l, 0} / {l, 1} lands on element 2*l / 2*l+1 (re / im).
        using PackedSpectrumIndex = alpaka::Vec<std::uint32_t, 2u>;
        auto const packedSpectrumExtents = PackedSpectrumIndex{spectrumCount, 2u};
        static_assert(std::is_same_v<
                      std::remove_cvref_t<decltype(packedSpectrumExtents)>::value_type,
                      PackedSpectrumIndex::value_type>);
        auto const packedSpectrumPitches
            = alpaka::Vec<std::size_t, 2u>{2u * sizeof(double), sizeof(double)};
        auto packedSpectrum = [&](double* data) {
            return alpaka::makeMdSpan(data, packedSpectrumExtents, packedSpectrumPitches);
        };
        // The replicated path keeps its FFT buffers as flat 1-D allocations.
        // Every access in this path (deposit, kernel build, spectrum
        // multiply, read-out) and the vendor plans (contiguous layout,
        // inembed == extents) use the dense x-fastest layout.  A 3-D
        // allocUnified instead pads each innermost row to the 128-byte
        // boundary whenever Mx is not a multiple of 16; alpaka's pitched
        // memset then leaves the padding bytes uncleared, and those bytes sit
        // inside the flat prefix the FFT reads.  Reused across the velocity
        // bins below, the stale padding carried the previous bin's charge
        // into the transform (characterised in tools/fft-size).
        auto rhoReal = alpaka::onHost::allocUnified<double>(device, static_cast<std::size_t>(paddedNodes));
        auto rhoSpec = alpaka::onHost::allocUnified<Complex>(device, static_cast<std::size_t>(spectrumCount));
        auto kernelReal = alpaka::onHost::allocUnified<double>(device, static_cast<std::size_t>(paddedNodes));
        auto kernelSpec = alpaka::onHost::allocUnified<Complex>(device, static_cast<std::size_t>(spectrumCount));
        auto outReal = alpaka::onHost::allocUnified<double>(device, static_cast<std::size_t>(paddedNodes));

        // Read-out geometry for the replicated path.  The c2r output stays
        // device-resident; a no-arithmetic gather copies the valid n^3 box into
        // a compact device buffer, one D2H copy moves it into pinned host
        // memory, and the host accumulate loop touches only the pinned buffer.
        // Before this, the OpenMP loop dereferenced the managed ``outReal`` and
        // paid a page migration on every first touch.  The staging buffers are
        // allocated only in the replicated branch below: the distributed path
        // reads its own z-slab and never gathers into them.
        std::uint32_t const outNx = static_cast<std::uint32_t>(grid.n[0]);
        std::uint32_t const outNy = static_cast<std::uint32_t>(grid.n[1]);
        std::uint32_t const outNz = static_cast<std::uint32_t>(grid.n[2]);
        std::size_t const outNodes
            = static_cast<std::size_t>(outNx) * static_cast<std::size_t>(outNy)
            * static_cast<std::size_t>(outNz);
        // CompactGatherKernel indexes with 32-bit values; guard the narrowing so a
        // future raise of kMaxPaddedNodes cannot silently truncate the count.
        if(outNodes > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
            throw std::runtime_error(
                "computeField: output node count " + std::to_string(outNodes)
                + " exceeds the 32-bit gathered-index limit");

        // Serial host TSC deposit of a particle list, in list order.  Same
        // stencil as the device kernel, without atomicAdd's unspecified order.
        auto depositHostImpl = [&](
                               auto tag,
                               std::vector<std::size_t> const& list,
                               double* rho,
                               std::uint32_t rhoStride,
                               std::uint32_t zOffset,
                               std::uint32_t zExtent) {
            constexpr Shape S = decltype(tag)::value;
            double const support = shapeSupport<S>();
            double const org[3] = {grid.origin[0], grid.origin[1], grid.origin[2]};
            double const sp[3] = {grid.spacing[0], grid.spacing[1], grid.spacing[2]};
            for(std::size_t const p : list)
            {
                double const pos[3] = {bunch.x[p], bunch.y[p], bunch.z[p]};
                long lo[3];
                long hi[3];
                double t[3];
                for(int a = 0; a < 3; ++a)
                {
                    t[a] = (pos[a] - org[a]) / sp[a];
                    lo[a] = static_cast<long>(::ceil(t[a] - support));
                    hi[a] = static_cast<long>(::floor(t[a] + support));
                }
                double const q = bunch.weight[p] * bunch.charge[p];
                for(long ia = lo[0]; ia <= hi[0]; ++ia)
                {
                    double const wa = shapeWeight<S>(static_cast<double>(ia) - t[0]);
                    if(wa == 0.0)
                        continue;
                    long const ma = wrapPaddedIndex(ia, Mx);
                    for(long ib = lo[1]; ib <= hi[1]; ++ib)
                    {
                        double const wb = wa * shapeWeight<S>(static_cast<double>(ib) - t[1]);
                        if(wb == 0.0)
                            continue;
                        long const mb = wrapPaddedIndex(ib, My);
                        for(long ic = lo[2]; ic <= hi[2]; ++ic)
                        {
                            double const wc = wb * shapeWeight<S>(static_cast<double>(ic) - t[2]);
                            if(wc == 0.0)
                                continue;
                            long const mc = wrapPaddedIndex(ic, Mz);
                            if(mc < static_cast<long>(zOffset)
                               || mc >= static_cast<long>(zOffset + zExtent))
                                continue;
                            auto const idx = static_cast<std::size_t>(ma)
                                + static_cast<std::size_t>(Mx)
                                    * (static_cast<std::size_t>(mb)
                                       + static_cast<std::size_t>(My)
                                           * static_cast<std::size_t>(mc - zOffset));
                            rho[idx * rhoStride] += wc * q;
                        }
                    }
                }
            }
        };

        auto depositHost = [&](
                               std::vector<std::size_t> const& list,
                               double* rho,
                               std::uint32_t rhoStride,
                               std::uint32_t zOffset,
                               std::uint32_t zExtent) {
            if(grid.shape == Shape::CIC)
                depositHostImpl(
                    std::integral_constant<Shape, Shape::CIC>{},
                    list,
                    rho,
                    rhoStride,
                    zOffset,
                    zExtent);
            else if(grid.shape == Shape::PQS)
                depositHostImpl(
                    std::integral_constant<Shape, Shape::PQS>{},
                    list,
                    rho,
                    rhoStride,
                    zOffset,
                    zExtent);
            else
                depositHostImpl(
                    std::integral_constant<Shape, Shape::TSC>{},
                    list,
                    rho,
                    rhoStride,
                    zOffset,
                    zExtent);
        };

        // Device TSC deposit of a particle list into the (already zeroed) rho.
        // ``rhoView`` is a 3-D span over the target slab: a double box in the
        // replicated path and a Complex slab in the distributed path.
        auto depositDevice = [&](
                                 std::vector<std::size_t> const& list,
                                 auto rhoView,
                                 std::uint32_t zOffset,
                                 std::uint32_t zExtent) {
            auto const numParticles = static_cast<std::uint32_t>(list.size());
            auto posX = alpaka::onHost::allocUnified<double>(device, numParticles);
            auto posY = alpaka::onHost::allocUnified<double>(device, numParticles);
            auto posZ = alpaka::onHost::allocUnified<double>(device, numParticles);
            auto charge = alpaka::onHost::allocUnified<double>(device, numParticles);
            for(std::uint32_t m = 0; m < numParticles; ++m)
            {
                std::size_t const p = list[m];
                posX.data()[m] = bunch.x[p];
                posY.data()[m] = bunch.y[p];
                posZ.data()[m] = bunch.z[p];
                charge.data()[m] = bunch.weight[p] * bunch.charge[p];
            }
            std::uint32_t const frame = 64u;
            auto launch = [&](auto tag) {
                constexpr Shape S = decltype(tag)::value;
                auto const particleExtents = alpaka::Vec<std::uint32_t, 1u>{numParticles};
                static_assert(std::is_same_v<
                              std::remove_cvref_t<decltype(particleExtents)>::value_type,
                              std::uint32_t>);
                queue.enqueue(
                    alpaka::onHost::FrameSpec{(numParticles + frame - 1u) / frame, frame},
                    TscDepositKernel<S>{},
                    alpaka::makeMdSpan(posX.data(), particleExtents),
                    alpaka::makeMdSpan(posY.data(), particleExtents),
                    alpaka::makeMdSpan(posZ.data(), particleExtents),
                    alpaka::makeMdSpan(charge.data(), particleExtents),
                    numParticles,
                    rhoView,
                    zOffset,
                    zExtent,
                    Mx,
                    My,
                    Mz,
                    alpaka::Vec{grid.origin[0], grid.origin[1], grid.origin[2]},
                    alpaka::Vec{grid.spacing[0], grid.spacing[1], grid.spacing[2]});
            };
            if(grid.shape == Shape::CIC)
                launch(std::integral_constant<Shape, Shape::CIC>{});
            else if(grid.shape == Shape::PQS)
                launch(std::integral_constant<Shape, Shape::PQS>{});
            else
                launch(std::integral_constant<Shape, Shape::TSC>{});
            // The host queue is non-blocking (a CallbackThread worker runs the
            // kernel), so the particle buffers must stay alive until the wait.
            // Keep this wait inside the block that owns posX/posY/posZ/charge.
            alpaka::onHost::wait(queue);
        };

        // ---------------------------------------------------------------- FFT
        auto r2cPlan
            = alpaka::fft::onHost::PlanBuilder<double, Extents3>{realExt}.r2c().outOfPlace().build(device);
        auto c2rPlan
            = alpaka::fft::onHost::PlanBuilder<double, Extents3>{realExt}.c2r().outOfPlace().build(device);
#if defined(ALPAKAV_DEP_ROCFFT) && ALPAKAV_DEP_ROCFFT
        // Deviation from the pure-alpaka rule, kept on purpose.  This is the only
        // native vendor API call in the tree: the HIP-only rocfft_cleanup() from
        // <rocfft/rocfft.h>, compiled only when the ROCm/HIP backend is built.
        // It is required because ROCm 7.2 has a static-destruction-order bug in
        // librocfft's process-global HIP-stream pool.  rocFFT destroys that pool
        // from a library static destructor at unload, and on ROCm 7.2 that
        // destructor calls hipStreamDestroy after the HIP runtime has finalised
        // and segfaults at exit.  Without this teardown the picBunch.Io test
        // regresses on ROCm 7.2 (it was fixed by commit 833b627).
        //
        // Do not call cleanup inline.  The pool is process-global, so an inline
        // call at the end of the first computeField destroys it while later
        // computeField calls may still create plans; those then fail with
        // rocfft_plan_create error 1.  Register a process-exit handler instead.
        //
        // Ordering: C++ runs atexit handlers in reverse order of registration.
        // alpakaVendor registers its own rocfft_cleanup atexit hook when the
        // first plan is created, and the HIP runtime registers its static
        // destructors before that.  Registering this handler after both plans
        // are built therefore makes it run before the HIP static destructors,
        // while the plans (function locals, already destroyed on return from
        // computeField) are gone and the HIP runtime is still alive.
        //
        // Idempotency: std::call_once registers the handler exactly once per
        // process.  After this handler runs, alpakaVendor's later (earlier
        // registered) atexit hook calls rocfft_cleanup() again; that second
        // call is assumed to be a no-op by librocfft.  This idempotency is the
        // documented contract of rocfft_cleanup() and is relied on here.
        //
        // Temporary ROCm 7.2 workaround. See docs/rocm-fft-teardown.md.
        // alpakaVendor's PlanImpl destructor owns the plan lifetime correctly
        // but cannot fix the process-global teardown, and the pin has no public
        // alpaka::fft::onHost::finalize().  The clean fix is to add that
        // finalize() (idempotent, once-guarded) to alpakaVendor; then replace
        // this handler with that call and drop the rocfft/rocfft.h include.
        static std::once_flag rocfftAtexitOnce;
        std::call_once(
            rocfftAtexitOnce,
            [] { std::atexit([] { static_cast<void>(rocfft_cleanup()); }); });
#endif

        double const scale = (kRefinement * kRefinement * kRefinement)
            / (static_cast<double>(Mx) * static_cast<double>(My) * static_cast<double>(Mz));
        double const eFactor = kCoulombConstant;
        double const bFactor = kCoulombConstant / kSpeedOfLight;

        // Component/bin parallelism.  Every rank deposits and transforms the
        // same source spectrum, then owns a disjoint subset of the six output
        // components.  The replicated path keeps the whole transform on each
        // rank; the distributed path decomposes the padded box by z-slabs and
        // uses alltoall transposes.  The six component arrays are summed after
        // the replicated loop; the distributed path reduces each component as
        // soon as its local z-slab has been read out.
        ParallelContext const* par = options.parallel;
        int const parSize = (par && par->distributed()) ? par->size : 1;
        int const parRank = (par && par->distributed()) ? par->rank : 0;

        // Auto uses the slab algorithm only above the configured padded-node
        // threshold; the explicit mode also runs it on one rank so its written
        // field can be compared with a multi-rank run.  The end-to-end field is
        // covered by Fft.DistributedProductionMatchesReplicatedWithDeviceDeposit.
        // With MPI every rank must keep only its slab: auto uses the
        // distributed transform for any multi-rank run.  Single-rank auto keeps
        // the replicated path until the padded box exceeds the configured
        // threshold, where the replicated buffers would be too large.
        bool const distributedRequested = options.fftMode == FftMode::Distributed
            || (options.fftMode == FftMode::Auto
                && (parSize > 1 || paddedNodes > options.distributedFftMinNodes));
        bool const useDistributed = distributedRequested && par != nullptr
            && (parSize > 1 || options.fftMode == FftMode::Distributed);

        // ---------------------------------------------------------- velocity moments
        // Opt-in (candidate C1): replace the per-velocity-box FFT chain with a
        // one-inverse-transform moment sum.  The point kernel is expanded in a
        // multivariate Taylor series about the absolute-charge-weighted mean u0,
        // the moment densities M_a are deposited once, and each component is the
        // inverse transform of sum_a (FFT M_a)(FFT K_a^c).  It always uses the
        // replicated FFT (full transform per rank, component-parallel), independent
        // of options.fftMode.  When the policy is disabled, infeasible, or (auto)
        // not converged this block returns without a result and the binned path
        // below runs unchanged, so the default stays bit-for-bit identical.
        if(options.velocityMoments.mode != VelocityMomentsOptions::Mode::Off)
        {
            auto const& vm = options.velocityMoments;
            VelocityMomentsReport localReport;
            VelocityMomentsReport& rep = outReport ? *outReport : localReport;
            rep.feasible = true;
            std::optional<Field> vmField;
            constexpr double kMomentAutoTol = 0.05;
            std::size_t const nTerms3 = detail::velocityMomentTerms(3u);

            if(bunch.size() > 0u && bunch.size() <= 0xffffffffu)
            {
                std::array<double, 3> u0{};
                double maxSpread = 0.0;
                try
                {
                    // The near shell pins the centre so its per-tile
                    // subtraction uses the same expansion as the global mesh.
                    if(options.velocityMomentsCentre)
                        u0 = *options.velocityMomentsCentre;
                    else
                        u0 = picBunch::velocityMomentsCentre(bunch);
                    maxSpread = velocityMomentsRadius(bunch, u0);
                }
                catch(std::exception const&)
                {
                    rep.feasible = false;
                    rep.fellBack = true;
                }
                if(rep.feasible)
                {
                    rep.u0 = u0;
                    rep.maxSpread = maxSpread;
                    // Memory feasibility: keep the moment scratch (20 moment
                    // spectra + 20 kernel planes + the convergence probe) under a
                    // guard so an oversized box falls back instead of exhausting
                    // the device.  The probe is computed for explicit orders too
                    // (see below), so it is always budgeted.
                    std::uint64_t const momentSpecBytes
                        = nTerms3 * static_cast<std::uint64_t>(spectrumCount) * sizeof(Complex);
                    std::uint64_t const kernelMomBytes
                        = nTerms3 * paddedNodes * sizeof(double);
                    std::uint64_t const probeBytes
                        = static_cast<std::uint64_t>(spectrumCount) * sizeof(Complex);
                    constexpr std::uint64_t kMomentMaxScratch
                        = 8ull * 1024ull * 1024ull * 1024ull;
                    if(momentSpecBytes + kernelMomBytes + probeBytes > kMomentMaxScratch)
                    {
                        rep.feasible = false;
                        rep.fellBack = true;
                    }
                }
                if(rep.feasible)
                {
                    std::size_t const orderUsed
                        = (vm.mode == VelocityMomentsOptions::Mode::Explicit)
                            ? (std::min<std::size_t>(vm.order, 3u))
                            : 3u;
                    std::uint32_t const N = static_cast<std::uint32_t>(bunch.size());

                    // Whole-bunch particle arrays (the moment path deposits all
                    // particles at once, not per velocity box).
                    auto px = alpaka::onHost::allocUnified<double>(device, N);
                    auto py = alpaka::onHost::allocUnified<double>(device, N);
                    auto pz = alpaka::onHost::allocUnified<double>(device, N);
                    auto pux = alpaka::onHost::allocUnified<double>(device, N);
                    auto puy = alpaka::onHost::allocUnified<double>(device, N);
                    auto puz = alpaka::onHost::allocUnified<double>(device, N);
                    auto pq = alpaka::onHost::allocUnified<double>(device, N);
                    for(std::uint32_t m = 0; m < N; ++m)
                    {
                        px.data()[m] = bunch.x[m];
                        py.data()[m] = bunch.y[m];
                        pz.data()[m] = bunch.z[m];
                        pux.data()[m] = bunch.ux[m];
                        puy.data()[m] = bunch.uy[m];
                        puz.data()[m] = bunch.uz[m];
                        pq.data()[m] = bunch.weight[m] * bunch.charge[m];
                    }

                    auto const moments = velocityMomentMultiIndices(3u);
                    // One contiguous coefficient-plane buffer replaces the former
                    // device-side double* array.  The build kernel addresses it as
                    // a 4-D IMdSpan [plane][z][y][x]; each plane is also handed to
                    // the FFT as a dense MdSpan view.  Round the per-plane pitch
                    // up to a 128-byte (16 double) boundary so a plane view keeps
                    // the alignment a separate allocation used to have.
                    auto firstMom = alpaka::onHost::allocUnified<Complex>(
                        device,
                        static_cast<std::size_t>(spectrumCount));
                    using MomBuf = decltype(firstMom);
                    std::vector<MomBuf> momentBufs;
                    momentBufs.reserve(nTerms3);
                    momentBufs.push_back(std::move(firstMom));
                    for(std::size_t a = 1u; a < nTerms3; ++a)
                        momentBufs.push_back(
                            alpaka::onHost::allocUnified<Complex>(
                                device,
                                static_cast<std::size_t>(spectrumCount)));
                    std::size_t const planeStride
                        = ((static_cast<std::size_t>(paddedNodes) + 15u) / 16u) * 16u;
                    auto kernelPlaneData = alpaka::onHost::allocUnified<double>(
                        device,
                        nTerms3 * planeStride);
                    auto planeView = [&](std::size_t a) {
                        return alpaka::makeMdSpan(
                            kernelPlaneData.data() + a * planeStride,
                            realExt);
                    };
                    auto const kernelPlaneExtents = alpaka::Vec<std::uint32_t, 4u>{
                        static_cast<std::uint32_t>(nTerms3),
                        Mz,
                        My,
                        Mx};
                    auto const kernelPlanePitches = alpaka::Vec<std::size_t, 4u>{
                        planeStride * sizeof(double),
                        static_cast<std::size_t>(Mx) * My * sizeof(double),
                        static_cast<std::size_t>(Mx) * sizeof(double),
                        sizeof(double)};
                    auto kernelPlanesView = alpaka::makeMdSpan(
                        kernelPlaneData.data(),
                        kernelPlaneExtents,
                        kernelPlanePitches);
                    alpaka::onHost::wait(queue);
                    // The order-3 remainder is needed for the convergence probe
                    // in explicit mode as well; both orders validate against it.
                    auto o3Spec = alpaka::onHost::allocUnified<Complex>(
                        device,
                        static_cast<std::size_t>(spectrumCount));

                    auto launchMomentDeposit = [&](int a1, int a2, int a3) {
                        auto launch = [&](auto tag) {
                            constexpr Shape S = decltype(tag)::value;
                            auto const particleExtents = alpaka::Vec<std::uint32_t, 1u>{N};
                            auto const rhoExtents = alpaka::Vec<std::uint32_t, 3u>{Mz, My, Mx};
                            static_assert(std::is_same_v<
                                          std::remove_cvref_t<decltype(particleExtents)>::value_type,
                                          std::uint32_t>);
                            static_assert(std::is_same_v<
                                          std::remove_cvref_t<decltype(rhoExtents)>::value_type,
                                          std::uint32_t>);
                            queue.enqueue(
                                alpaka::onHost::FrameSpec{(N + 63u) / 64u, 64u},
                                MomentDepositKernel<S>{},
                                alpaka::makeMdSpan(px.data(), particleExtents),
                                alpaka::makeMdSpan(py.data(), particleExtents),
                                alpaka::makeMdSpan(pz.data(), particleExtents),
                                alpaka::makeMdSpan(pux.data(), particleExtents),
                                alpaka::makeMdSpan(puy.data(), particleExtents),
                                alpaka::makeMdSpan(puz.data(), particleExtents),
                                alpaka::makeMdSpan(pq.data(), particleExtents),
                                N,
                                alpaka::makeMdSpan(rhoReal.data(), rhoExtents),
                                0u,
                                Mz,
                                Mx,
                                My,
                                Mz,
                                alpaka::Vec{grid.origin[0], grid.origin[1], grid.origin[2]},
                                alpaka::Vec{grid.spacing[0], grid.spacing[1], grid.spacing[2]},
                                alpaka::Vec{u0[0], u0[1], u0[2]},
                                a1,
                                a2,
                                a3);
                        };
                        if(grid.shape == Shape::CIC)
                            launch(std::integral_constant<Shape, Shape::CIC>{});
                        else if(grid.shape == Shape::PQS)
                            launch(std::integral_constant<Shape, Shape::PQS>{});
                        else
                            launch(std::integral_constant<Shape, Shape::TSC>{});
                    };

                    // ---- deposit + FFT every moment density M_a (once)
                    auto const depStart = Clock::now();
                    if(!planCaptured)
                    {
                        times.planMs = elapsedMs(callStart);
                        planCaptured = true;
                    }
                    for(auto const& mi : moments)
                    {
                        alpaka::onHost::memset(queue, rhoReal, std::uint8_t{0});
                        launchMomentDeposit(mi.a1, mi.a2, mi.a3);
                        alpaka::fft::onHost::executeForward(
                            queue, r2cPlan, rhoReal, momentBufs[static_cast<std::size_t>(mi.idx)]);
                        alpaka::onHost::wait(queue);
                    }
                    times.depositMs += elapsedMs(depStart);

                    // ---- field = sum_{|a|<=order} conv(M_a, K_a^c) per component
                    Field vf;
                    vf.grid = grid;
                    for(auto& comp : vf.data)
                        comp.assign(grid.size(), 0.0);
                    // Read-out accumulator on the queue's device: the field is
                    // filled by an alpaka onHost::transform and copied to the
                    // host vector once per component.  The two-double probe
                    // accumulator persists across components so the running sum
                    // keeps the original addition order.
                    auto readoutAcc = alpaka::onHost::allocUnified<double>(
                        device,
                        static_cast<std::size_t>(outNodes));
                    auto probeAcc = alpaka::onHost::allocUnified<double>(device, std::size_t{2u});
                    alpaka::onHost::memset(queue, probeAcc, std::uint8_t{0});
                    auto const fldStart = Clock::now();
                    for(int c = 0; c < 6; ++c)
                    {
                        if(parSize > 1 && (c % parSize) != parRank)
                            continue; // another rank owns this component
                        auto const kernelStart = Clock::now();
                        queue.enqueue(
                            alpaka::onHost::FrameSpec{
                                ceilDivVec(Extents3{Mx, My, Mz}, Extents3::fill(4u)),
                                Extents3::fill(4u)},
                            MomentKernelBuildKernel<3u>{},
                            kernelPlanesView,
                            0u,
                            Mz,
                            Mx,
                            My,
                            Mz,
                            c,
                            alpaka::Vec{
                                kYeeOffset[c][0],
                                kYeeOffset[c][1],
                                kYeeOffset[c][2]},
                            alpaka::Vec{cell[0], cell[1], cell[2]},
                            kRefinement,
                            alpaka::Vec{u0[0], u0[1], u0[2]});
                        alpaka::onHost::wait(queue);
                        times.kernelMs += elapsedMs(kernelStart);

                        alpaka::onHost::memset(queue, rhoSpec, std::uint8_t{0});
                        alpaka::onHost::memset(queue, o3Spec, std::uint8_t{0});
                        auto const specStart = Clock::now();
                        for(auto const& mi : moments)
                        {
                            bool const inFull
                                = static_cast<std::size_t>(mi.tot) <= orderUsed;
                            bool const inOrder3 = mi.tot == 3;
                            if(!inFull && !inOrder3)
                                continue;
                            alpaka::fft::onHost::executeForward(
                                queue,
                                r2cPlan,
                                planeView(static_cast<std::size_t>(mi.idx)),
                                kernelSpec);
                            if(inFull)
                                queue.enqueue(
                                    alpaka::onHost::FrameSpec{(spectrumCount + 63u) / 64u, 64u},
                                    SpectrumAccumulateKernel{},
                                    packedSpectrum(reinterpret_cast<double*>(rhoSpec.data())),
                                    packedSpectrum(reinterpret_cast<double*>(
                                        momentBufs[static_cast<std::size_t>(mi.idx)].data())),
                                    packedSpectrum(reinterpret_cast<double*>(kernelSpec.data())),
                                    spectrumCount);
                            // The order-3 remainder is needed for the probe even
                            // when the delivered order is 2.
                            if(inOrder3)
                            {
                                queue.enqueue(
                                    alpaka::onHost::FrameSpec{(spectrumCount + 63u) / 64u, 64u},
                                    SpectrumAccumulateKernel{},
                                    packedSpectrum(reinterpret_cast<double*>(o3Spec.data())),
                                    packedSpectrum(reinterpret_cast<double*>(
                                        momentBufs[static_cast<std::size_t>(mi.idx)].data())),
                                    packedSpectrum(reinterpret_cast<double*>(kernelSpec.data())),
                                    spectrumCount);
                            }
                        }
                        alpaka::onHost::wait(queue);
                        // a-posteriori probe (k-space): order-3 remainder against
                        // the order-`orderUsed` field.  Explicit orders use the
                        // same probe so they cannot silently return an
                        // unconverged series.
                        {
                            double const* const f
                                = reinterpret_cast<double const*>(rhoSpec.data());
                            double const* const o
                                = reinterpret_cast<double const*>(o3Spec.data());
                            queue.enqueue(
                                alpaka::onHost::FrameSpec{1u, 1u},
                                detail::SpectrumNormKernel{},
                                f,
                                o,
                                spectrumCount,
                                probeAcc.data());
                        }
                        alpaka::fft::onHost::executeBackward(queue, c2rPlan, rhoSpec, outReal);
                        alpaka::onHost::wait(queue);
                        times.fftMs += elapsedMs(specStart);

                        auto const readoutStart = Clock::now();
                        double const factor = scale * (c < 3 ? eFactor : bFactor);
                        std::uint32_t const outNz = static_cast<std::uint32_t>(grid.n[2]);
                        std::uint32_t const outNy = static_cast<std::uint32_t>(grid.n[1]);
                        std::uint32_t const outNx = static_cast<std::uint32_t>(grid.n[0]);
                        // Element-wise read-out as an alpaka host algorithm.  The
                        // compact view of the padded c2r output uses the padded
                        // x/y pitches, so the transform visits exactly the
                        // (k, j, i) subregion the former loop did.  Each output
                        // element sees the same two doubles and the same
                        // ``acc + in * factor`` as before; the functor is
                        // deliberately SIMD-safe, so the tests pin a <= 1 ulp /
                        // <= 2e-15 tolerance instead of bit-exactness.
                        auto const readoutExtents
                            = alpaka::Vec<std::uint32_t, 3u>{outNz, outNy, outNx};
                        auto const outPitches = alpaka::Vec<std::size_t, 3u>{
                            static_cast<std::size_t>(outNx) * outNy * sizeof(double),
                            static_cast<std::size_t>(outNx) * sizeof(double),
                            sizeof(double)};
                        auto const inPitches = alpaka::Vec<std::size_t, 3u>{
                            static_cast<std::size_t>(Mx) * My * sizeof(double),
                            static_cast<std::size_t>(Mx) * sizeof(double),
                            sizeof(double)};
                        auto outView = alpaka::makeMdSpan(
                            readoutAcc.data(),
                            readoutExtents,
                            outPitches);
                        auto inView = alpaka::makeMdSpan(outReal.data(), readoutExtents, inPitches);
                        alpaka::onHost::memset(queue, readoutAcc, std::uint8_t{0});
                        alpaka::onHost::transform(
                            queue,
                            outView,
                            ReadoutScaleAdd{factor},
                            outView,
                            inView);
                        alpaka::onHost::wait(queue);
                        alpaka::onHost::memcpy(queue, vf.data[c], readoutAcc);
                        alpaka::onHost::wait(queue);
                        times.readoutMs += elapsedMs(readoutStart);
                    }

                    // Convergence probe: ||F_3|| / ||F_0..F_orderUsed||.  For
                    // order 2 it estimates the next term; for order 3 it is the
                    // existing order-2-vs-3 k-space probe.  Auto and explicit
                    // share it.
                    alpaka::onHost::wait(queue);
                    double probe = 0.0;
                    {
                        double normSum[2] = {probeAcc.data()[0], probeAcc.data()[1]};
                        if(parSize > 1)
                            par->allreduceSum(normSum, 2);
                        double const fullN = std::sqrt(normSum[0]);
                        double const o3N = std::sqrt(normSum[1]);
                        probe = (fullN > 0.0) ? (o3N / fullN) : 0.0;
                        rep.probe = probe;
                    }
                    // Every rank owns the same field after the reduction, so the
                    // finiteness verdict and the fallback decision agree.
                    if(parSize > 1)
                        for(int c = 0; c < 6; ++c)
                            par->allreduceSum(vf.data[c].data(), vf.data[c].size());
                    bool nonFinite = false;
                    for(int c = 0; c < 6 && !nonFinite; ++c)
                        for(double const value : vf.data[c])
                            if(!std::isfinite(value))
                            {
                                nonFinite = true;
                                break;
                            }
                    bool const probeConverged = (probe < kMomentAutoTol);
                    bool const forced = vm.mode == VelocityMomentsOptions::Mode::Explicit
                        && vm.allowUnconverged;
                    if(nonFinite)
                    {
                        // A non-finite result is never delivered, not even with
                        // the diagnostic escape hatch.
                        rep.nonFinite = true;
                        rep.fellBack = true;
                        std::fprintf(
                            stderr,
                            "picBunch: warning: velocity-moment field contains non-finite "
                            "values (probe %.3e); falling back to the binned path\n",
                            probe);
                    }
                    else if(probeConverged || forced)
                    {
                        rep.used = true;
                        rep.order = static_cast<int>(orderUsed);
                        if(!probeConverged)
                        {
                            rep.allowUnconverged = true;
                            std::fprintf(
                                stderr,
                                "picBunch: warning: velocity-moment order %zu forced with "
                                "probe %.3e above the %.3f budget "
                                "(--velocity-moments-allow-unconverged); the delivered "
                                "field is not converged\n",
                                orderUsed,
                                probe,
                                kMomentAutoTol);
                        }
                        vmField = std::move(vf);
                    }
                    else
                    {
                        rep.fellBack = true;
                        std::fprintf(
                            stderr,
                            "picBunch: velocity-moment order %zu probe %.3e exceeds the "
                            "%.3f budget; falling back to the binned path (raise "
                            "--tolerance or --max-velocity-bins, or pass "
                            "--velocity-moments-allow-unconverged for a diagnostic field)\n",
                            orderUsed,
                            probe,
                            kMomentAutoTol);
                    }
                }
            }
            if(vmField)
                return *vmField;
        }

        if(useDistributed)
        {
            detail::DistributedFft3D dfft{device, queue, Mx, My, Mz, par};
            std::uint32_t const nz = dfft.zCount();
            std::uint32_t const zOffset = dfft.zOffset();
            auto* workReal = reinterpret_cast<double*>(dfft.work());
            Extents3 const slabExt{Mx, My, nz};
            Extents3 const slabFrame = Extents3::fill(4u);
            // The work buffer is a x-fastest Complex z-slab; both the kernel
            // build (real part) and the gather (real part) index it as a 3-D span.
            auto const slabWorkExtents = alpaka::Vec<std::uint32_t, 3u>{nz, My, Mx};
            auto const slabWorkPitches = alpaka::Vec<std::size_t, 3u>{
                static_cast<std::size_t>(Mx) * My * sizeof(Complex),
                static_cast<std::size_t>(Mx) * sizeof(Complex),
                sizeof(Complex)};
            auto slabWorkView = alpaka::makeMdSpan(
                dfft.work(),
                slabWorkExtents,
                slabWorkPitches);
            using SlabWorkView = std::remove_cvref_t<decltype(slabWorkView)>;
            using SlabWorkViewExtents = typename MdSpanExtents<SlabWorkView>::type;
            static_assert(std::is_same_v<typename SlabWorkViewExtents::value_type, std::uint32_t>);

            // The inverse transform leaves a z-slab of the *padded* grid.  Only
            // the output planes [0, grid.n[2]) are written, and those padded
            // planes are not balanced across ranks (padding 2 leaves the whole
            // output on the first half of the ranks).  Redistribute the
            // read-out output once per (bin, component) so every rank owns the
            // same contiguous z-slab the chunked writer writes.  No rank ever
            // allocates or gathers the full output field.
            std::uint32_t const outNx = static_cast<std::uint32_t>(grid.n[0]);
            std::uint32_t const outNy = static_cast<std::uint32_t>(grid.n[1]);
            std::uint32_t const outNz = static_cast<std::uint32_t>(grid.n[2]);
            std::size_t const planeOut
                = static_cast<std::size_t>(outNx) * static_cast<std::size_t>(outNy);

            std::size_t globalNz = grid.n[2];
            std::array<double, 3> writeOrigin = grid.origin;
            if(options.globalGridSize)
            {
                globalNz = (*options.globalGridSize)[2];
                writeOrigin = options.globalGridOrigin.value_or(
                    std::array<double, 3>{0.0, 0.0, 0.0});
            }
            long long const base2 = static_cast<long long>(std::llround(
                (grid.origin[2] - writeOrigin[2]) / grid.spacing[2]));
            auto const writerRange = [&](int r) {
                std::size_t const sz = static_cast<std::size_t>(r);
                return std::pair<long long, long long>{
                    static_cast<long long>((sz * globalNz) / static_cast<std::size_t>(parSize)),
                    static_cast<long long>(
                        ((sz + 1u) * globalNz) / static_cast<std::size_t>(parSize))};
            };
            auto const writerOwner = [&](long long globalZ) {
                for(int r = 0; r < parSize; ++r)
                {
                    auto const range = writerRange(r);
                    if(globalZ >= range.first && globalZ < range.second)
                        return r;
                }
                return parSize - 1;
            };

            auto const ownRange = writerRange(parRank);
            long long const ownLo = std::max(ownRange.first, base2);
            long long const ownHi = std::min(
                ownRange.second, base2 + static_cast<long long>(outNz));
            field.zSlab = true;
            field.slabOffset = (ownHi > ownLo)
                ? static_cast<std::size_t>(ownLo - base2)
                : 0u;
            field.slabCount = (ownHi > ownLo)
                ? static_cast<std::size_t>(ownHi - ownLo)
                : 0u;
            for(auto& component : field.data)
                component.assign(field.slabCount * planeOut, 0.0);

            // Padded output planes that this rank received after the inverse.
            long long const localLo = static_cast<long long>(zOffset);
            long long const localHi = std::min(
                static_cast<long long>(zOffset) + static_cast<long long>(nz),
                static_cast<long long>(outNz));
            // Read-out staging for this rank's local slab.  The gather-scale
            // kernel writes a compact x-fastest device buffer; one D2H copy
            // moves it into pinned host memory and the redistribution below
            // then touches only that pinned buffer.  Before this, the host
            // redistribution dereferenced the unified ``dfft.work()`` slab and
            // paid a managed-page migration on every first touch.  Allocate at
            // least one element so a rank without output planes still owns a
            // valid (never copied) buffer.
            std::uint32_t const localCount = (localHi > localLo)
                ? static_cast<std::uint32_t>(localHi - localLo)
                : 0u;
            std::size_t const stagedCount = planeOut * static_cast<std::size_t>(localCount);
            std::size_t const stagedCapacity = std::max<std::size_t>(stagedCount, 1u);
            auto slabStagingDev = alpaka::onHost::alloc<double>(device, stagedCapacity);
            auto slabStagingHost = alpaka::onHost::allocMapped<double>(device, stagedCapacity);

            std::vector<int> sendCounts(static_cast<std::size_t>(parSize), 0);
            std::vector<int> sendDispls(static_cast<std::size_t>(parSize), 0);
            std::vector<int> recvCounts(static_cast<std::size_t>(parSize), 0);
            std::vector<int> recvDispls(static_cast<std::size_t>(parSize), 0);

            long long sendTotal = 0;
            for(long long k = localLo; k < localHi; ++k)
            {
                int const dest = writerOwner(base2 + k);
                sendCounts[static_cast<std::size_t>(dest)]
                    += static_cast<int>(planeOut);
                sendTotal += static_cast<long long>(planeOut);
            }
            long long sendPos = 0;
            for(int r = 0; r < parSize; ++r)
            {
                sendDispls[static_cast<std::size_t>(r)] = static_cast<int>(sendPos);
                sendPos += sendCounts[static_cast<std::size_t>(r)];
            }
            long long recvPos = 0;
            for(int r = 0; r < parSize; ++r)
            {
                auto const z = detail::slabRange(Mz, parSize, r);
                long long const srcLo = std::min<long long>(z.first, outNz);
                long long const srcHi = std::min<long long>(
                    static_cast<long long>(z.first) + static_cast<long long>(z.second),
                    static_cast<long long>(outNz));
                long long const lo = std::max(srcLo, static_cast<long long>(field.slabOffset));
                long long const hi = std::min(
                    srcHi,
                    static_cast<long long>(field.slabOffset)
                        + static_cast<long long>(field.slabCount));
                long long const planes = std::max(0ll, hi - lo);
                recvCounts[static_cast<std::size_t>(r)]
                    = static_cast<int>(planes * static_cast<long long>(planeOut));
                recvDispls[static_cast<std::size_t>(r)] = static_cast<int>(recvPos);
                recvPos += static_cast<long long>(planes) * static_cast<long long>(planeOut);
            }
            if(sendTotal > static_cast<long long>(std::numeric_limits<int>::max())
               || recvPos > static_cast<long long>(std::numeric_limits<int>::max()))
                throw std::runtime_error(
                    "computeField: an output-slab redistribution exceeds the MPI int count range");

            std::vector<double> sendBuffer(static_cast<std::size_t>(sendTotal));
            std::vector<double> recvBuffer;
            double dummy = 0.0;
            std::array<bool, 6> componentWritten{};
            auto const redistribute = [&](int c, bool first) {
                std::vector<int> cursor = sendDispls;
                for(long long k = localLo; k < localHi; ++k)
                {
                    int const dest = writerOwner(base2 + k);
                    long long const localZ = k - static_cast<long long>(zOffset);
                    double* dst = sendBuffer.data() + cursor[static_cast<std::size_t>(dest)];
                    // The pinned staging buffer is already compact and x-fastest,
                    // so a plane is one contiguous block.  This copies the exact
                    // values the former host loop read from the scaled slab.
                    double const* staged = slabStagingHost.data()
                        + static_cast<std::size_t>(localZ) * planeOut;
                    for(std::size_t l = 0; l < planeOut; ++l)
                        dst[l] = staged[l];
                    cursor[static_cast<std::size_t>(dest)] += static_cast<int>(planeOut);
                }
                double* recv = first && !field.data[c].empty()
                    ? field.data[c].data()
                    : (recvBuffer.empty() ? &dummy : recvBuffer.data());
                par->alltoallv(
                    sendBuffer.empty() ? &dummy : sendBuffer.data(),
                    sendCounts.data(),
                    sendDispls.data(),
                    recv,
                    recvCounts.data(),
                    recvDispls.data());
                if(!first)
                    for(std::size_t l = 0; l < field.data[c].size(); ++l)
                        field.data[c][l] += recvBuffer[l];
            };

            for(std::size_t b = 0; b < binCount; ++b)
            {
                if(members[b].empty())
                    continue;
                if(members[b].size() > 0xffffffffu)
                    throw std::runtime_error("computeField: particle count exceeds uint32 range");

                // -------------------------------------------------------- deposit
                auto const depositStart = Clock::now();
                if(!planCaptured)
                {
                    times.planMs = elapsedMs(callStart);
                    planCaptured = true;
                }
                dfft.clearWork();
                if(options.deterministicDeposit)
                    depositHost(members[b], workReal, 2u, zOffset, nz);
                else
                    depositDevice(
                        members[b],
                        alpaka::makeMdSpan(
                            dfft.work(),
                            alpaka::Vec<std::uint32_t, 3u>{nz, My, Mx}),
                        zOffset,
                        nz);
                times.depositMs += elapsedMs(depositStart);

                // ------------------------------------------- rho forward transform
                dfft.forwardWorkToSpectrum();

                std::array<double, 3> const uRep{
                    binning.representativeU[b][0],
                    binning.representativeU[b][1],
                    binning.representativeU[b][2]};

                for(int c = 0; c < 6; ++c)
                {
                    // -------------------------------------------------------- kernel
                    auto const kernelStart = Clock::now();
                    queue.enqueue(
                        alpaka::onHost::FrameSpec{
                            ceilDivVec(slabExt, slabFrame),
                            slabFrame},
                        KernelBuildKernel{},
                        slabWorkView,
                        zOffset,
                        nz,
                        Mx,
                        My,
                        Mz,
                        c,
                        alpaka::Vec{
                            kYeeOffset[c][0],
                            kYeeOffset[c][1],
                            kYeeOffset[c][2]},
                        alpaka::Vec{cell[0], cell[1], cell[2]},
                        kRefinement,
                        alpaka::Vec{uRep[0], uRep[1], uRep[2]});
                    alpaka::onHost::wait(queue);
                    times.kernelMs += elapsedMs(kernelStart);

                    // ------------------------------- kernel forward + product + inverse
                    dfft.forwardWorkToKernelSpectrum();
                    dfft.multiply();
                    dfft.inverseToWork();

                    // ---------------------------------------------------------- read-out
                    auto const readoutStart = Clock::now();
                    double const factor = scale * (c < 3 ? eFactor : bFactor);
                    // Gather and scale the slab on the device into the compact
                    // staging buffer, then hand each output z plane to the rank
                    // that owns it in the writer layout.  A single D2H copy into
                    // pinned host memory precedes the host redistribution, which
                    // therefore never dereferences the unified FFT slab.
                    if(localCount > 0u)
                    {
                        queue.enqueue(
                            alpaka::onHost::FrameSpec{
                                static_cast<std::uint32_t>((stagedCount + 63u) / 64u),
                                64u},
                            SlabGatherScaleKernel{},
                            slabWorkView,
                            alpaka::makeMdSpan(
                                slabStagingDev.data(),
                                alpaka::Vec<std::uint32_t, 1u>{
                                    static_cast<std::uint32_t>(stagedCapacity)}),
                            outNx,
                            outNy,
                            localCount,
                            factor);
                        alpaka::onHost::wait(queue);
                        alpaka::onHost::memcpy(queue, slabStagingHost, slabStagingDev);
                        alpaka::onHost::wait(queue);
                    }
                    bool const first = !componentWritten[static_cast<std::size_t>(c)];
                    if(!first && recvBuffer.size() != field.data[c].size())
                        recvBuffer.assign(field.data[c].size(), 0.0);
                    redistribute(c, first);
                    componentWritten[static_cast<std::size_t>(c)] = true;
                    times.readoutMs += elapsedMs(readoutStart);
                }
            }
            times.fftMs += dfft.fftMilliseconds();
            times.transposeMs += dfft.transposeMilliseconds();
            return field;
        }

        // Replicated path: one whole FFT per rank, component-parallel across ranks.
        for(auto& component : field.data)
            component.assign(grid.size(), 0.0);

        // Read-out staging exists only on this path; the distributed branch above
        // returns before reaching it, so a rank that never gathers allocates nothing.
        auto outStagingDev = alpaka::onHost::alloc<double>(device, outNodes);
        // One device accumulation target per component.  The element-wise
        // read-out runs as an alpaka onHost::transform on the queue; the host
        // vectors are filled by a single copy after the bin loop.
        using FieldAccBuffer = decltype(alpaka::onHost::alloc<double>(device, outNodes));
        std::vector<FieldAccBuffer> fieldAccs;
        fieldAccs.reserve(6u);
        for(int c = 0; c < 6; ++c)
        {
            fieldAccs.push_back(alpaka::onHost::alloc<double>(device, outNodes));
            alpaka::onHost::memset(queue, fieldAccs.back(), std::uint8_t{0});
        }

        // Replicated path: one whole FFT per rank, component-parallel across ranks.
        for(std::size_t b = 0; b < binCount; ++b)
        {
            if(members[b].empty())
                continue;
            if(members[b].size() > 0xffffffffu)
                throw std::runtime_error("computeField: particle count exceeds uint32 range");

            // -------------------------------------------------------- deposit
            auto const depositStart = Clock::now();
            if(!planCaptured)
            {
                times.planMs = elapsedMs(callStart);
                planCaptured = true;
            }
            alpaka::onHost::memset(queue, rhoReal, std::uint8_t{0});
            if(options.deterministicDeposit)
            {
                // The memset is queued on the device; finish it before the host
                // touches the same unified buffer.
                alpaka::onHost::wait(queue);
                depositHost(members[b], rhoReal.data(), 1u, 0u, Mz);
            }
            else
            {
                alpaka::onHost::wait(queue);
                depositDevice(
                    members[b],
                    alpaka::makeMdSpan(
                        rhoReal.data(),
                        alpaka::Vec<std::uint32_t, 3u>{Mz, My, Mx}),
                    0u,
                    Mz);
            }
            times.depositMs += elapsedMs(depositStart);

            // ------------------------------------------------------------- FFT
            auto const fftStart = Clock::now();
            alpaka::fft::onHost::executeForward(queue, r2cPlan, rhoReal, rhoSpec);
            alpaka::onHost::wait(queue);
            times.fftMs += elapsedMs(fftStart);

            auto kernelRealView = alpaka::makeMdSpan(
                kernelReal.data(),
                alpaka::Vec<std::uint32_t, 3u>{Mz, My, Mx});
            auto outRealView = alpaka::makeMdSpan(
                outReal.data(),
                alpaka::Vec<std::uint32_t, 3u>{Mz, My, Mx});
            using OutRealView = std::remove_cvref_t<decltype(outRealView)>;
            using OutRealViewExtents = typename MdSpanExtents<OutRealView>::type;
            static_assert(std::is_same_v<typename OutRealViewExtents::value_type, std::uint32_t>);

            std::array<double, 3> const uRep{
                binning.representativeU[b][0],
                binning.representativeU[b][1],
                binning.representativeU[b][2]};

            for(int c = 0; c < 6; ++c)
            {
                if(parSize > 1 && (c % parSize) != parRank)
                    continue; // another rank owns this component
                // ------------------------------------------------------------ kernel
                auto const kernelStart = Clock::now();
                queue.enqueue(
                    alpaka::onHost::FrameSpec{
                        ceilDivVec(Extents3{Mx, My, Mz}, Extents3::fill(4u)),
                        Extents3::fill(4u)},
                    KernelBuildKernel{},
                    kernelRealView,
                    0u,
                    Mz,
                    Mx,
                    My,
                    Mz,
                    c,
                    alpaka::Vec{
                        kYeeOffset[c][0],
                        kYeeOffset[c][1],
                        kYeeOffset[c][2]},
                    alpaka::Vec{cell[0], cell[1], cell[2]},
                    kRefinement,
                    alpaka::Vec{uRep[0], uRep[1], uRep[2]});
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
                    packedSpectrum(reinterpret_cast<double*>(kernelSpec.data())),
                    packedSpectrum(reinterpret_cast<double*>(rhoSpec.data())),
                    spectrumCount);
                alpaka::onHost::wait(queue);
                alpaka::fft::onHost::executeBackward(queue, c2rPlan, kernelSpec, outReal);
                alpaka::onHost::wait(queue);
                times.fftMs += elapsedMs(spectrumStart);

                // ---------------------------------------------------------- read-out
                auto const readoutStart = Clock::now();
                double const factor = scale * (c < 3 ? eFactor : bFactor);
                // Every (k,j,i) owns one output node, so the gather keeps exactly
                // one deterministic write per node.  The padded c2r output is
                // gathered on the device into a compact n^3 buffer (pure copy,
                // no arithmetic); the accumulate is an alpaka onHost::transform
                // on the same compact device buffer.  Per node it performs the
                // same ``acc + staged * factor`` as the former host loop; the
                // functor is deliberately SIMD-safe, so the tests pin a
                // <= 1 ulp / <= 2e-15 tolerance, independent of executor threads.
                std::uint32_t const outCount = static_cast<std::uint32_t>(outNodes);
                auto const compactExtents = alpaka::Vec<std::uint32_t, 1u>{outCount};
                static_assert(std::is_same_v<
                              std::remove_cvref_t<decltype(compactExtents)>::value_type,
                              std::uint32_t>);
                queue.enqueue(
                    alpaka::onHost::FrameSpec{(outCount + 63u) / 64u, 64u},
                    detail::CompactGatherKernel{},
                    outRealView,
                    alpaka::makeMdSpan(outStagingDev.data(), compactExtents),
                    outNx,
                    outNy,
                    outNz);
                auto accView = alpaka::makeMdSpan(
                    fieldAccs[static_cast<std::size_t>(c)].data(),
                    compactExtents);
                auto stagedView = alpaka::makeMdSpan(outStagingDev.data(), compactExtents);
                alpaka::onHost::transform(
                    queue,
                    accView,
                    ReadoutScaleAdd{factor},
                    accView,
                    stagedView);
                alpaka::onHost::wait(queue);
                times.readoutMs += elapsedMs(readoutStart);
            }
        }

        // Materialise the device accumulators into the host field, once.  The
        // wait below is the queue/host boundary: after it no in-flight device
        // work can still be writing the read-out result.
        for(int c = 0; c < 6; ++c)
            alpaka::onHost::memcpy(queue, field.data[c], fieldAccs[static_cast<std::size_t>(c)]);
        alpaka::onHost::wait(queue);

        if(parSize > 1)
            for(int c = 0; c < 6; ++c)
                par->allreduceSum(field.data[c].data(), field.data[c].size());

        return field;
    }
} // namespace picBunch
