// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/Parallel.hpp"
#include "picBunch/detail/FftPlanCache.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/fft.hpp>
#include <alpaka/math/Complex.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace picBunch
{
    namespace detail
    {
        /*! Largest per-rank padded node count the distributed FFT accepts.
         *
         * This comes from the vendor FFT element ceiling, not from any byte
         * offset: cuFFT documents 4 G elements per transform, i.e. 2^32 complex
         * nodes, memory permitting.  The pinned alpakaVendor calls
         * cufftMakePlanMany64 with long long parameters and rocFFT with size_t,
         * so no plan value narrows and no vendor-side check is needed here.  All
         * offsets, counts and pitches in this path are 64-bit; the guard only
         * refuses a size past the vendor ceiling.
         */
        inline constexpr std::size_t kMaxDistributedNodesPerRank = std::size_t{1} << 32;

        //! Balanced contiguous split of ``total`` items over ``size`` ranks.
        inline std::pair<std::uint32_t, std::uint32_t> slabRange(
            std::uint32_t total,
            int size,
            int rank)
        {
            auto const n = static_cast<std::uint32_t>(size);
            auto const r = static_cast<std::uint32_t>(rank);
            std::uint32_t const base = total / n;
            std::uint32_t const remainder = total % n;
            std::uint32_t const begin = r * base + std::min(r, remainder);
            std::uint32_t const count = base + (r < remainder ? 1u : 0u);
            return {begin, count};
        }

        /*! Distributed 3-D complex-to-complex FFT on a z-slab decomposition.
         *
         * The padded grid is split into contiguous z-slabs.  Each rank holds all
         * x and y nodes for its z planes, performs the local x and y 1-D
         * transforms, and then transposes the data with ``MPI_Alltoallv`` so
         * that the z axis becomes local.  The z transform, the multiplication
         * with the kernel spectrum and the inverse z transform follow in that
         * layout; a second transpose and the inverse y/x transforms return the
         * real-space slab.  This is an exact global transform, not a per-rank
         * transform on a truncated subdomain.
         *
         * The class owns four complex work arrays, each at most ``Mx*My*Mz/nRank``
         * complex values (plus remainder imbalance).  The caller fills
         * ``work()`` with the real-space slab (real part; imaginary part zero),
         * calls ``forwardWorkToSpectrum()``, or the kernel variant, multiplies
         * the two spectra, and calls ``inverseToWork()``.
         */
        template<typename T_Device, typename T_Queue>
        class DistributedFft3D
        {
        public:
            using Complex = alpaka::math::Complex<double>;
            using Extents1 = alpaka::Vec<std::uint32_t, 1u>;
            using Buffer = decltype(alpaka::onHost::allocUnified<Complex>(
                std::declval<T_Device&>(),
                std::size_t{1u}));
            using Plan = decltype(alpaka::fft::onHost::PlanBuilder<Complex, Extents1>{Extents1{1u}}
                                      .c2c()
                                      .inPlace()
                                      .build(std::declval<T_Device&>()));

            DistributedFft3D(
                T_Device& device,
                T_Queue& queue,
                std::uint32_t Mx,
                std::uint32_t My,
                std::uint32_t Mz,
                ParallelContext const* parallel)
                : m_queue{queue}
                , m_parallel{parallel}
                , m_Mx{Mx}
                , m_My{My}
                , m_Mz{Mz}
            {
                if(parallel == nullptr || parallel->size < 1)
                    throw std::invalid_argument(
                        "DistributedFft3D requires a ParallelContext");
                auto const z = slabRange(Mz, parallel->size, parallel->rank);
                auto const x = slabRange(Mx, parallel->size, parallel->rank);
                m_z0 = z.first;
                m_nz = z.second;
                m_x0 = x.first;
                m_nx = x.second;
                if(m_nz == 0u || m_nx == 0u)
                    throw std::runtime_error(
                        "DistributedFft3D: more ranks than padded planes on an axis");

                std::size_t const workSize
                    = static_cast<std::size_t>(Mx) * My * m_nz;
                std::size_t const spectrumSize
                    = static_cast<std::size_t>(m_nx) * My * Mz;
                m_chunkSize = std::max(
                    static_cast<std::size_t>(Mx) * m_nz,
                    static_cast<std::size_t>(Mz) * m_nx);
                m_transposeWindow = resolveTransposeWindow(
                    static_cast<std::size_t>(parallel->size));

                // Hard safety net around the vendor element ceiling (see
                // kMaxDistributedNodesPerRank).  Nothing below this guard needs
                // it: the arithmetic is 64-bit and the MPI transfer is chunked,
                // so only a slab past the vendor ceiling is refused.
                if(workSize > kMaxDistributedNodesPerRank
                   || spectrumSize > kMaxDistributedNodesPerRank)
                    throw std::runtime_error(
                        "DistributedFft3D: rank " + std::to_string(parallel->rank) + " of "
                        + std::to_string(parallel->size) + " requires "
                        + std::to_string(std::max(workSize, spectrumSize))
                        + " padded nodes per rank (work " + std::to_string(workSize)
                        + ", spectrum " + std::to_string(spectrumSize)
                        + "), above the vendor element ceiling of "
                        + std::to_string(kMaxDistributedNodesPerRank) + " nodes per transform");

                // The plan batch is stored in the 32-bit extent value type used
                // here.  With the guard above, Mx >= 2 and Mz >= 2 keep both
                // batches below 2^31, so this cast cannot truncate.
                auto const batchX = static_cast<std::uint32_t>(
                    static_cast<std::uint64_t>(My) * m_nz);
                auto const batchZ = static_cast<std::uint32_t>(
                    static_cast<std::uint64_t>(m_nx) * My);

                // The scalar std::size_t count already creates a Vec<size_t,1>
                // extent, so these buffers and their pitches are 64-bit.
                m_work.emplace(alpaka::onHost::allocUnified<Complex>(device, workSize));
                m_send.emplace(alpaka::onHost::allocUnified<Complex>(device, m_chunkSize));
                m_recv.emplace(alpaka::onHost::allocUnified<Complex>(device, m_chunkSize));
                m_spectrum.emplace(alpaka::onHost::allocUnified<Complex>(device, spectrumSize));
                m_kernel.emplace(alpaka::onHost::allocUnified<Complex>(device, spectrumSize));

                // Every plan is keyed by the complete layout.  The x/y/z
                // transforms use different extents, batch counts, strides and
                // distances, and the replicated path's full-grid plans must
                // never be reused here.  A key mismatch builds a new plan;
                // alternating layouts therefore rebuilds instead of failing
                // with a vendor execution error.
                auto const shape = std::array<std::uint32_t, 3u>{Mx, My, Mz};
                auto const slab
                    = std::array<std::uint32_t, 4u>{m_nx, m_nz, m_z0, static_cast<std::uint32_t>(parallel->size)};
                auto const zeroStride = alpaka::Vec<std::size_t, 1u>{};
                auto const xStride = alpaka::Vec<std::size_t, 1u>{Mx * sizeof(Complex)};
                auto const yStride
                    = alpaka::Vec<std::size_t, 1u>{static_cast<std::size_t>(m_nx) * My * sizeof(Complex)};
                auto const xKey = detail::makeFftPlanKey<Complex>(
                    device,
                    alpaka::fft::Transform::c2c,
                    Extents1{Mx},
                    batchX,
                    zeroStride,
                    zeroStride,
                    Mx * sizeof(Complex),
                    Mx * sizeof(Complex),
                    alpaka::fft::Placement::inPlace,
                    detail::FftLayoutKind::DistributedSlab,
                    shape,
                    slab);
                auto const yKey = detail::makeFftPlanKey<Complex>(
                    device,
                    alpaka::fft::Transform::c2c,
                    Extents1{My},
                    static_cast<std::uint32_t>(Mx),
                    xStride,
                    xStride,
                    sizeof(Complex),
                    sizeof(Complex),
                    alpaka::fft::Placement::inPlace,
                    detail::FftLayoutKind::DistributedSlab,
                    shape,
                    slab);
                auto const zKey = detail::makeFftPlanKey<Complex>(
                    device,
                    alpaka::fft::Transform::c2c,
                    Extents1{Mz},
                    batchZ,
                    yStride,
                    yStride,
                    sizeof(Complex),
                    sizeof(Complex),
                    alpaka::fft::Placement::inPlace,
                    detail::FftLayoutKind::DistributedSlab,
                    shape,
                    slab);
                m_xPlan = acquirePlan(
                    device,
                    xKey,
                    [&] {
                        return alpaka::fft::onHost::PlanBuilder<Complex, Extents1>{Extents1{Mx}}
                            .c2c()
                            .inPlace()
                            .batch(batchX)
                            .distances(Mx * sizeof(Complex), Mx * sizeof(Complex))
                            .build(device);
                    });
                m_yPlan = acquirePlan(
                    device,
                    yKey,
                    [&] {
                        return alpaka::fft::onHost::PlanBuilder<Complex, Extents1>{Extents1{My}}
                            .c2c()
                            .inPlace()
                            .batch(Mx)
                            .strides(Mx * sizeof(Complex), Mx * sizeof(Complex))
                            .distances(sizeof(Complex), sizeof(Complex))
                            .build(device);
                    });
                m_zPlan = acquirePlan(
                    device,
                    zKey,
                    [&] {
                        return alpaka::fft::onHost::PlanBuilder<Complex, Extents1>{Extents1{Mz}}
                            .c2c()
                            .inPlace()
                            .batch(batchZ)
                            .strides(
                                static_cast<std::size_t>(m_nx) * My * sizeof(Complex),
                                static_cast<std::size_t>(m_nx) * My * sizeof(Complex))
                            .distances(sizeof(Complex), sizeof(Complex))
                            .build(device);
                    });

                buildTransposeCounts();
            }

            [[nodiscard]] std::uint32_t zOffset() const
            {
                return m_z0;
            }
            [[nodiscard]] std::uint32_t zCount() const
            {
                return m_nz;
            }
            [[nodiscard]] std::uint32_t xOffset() const
            {
                return m_x0;
            }
            [[nodiscard]] std::uint32_t xCount() const
            {
                return m_nx;
            }
            [[nodiscard]] std::size_t workSize() const
            {
                return static_cast<std::size_t>(m_Mx) * m_My * m_nz;
            }
            [[nodiscard]] Complex* work()
            {
                return m_work->data();
            }
            [[nodiscard]] Complex* spectrum()
            {
                return m_spectrum->data();
            }
            [[nodiscard]] Complex* kernelSpectrum()
            {
                return m_kernel->data();
            }
            //! Owning view of the local kernel spectrum; keeps the alpaka API
            //! trait that a bare pointer view loses for onHost::memcpy.
            [[nodiscard]] Buffer& kernelSpectrumBuffer()
            {
                return *m_kernel;
            }

            void clearWork()
            {
                alpaka::onHost::memset(m_queue, *m_work, std::uint8_t{0});
                alpaka::onHost::wait(m_queue);
            }

            //! ``work`` (real slab) -> ``spectrum``.
            void forwardWorkToSpectrum()
            {
                forwardTo(*m_spectrum);
            }

            //! ``work`` (real slab) -> ``kernelSpectrum``.
            void forwardWorkToKernelSpectrum()
            {
                forwardTo(*m_kernel);
            }

            //! Element-wise ``spectrum *= kernelSpectrum``.
            void multiply()
            {
                // The per-rank spectrum is up to 2^31 complex nodes, so the
                // element count and the byte pitches must be 64-bit.  With
                // 32-bit pitches the MdSpan byte offset wrapped at 4 GiB,
                // re-multiplying the low spectrum and leaving the high end
                // untouched (the large-grid field corruption).
                std::size_t const count = static_cast<std::size_t>(m_nx) * m_My * m_Mz;
                // Multiply the kernel spectrum in place so the source spectrum
                // survives for the remaining components.
                // Packed two-double layout: {complex index, re/im}. The pitches
                // keep the byte strides (x-fastest, re/im interleaved).
                using PackedIndex2 = alpaka::Vec<std::size_t, 2u>;
                auto const extents = PackedIndex2{count, 2u};
                auto const pitches = alpaka::Vec<std::size_t, 2u>{
                    2u * sizeof(double),
                    sizeof(double)};
                auto a = alpaka::makeMdSpan(
                    reinterpret_cast<double*>(m_kernel->data()),
                    extents,
                    pitches);
                auto b = alpaka::makeMdSpan(
                    reinterpret_cast<double const*>(m_spectrum->data()),
                    extents,
                    pitches);
                static_assert(std::is_same_v<
                              std::remove_cvref_t<decltype(extents)>::value_type,
                              PackedIndex2::value_type>);
                static_assert(std::is_same_v<
                              typename std::remove_cvref_t<
                                  decltype(std::declval<decltype(a) const&>().getExtents())>::value_type,
                              PackedIndex2::value_type>);
                static_assert(std::is_same_v<
                              typename std::remove_cvref_t<
                                  decltype(std::declval<decltype(b) const&>().getExtents())>::value_type,
                              PackedIndex2::value_type>);
                // The MdSpan byte offsets are size_t only if the pitch type is
                // size_t; pin it so a future edit cannot silently narrow it.
                static_assert(std::is_same_v<
                              typename std::remove_cvref_t<decltype(a)>::index_type,
                              std::size_t>);
                static_assert(std::is_same_v<
                              typename std::remove_cvref_t<decltype(b)>::index_type,
                              std::size_t>);
                std::uint32_t const frame = 64u;
                std::size_t const blocks = (count + frame - 1u) / frame;
                if(blocks > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
                    throw std::runtime_error(
                        "DistributedFft3D: spectrum multiply requires "
                        + std::to_string(blocks) + " blocks on rank "
                        + std::to_string(m_parallel->rank)
                        + ", above the 32-bit kernel launch grid");
                m_queue.enqueue(
                    alpaka::onHost::FrameSpec{static_cast<std::uint32_t>(blocks), frame},
                    MultiplyKernel{},
                    a,
                    b,
                    count);
                alpaka::onHost::wait(m_queue);
            }

            //! ``spectrum`` -> ``work`` (real slab in the real part).
            void inverseToWork()
            {
                auto const t0 = Clock::now();
                alpaka::fft::onHost::executeBackward(m_queue, *m_zPlan, *m_kernel, *m_kernel);
                alpaka::onHost::wait(m_queue);
                m_fftMs += elapsed(t0);

                auto const t1 = Clock::now();
                for(std::uint32_t y = 0; y < m_My; ++y)
                    transposeBackward(y);
                m_transposeMs += elapsed(t1);

                auto const t2 = Clock::now();
                executeYPlanes(false);
                alpaka::fft::onHost::executeBackward(m_queue, *m_xPlan, *m_work, *m_work);
                alpaka::onHost::wait(m_queue);
                m_fftMs += elapsed(t2);
            }

            // ---------------------------------------------------- staged exchange
            // The staged chunked path reuses this class unchanged for the local
            // transforms and the pack/unpack arithmetic; only the collective
            // call is replaced.  These entry points expose exactly the pieces
            // that forwardTo()/inverseToWork() use, so the staged path and the
            // MPI path execute the same operations in the same order.

            //! x transform plus the y planes, the local part of the forward pass.
            void forwardLocal()
            {
                auto const t0 = Clock::now();
                alpaka::fft::onHost::executeForward(m_queue, *m_xPlan, *m_work, *m_work);
                executeYPlanes(true);
                alpaka::onHost::wait(m_queue);
                m_fftMs += elapsed(t0);
            }

            //! inverse y planes plus the inverse x transform.
            void inverseLocal()
            {
                auto const t0 = Clock::now();
                executeYPlanes(false);
                alpaka::fft::onHost::executeBackward(m_queue, *m_xPlan, *m_work, *m_work);
                alpaka::onHost::wait(m_queue);
                m_fftMs += elapsed(t0);
            }

            //! z transform of the source (``kernel == false``) or kernel spectrum.
            void forwardZ(bool kernel)
            {
                auto const t0 = Clock::now();
                auto& buffer = kernel ? *m_kernel : *m_spectrum;
                alpaka::fft::onHost::executeForward(m_queue, *m_zPlan, buffer, buffer);
                alpaka::onHost::wait(m_queue);
                m_fftMs += elapsed(t0);
            }

            //! inverse z transform of the product held in the kernel spectrum.
            void inverseZ()
            {
                auto const t0 = Clock::now();
                alpaka::fft::onHost::executeBackward(m_queue, *m_zPlan, *m_kernel, *m_kernel);
                alpaka::onHost::wait(m_queue);
                m_fftMs += elapsed(t0);
            }

            void packForwardSlice(std::uint32_t y)
            {
                packForward(y);
            }
            void unpackForwardSlice(std::uint32_t y)
            {
                unpackForward(y, *m_spectrum);
            }
            void unpackForwardSliceKernel(std::uint32_t y)
            {
                unpackForward(y, *m_kernel);
            }
            void packBackwardSlice(std::uint32_t y)
            {
                packBackward(y);
            }
            void unpackBackwardSlice(std::uint32_t y)
            {
                unpackBackward(y);
            }

            [[nodiscard]] Complex* spectrumData()
            {
                return m_spectrum->data();
            }
            [[nodiscard]] Complex* kernelData()
            {
                return m_kernel->data();
            }
            [[nodiscard]] double* sendDoubles()
            {
                return reinterpret_cast<double*>(m_send->data());
            }
            [[nodiscard]] double* recvDoubles()
            {
                return reinterpret_cast<double*>(m_recv->data());
            }
            //! Doubles in one forward send slice (all destinations).
            [[nodiscard]] std::size_t forwardSendDoubles() const
            {
                return static_cast<std::size_t>(m_Mx) * m_nz * 2u;
            }
            //! Doubles in one backward send slice (all destinations).
            [[nodiscard]] std::size_t backwardSendDoubles() const
            {
                return static_cast<std::size_t>(m_Mz) * m_nx * 2u;
            }
            [[nodiscard]] std::vector<std::size_t> const& recvDisplacements() const
            {
                return m_recvDispls;
            }
            [[nodiscard]] std::vector<std::size_t> const& backRecvDisplacements() const
            {
                return m_backRecvDispls;
            }
            [[nodiscard]] std::vector<std::size_t> const& backRecvCounts() const
            {
                return m_backRecvCounts;
            }

            [[nodiscard]] double fftMilliseconds() const
            {
                return m_fftMs;
            }
            [[nodiscard]] double transposeMilliseconds() const
            {
                return m_transposeMs;
            }

        private:
            template<typename T_Builder>
            static std::shared_ptr<Plan> acquirePlan(
                T_Device& device,
                detail::FftPlanKey const& key,
                T_Builder&& builder)
            {
#if defined(ALPAKAV_DEP_ROCFFT) && ALPAKAV_DEP_ROCFFT
                // The HIP teardown workaround still owns plans per process
                // call; do not put them into the process cache.
                static_cast<void>(device);
                static_cast<void>(key);
                return std::make_shared<Plan>(builder());
#else
                static_cast<void>(device);
                return detail::FftPlanCache<Plan>::acquire(key, std::forward<T_Builder>(builder));
#endif
            }

            //! Resolve the MPI transfer window once per transform.
            /*!
             * The largest window that keeps every count, displacement and the
             * packed window total inside int is INT_MAX / (2 * ranks).
             *
             * PICBUNCH_FFT_TRANSPOSE_WINDOW is a test-only hook: a small value
             * forces many chunk calls on a small grid so the window scheduling
             * is covered without a huge slice.  It is read once here, not per
             * y-plane, and any positive value is correct.
             */
            static std::size_t resolveTransposeWindow(std::size_t ranks)
            {
                std::size_t const intMax
                    = static_cast<std::size_t>(std::numeric_limits<int>::max());
                std::size_t window = intMax / (2u * std::max<std::size_t>(ranks, 1u));
                window = std::max<std::size_t>(window, 1u);
                if(char const* text = std::getenv("PICBUNCH_FFT_TRANSPOSE_WINDOW"))
                {
                    long long const parsed = std::strtoll(text, nullptr, 10);
                    if(parsed > 0)
                        window = std::min<std::size_t>(
                            static_cast<std::size_t>(parsed),
                            window);
                }
                return window;
            }

            using Clock = std::chrono::steady_clock;

            static double elapsed(Clock::time_point const& start)
            {
                return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            }


        public:
            //! Element-wise complex product in the packed two-double layout.
            struct MultiplyKernel
            {
                template<typename TAcc>
                ALPAKA_FN_ACC void operator()(
                    TAcc const& acc,
                    alpaka::concepts::IMdSpan auto a,
                    alpaka::concepts::IMdSpan auto const& b,
                    std::size_t count) const
                {
                    for(auto [l] : alpaka::onAcc::makeIdxMap(
                            acc,
                            alpaka::onAcc::worker::threadsInGrid,
                            alpaka::IdxRange{count}))
                    {
                        auto const re = alpaka::Vec{l, std::size_t{0u}};
                        auto const im = alpaka::Vec{l, std::size_t{1u}};
                        double const ar = a[re];
                        double const ai = a[im];
                        double const br = b[re];
                        double const bi = b[im];
                        a[re] = ar * br - ai * bi;
                        a[im] = ar * bi + ai * br;
                    }
                }
            };

        private:
            //! Apply the y transform plane by plane.
            //!
            //! The y lines of one z plane are interleaved with a constant
            //! distance (x is fastest), but two consecutive planes are not
            //! contiguous, so a single batched plan cannot cover all planes.
            void executeYPlanes(bool forward)
            {
                using SubExtents = alpaka::Vec<std::size_t, 1u>;
                static_assert(std::is_same_v<
                              typename SubExtents::value_type,
                              std::size_t>);
                std::size_t const planeSize = static_cast<std::size_t>(m_Mx) * m_My;
                for(std::uint32_t zl = 0; zl < m_nz; ++zl)
                {
                    auto plane = m_work->getSubView(
                        SubExtents{static_cast<std::size_t>(zl) * planeSize},
                        SubExtents{planeSize});
                    if(forward)
                        alpaka::fft::onHost::executeForward(m_queue, *m_yPlan, plane, plane);
                    else
                        alpaka::fft::onHost::executeBackward(m_queue, *m_yPlan, plane, plane);
                }
            }

            void buildTransposeCounts()
            {
                int const size = m_parallel->size;
                m_sendCounts.assign(static_cast<std::size_t>(size), 0);
                m_sendDispls.assign(static_cast<std::size_t>(size), 0);
                m_recvCounts.assign(static_cast<std::size_t>(size), 0);
                m_recvDispls.assign(static_cast<std::size_t>(size), 0);
                m_backSendCounts.assign(static_cast<std::size_t>(size), 0);
                m_backSendDispls.assign(static_cast<std::size_t>(size), 0);
                m_backRecvCounts.assign(static_cast<std::size_t>(size), 0);
                m_backRecvDispls.assign(static_cast<std::size_t>(size), 0);

                // Counts and displacements are double elements (two per
                // complex node) and stay 64-bit.  MPI_Alltoallv only accepts
                // int, so alltoallvChunked() splits a slice into windows whose
                // offsets still fit int instead of refusing a large slab.
                std::size_t sendPos = 0;
                std::size_t recvPos = 0;
                std::size_t backSendPos = 0;
                std::size_t backRecvPos = 0;
                for(int r = 0; r < size; ++r)
                {
                    auto const z = slabRange(m_Mz, size, r);
                    auto const x = slabRange(m_Mx, size, r);
                    auto const sz = static_cast<std::size_t>(r);
                    // Forward: one y slice, dest r gets all local z and its x range.
                    m_sendCounts[sz] = static_cast<std::size_t>(m_nz) * x.second * 2u;
                    m_sendDispls[sz] = sendPos;
                    sendPos += m_sendCounts[sz];
                    // Receive from r: our local x, all of r's z planes.
                    m_recvCounts[sz] = static_cast<std::size_t>(m_nx) * z.second * 2u;
                    m_recvDispls[sz] = recvPos;
                    recvPos += m_recvCounts[sz];

                    // Backward: dest r owns its z range.
                    m_backSendCounts[sz] = static_cast<std::size_t>(m_nx) * z.second * 2u;
                    m_backSendDispls[sz] = backSendPos;
                    backSendPos += m_backSendCounts[sz];
                    m_backRecvCounts[sz] = static_cast<std::size_t>(x.second) * m_nz * 2u;
                    m_backRecvDispls[sz] = backRecvPos;
                    backRecvPos += m_backRecvCounts[sz];
                }
                std::size_t const forwardSend = static_cast<std::size_t>(m_Mx) * m_nz * 2u;
                std::size_t const forwardRecv = static_cast<std::size_t>(m_nx) * m_Mz * 2u;
                std::size_t const backwardSend = static_cast<std::size_t>(m_nx) * m_Mz * 2u;
                std::size_t const backwardRecv = static_cast<std::size_t>(m_Mx) * m_nz * 2u;
                if(sendPos != forwardSend || recvPos != forwardRecv || backSendPos != backwardSend
                   || backRecvPos != backwardRecv)
                    throw std::runtime_error("DistributedFft3D: inconsistent transpose counts");
            }

            //! ``MPI_Alltoallv`` with 64-bit slices, transferred in int windows.
            /*!
             * Counts and displacements are double elements and may exceed
             * ``INT_MAX``.  MPI_Alltoallv cannot express that in one call, so a
             * large slice is split into per-peer windows of at most
             * ``window`` elements.  Both sides derive the window lengths from
             * the same pairwise count (a send block and the matching receive
             * block have equal lengths), pack the window compactly and use
             * advancing offsets that always fit int.  The transfer is never
             * refused and the domain is never reduced.
             *
             * The common case (total slice <= window) goes straight to MPI with
             * the original layout and no copy.  Window length is bounded by
             * ``INT_MAX / (2 * ranks)`` so the packed window total fits int for
             * any rank count.
             */
            void alltoallvChunked(
                double* send,
                std::vector<std::size_t> const& sendCounts,
                std::vector<std::size_t> const& sendDispls,
                double* recv,
                std::vector<std::size_t> const& recvCounts,
                std::vector<std::size_t> const& recvDispls)
            {
                std::size_t const size = sendCounts.size();
                auto const end = [](std::vector<std::size_t> const& counts,
                                    std::vector<std::size_t> const& displs) {
                    std::size_t total = 0u;
                    for(std::size_t r = 0; r < counts.size(); ++r)
                        total = std::max(total, displs[r] + counts[r]);
                    return total;
                };
                std::size_t const sendTotal = end(sendCounts, sendDispls);
                std::size_t const recvTotal = end(recvCounts, recvDispls);
                std::size_t const total = std::max(sendTotal, recvTotal);

                // Window resolved once per transform in the constructor.
                std::size_t const window = m_transposeWindow;

                // MPI_Alltoallv takes int.  Never narrow silently: the window
                // bound keeps these values inside int, and this check rejects
                // loudly if the bound or a count ever exceeds it.
                auto const checkedInt = [&](std::size_t value, char const* what) {
                    if(value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                        throw std::runtime_error(
                            std::string{"DistributedFft3D: "} + what + " requires "
                            + std::to_string(value) + " doubles on rank "
                            + std::to_string(m_parallel->rank)
                            + " but MPI_Alltoallv counts are limited to "
                            + std::to_string(std::numeric_limits<int>::max()));
                    return static_cast<int>(value);
                };
                std::vector<int> sc(size, 0);
                std::vector<int> sd(size, 0);
                std::vector<int> rc(size, 0);
                std::vector<int> rd(size, 0);

                if(total <= window)
                {
                    for(std::size_t r = 0; r < size; ++r)
                    {
                        sc[r] = checkedInt(sendCounts[r], "transpose send");
                        sd[r] = checkedInt(sendDispls[r], "transpose send displacement");
                        rc[r] = checkedInt(recvCounts[r], "transpose receive");
                        rd[r] = checkedInt(recvDispls[r], "transpose receive displacement");
                    }
                    m_parallel->alltoallv(send, sc.data(), sd.data(), recv, rc.data(), rd.data());
                    return;
                }

                // Compact staging for one window.  Allocated once; the chunked
                // path is only taken by a slice above ~2^31 doubles.
                std::vector<double> sendChunk;
                std::vector<double> recvChunk;
                for(std::size_t base = 0u; base < total; base += window)
                {
                    std::size_t sendPacked = 0u;
                    std::size_t recvPacked = 0u;
                    for(std::size_t r = 0; r < size; ++r)
                    {
                        std::size_t const sendLen
                            = std::min(window, sendCounts[r] > base ? sendCounts[r] - base : 0u);
                        sc[r] = checkedInt(sendLen, "transpose window send");
                        sd[r] = checkedInt(sendPacked, "transpose window send displacement");
                        sendPacked += sendLen;
                        std::size_t const recvLen
                            = std::min(window, recvCounts[r] > base ? recvCounts[r] - base : 0u);
                        rc[r] = checkedInt(recvLen, "transpose window receive");
                        rd[r] = checkedInt(recvPacked, "transpose window receive displacement");
                        recvPacked += recvLen;
                    }
                    if(sendChunk.size() < sendPacked)
                        sendChunk.resize(sendPacked);
                    if(recvChunk.size() < recvPacked)
                        recvChunk.resize(recvPacked);
                    for(std::size_t r = 0; r < size; ++r)
                    {
                        if(sc[r] == 0)
                            continue;
                        std::memcpy(
                            sendChunk.data() + sd[r],
                            send + sendDispls[r] + base,
                            static_cast<std::size_t>(sc[r]) * sizeof(double));
                    }
                    m_parallel->alltoallv(
                        sendChunk.data(),
                        sc.data(),
                        sd.data(),
                        recvChunk.data(),
                        rc.data(),
                        rd.data());
                    for(std::size_t r = 0; r < size; ++r)
                    {
                        if(rc[r] == 0)
                            continue;
                        std::memcpy(
                            recv + recvDispls[r] + base,
                            recvChunk.data() + rd[r],
                            static_cast<std::size_t>(rc[r]) * sizeof(double));
                    }
                }
            }


            void forwardTo(Buffer& destination)
            {
                auto const t0 = Clock::now();
                alpaka::fft::onHost::executeForward(m_queue, *m_xPlan, *m_work, *m_work);
                executeYPlanes(true);
                alpaka::onHost::wait(m_queue);
                m_fftMs += elapsed(t0);

                auto const t1 = Clock::now();
                for(std::uint32_t y = 0; y < m_My; ++y)
                {
                    packForward(y);
                    alltoallvChunked(
                        reinterpret_cast<double*>(m_send->data()),
                        m_sendCounts,
                        m_sendDispls,
                        reinterpret_cast<double*>(m_recv->data()),
                        m_recvCounts,
                        m_recvDispls);
                    unpackForward(y, destination);
                }
                m_transposeMs += elapsed(t1);

                auto const t2 = Clock::now();
                alpaka::fft::onHost::executeForward(m_queue, *m_zPlan, destination, destination);
                alpaka::onHost::wait(m_queue);
                m_fftMs += elapsed(t2);
            }

            //! Pack the y slice of the z-slab work array for the forward alltoall.
            void packForward(std::uint32_t y)
            {
                for(int r = 0; r < m_parallel->size; ++r)
                {
                    auto const x = slabRange(m_Mx, m_parallel->size, r);
                    auto const sz = static_cast<std::size_t>(r);
                    Complex* dst = m_send->data() + m_sendDispls[sz] / 2;
                    for(std::uint32_t zl = 0; zl < m_nz; ++zl)
                    {
                        Complex const* src = m_work->data()
                            + static_cast<std::size_t>(m_Mx)
                                * (static_cast<std::size_t>(y) + static_cast<std::size_t>(m_My) * zl)
                            + x.first;
                        std::memcpy(
                            dst + static_cast<std::size_t>(zl) * x.second,
                            src,
                            static_cast<std::size_t>(x.second) * sizeof(Complex));
                    }
                }
            }

            //! Scatter the received y slice into the transposed z-local layout.
            void unpackForward(std::uint32_t y, Buffer& destination)
            {
                for(int r = 0; r < m_parallel->size; ++r)
                {
                    auto const z = slabRange(m_Mz, m_parallel->size, r);
                    auto const sz = static_cast<std::size_t>(r);
                    Complex const* src = m_recv->data() + m_recvDispls[sz] / 2;
                    for(std::uint32_t zl = 0; zl < z.second; ++zl)
                    {
                        Complex* dst = destination.data()
                            + static_cast<std::size_t>(m_nx)
                                * (static_cast<std::size_t>(y)
                                   + static_cast<std::size_t>(m_My) * (z.first + zl));
                        std::memcpy(
                            dst,
                            src + static_cast<std::size_t>(zl) * m_nx,
                            static_cast<std::size_t>(m_nx) * sizeof(Complex));
                    }
                }
            }

            //! Pack one y slice of the z-local product for the backward exchange.
            void packBackward(std::uint32_t y)
            {
                for(int r = 0; r < m_parallel->size; ++r)
                {
                    auto const z = slabRange(m_Mz, m_parallel->size, r);
                    auto const sz = static_cast<std::size_t>(r);
                    Complex* dst = m_send->data() + m_backSendDispls[sz] / 2;
                    for(std::uint32_t zl = 0; zl < z.second; ++zl)
                    {
                        Complex const* src = m_kernel->data()
                            + static_cast<std::size_t>(m_nx)
                                * (static_cast<std::size_t>(y)
                                   + static_cast<std::size_t>(m_My) * (z.first + zl));
                        std::memcpy(
                            dst + static_cast<std::size_t>(zl) * m_nx,
                            src,
                            static_cast<std::size_t>(m_nx) * sizeof(Complex));
                    }
                }
            }

            //! Scatter one received y slice back into the z-slab work array.
            void unpackBackward(std::uint32_t y)
            {
                for(int r = 0; r < m_parallel->size; ++r)
                {
                    auto const x = slabRange(m_Mx, m_parallel->size, r);
                    auto const sz = static_cast<std::size_t>(r);
                    Complex const* src = m_recv->data() + m_backRecvDispls[sz] / 2;
                    for(std::uint32_t zl = 0; zl < m_nz; ++zl)
                    {
                        Complex* dst = m_work->data()
                            + static_cast<std::size_t>(m_Mx)
                                * (static_cast<std::size_t>(y) + static_cast<std::size_t>(m_My) * zl)
                            + x.first;
                        std::memcpy(
                            dst,
                            src + static_cast<std::size_t>(zl) * x.second,
                            static_cast<std::size_t>(x.second) * sizeof(Complex));
                    }
                }
            }

            //! Transpose one y slice back from the z-local spectrum to the z-slab work array.
            void transposeBackward(std::uint32_t y)
            {
                packBackward(y);
                alltoallvChunked(
                    reinterpret_cast<double*>(m_send->data()),
                    m_backSendCounts,
                    m_backSendDispls,
                    reinterpret_cast<double*>(m_recv->data()),
                    m_backRecvCounts,
                    m_backRecvDispls);
                unpackBackward(y);
            }

            T_Queue& m_queue;
            ParallelContext const* m_parallel;
            std::uint32_t m_Mx;
            std::uint32_t m_My;
            std::uint32_t m_Mz;
            std::uint32_t m_z0 = 0;
            std::uint32_t m_nz = 0;
            std::uint32_t m_x0 = 0;
            std::uint32_t m_nx = 0;
            std::size_t m_chunkSize = 0;
            //! MPI_Alltoallv window in doubles, resolved once per transform.
            std::size_t m_transposeWindow = 0;
            std::optional<Buffer> m_work;
            std::optional<Buffer> m_send;
            std::optional<Buffer> m_recv;
            std::optional<Buffer> m_spectrum;
            std::optional<Buffer> m_kernel;
            std::shared_ptr<Plan> m_xPlan;
            std::shared_ptr<Plan> m_yPlan;
            std::shared_ptr<Plan> m_zPlan;
            std::vector<std::size_t> m_sendCounts;
            std::vector<std::size_t> m_sendDispls;
            std::vector<std::size_t> m_recvCounts;
            std::vector<std::size_t> m_recvDispls;
            std::vector<std::size_t> m_backSendCounts;
            std::vector<std::size_t> m_backSendDispls;
            std::vector<std::size_t> m_backRecvCounts;
            std::vector<std::size_t> m_backRecvDispls;
            double m_fftMs = 0.0;
            double m_transposeMs = 0.0;
        };
    } // namespace detail
} // namespace picBunch
