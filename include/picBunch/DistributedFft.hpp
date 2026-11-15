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
        //! Largest per-rank padded node count the distributed FFT accepts.
        /*!
         * Every rank-local element count is still carried in 32-bit index
         * variables (``alpaka::Vec<std::uint32_t, N>`` extents, the vendor
         * plan batch and the kernel launch grid).  All byte offsets are
         * ``std::size_t`` after the 64-bit fix, so the node count, not the byte
         * count, is the remaining wall.  2^31 keeps every element count and
         * every per-rank transpose (two doubles per complex node) inside a
         * signed 32-bit range with room for the plane split; the exact
         * ``MPI_Alltoallv`` int-count limit is checked when the transpose
         * tables are built.
         */
        inline constexpr std::size_t kMaxDistributedNodesPerRank = std::size_t{1} << 31;

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
                // Safety net for the 32-bit element-index space that remains
                // in the snapshot and spectrum views and in the vendor plan
                // batch.  Byte offsets are 64-bit, so this is the true
                // remaining wall; refuse with the measured size instead of
                // wrapping an index or a byte offset.
                if(workSize > kMaxDistributedNodesPerRank
                   || spectrumSize > kMaxDistributedNodesPerRank)
                    throw std::runtime_error(
                        "DistributedFft3D: rank " + std::to_string(parallel->rank) + " of "
                        + std::to_string(parallel->size) + " requires "
                        + std::to_string(std::max(workSize, spectrumSize))
                        + " padded nodes per rank (work " + std::to_string(workSize)
                        + ", spectrum " + std::to_string(spectrumSize)
                        + "), above the per-rank distributed FFT limit of "
                        + std::to_string(kMaxDistributedNodesPerRank)
                        + " nodes; use more ranks or a smaller grid");
                m_chunkSize = std::max(
                    static_cast<std::size_t>(Mx) * m_nz,
                    static_cast<std::size_t>(Mz) * m_nx);

                // The vendor plan API takes the batch as a 32-bit value.  Form
                // it in 64 bits so a large slab cannot wrap here.
                auto const batchX = checkedU32(
                    static_cast<std::uint64_t>(My) * m_nz,
                    "x-transform batch");
                auto const batchZ = checkedU32(
                    static_cast<std::uint64_t>(m_nx) * My,
                    "z-transform batch");

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

            using Clock = std::chrono::steady_clock;

            static double elapsed(Clock::time_point const& start)
            {
                return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            }

            //! Narrow a count to the vendor plan API's 32-bit batch field.
            static std::uint32_t checkedU32(std::uint64_t value, char const* what)
            {
                if(value > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()))
                    throw std::runtime_error(
                        std::string{"DistributedFft3D: "} + what + " requires "
                        + std::to_string(value) + " items but the vendor FFT plan API is limited to "
                        + std::to_string(std::numeric_limits<std::uint32_t>::max()));
                return static_cast<std::uint32_t>(value);
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

                // MPI_Alltoallv takes int counts and displacements.  The
                // counts are double elements (two per complex node), so form
                // every value in 64 bits and refuse before narrowing.
                auto const checkedInt = [&](long long value, char const* what) {
                    if(value < 0 || value > static_cast<long long>(std::numeric_limits<int>::max()))
                        throw std::runtime_error(
                            std::string{"DistributedFft3D: "} + what + " requires "
                            + std::to_string(value) + " doubles on rank "
                            + std::to_string(m_parallel->rank) + " of "
                            + std::to_string(size) + " but MPI_Alltoallv counts are limited to "
                            + std::to_string(std::numeric_limits<int>::max()));
                    return static_cast<int>(value);
                };
                long long sendPos = 0;
                long long recvPos = 0;
                long long backSendPos = 0;
                long long backRecvPos = 0;
                for(int r = 0; r < size; ++r)
                {
                    auto const z = slabRange(m_Mz, size, r);
                    auto const x = slabRange(m_Mx, size, r);
                    auto const sz = static_cast<std::size_t>(r);
                    // Forward: one y slice, dest r gets all local z and its x range.
                    m_sendCounts[sz] = checkedInt(
                        static_cast<long long>(m_nz) * x.second * 2ll,
                        "forward transpose send");
                    m_sendDispls[sz] = checkedInt(sendPos, "forward transpose send displacement");
                    sendPos += static_cast<long long>(m_nz) * x.second * 2ll;
                    // Receive from r: our local x, all of r's z planes.
                    m_recvCounts[sz] = checkedInt(
                        static_cast<long long>(m_nx) * z.second * 2ll,
                        "forward transpose receive");
                    m_recvDispls[sz] = checkedInt(recvPos, "forward transpose receive displacement");
                    recvPos += static_cast<long long>(m_nx) * z.second * 2ll;

                    // Backward: dest r owns its z range.
                    m_backSendCounts[sz] = checkedInt(
                        static_cast<long long>(m_nx) * z.second * 2ll,
                        "backward transpose send");
                    m_backSendDispls[sz] = checkedInt(backSendPos, "backward transpose send displacement");
                    backSendPos += static_cast<long long>(m_nx) * z.second * 2ll;
                    m_backRecvCounts[sz] = checkedInt(
                        static_cast<long long>(x.second) * m_nz * 2ll,
                        "backward transpose receive");
                    m_backRecvDispls[sz] = checkedInt(backRecvPos, "backward transpose receive displacement");
                    backRecvPos += static_cast<long long>(x.second) * m_nz * 2ll;
                }
                long long const forwardSend = static_cast<long long>(m_Mx) * m_nz * 2ll;
                long long const forwardRecv = static_cast<long long>(m_nx) * m_Mz * 2ll;
                long long const backwardSend = static_cast<long long>(m_nx) * m_Mz * 2ll;
                long long const backwardRecv = static_cast<long long>(m_Mx) * m_nz * 2ll;
                if(sendPos != forwardSend || recvPos != forwardRecv || backSendPos != backwardSend
                   || backRecvPos != backwardRecv)
                    throw std::runtime_error("DistributedFft3D: inconsistent transpose counts");
                static_cast<void>(checkedInt(forwardSend, "forward transpose total"));
                static_cast<void>(checkedInt(forwardRecv, "forward transpose total"));
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
                    m_parallel->alltoallv(
                        reinterpret_cast<double*>(m_send->data()),
                        m_sendCounts.data(),
                        m_sendDispls.data(),
                        reinterpret_cast<double*>(m_recv->data()),
                        m_recvCounts.data(),
                        m_recvDispls.data());
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

            //! Transpose one y slice back from the z-local spectrum to the z-slab work array.
            void transposeBackward(std::uint32_t y)
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
                m_parallel->alltoallv(
                    reinterpret_cast<double*>(m_send->data()),
                    m_backSendCounts.data(),
                    m_backSendDispls.data(),
                    reinterpret_cast<double*>(m_recv->data()),
                    m_backRecvCounts.data(),
                    m_backRecvDispls.data());
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
            std::optional<Buffer> m_work;
            std::optional<Buffer> m_send;
            std::optional<Buffer> m_recv;
            std::optional<Buffer> m_spectrum;
            std::optional<Buffer> m_kernel;
            std::shared_ptr<Plan> m_xPlan;
            std::shared_ptr<Plan> m_yPlan;
            std::shared_ptr<Plan> m_zPlan;
            std::vector<int> m_sendCounts;
            std::vector<int> m_sendDispls;
            std::vector<int> m_recvCounts;
            std::vector<int> m_recvDispls;
            std::vector<int> m_backSendCounts;
            std::vector<int> m_backSendDispls;
            std::vector<int> m_backRecvCounts;
            std::vector<int> m_backRecvDispls;
            double m_fftMs = 0.0;
            double m_transposeMs = 0.0;
        };
    } // namespace detail
} // namespace picBunch
