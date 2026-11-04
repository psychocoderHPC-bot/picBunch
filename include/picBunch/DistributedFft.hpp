// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/Parallel.hpp"

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
#include <utility>
#include <vector>

namespace picBunch
{
    namespace detail
    {
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
                if(parallel == nullptr || parallel->size <= 1)
                    throw std::invalid_argument(
                        "DistributedFft3D requires a distributed ParallelContext");
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

                m_work.emplace(alpaka::onHost::allocUnified<Complex>(device, workSize));
                m_send.emplace(alpaka::onHost::allocUnified<Complex>(device, m_chunkSize));
                m_recv.emplace(alpaka::onHost::allocUnified<Complex>(device, m_chunkSize));
                m_spectrum.emplace(alpaka::onHost::allocUnified<Complex>(device, spectrumSize));
                m_kernel.emplace(alpaka::onHost::allocUnified<Complex>(device, spectrumSize));

                m_xPlan.emplace(
                    alpaka::fft::onHost::PlanBuilder<Complex, Extents1>{Extents1{Mx}}
                        .c2c()
                        .inPlace()
                        .batch(My * m_nz)
                        .distances(Mx * sizeof(Complex), Mx * sizeof(Complex))
                        .build(device));
                m_yPlan.emplace(
                    alpaka::fft::onHost::PlanBuilder<Complex, Extents1>{Extents1{My}}
                        .c2c()
                        .inPlace()
                        .batch(Mx)
                        .strides(Mx * sizeof(Complex), Mx * sizeof(Complex))
                        .distances(sizeof(Complex), sizeof(Complex))
                        .build(device));
                m_zPlan.emplace(
                    alpaka::fft::onHost::PlanBuilder<Complex, Extents1>{Extents1{Mz}}
                        .c2c()
                        .inPlace()
                        .batch(m_nx * My)
                        .strides(
                            static_cast<std::size_t>(m_nx) * My * sizeof(Complex),
                            static_cast<std::size_t>(m_nx) * My * sizeof(Complex))
                        .distances(sizeof(Complex), sizeof(Complex))
                        .build(device));

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
                auto const count = static_cast<std::uint32_t>(m_nx * m_My * m_Mz);
                // Multiply the kernel spectrum in place so the source spectrum
                // survives for the remaining components.
                auto* a = reinterpret_cast<double*>(m_kernel->data());
                auto const* b = reinterpret_cast<double const*>(m_spectrum->data());
                m_queue.enqueue(
                    alpaka::onHost::FrameSpec{(count + 63u) / 64u, 64u},
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
                        a[off] = ar * br - ai * bi;
                        a[off + 1u] = ar * bi + ai * br;
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
                    m_sendCounts[sz] = static_cast<int>(
                        static_cast<long long>(m_nz) * x.second * 2ll);
                    m_sendDispls[sz] = static_cast<int>(sendPos);
                    sendPos += static_cast<long long>(m_nz) * x.second * 2ll;
                    // Receive from r: our local x, all of r's z planes.
                    m_recvCounts[sz] = static_cast<int>(
                        static_cast<long long>(m_nx) * z.second * 2ll);
                    m_recvDispls[sz] = static_cast<int>(recvPos);
                    recvPos += static_cast<long long>(m_nx) * z.second * 2ll;

                    // Backward: dest r owns its z range.
                    m_backSendCounts[sz] = static_cast<int>(
                        static_cast<long long>(m_nx) * z.second * 2ll);
                    m_backSendDispls[sz] = static_cast<int>(backSendPos);
                    backSendPos += static_cast<long long>(m_nx) * z.second * 2ll;
                    m_backRecvCounts[sz] = static_cast<int>(
                        static_cast<long long>(x.second) * m_nz * 2ll);
                    m_backRecvDispls[sz] = static_cast<int>(backRecvPos);
                    backRecvPos += static_cast<long long>(x.second) * m_nz * 2ll;
                }
                long long const forwardSend = static_cast<long long>(m_Mx) * m_nz * 2ll;
                long long const forwardRecv = static_cast<long long>(m_nx) * m_Mz * 2ll;
                long long const backwardSend = static_cast<long long>(m_nx) * m_Mz * 2ll;
                long long const backwardRecv = static_cast<long long>(m_Mx) * m_nz * 2ll;
                if(sendPos != forwardSend || recvPos != forwardRecv || backSendPos != backwardSend
                   || backRecvPos != backwardRecv)
                    throw std::runtime_error("DistributedFft3D: inconsistent transpose counts");
                if(forwardSend > static_cast<long long>(std::numeric_limits<int>::max())
                   || forwardRecv > static_cast<long long>(std::numeric_limits<int>::max()))
                    throw std::runtime_error(
                        "DistributedFft3D: a transpose slice exceeds the MPI int count range");
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
                            + static_cast<std::size_t>(m_Mx) * (y + m_My * zl) + x.first;
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
                            + static_cast<std::size_t>(m_nx) * (y + m_My * (z.first + zl));
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
                            + static_cast<std::size_t>(m_nx) * (y + m_My * (z.first + zl));
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
                            + static_cast<std::size_t>(m_Mx) * (y + m_My * zl) + x.first;
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
            std::optional<Plan> m_xPlan;
            std::optional<Plan> m_yPlan;
            std::optional<Plan> m_zPlan;
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
