// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/ChunkedFft.hpp"
#include "picBunch/DistributedFft.hpp"
#include "picBunch/Parallel.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/math/Complex.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace picBunch::detail
{
    /*! The distributed FFT with its collective exchange replaced by staging.
     *
     * The algorithm, slab decomposition, pack/unpack index arithmetic and the
     * transforms are the ones in ``DistributedFft3D``.  One process plays every
     * virtual rank in turn and the ``MPI_Alltoallv`` slice is written to, then
     * read from, a file at the deterministic offsets of the all-to-all matrix.
     *
     * The exchange is split into three passes so that only one virtual rank's
     * buffers are alive:
     *
     *  1. every z-slab rank runs the local x/y transforms and writes its packed
     *     slices (one per y) to the staging file;
     *  2. every x-slab rank reads the slices of its column, unpacks, runs the z
     *     transform, multiplies the kernel and writes the backward slices;
     *  3. every z-slab rank reads its backward column, unpacks, runs the inverse
     *     y/x transforms and emits the output planes.
     *
     * The file layout is the per-y concatenation of the existing send/recv
     * buffers: for one y the all-to-all matrix is stored row by row (source
     * rank), and a destination reads its column.  The offsets are derived from
     * the same ``slabRange`` split the MPI path uses, so a block has the same
     * bytes and the same partner as in the collective.
     *
     * Peak resident memory is one ``DistributedFft3D``:
     * ``16 * (Mx*My*nz + 2*nx*My*Mz + 2*max(Mx*nz, Mz*nx))`` bytes, with
     * ``nz = ceil(Mz/N)`` and ``nx = ceil(Mx/N)``.  It does not grow with the
     * padded volume once N is fixed.
     */
    template<typename T_Device, typename T_Queue>
    class StagedDistributedSolver
    {
    public:
        using Complex = alpaka::math::Complex<double>;
        using SlabFill = std::function<void(
            std::uint32_t zOffset,
            std::uint32_t zCount,
            Complex* work)>;
        using SlabEmit = std::function<void(
            std::size_t zOffset,
            std::size_t zCount,
            double const* planes)>;

        StagedDistributedSolver(
            T_Device& device,
            T_Queue& queue,
            std::uint32_t Mx,
            std::uint32_t My,
            std::uint32_t Mz,
            int virtualRanks,
            std::string stagingDirectory,
            bool keepStaging,
            ChunkedSolveStats* stats)
            : m_device{device}
            , m_queue{queue}
            , m_Mx{Mx}
            , m_My{My}
            , m_Mz{Mz}
            , m_ranks{virtualRanks}
            , m_directory{std::move(stagingDirectory)}
            , m_keepStaging{keepStaging}
            , m_stats{stats}
        {
            if(m_ranks < 1)
                throw std::invalid_argument(
                    "StagedDistributedSolver: virtual rank count must be positive");
            if(m_ranks > static_cast<int>(std::min(Mx, Mz)))
                throw std::invalid_argument(
                    "StagedDistributedSolver: more virtual ranks than padded planes "
                    "on an axis");
            if(Mx < 1u || My < 1u || Mz < 1u)
                throw std::invalid_argument("StagedDistributedSolver: extents must be positive");

            m_contexts.resize(static_cast<std::size_t>(m_ranks));
            for(int r = 0; r < m_ranks; ++r)
            {
                m_contexts[static_cast<std::size_t>(r)].size = m_ranks;
                m_contexts[static_cast<std::size_t>(r)].rank = r;
            }

            m_nz.resize(static_cast<std::size_t>(m_ranks));
            m_z0.resize(static_cast<std::size_t>(m_ranks));
            m_nx.resize(static_cast<std::size_t>(m_ranks));
            m_x0.resize(static_cast<std::size_t>(m_ranks));
            m_fwdRowPrefix.assign(static_cast<std::size_t>(m_ranks) + 1u, 0u);
            m_specPrefix.assign(static_cast<std::size_t>(m_ranks) + 1u, 0u);
            m_bwdRowPrefix.assign(static_cast<std::size_t>(m_ranks) + 1u, 0u);
            for(int r = 0; r < m_ranks; ++r)
            {
                auto const z = slabRange(Mz, m_ranks, r);
                auto const x = slabRange(Mx, m_ranks, r);
                auto const sz = static_cast<std::size_t>(r);
                m_z0[sz] = z.first;
                m_nz[sz] = z.second;
                m_x0[sz] = x.first;
                m_nx[sz] = x.second;
                m_fwdRowPrefix[sz + 1u]
                    = m_fwdRowPrefix[sz] + static_cast<std::uint64_t>(m_Mx) * m_nz[sz] * 2u;
                m_specPrefix[sz + 1u]
                    = m_specPrefix[sz] + static_cast<std::uint64_t>(m_nx[sz]) * m_My * m_Mz;
                m_bwdRowPrefix[sz + 1u]
                    = m_bwdRowPrefix[sz] + static_cast<std::uint64_t>(m_Mz) * m_nx[sz] * 2u;
            }
            m_sliceDoubles = static_cast<std::uint64_t>(m_Mx) * m_Mz * 2u;

            std::error_code ec;
            std::filesystem::create_directories(m_directory, ec);
            if(ec)
                throw std::runtime_error(
                    "staged solve: cannot create staging directory '" + m_directory
                    + "': " + ec.message());
            m_prefix = m_directory + "/staged-" + std::to_string(::getpid());

            if(m_stats)
            {
                std::uint64_t const nzMax = *std::max_element(m_nz.begin(), m_nz.end());
                std::uint64_t const nxMax = *std::max_element(m_nx.begin(), m_nx.end());
                std::uint64_t const bytes = 16u
                    * (static_cast<std::uint64_t>(m_Mx) * m_My * nzMax
                       + 2u * nxMax * m_My * m_Mz
                       + 2u * std::max<std::uint64_t>(
                             static_cast<std::uint64_t>(m_Mx) * nzMax,
                             static_cast<std::uint64_t>(m_Mz) * nxMax));
                m_stats->peakChunkBytes = std::max(m_stats->peakChunkBytes, bytes);
            }
        }

        StagedDistributedSolver(StagedDistributedSolver const&) = delete;
        StagedDistributedSolver& operator=(StagedDistributedSolver const&) = delete;

        //! Forward exchange of the deposited source; leaves the source spectrum
        //! in the per-x-slab staging file.
        void forwardSource(SlabFill const& fill)
        {
            auto const wallStart = std::chrono::steady_clock::now();
            StagingFile exchange = makeStaging("source-forward");
            for(int s = 0; s < m_ranks; ++s)
            {
                DistributedFft3D fft{fftFor(s)};
                fft.clearWork();
                fill(fft.zOffset(), fft.zCount(), fft.work());
                fft.forwardLocal();
                std::uint64_t const rowBase
                    = m_fwdRowPrefix[static_cast<std::size_t>(s)];
                for(std::uint32_t y = 0; y < m_My; ++y)
                {
                    fft.packForwardSlice(y);
                    exchange.writeAt(
                        fft.sendDoubles(),
                        fft.forwardSendDoubles() * sizeof(double),
                        (static_cast<std::uint64_t>(y) * m_sliceDoubles + rowBase) * sizeof(double));
                    if(m_stats)
                        m_stats->sourceWrite += fft.forwardSendDoubles() * sizeof(double);
                }
                accountFft(fft);
            }

            if(std::getenv("PICBUNCH_TEST_TRUNCATE_STAGING") != nullptr)
                ::ftruncate(exchange.fd(), 1024);
            StagingFile spectrum = makeStaging("source-spectrum");
            for(int d = 0; d < m_ranks; ++d)
            {
                DistributedFft3D fft{fftFor(d)};
                for(std::uint32_t y = 0; y < m_My; ++y)
                {
                    readForwardColumn(fft, exchange, d, y, false);
                    fft.unpackForwardSlice(y);
                }
                fft.forwardZ(false);
                std::uint64_t const specBytes
                    = static_cast<std::uint64_t>(m_nx[static_cast<std::size_t>(d)]) * m_My * m_Mz
                    * sizeof(Complex);
                spectrum.writeAt(
                    fft.spectrumData(),
                    specBytes,
                    m_specPrefix[static_cast<std::size_t>(d)] * sizeof(Complex));
                if(m_stats)
                    m_stats->sourceSpecWrite += specBytes;
                accountFft(fft);
            }
            exchange.reset();
            m_sourceSpectrum = std::move(spectrum);
            accountPhase(wallStart);
        }

        //! Forward kernel, multiply by the stored source spectrum, inverse, and
        //! stream the scaled output z-slabs to ``emit``.
        void convolveComponent(
            SlabFill const& fillKernel,
            std::uint32_t outNx,
            std::uint32_t outNy,
            std::uint32_t outNz,
            double factor,
            SlabEmit const& emit)
        {
            if(!m_sourceSpectrum.has_value())
                throw std::runtime_error(
                    "StagedDistributedSolver: forwardSource() was not called");
            auto const wallStart = std::chrono::steady_clock::now();

            StagingFile kernelExchange = makeStaging("kernel-forward");
            for(int s = 0; s < m_ranks; ++s)
            {
                DistributedFft3D fft{fftFor(s)};
                fft.clearWork();
                fillKernel(fft.zOffset(), fft.zCount(), fft.work());
                fft.forwardLocal();
                std::uint64_t const rowBase
                    = m_fwdRowPrefix[static_cast<std::size_t>(s)];
                for(std::uint32_t y = 0; y < m_My; ++y)
                {
                    fft.packForwardSlice(y);
                    kernelExchange.writeAt(
                        fft.sendDoubles(),
                        fft.forwardSendDoubles() * sizeof(double),
                        (static_cast<std::uint64_t>(y) * m_sliceDoubles + rowBase) * sizeof(double));
                    if(m_stats)
                        m_stats->kernelWrite += fft.forwardSendDoubles() * sizeof(double);
                }
                accountFft(fft);
            }

            StagingFile backward = makeStaging("product-backward");
            for(int d = 0; d < m_ranks; ++d)
            {
                DistributedFft3D fft{fftFor(d)};
                auto const sd = static_cast<std::size_t>(d);
                std::uint64_t const specBytes
                    = static_cast<std::uint64_t>(m_nx[sd]) * m_My * m_Mz * sizeof(Complex);
                m_sourceSpectrum->readAt(
                    fft.spectrumData(),
                    specBytes,
                    m_specPrefix[sd] * sizeof(Complex));
                if(m_stats)
                    m_stats->sourceSpecRead += specBytes;
                for(std::uint32_t y = 0; y < m_My; ++y)
                {
                    readForwardColumn(fft, kernelExchange, d, y, true);
                    fft.unpackForwardSliceKernel(y);
                }
                fft.forwardZ(true);
                fft.multiply();
                fft.inverseZ();
                std::uint64_t const rowBase
                    = m_bwdRowPrefix[static_cast<std::size_t>(d)];
                for(std::uint32_t y = 0; y < m_My; ++y)
                {
                    fft.packBackwardSlice(y);
                    backward.writeAt(
                        fft.sendDoubles(),
                        fft.backwardSendDoubles() * sizeof(double),
                        (static_cast<std::uint64_t>(y) * m_sliceDoubles + rowBase) * sizeof(double));
                    if(m_stats)
                        m_stats->backWrite += fft.backwardSendDoubles() * sizeof(double);
                }
                accountFft(fft);
            }
            kernelExchange.reset();

            for(int s = 0; s < m_ranks; ++s)
            {
                DistributedFft3D fft{fftFor(s)};
                for(std::uint32_t y = 0; y < m_My; ++y)
                {
                    readBackwardColumn(fft, backward, s, y);
                    fft.unpackBackwardSlice(y);
                }
                fft.inverseLocal();
                accountFft(fft);
                gatherAndEmit(fft, outNx, outNy, outNz, factor, emit);
            }
            backward.reset();
            accountPhase(wallStart);
            if(m_stats)
            {
                m_stats->fftSeconds += 0.0; // per-object time is added in accountFft
            }
        }

    private:
        struct VirtualContext : ParallelContext
        {
        };

        DistributedFft3D<T_Device, T_Queue> fftFor(int rank)
        {
            return DistributedFft3D<T_Device, T_Queue>{
                m_device,
                m_queue,
                m_Mx,
                m_My,
                m_Mz,
                &m_contexts[static_cast<std::size_t>(rank)]};
        }

        StagingFile makeStaging(std::string const& tag)
        {
            static std::uint64_t counter = 0;
            StagingFile file(
                m_prefix + "-" + tag + "-" + std::to_string(counter++) + ".bin",
                m_stats);
            file.keep(m_keepStaging);
            return file;
        }

        //! Read the column block (source ``s``, destination ``d``) for one y.
        void readForwardColumn(
            DistributedFft3D<T_Device, T_Queue>& fft,
            StagingFile& exchange,
            int d,
            std::uint32_t y,
            bool kernel)
        {
            auto const sd = static_cast<std::size_t>(d);
            auto const& displs = fft.recvDisplacements();
            for(int s = 0; s < m_ranks; ++s)
            {
                auto const ss = static_cast<std::size_t>(s);
                std::uint64_t const count
                    = static_cast<std::uint64_t>(m_nz[ss]) * m_nx[sd] * 2u;
                std::uint64_t const offset
                    = static_cast<std::uint64_t>(y) * m_sliceDoubles
                    + m_fwdRowPrefix[ss]
                    + static_cast<std::uint64_t>(m_nz[ss]) * m_x0[sd] * 2u;
                exchange.readAt(
                    fft.recvDoubles() + displs[ss],
                    count * sizeof(double),
                    offset * sizeof(double));
                if(m_stats)
                {
                    if(kernel)
                        m_stats->kernelRead += count * sizeof(double);
                    else
                        m_stats->sourceRead += count * sizeof(double);
                }
            }
        }

        void readBackwardColumn(
            DistributedFft3D<T_Device, T_Queue>& fft,
            StagingFile& backward,
            int s,
            std::uint32_t y)
        {
            auto const ss = static_cast<std::size_t>(s);
            auto const& displs = fft.backRecvDisplacements();
            for(int r = 0; r < m_ranks; ++r)
            {
                auto const rr = static_cast<std::size_t>(r);
                std::uint64_t const count
                    = static_cast<std::uint64_t>(m_nx[rr]) * m_nz[ss] * 2u;
                std::uint64_t const offset
                    = static_cast<std::uint64_t>(y) * m_sliceDoubles
                    + m_bwdRowPrefix[rr]
                    + static_cast<std::uint64_t>(m_nx[rr]) * m_z0[ss] * 2u;
                backward.readAt(
                    fft.recvDoubles() + displs[rr],
                    count * sizeof(double),
                    offset * sizeof(double));
                if(m_stats)
                    m_stats->backRead += count * sizeof(double);
            }
        }

        void gatherAndEmit(
            DistributedFft3D<T_Device, T_Queue>& fft,
            std::uint32_t outNx,
            std::uint32_t outNy,
            std::uint32_t outNz,
            double factor,
            SlabEmit const& emit)
        {
            std::uint32_t const z0 = fft.zOffset();
            std::uint32_t const nz = fft.zCount();
            if(z0 >= outNz)
                return;
            std::uint32_t const planes = std::min(nz, outNz - z0);
            std::size_t const planeOut
                = static_cast<std::size_t>(outNx) * outNy;
            std::vector<double> output(static_cast<std::size_t>(planes) * planeOut);
            Complex const* work = fft.work();
            for(std::uint32_t k = 0; k < planes; ++k)
                for(std::uint32_t y = 0; y < outNy; ++y)
                    for(std::uint32_t x = 0; x < outNx; ++x)
                        output[(static_cast<std::size_t>(k) * outNy + y) * outNx + x]
                            = work[static_cast<std::size_t>(m_Mx)
                                       * (static_cast<std::size_t>(y)
                                          + static_cast<std::size_t>(m_My) * k)
                                   + x]
                                  .real()
                            * factor;
            emit(z0, planes, output.data());
        }

        void accountFft(DistributedFft3D<T_Device, T_Queue> const& fft)
        {
            if(m_stats)
                m_stats->fftSeconds += fft.fftMilliseconds() / 1000.0;
        }

        void accountPhase(std::chrono::steady_clock::time_point const& start)
        {
            if(m_stats)
                m_stats->transposeSeconds
                    += std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                           .count();
        }

        T_Device& m_device;
        T_Queue& m_queue;
        std::uint32_t m_Mx;
        std::uint32_t m_My;
        std::uint32_t m_Mz;
        int m_ranks;
        std::string m_directory;
        std::string m_prefix;
        bool m_keepStaging = false;
        ChunkedSolveStats* m_stats = nullptr;

        std::vector<VirtualContext> m_contexts;
        std::vector<std::uint32_t> m_nz;
        std::vector<std::uint32_t> m_z0;
        std::vector<std::uint32_t> m_nx;
        std::vector<std::uint32_t> m_x0;
        std::vector<std::uint64_t> m_fwdRowPrefix;
        std::vector<std::uint64_t> m_specPrefix;
        std::vector<std::uint64_t> m_bwdRowPrefix;
        std::uint64_t m_sliceDoubles = 0;
        std::optional<StagingFile> m_sourceSpectrum;
    };

} // namespace picBunch::detail
