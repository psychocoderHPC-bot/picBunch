// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include <alpaka/alpaka.hpp>

#include <cstddef>
#include <cstdint>

namespace picBunch
{
    namespace detail
    {
        /** Copy the valid ``nx*ny*nz`` subregion of the padded ``mx*my*mz`` real grid
         *  into a compact x-fastest staging buffer.
         *
         *  Pure copy: no arithmetic, no reordering, so the staged doubles are
         *  bit-identical to the source.  This keeps the host read-out loop off the
         *  unified FFT output buffer, whose page migration dominated the measured
         *  wall time.
         *
         *  The caller guarantees ``nx*ny*nz <= UINT32_MAX`` (computeField checks
         *  this explicitly before it enqueues the kernel).
         */
        struct CompactGatherKernel
        {
            template<typename TAcc>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                double const* src,
                double* dst,
                std::uint32_t nx,
                std::uint32_t ny,
                std::uint32_t nz,
                std::uint32_t mx,
                std::uint32_t my) const
            {
                std::uint32_t const count = nx * ny * nz;
                for(auto [l] : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{count}))
                {
                    std::uint32_t const i = l % nx;
                    std::uint32_t const rem = l / nx;
                    std::uint32_t const j = rem % ny;
                    std::uint32_t const k = rem / ny;
                    dst[l] = src[
                        static_cast<std::size_t>(i)
                        + static_cast<std::size_t>(mx)
                            * (static_cast<std::size_t>(j)
                               + static_cast<std::size_t>(my) * static_cast<std::size_t>(k))];
                }
            }
        };

        /** Serial host reference for CompactGatherKernel.
         *
         *  Independent of the device kernel so a regression test can compare the
         *  device result bit-for-bit against the intended indexing.  The traversal
         *  is x-fastest and writes ``dst[l]`` for ``l = i + nx*(j + ny*k)``.
         */
        inline void gatherCompactHost(
            double const* src,
            double* dst,
            std::uint32_t nx,
            std::uint32_t ny,
            std::uint32_t nz,
            std::uint32_t mx,
            std::uint32_t my)
        {
            std::size_t l = 0;
            for(std::uint32_t k = 0; k < nz; ++k)
                for(std::uint32_t j = 0; j < ny; ++j)
                    for(std::uint32_t i = 0; i < nx; ++i, ++l)
                        dst[l] = src[
                            static_cast<std::size_t>(i)
                            + static_cast<std::size_t>(mx)
                                * (static_cast<std::size_t>(j)
                                   + static_cast<std::size_t>(my) * static_cast<std::size_t>(k))];
        }
    } // namespace detail
} // namespace picBunch
