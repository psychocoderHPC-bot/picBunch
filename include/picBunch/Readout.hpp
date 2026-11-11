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
         *  The source is a 3-D MdSpan over the padded box; the destination is a
         *  1-D MdSpan.  Both index types are provided by the spans, so the former
         *  ``mx``/``my`` pitch arguments are gone.
         *
         *  The caller guarantees ``nx*ny*nz <= UINT32_MAX`` (computeField checks
         *  this explicitly before it enqueues the kernel).
         */
        struct CompactGatherKernel
        {
            template<typename TAcc>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                alpaka::concepts::IMdSpan auto const& src,
                alpaka::concepts::IMdSpan auto dst,
                std::uint32_t nx,
                std::uint32_t ny,
                std::uint32_t nz) const
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
                    dst[l] = src[alpaka::Vec{k, j, i}];
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
            alpaka::concepts::IMdSpan auto const& src,
            alpaka::concepts::IMdSpan auto dst,
            std::uint32_t nx,
            std::uint32_t ny,
            std::uint32_t nz)
        {
            std::size_t l = 0;
            for(std::uint32_t k = 0; k < nz; ++k)
                for(std::uint32_t j = 0; j < ny; ++j)
                    for(std::uint32_t i = 0; i < nx; ++i, ++l)
                        // size_t linear index into 32-bit extents; count <= kMaxPaddedNodes (2^28).
                        dst[l] = src[alpaka::Vec{k, j, i}];
        }

    } // namespace detail
} // namespace picBunch
