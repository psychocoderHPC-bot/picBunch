// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group MpiReadout: the distributed read-out must gather and scale the local
// complex z-slab on the device into a compact double staging buffer without
// changing any value.  The host redistribution then reads that compact buffer
// instead of dereferencing the unified FFT work slab.
//
// A determinism run of computeField cannot catch a deterministic wrong gather,
// because both runs share the same wrong mapping.  These tests therefore compare
// the device gather-scale bit for bit against an independent serial host
// reference over non-cubic, padded shapes and several local slab depths.
//
// The kernel body below mirrors src/Compute.cpp's SlabGatherScaleKernel.  It is
// duplicated here on purpose: the production functor lives in an anonymous
// namespace and cannot be included.  Keep the two in sync.
//
// Registration (CMakeLists.txt is owned elsewhere): add test_mpi_readout.cpp to
// the picBunchUnitTests sources and add ``MpiReadout`` to the foreach(group ...)
// list in tests/CMakeLists.txt.  With MPI available the same executable is
// meaningful under ``mpirun -n 1`` and ``mpirun -n 2`` because the read-out
// stage is per-rank; a multi-rank launch additionally exercises the
// redistribution, which this file does not cover.

#include "TestDevice.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/math/Complex.hpp>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace
{
    using Complex = alpaka::math::Complex<double>;

    //! Same gather-scale as Compute.cpp's distributed read-out.
    struct SlabGatherScaleKernel
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            Complex const* work,
            double* dst,
            std::uint32_t nx,
            std::uint32_t ny,
            std::uint32_t nz,
            std::uint32_t mx,
            std::uint32_t my,
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
                std::size_t const idx = i
                    + static_cast<std::size_t>(mx)
                        * (j + static_cast<std::size_t>(my) * k);
                dst[l] = work[idx].real() * factor;
            }
        }
    };

    struct SlabShape
    {
        std::uint32_t nx;
        std::uint32_t ny;
        std::uint32_t nz;
        std::uint32_t mx;
        std::uint32_t my;
        std::uint32_t mz;
    };

    //! Deterministic, exactly representable value; a wrong stride or order gives
    //! a different double, so a bitwise comparison is meaningful.
    double sourceValue(std::uint32_t i, std::uint32_t j, std::uint32_t k, int pass)
    {
        std::uint64_t h = static_cast<std::uint64_t>(i) * 1000003u
                        + static_cast<std::uint64_t>(j) * 10007u
                        + static_cast<std::uint64_t>(k) * 131u
                        + static_cast<std::uint64_t>(pass + 1) * 17u;
        h ^= h >> 17;
        h *= 0x9e3779b97f4a7c15ull;
        h ^= h >> 29;
        return static_cast<double>(static_cast<std::int64_t>(h % 2000001ull) - 1000000ll) / 1024.0;
    }
} // namespace

TEST(MpiReadout, SlabGatherScaleMatchesSerialReferenceBitwise)
{
    // Non-cubic, odd extents catch stride and order mistakes; the padded mx/my
    // differ from the compact extents so a gather that ignores the padding
    // fails; nz < mz models a rank that owns only part of the padded box.
    std::array<SlabShape, 4> const shapes{{
        {5, 7, 3, 5, 7, 3},
        {7, 5, 3, 14, 15, 9},
        {6, 5, 4, 12, 10, 8},
        {9, 7, 2, 18, 21, 11},
    }};
    std::array<double, 3> const factors{0.25, 3.0 / 7.0, -11.0 / 32.0};

    picBunchTest::runOnDevice(
        [&](auto& device, auto& queue)
        {
            for(SlabShape const& s : shapes)
            {
                std::size_t const paddedCount
                    = static_cast<std::size_t>(s.mx) * s.my * s.mz;
                std::size_t const compactCount
                    = static_cast<std::size_t>(s.nx) * s.ny * s.nz;

                auto src = alpaka::onHost::allocUnified<Complex>(device, paddedCount);
                auto gatheredDev = alpaka::onHost::alloc<double>(device, compactCount);
                auto gathered = alpaka::onHost::allocMapped<double>(device, compactCount);
                std::vector<double> reference(compactCount);

                // Independent serial reference: nested x-fastest loops, no
                // shared index arithmetic with the device kernel.
                for(std::uint32_t k = 0; k < s.mz; ++k)
                    for(std::uint32_t j = 0; j < s.my; ++j)
                        for(std::uint32_t i = 0; i < s.mx; ++i)
                            src.data()[
                                static_cast<std::size_t>(i)
                                + static_cast<std::size_t>(s.mx)
                                    * (static_cast<std::size_t>(j)
                                       + static_cast<std::size_t>(s.my) * k)]
                                = Complex{sourceValue(i, j, k, 0), sourceValue(i, j, k, 1)};

                for(double factor : factors)
                {
                    std::size_t l = 0;
                    for(std::uint32_t k = 0; k < s.nz; ++k)
                        for(std::uint32_t j = 0; j < s.ny; ++j)
                            for(std::uint32_t i = 0; i < s.nx; ++i, ++l)
                                reference[l]
                                    = src.data()[
                                          static_cast<std::size_t>(i)
                                          + static_cast<std::size_t>(s.mx)
                                              * (static_cast<std::size_t>(j)
                                                 + static_cast<std::size_t>(s.my) * k)]
                                          .real()
                                    * factor;

                    queue.enqueue(
                        alpaka::onHost::FrameSpec{
                            static_cast<std::uint32_t>((compactCount + 63u) / 64u),
                            64u},
                        SlabGatherScaleKernel{},
                        src.data(),
                        gatheredDev.data(),
                        s.nx,
                        s.ny,
                        s.nz,
                        s.mx,
                        s.my,
                        factor);
                    alpaka::onHost::memcpy(queue, gathered, gatheredDev);
                    alpaka::onHost::wait(queue);

                    ASSERT_EQ(
                        0,
                        std::memcmp(
                            gathered.data(),
                            reference.data(),
                            compactCount * sizeof(double)))
                        << "shape " << s.nx << "x" << s.ny << "x" << s.nz << " padded "
                        << s.mx << "x" << s.my << "x" << s.mz << " factor " << factor
                        << ": device gather-scale differs from the serial reference";
                }
            }
        });
}

TEST(MpiReadout, SlabGatherScaleUsesRankLocalSlabBase)
{
    // In the distributed path the kernel receives dfft.work(), which points at
    // this rank's local z-slab inside the padded box.  Verify that passing a slab
    // base pointer at zOffset gives exactly the serial extraction of planes
    // [zOffset, zOffset+nz) with the padded mx*my stride.
    picBunchTest::runOnDevice(
        [&](auto& device, auto& queue)
        {
            constexpr std::uint32_t nx = 6;
            constexpr std::uint32_t ny = 5;
            constexpr std::uint32_t nz = 3;
            constexpr std::uint32_t mx = 13;
            constexpr std::uint32_t my = 10;
            constexpr std::uint32_t mz = 9;
            constexpr std::uint32_t zOffset = 4;
            constexpr double factor = 3.0 / 7.0;
            std::size_t const paddedCount = static_cast<std::size_t>(mx) * my * mz;
            std::size_t const compactCount = static_cast<std::size_t>(nx) * ny * nz;

            auto src = alpaka::onHost::allocUnified<Complex>(device, paddedCount);
            auto gatheredDev = alpaka::onHost::alloc<double>(device, compactCount);
            auto gathered = alpaka::onHost::allocMapped<double>(device, compactCount);
            std::vector<double> reference(compactCount);

            for(std::uint32_t k = 0; k < mz; ++k)
                for(std::uint32_t j = 0; j < my; ++j)
                    for(std::uint32_t i = 0; i < mx; ++i)
                        src.data()[
                            static_cast<std::size_t>(i)
                            + static_cast<std::size_t>(mx)
                                * (static_cast<std::size_t>(j)
                                   + static_cast<std::size_t>(my) * k)]
                            = Complex{sourceValue(i, j, k, 0), sourceValue(i, j, k, 1)};

            std::size_t l = 0;
            for(std::uint32_t k = 0; k < nz; ++k)
                for(std::uint32_t j = 0; j < ny; ++j)
                    for(std::uint32_t i = 0; i < nx; ++i, ++l)
                        reference[l]
                            = src.data()[
                                  static_cast<std::size_t>(i)
                                  + static_cast<std::size_t>(mx)
                                      * (static_cast<std::size_t>(j)
                                         + static_cast<std::size_t>(my)
                                             * (zOffset + k))]
                                  .real()
                            * factor;

            Complex* slabBase = src.data()
                + static_cast<std::size_t>(zOffset) * mx * my;
            queue.enqueue(
                alpaka::onHost::FrameSpec{
                    static_cast<std::uint32_t>((compactCount + 63u) / 64u),
                    64u},
                SlabGatherScaleKernel{},
                slabBase,
                gatheredDev.data(),
                nx,
                ny,
                nz,
                mx,
                my,
                factor);
            alpaka::onHost::memcpy(queue, gathered, gatheredDev);
            alpaka::onHost::wait(queue);

            ASSERT_EQ(
                0,
                std::memcmp(
                    gathered.data(),
                    reference.data(),
                    compactCount * sizeof(double)))
                << "rank-local slab base produced a shifted or wrong gather";
        });
}
