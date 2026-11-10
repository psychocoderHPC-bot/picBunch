// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group Readout: the replicated read-out gather (CompactGatherKernel) must map
// the valid n^3 subregion of the padded c2r output to the compact host buffer
// with exactly the intended x-fastest indexing.  A determinism test that runs
// computeField twice cannot catch a deterministic wrong gather, because both
// runs share the same wrong mapping.  These tests compare all six output
// components against an independent host gather, bit for bit, on shapes with
// non-cubic extents and an odd output size.

#include "picBunch/Readout.hpp"

#include "TestDevice.hpp"

#include <alpaka/alpaka.hpp>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace
{
    struct GatherShape
    {
        std::uint32_t nx;
        std::uint32_t ny;
        std::uint32_t nz;
        std::uint32_t mx;
        std::uint32_t my;
        std::uint32_t mz;
    };

    //! Deterministic, asymmetric, exactly representable source value.
    //!
    //! The hashed index terms make a transposed or shifted gather produce a
    //! different double; the /1024.0 keeps every value exact so the bitwise
    //! comparison cannot be masked by rounding.  ``component`` separates the
    //! six field components so a component mix-up is also visible.
    double gatherSourceValue(std::uint32_t i, std::uint32_t j, std::uint32_t k, int component)
    {
        std::uint64_t h = static_cast<std::uint64_t>(i) * 1000003u
                        + static_cast<std::uint64_t>(j) * 10007u
                        + static_cast<std::uint64_t>(k) * 131u
                        + static_cast<std::uint64_t>(component + 1) * 17u;
        h ^= h >> 17;
        h *= 0x9e3779b97f4a7c15ull;
        h ^= h >> 29;
        return static_cast<double>(static_cast<std::int64_t>(h % 2000001ull) - 1000000ll) / 1024.0;
    }

    char const* const kComponentNames[6] = {"Ex", "Ey", "Ez", "Bx", "By", "Bz"};
} // namespace

TEST(Readout, CompactGatherMatchesHostBitwiseAllComponents)
{
    // Non-cubic shapes catch an x/y/z stride transposition; the odd output
    // extents catch indexing that assumes a power-of-two or even grid.  Padded
    // extents differ from the compact extents in every case so a gather that
    // forgets the padding stride also fails.
    std::array<GatherShape, 4> const shapes{{
        {5, 7, 3, 5, 7, 3},      // non-cubic, unpadded
        {7, 5, 3, 14, 15, 9},    // non-cubic odd output, padded
        {6, 5, 4, 12, 10, 8},    // non-cubic, padded
        {9, 7, 5, 18, 21, 15},   // non-cubic odd output, padded
    }};

    picBunchTest::runOnDevice(
        [&](auto& device, auto& queue)
        {
            for(GatherShape const& s : shapes)
            {
                std::size_t const paddedCount
                    = static_cast<std::size_t>(s.mx) * s.my * s.mz;
                std::uint32_t const compactCount = s.nx * s.ny * s.nz;

                auto src = alpaka::onHost::allocUnified<double>(device, paddedCount);
                auto gatheredDev = alpaka::onHost::alloc<double>(device, compactCount);
                auto gathered = alpaka::onHost::allocMapped<double>(device, compactCount);
                std::vector<double> hostReference(compactCount);
                std::vector<double> sourceSnapshot(paddedCount);

                for(int c = 0; c < 6; ++c)
                {
                    for(std::uint32_t k = 0; k < s.mz; ++k)
                        for(std::uint32_t j = 0; j < s.my; ++j)
                            for(std::uint32_t i = 0; i < s.mx; ++i)
                                sourceSnapshot[
                                    static_cast<std::size_t>(i)
                                    + static_cast<std::size_t>(s.mx)
                                        * (static_cast<std::size_t>(j)
                                           + static_cast<std::size_t>(s.my) * k)]
                                    = gatherSourceValue(i, j, k, c);
                    for(std::size_t l = 0; l < paddedCount; ++l)
                        src.data()[l] = sourceSnapshot[l];

                    auto const paddedExtents = alpaka::Vec<std::uint32_t, 3u>{s.mz, s.my, s.mx};
                    auto const compactExtents = alpaka::Vec<std::uint32_t, 1u>{compactCount};
                    picBunch::detail::gatherCompactHost(
                        alpaka::makeMdSpan(sourceSnapshot.data(), paddedExtents),
                        alpaka::makeMdSpan(hostReference.data(), compactExtents),
                        s.nx,
                        s.ny,
                        s.nz);

                    queue.enqueue(
                        alpaka::onHost::FrameSpec{(compactCount + 63u) / 64u, 64u},
                        picBunch::detail::CompactGatherKernel{},
                        alpaka::makeMdSpan(src.data(), paddedExtents),
                        alpaka::makeMdSpan(gatheredDev.data(), compactExtents),
                        s.nx,
                        s.ny,
                        s.nz);
                    alpaka::onHost::memcpy(queue, gathered, gatheredDev);
                    alpaka::onHost::wait(queue);

                    ASSERT_EQ(
                        0,
                        std::memcmp(
                            gathered.data(),
                            hostReference.data(),
                            compactCount * sizeof(double)))
                        << "component " << kComponentNames[c] << " shape " << s.nx << "x"
                        << s.ny << "x" << s.nz << " padded " << s.mx << "x" << s.my << "x"
                        << s.mz << ": device gather differs from the host reference";
                }
            }
        });
}
