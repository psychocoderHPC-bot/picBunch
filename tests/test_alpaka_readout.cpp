// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group AlpakaReadout: the read-out element-wise host paths now run through
// alpaka onHost algorithms / small index-mapped kernels.  These tests pin the
// two properties that make that replacement safe:
//
//  1. bit-identical numerics against an independent plain serial loop, using
//     the same operation order and the same double types;
//  2. determinism that does not depend on the executor's thread count, checked
//     by comparing the default (parallel) executor with the serial executor.
//
// The functors mirror src/Compute.cpp: ReadoutScaleAdd is the ScalarFunc used
// by the pitched 3D read-out and the compact accumulate; SlabScaleKernel is the
// indexed complex scale of the distributed read-out.

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
    //! Same scalar functor as Compute.cpp's read-out transform.
    struct ReadoutScaleAdd
    {
        double factor;

        ALPAKA_FN_ACC double operator()(double acc, double in) const
        {
            return acc + in * factor;
        }
    };

    //! Same indexed complex scale as Compute.cpp's distributed read-out.
    struct SlabScaleKernel
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            alpaka::math::Complex<double>* work,
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
                auto& value = work[idx];
                value = alpaka::math::Complex<double>{value.real() * factor, 0.0};
            }
        }
    };

    //! Deterministic, exactly representable value; a wrong stride or order gives
    //! a different double, so a bitwise compare is meaningful.
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

    struct PaddedShape
    {
        std::uint32_t nx;
        std::uint32_t ny;
        std::uint32_t nz;
        std::uint32_t mx;
        std::uint32_t my;
        std::uint32_t mz;
    };

    //! Reference: pitched 3D accumulate in plain serial loops.
    std::vector<double> pitchedReference(
        PaddedShape const& s,
        std::vector<double> const& initial,
        std::array<double, 3> const& factors)
    {
        std::vector<double> out = initial;
        for(int pass = 0; pass < 3; ++pass)
        {
            for(std::uint32_t k = 0; k < s.nz; ++k)
                for(std::uint32_t j = 0; j < s.ny; ++j)
                    for(std::uint32_t i = 0; i < s.nx; ++i)
                    {
                        std::size_t const outIdx = static_cast<std::size_t>(i)
                            + static_cast<std::size_t>(s.nx)
                                * (static_cast<std::size_t>(j)
                                   + static_cast<std::size_t>(s.ny) * static_cast<std::size_t>(k));
                        out[outIdx] = out[outIdx] + sourceValue(i, j, k, pass) * factors[static_cast<std::size_t>(pass)];
                    }
        }
        return out;
    }
} // namespace

TEST(AlpakaReadout, PitchedTransformMatchesSerialLoopBitwise)
{
    std::array<PaddedShape, 4> const shapes{{
        {5, 7, 3, 5, 7, 3},
        {7, 5, 3, 14, 15, 9},
        {6, 5, 4, 12, 10, 8},
        {9, 7, 5, 18, 21, 15},
    }};

    picBunchTest::runOnDevice(
        [&](auto& device, auto& queue)
        {
            for(PaddedShape const& s : shapes)
            {
                std::size_t const paddedCount
                    = static_cast<std::size_t>(s.mx) * s.my * s.mz;
                std::size_t const compactCount
                    = static_cast<std::size_t>(s.nx) * s.ny * s.nz;
                std::array<double, 3> const factors{{1.5, 0.125, 3.0}};

                std::vector<double> initial(compactCount);
                std::vector<double> sourceOfPass(paddedCount);
                // Non-zero initial accumulator proves the transform accumulates
                // rather than overwrites; the padded halo is filled with
                // non-zero garbage so a missing pitch difference shows up.
                for(std::size_t l = 0; l < compactCount; ++l)
                    initial[l] = static_cast<double>(static_cast<std::int64_t>(l % 37u) - 18)
                        / 256.0;

                auto src = alpaka::onHost::allocUnified<double>(device, paddedCount);
                auto acc = alpaka::onHost::allocUnified<double>(device, compactCount);
                for(std::size_t l = 0; l < compactCount; ++l)
                    acc.data()[l] = initial[l];

                auto const extents
                    = alpaka::Vec<std::uint32_t, 3u>{s.nz, s.ny, s.nx};
                auto const accPitches = alpaka::Vec<std::size_t, 3u>{
                    static_cast<std::size_t>(s.nx) * s.ny * sizeof(double),
                    static_cast<std::size_t>(s.nx) * sizeof(double),
                    sizeof(double)};
                auto const srcPitches = alpaka::Vec<std::size_t, 3u>{
                    static_cast<std::size_t>(s.mx) * s.my * sizeof(double),
                    static_cast<std::size_t>(s.mx) * sizeof(double),
                    sizeof(double)};

                for(int pass = 0; pass < 3; ++pass)
                {
                    for(std::uint32_t k = 0; k < s.mz; ++k)
                        for(std::uint32_t j = 0; j < s.my; ++j)
                            for(std::uint32_t i = 0; i < s.mx; ++i)
                                sourceOfPass[static_cast<std::size_t>(i)
                                    + static_cast<std::size_t>(s.mx)
                                        * (static_cast<std::size_t>(j)
                                           + static_cast<std::size_t>(s.my) * k)]
                                    = sourceValue(i, j, k, pass);
                    for(std::size_t l = 0; l < paddedCount; ++l)
                        src.data()[l] = sourceOfPass[l];

                    auto outView = alpaka::makeMdSpan(acc.data(), extents, accPitches);
                    auto inView = alpaka::makeMdSpan(src.data(), extents, srcPitches);
                    double const factor = factors[static_cast<std::size_t>(pass)];
                    alpaka::onHost::transform(
                        queue,
                        outView,
                        alpaka::ScalarFunc{ReadoutScaleAdd{factor}},
                        outView,
                        inView);
                    alpaka::onHost::wait(queue);
                }

                std::vector<double> const reference = pitchedReference(s, initial, factors);
                ASSERT_EQ(
                    0,
                    std::memcmp(acc.data(), reference.data(), compactCount * sizeof(double)))
                    << "shape " << s.nx << "x" << s.ny << "x" << s.nz << " padded "
                    << s.mx << "x" << s.my << "x" << s.mz;
            }
        });
}

TEST(AlpakaReadout, CompactAccumulateMatchesSerialLoopBitwise)
{
    picBunchTest::runOnDevice(
        [&](auto& device, auto& queue)
        {
            std::uint32_t const count = 10007u;
            std::array<double, 3> const factors{{0.75, -1.25, 2.0}};

            std::vector<double> initial(count);
            for(std::uint32_t l = 0; l < count; ++l)
                initial[l] = static_cast<double>(static_cast<std::int64_t>(l % 29u) - 14)
                    / 128.0;

            auto acc = alpaka::onHost::allocUnified<double>(device, count);
            auto staged = alpaka::onHost::allocUnified<double>(device, count);
            for(std::uint32_t l = 0; l < count; ++l)
                acc.data()[l] = initial[l];

            std::vector<double> reference = initial;
            for(int pass = 0; pass < 3; ++pass)
            {
                double const factor = factors[static_cast<std::size_t>(pass)];
                for(std::uint32_t l = 0; l < count; ++l)
                    staged.data()[l] = sourceValue(l % 101u, l / 101u, pass, pass);
                for(std::uint32_t l = 0; l < count; ++l)
                    reference[l] = reference[l] + staged.data()[l] * factor;

                auto const extents = alpaka::Vec<std::uint32_t, 1u>{count};
                auto accView = alpaka::makeMdSpan(acc.data(), extents);
                auto stagedView = alpaka::makeMdSpan(staged.data(), extents);
                alpaka::onHost::transform(
                    queue,
                    accView,
                    alpaka::ScalarFunc{ReadoutScaleAdd{factor}},
                    accView,
                    stagedView);
                alpaka::onHost::wait(queue);
            }

            ASSERT_EQ(0, std::memcmp(acc.data(), reference.data(), count * sizeof(double)));
        });
}

TEST(AlpakaReadout, DistributedSlabScaleMatchesSerialLoopBitwise)
{
    std::array<PaddedShape, 3> const shapes{{
        {5, 7, 3, 5, 7, 3},
        {7, 5, 3, 14, 15, 9},
        {4, 6, 5, 9, 11, 5},
    }};

    picBunchTest::runOnDevice(
        [&](auto& device, auto& queue)
        {
            using Complex = alpaka::math::Complex<double>;
            for(PaddedShape const& s : shapes)
            {
                // The slab covers nz output planes with all nx/ny nodes.
                std::size_t const workCount
                    = static_cast<std::size_t>(s.mx) * s.my * s.mz;
                std::vector<Complex> host(workCount);
                for(std::uint32_t k = 0; k < s.mz; ++k)
                    for(std::uint32_t j = 0; j < s.my; ++j)
                        for(std::uint32_t i = 0; i < s.mx; ++i)
                            host[static_cast<std::size_t>(i)
                                + static_cast<std::size_t>(s.mx)
                                    * (static_cast<std::size_t>(j)
                                       + static_cast<std::size_t>(s.my) * k)]
                                = Complex{
                                    sourceValue(i, j, k, 11),
                                    sourceValue(i, j, k, 12)};

                auto work = alpaka::onHost::allocUnified<Complex>(device, workCount);
                for(std::size_t l = 0; l < workCount; ++l)
                    work.data()[l] = host[l];

                std::uint64_t const scaleCount
                    = static_cast<std::uint64_t>(s.nx) * s.ny * s.nz;
                queue.enqueue(
                    alpaka::onHost::FrameSpec{
                        static_cast<std::uint32_t>((scaleCount + 63u) / 64u),
                        64u},
                    SlabScaleKernel{},
                    work.data(),
                    s.nx,
                    s.ny,
                    s.nz,
                    s.mx,
                    s.my,
                    1.375);
                alpaka::onHost::wait(queue);

                std::vector<Complex> const reference = [&]
                {
                    std::vector<Complex> out = host;
                    for(std::uint32_t k = 0; k < s.nz; ++k)
                        for(std::uint32_t j = 0; j < s.ny; ++j)
                            for(std::uint32_t i = 0; i < s.nx; ++i)
                            {
                                std::size_t const idx = static_cast<std::size_t>(i)
                                    + static_cast<std::size_t>(s.mx)
                                        * (static_cast<std::size_t>(j)
                                           + static_cast<std::size_t>(s.my)
                                               * static_cast<std::size_t>(k));
                                Complex const v = out[idx];
                                out[idx] = Complex{v.real() * 1.375, 0.0};
                            }
                    return out;
                }();

                ASSERT_EQ(
                    0,
                    std::memcmp(work.data(), reference.data(), workCount * sizeof(Complex)))
                    << "shape " << s.nx << "x" << s.ny << "x" << s.nz << " padded "
                    << s.mx << "x" << s.my << "x" << s.mz;
            }
        });
}

TEST(AlpakaReadout, TransformResultIsIndependentOfExecutorThreadCount)
{
    picBunchTest::runOnDevice(
        [&](auto& device, auto& queue)
        {
            std::uint32_t const count = 20011u;
            double const factor = 1.0 + 1.0 / 64.0;

            auto src = alpaka::onHost::allocUnified<double>(device, count);
            auto accParallel = alpaka::onHost::allocUnified<double>(device, count);
            auto accSerial = alpaka::onHost::allocUnified<double>(device, count);
            auto accRepeat = alpaka::onHost::allocUnified<double>(device, count);
            for(std::uint32_t l = 0; l < count; ++l)
            {
                src.data()[l] = sourceValue(l % 97u, l / 97u, 3, 5);
                accParallel.data()[l] = static_cast<double>(l % 13u) / 32.0;
                accSerial.data()[l] = accParallel.data()[l];
                accRepeat.data()[l] = accParallel.data()[l];
            }

            auto const extents = alpaka::Vec<std::uint32_t, 1u>{count};
            auto srcView = alpaka::makeMdSpan(src.data(), extents);

            auto runTransform = [&](auto& acc)
            {
                auto accView = alpaka::makeMdSpan(acc.data(), extents);
                alpaka::onHost::transform(
                    queue,
                    accView,
                    alpaka::ScalarFunc{ReadoutScaleAdd{factor}},
                    accView,
                    srcView);
                alpaka::onHost::wait(queue);
            };

            runTransform(accParallel);
            runTransform(accRepeat);
            ASSERT_EQ(
                0,
                std::memcmp(accParallel.data(), accRepeat.data(), count * sizeof(double)))
                << "default executor is not deterministic across repeated runs";

            // Serial vs the default (parallel) executor on a host CPU queue.
            // This comparison runs on every backend: the read-out transform is
            // element-wise and backend independent, and the host CPU device is
            // the one device whose serial executor is available in CUDA and HIP
            // builds too.
            auto hostSelector = alpaka::onHost::makeDeviceSelector(
                alpaka::api::host,
                alpaka::deviceKind::cpu);
            ASSERT_TRUE(hostSelector.isAvailable())
                << "no host CPU device available for the executor comparison";
            auto hostDevice = hostSelector.makeDevice(0);
            auto hostQueue = hostDevice.makeQueue();
            auto hostSrc = alpaka::onHost::allocUnified<double>(hostDevice, count);
            auto hostAccParallel = alpaka::onHost::allocUnified<double>(hostDevice, count);
            auto hostAccSerial = alpaka::onHost::allocUnified<double>(hostDevice, count);
            for(std::uint32_t l = 0; l < count; ++l)
            {
                hostSrc.data()[l] = sourceValue(l % 97u, l / 97u, 3, 5);
                hostAccParallel.data()[l] = static_cast<double>(l % 13u) / 32.0;
                hostAccSerial.data()[l] = hostAccParallel.data()[l];
            }
            auto hostSrcView = alpaka::makeMdSpan(hostSrc.data(), extents);
            auto hostParallelView = alpaka::makeMdSpan(hostAccParallel.data(), extents);
            auto hostSerialView = alpaka::makeMdSpan(hostAccSerial.data(), extents);
            alpaka::onHost::transform(
                hostQueue,
                hostParallelView,
                alpaka::ScalarFunc{ReadoutScaleAdd{factor}},
                hostParallelView,
                hostSrcView);
            alpaka::onHost::transform(
                hostQueue,
                alpaka::exec::cpuSerial,
                hostSerialView,
                alpaka::ScalarFunc{ReadoutScaleAdd{factor}},
                hostSerialView,
                hostSrcView);
            alpaka::onHost::wait(hostQueue);
            ASSERT_EQ(
                0,
                std::memcmp(
                    hostAccParallel.data(),
                    hostAccSerial.data(),
                    count * sizeof(double)))
                << "read-out result depends on the executor thread count";
            // The backend-queue default must agree with the host serial result.
            ASSERT_EQ(
                0,
                std::memcmp(accParallel.data(), hostAccSerial.data(), count * sizeof(double)))
                << "read-out result depends on the backend";
        });
}
