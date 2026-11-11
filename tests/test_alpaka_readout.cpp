// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group AlpakaReadout: the read-out element-wise host paths now run through
// alpaka onHost algorithms / small index-mapped kernels.  These tests pin the
// two properties that make that replacement safe:
//
//  1. agreement with an independent plain serial loop, using the same operation
//     order and the same double types, within the documented read-out tolerance
//     (<= 1 ulp of the result or <= 2e-15 absolute; bit-identical passes);
//  2. determinism that does not depend on the executor's thread count.
//
// The functor mirrors src/Compute.cpp: ReadoutScaleAdd is the plain (SIMD-safe)
// transform functor used by the pitched 3D read-out and the compact accumulate.
// It is deliberately SIMD-safe (no scalar-only wrapper), so alpaka may lower it
// onto SIMD packs; the tolerance below bounds the last-bit difference that a
// different vector lowering can introduce.  The distributed read-out now
// uses the gather/scale kernel covered by the MpiReadout group.

#include "TestDevice.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/math/Complex.hpp>
#include <gtest/gtest.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

namespace
{
    //! Same SIMD-safe functor as Compute.cpp's read-out transform.  The generic
    //! arguments let alpaka pass Simd packs; no scalar-only wrapper is used.
    struct ReadoutScaleAdd
    {
        double factor;

        ALPAKA_FN_ACC auto operator()(auto const& acc, auto const& in) const
        {
            return acc + in * factor;
        }
    };

    //! Read-out tolerance pinned by the SIMD-safe functor: a value passes when
    //! it is bit-identical, or within one ulp of the reference, or within an
    //! absolute 2e-15.  Returns the number of elements outside the bound and the
    //! worst observed ulp / absolute / relative distance for the test log.
    struct ReadoutDifference
    {
        std::size_t violating{};
        std::size_t differing{};
        double maxAbs{};
        double maxRel{};
        double maxUlp{};
    };

    ReadoutDifference readoutDifference(
        double const* got,
        double const* ref,
        std::size_t count)
    {
        ReadoutDifference d;
        for(std::size_t l = 0; l < count; ++l)
        {
            double const a = got[l];
            double const b = ref[l];
            if(std::memcmp(&a, &b, sizeof(double)) == 0)
                continue;
            ++d.differing;
            double const diff = std::abs(a - b);
            double const mag = std::max(std::abs(a), std::abs(b));
            double const ulp = std::abs(
                std::nextafter(b, std::numeric_limits<double>::infinity()) - b);
            double const ulpDistance = (ulp > 0.0) ? diff / ulp : diff;
            d.maxAbs = std::max(d.maxAbs, diff);
            d.maxRel = std::max(d.maxRel, (mag > 0.0) ? diff / mag : diff);
            d.maxUlp = std::max(d.maxUlp, ulpDistance);
            // <= 1 ulp or <= 2e-15 absolute.
            if(!(ulpDistance <= 1.0 || diff <= 2.0e-15))
                ++d.violating;
        }
        return d;
    }

    void expectReadoutWithinTolerance(
        double const* got,
        double const* ref,
        std::size_t count,
        char const* label)
    {
        auto const d = readoutDifference(got, ref, count);
        std::printf(
            "AlpakaReadout %s: differing=%zu/%zu maxUlp=%.3g maxAbs=%.3g maxRel=%.3g\n",
            label,
            d.differing,
            count,
            d.maxUlp,
            d.maxAbs,
            d.maxRel);
        EXPECT_EQ(0u, d.violating)
            << label << ": " << d.differing << "/" << count
            << " elements differ; maxUlp=" << d.maxUlp << " maxAbs=" << d.maxAbs
            << " maxRel=" << d.maxRel
            << " (allowed: <= 1 ulp or <= 2e-15 absolute)";
    }

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

TEST(AlpakaReadout, PitchedTransformMatchesSerialLoopWithinTolerance)
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
                        ReadoutScaleAdd{factor},
                        outView,
                        inView);
                    alpaka::onHost::wait(queue);
                }

                std::vector<double> const reference = pitchedReference(s, initial, factors);
                expectReadoutWithinTolerance(
                    acc.data(),
                    reference.data(),
                    compactCount,
                    "pitched transform");
            }
        });
}

TEST(AlpakaReadout, CompactAccumulateMatchesSerialLoopWithinTolerance)
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
                    ReadoutScaleAdd{factor},
                    accView,
                    stagedView);
                alpaka::onHost::wait(queue);
            }

            expectReadoutWithinTolerance(
                acc.data(),
                reference.data(),
                count,
                "compact accumulate");
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
                    ReadoutScaleAdd{factor},
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
                ReadoutScaleAdd{factor},
                hostParallelView,
                hostSrcView);
            alpaka::onHost::transform(
                hostQueue,
                alpaka::exec::cpuSerial,
                hostSerialView,
                ReadoutScaleAdd{factor},
                hostSerialView,
                hostSrcView);
            alpaka::onHost::wait(hostQueue);
            expectReadoutWithinTolerance(
                hostAccParallel.data(),
                hostAccSerial.data(),
                count,
                "host parallel vs serial executor");
            // The backend-queue default must agree with the host serial result.
            expectReadoutWithinTolerance(
                accParallel.data(),
                hostAccSerial.data(),
                count,
                "backend vs host serial");
        });
}

//! Compile- and run-time pin for the span/index agreement introduced with the
//! ported kernels: a size_t extent vector must produce a span whose extent
//! value type is size_t, and indexing with the same type must round-trip.
TEST(AlpakaReadout, SpanExtentIndexTypeAgreement)
{
    using Index1D = alpaka::Vec<std::size_t, 1u>;
    std::vector<double> data(7u);
    Index1D const extents{data.size()};
    auto span = alpaka::makeMdSpan(data.data(), extents);
    static_assert(std::is_same_v<
                  typename std::remove_cvref_t<decltype(span.getExtents())>::value_type,
                  Index1D::value_type>);
    for(std::size_t i = 0; i < data.size(); ++i)
        span[Index1D{i}] = static_cast<double>(i) + 0.5;
    for(std::size_t i = 0; i < data.size(); ++i)
        EXPECT_EQ(span[Index1D{i}], static_cast<double>(i) + 0.5);
    EXPECT_EQ(span.getExtents()[0u], data.size());
}
