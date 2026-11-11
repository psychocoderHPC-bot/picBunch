// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group 3: FFT.  Round-trip of a known array, cyclic convolution against a
// direct convolution on a tiny grid, and the R2C/C2R path against C2C on the
// same input.  The last comparison settles the packed DC/Nyquist handling
// empirically: preserving the imaginary part of those planes must match C2C,
// zeroing it must not.

#include "TestDevice.hpp"

#include "picBunch/Compute.hpp"
#include "picBunch/Parallel.hpp"
#include "picBunch/detail/SpectrumNorm.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/fft.hpp>
#include <alpaka/math/Complex.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
    using Complex = alpaka::math::Complex<double>;
    using Extents = alpaka::Vec<std::uint32_t, 3u>;

    constexpr std::uint32_t kSize = 4u;
    constexpr std::size_t kCount = std::size_t{kSize} * kSize * kSize;

    std::size_t flat(std::uint32_t i, std::uint32_t j, std::uint32_t k)
    {
        return i + std::size_t{kSize} * (j + std::size_t{kSize} * k);
    }

    double referenceValue(std::size_t l)
    {
        return std::sin(0.37 * static_cast<double>(l)) + 0.25 * std::cos(0.11 * static_cast<double>(l));
    }

    void fillReal(auto& buffer)
    {
        for(std::size_t l = 0; l < kCount; ++l)
            buffer.data()[l] = referenceValue(l);
    }
} // namespace

TEST(Fft, RoundTripKnownArray)
{
    picBunchTest::runOnDevice(
        [](auto& device, auto& queue)
        {
            Extents const ext = Extents::fill(kSize);
            auto in = alpaka::onHost::allocUnified<Complex>(device, ext);
            auto spectrum = alpaka::onHost::allocUnified<Complex>(device, ext);
            auto out = alpaka::onHost::allocUnified<Complex>(device, ext);
            auto plan = alpaka::fft::onHost::PlanBuilder<Complex, Extents>{ext}.c2c().build(device);

            for(std::size_t l = 0; l < kCount; ++l)
            {
                double const v = referenceValue(l);
                in.data()[l] = Complex{v, 0.0};
            }
            alpaka::fft::onHost::executeForward(queue, plan, in, spectrum);
            alpaka::fft::onHost::executeBackward(queue, plan, spectrum, out);
            alpaka::onHost::wait(queue);

            double const norm = 1.0 / static_cast<double>(kCount);
            double maxError = 0.0;
            for(std::size_t l = 0; l < kCount; ++l)
            {
                double const dr = out.data()[l].real() * norm - referenceValue(l);
                double const di = out.data()[l].imag() * norm;
                maxError = std::max(maxError, std::hypot(dr, di));
            }
            EXPECT_LT(maxError, 1.0e-12) << "round-trip error";

            // Delta pulse at the origin has an all-ones spectrum.
            for(std::size_t l = 0; l < kCount; ++l)
                in.data()[l] = Complex{0.0, 0.0};
            in.data()[0] = Complex{1.0, 0.0};
            alpaka::fft::onHost::executeForward(queue, plan, in, spectrum);
            alpaka::onHost::wait(queue);
            double maxDelta = 0.0;
            for(std::size_t l = 0; l < kCount; ++l)
                maxDelta = std::max(
                    maxDelta,
                    std::hypot(spectrum.data()[l].real() - 1.0, spectrum.data()[l].imag()));
            EXPECT_LT(maxDelta, 1.0e-12) << "delta spectrum";
        });
}

TEST(Fft, ConvolutionMatchesDirect)
{
    picBunchTest::runOnDevice(
        [](auto& device, auto& queue)
        {
            Extents const ext = Extents::fill(kSize);
            auto aReal = alpaka::onHost::allocUnified<double>(device, ext);
            auto bReal = alpaka::onHost::allocUnified<double>(device, ext);
            auto a = alpaka::onHost::allocUnified<Complex>(device, ext);
            auto b = alpaka::onHost::allocUnified<Complex>(device, ext);
            auto sa = alpaka::onHost::allocUnified<Complex>(device, ext);
            auto sb = alpaka::onHost::allocUnified<Complex>(device, ext);
            auto product = alpaka::onHost::allocUnified<Complex>(device, ext);
            auto out = alpaka::onHost::allocUnified<Complex>(device, ext);
            auto plan = alpaka::fft::onHost::PlanBuilder<Complex, Extents>{ext}.c2c().build(device);

            for(std::size_t l = 0; l < kCount; ++l)
            {
                double const av = std::sin(0.21 * static_cast<double>(l)) + 0.5;
                double const bv = std::cos(0.13 * static_cast<double>(l)) - 0.2;
                aReal.data()[l] = av;
                bReal.data()[l] = bv;
                a.data()[l] = Complex{av, 0.0};
                b.data()[l] = Complex{bv, 0.0};
            }
            alpaka::fft::onHost::executeForward(queue, plan, a, sa);
            alpaka::fft::onHost::executeForward(queue, plan, b, sb);
            alpaka::onHost::wait(queue);
            for(std::size_t l = 0; l < kCount; ++l)
            {
                double const ar = sa.data()[l].real();
                double const ai = sa.data()[l].imag();
                double const br = sb.data()[l].real();
                double const bi = sb.data()[l].imag();
                product.data()[l] = Complex{ar * br - ai * bi, ar * bi + ai * br};
            }
            alpaka::fft::onHost::executeBackward(queue, plan, product, out);
            alpaka::onHost::wait(queue);

            double const norm = 1.0 / static_cast<double>(kCount);
            double maxError = 0.0;
            for(std::uint32_t i = 0; i < kSize; ++i)
                for(std::uint32_t j = 0; j < kSize; ++j)
                    for(std::uint32_t k = 0; k < kSize; ++k)
                    {
                        double direct = 0.0;
                        for(std::uint32_t di = 0; di < kSize; ++di)
                            for(std::uint32_t dj = 0; dj < kSize; ++dj)
                                for(std::uint32_t dk = 0; dk < kSize; ++dk)
                                    direct += aReal.data()[flat(di, dj, dk)]
                                        * bReal.data()[flat(
                                            (i + kSize - di) % kSize,
                                            (j + kSize - dj) % kSize,
                                            (k + kSize - dk) % kSize)];
                        maxError = std::max(
                            maxError,
                            std::abs(out.data()[flat(i, j, k)].real() * norm - direct));
                    }
            EXPECT_LT(maxError, 1.0e-11) << "FFT convolution vs direct";
        });
}

TEST(Fft, R2cMatchesC2cAndPackedPlanesMustBePreserved)
{
    picBunchTest::runOnDevice(
        [](auto& device, auto& queue)
        {
            Extents const realExt = Extents::fill(kSize);
            std::uint32_t const cp = kSize / 2u + 1u;
            Extents const specExt{kSize, kSize, cp};

            auto aReal = alpaka::onHost::allocUnified<double>(device, realExt);
            auto bReal = alpaka::onHost::allocUnified<double>(device, realExt);
            auto a = alpaka::onHost::allocUnified<Complex>(device, realExt);
            auto b = alpaka::onHost::allocUnified<Complex>(device, realExt);
            auto sa = alpaka::onHost::allocUnified<Complex>(device, realExt);
            auto sb = alpaka::onHost::allocUnified<Complex>(device, realExt);
            auto c2cProduct = alpaka::onHost::allocUnified<Complex>(device, realExt);
            auto c2cOut = alpaka::onHost::allocUnified<Complex>(device, realExt);

            auto ra = alpaka::onHost::allocUnified<Complex>(device, specExt);
            auto rb = alpaka::onHost::allocUnified<Complex>(device, specExt);
            auto rProduct = alpaka::onHost::allocUnified<Complex>(device, specExt);
            auto rOut = alpaka::onHost::allocUnified<double>(device, realExt);
            auto rOutZeroed = alpaka::onHost::allocUnified<double>(device, realExt);

            auto c2c = alpaka::fft::onHost::PlanBuilder<Complex, Extents>{realExt}.c2c().build(device);
            auto r2c = alpaka::fft::onHost::PlanBuilder<double, Extents>{realExt}.r2c().outOfPlace().build(device);
            auto c2r = alpaka::fft::onHost::PlanBuilder<double, Extents>{realExt}.c2r().outOfPlace().build(device);

            for(std::size_t l = 0; l < kCount; ++l)
            {
                double const av = std::sin(0.21 * static_cast<double>(l)) + 0.5;
                double const bv = std::cos(0.13 * static_cast<double>(l)) - 0.2;
                aReal.data()[l] = av;
                bReal.data()[l] = bv;
                a.data()[l] = Complex{av, 0.0};
                b.data()[l] = Complex{bv, 0.0};
            }

            // C2C reference.
            alpaka::fft::onHost::executeForward(queue, c2c, a, sa);
            alpaka::fft::onHost::executeForward(queue, c2c, b, sb);
            alpaka::onHost::wait(queue);
            for(std::size_t l = 0; l < kCount; ++l)
            {
                double const ar = sa.data()[l].real();
                double const ai = sa.data()[l].imag();
                double const br = sb.data()[l].real();
                double const bi = sb.data()[l].imag();
                c2cProduct.data()[l] = Complex{ar * br - ai * bi, ar * bi + ai * br};
            }
            alpaka::fft::onHost::executeBackward(queue, c2c, c2cProduct, c2cOut);
            alpaka::onHost::wait(queue);

            // R2C product, packed half-spectra, planes preserved.
            alpaka::fft::onHost::executeForward(queue, r2c, aReal, ra);
            alpaka::fft::onHost::executeForward(queue, r2c, bReal, rb);
            alpaka::onHost::wait(queue);
            std::size_t const specCount = std::size_t{kSize} * kSize * cp;
            for(std::size_t l = 0; l < specCount; ++l)
            {
                double const ar = ra.data()[l].real();
                double const ai = ra.data()[l].imag();
                double const br = rb.data()[l].real();
                double const bi = rb.data()[l].imag();
                rProduct.data()[l] = Complex{ar * br - ai * bi, ar * bi + ai * br};
            }
            alpaka::fft::onHost::executeBackward(queue, c2r, rProduct, rOut);
            alpaka::onHost::wait(queue);

            double const norm = 1.0 / static_cast<double>(kCount);
            double maxPreserved = 0.0;
            for(std::size_t l = 0; l < kCount; ++l)
                maxPreserved = std::max(
                    maxPreserved,
                    std::abs(rOut.data()[l] * norm - c2cOut.data()[l].real() * norm));
            EXPECT_LT(maxPreserved, 1.0e-12) << "R2C vs C2C with preserved packed planes";

            // Negative control: zero the imaginary part of the kx=0 and kx=Nyquist
            // planes, which is what treating them as real 1-D transforms does.
            for(std::size_t l = 0; l < specCount; ++l)
            {
                std::uint32_t const kx = static_cast<std::uint32_t>(l % cp);
                double const re = rProduct.data()[l].real();
                double const im = (kx == 0u || kx + 1u == cp) ? 0.0 : rProduct.data()[l].imag();
                rProduct.data()[l] = Complex{re, im};
            }
            alpaka::fft::onHost::executeBackward(queue, c2r, rProduct, rOutZeroed);
            alpaka::onHost::wait(queue);
            double maxZeroed = 0.0;
            double scale = 0.0;
            for(std::size_t l = 0; l < kCount; ++l)
            {
                scale = std::max(scale, std::abs(c2cOut.data()[l].real() * norm));
                maxZeroed = std::max(
                    maxZeroed,
                    std::abs(rOutZeroed.data()[l] * norm - c2cOut.data()[l].real() * norm));
            }
            // This is the empirical verdict: zeroing the packed planes is wrong
            // by an O(1) fraction of the field, preserving them is exact.
            std::printf(
                "R2C vs C2C: preserved=%.3e (relative %.3e), zeroed=%.3e (relative %.3e)\n",
                maxPreserved,
                maxPreserved / (scale + 1.0e-30),
                maxZeroed,
                maxZeroed / (scale + 1.0e-30));
            EXPECT_GT(maxZeroed, 1.0e-2 * (scale + 1.0e-30))
                << "zeroing the DC/Nyquist imaginary planes must change the result";
        });
}

namespace
{
    //! Single-rank parallel context: copies the one block so the distributed
    //! path can be exercised in-process without MPI.
    struct SelfContext : picBunch::ParallelContext
    {
        void alltoallv(
            double* send,
            int const* sendCounts,
            int const* sendDispls,
            double* recv,
            int const* recvCounts,
            int const* recvDispls) const override
        {
            if(sendCounts != nullptr && recvCounts != nullptr && sendCounts[0] > 0)
                std::copy_n(send + sendDispls[0], sendCounts[0], recv + recvDispls[0]);
        }
    };
} // namespace

TEST(Fft, MixedLayoutPlanAndSpectrumCacheOrders)
{
    // Regression for the plan cache that was keyed on the padded extents only.
    // A process that alternates the replicated full-grid r2c/c2r plans with the
    // distributed batched c2c plans reused an incompatible cuFFT plan and
    // failed with CUFFT_EXEC_FAILED.  The complete plan key must rebuild on a
    // layout mismatch, and the kernel-spectrum cache must return the same
    // values as a fresh build.
    //
    // The sequences below start with the distributed layout (first) and then
    // with the replicated layout (second), and each repeats every layout three
    // times so the second-miss kernel insertion and the later cache hits are
    // exercised in both directions.
    picBunch::Bunch bunch;
    bunch.resize(4);
    bunch.x = {-2.0e-6, 0.5e-6, 1.5e-6, 3.0e-6};
    bunch.y = {0.5e-6, -1.5e-6, 2.0e-6, 0.0};
    bunch.z = {1.0e-6, -0.5e-6, 0.0, 2.5e-6};
    bunch.ux = {0.0, 0.3, -0.2, 0.5};
    bunch.uy = {0.0, -0.1, 0.4, 0.0};
    bunch.uz = {0.0, 0.2, 0.0, -0.3};
    bunch.weight = {1.0, 2.0, 0.5, 1.5};
    bunch.charge = {
        -picBunch::kElementaryCharge,
        -picBunch::kElementaryCharge,
        -picBunch::kElementaryCharge,
        -picBunch::kElementaryCharge};

    SelfContext context;
    picBunch::ComputeOptions options;
    options.cellSize = {1.0e-6, 1.0e-6, 1.0e-6};
    options.gridSize = {8, 8, 8};
    options.gridOrigin = {-4.0e-6, -4.0e-6, -4.0e-6};
    options.fftPadding = 2u;
    options.deterministicDeposit = true;
    options.parallel = &context;

    auto run = [&](picBunch::FftMode mode) {
        options.fftMode = mode;
        return picBunch::computeField(bunch, options);
    };

    auto expectBitIdentical = [](picBunch::Field const& expected, picBunch::Field const& actual) {
        for(int c = 0; c < 6; ++c)
        {
            if(expected.data[c].size() != actual.data[c].size())
            {
                ADD_FAILURE() << "component " << c << " size differs";
                continue;
            }
            EXPECT_EQ(
                0,
                std::memcmp(
                    expected.data[c].data(),
                    actual.data[c].data(),
                    expected.data[c].size() * sizeof(double)))
                << "component " << c << " differs between repeated calls";
        }
    };

    auto runOrder = [&](picBunch::FftMode first, picBunch::FftMode second) {
        std::array<picBunch::Field, 3> firstFields;
        std::array<picBunch::Field, 3> secondFields;
        for(int i = 0; i < 3; ++i)
        {
            firstFields[static_cast<std::size_t>(i)] = run(first);
            secondFields[static_cast<std::size_t>(i)] = run(second);
        }
        for(int i = 1; i < 3; ++i)
        {
            expectBitIdentical(firstFields[0], firstFields[static_cast<std::size_t>(i)]);
            expectBitIdentical(secondFields[0], secondFields[static_cast<std::size_t>(i)]);
        }
        return std::make_pair(firstFields[0], secondFields[0]);
    };

    // Distributed-first sequence, then replicated-first sequence.
    auto const distributedFirst
        = runOrder(picBunch::FftMode::Distributed, picBunch::FftMode::Replicated);
    auto const replicatedFirst
        = runOrder(picBunch::FftMode::Replicated, picBunch::FftMode::Distributed);

    expectBitIdentical(distributedFirst.first, replicatedFirst.second);
    expectBitIdentical(distributedFirst.second, replicatedFirst.first);

    double peak = 0.0;
    double worst = 0.0;
    for(int c = 0; c < 6; ++c)
        for(std::size_t l = 0; l < distributedFirst.second.data[c].size(); ++l)
        {
            peak = std::max(peak, std::abs(distributedFirst.second.data[c][l]));
            worst = std::max(
                worst,
                std::abs(
                    distributedFirst.second.data[c][l]
                    - replicatedFirst.first.data[c][l]));
        }
    ASSERT_GT(peak, 0.0);
    EXPECT_LT(worst / peak, 1.0e-12)
        << "the two layouts must agree after repeated alternating calls";
}

TEST(Fft, DistributedProductionMatchesReplicatedWithDeviceDeposit)
{
    // Regression: the device TSC deposit ignored rhoStride, so the distributed
    // path (stride 2, interleaved complex) deposited at half the index.  The
    // deterministic host deposit was correct, so this test must use the device
    // deposit (deterministicDeposit = false).
    picBunch::Bunch bunch;
    bunch.resize(4);
    bunch.x = {-2.0e-6, 0.5e-6, 1.5e-6, 3.0e-6};
    bunch.y = {0.5e-6, -1.5e-6, 2.0e-6, 0.0};
    bunch.z = {1.0e-6, -0.5e-6, 0.0, 2.5e-6};
    bunch.ux = {0.0, 0.3, -0.2, 0.5};
    bunch.uy = {0.0, -0.1, 0.4, 0.0};
    bunch.uz = {0.0, 0.2, 0.0, -0.3};
    bunch.weight = {1.0, 2.0, 0.5, 1.5};
    bunch.charge = {
        -picBunch::kElementaryCharge,
        -picBunch::kElementaryCharge,
        -picBunch::kElementaryCharge,
        -picBunch::kElementaryCharge};

    picBunch::ComputeOptions options;
    options.cellSize = {1.0e-6, 1.0e-6, 1.0e-6};
    options.gridSize = {16, 16, 16};
    options.gridOrigin = {-8.0e-6, -8.0e-6, -8.0e-6};
    options.fftPadding = 2u;
    options.deterministicDeposit = false;

    options.fftMode = picBunch::FftMode::Replicated;
    auto const replicated = picBunch::computeField(bunch, options);

    SelfContext context;
    options.fftMode = picBunch::FftMode::Distributed;
    options.parallel = &context;
    auto const distributed = picBunch::computeField(bunch, options);

    double peak = 0.0;
    double worst = 0.0;
    for(int c = 0; c < 6; ++c)
    {
        ASSERT_EQ(replicated.data[c].size(), distributed.data[c].size());
        for(std::size_t l = 0; l < replicated.data[c].size(); ++l)
        {
            peak = std::max(peak, std::abs(replicated.data[c][l]));
            worst = std::max(
                worst, std::abs(replicated.data[c][l] - distributed.data[c][l]));
        }
    }
    ASSERT_GT(peak, 0.0);
    EXPECT_LT(worst / peak, 1.0e-12)
        << "distributed production field must match the replicated field";

    // The automatic policy must select the same validated path above its
    // padded-node threshold.
    options.fftMode = picBunch::FftMode::Auto;
    options.distributedFftMinNodes = 1u;
    auto const automatic = picBunch::computeField(bunch, options);
    double autoWorst = 0.0;
    for(int c = 0; c < 6; ++c)
        for(std::size_t l = 0; l < replicated.data[c].size(); ++l)
            autoWorst = std::max(
                autoWorst,
                std::abs(replicated.data[c][l] - automatic.data[c][l]));
    EXPECT_LT(autoWorst / peak, 1.0e-12)
        << "auto-selected distributed path must match the replicated field";
}

TEST(Fft, SpectrumNormMatchesSerialReference)
{
    picBunchTest::runOnDevice(
        [](auto& device, auto& queue)
        {
            constexpr std::uint32_t count = 257u;
            auto full = alpaka::onHost::allocUnified<double>(device, 2u * count);
            auto order3 = alpaka::onHost::allocUnified<double>(device, 2u * count);
            auto accNorm = alpaka::onHost::allocUnified<double>(device, 2u);

            // Same input on both sides, including negative and mixed-magnitude
            // components so the accumulation order matters.
            std::vector<double> refFull(2u * count);
            std::vector<double> refOrder3(2u * count);
            for(std::uint32_t l = 0; l < 2u * count; ++l)
            {
                refFull[l] = referenceValue(l) * ((l % 3u) == 0u ? -1.0 : 1.0);
                refOrder3[l] = 1.0e-3 * referenceValue(l + 7u);
                full.data()[l] = refFull[l];
                order3.data()[l] = refOrder3[l];
            }
            accNorm.data()[0] = 0.0;
            accNorm.data()[1] = 0.0;

            queue.enqueue(
                alpaka::onHost::FrameSpec{1u, 1u},
                picBunch::detail::SpectrumNormKernel{},
                static_cast<double const*>(full.data()),
                static_cast<double const*>(order3.data()),
                count,
                accNorm.data());
            alpaka::onHost::wait(queue);

            // Independent serial reference: exactly the kernel's order.
            double ref0 = 0.0;
            double ref1 = 0.0;
            for(std::uint32_t i = 0; i < count; ++i)
            {
                double const fr = refFull[2u * i];
                double const fi = refFull[2u * i + 1u];
                double const orr = refOrder3[2u * i];
                double const oi = refOrder3[2u * i + 1u];
                ref0 += fr * fr + fi * fi;
                ref1 += orr * orr + oi * oi;
            }
            // The addition order is fixed by the single-worker kernel.  A
            // device backend may still contract the multiply-add into an FMA,
            // so the comparison is a documented tolerance rather than a bit
            // comparison; the CPU backend matches bit for bit.
            EXPECT_NEAR(accNorm.data()[0], ref0, 1.0e-13 * std::max(1.0, std::abs(ref0)));
            EXPECT_NEAR(accNorm.data()[1], ref1, 1.0e-13 * std::max(1.0, std::abs(ref1)));
        });
}
