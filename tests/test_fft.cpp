// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group 3: FFT.  Round-trip of a known array, cyclic convolution against a
// direct convolution on a tiny grid, and the R2C/C2R path against C2C on the
// same input.  The last comparison settles the packed DC/Nyquist handling
// empirically: preserving the imaginary part of those planes must match C2C,
// zeroing it must not.

#include "TestDevice.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/fft.hpp>
#include <alpaka/math/Complex.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

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
            EXPECT_GT(maxZeroed, 1.0e-2 * (scale + 1.0e-30))
                << "zeroing the DC/Nyquist imaginary planes must change the result";
        });
}
