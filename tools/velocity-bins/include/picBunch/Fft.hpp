// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/fft.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace picBunch
{
    /** Result of a minimal complex-to-complex FFT self test.
     *
     * ``maxDeltaSpectrumError`` is the largest deviation of the forward
     * transform of a delta pulse at the origin from an all-ones spectrum.
     * ``maxRoundTripError`` is measured after dividing the backward transform
     * by ``n^3`` (the vendor plans are unnormalized).
     */
    struct FftSelfTest
    {
        bool planBuilt = false;
        double maxDeltaSpectrumError = 0.0;
        double maxRoundTripError = 0.0;
        std::string backend;
    };

    namespace detail
    {
        constexpr char const* fftBackendName()
        {
#if defined(ALPAKAV_DEP_CUFFT) && ALPAKAV_DEP_CUFFT
            return "cuFFT";
#elif defined(ALPAKAV_DEP_ROCFFT) && ALPAKAV_DEP_ROCFFT
            return "rocFFT";
#elif defined(ALPAKAV_DEP_FFTW) && ALPAKAV_DEP_FFTW
            return "FFTW";
#else
            return "none";
#endif
        }
    } // namespace detail

    /** Build one double-precision 3-D C2C plan on ``device`` and verify it.
     *
     * The plan is created through the alpakaVendor portable FFT API, so the
     * call exercises cuFFT, rocFFT or FFTW depending on the build.  The
     * device and queue types stay generic: alpaka's device types are templated
     * on api and device kind.
     */
    template<typename T_Device, typename T_Queue>
    FftSelfTest runFftSelfTest(T_Device& device, T_Queue& queue, std::uint32_t n)
    {
        using Complex = alpaka::math::Complex<double>;
        using Extents = alpaka::Vec<std::uint32_t, 3u>;

        FftSelfTest result;
        result.backend = detail::fftBackendName();

        std::size_t const count = std::size_t{n} * n * n;
        auto const extents = Extents::fill(n);

        auto in = alpaka::onHost::allocUnified<Complex>(device, extents);
        auto spectrum = alpaka::onHost::allocUnified<Complex>(device, extents);
        auto out = alpaka::onHost::allocUnified<Complex>(device, extents);

        auto plan = alpaka::fft::onHost::PlanBuilder<Complex, Extents>{extents}.c2c().build(device);
        result.planBuilt = true;

        // delta pulse at the origin: the forward transform is exactly all ones
        for(std::size_t l = 0; l < count; ++l)
            in.data()[l] = Complex{0.0, 0.0};
        in.data()[0] = Complex{1.0, 0.0};
        alpaka::fft::onHost::executeForward(queue, plan, in, spectrum);
        alpaka::onHost::wait(queue);
        for(std::size_t l = 0; l < count; ++l)
            result.maxDeltaSpectrumError = std::max(
                result.maxDeltaSpectrumError,
                std::hypot(spectrum.data()[l].real() - 1.0, spectrum.data()[l].imag()));

        // deterministic pattern: forward + backward must return count*input
        std::vector<Complex> reference(count);
        for(std::size_t l = 0; l < count; ++l)
        {
            double const v = std::sin(0.1 * static_cast<double>(l)) + 0.5 * std::cos(0.03 * static_cast<double>(l));
            in.data()[l] = Complex{v, 0.0};
            reference[l] = Complex{v, 0.0};
        }
        alpaka::fft::onHost::executeForward(queue, plan, in, spectrum);
        alpaka::fft::onHost::executeBackward(queue, plan, spectrum, out);
        alpaka::onHost::wait(queue);
        double const norm = 1.0 / static_cast<double>(count);
        for(std::size_t l = 0; l < count; ++l)
        {
            double const dr = out.data()[l].real() * norm - reference[l].real();
            double const di = out.data()[l].imag() * norm - reference[l].imag();
            result.maxRoundTripError = std::max(result.maxRoundTripError, std::hypot(dr, di));
        }

        return result;
    }

} // namespace picBunch
