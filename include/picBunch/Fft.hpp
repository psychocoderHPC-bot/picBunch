// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <string>

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

    /** Build one double-precision 3-D C2C plan on ``device`` and verify it.
     *
     * The plan is created through the alpakaVendor portable FFT API, so the
     * call exercises cuFFT, rocFFT or FFTW depending on the build.
     */
    FftSelfTest runFftSelfTest(alpaka::onHost::Device& device, alpaka::onHost::Queue& queue, std::uint32_t n);

} // namespace picBunch
