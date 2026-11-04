// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Trivial end-to-end smoke test: proves that the pinned alpaka/alpakaVendor
// stack, the selected backend, the FFT plan and the openPMD/HDF5 build are
// usable.  It is not a physics test; the numerical gates live in docs/algorithm.md
// and are filled in during validation.

#include "picBunch/Deposit.hpp"
#include "picBunch/Fft.hpp"
#include "picBunch/OpenPMD.hpp"
#include "picBunch/Types.hpp"

#include <alpaka/alpaka.hpp>

#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

namespace
{
    template<typename T_Device, typename T_Queue>
    int run(T_Device& device, T_Queue& queue)
    {
        std::printf("picBunch smoke test\n");
        std::printf("  alpaka        : %d.%d.%d\n", ALPAKA_VERSION_MAJOR, ALPAKA_VERSION_MINOR, ALPAKA_VERSION_PATCH);
        std::printf(
            "  api / kind    : %s / %s\n",
            std::string{alpaka::onHost::getStaticName(alpaka::getApi(device))}.c_str(),
            std::string{alpaka::onHost::getStaticName(alpaka::getDeviceKind(device))}.c_str());
        std::printf("  device        : %s\n", device.getName().c_str());

        // deposit: one macro particle at the grid centre must deposit exactly its charge
        picBunch::Grid grid;
        grid.n = {16, 16, 16};
        grid.spacing = {1.0, 1.0, 1.0};
        grid.origin = {0.0, 0.0, 0.0};
        picBunch::Bunch bunch;
        bunch.resize(1);
        bunch.x[0] = 8.0;
        bunch.y[0] = 8.0;
        bunch.z[0] = 8.0;
        bunch.weight[0] = 1.0;
        bunch.charge[0] = 1.0;
        auto const rho = picBunch::depositTSC(bunch, grid);
        double const deposited = std::accumulate(rho.begin(), rho.end(), 0.0);
        std::printf("  deposit       : sum(rho)=%.15e (expected 1)\n", deposited);
        bool const depositOk = std::abs(deposited - 1.0) < 1.0e-12;

        // FFT plan build and a delta/round-trip self test
        auto const fft = picBunch::runFftSelfTest(device, queue, 16u);
        std::printf("  fft backend   : %s\n", fft.backend.c_str());
        std::printf("  fft plan      : %s\n", fft.planBuilt ? "build OK" : "FAILED");
        std::printf("  fft deltaspec : max|F-1|=%.3e\n", fft.maxDeltaSpectrumError);
        std::printf("  fft roundtrip : max|out/N^3-in|=%.3e\n", fft.maxRoundTripError);
        bool const fftOk = fft.planBuilt && fft.maxDeltaSpectrumError < 1.0e-10 && fft.maxRoundTripError < 1.0e-10;

        auto const io = picBunch::openPMDInfo();
        std::printf("  openPMD       : %s (HDF5 backend: %s)\n", io.version.c_str(), io.hdf5 ? "yes" : "no");

        bool const ok = depositOk && fftOk && io.hdf5;
        std::printf("%s\n", ok ? "OK" : "FAILED");
        return ok ? 0 : 1;
    }
} // namespace

int main()
{
#if defined(PICBUNCH_BACKEND_CUDA)
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu);
#elif defined(PICBUNCH_BACKEND_HIP)
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::hip, alpaka::deviceKind::amdGpu);
#else
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::host, alpaka::deviceKind::cpu);
#endif
    if(!selector.isAvailable())
    {
        std::fprintf(stderr, "smoke: no device available for the selected backend\n");
        return 1;
    }
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    return run(device, queue);
}
