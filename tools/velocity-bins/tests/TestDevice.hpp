// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include <gtest/gtest.h>

#include <alpaka/alpaka.hpp>

namespace picBunchTest
{
    /** Run ``fn(device, queue)`` on the compiled backend's device zero.
     *
     * The device type depends on the backend, so the selection happens inside
     * the preprocessor branch and the callable is a template.  ``fn`` must
     * return void; gtest assertions are allowed inside it.
     */
    template<typename T_Fn>
    void runOnDevice(T_Fn&& fn)
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
            ADD_FAILURE() << "no device available for the compiled backend";
            return;
        }
        auto device = selector.makeDevice(0);
        auto queue = device.makeQueue();
        fn(device, queue);
    }
} // namespace picBunchTest
