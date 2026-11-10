// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Shared executor selection for the host particle-bookkeeping reductions.
//
// Production call sites keep their own default (the far-tail pass uses the
// queue's default parallel executor, the binning pass uses cpuSerial).  The
// test suite overrides the choice here so the same production kernels can be
// run with both executors and their results compared bit for bit.
#pragma once

namespace picBunch::detail
{
    //! Executor override for the host reduction kernels.
    enum class HostReductionExecutor
    {
        //! Use the call site's production default.
        Default,
        //! Force alpaka::exec::cpuSerial.
        Serial,
        //! Force the queue's default (parallel) executor.
        Parallel
    };

    //! Process-wide override; tests set it, production leaves it at Default.
    inline HostReductionExecutor& hostReductionExecutorOverride() noexcept
    {
        static HostReductionExecutor value = HostReductionExecutor::Default;
        return value;
    }
} // namespace picBunch::detail
