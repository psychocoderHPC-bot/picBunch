// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Scalars that are only min/max extrema of particle arrays are order
// independent, so they are folded with the alpaka host reduction instead of a
// hand-written serial accumulator.  The functors carry the ADL
// ``alpakaAtomicInvoke`` overload the alpaka reduction needs to write the block
// result, and they reduce Simd packs directly so no ``ScalarFunc`` wrapper is
// required.
//
// ``HostReductionExecutor::Serial`` and ``Default`` keep the historical
// single-threaded ``cpuSerial`` fold.  ``Parallel`` selects alpaka's OpenMP
// block executor so the reduction really runs on the host thread pool; when
// alpaka is built without OpenMP it falls back to ``cpuSerial``.
#pragma once

#include "picBunch/detail/HostReduction.hpp"

#include <alpaka/alpaka.hpp>

#include <cstddef>

namespace picBunch::detail
{
    //! Element-wise minimum for scalars and alpaka Simd packs.
    template<typename T>
    ALPAKA_FN_HOST_ACC constexpr auto elementMin(T const& a, T const& b)
    {
        if constexpr(alpaka::concepts::Simd<T>)
        {
            auto result = a;
            alpaka::where(b < a, result) = b;
            return result;
        }
        else
            return (b < a) ? b : a;
    }

    //! Element-wise maximum for scalars and alpaka Simd packs.
    template<typename T>
    ALPAKA_FN_HOST_ACC constexpr auto elementMax(T const& a, T const& b)
    {
        if constexpr(alpaka::concepts::Simd<T>)
        {
            auto result = a;
            alpaka::where(a < b, result) = b;
            return result;
        }
        else
            return (a < b) ? b : a;
    }

    struct MinValue
    {
        template<typename T>
        ALPAKA_FN_HOST_ACC constexpr auto operator()(T const& a, T const& b) const
        {
            return elementMin(a, b);
        }
    };

    ALPAKA_FN_ACC void alpakaAtomicInvoke(
        MinValue const&,
        alpaka::onAcc::concepts::Acc auto const& acc,
        auto* dest,
        auto const src)
    {
        alpaka::onAcc::atomicMin(acc, dest, src);
    }

    struct MaxValue
    {
        template<typename T>
        ALPAKA_FN_HOST_ACC constexpr auto operator()(T const& a, T const& b) const
        {
            return elementMax(a, b);
        }
    };

    ALPAKA_FN_ACC void alpakaAtomicInvoke(
        MaxValue const&,
        alpaka::onAcc::concepts::Acc auto const& acc,
        auto* dest,
        auto const src)
    {
        alpaka::onAcc::atomicMax(acc, dest, src);
    }

    //! Dedicated CPU-side alpaka queue for the host reductions.
    inline auto& reductionQueue()
    {
        static auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::host, alpaka::deviceKind::cpu);
        static auto device = selector.makeDevice(0);
        static auto queue = device.makeQueue();
        return queue;
    }

    //! One-element device result, reused per thread.
    template<typename T>
    auto& reductionDeviceSlot()
    {
        static thread_local auto slot
            = alpaka::onHost::allocDeferred<T>(reductionQueue(), alpaka::Vec<std::size_t, 1u>{1u});
        return slot;
    }

    //! Host mirror of the one-element reduction result, reused per thread.
    template<typename T>
    auto& reductionHostSlot()
    {
        static thread_local auto slot = alpaka::onHost::allocHostLike(reductionDeviceSlot<T>());
        return slot;
    }

    //! Executor selected by ``HostReductionExecutor::Parallel``.
    //!
    //! The OpenMP block executor exists only when alpaka saw an OpenMP-enabled
    //! compiler.  Without OpenMP the enum keeps its historical serial meaning so
    //! this header still compiles and behaves correctly on a bare host build.
#if ALPAKA_OMP
    inline constexpr auto parallelHostReductionExecutor = alpaka::exec::cpuOmpBlocks;
#else
    inline constexpr auto parallelHostReductionExecutor = alpaka::exec::cpuSerial;
#endif

    //! Run a reduction callable with the executor selected by the override.
    template<typename T_Fn>
    void dispatchHostReductionExecutor(HostReductionExecutor executor, T_Fn&& fn)
    {
        switch(executor)
        {
        case HostReductionExecutor::Parallel:
            fn(parallelHostReductionExecutor);
            break;
        case HostReductionExecutor::Serial:
        case HostReductionExecutor::Default:
        default:
            fn(alpaka::exec::cpuSerial);
            break;
        }
    }

    template<typename T_Value, typename T_Reduce, typename... TInputs>
    T_Value runReduce(T_Value neutral, T_Reduce const& reduceFn, TInputs const&... inputs)
    {
        auto& queue = reductionQueue();
        auto& device = reductionDeviceSlot<T_Value>();
        auto& host = reductionHostSlot<T_Value>();
        dispatchHostReductionExecutor(
            hostReductionExecutorOverride(),
            [&](auto const executor)
            {
                alpaka::onHost::reduce(queue, executor, neutral, device, reduceFn, inputs...);
            });
        alpaka::onHost::memcpy(queue, host, device);
        alpaka::onHost::wait(queue);
        return *host.data();
    }

    template<typename T_Value, typename T_Reduce, typename T_Transform, typename... TInputs>
    T_Value runTransformReduce(
        T_Value neutral,
        T_Reduce const& reduceFn,
        T_Transform const& transformFn,
        TInputs const&... inputs)
    {
        auto& queue = reductionQueue();
        auto& device = reductionDeviceSlot<T_Value>();
        auto& host = reductionHostSlot<T_Value>();
        dispatchHostReductionExecutor(
            hostReductionExecutorOverride(),
            [&](auto const executor)
            {
                alpaka::onHost::transformReduce(
                    queue, executor, neutral, device, reduceFn, transformFn, inputs...);
            });
        alpaka::onHost::memcpy(queue, host, device);
        alpaka::onHost::wait(queue);
        return *host.data();
    }

    template<typename T_Value, alpaka::concepts::IMdSpan TInput>
    T_Value reduceMin(TInput const& input, T_Value neutral)
    {
        return runReduce(neutral, MinValue{}, input);
    }

    template<typename T_Value, alpaka::concepts::IMdSpan TInput>
    T_Value reduceMax(TInput const& input, T_Value neutral)
    {
        return runReduce(neutral, MaxValue{}, input);
    }

    template<typename T_Value, typename T_Transform, typename... TInputs>
    T_Value transformReduceMin(T_Value neutral, T_Transform const& transformFn, TInputs const&... inputs)
    {
        return runTransformReduce(neutral, MinValue{}, transformFn, inputs...);
    }

    template<typename T_Value, typename T_Transform, typename... TInputs>
    T_Value transformReduceMax(T_Value neutral, T_Transform const& transformFn, TInputs const&... inputs)
    {
        return runTransformReduce(neutral, MaxValue{}, transformFn, inputs...);
    }
} // namespace picBunch::detail
