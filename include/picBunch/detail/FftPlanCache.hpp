// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/fft.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <utility>

namespace picBunch
{
    namespace detail
    {
        //! Which buffer layout a plan was built for.
        //!
        //! The replicated path transforms a dense full grid; the distributed
        //! path transforms a rank-local slab through batched, strided c2c
        //! plans.  A plan from one layout is never valid for the other, even
        //! when the extents happen to match.
        enum class FftLayoutKind : std::uint8_t
        {
            Replicated = 0u,
            DistributedSlab = 1u
        };

        /*! Complete identity of an FFT plan.
         *
         * The previous single-slot cache keyed on ``(Mx, My, Mz)`` only.  That
         * is not enough: a cuFFT plan also captures the transform type, the
         * batch/strides/distances, the placement, the value type and the
         * device.  Reusing it across layouts produced ``CUFFT_EXEC_FAILED``
         * when a process alternated between the replicated full-grid r2c/c2r
         * plans and the distributed batched c2c plans.  Every field that the
         * vendor library reads at plan construction is part of this key.
         */
        struct FftPlanKey
        {
            std::type_index valueType{typeid(void)};
            std::string backend;
            int device = -1;
            int transform = 0;
            std::uint32_t dim = 0u;
            std::array<std::uint32_t, 3u> extents{};
            std::uint32_t batch = 1u;
            std::array<std::size_t, 3u> inStrides{};
            std::array<std::size_t, 3u> outStrides{};
            std::size_t inDistance = 0u;
            std::size_t outDistance = 0u;
            int placement = 0;
            FftLayoutKind layout = FftLayoutKind::Replicated;
            std::array<std::uint32_t, 3u> shape{};
            //! nx, nz, z0, ranks; all zero for a replicated plan.
            std::array<std::uint32_t, 4u> slab{};

            bool operator==(FftPlanKey const&) const = default;

            bool operator<(FftPlanKey const& other) const
            {
                return std::tie(
                           valueType,
                           backend,
                           device,
                           transform,
                           dim,
                           extents,
                           batch,
                           inStrides,
                           outStrides,
                           inDistance,
                           outDistance,
                           placement,
                           layout,
                           shape,
                           slab)
                    < std::tie(
                           other.valueType,
                           other.backend,
                           other.device,
                           other.transform,
                           other.dim,
                           other.extents,
                           other.batch,
                           other.inStrides,
                           other.outStrides,
                           other.inDistance,
                           other.outDistance,
                           other.placement,
                           other.layout,
                           other.shape,
                           other.slab);
            }
        };

        template<typename T_Device>
        inline std::string fftBackendIdentity(T_Device& device)
        {
            std::string const api
                = typeid(std::remove_cvref_t<decltype(device.getApi())>).name();
            return api + "#" + std::to_string(static_cast<long long>(device.getNativeHandle()));
        }

        template<typename T_Vec>
        inline std::array<std::uint32_t, 3u> fftExtentArray(T_Vec const& extents)
        {
            std::array<std::uint32_t, 3u> result{};
            for(std::uint32_t i = 0u; i < T_Vec::dim() && i < 3u; ++i)
                result[i] = static_cast<std::uint32_t>(extents[i]);
            return result;
        }

        template<typename T_Vec>
        inline std::array<std::size_t, 3u> fftStrideArray(T_Vec const& strides)
        {
            std::array<std::size_t, 3u> result{};
            for(std::uint32_t i = 0u; i < T_Vec::dim() && i < 3u; ++i)
                result[i] = static_cast<std::size_t>(strides[i]);
            return result;
        }

        //! Key for a plan with the default contiguous layout, batch 1 and out-of-place placement.
        template<typename T_Value, typename T_Device, typename T_Extents>
        inline FftPlanKey makeFftPlanKey(
            T_Device& device,
            alpaka::fft::Transform transform,
            T_Extents const& extents,
            FftLayoutKind layout,
            std::array<std::uint32_t, 3u> const& shape,
            std::array<std::uint32_t, 4u> const& slab = {})
        {
            FftPlanKey key;
            key.valueType = std::type_index{typeid(T_Value)};
            key.backend = fftBackendIdentity(device);
            key.device = static_cast<int>(device.getNativeHandle());
            key.transform = static_cast<int>(transform);
            key.dim = T_Extents::dim();
            key.extents = fftExtentArray(extents);
            key.batch = 1u;
            key.placement = static_cast<int>(alpaka::fft::Placement::outOfPlace);
            key.layout = layout;
            key.shape = shape;
            key.slab = slab;
            return key;
        }

        //! Key for a fully specified plan layout.
        template<typename T_Value, typename T_Device, typename T_Extents, typename T_Strides>
        inline FftPlanKey makeFftPlanKey(
            T_Device& device,
            alpaka::fft::Transform transform,
            T_Extents const& extents,
            std::uint32_t batch,
            T_Strides const& inStrides,
            T_Strides const& outStrides,
            std::size_t inDistance,
            std::size_t outDistance,
            alpaka::fft::Placement placement,
            FftLayoutKind layout,
            std::array<std::uint32_t, 3u> const& shape,
            std::array<std::uint32_t, 4u> const& slab)
        {
            FftPlanKey key = makeFftPlanKey<T_Value>(device, transform, extents, layout, shape, slab);
            key.batch = batch;
            key.inStrides = fftStrideArray(inStrides);
            key.outStrides = fftStrideArray(outStrides);
            key.inDistance = inDistance;
            key.outDistance = outDistance;
            key.placement = static_cast<int>(placement);
            return key;
        }

        /*! Process-lifetime, per-thread FFT plan cache.
         *
         * Plans are shape/layout-only objects (vendor handle plus layout), so
         * they can be reused across ``computeField`` calls and across the
         * replicated and distributed branches of the same process.  The lookup
         * key is complete; a different key builds a new plan instead of
         * reusing an incompatible one.  A small LRU bound keeps the number of
         * vendor handles finite for a process that sweeps many shapes.
         */
        template<typename T_Plan>
        class FftPlanCache
        {
        public:
            template<typename T_Builder>
            static std::shared_ptr<T_Plan> acquire(FftPlanKey const& key, T_Builder&& builder)
            {
                struct Entry
                {
                    std::shared_ptr<T_Plan> plan;
                    typename std::list<FftPlanKey>::iterator lru;
                };
                struct State
                {
                    std::map<FftPlanKey, Entry> entries;
                    std::list<FftPlanKey> order;
                };

                static auto* states = new std::map<std::thread::id, State>{};
                static auto* guard = new std::mutex{};
                std::lock_guard lock{*guard};
                State& state = (*states)[std::this_thread::get_id()];

                auto it = state.entries.find(key);
                if(it != state.entries.end())
                {
                    state.order.splice(state.order.begin(), state.order, it->second.lru);
                    return it->second.plan;
                }

                if(state.entries.size() >= kMaxPlansPerThread)
                {
                    FftPlanKey const victim = state.order.back();
                    state.order.pop_back();
                    state.entries.erase(victim);
                }

                auto plan = std::make_shared<T_Plan>(builder());
                state.order.push_front(key);
                state.entries.emplace(key, Entry{plan, state.order.begin()});
                return plan;
            }

        private:
            static constexpr std::size_t kMaxPlansPerThread = 24u;
        };
    } // namespace detail
} // namespace picBunch
