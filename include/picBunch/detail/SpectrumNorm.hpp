// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Deterministic k-space norm accumulation used by the a-posteriori probe.
#pragma once

#include <alpaka/alpaka.hpp>

#include <cstdint>

namespace picBunch::detail
{
    //! Deterministic k-space norm accumulation.
    //!
    //! One worker walks the half-spectrum in the original ``i`` order and
    //! accumulates directly into the persistent two-double buffer, so the
    //! floating-point addition order (and therefore the probe) is identical to
    //! the former serial host loop, independent of the device block size.
    struct SpectrumNormKernel
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            double const* full,
            double const* order3,
            std::uint32_t count,
            double* accNorm) const
        {
            for(auto [unused] : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInGrid,
                    alpaka::IdxRange{std::size_t{1u}}))
            {
                alpaka::unused(unused);
                for(std::uint32_t i = 0; i < count; ++i)
                {
                    double const fr = full[2u * i];
                    double const fi = full[2u * i + 1u];
                    double const orr = order3[2u * i];
                    double const oi = order3[2u * i + 1u];
                    accNorm[0] += fr * fr + fi * fi;
                    accNorm[1] += orr * orr + oi * oi;
                }
            }
        }
    };
} // namespace picBunch::detail
