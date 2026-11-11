// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Host policy helpers of the velocity-moment spectral decomposition: the
// absolute-charge-weighted expansion centre ``u0`` and the velocity-space radius
// used by the feasibility/fallback decision.  See ``picBunch/VelocityMoments.hpp``
// for the method.

#include "picBunch/VelocityMoments.hpp"

#include "picBunch/detail/Reduction.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace picBunch
{
    namespace
    {
        //! Squared distance of the beam-frame momentum from the expansion centre.
        struct MomentumDistanceSquared
        {
            Vec3 u0;

            template<typename T>
            ALPAKA_FN_ACC constexpr auto operator()(T const& ux, T const& uy, T const& uz) const
            {
                auto const dx = ux - u0[0];
                auto const dy = uy - u0[1];
                auto const dz = uz - u0[2];
                return dx * dx + dy * dy + dz * dz;
            }
        };
    } // namespace

    Vec3 velocityMomentsCentre(Bunch const& bunch)
    {
        std::size_t const n = bunch.size();
        if(n == 0u)
            throw std::runtime_error("velocityMomentsCentre: empty bunch");
        double sum[3] = {0.0, 0.0, 0.0};
        double w = 0.0;
        for(std::size_t p = 0; p < n; ++p)
        {
            double const ap = std::abs(bunch.weight[p] * bunch.charge[p]);
            if(!(std::isfinite(bunch.ux[p]) && std::isfinite(bunch.uy[p])
                 && std::isfinite(bunch.uz[p])))
                throw std::invalid_argument("velocityMomentsCentre: a momentum is not finite");
            sum[0] += ap * bunch.ux[p];
            sum[1] += ap * bunch.uy[p];
            sum[2] += ap * bunch.uz[p];
            w += ap;
        }
        if(w == 0.0)
            throw std::runtime_error("velocityMomentsCentre: the bunch carries no charge");
        return {sum[0] / w, sum[1] / w, sum[2] / w};
    }

    double velocityMomentsRadius(Bunch const& bunch, Vec3 const& u0)
    {
        std::size_t const n = bunch.size();
        if(n == 0u)
            return 0.0;
        auto const extent = alpaka::Vec<std::size_t, 1u>{n};
        // sqrt is monotonic, so the largest squared distance belongs to the
        // particle with the largest distance; taking the square root once is
        // bit-identical to the former per-particle maximum of sqrt.
        double const r2 = detail::transformReduceMax(
            0.0,
            MomentumDistanceSquared{u0},
            alpaka::makeMdSpan(bunch.ux.data(), extent),
            alpaka::makeMdSpan(bunch.uy.data(), extent),
            alpaka::makeMdSpan(bunch.uz.data(), extent));
        return std::sqrt(r2);
    }

} // namespace picBunch
