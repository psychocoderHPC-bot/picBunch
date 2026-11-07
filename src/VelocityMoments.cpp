// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Host policy helpers of the velocity-moment spectral decomposition: the
// absolute-charge-weighted expansion centre ``u0`` and the velocity-space radius
// used by the feasibility/fallback decision.  See ``picBunch/VelocityMoments.hpp``
// for the method.

#include "picBunch/VelocityMoments.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace picBunch
{
    std::array<double, 3> velocityMomentsCentre(Bunch const& bunch)
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

    double velocityMomentsRadius(Bunch const& bunch, std::array<double, 3> const& u0)
    {
        double r = 0.0;
        for(std::size_t p = 0; p < bunch.size(); ++p)
        {
            double const d[3] = {
                bunch.ux[p] - u0[0],
                bunch.uy[p] - u0[1],
                bunch.uz[p] - u0[2]};
            r = std::max(r, std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]));
        }
        return r;
    }

} // namespace picBunch
