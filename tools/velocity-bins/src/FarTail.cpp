// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Multipole far field used to fill a global output domain outside the exact
// compute box.  This replaces the former exact-zero fill; the expansion is
// valid beyond ``FarTail::switchDistance`` and is measured against an
// independent direct sum in the validation suite.

#include "picBunch/FarTail.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace picBunch
{

    FarTail deriveFarTail(Bunch const& bunch, double switchFactor)
    {
        if(bunch.size() == 0u)
            throw std::runtime_error("deriveFarTail: empty bunch");
        if(!(switchFactor > 0.0) || !std::isfinite(switchFactor))
            throw std::invalid_argument("deriveFarTail: switchFactor must be finite and positive");

        auto const bounds = bunch.bounds();
        std::array<double, 3> centre{};
        for(int a = 0; a < 3; ++a)
            centre[a] = 0.5 * (bounds[0][a] + bounds[1][a]);

        FarTail tail;
        tail.centre = centre;
        double absoluteCharge = 0.0;
        std::array<double, 3> velocitySum{0.0, 0.0, 0.0};
        for(std::size_t p = 0; p < bunch.size(); ++p)
        {
            double const q = bunch.weight[p] * bunch.charge[p];
            std::array<double, 3> const rel{
                bunch.x[p] - centre[0],
                bunch.y[p] - centre[1],
                bunch.z[p] - centre[2]};
            tail.charge += q;
            absoluteCharge += std::abs(q);
            for(int a = 0; a < 3; ++a)
                tail.dipole[a] += q * rel[a];
            double const r2 = rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2];
            for(int i = 0; i < 3; ++i)
                for(int j = 0; j < 3; ++j)
                    tail.quadrupole[i * 3 + j]
                        += q * (3.0 * rel[i] * rel[j] - (i == j ? r2 : 0.0));
            tail.supportRadius = std::max(tail.supportRadius, std::sqrt(r2));

            double const u2 = bunch.ux[p] * bunch.ux[p] + bunch.uy[p] * bunch.uy[p]
                + bunch.uz[p] * bunch.uz[p];
            double const gamma = std::sqrt(1.0 + u2);
            tail.gammaMax = std::max(tail.gammaMax, gamma);
            // Charge-weighted mean velocity; a nearly neutral bunch falls back to
            // the absolute-charge weights so the direction is still defined.
            double const weightVelocity = (std::abs(tail.charge) > 0.0) ? q : std::abs(q);
            for(int a = 0; a < 3; ++a)
            {
                double const u = (a == 0) ? bunch.ux[p] : (a == 1) ? bunch.uy[p] : bunch.uz[p];
                velocitySum[a] += weightVelocity * kSpeedOfLight * u / gamma;
            }
        }
        double const velocityWeight
            = (std::abs(tail.charge) > 0.0) ? tail.charge : absoluteCharge;
        if(velocityWeight != 0.0)
            for(int a = 0; a < 3; ++a)
                tail.meanVelocity[a] = velocitySum[a] / velocityWeight;
        tail.switchDistance = switchFactor * tail.supportRadius;
        return tail;
    }

    std::array<double, 3> evaluateFarTailE(
        FarTail const& tail,
        std::array<double, 3> const& position)
    {
        std::array<double, 3> const r{
            position[0] - tail.centre[0],
            position[1] - tail.centre[1],
            position[2] - tail.centre[2]};
        double const d2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        if(d2 == 0.0)
            return {0.0, 0.0, 0.0};
        double const d = std::sqrt(d2);
        std::array<double, 3> const n{r[0] / d, r[1] / d, r[2] / d};

        double const pDotN
            = tail.dipole[0] * n[0] + tail.dipole[1] * n[1] + tail.dipole[2] * n[2];
        std::array<double, 3> dTimesN{};
        for(int i = 0; i < 3; ++i)
            dTimesN[i] = tail.quadrupole[i * 3 + 0] * n[0]
                + tail.quadrupole[i * 3 + 1] * n[1]
                + tail.quadrupole[i * 3 + 2] * n[2];
        double const nDotDn
            = n[0] * dTimesN[0] + n[1] * dTimesN[1] + n[2] * dTimesN[2];

        double const monopole = kCoulombConstant * tail.gammaMax * tail.charge / d2;
        double const dipoleScale = kCoulombConstant * tail.gammaMax * 2.0 / (d2 * d);
        double const quadrupoleScale
            = kCoulombConstant * tail.gammaMax * 0.5 / (d2 * d2);
        std::array<double, 3> field{};
        for(int a = 0; a < 3; ++a)
        {
            double const dipoleTerm = 1.5 * pDotN * n[a] - 0.5 * tail.dipole[a];
            double const quadrupoleTerm = 5.0 * nDotDn * n[a] - 2.0 * dTimesN[a];
            field[a] = monopole * n[a] + dipoleScale * dipoleTerm
                + quadrupoleScale * quadrupoleTerm;
        }
        return field;
    }

    std::array<double, 3> evaluateFarTailB(
        FarTail const& tail,
        std::array<double, 3> const& position)
    {
        auto const e = evaluateFarTailE(tail, position);
        std::array<double, 3> const& v = tail.meanVelocity;
        double const inverseC2 = 1.0 / (kSpeedOfLight * kSpeedOfLight);
        return {
            (v[1] * e[2] - v[2] * e[1]) * inverseC2,
            (v[2] * e[0] - v[0] * e[2]) * inverseC2,
            (v[0] * e[1] - v[1] * e[0]) * inverseC2};
    }

} // namespace picBunch
