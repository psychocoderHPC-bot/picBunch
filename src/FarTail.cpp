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
        // Both the charge-weighted and the absolute-charge-weighted mean
        // velocity are accumulated in the same pass.  Which one to use is
        // decided once, from the TOTAL charge, after the loop: a neutral bunch
        // uses the absolute-charge weights.  The weighting therefore never
        // flips with the running partial charge sum, which made the direction
        // of a nearly neutral bunch effectively noise.
        std::array<double, 3> velocitySumCharge{0.0, 0.0, 0.0};
        std::array<double, 3> velocitySumAbs{0.0, 0.0, 0.0};
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
            double const invGamma = 1.0 / gamma;
            for(int a = 0; a < 3; ++a)
            {
                double const u = (a == 0) ? bunch.ux[p] : (a == 1) ? bunch.uy[p] : bunch.uz[p];
                double const v = kSpeedOfLight * u * invGamma; // particle velocity
                velocitySumCharge[a] += q * v;
                velocitySumAbs[a] += std::abs(q) * v;
            }
        }
        // The weighting choice depends on the TOTAL charge only.  A bunch whose
        // net charge is negligible against its absolute content is neutral and
        // uses the absolute-charge weights, so the mean velocity direction stays
        // defined.
        bool const neutral
            = (absoluteCharge == 0.0)
            || (std::abs(tail.charge) <= 1.0e-12 * absoluteCharge);
        std::array<double, 3> const& velocitySum = neutral ? velocitySumAbs : velocitySumCharge;
        double const velocityWeight = neutral ? absoluteCharge : tail.charge;
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

        // Motion of the compact bunch: beta = meanVelocity / c, with
        // 1/gamma^2 = 1 - |beta|^2.  bhat is the unit direction of motion.
        double const c = kSpeedOfLight;
        std::array<double, 3> const beta{
            tail.meanVelocity[0] / c,
            tail.meanVelocity[1] / c,
            tail.meanVelocity[2] / c};
        double const betaSq = beta[0] * beta[0] + beta[1] * beta[1] + beta[2] * beta[2];
        // A valid bunch has a subluminal mean velocity; a superluminal mean (a
        // pathological mixed-sign, near-neutral bunch with a small net charge
        // whose charge-weighted mean diverges) would make 1 - |beta|^2
        // non-positive, so clamp it to keep the field finite.
        double const invG2 = std::max(1.0 - betaSq, 1.0e-15);
        double const gamma = 1.0 / std::sqrt(invG2);
        double const bnorm = std::sqrt(betaSq);
        std::array<double, 3> const bhat{
            (bnorm > 0.0) ? beta[0] / bnorm : 0.0,
            (bnorm > 0.0) ? beta[1] / bnorm : 0.0,
            (bnorm > 0.0) ? beta[2] / bnorm : 0.0};

        // Static dipole + quadrupole field (the monopole is handled separately,
        // as the exact Lienard point field below).
        double const dipoleScale = kCoulombConstant * 2.0 / (d2 * d);
        double const quadrupoleScale = kCoulombConstant * 0.5 / (d2 * d2);
        std::array<double, 3> high{};
        for(int a = 0; a < 3; ++a)
        {
            double const dipoleTerm = 1.5 * pDotN * n[a] - 0.5 * tail.dipole[a];
            double const quadrupoleTerm = 5.0 * nDotDn * n[a] - 2.0 * dTimesN[a];
            high[a] = dipoleScale * dipoleTerm + quadrupoleScale * quadrupoleTerm;
        }

        // Asymptotic moving field of a uniformly moving compact bunch.
        //
        // Monopole: the exact Lienard point field of the total charge moving at
        // the mean velocity.  This is exact for a point source in every
        // direction; it is the on-axis 1/gamma^2 / transverse gamma form.
        double const rb = r[0] * beta[0] + r[1] * beta[1] + r[2] * beta[2];
        double const denom = d2 * invG2 + rb * rb; // r^2 (1-beta^2) + (r.beta)^2
        double const monopoleF
            = (denom > 0.0)
            ? kCoulombConstant * tail.charge * invG2 / (denom * std::sqrt(denom))
            : 0.0;

        // Higher multipoles: keep their static shape but transform it by the
        // motion, scaling the component parallel to beta by 1/gamma^2 and the
        // perpendicular one by gamma.  This is an asymptotic moving multipole,
        // not an exact Lienard multipole sum.  For beta = 0 the transform is the
        // identity and the whole expression reduces exactly to the static
        // multipole field (monopole + dipole + quadrupole, no gammaMax factor).
        if(betaSq > 0.0)
        {
            double const ePar
                = high[0] * bhat[0] + high[1] * bhat[1] + high[2] * bhat[2];
            double const s = invG2 - gamma;
            for(int a = 0; a < 3; ++a)
                high[a] = gamma * high[a] + s * ePar * bhat[a];
        }
        std::array<double, 3> field{};
        for(int a = 0; a < 3; ++a)
            field[a] = high[a] + monopoleF * r[a];
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
