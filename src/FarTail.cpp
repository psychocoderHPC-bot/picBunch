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
            double const r2 = rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2];
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

        // Dipole and quadrupole are accumulated in the bunch rest frame.  For a
        // rigidly moving distribution the field is the Lorentz transform of the
        // rest-frame static field, so the moments must be taken there: the
        // longitudinal separation is stretched by gamma.  Keeping lab-frame
        // moments and scaling the lab static field is not the same operation
        // for the higher multipoles.
        double const meanSpeed2 = tail.meanVelocity[0] * tail.meanVelocity[0]
            + tail.meanVelocity[1] * tail.meanVelocity[1]
            + tail.meanVelocity[2] * tail.meanVelocity[2];
        double const meanGamma2
            = 1.0 / std::max(1.0 - meanSpeed2 / (kSpeedOfLight * kSpeedOfLight), 1.0e-300);
        double const meanGamma = std::sqrt(meanGamma2);
        double const meanSpeed = std::sqrt(meanSpeed2);
        std::array<double, 3> const nhat{
            (meanSpeed > 0.0) ? tail.meanVelocity[0] / meanSpeed : 0.0,
            (meanSpeed > 0.0) ? tail.meanVelocity[1] / meanSpeed : 0.0,
            (meanSpeed > 0.0) ? tail.meanVelocity[2] / meanSpeed : 0.0};
        for(std::size_t p = 0; p < bunch.size(); ++p)
        {
            double const q = bunch.weight[p] * bunch.charge[p];
            std::array<double, 3> const rel{
                bunch.x[p] - centre[0],
                bunch.y[p] - centre[1],
                bunch.z[p] - centre[2]};
            double const rn = rel[0] * nhat[0] + rel[1] * nhat[1] + rel[2] * nhat[2];
            std::array<double, 3> rest{};
            for(int a = 0; a < 3; ++a)
                rest[a] = rel[a] + (meanGamma - 1.0) * rn * nhat[a];
            double const r2 = rest[0] * rest[0] + rest[1] * rest[1] + rest[2] * rest[2];
            for(int a = 0; a < 3; ++a)
                tail.dipole[a] += q * rest[a];
            for(int i = 0; i < 3; ++i)
                for(int j = 0; j < 3; ++j)
                    tail.quadrupole[i * 3 + j]
                        += q * (3.0 * rest[i] * rest[j] - (i == j ? r2 : 0.0));
        }
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

        // Motion of the compact bunch: beta = meanVelocity / c, with
        // 1/gamma^2 = 1 - |beta|^2.  bhat is the unit direction of motion.
        double const c = kSpeedOfLight;
        std::array<double, 3> const beta{
            tail.meanVelocity[0] / c,
            tail.meanVelocity[1] / c,
            tail.meanVelocity[2] / c};
        double const betaSq = beta[0] * beta[0] + beta[1] * beta[1] + beta[2] * beta[2];
        // A valid bunch has a subluminal mean velocity; clamp the pathological
        // superluminal mixed-sign case so the field stays finite.
        double const invG2 = std::max(1.0 - betaSq, 1.0e-15);
        double const gamma = 1.0 / std::sqrt(invG2);
        double const bnorm = std::sqrt(betaSq);
        std::array<double, 3> const bhat{
            (bnorm > 0.0) ? beta[0] / bnorm : 0.0,
            (bnorm > 0.0) ? beta[1] / bnorm : 0.0,
            (bnorm > 0.0) ? beta[2] / bnorm : 0.0};

        // Rest-frame observation vector: longitudinal separation is stretched
        // by gamma.  The dipole and quadrupole in FarTail are rest-frame
        // moments, so their static field must be evaluated at R', not at R.
        double const rn = r[0] * bhat[0] + r[1] * bhat[1] + r[2] * bhat[2];
        std::array<double, 3> const rRest{
            r[0] + (gamma - 1.0) * rn * bhat[0],
            r[1] + (gamma - 1.0) * rn * bhat[1],
            r[2] + (gamma - 1.0) * rn * bhat[2]};
        double const d2r = rRest[0] * rRest[0] + rRest[1] * rRest[1]
            + rRest[2] * rRest[2];
        double const dr = std::sqrt(d2r);
        std::array<double, 3> const nr{
            (dr > 0.0) ? rRest[0] / dr : 0.0,
            (dr > 0.0) ? rRest[1] / dr : 0.0,
            (dr > 0.0) ? rRest[2] / dr : 0.0};

        double const pDotN
            = tail.dipole[0] * nr[0] + tail.dipole[1] * nr[1] + tail.dipole[2] * nr[2];
        std::array<double, 3> dTimesN{};
        for(int i = 0; i < 3; ++i)
            dTimesN[i] = tail.quadrupole[i * 3 + 0] * nr[0]
                + tail.quadrupole[i * 3 + 1] * nr[1]
                + tail.quadrupole[i * 3 + 2] * nr[2];
        double const nDotDn
            = nr[0] * dTimesN[0] + nr[1] * dTimesN[1] + nr[2] * dTimesN[2];

        // Rest-frame static dipole + quadrupole field.  The monopole is the
        // exact Lienard point field below and is not part of this expansion.
        double const dipoleScale = kCoulombConstant * 2.0 / (d2r * dr);
        double const quadrupoleScale = kCoulombConstant * 0.5 / (d2r * d2r);
        std::array<double, 3> high{};
        for(int a = 0; a < 3; ++a)
        {
            double const dipoleTerm = 1.5 * pDotN * nr[a] - 0.5 * tail.dipole[a];
            double const quadrupoleTerm = 5.0 * nDotDn * nr[a] - 2.0 * dTimesN[a];
            high[a] = dipoleScale * dipoleTerm + quadrupoleScale * quadrupoleTerm;
        }

        // Boost the rest-frame higher multipoles to the lab: parallel component
        // unchanged, perpendicular component times gamma.
        if(betaSq > 0.0)
        {
            double const ePar
                = high[0] * bhat[0] + high[1] * bhat[1] + high[2] * bhat[2];
            for(int a = 0; a < 3; ++a)
                high[a] = high[a] + (gamma - 1.0) * (high[a] - ePar * bhat[a]);
        }

        // Monopole: exact Lienard point field of the total charge in the lab.
        double const rb = r[0] * beta[0] + r[1] * beta[1] + r[2] * beta[2];
        double const denom = d2 * invG2 + rb * rb;
        double const monopoleF
            = (denom > 0.0)
            ? kCoulombConstant * tail.charge * invG2 / (denom * std::sqrt(denom))
            : 0.0;

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
