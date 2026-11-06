// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Tolerance-driven velocity-space binning for multi-velocity bunches.  See
// ``picBunch/VelocityBinning.hpp`` for the model and the policy.

#include "picBunch/VelocityBinning.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace picBunch
{
    namespace
    {
        using Vec3 = std::array<double, 3>;

        double dot(Vec3 const& a, Vec3 const& b)
        {
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        }

        Vec3 cross(Vec3 const& a, Vec3 const& b)
        {
            return {a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0]};
        }

        double norm(Vec3 const& a)
        {
            return std::sqrt(dot(a, a));
        }

        Vec3 normalize(Vec3 const& a)
        {
            double const n = norm(a);
            return {a[0] / n, a[1] / n, a[2] / n};
        }

        //! Beam-frame coordinates of every particle plus the per-axis extent.
        struct BeamFrame
        {
            std::array<Vec3, 3> basis; //!< rows e1, e2, e3 (lab frame)
            std::vector<Vec3> uBeam;   //!< per particle, beam frame
            std::array<double, 3> uMin{0.0, 0.0, 0.0};
            std::array<double, 3> uMax{0.0, 0.0, 0.0};
            std::array<double, 3> range{0.0, 0.0, 0.0};
            bool active[3] = {false, false, false};
        };

        BeamFrame makeBeamFrame(
            std::size_t count,
            std::vector<Vec3> const& uLab,
            std::vector<double> const& absCharge,
            double scaleU)
        {
            BeamFrame frame;
            // Charge-weighted mean of the unit momentum directions sets the
            // beam axis; energy spread must not tilt it.
            Vec3 meanDir{0.0, 0.0, 0.0};
            double dirWeight = 0.0;
            for(std::size_t p = 0; p < count; ++p)
            {
                double const um = norm(uLab[p]);
                if(um > 0.0)
                {
                    meanDir[0] += absCharge[p] * uLab[p][0] / um;
                    meanDir[1] += absCharge[p] * uLab[p][1] / um;
                    meanDir[2] += absCharge[p] * uLab[p][2] / um;
                    dirWeight += absCharge[p];
                }
            }
            Vec3 const e3 = (dirWeight > 0.0 && norm(meanDir) > 1.0e-12) ? normalize(meanDir)
                                                                         : Vec3{0.0, 0.0, 1.0};
            // Reference axis least aligned with e3 keeps the cross product
            // well conditioned.
            Vec3 const reference = (std::abs(e3[2]) < 0.9) ? Vec3{0.0, 0.0, 1.0}
                                                           : Vec3{1.0, 0.0, 0.0};
            Vec3 const e1 = normalize(cross(reference, e3));
            Vec3 const e2 = cross(e3, e1);
            frame.basis = {e1, e2, e3};

            frame.uBeam.resize(count);
            for(int i = 0; i < 3; ++i)
            {
                frame.uMin[i] = std::numeric_limits<double>::infinity();
                frame.uMax[i] = -std::numeric_limits<double>::infinity();
            }
            for(std::size_t p = 0; p < count; ++p)
            {
                Vec3 const& u = uLab[p];
                frame.uBeam[p] = {dot(u, e1), dot(u, e2), dot(u, e3)};
                for(int i = 0; i < 3; ++i)
                {
                    frame.uMin[i] = std::min(frame.uMin[i], frame.uBeam[p][i]);
                    frame.uMax[i] = std::max(frame.uMax[i], frame.uBeam[p][i]);
                }
            }
            for(int i = 0; i < 3; ++i)
            {
                frame.range[i] = frame.uMax[i] - frame.uMin[i];
                // A numerically zero extent is degenerate: one box covers it.
                if(frame.range[i] <= 1.0e-12 * scaleU)
                {
                    frame.range[i] = 0.0;
                    frame.uMax[i] = frame.uMin[i];
                }
                frame.active[i] = frame.range[i] > 0.0;
            }
            return frame;
        }

        //! Axis split of a total box budget ``B`` over the active axes.
        //!
        //! Minimizes the geometric part of the model, ``sqrt((r1/N1)^2 +
        //! (r2/N2)^2 + (r3/(N3*gamma))^2)`` (perpendicular box width plus beam-
        //! axis width in ``Delta_gamma/gamma`` units), over splits
        //! ``N1*N2*N3 = B``.  With two active axes the minimum is closed-form
        //! (``N_i`` proportional to ``sqrt(range_i)``); with three, a grid over
        //! the power-of-two factors of ``B`` (which is always a power of two).
        std::array<std::size_t, 3> chooseSplit(
            std::size_t B,
            std::array<double, 3> const& range,
            bool const* active,
            double gammaWeighted)
        {
            std::array<std::size_t, 3> N{1u, 1u, 1u};
            int a1 = -1, a2 = -1, a3 = -1;
            for(int i = 0; i < 3; ++i)
            {
                if(!active[i])
                    continue;
                if(a1 < 0)
                    a1 = i;
                else if(a2 < 0)
                    a2 = i;
                else
                    a3 = i;
            }
            if(a1 < 0)
                return N; // no spread at all
            if(a3 < 0)
            {
                if(a2 < 0)
                {
                    N[static_cast<std::size_t>(a1)] = B;
                    return N;
                }
                double const r1 = range[static_cast<std::size_t>(a1)];
                double const r2 = range[static_cast<std::size_t>(a2)];
                std::size_t n1 = static_cast<std::size_t>(std::llround(
                    std::sqrt(static_cast<double>(B) * r1 / r2)));
                n1 = std::max<std::size_t>(1u, std::min(n1, B));
                N[static_cast<std::size_t>(a1)] = n1;
                N[static_cast<std::size_t>(a2)] = std::max<std::size_t>(1u, B / n1);
                return N;
            }
            auto geometricScore = [&](std::array<std::size_t, 3> const& cand) {
                double s = 0.0;
                for(int i = 0; i < 3; ++i)
                    if(active[i])
                    {
                        double const width = range[i] / static_cast<double>(cand[i]);
                        s += (i < 2) ? width * width
                                     : (width / gammaWeighted) * (width / gammaWeighted);
                    }
                return std::sqrt(s);
            };
            std::array<std::size_t, 3> best{1u, 1u, 1u};
            double bestScore = std::numeric_limits<double>::infinity();
            for(std::size_t n1 = 1; n1 <= B; n1 *= 2u)
            {
                for(std::size_t n2 = 1; n2 <= B / n1; n2 *= 2u)
                {
                    if((B / n1) % n2 != 0u)
                        continue;
                    std::array<std::size_t, 3> const cand{n1, n2, B / (n1 * n2)};
                    double const score = geometricScore(cand);
                    if(score < bestScore)
                    {
                        bestScore = score;
                        best = cand;
                    }
                }
            }
            return best;
        }

        //! Assign every particle to its box for a split and evaluate the
        //! calibrated content model.  ``out`` receives the geometry, the
        //! representatives and the per-particle assignment.
        double evaluateSplit(
            Bunch const& bunch,
            BeamFrame const& frame,
            std::array<std::size_t, 3> const& N,
            std::vector<double> const& absCharge,
            double totalAbsoluteCharge,
            double modelConstant,
            VelocityBinning& out)
        {
            std::size_t const count = bunch.size();
            std::size_t const boxes = N[0] * N[1] * N[2];

            out.perAxis = N;
            out.count = boxes;
            out.basis = frame.basis;
            out.uMin = frame.uMin;
            out.uMax = frame.uMax;
            out.representativeU.resize(boxes);

            // Representative: box centre in the beam frame, mapped to the lab
            // frame.  One box: the charge-weighted mean u.
            if(boxes == 1u)
            {
                Vec3 sum{0.0, 0.0, 0.0};
                double w = 0.0;
                for(std::size_t p = 0; p < count; ++p)
                {
                    sum[0] += absCharge[p] * bunch.ux[p];
                    sum[1] += absCharge[p] * bunch.uy[p];
                    sum[2] += absCharge[p] * bunch.uz[p];
                    w += absCharge[p];
                }
                out.representativeU[0] = {sum[0] / w, sum[1] / w, sum[2] / w};
            }
            else
            {
                for(std::size_t i3 = 0; i3 < N[2]; ++i3)
                    for(std::size_t i2 = 0; i2 < N[1]; ++i2)
                        for(std::size_t i1 = 0; i1 < N[0]; ++i1)
                        {
                            std::size_t const box = i1 + N[0] * (i2 + N[1] * i3);
                            double const centre[3] = {
                                frame.uMin[0] + (static_cast<double>(i1) + 0.5) * frame.range[0]
                                    / static_cast<double>(N[0]),
                                frame.uMin[1] + (static_cast<double>(i2) + 0.5) * frame.range[1]
                                    / static_cast<double>(N[1]),
                                frame.uMin[2] + (static_cast<double>(i3) + 0.5) * frame.range[2]
                                    / static_cast<double>(N[2])};
                            for(int a = 0; a < 3; ++a)
                                out.representativeU[box][a]
                                    = centre[0] * frame.basis[0][a] + centre[1] * frame.basis[1][a]
                                        + centre[2] * frame.basis[2][a];
                        }
            }

            std::vector<std::size_t> boxOf(count, 0u);
            if(boxes > 1u)
            {
                double const width[3] = {
                    (frame.range[0] > 0.0) ? frame.range[0] / static_cast<double>(N[0]) : 1.0,
                    (frame.range[1] > 0.0) ? frame.range[1] / static_cast<double>(N[1]) : 1.0,
                    (frame.range[2] > 0.0) ? frame.range[2] / static_cast<double>(N[2]) : 1.0};
                for(std::size_t p = 0; p < count; ++p)
                {
                    std::size_t idx[3] = {};
                    for(int a = 0; a < 3; ++a)
                    {
                        if(frame.range[a] <= 0.0)
                            continue;
                        long t = static_cast<long>(std::floor((frame.uBeam[p][a] - frame.uMin[a]) / width[a]));
                        idx[a] = (t < 0) ? 0u
                                         : (static_cast<std::size_t>(t) >= N[a] ? N[a] - 1u
                                                                                : static_cast<std::size_t>(t));
                    }
                    boxOf[p] = idx[0] + N[0] * (idx[1] + N[1] * idx[2]);
                }
            }
            out.boxOf = boxOf;

            // Content model: per box, absolute-charge fraction, gamma range, the
            // largest member angle against the representative, and the largest
            // member-vs-representative gamma deviation.  The boxes are weighted
            // by their absolute-charge content (never the signed sum) so that
            // opposite charges cannot erase a local error.
            std::vector<double> boxAbsCharge(boxes, 0.0);
            std::vector<double> boxGammaMin(boxes, 0.0), boxGammaMax(boxes, 0.0);
            std::vector<bool> boxUsed(boxes, false);
            std::vector<double> particleGamma(count);
            for(std::size_t p = 0; p < count; ++p)
            {
                double const u2 = bunch.ux[p] * bunch.ux[p] + bunch.uy[p] * bunch.uy[p]
                    + bunch.uz[p] * bunch.uz[p];
                particleGamma[p] = std::sqrt(1.0 + u2);
                std::size_t const b = boxOf[p];
                boxAbsCharge[b] += absCharge[p];
                if(boxUsed[b])
                {
                    boxGammaMin[b] = std::min(boxGammaMin[b], particleGamma[p]);
                    boxGammaMax[b] = std::max(boxGammaMax[b], particleGamma[p]);
                }
                else
                {
                    boxGammaMin[b] = particleGamma[p];
                    boxGammaMax[b] = particleGamma[p];
                    boxUsed[b] = true;
                }
            }
            double modelSum = 0.0;
            for(std::size_t b = 0; b < boxes; ++b)
            {
                if(!boxUsed[b])
                    continue;
                Vec3 const& uRep = out.representativeU[b];
                double const repNorm = norm(uRep);
                double const gammaRep = std::sqrt(1.0 + repNorm * repNorm);
                double maxAngle = 0.0;
                double maxGammaDev = 0.0;
                bool anyMoving = false;
                for(std::size_t p = 0; p < count; ++p)
                {
                    if(boxOf[p] != b)
                        continue;
                    double const u2 = bunch.ux[p] * bunch.ux[p] + bunch.uy[p] * bunch.uy[p]
                        + bunch.uz[p] * bunch.uz[p];
                    double const uMag = std::sqrt(u2);
                    if(uMag > 0.0)
                    {
                        anyMoving = true;
                        // The angle against the representative is only defined
                        // when the representative is itself moving.
                        if(repNorm > 0.0)
                        {
                            double const cosA
                                = std::max(
                                      -1.0,
                                      std::min(
                                          1.0, (bunch.ux[p] * uRep[0] + bunch.uy[p] * uRep[1]
                                                 + bunch.uz[p] * uRep[2])
                                                / (uMag * repNorm)));
                            maxAngle = std::max(maxAngle, std::acos(cosA));
                        }
                    }
                    maxGammaDev = std::max(
                        maxGammaDev, std::abs(particleGamma[p] - gammaRep) / gammaRep);
                }
                // A stationary representative for a moving member is a full
                // angular mismatch: the representative kernel is a static one.
                if(repNorm == 0.0 && anyMoving)
                    maxAngle = std::acos(-1.0);
                double const gamma = boxGammaMax[b];
                modelSum += (boxAbsCharge[b] / totalAbsoluteCharge)
                    * (gamma * maxAngle + (boxGammaMax[b] - boxGammaMin[b]) / gamma
                       + maxGammaDev);
            }
            out.modelSum = modelSum;
            out.modelConstant = modelConstant;
            out.modelError = modelConstant * modelSum;
            return modelSum;
        }
    } // namespace

    std::size_t VelocityBinning::boxIndex(double u1, double u2, double u3) const
    {
        double const coords[3] = {u1, u2, u3};
        std::size_t idx[3] = {};
        for(int a = 0; a < 3; ++a)
        {
            if(uMax[a] - uMin[a] <= 0.0)
            {
                idx[a] = 0u;
                continue;
            }
            double const width = (uMax[a] - uMin[a]) / static_cast<double>(perAxis[a]);
            long t = static_cast<long>(std::floor((coords[a] - uMin[a]) / width));
            idx[a] = (t < 0) ? 0u
                             : (static_cast<std::size_t>(t) >= perAxis[a] ? perAxis[a] - 1u
                                                                          : static_cast<std::size_t>(t));
        }
        return idx[0] + perAxis[0] * (idx[1] + perAxis[1] * idx[2]);
    }

    VelocityBinning deriveVelocityBinning(
        Bunch const& bunch,
        double tolerance,
        VelocityBinningOptions const& policy)
    {
        std::size_t const count = bunch.size();
        if(count == 0u)
            throw std::runtime_error("deriveVelocityBinning: empty bunch");
        if(!(tolerance > 0.0) || !std::isfinite(tolerance))
            throw std::invalid_argument("deriveVelocityBinning: tolerance must be finite and positive");
        if(!(policy.modelConstant > 0.0) || !std::isfinite(policy.modelConstant))
            throw std::invalid_argument("deriveVelocityBinning: modelConstant must be finite and positive");
        if(policy.maxBins < 1u)
            throw std::invalid_argument("deriveVelocityBinning: maxBins must be >= 1");
        if(policy.mode == VelocityBinningOptions::Mode::Explicit && policy.explicitBins < 1u)
            throw std::invalid_argument("deriveVelocityBinning: explicitBins must be >= 1");

        std::vector<Vec3> uLab(count);
        std::vector<double> absCharge(count);
        double totalAbsoluteCharge = 0.0;
        double scaleU = 1.0;
        for(std::size_t p = 0; p < count; ++p)
        {
            uLab[p] = {bunch.ux[p], bunch.uy[p], bunch.uz[p]};
            for(double const v : uLab[p])
            {
                if(!std::isfinite(v))
                    throw std::invalid_argument("deriveVelocityBinning: a momentum is not finite");
                scaleU = std::max(scaleU, std::abs(v));
            }
            double const q = bunch.weight[p] * bunch.charge[p];
            if(!std::isfinite(q))
                throw std::invalid_argument("deriveVelocityBinning: a particle charge is not finite");
            absCharge[p] = std::abs(q);
            totalAbsoluteCharge += std::abs(q);
        }
        if(totalAbsoluteCharge == 0.0)
            throw std::runtime_error("deriveVelocityBinning: the bunch carries no charge");

        // Historical single-velocity check, unchanged: within 1e-12 relative the
        // bunch is one velocity class and the kernel is exact.
        bool sharedVelocity = true;
        for(std::size_t p = 1; p < count && sharedVelocity; ++p)
            if(std::abs(bunch.ux[p] - bunch.ux[0]) > 1.0e-12 * scaleU
               || std::abs(bunch.uy[p] - bunch.uy[0]) > 1.0e-12 * scaleU
               || std::abs(bunch.uz[p] - bunch.uz[0]) > 1.0e-12 * scaleU)
                sharedVelocity = false;

        VelocityBinning out;
        out.derived = true;
        out.modelConstant = policy.modelConstant;

        if(sharedVelocity)
        {
            out.multiVelocity = false;
            out.count = 1u;
            out.perAxis = {1u, 1u, 1u};
            out.representativeU = {uLab[0]};
            out.boxOf.assign(count, 0u);
            out.modelSum = 0.0;
            out.modelError = 0.0;
            return out;
        }
        out.multiVelocity = true;

        BeamFrame const frame = makeBeamFrame(count, uLab, absCharge, scaleU);
        double gammaWeighted = 0.0;
        {
            double w = 0.0;
            for(std::size_t p = 0; p < count; ++p)
            {
                gammaWeighted += absCharge[p] * std::sqrt(1.0 + dot(uLab[p], uLab[p]));
                w += absCharge[p];
            }
            gammaWeighted /= w;
        }

        auto splitsFor = [&](std::size_t B) {
            return chooseSplit(B, frame.range, frame.active, gammaWeighted);
        };
        auto modelSumFor = [&](std::size_t B, VelocityBinning& probe) {
            return evaluateSplit(
                bunch, frame, splitsFor(B), absCharge, totalAbsoluteCharge,
                policy.modelConstant, probe);
        };

        std::size_t targetBins = 0u;
        if(policy.mode == VelocityBinningOptions::Mode::Explicit)
        {
            // Diagnostics run whatever is requested; the model is reported,
            // not enforced.
            targetBins = policy.explicitBins;
        }
        else
        {
            // Smallest power of two whose best split meets the model.
            for(std::size_t B = 1u; B <= policy.maxBins; B *= 2u)
            {
                VelocityBinning probe;
                double const modelSum = modelSumFor(B, probe);
                if(policy.modelConstant * modelSum <= tolerance)
                {
                    targetBins = B;
                    break;
                }
            }
            if(targetBins == 0u)
            {
                // Report the count the model would need, beyond the cap.
                std::size_t required = policy.maxBins;
                double lastError = 0.0;
                for(std::size_t B = policy.maxBins * 2u; B <= (1u << 30); B *= 2u)
                {
                    VelocityBinning probe;
                    double const modelSum = modelSumFor(B, probe);
                    lastError = policy.modelConstant * modelSum;
                    required = B;
                    if(lastError <= tolerance)
                        break;
                }
                throw std::runtime_error(
                    "deriveVelocityBinning: the velocity-bin model requires at least "
                    + std::to_string(required)
                    + " boxes for tolerance " + std::to_string(tolerance)
                    + " (estimated error at the " + std::to_string(policy.maxBins)
                    + "-box cap: " + std::to_string(lastError)
                    + "); reduce the momentum spread, raise --tolerance, or raise --max-velocity-bins");
            }
        }

        evaluateSplit(
            bunch, frame, splitsFor(targetBins), absCharge, totalAbsoluteCharge,
            policy.modelConstant, out);
        return out;
    }

} // namespace picBunch
