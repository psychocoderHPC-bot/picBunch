// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Multipole far field used to fill a global output domain outside the exact
// compute box.  This replaces the former exact-zero fill; the expansion is
// valid beyond ``FarTail::switchDistance`` and is measured against an
// independent direct sum in the validation suite.

#include "picBunch/FarTail.hpp"

#include "picBunch/detail/HostReduction.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace picBunch
{
    namespace
    {
        //! Dedicated CPU-side alpaka queue for the particle bookkeeping kernels.
        //!
        //! The Bunch lives in host ``std::vector`` storage.  Running the kernels on
        //! the host API keeps the data where it is, independent of whether the
        //! field solver was built for CUDA/HIP.
        inline auto& hostReductionQueue()
        {
            static auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::host, alpaka::deviceKind::cpu);
            static auto device = selector.makeDevice(0);
            static auto queue = device.makeQueue();
            return queue;
        }

        template<typename TKernel, typename... TArgs>
        void runHostKernel(std::size_t count, TKernel const& kernel, TArgs... args)
        {
            if(count == 0u)
                return;
            std::uint32_t const frame = 64u;
            auto const blocks = static_cast<std::uint32_t>((count + frame - 1u) / frame);
            auto& queue = hostReductionQueue();
            switch(detail::hostReductionExecutorOverride())
            {
            case detail::HostReductionExecutor::Serial:
                queue.enqueue(
                    alpaka::onHost::FrameSpec{blocks, frame, alpaka::exec::cpuSerial}, kernel, args...);
                break;
            case detail::HostReductionExecutor::Parallel:
            case detail::HostReductionExecutor::Default:
            default:
                queue.enqueue(alpaka::onHost::FrameSpec{blocks, frame}, kernel, args...);
                break;
            }
            alpaka::onHost::wait(queue);
        }

        //! First far-tail pass: support distance and gamma.
        //!
        //! Only the transcendental per-particle work is offloaded.  The charge,
        //! the absolute charge and the velocity are recomputed in the ordered
        //! host accumulation with the same expressions as the reference loop, so
        //! the sums stay bit-identical and need less scratch memory.
        struct FarTailPassKernel
        {
            template<typename TAcc>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                std::size_t count,
                alpaka::concepts::IMdSpan auto const& x,
                alpaka::concepts::IMdSpan auto const& y,
                alpaka::concepts::IMdSpan auto const& z,
                alpaka::concepts::IMdSpan auto const& ux,
                alpaka::concepts::IMdSpan auto const& uy,
                alpaka::concepts::IMdSpan auto const& uz,
                alpaka::Vec<double, 3u> const centre,
                alpaka::concepts::IMdSpan auto supportDistance,
                alpaka::concepts::IMdSpan auto gamma) const
            {
                for(auto [p] : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{count}))
                {
                    auto const rel = alpaka::Vec{x[p], y[p], z[p]} - centre;
                    double const r2 = rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2];
                    supportDistance[p] = alpaka::math::sqrt(r2);
                    double const u2 = ux[p] * ux[p] + uy[p] * uy[p] + uz[p] * uz[p];
                    gamma[p] = alpaka::math::sqrt(1.0 + u2);
                }
            }
        };

        //! Second far-tail pass: rest-frame separation vector.
        struct RestFrameKernel
        {
            template<typename TAcc>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                std::size_t count,
                alpaka::concepts::IMdSpan auto const& x,
                alpaka::concepts::IMdSpan auto const& y,
                alpaka::concepts::IMdSpan auto const& z,
                alpaka::Vec<double, 3u> const centre,
                alpaka::Vec<double, 3u> const nhat,
                double meanGamma,
                alpaka::concepts::IMdSpan auto rest) const
            {
                for(auto [p] : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{count}))
                {
                    auto const rel = alpaka::Vec{x[p], y[p], z[p]} - centre;
                    double const rn = rel[0] * nhat[0] + rel[1] * nhat[1] + rel[2] * nhat[2];
                    for(int a = 0; a < 3; ++a)
                        rest[3u * p + static_cast<std::size_t>(a)]
                            = rel[a] + (meanGamma - 1.0) * rn * nhat[a];
                }
            }
        };
    } // namespace

    FarTail deriveFarTail(
        Bunch const& bunch,
        double switchFactor,
        Vec3 const& cellSize,
        int groupCells)
    {
        if(bunch.size() == 0u)
            throw std::runtime_error("deriveFarTail: empty bunch");
        if(!(switchFactor > 0.0) || !std::isfinite(switchFactor))
            throw std::invalid_argument("deriveFarTail: switchFactor must be finite and positive");

        auto const bounds = bunch.bounds();
        Vec3 centre{};
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
        Vec3 velocitySumCharge{0.0, 0.0, 0.0};
        Vec3 velocitySumAbs{0.0, 0.0, 0.0};
        std::size_t const count = bunch.size();
        // Reused scratch; the function is not called concurrently on one thread.
        thread_local std::vector<double> supportDistance;
        thread_local std::vector<double> gammaOf;
        supportDistance.resize(count);
        gammaOf.resize(count);
        auto const particleExtent = alpaka::Vec<std::size_t, 1u>{count};
        auto const xView = alpaka::makeMdSpan(std::as_const(bunch.x).data(), particleExtent);
        static_assert(std::is_same_v<
                      typename std::remove_cvref_t<
                          decltype(std::declval<decltype(xView) const&>().getExtents())>::value_type,
                      std::size_t>);
        auto const yView = alpaka::makeMdSpan(std::as_const(bunch.y).data(), particleExtent);
        auto const zView = alpaka::makeMdSpan(std::as_const(bunch.z).data(), particleExtent);
        auto const uxView = alpaka::makeMdSpan(std::as_const(bunch.ux).data(), particleExtent);
        auto const uyView = alpaka::makeMdSpan(std::as_const(bunch.uy).data(), particleExtent);
        auto const uzView = alpaka::makeMdSpan(std::as_const(bunch.uz).data(), particleExtent);
        auto const centreVec = alpaka::Vec{centre[0], centre[1], centre[2]};
        auto const supportDistanceView = alpaka::makeMdSpan(supportDistance.data(), particleExtent);
        auto const gammaView = alpaka::makeMdSpan(gammaOf.data(), particleExtent);
        runHostKernel(
            count,
            FarTailPassKernel{},
            count,
            xView,
            yView,
            zView,
            uxView,
            uyView,
            uzView,
            centreVec,
            supportDistanceView,
            gammaView);
        // The running sums keep the historical particle order.  Reordering a
        // floating-point sum would make the mean velocity depend on the thread
        // count, so the reduction itself stays serial.
        for(std::size_t p = 0; p < count; ++p)
        {
            double const q = bunch.weight[p] * bunch.charge[p];
            tail.charge += q;
            double const absQ = std::abs(q);
            absoluteCharge += absQ;
            tail.supportRadius = std::max(tail.supportRadius, supportDistance[p]);
            tail.gammaMax = std::max(tail.gammaMax, gammaOf[p]);
            double const invGamma = 1.0 / gammaOf[p];
            for(int a = 0; a < 3; ++a)
            {
                double const u = (a == 0) ? bunch.ux[p] : (a == 1) ? bunch.uy[p] : bunch.uz[p];
                double const v = kSpeedOfLight * u * invGamma; // particle velocity
                velocitySumCharge[a] += q * v;
                velocitySumAbs[a] += absQ * v;
            }
        }
        // The weighting choice depends on the TOTAL charge only.  A bunch whose
        // net charge is negligible against its absolute content is neutral and
        // uses the absolute-charge weights, so the mean velocity direction stays
        // defined.
        bool const neutral
            = (absoluteCharge == 0.0)
            || (std::abs(tail.charge) <= 1.0e-12 * absoluteCharge);
        Vec3 const& velocitySum = neutral ? velocitySumAbs : velocitySumCharge;
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
        Vec3 const nhat{
            (meanSpeed > 0.0) ? tail.meanVelocity[0] / meanSpeed : 0.0,
            (meanSpeed > 0.0) ? tail.meanVelocity[1] / meanSpeed : 0.0,
            (meanSpeed > 0.0) ? tail.meanVelocity[2] / meanSpeed : 0.0};
        thread_local std::vector<double> restFrame;
        restFrame.resize(3u * count);
        auto const nhatVec = alpaka::Vec{nhat[0], nhat[1], nhat[2]};
        auto const restExtent = alpaka::Vec<std::size_t, 1u>{3u * count};
        auto const restView = alpaka::makeMdSpan(restFrame.data(), restExtent);
        static_assert(std::is_same_v<
                      typename std::remove_cvref_t<
                          decltype(std::declval<decltype(restView) const&>().getExtents())>::value_type,
                      std::size_t>);
        runHostKernel(
            count,
            RestFrameKernel{},
            count,
            xView,
            yView,
            zView,
            centreVec,
            nhatVec,
            meanGamma,
            restView);
        // Ordered accumulation: the multipoles are sums and their rounding must
        // not follow the executor's thread count.
        for(std::size_t p = 0; p < count; ++p)
        {
            double const q = bunch.weight[p] * bunch.charge[p];
            double const rest[3]
                = {restFrame[3u * p], restFrame[3u * p + 1u], restFrame[3u * p + 2u]};
            double const r2 = rest[0] * rest[0] + rest[1] * rest[1] + rest[2] * rest[2];
            for(int a = 0; a < 3; ++a)
                tail.dipole[a] += q * rest[a];
            for(int i = 0; i < 3; ++i)
                for(int j = 0; j < 3; ++j)
                    tail.quadrupole[i * 3 + j]
                        += q * (3.0 * rest[i] * rest[j] - (i == j ? r2 : 0.0));
        }
        tail.switchDistance = switchFactor * tail.supportRadius;

        if(groupCells > 0 && cellSize[0] > 0.0 && cellSize[1] > 0.0 && cellSize[2] > 0.0)
        {
            Vec3 extent{};
            for(int a = 0; a < 3; ++a)
                extent[a] = static_cast<double>(groupCells) * cellSize[a];
            std::array<long, 3> dims{};
            for(int a = 0; a < 3; ++a)
            {
                dims[a] = static_cast<long>(
                    std::floor((bounds[1][a] - bounds[0][a]) / extent[a])) + 1;
                dims[a] = std::max(1L, dims[a]);
            }
            long long const cellCount = static_cast<long long>(dims[0])
                * static_cast<long long>(dims[1]) * static_cast<long long>(dims[2]);
            if(cellCount > 0 && cellCount <= 1'000'000ll)
            {
                std::vector<std::vector<std::size_t>> members(
                    static_cast<std::size_t>(cellCount));
                for(std::size_t p = 0; p < bunch.size(); ++p)
                {
                    long idx[3];
                    for(int a = 0; a < 3; ++a)
                    {
                        double const value = (a == 0) ? bunch.x[p] : (a == 1) ? bunch.y[p] : bunch.z[p];
                        long t = static_cast<long>(
                            std::floor((value - bounds[0][a]) / extent[a]));
                        idx[a] = std::max(0L, std::min(t, dims[a] - 1));
                    }
                    std::size_t const flat = static_cast<std::size_t>(
                        idx[0] + dims[0] * (idx[1] + dims[1] * idx[2]));
                    members[flat].push_back(p);
                }
                for(auto const& groupMembers : members)
                {
                    if(groupMembers.empty())
                        continue;
                    Bunch sub;
                    sub.resize(groupMembers.size());
                    for(std::size_t k = 0; k < groupMembers.size(); ++k)
                    {
                        std::size_t const p = groupMembers[k];
                        sub.x[k] = bunch.x[p];
                        sub.y[k] = bunch.y[p];
                        sub.z[k] = bunch.z[p];
                        sub.ux[k] = bunch.ux[p];
                        sub.uy[k] = bunch.uy[p];
                        sub.uz[k] = bunch.uz[p];
                        sub.weight[k] = bunch.weight[p];
                        sub.charge[k] = bunch.charge[p];
                        sub.restMass[k] = bunch.restMass.empty() ? 0.0 : bunch.restMass[p];
                    }
                    tail.groups.push_back(
                        deriveFarTail(sub, switchFactor, Vec3{0.0, 0.0, 0.0}, 0));
                }
            }
        }
        return tail;
    }

    Vec3 evaluateFarTailE(
        FarTail const& tail,
        Vec3 const& position)
    {
        Vec3 const r{
            position[0] - tail.centre[0],
            position[1] - tail.centre[1],
            position[2] - tail.centre[2]};
        double const d2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        if(d2 == 0.0)
            return {0.0, 0.0, 0.0};
        if(!tail.groups.empty())
        {
            Vec3 sum{0.0, 0.0, 0.0};
            for(auto const& group : tail.groups)
            {
                auto const value = evaluateFarTailE(group, position);
                for(int a = 0; a < 3; ++a)
                    sum[a] += value[a];
            }
            return sum;
        }
        double const d = std::sqrt(d2);
        Vec3 const n{r[0] / d, r[1] / d, r[2] / d};

        // Motion of the compact bunch: beta = meanVelocity / c, with
        // 1/gamma^2 = 1 - |beta|^2.  bhat is the unit direction of motion.
        double const c = kSpeedOfLight;
        Vec3 const beta{
            tail.meanVelocity[0] / c,
            tail.meanVelocity[1] / c,
            tail.meanVelocity[2] / c};
        double const betaSq = beta[0] * beta[0] + beta[1] * beta[1] + beta[2] * beta[2];
        // A valid bunch has a subluminal mean velocity; clamp the pathological
        // superluminal mixed-sign case so the field stays finite.
        double const invG2 = std::max(1.0 - betaSq, 1.0e-15);
        double const gamma = 1.0 / std::sqrt(invG2);
        double const bnorm = std::sqrt(betaSq);
        Vec3 const bhat{
            (bnorm > 0.0) ? beta[0] / bnorm : 0.0,
            (bnorm > 0.0) ? beta[1] / bnorm : 0.0,
            (bnorm > 0.0) ? beta[2] / bnorm : 0.0};

        // Rest-frame observation vector: longitudinal separation is stretched
        // by gamma.  The dipole and quadrupole in FarTail are rest-frame
        // moments, so their static field must be evaluated at R', not at R.
        double const rn = r[0] * bhat[0] + r[1] * bhat[1] + r[2] * bhat[2];
        Vec3 const rRest{
            r[0] + (gamma - 1.0) * rn * bhat[0],
            r[1] + (gamma - 1.0) * rn * bhat[1],
            r[2] + (gamma - 1.0) * rn * bhat[2]};
        double const d2r = rRest[0] * rRest[0] + rRest[1] * rRest[1]
            + rRest[2] * rRest[2];
        double const dr = std::sqrt(d2r);
        Vec3 const nr{
            (dr > 0.0) ? rRest[0] / dr : 0.0,
            (dr > 0.0) ? rRest[1] / dr : 0.0,
            (dr > 0.0) ? rRest[2] / dr : 0.0};

        double const pDotN
            = tail.dipole[0] * nr[0] + tail.dipole[1] * nr[1] + tail.dipole[2] * nr[2];
        Vec3 dTimesN{};
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
        Vec3 high{};
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

        Vec3 field{};
        for(int a = 0; a < 3; ++a)
            field[a] = high[a] + monopoleF * r[a];
        return field;
    }

    Vec3 evaluateFarTailB(
        FarTail const& tail,
        Vec3 const& position)
    {
        auto const e = evaluateFarTailE(tail, position);
        Vec3 const& v = tail.meanVelocity;
        double const inverseC2 = 1.0 / (kSpeedOfLight * kSpeedOfLight);
        return {
            (v[1] * e[2] - v[2] * e[1]) * inverseC2,
            (v[2] * e[0] - v[0] * e[2]) * inverseC2,
            (v[0] * e[1] - v[1] * e[0]) * inverseC2};
    }

    FarTailBatch makeFarTailBatch(FarTail const& tail)
    {
        FarTailBatch batch;
        batch.centre = tail.centre;
        batch.meanVelocity = tail.meanVelocity;

        auto append = [&batch](FarTail const& entry)
        {
            batch.centreX.push_back(entry.centre[0]);
            batch.centreY.push_back(entry.centre[1]);
            batch.centreZ.push_back(entry.centre[2]);
            batch.charge.push_back(entry.charge);
            batch.dipoleX.push_back(entry.dipole[0]);
            batch.dipoleY.push_back(entry.dipole[1]);
            batch.dipoleZ.push_back(entry.dipole[2]);
            for(int i = 0; i < 9; ++i)
                batch.quadrupole[static_cast<std::size_t>(i)].push_back(entry.quadrupole[i]);

            // Precompute the boost geometry once per group instead of once per
            // node and group.  The expressions match evaluateFarTailE exactly.
            double const c = kSpeedOfLight;
            Vec3 const beta{
                entry.meanVelocity[0] / c,
                entry.meanVelocity[1] / c,
                entry.meanVelocity[2] / c};
            double const betaSq = beta[0] * beta[0] + beta[1] * beta[1] + beta[2] * beta[2];
            double const invG2 = std::max(1.0 - betaSq, 1.0e-15);
            double const gamma = 1.0 / std::sqrt(invG2);
            double const bnorm = std::sqrt(betaSq);
            batch.betaX.push_back(beta[0]);
            batch.betaY.push_back(beta[1]);
            batch.betaZ.push_back(beta[2]);
            batch.bhatX.push_back((bnorm > 0.0) ? beta[0] / bnorm : 0.0);
            batch.bhatY.push_back((bnorm > 0.0) ? beta[1] / bnorm : 0.0);
            batch.bhatZ.push_back((bnorm > 0.0) ? beta[2] / bnorm : 0.0);
            batch.invGamma2.push_back(invG2);
            batch.gammaMinusOne.push_back(gamma - 1.0);
        };

        if(tail.groups.empty())
        {
            append(tail);
        }
        else
        {
            std::size_t const groups = tail.groups.size();
            batch.centreX.reserve(groups);
            batch.centreY.reserve(groups);
            batch.centreZ.reserve(groups);
            batch.charge.reserve(groups);
            batch.dipoleX.reserve(groups);
            batch.dipoleY.reserve(groups);
            batch.dipoleZ.reserve(groups);
            for(auto& component : batch.quadrupole)
                component.reserve(groups);
            batch.betaX.reserve(groups);
            batch.betaY.reserve(groups);
            batch.betaZ.reserve(groups);
            batch.bhatX.reserve(groups);
            batch.bhatY.reserve(groups);
            batch.bhatZ.reserve(groups);
            batch.invGamma2.reserve(groups);
            batch.gammaMinusOne.reserve(groups);
            for(auto const& group : tail.groups)
            {
                if(!group.groups.empty())
                    throw std::invalid_argument(
                        "makeFarTailBatch: nested far-tail groups are not supported");
                append(group);
            }
        }
        batch.count = batch.centreX.size();
        return batch;
    }

    namespace
    {
        /*! Evaluate ``W`` consecutive flattened groups at one position.
         *
         * The group loop is split into stages that each read contiguous arrays
         * and write one staging array.  Every stage is a plain elementwise loop
         * over ``W`` lanes, so the compiler vectorises it and the independent
         * lanes hide the ``sqrt``/divide latency.  The expressions and their
         * association match :cpp:func:`evaluateFarTailE` term by term, so the
         * results are bit-identical.
         */
        template<std::size_t W>
        void evaluateBatchBlock(
            FarTailBatch const& batch,
            std::size_t base,
            Vec3 const& position,
            std::array<double, W>& ex,
            std::array<double, W>& ey,
            std::array<double, W>& ez)
        {
            std::array<double, W> r0;
            std::array<double, W> r1;
            std::array<double, W> r2;
            for(std::size_t k = 0; k < W; ++k)
            {
                std::size_t const i = base + k;
                r0[k] = position[0] - batch.centreX[i];
                r1[k] = position[1] - batch.centreY[i];
                r2[k] = position[2] - batch.centreZ[i];
            }
            std::array<double, W> d2;
            for(std::size_t k = 0; k < W; ++k)
                d2[k] = r0[k] * r0[k] + r1[k] * r1[k] + r2[k] * r2[k];
            std::array<double, W> active;
            for(std::size_t k = 0; k < W; ++k)
                active[k] = (d2[k] == 0.0) ? 0.0 : 1.0;

            // Rest-frame observation vector.
            std::array<double, W> rr0;
            std::array<double, W> rr1;
            std::array<double, W> rr2;
            for(std::size_t k = 0; k < W; ++k)
            {
                std::size_t const i = base + k;
                double const rn = r0[k] * batch.bhatX[i] + r1[k] * batch.bhatY[i]
                    + r2[k] * batch.bhatZ[i];
                double const g1 = batch.gammaMinusOne[i];
                rr0[k] = r0[k] + g1 * rn * batch.bhatX[i];
                rr1[k] = r1[k] + g1 * rn * batch.bhatY[i];
                rr2[k] = r2[k] + g1 * rn * batch.bhatZ[i];
            }
            std::array<double, W> d2r;
            for(std::size_t k = 0; k < W; ++k)
                d2r[k] = rr0[k] * rr0[k] + rr1[k] * rr1[k] + rr2[k] * rr2[k];
            std::array<double, W> dr;
            std::array<double, W> restActive;
            for(std::size_t k = 0; k < W; ++k)
            {
                restActive[k] = (d2r[k] > 0.0) ? 1.0 : 0.0;
                dr[k] = std::sqrt((d2r[k] > 0.0) ? d2r[k] : 1.0);
            }
            std::array<double, W> nr0;
            std::array<double, W> nr1;
            std::array<double, W> nr2;
            for(std::size_t k = 0; k < W; ++k)
            {
                double const safe = (restActive[k] != 0.0) ? dr[k] : 1.0;
                nr0[k] = (restActive[k] != 0.0) ? rr0[k] / safe : 0.0;
                nr1[k] = (restActive[k] != 0.0) ? rr1[k] / safe : 0.0;
                nr2[k] = (restActive[k] != 0.0) ? rr2[k] / safe : 0.0;
            }

            std::array<double, W> pDotN;
            for(std::size_t k = 0; k < W; ++k)
            {
                std::size_t const i = base + k;
                pDotN[k] = batch.dipoleX[i] * nr0[k] + batch.dipoleY[i] * nr1[k]
                    + batch.dipoleZ[i] * nr2[k];
            }
            std::array<double, W> qn0;
            std::array<double, W> qn1;
            std::array<double, W> qn2;
            for(std::size_t k = 0; k < W; ++k)
            {
                std::size_t const i = base + k;
                qn0[k] = batch.quadrupole[0][i] * nr0[k] + batch.quadrupole[1][i] * nr1[k]
                    + batch.quadrupole[2][i] * nr2[k];
                qn1[k] = batch.quadrupole[3][i] * nr0[k] + batch.quadrupole[4][i] * nr1[k]
                    + batch.quadrupole[5][i] * nr2[k];
                qn2[k] = batch.quadrupole[6][i] * nr0[k] + batch.quadrupole[7][i] * nr1[k]
                    + batch.quadrupole[8][i] * nr2[k];
            }
            std::array<double, W> nDotDn;
            for(std::size_t k = 0; k < W; ++k)
                nDotDn[k] = nr0[k] * qn0[k] + nr1[k] * qn1[k] + nr2[k] * qn2[k];

            std::array<double, W> dipoleScale;
            std::array<double, W> quadrupoleScale;
            for(std::size_t k = 0; k < W; ++k)
            {
                double const drSafe = (restActive[k] != 0.0) ? dr[k] : 1.0;
                dipoleScale[k] = kCoulombConstant * 2.0 / (d2r[k] * drSafe);
                quadrupoleScale[k] = kCoulombConstant * 0.5 / (d2r[k] * d2r[k]);
            }
            std::array<double, W> h0;
            std::array<double, W> h1;
            std::array<double, W> h2;
            for(std::size_t k = 0; k < W; ++k)
            {
                std::size_t const i = base + k;
                double const dipoleTerm0 = (1.5 * pDotN[k]) * nr0[k] - 0.5 * batch.dipoleX[i];
                double const dipoleTerm1 = (1.5 * pDotN[k]) * nr1[k] - 0.5 * batch.dipoleY[i];
                double const dipoleTerm2 = (1.5 * pDotN[k]) * nr2[k] - 0.5 * batch.dipoleZ[i];
                double const quadrupoleTerm0 = (5.0 * nDotDn[k]) * nr0[k] - 2.0 * qn0[k];
                double const quadrupoleTerm1 = (5.0 * nDotDn[k]) * nr1[k] - 2.0 * qn1[k];
                double const quadrupoleTerm2 = (5.0 * nDotDn[k]) * nr2[k] - 2.0 * qn2[k];
                h0[k] = dipoleScale[k] * dipoleTerm0 + quadrupoleScale[k] * quadrupoleTerm0;
                h1[k] = dipoleScale[k] * dipoleTerm1 + quadrupoleScale[k] * quadrupoleTerm1;
                h2[k] = dipoleScale[k] * dipoleTerm2 + quadrupoleScale[k] * quadrupoleTerm2;
            }

            // Boost to the lab.  For a stationary group ``gammaMinusOne == 0``
            // exactly, so the add is a no-op and the scalar reference's branch
            // is not needed.
            std::array<double, W> ePar;
            for(std::size_t k = 0; k < W; ++k)
            {
                std::size_t const i = base + k;
                ePar[k] = h0[k] * batch.bhatX[i] + h1[k] * batch.bhatY[i]
                    + h2[k] * batch.bhatZ[i];
            }
            for(std::size_t k = 0; k < W; ++k)
            {
                std::size_t const i = base + k;
                double const g1 = batch.gammaMinusOne[i];
                h0[k] = h0[k] + g1 * (h0[k] - ePar[k] * batch.bhatX[i]);
                h1[k] = h1[k] + g1 * (h1[k] - ePar[k] * batch.bhatY[i]);
                h2[k] = h2[k] + g1 * (h2[k] - ePar[k] * batch.bhatZ[i]);
            }

            // Exact Lienard monopole of the group charge.
            std::array<double, W> rb;
            for(std::size_t k = 0; k < W; ++k)
            {
                std::size_t const i = base + k;
                rb[k] = r0[k] * batch.betaX[i] + r1[k] * batch.betaY[i]
                    + r2[k] * batch.betaZ[i];
            }
            std::array<double, W> monopoleF;
            for(std::size_t k = 0; k < W; ++k)
            {
                std::size_t const i = base + k;
                double const invG2 = batch.invGamma2[i];
                double const denom = d2[k] * invG2 + rb[k] * rb[k];
                monopoleF[k] = (denom > 0.0)
                    ? kCoulombConstant * batch.charge[i] * invG2 / (denom * std::sqrt(denom))
                    : 0.0;
            }
            for(std::size_t k = 0; k < W; ++k)
            {
                ex[k] = (active[k] != 0.0) ? (h0[k] + monopoleF[k] * r0[k]) : 0.0;
                ey[k] = (active[k] != 0.0) ? (h1[k] + monopoleF[k] * r1[k]) : 0.0;
                ez[k] = (active[k] != 0.0) ? (h2[k] + monopoleF[k] * r2[k]) : 0.0;
            }
        }
    } // namespace

    Vec3 evaluateFarTailE(
        FarTailBatch const& batch,
        Vec3 const& position)
    {
        Vec3 const r{
            position[0] - batch.centre[0],
            position[1] - batch.centre[1],
            position[2] - batch.centre[2]};
        double const d2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        if(d2 == 0.0)
            return {0.0, 0.0, 0.0};

        Vec3 sum{0.0, 0.0, 0.0};
        constexpr std::size_t kWidth = 8;
        std::size_t i = 0;
        for(; i + kWidth <= batch.count; i += kWidth)
        {
            std::array<double, kWidth> ex{};
            std::array<double, kWidth> ey{};
            std::array<double, kWidth> ez{};
            evaluateBatchBlock<kWidth>(batch, i, position, ex, ey, ez);
            // Keep the group order of the recursive sum so the result stays
            // bit-identical.
            for(std::size_t k = 0; k < kWidth; ++k)
            {
                sum[0] += ex[k];
                sum[1] += ey[k];
                sum[2] += ez[k];
            }
        }
        for(; i < batch.count; ++i)
        {
            std::array<double, 1> ex{};
            std::array<double, 1> ey{};
            std::array<double, 1> ez{};
            evaluateBatchBlock<1>(batch, i, position, ex, ey, ez);
            sum[0] += ex[0];
            sum[1] += ey[0];
            sum[2] += ez[0];
        }
        return sum;
    }

    Vec3 evaluateFarTailB(
        FarTailBatch const& batch,
        Vec3 const& position)
    {
        auto const e = evaluateFarTailE(batch, position);
        Vec3 const& v = batch.meanVelocity;
        double const inverseC2 = 1.0 / (kSpeedOfLight * kSpeedOfLight);
        return {
            (v[1] * e[2] - v[2] * e[1]) * inverseC2,
            (v[2] * e[0] - v[0] * e[2]) * inverseC2,
            (v[0] * e[1] - v[1] * e[0]) * inverseC2};
    }


} // namespace picBunch
