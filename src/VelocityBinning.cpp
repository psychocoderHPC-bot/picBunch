// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Tolerance-driven velocity-space binning for multi-velocity bunches.  See
// ``picBunch/VelocityBinning.hpp`` for the model and the policy.

#include "picBunch/VelocityBinning.hpp"

#include "picBunch/detail/HostReduction.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace picBunch
{
    namespace
    {
        using Vec3 = Vec3;
        using KernelVec3 = alpaka::Vec<double, 3u>;
        using KernelIndex3 = alpaka::Vec<std::size_t, 3u>;

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
            // Measured on the 64-thread host: one OpenMP launch costs about
            // 7 ms, which exceeds the work of these binning kernels.  A serial
            // alpaka kernel keeps the conversion and avoids that fixed cost.
            switch(detail::hostReductionExecutorOverride())
            {
            case detail::HostReductionExecutor::Parallel:
                queue.enqueue(alpaka::onHost::FrameSpec{blocks, frame}, kernel, args...);
                break;
            case detail::HostReductionExecutor::Serial:
            case detail::HostReductionExecutor::Default:
            default:
                queue.enqueue(
                    alpaka::onHost::FrameSpec{blocks, frame, alpaka::exec::cpuSerial}, kernel, args...);
                break;
            }
            alpaka::onHost::wait(queue);
        }

        //! Launch a kernel that reduces scalar extrema/invalid indices.  The
        //! serial executor can update the outputs directly; only the parallel
        //! executor needs atomics.  Both paths use the same per-particle
        //! arithmetic, and min/max are order independent.
        template<typename TKernelSerial, typename TKernelAtomic, typename... TArgs>
        void runHostReductionKernel(std::size_t count, TArgs... args)
        {
            if(count == 0u)
                return;
            std::uint32_t const frame = 64u;
            auto const blocks = static_cast<std::uint32_t>((count + frame - 1u) / frame);
            auto& queue = hostReductionQueue();
            switch(detail::hostReductionExecutorOverride())
            {
            case detail::HostReductionExecutor::Parallel:
                queue.enqueue(alpaka::onHost::FrameSpec{blocks, frame}, TKernelAtomic{}, args...);
                break;
            case detail::HostReductionExecutor::Serial:
            case detail::HostReductionExecutor::Default:
            default:
                queue.enqueue(
                    alpaka::onHost::FrameSpec{blocks, frame, alpaka::exec::cpuSerial},
                    TKernelSerial{},
                    args...);
                break;
            }
            alpaka::onHost::wait(queue);
        }

        //! Per-particle lab-frame bookkeeping: absolute charge, gamma, and the
        //! scalar reductions the binning model needs.  The serial executor
        //! updates them directly; the parallel executor uses alpaka atomics.
        //! Min/max are exact and order independent.
        template<bool TAtomic>
        struct LabMomentumKernel
        {
            template<
                typename TAcc,
                alpaka::concepts::IMdSpan TUx,
                alpaka::concepts::IMdSpan TUy,
                alpaka::concepts::IMdSpan TUz,
                alpaka::concepts::IMdSpan TWeight,
                alpaka::concepts::IMdSpan TCharge,
                alpaka::concepts::IMdSpan TAbsoluteCharge,
                alpaka::concepts::IMdSpan TGamma,
                alpaka::concepts::IMdSpan TScaleU,
                alpaka::concepts::IMdSpan TFirstBadMomentum,
                alpaka::concepts::IMdSpan TFirstBadCharge>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                std::size_t count,
                TUx const& ux,
                TUy const& uy,
                TUz const& uz,
                TWeight const& weight,
                TCharge const& charge,
                TAbsoluteCharge absoluteCharge,
                TGamma gamma,
                TScaleU scaleU,
                TFirstBadMomentum firstBadMomentum,
                TFirstBadCharge firstBadCharge) const
            {
                for(auto [p] : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{count}))
                {
                    bool const badMomentum = !(alpaka::math::isfinite(ux[p])
                                               && alpaka::math::isfinite(uy[p])
                                               && alpaka::math::isfinite(uz[p]));
                    bool const badCharge = !alpaka::math::isfinite(weight[p] * charge[p]);
                    if constexpr(TAtomic)
                    {
                        if(badMomentum)
                            alpaka::onAcc::atomicMin(acc, firstBadMomentum.data(), p);
                        if(badCharge)
                            alpaka::onAcc::atomicMin(acc, firstBadCharge.data(), p);
                        if(!badMomentum)
                        {
                            double const absMax = alpaka::math::max(
                                alpaka::math::abs(ux[p]),
                                alpaka::math::max(alpaka::math::abs(uy[p]), alpaka::math::abs(uz[p])));
                            alpaka::onAcc::atomicMax(acc, scaleU.data(), absMax);
                        }
                    }
                    else
                    {
                        std::size_t const index = static_cast<std::size_t>(p);
                        if(badMomentum)
                            firstBadMomentum[0] = std::min(firstBadMomentum[0], index);
                        if(badCharge)
                            firstBadCharge[0] = std::min(firstBadCharge[0], index);
                        if(!badMomentum)
                        {
                            double const absMax = alpaka::math::max(
                                alpaka::math::abs(ux[p]),
                                alpaka::math::max(alpaka::math::abs(uy[p]), alpaka::math::abs(uz[p])));
                            scaleU[0] = std::max(scaleU[0], absMax);
                        }
                    }
                    absoluteCharge[p] = alpaka::math::abs(weight[p] * charge[p]);
                    double const u2 = ux[p] * ux[p] + uy[p] * uy[p] + uz[p] * uz[p];
                    gamma[p] = alpaka::math::sqrt(1.0 + u2);
                }
            }
        };

        //! Beam-frame momentum of every particle plus the per-axis extrema.
        template<bool TAtomic>
        struct BeamFrameKernel
        {
            template<
                typename TAcc,
                alpaka::concepts::IMdSpan TUx,
                alpaka::concepts::IMdSpan TUy,
                alpaka::concepts::IMdSpan TUz,
                alpaka::concepts::IMdSpan TBeam,
                alpaka::concepts::IMdSpan TUMin,
                alpaka::concepts::IMdSpan TUMax>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                std::size_t count,
                TUx const& ux,
                TUy const& uy,
                TUz const& uz,
                KernelVec3 const e1,
                KernelVec3 const e2,
                KernelVec3 const e3,
                TBeam beam,
                TUMin uMin,
                TUMax uMax) const
            {
                for(auto [p] : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{count}))
                {
                    auto const u = KernelVec3{ux[p], uy[p], uz[p]};
                    // The pin has no Vec dot product; keep the explicit
                    // three-term sums, including their association order.
                    auto const b = KernelVec3{
                        u[0] * e1[0] + u[1] * e1[1] + u[2] * e1[2],
                        u[0] * e2[0] + u[1] * e2[1] + u[2] * e2[2],
                        u[0] * e3[0] + u[1] * e3[1] + u[2] * e3[2]};
                    for(std::uint32_t a = 0u; a < 3u; ++a)
                    {
                        beam[3u * p + a] = b[a];
                        if constexpr(TAtomic)
                        {
                            alpaka::onAcc::atomicMin(acc, uMin.data() + a, b[a]);
                            alpaka::onAcc::atomicMax(acc, uMax.data() + a, b[a]);
                        }
                        else
                        {
                            uMin[a] = std::min(uMin[a], b[a]);
                            uMax[a] = std::max(uMax[a], b[a]);
                        }
                    }
                }
            }
        };

        //! Assign every particle to its box for an existing split.
        struct BoxAssignmentKernel
        {
            template<typename TAcc>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                std::size_t count,
                alpaka::concepts::IMdSpan auto const& beam,
                KernelIndex3 const N,
                KernelVec3 const uMin,
                KernelVec3 const range,
                alpaka::concepts::IMdSpan auto const& width,
                alpaka::concepts::IMdSpan auto boxOf) const
            {
                for(auto [p] : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{count}))
                {
                    KernelIndex3 idx{0u, 0u, 0u};
                    for(std::uint32_t a = 0u; a < 3u; ++a)
                    {
                        if(range[a] <= 0.0)
                            continue;
                        long t = static_cast<long>(
                            alpaka::math::floor((beam[3u * p + a] - uMin[a]) / width[a]));
                        idx[a] = (t < 0) ? 0u
                                         : (static_cast<std::size_t>(t) >= N[a] ? N[a] - 1u
                                                                                : static_cast<std::size_t>(t));
                    }
                    boxOf[p] = idx[0] + N[0] * (idx[1] + N[1] * idx[2]);
                }
            }
        };

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
            Vec3 uMin{0.0, 0.0, 0.0};
            Vec3 uMax{0.0, 0.0, 0.0};
            Vec3 range{0.0, 0.0, 0.0};
            bool active[3] = {false, false, false};
        };

        BeamFrame makeBeamFrame(
            Bunch const& bunch,
            std::vector<double> const& absCharge,
            double scaleU)
        {
            std::size_t const count = bunch.size();
            BeamFrame frame;
            // Charge-weighted mean of the unit momentum directions sets the
            // beam axis; energy spread must not tilt it.  The accumulation is
            // order sensitive, so it stays a serial host reduction rather than
            // a parallel kernel.
            Vec3 meanDir{0.0, 0.0, 0.0};
            double dirWeight = 0.0;
            for(std::size_t p = 0; p < count; ++p)
            {
                double const um
                    = std::sqrt(bunch.ux[p] * bunch.ux[p] + bunch.uy[p] * bunch.uy[p]
                                + bunch.uz[p] * bunch.uz[p]);
                if(um > 0.0)
                {
                    meanDir[0] += absCharge[p] * bunch.ux[p] / um;
                    meanDir[1] += absCharge[p] * bunch.uy[p] / um;
                    meanDir[2] += absCharge[p] * bunch.uz[p] / um;
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
            // The beam-frame vectors are contiguous, so the kernel writes
            // the beam-frame components in place.  The extrema are reduced in
            // the same pass.
            auto const particleExtent = alpaka::Vec<std::size_t, 1u>{count};
            static_assert(std::is_same_v<
                          std::remove_cvref_t<decltype(particleExtent)>::value_type,
                          KernelIndex3::value_type>);
            runHostReductionKernel<BeamFrameKernel<false>, BeamFrameKernel<true>>(
                count,
                count,
                alpaka::makeMdSpan(bunch.ux.data(), particleExtent),
                alpaka::makeMdSpan(bunch.uy.data(), particleExtent),
                alpaka::makeMdSpan(bunch.uz.data(), particleExtent),
                KernelVec3{e1[0], e1[1], e1[2]},
                KernelVec3{e2[0], e2[1], e2[2]},
                KernelVec3{e3[0], e3[1], e3[2]},
                alpaka::makeMdSpan(
                    &frame.uBeam[0][0],
                    alpaka::Vec<std::size_t, 1u>{3u * count}),
                alpaka::makeMdSpan(&frame.uMin[0], alpaka::Vec<std::size_t, 1u>{3u}),
                alpaka::makeMdSpan(&frame.uMax[0], alpaka::Vec<std::size_t, 1u>{3u}));
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
        Idx3 chooseSplit(
            std::size_t B,
            Vec3 const& range,
            bool const* active,
            double gammaWeighted)
        {
            Idx3 N{1u, 1u, 1u};
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
            auto geometricScore = [&](Idx3 const& cand) {
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
            Idx3 best{1u, 1u, 1u};
            double bestScore = std::numeric_limits<double>::infinity();
            for(std::size_t n1 = 1; n1 <= B; n1 *= 2u)
            {
                for(std::size_t n2 = 1; n2 <= B / n1; n2 *= 2u)
                {
                    if((B / n1) % n2 != 0u)
                        continue;
                    Idx3 const cand{n1, n2, B / (n1 * n2)};
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
            Idx3 const& N,
            std::vector<double> const& absCharge,
            std::vector<double> const& particleGamma,
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
            // frame.  For a multi-box split it is replaced below by the
            // absolute-charge-weighted mean of the members.  One box: the
            // charge-weighted mean u.
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
                // Only one box per axis is cheap; the nested loop is control
                // geometry over the box grid, not a particle loop, so it stays
                // serial rather than paying a kernel launch for O(boxes) work.
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
                runHostKernel(
                    count,
                    BoxAssignmentKernel{},
                    count,
                    alpaka::makeMdSpan(
                        &frame.uBeam[0][0],
                        alpaka::Vec<std::size_t, 1u>{3u * count}),
                    KernelIndex3{N[0], N[1], N[2]},
                    KernelVec3{frame.uMin[0], frame.uMin[1], frame.uMin[2]},
                    KernelVec3{frame.range[0], frame.range[1], frame.range[2]},
                    alpaka::makeMdSpan(width, alpaka::Vec<std::size_t, 1u>{3u}),
                    alpaka::makeMdSpan(
                        boxOf.data(),
                        alpaka::Vec<std::size_t, 1u>{count}));
            }
            out.boxOf = boxOf;

            // Replace the geometric box centre by the absolute-charge-weighted
            // mean momentum of the box members and collect the per-box content
            // statistics in the same pass.  The mean minimises the angular and
            // gamma deviations the model charges for, so it is the correct
            // kernel representative at zero extra kernel cost.  Empty boxes
            // keep the centre (they contribute nothing to the field).
            std::vector<Vec3> sumU(boxes);
            std::vector<double> sumW(boxes, 0.0);
            std::vector<double> boxAbsCharge(boxes, 0.0);
            std::vector<double> boxGammaMin(boxes, 0.0), boxGammaMax(boxes, 0.0);
            std::vector<bool> boxUsed(boxes, false);
            for(std::size_t p = 0; p < count; ++p)
            {
                std::size_t const b = boxOf[p];
                double const w = absCharge[p];
                sumU[b][0] += w * bunch.ux[p];
                sumU[b][1] += w * bunch.uy[p];
                sumU[b][2] += w * bunch.uz[p];
                sumW[b] += w;
                boxAbsCharge[b] += w;
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
            if(boxes > 1u)
                for(std::size_t b = 0; b < boxes; ++b)
                    if(sumW[b] > 0.0)
                        for(int a = 0; a < 3; ++a)
                            out.representativeU[b][a] = sumU[b][a] / sumW[b];

            // Largest member angle against the representative.  The maximum is
            // order independent, so one pass over the particles is equivalent
            // to the former per-box scan over all particles and stays O(count).
            std::vector<double> repNorm(boxes, 0.0);
            std::vector<double> maxAngle(boxes, 0.0);
            std::vector<bool> anyMoving(boxes, false);
            for(std::size_t b = 0; b < boxes; ++b)
                if(boxUsed[b])
                    repNorm[b] = norm(out.representativeU[b]);
            for(std::size_t p = 0; p < count; ++p)
            {
                std::size_t const b = boxOf[p];
                double const u2 = bunch.ux[p] * bunch.ux[p] + bunch.uy[p] * bunch.uy[p]
                    + bunch.uz[p] * bunch.uz[p];
                double const uMag = std::sqrt(u2);
                if(uMag > 0.0)
                {
                    anyMoving[b] = true;
                    // The angle against the representative is only defined
                    // when the representative is itself moving.
                    if(repNorm[b] > 0.0)
                    {
                        Vec3 const& uRep = out.representativeU[b];
                        double const cosA
                            = std::max(
                                  -1.0,
                                  std::min(
                                      1.0, (bunch.ux[p] * uRep[0] + bunch.uy[p] * uRep[1]
                                             + bunch.uz[p] * uRep[2])
                                            / (uMag * repNorm[b])));
                        maxAngle[b] = std::max(maxAngle[b], std::acos(cosA));
                    }
                }
            }

            double modelSum = 0.0;
            for(std::size_t b = 0; b < boxes; ++b)
            {
                if(!boxUsed[b])
                    continue;
                // A stationary representative for a moving member is a full
                // angular mismatch: the representative kernel is a static one.
                double const angle
                    = (repNorm[b] == 0.0 && anyMoving[b]) ? std::acos(-1.0) : maxAngle[b];
                double const gamma = boxGammaMax[b];
                // Energy term: a relative gamma shift changes the boosted
                // pattern by at most 2*Delta_gamma/gamma at a node.  The former
                // (Delta_gamma/gamma + maxGammaDev) pair charged ~1.5
                // Delta_gamma/gamma for the same effect and double-counted it;
                // the archived calibration points show that pair over-predicting
                // the measured energy-spread error by ~26x, so a single
                // node-bounded term is used instead.
                double const deltaGamma = (boxGammaMax[b] - boxGammaMin[b]) / gamma;
                modelSum += (boxAbsCharge[b] / totalAbsoluteCharge)
                    * (gamma * angle + 2.0 * deltaGamma / gamma);
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

        std::vector<double> absCharge(count);
        std::vector<double> particleGamma(count);
        double scaleU = 1.0;
        // The earliest offending particle is tracked so the exception still
        // names the first failure in particle order.
        std::size_t firstBadMomentum = count;
        std::size_t firstBadCharge = count;
        auto const particleExtent = alpaka::Vec<std::size_t, 1u>{count};
        auto const scalarExtent = alpaka::Vec<std::size_t, 1u>{1u};
        static_assert(std::is_same_v<
                      std::remove_cvref_t<decltype(particleExtent)>::value_type,
                      KernelIndex3::value_type>);
        static_assert(std::is_same_v<
                      std::remove_cvref_t<decltype(scalarExtent)>::value_type,
                      KernelIndex3::value_type>);
        runHostReductionKernel<LabMomentumKernel<false>, LabMomentumKernel<true>>(
            count,
            count,
            alpaka::makeMdSpan(bunch.ux.data(), particleExtent),
            alpaka::makeMdSpan(bunch.uy.data(), particleExtent),
            alpaka::makeMdSpan(bunch.uz.data(), particleExtent),
            alpaka::makeMdSpan(bunch.weight.data(), particleExtent),
            alpaka::makeMdSpan(bunch.charge.data(), particleExtent),
            alpaka::makeMdSpan(absCharge.data(), particleExtent),
            alpaka::makeMdSpan(particleGamma.data(), particleExtent),
            alpaka::makeMdSpan(&scaleU, scalarExtent),
            alpaka::makeMdSpan(&firstBadMomentum, scalarExtent),
            alpaka::makeMdSpan(&firstBadCharge, scalarExtent));
        if(firstBadMomentum < count || firstBadCharge < count)
        {
            if(firstBadMomentum <= firstBadCharge)
                throw std::invalid_argument("deriveVelocityBinning: a momentum is not finite");
            throw std::invalid_argument("deriveVelocityBinning: a particle charge is not finite");
        }

        double totalAbsoluteCharge = 0.0;
        double gammaWeighted = 0.0;
        for(std::size_t p = 0; p < count; ++p)
        {
            totalAbsoluteCharge += absCharge[p];
            gammaWeighted += absCharge[p] * particleGamma[p];
        }
        if(totalAbsoluteCharge == 0.0)
            throw std::runtime_error("deriveVelocityBinning: the bunch carries no charge");
        gammaWeighted /= totalAbsoluteCharge;

        // Historical single-velocity check, unchanged: within 1e-12 relative the
        // bunch is one velocity class and the kernel is exact.  This is
        // early-exit control logic with no arithmetic, so it stays serial.
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
            out.representativeU = {Vec3{bunch.ux[0], bunch.uy[0], bunch.uz[0]}};
            out.boxOf.assign(count, 0u);
            out.modelSum = 0.0;
            out.modelError = 0.0;
            return out;
        }
        out.multiVelocity = true;

        BeamFrame const frame = makeBeamFrame(bunch, absCharge, scaleU);

        auto splitsFor = [&](std::size_t B) {
            return chooseSplit(B, frame.range, frame.active, gammaWeighted);
        };
        auto modelSumFor = [&](std::size_t B, VelocityBinning& probe) {
            return evaluateSplit(
                bunch, frame, splitsFor(B), absCharge, particleGamma, totalAbsoluteCharge,
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
                // Report the count the model would need beyond the cap, and the
                // model error at the cap itself (a different number: the loop
                // above already failed at the cap before continuing).
                VelocityBinning capProbe;
                double const capError
                    = policy.modelConstant * modelSumFor(policy.maxBins, capProbe);
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
                    + " (estimated error at " + std::to_string(required)
                    + " boxes: " + std::to_string(lastError)
                    + "; at the " + std::to_string(policy.maxBins)
                    + "-box cap: " + std::to_string(capError)
                    + "); reduce the momentum spread, raise --tolerance, or raise --max-velocity-bins");
            }
        }

        evaluateSplit(
            bunch, frame, splitsFor(targetBins), absCharge, particleGamma, totalAbsoluteCharge,
            policy.modelConstant, out);
        return out;
    }

} // namespace picBunch
