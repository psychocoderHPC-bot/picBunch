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
#include <numeric>
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

        //! Assign every particle to its box for an existing split.  Uniform
        //! and quantile edge rules share this kernel: the per-axis boundaries
        //! (``N[a] + 1`` values per axis, concatenated in axis order) are the
        //! only geometry input, so the kernel is the half-open floor lookup
        //! against a general non-uniform edge set.  A particle exactly on a
        //! cut lands in the box to the right of the cut.
        //! Per-axis box lookup against explicit boundaries (the per-axis edge
        //! sets, uniform or quantile).  Returns the box index of ``coord``
        //! for an axis with ``nBoxes`` boxes whose boundaries are the
        //! ``nBoxes + 1`` values ``bounds[0..nBoxes]`` (ascending; a
        //! degenerate axis has all values equal).  Half-open: a coordinate
        //! exactly on an interior boundary takes the box to the right,
        //! matching the historical uniform rule's ``floor`` assignment.
        //! Device-compatible (no host-only constructs).
        [[nodiscard]] constexpr std::size_t boxIndexOfBounds(
            double coord,
            std::size_t nBoxes,
            double const* bounds) noexcept
        {
            // A degenerate axis (all boundaries equal) has no extent; the
            // historical rule put every particle in box 0.
            double const lo = bounds[0u];
            double hi = lo;
            for(std::size_t b = 1u; b <= nBoxes; ++b)
                hi = (bounds[b] > hi) ? bounds[b] : hi;
            if(hi <= lo)
                return 0u;
            long t = 0;
            for(long b = 0; b < static_cast<long>(nBoxes) + 1; ++b)
                if(bounds[static_cast<std::size_t>(b)] < coord)
                    ++t;
            return (t <= 0) ? 0u
                            : (static_cast<std::size_t>(t - 1) >= nBoxes
                                   ? nBoxes - 1u
                                   : static_cast<std::size_t>(t - 1));
        }

        //! Assign every particle to its box for an existing split.  Uniform
        //! and quantile edge rules share this kernel: the per-axis boundaries
        //! (``N[a] + 1`` values per axis, concatenated in axis order) are the
        //! only geometry input, so the kernel is the half-open floor lookup
        //! against a general non-uniform edge set.
        struct BoxAssignmentKernel2
        {
            template<typename TAcc>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                std::size_t count,
                alpaka::concepts::IMdSpan auto const& beam,
                KernelIndex3 const N,
                alpaka::concepts::IMdSpan auto const& boundaries,
                alpaka::concepts::IMdSpan auto boxOf) const
            {
                using IdxT = std::remove_cv_t<decltype(N[0])>;
                IdxT const n0 = N[0];
                IdxT const n1 = N[1];
                IdxT const n2 = N[2];
                double const* const b0 = &boundaries[0u];
                double const* const b1 = &boundaries[static_cast<std::size_t>(n0) + 1u];
                double const* const b2
                    = &boundaries[static_cast<std::size_t>(n0) + 1u
                                  + static_cast<std::size_t>(n1) + 1u];
                for(auto [p] : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{count}))
                {
                    KernelIndex3 idx{
                        static_cast<IdxT>(boxIndexOfBounds(beam[3u * p + 0u], n0, b0)),
                        static_cast<IdxT>(boxIndexOfBounds(beam[3u * p + 1u], n1, b1)),
                        static_cast<IdxT>(boxIndexOfBounds(beam[3u * p + 2u], n2, b2))};
                    boxOf[p] = idx[0] + n0 * (idx[1] + n1 * idx[2]);
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

        //! Measurements behind the near-single-velocity one-box fallback.
        struct SingleVelocityMeasure
        {
            //! Charge-weighted relative spread of ``|u|``.
            double relativeSpread = 0.0;
            //! Largest particle angle (rad) to the one-box representative.
            double maxAngle = 0.0;
        };

        /*! Measure how close ``bunch`` is to a single velocity class.
         *
         * The relative spread uses the charge-weighted RMS deviation of ``|u|``
         * from its charge-weighted mean, so neither a few high-charge outliers
         * nor opposite charges can hide a spread.  The angle is measured
         * against the one-box representative (the charge-weighted mean
         * momentum), exactly the representative the one-box path would use.
         */
        SingleVelocityMeasure measureSingleVelocity(
            Bunch const& bunch,
            std::vector<double> const& absCharge,
            double totalAbsoluteCharge)
        {
            std::size_t const count = bunch.size();
            SingleVelocityMeasure measure;
            Vec3 sumU{0.0, 0.0, 0.0};
            double meanMagnitude = 0.0;
            for(std::size_t p = 0; p < count; ++p)
            {
                Vec3 const u{bunch.ux[p], bunch.uy[p], bunch.uz[p]};
                sumU[0] += absCharge[p] * u[0];
                sumU[1] += absCharge[p] * u[1];
                sumU[2] += absCharge[p] * u[2];
                meanMagnitude += absCharge[p] * norm(u);
            }
            meanMagnitude /= totalAbsoluteCharge;
            if(!(meanMagnitude > 0.0))
            {
                // A bunch of stationary particles never reaches here (the
                // shared-velocity check returns first).  Fail the angle test
                // explicitly so an unexpected zero mean cannot qualify.
                measure.maxAngle = std::acos(-1.0);
                return measure;
            }
            Vec3 const representative{
                sumU[0] / totalAbsoluteCharge,
                sumU[1] / totalAbsoluteCharge,
                sumU[2] / totalAbsoluteCharge};
            double const representativeNorm = norm(representative);
            double weightedVariance = 0.0;
            for(std::size_t p = 0; p < count; ++p)
            {
                Vec3 const u{bunch.ux[p], bunch.uy[p], bunch.uz[p]};
                double const magnitude = norm(u);
                double const deviation = magnitude - meanMagnitude;
                weightedVariance += absCharge[p] * deviation * deviation;
                if(magnitude > 0.0)
                {
                    // A zero representative for moving members is the full
                    // angular mismatch, matching the content model.
                    double angle = std::acos(-1.0);
                    if(representativeNorm > 0.0)
                    {
                        double const cosine
                            = std::max(
                                -1.0,
                                std::min(1.0, dot(u, representative) / (magnitude * representativeNorm)));
                        angle = std::acos(cosine);
                    }
                    measure.maxAngle = std::max(measure.maxAngle, angle);
                }
            }
            measure.relativeSpread
                = std::sqrt(weightedVariance / totalAbsoluteCharge) / meanMagnitude;
            return measure;
        }

        //! Per-axis interior edges of a quantile split: ``N[a] - 1`` cut
        //! positions of axis ``a`` at equal cumulative charge fraction of the
        //! particles' beam-frame coordinate, plus the per-axis extent needed
        //! by the box geometry.  ``frac = q / N[a]`` (the median, not the
        //! mean, of the cumulative fractions) keeps a particle exactly on a
        //! cut in the box above it, so the charge fractions per bin are
        //! (approximately) equal: the half-open floor assignment then puts
        //! the cut particle in box ``q`` (the box to the right).
        //!
        //! The sort is a stable, charge-weighted quantile (no interpolation
        //! between distinct coordinates), deterministic in particle order for
        //! a deterministic fixture.
        struct QuantileAxis
        {
            std::vector<double> cuts; //!< size N[a] - 1 (empty when N[a] == 1)
            double range = 0.0;      //!< uMax[a] - uMin[a]
        };

        QuantileAxis quantileAxis(
            std::vector<Vec3> const& uBeam,
            int a,
            std::size_t nBoxes,
            std::vector<double> const& absCharge,
            double totalAbsoluteCharge,
            double uMin,
            double uMax)
        {
            QuantileAxis out;
            if(nBoxes <= 1u || uMax - uMin <= 0.0)
                return out;
            std::vector<std::size_t> order(uBeam.size());
            std::iota(order.begin(), order.end(), 0u);
            std::stable_sort(order.begin(), order.end(), [&](std::size_t const p, std::size_t const q) {
                return uBeam[p][static_cast<std::size_t>(a)]
                    < uBeam[q][static_cast<std::size_t>(a)];
            });
            out.range = uMax - uMin;
            out.cuts.resize(nBoxes - 1u);
            double cumulative = 0.0;
            std::size_t cursor = 0u;
            for(std::size_t q = 0; q + 1u < nBoxes; ++q)
            {
                double const target = (static_cast<double>(q + 1u)
                                       / static_cast<double>(nBoxes))
                                      * totalAbsoluteCharge;
                while(cursor < order.size() && cumulative < target)
                    cumulative += absCharge[order[cursor++]];
                out.cuts[q] = uBeam[order[cursor]][static_cast<std::size_t>(a)];
            }
            return out;
        }

        //! Adaptive per-axis edges: dynamic programming over candidate cut
        //! positions that minimises the calibrated content model for a fixed
        //! box count ``nBoxes``.
        //!
        //! The box decomposition is a product of per-axis partitions, and the
        //! content model is additive over the 3-D boxes.  A box's content
        //! factorises into per-axis 1-D bins (the members of a box are exactly
        //! the particles in the corresponding per-axis bins, because the
        //! angular representative, the per-box |u| and the per-box gamma are
        //! computed from those members alone), so the per-box model term is a
        //! product of three 1-D factors.  The sum over the 3-D boxes therefore
        //! separates into three independent per-axis sums, and the global
        //! minimum is obtained by minimising each axis's 1-D cost on its own:
        //! the DP needs only the charge, the |u| and the gamma of each particle,
        //! not the full box assignment.
        //!
        //! Complexity: for an axis with ``K`` candidate cuts and ``nBoxes``
        //! boxes the DP is O(K^2 * nBoxes).  Candidates are the distinct
        //! reduced-velocity values (a reduced set of at most ``kAdaptiveEdgeMaxCandidates``
        //! is used when there are more), so for a few thousand particles and a
        //! few dozen candidates the cost is a few milliseconds at startup and
        //! the deposition cost is unchanged (the box geometry and the field
        //! construction are identical for all edge rules).
        //!
        //! A 1-D bin is [lo, hi); the half-open assignment puts a particle
        //! exactly on a cut in the bin to the right, so the last bin is closed
        //! on the right.  ``binCost`` is the per-axis 1-D content of the
        //! calibrated model for particles ``[first, last)``.
        std::vector<double> adaptiveAxisCuts(
            std::vector<Vec3> const& uBeam,
            int a,
            std::size_t nBoxes,
            std::vector<double> const& absCharge,
            std::vector<double> const& particleGamma,
            double totalAbsoluteCharge,
            double modelConstant,
            bool inBunch)
        {
            constexpr std::size_t kAdaptiveEdgeMaxCandidates = 96u;
            std::vector<double> out;
            if(nBoxes <= 1u)
                return out;
            std::size_t const count = uBeam.size();
            if(count <= 1u)
            {
                out.assign(nBoxes - 1u, 0.0);
                return out;
            }
            // Sort by the reduced velocity on axis ``a``.
            std::vector<std::size_t> order(count);
            std::iota(order.begin(), order.end(), 0u);
            std::stable_sort(
                order.begin(), order.end(), [&](std::size_t const p, std::size_t const q) {
                    return uBeam[p][static_cast<std::size_t>(a)]
                        < uBeam[q][static_cast<std::size_t>(a)];
                });
            // Candidate cut positions: distinct reduced-velocity values
            // (reduced to ``kAdaptiveEdgeMaxCandidates`` when there are more).
            std::vector<double> cand;
            cand.reserve(std::min(count, kAdaptiveEdgeMaxCandidates));
            double const uLo = uBeam[order.front()][static_cast<std::size_t>(a)];
            double const uHi = uBeam[order.back()][static_cast<std::size_t>(a)];
            if(uHi - uLo <= 0.0)
            {
                out.assign(nBoxes - 1u, uLo);
                return out;
            }
            std::size_t const distinct = [count, &uBeam, a, &order] {
                std::size_t d = 1u;
                for(std::size_t i = 1u; i < count; ++i)
                    if(uBeam[order[i]][static_cast<std::size_t>(a)]
                       > uBeam[order[i - 1u]][static_cast<std::size_t>(a)])
                        ++d;
                return d;
            }();
            if(distinct <= kAdaptiveEdgeMaxCandidates)
            {
                // Candidate cuts at the midpoints between consecutive distinct
                // reduced-velocity values: the optimal cut for a given
                // partition is at a gap boundary, and a midpoint is in the gap
                // (so a half-open assignment splits the two sides cleanly).  A
                // cut exactly at a particle value would put that particle on the
                // right of the cut, misassigning the cluster.
                std::size_t prev = order.front();
                for(std::size_t i = 1u; i < count; ++i)
                {
                    if(uBeam[order[i]][static_cast<std::size_t>(a)]
                       > uBeam[prev][static_cast<std::size_t>(a)])
                    {
                        cand.push_back(0.5 * (uBeam[prev][static_cast<std::size_t>(a)]
                                              + uBeam[order[i]][static_cast<std::size_t>(a)]));
                        prev = order[i];
                    }
                }
            }
            else
            {
                // Reduced set: equal-charge quantile positions.
                for(std::size_t c = 1u; c < kAdaptiveEdgeMaxCandidates; ++c)
                {
                    double const target = static_cast<double>(c)
                        / static_cast<double>(kAdaptiveEdgeMaxCandidates)
                        * totalAbsoluteCharge;
                    double cumulative = 0.0;
                    std::size_t cursor = 0u;
                    while(cursor < count && cumulative < target)
                        cumulative += absCharge[order[cursor++]];
                    cand.push_back(
                        uBeam[order[cursor < count ? cursor : count - 1u]]
                                          [static_cast<std::size_t>(a)]);
                }
            }
            std::size_t const K = cand.size();
            if(K < nBoxes - 1u)
            {
                // Not enough distinct (or candidate) positions to place
                // nBoxes - 1 cuts: fall back to quantile edges for this axis.
                QuantileAxis const q = quantileAxis(
                    uBeam, a, nBoxes, absCharge, totalAbsoluteCharge, uLo, uHi);
                out.assign(nBoxes - 1u, uLo);
                for(std::size_t k = 0u; k + 1u < nBoxes; ++k)
                    out[k] = q.cuts[k];
                return out;
            }
            // Prefix sums over the sorted particle order.
            std::vector<double> prefixQ(count + 1u, 0.0);
            std::vector<double> prefixQw(count + 1u, 0.0);
            std::vector<double> prefixQu(count + 1u, 0.0);
            std::vector<double> prefixQu2(count + 1u, 0.0);
            std::vector<double> prefixQg(count + 1u, 0.0);
            std::vector<double> prefixQg2(count + 1u, 0.0);
            for(std::size_t i = 0u; i < count; ++i)
            {
                std::size_t const p = order[i];
                double const w = absCharge[p];
                double const mag = std::sqrt(
                    uBeam[p][0] * uBeam[p][0] + uBeam[p][1] * uBeam[p][1]
                    + uBeam[p][2] * uBeam[p][2]);
                double const g = particleGamma[p];
                prefixQ[i + 1u] = prefixQ[i] + w;
                prefixQw[i + 1u] = prefixQw[i] + w * w;
                prefixQu[i + 1u] = prefixQu[i] + w * mag;
                prefixQu2[i + 1u] = prefixQu2[i] + w * mag * mag;
                prefixQg[i + 1u] = prefixQg[i] + w * g;
                prefixQg2[i + 1u] = prefixQg2[i] + w * g * g;
            }
            // Per-axis 1-D content of the calibrated model for a 1-D bin.
            auto binCost = [&](std::size_t first, std::size_t last) -> double {
                if(last <= first)
                    return 0.0;
                double const q = prefixQ[last] - prefixQ[first];
                if(q <= 0.0)
                    return 0.0;
                double const uBar = (prefixQu[last] - prefixQu[first]) / q;
                double const uBar2 = (prefixQu2[last] - prefixQu2[first]) / q;
                double const gammaBar = (prefixQg[last] - prefixQg[first]) / q;
                double const gammaBar2 = (prefixQg2[last] - prefixQg2[first]) / q;
                double const spread2 = uBar2 - uBar * uBar;
                double const gSpread2 = gammaBar2 - gammaBar * gammaBar;
                double const angle = std::sqrt(std::max(0.0, spread2)) / std::max(uBar, 1.0e-30);
                double const energy = std::max(0.0, gSpread2) / std::max(gammaBar, 1.0e-30);
                return (q / totalAbsoluteCharge)
                    * modelConstant
                    * (kAngularModelConstant
                           * (inBunch ? kInBunchAngularGammaAmplification * gammaBar : 1.0)
                           * angle * angle
                       + 2.0 * energy / gammaBar);
            };
            // DP: best[i][j] = minimum 1-D cost of partitioning the first
            // ``i`` distinct candidate positions into ``j`` bins (the last cut
            // at candidate ``i``); the particle bin ``[i, i']`` is the range of
            // sorted particles whose reduced velocity lies in
            // ``[cand[i-1], cand[i'])``.  The outer bounds are added as
            // candidate 0 (uLo) and K+1 (uHi).
            std::vector<double> candAll(K + 2u);
            candAll[0] = uLo;
            for(std::size_t i = 0u; i < K; ++i)
                candAll[i + 1u] = cand[i];
            candAll[K + 1u] = uHi;
            // First particle index at or above candidate position ``i``.
            std::vector<std::size_t> firstIdx(K + 2u);
            for(std::size_t i = 0u; i <= K + 1u; ++i)
            {
                std::size_t lo = 0u, hi = count;
                while(lo < hi)
                {
                    std::size_t const mid = (lo + hi) / 2u;
                    if(uBeam[order[mid]][static_cast<std::size_t>(a)] < candAll[i])
                        lo = mid + 1u;
                    else
                        hi = mid;
                }
                firstIdx[i] = lo;
            }
            // The DP table is indexed by the bin count j in [1, nBoxes], so it
            // needs nBoxes + 1 rows (row 0 is unused).
            std::vector<std::vector<double>> best(nBoxes + 1u);
            std::vector<std::vector<std::size_t>> cutOf(nBoxes + 1u);
            for(std::size_t j = 1u; j <= nBoxes; ++j)
            {
                best[j].assign(K + 2u, std::numeric_limits<double>::infinity());
                cutOf[j].assign(K + 2u, 0u);
            }
            for(std::size_t i = 1u; i <= K + 1u; ++i)
            {
                if(i < 1u)
                    continue;
                best[1u][i] = binCost(firstIdx[0u], firstIdx[i]);
                cutOf[1u][i] = 0u;
            }
            for(std::size_t j = 2u; j <= nBoxes; ++j)
            {
                for(std::size_t i = j; i <= K + 1u; ++i) // need >= j-1 cuts
                {
                    double bestCost = std::numeric_limits<double>::infinity();
                    std::size_t bestPrev = 0u;
                    for(std::size_t i2 = j - 1u; i2 < i; ++i2)
                    {
                        double const prev = best[j - 1u][i2];
                        if(!std::isfinite(prev))
                            continue;
                        double const c = prev + binCost(firstIdx[i2], firstIdx[i]);
                        if(c < bestCost)
                        {
                            bestCost = c;
                            bestPrev = i2;
                        }
                    }
                    if(std::isfinite(bestCost))
                    {
                        best[j][i] = bestCost;
                        cutOf[j][i] = bestPrev;
                    }
                }
            }
            std::size_t i = K + 1u; // last bin ends at uHi
            std::vector<std::size_t> chosen;
            for(std::size_t j = nBoxes; j >= 2u; --j)
            {
                if(!std::isfinite(best[j][i]))
                {
                    // Not enough distinct candidates for this box count: fall
                    // back to quantile edges for this axis.
                    QuantileAxis const q = quantileAxis(
                        uBeam, a, nBoxes, absCharge, totalAbsoluteCharge, uLo, uHi);
                    out.assign(nBoxes - 1u, uLo);
                    for(std::size_t k = 0u; k + 1u < nBoxes; ++k)
                        out[k] = q.cuts[k];
                    return out;
                }
                chosen.push_back(cutOf[j][i]);
                i = cutOf[j][i];
            }
            std::reverse(chosen.begin(), chosen.end());
            // ``chosen`` holds candidate indices; dereference them into the
            // actual cut positions.
            out.reserve(chosen.size());
            for(std::size_t const idx : chosen)
                out.push_back(candAll[idx]);
            return out;
        }

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

        //! Per-axis ``nBoxes + 1`` boundaries for the box geometry and the
        //! assignment.  Uniform edges (``axis.cuts`` empty) divide the active
        //! extent equally; quantile edges use the cut positions with
        //! ``uMin`` / ``uMax`` as the outer bounds.  A degenerate axis (one
        //! box, or no extent) has all boundaries equal to ``uMin``: the box
        //! is a single point and the assignment kernel never divides by zero.
        std::vector<double> buildAxisBounds(
            BeamFrame const& frame,
            int a,
            std::size_t nBoxes,
            QuantileAxis const& axis)
        {
            std::vector<double> b(nBoxes + 1u);
            double const lo = frame.uMin[static_cast<std::size_t>(a)];
            double const hi = frame.uMax[static_cast<std::size_t>(a)];
            std::fill(b.begin(), b.end(), lo);
            if(nBoxes == 1u || hi <= lo)
                return b;
            if(axis.cuts.empty())
            {
                double const width = (hi - lo) / static_cast<double>(nBoxes);
                for(std::size_t i = 0; i <= nBoxes; ++i)
                    b[i] = lo + static_cast<double>(i) * width;
                return b;
            }
            b[0] = lo;
            for(std::size_t i = 0; i + 1u < nBoxes; ++i)
                b[i + 1u] = axis.cuts[i];
            b[nBoxes] = hi;
            // The adaptive DP cuts are sorted by construction; the quantile
            // cuts are sorted by the particle sort.  Enforce monotonicity so a
            // repeated candidate (a zero-extent bin) never inverts a boundary.
            for(std::size_t i = 1u; i < nBoxes; ++i)
                if(b[i] < b[i - 1u])
                    b[i] = b[i - 1u];
            return b;
        }

        //! Assign every particle to its box for a split and evaluate the
        //! calibrated content model.  ``out`` receives the geometry, the
        //! representatives and the per-particle assignment.
        //!
        //! When ``quantileAxes`` holds non-empty per-axis cut positions (and
        //! the axis is active), the assignment uses those cuts instead of
        //! uniform widths: the per-axis cumulative charge fraction along the
        //! axis is equal across the bins, which bounds the per-box angular
        //! spread by the per-box charge spread instead of by the box width
        //! (equal-width edges put a dense velocity cluster in a box whose
        //! angular spread can exceed the sparse outer box).  The geometric
        //! centre of each box (used only for empty boxes and the initial
        //! representative) is the midpoint of its two adjacent cuts (the
        //! global minimum for the first box, the maximum for the last), so
        //! the box geometry stays consistent with the assignment.
        double evaluateSplit(
            Bunch const& bunch,
            BeamFrame const& frame,
            Idx3 const& N,
            std::vector<double> const& absCharge,
            std::vector<double> const& particleGamma,
            double totalAbsoluteCharge,
            double modelConstant,
            bool inBunch,
            VelocityBinning& out,
            std::array<QuantileAxis, 3> const& quantileAxes = {})
        {
            std::size_t const count = bunch.size();
            std::size_t const boxes = N[0] * N[1] * N[2];

            // Per-axis effective range: the quantile cuts span the same extent
            // as the uniform rule (first cut at uMin, last at uMax, by
            // construction), so the range is unchanged; the kernel uses the
            // cuts instead of a constant width.
            out.perAxis = N;
            out.count = boxes;
            out.basis = frame.basis;
            out.uMin = frame.uMin;
            out.uMax = frame.uMax;
            out.representativeU.resize(boxes);

            // Box geometry (beam frame) per axis: uniform rule is the equal
            // subdivision of [uMin, uMax]; the quantile rule is the cut
            // positions, with uMin / uMax as the outer bounds.  The box centre
            // (used only for empty boxes and the initial representative) is
            // the midpoint of the two adjacent bounds, so the geometric
            // centres are consistent with whichever rule assigns the particles.
            std::vector<std::vector<double>> const bounds{
                buildAxisBounds(frame, 0, N[0], quantileAxes[0]),
                buildAxisBounds(frame, 1, N[1], quantileAxes[1]),
                buildAxisBounds(frame, 2, N[2], quantileAxes[2]),
            };
            auto axisCentre = [&](int a, std::size_t i) {
                return 0.5 * (bounds[static_cast<std::size_t>(a)][i]
                              + bounds[static_cast<std::size_t>(a)][i + 1u]);
            };

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
                                axisCentre(0, i1),
                                axisCentre(1, i2),
                                axisCentre(2, i3)};
                            for(int a = 0; a < 3; ++a)
                                out.representativeU[box][a]
                                    = centre[0] * frame.basis[0][a] + centre[1] * frame.basis[1][a]
                                        + centre[2] * frame.basis[2][a];
                        }
            }

            std::vector<std::size_t> boxOf(count, 0u);
            if(boxes > 1u)
            {
                // The assignment is the same per-axis floor lookup for both
                // rules: uniform edges divide the axis range equally, quantile
                // edges use the per-axis cut positions.  Both are carried in a
                // single per-axis array of ``N[a] + 1`` boundaries, so the
                // kernel needs only the array and the per-box width is not a
                // separate input.  A particle exactly on a cut lands in the
                // box to the right of the cut (half-open intervals), matching
                // the ``boxIndex`` behaviour of the uniform rule.
                std::vector<double> boundaries;
                boundaries.reserve((N[0] + 1u) + (N[1] + 1u) + (N[2] + 1u));
                for(int a = 0; a < 3; ++a)
                {
                    auto const& b = bounds[a];
                    for(int i = 0; i <= static_cast<int>(N[a]); ++i)
                        boundaries.push_back(b[i]);
                }
                runHostKernel(
                    count,
                    BoxAssignmentKernel2{},
                    count,
                    alpaka::makeMdSpan(
                        &frame.uBeam[0][0],
                        alpaka::Vec<std::size_t, 1u>{3u * count}),
                    KernelIndex3{N[0], N[1], N[2]},
                    alpaka::makeMdSpan(
                        boundaries.data(),
                        alpaka::Vec<std::size_t, 1u>{boundaries.size()}),
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
                // Angular term: the field is a superposition over all
                // particles, so the tilt errors of a box's members partially
                // cancel at a node instead of adding up to the worst member.
                // The measured binning error scales with the box angular
                // width squared (W37 calibration: 256-source gamma-500 bunches
                // at 50/150 mrad divergence, 1-16 boxes; the largest
                // measured/model ratio is 0.34 for C_ang = 1, a 3x safety
                // factor).  The former gamma * Delta_theta worst case
                // over-predicted the same bunch by 40x and drove the auto
                // policy to order 1e5 boxes.
                // In-bunch peak-normalised amplification: the in-bunch
                // transverse B and E peaks are suppressed by ~1/gamma for a
                // relativistic beam, so the in-bunch peak-normalised relative
                // error is amplified by ~gamma relative to the far-field
                // angular error (real-bunch calibration: measured in-bunch
                // driver 1.24 at one box vs model sum 1.14e-3, ratio ~1.1e3,
                // gamma ~ 490).  The far-field angular calibration
                // (kAngularModelConstant) is unchanged; this factor applies to
                // the in-bunch peak-normalised criterion the auto policy uses
                // to drive the in-bunch max below the tolerance.
                // The in-bunch peak-normalised amplification (gamma factor) is
                // applied only when the in-bunch criterion is active for this
                // call; otherwise the far-field angular calibration
                // (kAngularModelConstant) is unchanged.  The flag is threaded
                // through from the policy (see below).
                double const angular
                    = kAngularModelConstant
                      * (inBunch ? kInBunchAngularGammaAmplification * gamma : 1.0)
                      * angle * angle;
                // Energy term: a relative gamma shift changes the boosted
                // pattern by at most 2*Delta_gamma/gamma at a node.  The former
                // (Delta_gamma/gamma + maxGammaDev) pair charged ~1.5
                // Delta_gamma/gamma for the same effect and double-counted it;
                // the archived calibration points show that pair over-predicting
                // the measured energy-spread error by ~26x, so a single
                // node-bounded term is used instead.
                double const deltaGamma = (boxGammaMax[b] - boxGammaMin[b]) / gamma;
                modelSum += (boxAbsCharge[b] / totalAbsoluteCharge)
                    * (angular + 2.0 * deltaGamma / gamma);
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
        // With per-axis boundaries populated (``axisBoundaries``), use the
        // half-open floor lookup against the actual edges (uniform or
        // quantile); without them (a single-velocity bunch, or a binning not
        // derived through ``deriveVelocityBinning``), fall back to the
        // uniform-width rule from ``uMin`` / ``uMax`` / ``perAxis``.
        if(!axisBoundaries.empty())
        {
            std::size_t const offset[3] = {
                0u,
                perAxis[0] + 1u,
                perAxis[0] + 1u + perAxis[1] + 1u};
            for(int a = 0; a < 3; ++a)
            {
                idx[a] = boxIndexOfBounds(
                    coords[a], perAxis[static_cast<std::size_t>(a)],
                    &axisBoundaries[offset[static_cast<std::size_t>(a)]]);
            }
            return idx[0] + perAxis[0] * (idx[1] + perAxis[1] * idx[2]);
        }
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
        // Per-axis quantile cuts for a candidate split, computed from the
        // particle set at startup (the absolute charge of each particle, the
        // bin content the model charges for).  Recomputed per candidate count
        // during the auto search; the same function is reused for the final
        // split so the final geometry matches the model's geometry exactly.
        auto quantileAxesFor = [&](std::size_t B) -> std::array<QuantileAxis, 3> {
            std::array<QuantileAxis, 3> axes;
            Idx3 const N = chooseSplit(B, frame.range, frame.active, gammaWeighted);
            for(int a = 0; a < 3; ++a)
            {
                axes[a] = quantileAxis(
                    frame.uBeam,
                    a,
                    N[static_cast<std::size_t>(a)],
                    absCharge,
                    totalAbsoluteCharge,
                    frame.uMin[static_cast<std::size_t>(a)],
                    frame.uMax[static_cast<std::size_t>(a)]);
            }
            return axes;
        };
        // Per-axis adaptive (DP) cuts for a candidate split, computed from the
        // particle set at startup.  Recomputed per candidate count during the
        // auto search; the same function is reused for the final split so the
        // final geometry matches the model's geometry exactly.
        auto adaptiveAxesFor = [&](std::size_t B) -> std::array<QuantileAxis, 3> {
            std::array<QuantileAxis, 3> axes;
            Idx3 const N = chooseSplit(B, frame.range, frame.active, gammaWeighted);
            for(int a = 0; a < 3; ++a)
            {
                axes[a].cuts = adaptiveAxisCuts(
                    frame.uBeam, a, N[static_cast<std::size_t>(a)], absCharge, particleGamma,
                    totalAbsoluteCharge, policy.modelConstant, policy.inBunchCriterion);
                axes[a].range = frame.range[static_cast<std::size_t>(a)];
            }
            return axes;
        };
        auto modelSumFor = [&](std::size_t B, VelocityBinning& probe) {
            if(policy.edgeRule == VelocityBinning::EdgeRule::Quantile)
            {
                return evaluateSplit(
                    bunch, frame, splitsFor(B), absCharge, particleGamma, totalAbsoluteCharge,
                    policy.modelConstant, policy.inBunchCriterion, probe, quantileAxesFor(B));
            }
            if(policy.edgeRule == VelocityBinning::EdgeRule::Adaptive)
            {
                return evaluateSplit(
                    bunch, frame, splitsFor(B), absCharge, particleGamma, totalAbsoluteCharge,
                    policy.modelConstant, policy.inBunchCriterion, probe, adaptiveAxesFor(B));
            }
            return evaluateSplit(
                bunch, frame, splitsFor(B), absCharge, particleGamma, totalAbsoluteCharge,
                policy.modelConstant, policy.inBunchCriterion, probe);
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
                // A cold, nearly monoenergetic and nearly collinear bunch is
                // one velocity class: the gamma * Delta_theta worst-case term
                // can ask for millions of boxes even though one box is the
                // physical model.  Only the default-cap auto policy may take
                // this path, so an explicit --velocity-bins or a non-default
                // --max-velocity-bins keeps the historical failure.
                if(policy.mode == VelocityBinningOptions::Mode::Auto
                   && policy.maxBins == kDefaultMaxVelocityBins)
                {
                    SingleVelocityMeasure const measure
                        = measureSingleVelocity(bunch, absCharge, totalAbsoluteCharge);
                    // The one-box fallback is only physical when the one-box
                    // model error (now gamma-amplified for the in-bunch
                    // peak-normalised criterion) actually meets the tolerance.
                    // A high-gamma bunch whose in-bunch transverse B and E
                    // peaks are ~1/gamma suppressed can be "single-velocity" in
                    // spread yet still have a one-box in-bunch error far above
                    // the tolerance (real bunch: 0.74 at one box); in that case
                    // the fallback must not force one box -- the cap-escalation
                    // below is the honest result.
                    VelocityBinning oneBoxProbe;
                    double const oneBoxError
                        = policy.modelConstant * modelSumFor(1u, oneBoxProbe);
                    if(measure.relativeSpread <= kSingleVelocityRelativeSpread
                       && measure.maxAngle <= kSingleVelocityMaxAngle
                       && oneBoxError <= tolerance)
                    {
                        targetBins = 1u;
                        out.singleVelocityFallback = true;
                        out.singleVelocityRelativeSpread = measure.relativeSpread;
                        out.singleVelocityMaxAngle = measure.maxAngle;
                    }
                }
            }
            if(targetBins == 0u && policy.mode == VelocityBinningOptions::Mode::Auto
               && policy.inBunchCriterion)
            {
                // In-bunch criterion: the peak-normalised in-bunch error of a
                // high-gamma beam saturates above the tolerance.  The
                // independent cloud reference (the mesh deposit with the same
                // macroparticles smeared over a cell,
                // docs/measurements/velocity_bin_cloud_reference.txt) measured
                // the in-bunch relative error at 1e-3 for NO count up to 256
                // boxes: the beam-axis components (Bx/Ey/By) and the core 1/r
                // singularity keep the in-bunch components above the tolerance
                // at every box count (the per-component relative error
                // saturates at O(0.1-4.4) and the absolute in-bunch By at
                // ~3.3 T at 256 boxes).  The model error is a monotone decline
                // (the angular term scales with the per-box angular width
                // squared) with a gradual per-doubling improvement (~30-50
                // percent), so a "saturation point" (a halving threshold) is
                 // not well defined for the in-bunch curve.  The honest,
                 // practical result is therefore the in-bunch auto ceiling: the
                 // count at which the remaining accuracy gain no longer
                 // justifies another field solve.  The ceiling is
                 // kInBunchAutoMaxVelocityBins (16) whenever --max-velocity-bins
                 // is at or below the default cap (256): the 256 cap is never
                 // returned by the in-bunch auto path, even when the model
                 // error is still halving at the largest tested count.  Only a
                 // user who raises --max-velocity-bins ABOVE the default cap
                 // gets a higher auto ceiling (a deliberate opt-in to more
                 // field solves).  An explicit --velocity-bins (the far-field
                 // angular calibration) is unaffected.  The achieved accuracy
                 // is reported either way.
                 std::size_t const ceiling
                     = (policy.maxBins > kDefaultMaxVelocityBins)
                           ? policy.maxBins
                           : kInBunchAutoMaxVelocityBins;
                 std::size_t satBins = ceiling;
                for(std::size_t B = 1u; B <= ceiling; B *= 2u)
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
                    // No count up to the in-bunch auto ceiling meets the
                    // tolerance: return the ceiling (capped for cost), report
                    // the achieved accuracy, and never return the 256 cap.
                    targetBins = satBins;
                    out.saturated = true;
                    out.practicalCapApplied = (satBins == ceiling);
                    VelocityBinning satProbe;
                    double const satSum = modelSumFor(satBins, satProbe);
                    out.saturationError = policy.modelConstant * satSum;
                    // error at the count before the returned one (the
                    // improvement the returned count buys; diagnostic)
                    if(satBins > 1u)
                    {
                        VelocityBinning prevProbe;
                        double const prevSum = modelSumFor(satBins / 2u, prevProbe);
                        out.saturationPrevError = policy.modelConstant * prevSum;
                    }
                    std::fprintf(stderr,
                        "picBunch: velocity-bin auto: the in-bunch peak-normalised "
                        "model does not meet tolerance %g.  The in-bunch error of a "
                        "high-gamma beam saturates above the tolerance (no count up to "
                        "the 256-box cap reaches it), so the auto count is capped at "
                        "the %zu-box practical cost ceiling.  Achieved in-bunch model "
                        "error %g at %zu boxes (prev count %g).  The 1e-3 in-bunch "
                        "tolerance is not reachable by the binned model at this gamma; "
                        "validate a production field against a physics run.  Raise "
                        "--max-velocity-bins for more boxes.\\n",
                        tolerance, ceiling, out.saturationError, satBins,
                        out.saturationPrevError);
                }
            }
            if(targetBins == 0u)
            {
                {
                    // Report the count the model would need beyond the cap, and
                    // the model error at the cap itself (a different number: the
                    // loop above already failed at the cap before continuing).
                    VelocityBinning capProbe;
                    double const capError
                        = policy.modelConstant * modelSumFor(policy.maxBins, capProbe);
                    std::size_t required = policy.maxBins;
                    double lastError = capError;
                    for(std::size_t B = policy.maxBins * 2u; B <= (1u << 30); B *= 2u)
                    {
                        VelocityBinning probe;
                        double const modelSum = modelSumFor(B, probe);
                        lastError = policy.modelConstant * modelSum;
                        required = B;
                        if(lastError <= tolerance)
                            break;
                    }
                    std::string const remedy
                        = (lastError > tolerance)
                            ? " no count up to 2^30 boxes meets the tolerance"
                            : "";
                    throw std::runtime_error(
                        "deriveVelocityBinning: the velocity-bin model requires at least "
                        + std::to_string(required)
                        + " boxes for tolerance " + std::to_string(tolerance)
                        + " (estimated error at " + std::to_string(required)
                        + " boxes: " + std::to_string(lastError)
                        + "; at the " + std::to_string(policy.maxBins)
                        + "-box cap: " + std::to_string(capError)
                        + remedy
                        + "); reduce the momentum or angular spread, raise --tolerance, or raise --max-velocity-bins");
                }
            }
        }

        Idx3 const finalSplit = splitsFor(targetBins);
        std::array<QuantileAxis, 3> const finalAxes
            = (policy.edgeRule == VelocityBinning::EdgeRule::Adaptive)
                  ? adaptiveAxesFor(targetBins)
                  : quantileAxesFor(targetBins);
        out.edgeRule = policy.edgeRule;
        evaluateSplit(
            bunch, frame, finalSplit, absCharge, particleGamma, totalAbsoluteCharge,
            policy.modelConstant, policy.inBunchCriterion, out, finalAxes);
        // Populate ``axisBoundaries`` for the final split (uniform or
        // quantile).  A degenerate axis (no extent, or one box) has all
        // boundaries equal to ``uMin[a]``, so the half-open floor lookup in
        // ``boxIndex`` and the kernel never divide by zero.
        {
            std::size_t const N0 = out.perAxis[0], N1 = out.perAxis[1], N2 = out.perAxis[2];
            std::vector<double> b;
            b.reserve((N0 + 1u) + (N1 + 1u) + (N2 + 1u));
            for(int a = 0; a < 3; ++a)
            {
                std::size_t const n = out.perAxis[static_cast<std::size_t>(a)];
                double const lo = out.uMin[static_cast<std::size_t>(a)];
                double const hi = out.uMax[static_cast<std::size_t>(a)];
                if(n == 1u || hi <= lo)
                {
                    for(std::size_t i = 0; i <= n; ++i)
                        b.push_back(lo);
                    continue;
                }
                if(policy.edgeRule == VelocityBinning::EdgeRule::Uniform)
                {
                    double const width = (hi - lo) / static_cast<double>(n);
                    for(std::size_t i = 0; i <= n; ++i)
                        b.push_back(lo + static_cast<double>(i) * width);
                }
                else
                {
                    // Quantile or adaptive: the same cuts the model used during
                    // the search, so the final geometry matches the model's
                    // geometry exactly.
                    b.push_back(lo);
                    for(std::size_t i = 0; i + 1u < n; ++i)
                        b.push_back(finalAxes[static_cast<std::size_t>(a)].cuts[i]);
                    b.push_back(hi);
                }
            }
            out.axisBoundaries = std::move(b);
        }
        return out;
    }

} // namespace picBunch
