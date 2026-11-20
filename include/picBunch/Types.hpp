// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/Parallel.hpp"

#include <alpaka/Vec.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace picBunch
{
    /** Three-component physical vector (positions, momenta, cell sizes). */
    using Vec3 = alpaka::Vec<double, 3u>;
    /** Three-component index/extent vector in nodes or cells. */
    using Idx3 = alpaka::Vec<std::size_t, 3u>;
    /** Signed 64-bit cell-offset vector. */
    using IVec3 = alpaka::Vec<std::int64_t, 3u>;
    /** Unsigned 64-bit cell-extent vector. */
    using UVec3 = alpaka::Vec<std::uint64_t, 3u>;

    //! Particle shape used for the TSC-style charge deposit.
    enum class Shape
    {
        CIC, //!< first-order B-spline, support radius 1 cell
        TSC, //!< second-order B-spline, support radius 1.5 cells
        PQS  //!< third-order B-spline, support radius 2 cells
    };

    //! Number of nodes per physical PIC cell along one axis (``h`` in the algorithm spec).
    //!
    //! v0.1 keeps the mesh at cell resolution.  The tolerance-driven refinement policy is a
    //! later change; all code reads this constant from one place.
    inline constexpr double kRefinement = 1.0;
    //! Interpolation order of the assignment / read-out shape. 2 = quadratic B-spline (TSC).
    inline constexpr int kInterpolationOrder = 2;

    /** Uniform Cartesian output grid.
     *
     * ``n`` nodes per axis, node ``(i,j,k)`` sits at
     * ``origin + (i,j,k) * spacing``.  ``spacing`` is the physical node spacing, i.e.
     * ``kRefinement`` PIC cells.  ``fftPadding`` is the zero-padding factor of the
     * free-space convolution: the transform extent would be ``fftPadding * n``.
     * It is normally selected from ``ComputeOptions::tolerance`` (never below 2).
     * The factor is an oversampling count, not a halo count.  The deposit needs
     * ``M - n >= 2*floor(shapeSupport)`` halo nodes (2 for CIC/TSC, 4 for PQS);
     * ``depositHalo`` holds the per-axis nodes that the factor does not provide
     * (``makeGrid`` sets them, so padding 1 still gets its two halo nodes and
     * the extent becomes ``n + 2``).  ``paddedSize`` is the extent of both the
     * deposit buffer and the transform.
     */
    struct Grid
    {
        Idx3 n{48, 48, 48};
        Vec3 spacing{1.0, 1.0, 1.0}; //!< physical node spacing
        Vec3 origin{0.0, 0.0, 0.0}; //!< physical position of node (0,0,0)
        unsigned fftPadding{2};
        /*! Halo nodes appended above ``fftPadding * n`` per axis.
         *
         * Zero when the FFT padding factor already leaves the stencil enough
         * room.  ``makeGrid`` fills the deficit, so a padding-1 grid gets two
         * extra nodes for CIC/TSC instead of silently wrapping boundary charge.
         */
        Idx3 depositHalo{0u, 0u, 0u};
        //! Deposit shape; the near shell currently supports TSC only.
        Shape shape = Shape::TSC;

        //! Total number of output nodes.
        [[nodiscard]] std::size_t size() const
        {
            return n[0] * n[1] * n[2];
        }
        //! Number of nodes per axis of the padded FFT/deposit box.
        [[nodiscard]] Idx3 paddedSize() const
        {
            return {
                n[0] * fftPadding + depositHalo[0],
                n[1] * fftPadding + depositHalo[1],
                n[2] * fftPadding + depositHalo[2]};
        }
    };

    /** Macro-particle bunch, SI units for positions and charge.
     *
     * ``ux/uy/uz`` are dimensionless momenta ``p/(m c)``.  ``charge`` is the
     * charge of one macro particle in Coulomb, ``weight`` its statistical
     * weight.  The source weight deposited on the grid is ``weight*charge``.
     */
    struct Bunch
    {
        std::vector<double> x, y, z;
        std::vector<double> ux, uy, uz;
        std::vector<double> weight;
        std::vector<double> charge;
        //! SI rest mass per macro particle; empty or non-positive entries fall
        //! back to the configured PIConGPU mass unit in the checkpoint writer.
        std::vector<double> restMass;

        [[nodiscard]] std::size_t size() const
        {
            return x.size();
        }
        void resize(std::size_t n);

        /** min/max corner of the particle positions. */
        [[nodiscard]] std::array<Vec3, 2> bounds() const;
    };

    /** Six staggered field components on one grid.
     *
     * ``data[c]`` is stored with x fastest, then y, then z, on the *output*
     * grid (``grid.n``).  The field of component ``c`` lives at the staggered
     * position ``node + kYeeOffset[c]`` (in PIC-cell units).
     */
    struct Field
    {
        Grid grid;
        std::array<std::vector<double>, 6> data;
        /*! True when ``data`` holds only one contiguous z-slab of the grid.
         *
         * The distributed FFT returns this form: every rank owns
         * ``[slabOffset, slabOffset + slabCount)`` z planes with all x and y
         * nodes of those planes.  A replicated field keeps ``zSlab == false``
         * and every component sized ``grid.size()``.
         */
        bool zSlab = false;
        //! First z plane held by this rank (only meaningful when ``zSlab``).
        std::size_t slabOffset = 0;
        //! Number of z planes held by this rank (only meaningful when ``zSlab``).
        std::size_t slabCount = 0;

        //! z planes held locally (the full grid extent for a replicated field).
        [[nodiscard]] std::size_t planes() const
        {
            return zSlab ? slabCount : grid.n[2];
        }
        //! Number of stored values per component on this rank.
        [[nodiscard]] std::size_t localSize() const
        {
            return grid.n[0] * grid.n[1] * planes();
        }
    };

    /** One component z-slab emitted by the chunked field solve.
     *
     * ``data`` holds ``grid.n[0] * grid.n[1] * zCount`` values with x fastest,
     * covering global planes ``[zOffset, zOffset + zCount)``.  The chunked
     * writer stores exactly this slab at its dataset offset, so no full output
     * grid is ever materialised.
     */
    struct FieldChunk
    {
        Grid grid;
        int component = 0;
        std::size_t zOffset = 0;
        std::size_t zCount = 0;
        std::vector<double> data;
    };

    //! Sink for one ``FieldChunk``; called once per emitted output slab.
    using FieldChunkSink = std::function<void(FieldChunk const&)>;

    /*! One particle patch region of a generated species.
     *
     * ``offset`` is the lower cell index relative to the PIConGPU domain
     * origin and ``extent`` the number of cells covered.  Regions may overlap
     * by one cell; particle ownership is disjoint and is carried separately by
     * the cumulative ``numParticlesOffset`` written next to these arrays.
     */
    struct ParticlePatch
    {
        IVec3 offset{0, 0, 0};
        UVec3 extent{1, 1, 1};
    };

    //! Wall-clock time per pipeline stage, milliseconds.
    struct StageTimes
    {
        double readerMs = 0.0;
        //! Untimed setup before the first deposit: binning, grid, allocations,
        //! and the FFT plan/workspace build (r2c/c2r PlanBuilder::build).
        double planMs = 0.0;
        double depositMs = 0.0;
        double kernelMs = 0.0;
        double fftMs = 0.0;
        //! Time inside the distributed FFT's all-to-all transposes (pack, exchange, unpack).
        double transposeMs = 0.0;
        double readoutMs = 0.0;
        double writerMs = 0.0;
        /*! Measured staging traffic of a chunked solve (0 for the in-core paths).
         *
         * ``chunkReadBytes``/``chunkWrittenBytes`` are the byte counts the
         * out-of-core staging layer actually issued.  ``chunkPeakStagingBytes``
         * is the live staging-file high-water mark (disk, not RSS), and
         * ``chunkPeakMemoryBytes`` is the largest single chunk allocation.
         */
        std::uint64_t chunkReadBytes = 0;
        std::uint64_t chunkWrittenBytes = 0;
        std::uint64_t chunkPeakStagingBytes = 0;
        std::uint64_t chunkPeakMemoryBytes = 0;
        std::uint64_t chunkCells = 0;
    };

    /** Explicit near-shell mode.  See NearShell.hpp for the split and the cost. */
    struct NearShellOptions
    {
        enum class Mode
        {
            None,
            Exact,
            Fast,
            //! Fast evaluator on the accelerator; falls back to Fast on the host.
            FastGpu
        };

        Mode mode = Mode::None;
        //! Shell radius in PIC cells; 0 derives it from ComputeOptions::tolerance.
        double radiusCells = 0.0;
        //! Output tile edge in nodes for the local subtraction FFT.
        //!
        //! 0 selects the automatic rule, which samples the source distribution
        //! and chooses 2 when the near-set pair count clearly favours it, and 4
        //! otherwise.  A positive value is an explicit override and always wins.
        std::size_t tileNodes = 0;
        //! Diagnostic output: receives the tile edge actually used, including
        //! the automatic choice.  May be left null.
        std::size_t* resolvedTileNodes = nullptr;
        //! Diagnostic output: receives the number of nodes per component in the
        //! slab-local correction buffer.  It is at most the rank's field slab
        //! rounded out to whole tile rows, never the full grid once the field
        //! is distributed.  May be left null.
        std::size_t* resolvedCorrectionNodes = nullptr;
        //! Gauss order per panel of the fast remainder (default 5).
        int fastOrder = 5;
        //! Order of the fast oblique singular box.  0 selects the shared
        //! host/device `auto` rule (gamma-driven, per tile); an explicit value
        //! is clamped to an allowed even order in [4, 12].  The value 4 is the
        //! fast path and reproduces the historical device behaviour.
        int fastObliqueOrder = 0;
        //! Diagnostic counter: number of exact (source, node, component) evaluations.
        std::size_t* exactPairs = nullptr;
        //! Which TSC support cell classes use the exact evaluator (fast path):
        //! 0 = observer-containing cell only (hybrid), 1 = plus the six
        //! face-adjacent cells, 2 = all in-support cells, -1 = legacy only,
        //! -2 = automatic (default).  Automatic selects the observer cell at
        //! the default 1e-3 tolerance and all in-support cells at a tolerance
        //! <= 1e-4.  A tile with gamma >= 100 is always all-exact, because the
        //! legacy quadrature cannot resolve that contracted tube.
        int exactCellReach = -2;
        //! Diagnostic counter: exact evaluations among the total pair count.
        std::size_t* exactPairCount = nullptr;
    };

    /*! Safety factor of the velocity-bin angular error term.
     *
     * The angular term of the error model charges ``C_ang * Delta_theta_b^2``
     * per box (see ``VelocityBinning``); the old worst-case form charged
     * ``gamma * Delta_theta_b``, which over-predicts the measured superposition
     * error by orders of magnitude (a gamma-490 bunch at 50 mrad divergence
     * carries a one-box model sum of about 24.6 but a measured binning error of
     * about 6e-4, a 40x over-prediction; see
     * ``docs/measurements/velocity_bin_angular_calibration.csv``).
     *
     * ``C_ang = 1`` is a conservative bound on the measured binning error for
     * the three calibration bunches (gamma 500, 256 sources, 48^3 grid,
     * 1 um cells; angular sigma 50 mrad, 50 mrad, and 150 mrad): the largest
     * measured/model ratio is 0.34, so the safety factor is 3.
     * The bound is validated for a charge-weighted angular sigma below about
     * 0.3 rad and per-box angular width below about 0.26 rad, where the
     * superposition cancellation is near-complete.  Bunches outside that range
     * (for example, the archived 5 degree point, or a per-box width near the
     * beam-frame range) can lose the cancellation; the auto policy then fails
     * at the cap with the required count instead of silently under-binning.
     */
    inline constexpr double kAngularModelConstant = 1.0;

    /*! Amplification of the angular term for the in-bunch peak-normalised error.
     *
     * The far-field angular error (calibrated at ``kAngularModelConstant = 1``)
     * is a lower bound on the in-bunch peak-normalised relative error.  For a
     * relativistic beam the in-bunch transverse B and the in-bunch E peaks are
     * suppressed by ~1/gamma, so the in-bunch relative error is amplified by
     * ~gamma relative to the far-field one.  The real bunch calibration
     * (``docs/measurements/velocity_bin_real_bunch_calibration.txt``) measured
     * the in-bunch peak-normalised driver at ~1.24 for one box, against an
     * angular model term of ``C_ang * theta^2 = 6.4e-4`` (the B=1 model_sum
     * 1.136e-3 is angular 6.4e-4 plus the energy term 4.9e-4; the energy term
     * is the per-particle gamma deviation, not the angular term), a ratio of
     * ~1.9e3 for the angular term.  The gamma-amplification factor
     * ``kInBunchAngularGammaAmplification`` (a modest ~2.2, not the full
     * gamma ~ 490) reproduces the in-bunch scaling the auto policy needs to
     * escalate to the cap for a gamma ~ 490 beam; the residual spread across
     * the box counts is carried by the node-bounded energy term.  This factor
     * applies only to the in-bunch peak-normalised criterion; the far-field
     * angular calibration (``kAngularModelConstant``) is unchanged.
     */
    inline constexpr double kInBunchAngularGammaAmplification = 2.2;

    /*! Overall constant of the velocity-bin error model.
     *
     * The model (see ``VelocityBinning``) estimates the relative error of the
     * binned mesh field as ``modelConstant * modelSum``; ``modelSum`` already
     * carries the calibrated angular coefficient
     * ``kAngularModelConstant`` and the node-bounded energy coefficient.  The
     * shipped value is 1, the unit scaling of both calibrated terms; the CLI
     * exposes ``--bin-model-constant`` as a diagnostic override.
     */
    inline constexpr double kDefaultVelocityModelConstant = 1.0;

    /*! Upper bound of the auto binning policy (box count).
     *
     * Every box costs one deposit plus 13 FFTs on the padded grid, so the cap
     * is a cost guard, not an accuracy limit: the policy reports the box
     * count the model would need and fails when it exceeds the cap.
     */
    inline constexpr std::size_t kDefaultMaxVelocityBins = 256u;

     /*! Halving threshold of the velocity-bin error model.
      *
      * 0.5 means "a doubling cut the model error by at least half".  The
      * in-bunch peak-normalised error of a high-gamma beam is a monotone
      * decline with a gradual per-doubling improvement (~30-50 percent), so
      * this halving threshold is not a useful stop rule for the in-bunch
      * curve; the in-bunch auto policy therefore uses the practical cost
      * ceiling (``kInBunchAutoMaxVelocityBins``) instead of searching for a
      * saturation point.  The constant is kept as the documented halving
      * threshold of the error model and for the far-field auto path.
      */
     inline constexpr double kSaturationImprovement = 0.5;

     /*! Practical cost ceiling of the in-bunch auto policy (box count).
      *
      * The in-bunch peak-normalised error of a high-gamma beam saturates
      * above the tolerance: no count up to the 256-box cap reaches 1e-3 (see
      * ``kSaturationImprovement`` and
      * ``docs/measurements/velocity_bin_cloud_reference.txt``).  Refinement
      * keeps halving the error (the angular term scales with the per-box
      * angular width squared), so the saturation point is the count where the
      * per-doubling improvement stops being at least ``kSaturationImprovement``.
      * For a gamma ~ 490 beam that saturation point sits near or above the 256
      * cap, which costs 256 field solves and is not viable in production.  The
      * in-bunch auto policy therefore stops at this practical maximum instead:
      * it is the count where the remaining accuracy gain no longer justifies
      * another field solve, and the achieved accuracy is reported with the
      * fact that the count was capped for cost (the user sees both).  A user
      * may still raise ``--max-velocity-bins`` above it; the cap is a cost
      * default, not an accuracy limit.  16 boxes (16 field solves) is the
      * shipped value; it is a cost trade-off, revisited against the measured
      * per-box FFT cost if that changes.
      */
     inline constexpr std::size_t kInBunchAutoMaxVelocityBins = 16u;

     /*! Tolerance-driven velocity-space binning of a multi-velocity bunch.
     *
     * The mesh kernel is exact only for a single shared ``u``; it is linear in
     * charge, so the field of a mixed bunch is the sum over velocity classes
     * of the field of each class.  The particles are split into boxes in
     * ``u``-space and each box is evaluated with its box-centre ``u``:
     *
     * - the frame is aligned to the charge-weighted mean momentum direction
     *   ``e3`` (``e1``, ``e2`` complete it deterministically);
     * - the box geometry is a uniform grid over the observed ``u`` extent in
     *   that frame, ``N1 x N2 x N3`` boxes, width ``range_i / N_i`` per axis.
     *   The representative velocity of a box is the box centre; for one box it
     *   is the charge-weighted mean ``u``.
     *
     * The geometry depends only on the bunch ``u`` extent and the box counts,
     * never on which particles fall where.  That keeps a per-tile mesh
     * subtraction (the near shell) bin-consistent with the global mesh for
     * any sub-bunch, so the exact per-pair near term -- which evaluates every
     * source with its own ``u`` -- cancels the mesh exactly: the near shell
     * is bin-free.
     *
     * Error model.  Replacing a particle ``u`` by its box representative
     * perturbs the field by two mechanisms: a transverse tilt of the boosted
     * pattern, and an energy shift, of order ``(2/gamma) * Delta_gamma /
     * gamma`` at a node.  The field is a superposition over all particles, so
     * the tilt errors of the members of a box partially cancel at a node
     * instead of adding up to the worst member: the measured binning error
     * (binned field minus the per-particle-velocity continuum reference,
     * ``tools/velocity-bins/compare_velocity_bins.py``) scales with the box
     * angular *width* squared, not with ``gamma * width``.  With ``q_b`` the
     * absolute charge in box ``b``, ``Q`` the total absolute charge,
     * ``Delta_theta_b`` the largest angle between a member and the box
     * representative (``pi`` when the representative is zero and any member
     * is moving), and ``Delta_gamma_b`` the ``gamma`` range of the box, the
     * estimated relative error is
     *
     *   modelError = C * Sum_b (q_b / Q)
     *                 * (C_ang * Delta_theta_b^2
     *                    + (2/gamma_b) * Delta_gamma_b/gamma_b)
     *
     * with ``C = kDefaultVelocityModelConstant`` (CLI override) and
     * ``C_ang = kAngularModelConstant``.  The angular term is calibrated on
     * the W37 campaign (three synthetic bunches, gamma 500, 1-16 boxes; see
     * ``docs/measurements/velocity_bin_angular_calibration.csv``): ``C_ang = 1``
     * bounds the measured binning error with a safety factor of 3 (largest
     * measured/model ratio 0.34).  The former ``gamma * Delta_theta`` term
     * over-predicted that same bunch by 40x, which is what drove the auto
     * policy to order 1e5 boxes for a realistic high-gamma bunch.  The energy
     * term is the node-bounded single term that replaces the former
     * ``Delta_gamma/gamma + maxGammaDev`` pair, which double-counted the same
     * effect and over-predicted the archived 5% energy-spread points by
     * ~26x.  Both terms are conservative estimates, not guarantees: the
     * angular bound is validated for per-box angular widths below about
     * 0.26 rad and a charge-weighted angular sigma below about 0.3 rad, and
     * the energy term beyond gamma 10 at 5% spread and the measured box range
     * is not validated.
     *
     * ``modelSum`` is the uncalibrated sum; ``modelError`` is the calibrated
     * estimate.  The auto policy chooses the smallest power-of-two box count
     * whose best axis split drives ``modelError`` below the requested
     * tolerance.  At the default cap an essentially single-velocity bunch
     * (see ``kSingleVelocityRelativeSpread`` and ``kSingleVelocityMaxAngle``)
     * falls back to one box; any other infeasible requirement fails with the
     * needed count.
     */
    struct VelocityBinning
    {
        //! True once filled by ``deriveVelocityBinning`` (or by a fixed copy).
        bool derived = false;
        //! True when the bunch carries more than one velocity class.
        bool multiVelocity = false;
        //! Total box count ``N1*N2*N3`` (1 for a single-velocity bunch).
        std::size_t count = 1u;
        //! Box counts per beam-frame axis.
        Idx3 perAxis{1u, 1u, 1u};
        //! Orthonormal beam-frame basis, rows ``e1, e2, e3`` (lab frame).
        std::array<Vec3, 3> basis{{{1.0, 0.0, 0.0},
                                                    {0.0, 1.0, 0.0},
                                                    {0.0, 0.0, 1.0}}};
        //! Box extent in beam-frame ``u`` units, axis ``i``: ``[uMin[i], uMax[i]]``.
        Vec3 uMin{0.0, 0.0, 0.0};
        Vec3 uMax{0.0, 0.0, 0.0};
        //! Representative ``u`` (lab frame) per box, box index order
        //! ``i1 + N1*(i2 + N2*i3)``.
        std::vector<Vec3> representativeU;
        //! Box index of each particle; size equals the bunch size the binning
        //! was derived for.  A fixed binning applied to a sub-bunch must
        //! re-derive the assignment with ``boxIndex``.
        std::vector<std::size_t> boxOf;
        //! Uncalibrated model sum (dimensionless, charge-weighted; angular
        //! term with the shipped ``kAngularModelConstant``, energy term with
        //! its node-bounded coefficient).
        double modelSum = 0.0;
        //! Calibration constant used (``kDefaultVelocityModelConstant`` unless
        //! overridden).
        double modelConstant = kDefaultVelocityModelConstant;
        //! Calibrated estimate ``modelConstant * modelSum``.
        double modelError = 0.0;
        //! True when the auto policy used the near-single-velocity one-box
        //! fallback because the model exceeded the default cap.
        bool singleVelocityFallback = false;
        //! Charge-weighted relative ``|u|`` spread measured for that decision.
        double singleVelocityRelativeSpread = 0.0;
        //! Largest particle angle (rad) to the one-box representative measured
        //! for that decision.
        double singleVelocityMaxAngle = 0.0;
        //! True when the in-bunch criterion did not reach the tolerance within
        //! the in-bunch auto ceiling and the auto policy returned the ceiling
        //! (``kInBunchAutoMaxVelocityBins``, NOT the 256-box cap).  The in-bunch
        //! peak-normalised error of a high-gamma beam saturates above 1e-3
        //! (the beam-axis components and the core 1/r singularity keep it
        //! above) and is a monotone decline, so the ceiling -- the count where
        //! the remaining accuracy gain no longer justifies another field solve
        //! -- is the honest result; the achieved accuracy is ``modelError`` at
        //! that count.
        bool saturated = false;
        //! True when the in-bunch auto count is the practical cost ceiling
        //! ``kInBunchAutoMaxVelocityBins`` (the in-bunch error did not meet the
        //! tolerance at any count below it).  The returned count is the
        //! ceiling, not the 256 cap; the user is told the count was capped for
        //! cost and the achieved accuracy is reported.
        bool practicalCapApplied = false;
        //! The in-bunch model error at the count before the returned one
        //! (``modelConstant * modelSum`` at ``count / 2``); the improvement
        //! that the returned count buys.  Set only on the in-bunch auto path.
        double saturationPrevError = 0.0;
        //! The in-bunch model error at the returned count,
        //! ``modelConstant * modelSum``; the achieved accuracy.
        double saturationError = 0.0;

        //! Edge rule used to place the per-axis box edges.  The box
        //! decomposition (``perAxis``) and the field construction are identical
        //! for all rules; only the per-axis edge placement changes.  Populated
        //! by ``deriveVelocityBinning``; the default (``Uniform``) leaves the
        //! historical behaviour unchanged.
        enum class EdgeRule
        {
            Auto,      //!< build the uniform and adaptive partitions per axis, keep the lower predicted error (default)
            Adaptive,  //!< per-axis edges chosen by DP to minimise the content model
            Uniform,   //!< equal-width edges (the pre-adaptive behaviour; pass --velocity-bin-edges uniform)
            Quantile   //!< per-axis edges at equal cumulative charge fraction
        };
        EdgeRule edgeRule = EdgeRule::Auto;
        //! Per-axis box boundaries: ``N[0] + 1`` then ``N[1] + 1`` then
        //! ``N[2] + 1`` beam-frame ``u`` values, concatenated in axis order.
        //! Uniform rule: equal subdivision of ``[uMin[a], uMax[a]]``;
        //! quantile rule: ``uMin[a]``, the per-axis cut positions, ``uMax[a]``;
        //! adaptive rule: ``uMin[a]``, the DP-chosen cut positions, ``uMax[a]``.
        //! A degenerate axis (no extent, or one box) has all boundaries equal
        //! to ``uMin[a]``.  The half-open floor lookup against these
        //! boundaries is the assignment for all three rules (a particle exactly
        //! on an interior edge lands in the box to the right).  Populated by
        //! ``deriveVelocityBinning`` when ``multiVelocity`` is true; empty
        //! otherwise.
        std::vector<double> axisBoundaries;

        //! Per-axis edge rule used for ``axisBoundaries`` when
        //! ``edgeRule == Auto``: axis ``a`` holds the rule (``Uniform`` or
        //! ``Adaptive``) whose predicted content-model error is lower for that
        //! axis.  Empty for the explicit rules, where every axis uses
        //! ``edgeRule``.
        std::array<EdgeRule, 3> axisEdgeRule{};
        //! Per-axis predicted content-model error (``modelConstant`` scaled out;
        //! the same quantity the DP minimises) of the two candidate partitions
        //! under ``edgeRule == Auto``.  ``axisEdgeModelErrorUniform[a]`` is the
        //! uniform rule's 1-D error on axis ``a`` and
        //! ``axisEdgeModelErrorAdaptive[a]`` is the adaptive rule's.  Populated
        //! only for ``Auto`` and only for axes with more than one box; other
        //! entries are 0.
        std::array<double, 3> axisEdgeModelErrorUniform{};
        std::array<double, 3> axisEdgeModelErrorAdaptive{};

        //! Box of a particle with beam-frame ``u`` coordinates, for re-using a
        //! fixed binning on a sub-bunch (the near shell tile mesh).  Uses the
        //! per-axis boundaries (``axisBoundaries``) when populated, otherwise
        //! the uniform-width rule from ``uMin`` / ``uMax`` / ``perAxis``.
        [[nodiscard]] std::size_t boxIndex(double u1, double u2, double u3) const;
    };

    /*! Policy of the velocity binning. */
    struct VelocityBinningOptions
    {
        enum class Mode
        {
            Auto,    //!< box count from the error model and ``tolerance``
            Explicit //!< fixed box count (calibration, diagnostics)
        };

        Mode mode = Mode::Auto;
        //! Box count used by ``Explicit`` (must be >= 1).
        std::size_t explicitBins = 1u;
        //! Calibration constant ``C`` of the error model (diagnostic override).
        double modelConstant = kDefaultVelocityModelConstant;
        //! Box-count cap of the auto policy; exceeded requirements fail with
        //! the needed count instead of running.
        std::size_t maxBins = kDefaultMaxVelocityBins;
        //! Drive the bin count with the in-bunch peak-normalised error model
        //! (angular term amplified by ~gamma, the suppression of the in-bunch
        //! transverse B and E peaks of a relativistic beam) instead of the
        //! far-field angular calibration.  The in-bunch peak-normalised error
        //! of a high-gamma beam saturates above the tolerance (no count up to
        //! the 256-box cap reaches 1e-3), so when set and no count up to the
        //! in-bunch auto ceiling
        //! (``kInBunchAutoMaxVelocityBins`` unless the user raised
        //! ``maxBins``) meets the tolerance, the default-cap auto policy stops
        //! at the smallest count whose error still improved by at least
        //! ``kSaturationImprovement`` per doubling (``saturated``), or at the
        //! practical cost ceiling (``practicalCapApplied``) when it never
        //! stopped -- it never returns the 256 cap.  The achieved accuracy is
        //! reported.  Default off: the far-field angular calibration
        //! (``kAngularModelConstant``) is unchanged.
        bool inBunchCriterion = false;
        //! Edge rule for the per-axis box edges (``VelocityBinning::EdgeRule``):
        //! ``Auto`` (default) builds the uniform and the adaptive partitions
        //! for each axis, evaluates both with the calibrated content model, and
        //! keeps the partition with the lower predicted error per axis, so it
        //! cannot be worse than uniform by construction; ``Adaptive`` chooses
        //! the interior edges by dynamic programming to minimise the calibrated
        //! content model for the requested count (the per-axis problem is
        //! independent because the decomposition is a product of per-axis
        //! partitions); ``Uniform`` divides each active axis range equally,
        //! the pre-adaptive behaviour; ``Quantile`` places the interior edges
        //! at equal cumulative charge fraction along each axis, computed from
        //! the particle set at startup.  The box decomposition (``explicitBins``
        //! / the auto policy) and the field construction are unchanged; only
        //! the per-axis edge placement changes.  The count-auto policy (the
        //! smallest count whose model error meets the tolerance) runs against
        //! the selected rule, so the returned count is the count for that rule.
        VelocityBinning::EdgeRule edgeRule = VelocityBinning::EdgeRule::Auto;
    };

    /*! Policy of the velocity-moment spectral decomposition (candidate C1).
     *
     * A multi-velocity bunch is normally evaluated by the binned path (one FFT
     * chain per velocity box).  The moment path instead expands the point kernel
     * in a multivariate Taylor series about the absolute-charge-weighted mean
     * ``u0``, deposits the moment densities once, and takes one inverse
     * transform per component -- independent of the number of velocity classes.
     * See ``picBunch/VelocityMoments.hpp``.
     */
    struct VelocityMomentsOptions
    {
        enum class Mode
        {
            Off,       //!< binned path (default; unchanged behaviour)
            Auto,      //!< smallest order (2, then 3) whose a-posteriori probe meets tolerance
            Explicit   //!< fixed order (``order``, 2 or 3)
        };

        Mode mode = Mode::Off;
        //! Moment order for ``Explicit`` (2 or 3).
        std::size_t order = 2u;
        /*! Keep an explicit order even when its a-posteriori probe exceeds the
         * convergence budget.  Diagnostic escape hatch: the delivered field is
         * then known to be unconverged (the CLI warns and the report marks it).
         * It never overrides the non-finite guard.
         */
        bool allowUnconverged = false;
    };

    /** User-facing knobs of the field computation.
     *
     * The user sets an accuracy target, the physical cell size, and the empty
     * shell margin.  Bin counts, interpolation orders and FFT sizes are never
     * user input: ``tolerance`` selects the tunable discretisation choices,
     * the FFT zero padding and the velocity-box count, and the remaining
     * values are constants of the implementation.
     */
    //! Selects the replicated or the slab-decomposed global FFT.
    enum class FftMode
    {
        //! Replicated on small boxes, distributed when the padded box exceeds the per-rank limit.
        Auto,
        //! Always one whole FFT per rank (the historical path).
        Replicated,
        //! Always the distributed z-slab/pencil FFT; falls back to replicated with one rank.
        Distributed
    };

    //! Field-solve execution path.
    enum class SolveMode
    {
        //! Pick in-core when it fits the per-rank budget, else the chunked solve.
        Auto,
        //! Whole-grid FFT in host/device memory (the historical path).
        InCore,
        /*! Out-of-core chunked 3-D FFT with disk-backed transposes.
         *
         * One process splits the padded volume into chunks and processes one
         * chunk at a time.  The transform stays global (it is the same exact
         * 3-D transform); only the intermediate layouts are staged on disk.
         */
        Chunked
    };

    struct ComputeOptions
    {
        /*! Relative accuracy target of the delivered free-space convolution.
         *
         * The target refers to the discrete free-space TSC convolution, not to
         * the continuum Coulomb field near the source.  It selects the FFT
         * padding (see ``makeGrid``): padding 2 for targets that are at or above
         * the measured padding-2 error bound, padding 3 for tighter targets.
         * A target below the measured double-precision floor is rejected.
         */
        double tolerance = 1e-3;
        //! Physical PIC cell size (m). Node spacing is ``kRefinement * cellSize``.
        Vec3 cellSize{1.0, 1.0, 1.0};
        /*! Diagnostic override of the FFT zero-padding factor; 0 selects the
         * tolerance policy.  A non-zero value bypasses the policy.  The factor
         * is an oversampling multiplier (``M = fftPadding * n``), not a halo
         * count; ``makeGrid`` appends the deposit-halo deficit separately, so
         * ``fftPadding = 1`` keeps two halo nodes for CIC/TSC and remains a
         * valid no-oversampling choice.
         */
        unsigned fftPadding = 0;
        //! Empty shell margin around the particle bounding box, in PIC cells.
        double marginCells = 4.0;
        /*! Per-axis override of ``marginCells`` (PIC cells), used by the tolerated
         * global-domain writer.  The margin is measured from the corresponding
         * bounding-box face, so anisotropic boxes do not pay the largest axis'
         * margin on every axis.  Unset keeps the scalar ``marginCells``.
         */
        std::optional<Vec3> marginCellsPerAxis;
        /*! Output-grid overrides supplied by the input mesh.

         * The CLI fills ``meshGridSize`` from the mesh dataset shape (already
         * mapped to x,y,z by the reader) and ``meshGridOrigin`` from the mesh
         * ``gridGlobalOffset``.  They are defaults, not overrides: ``gridSize``
         * and ``gridOrigin`` still win when set.  Leaving both unset keeps the
         * bunch-bounding-box derivation.
         */
        std::optional<Idx3> meshGridSize;
        std::optional<Vec3> meshGridOrigin;
        /*! Explicit output-grid overrides (metres / node counts).

         * Unset values keep the mesh default or, when there is no mesh, the
         * automatic derivation from the bunch bounding box.  When set they are
         * used verbatim: an explicit ``--grid`` or ``--origin`` never blends with
         * the mesh or derived value.  Grid node counts must be even and positive
         * so the R2C transform keeps its even last extent.
         */
        std::optional<Idx3> gridSize;
        std::optional<Vec3> gridOrigin;
        /*! Global output domain written by the chunked writer.
         *
         * The computation stays on the (small) output grid derived above; these
         * two values only describe the domain that is written.  The compute grid
         * is aligned to this domain so every computed node lands on a global
         * node and no interpolation is needed.  Outside the compute box the
         * monopole/dipole/quadrupole far tail is evaluated; zeros are written
         * only on the diagnostic truncation path (``--allow-field-truncation``).
         * When unset, the written domain is the compute grid itself (the v0.1
         * behaviour).
         */
        std::optional<Idx3> globalGridSize;
        std::optional<Vec3> globalGridOrigin;
        /*! Serial host TSC deposit, used to make runs bitwise reproducible.
         *
         * The CUDA/HIP deposit uses double atomicAdd, whose summation order is
         * not fixed, so two runs of the same input differ by ~1 ULP.  This
         * diagnostic mode accumulates in particle order on the host instead.  It
         * is slower and is not intended for production, but it lets a rank-count
         * comparison be a bitwise comparison.
         */
        bool deterministicDeposit = false;
        /*! Host-side rank context for the component-parallel mesh and the
         * tile-parallel near shell.  Null means a single rank.  The mesh keeps
         * one full FFT per rank (no domain-decomposed FFT); ranks own disjoint
         * field components and the six arrays are summed after the loop.
         */
        ParallelContext const* parallel = nullptr;
        /*! Rank-local device index of the compiled backend.
         *
         * Filled from the automatic rank-to-device mapping after the option
         * broadcast, so it is deliberately not part of the wire format or the
         * fingerprint.  Every rank keeps its own value.
         */
        unsigned deviceIndex = 0;
        //! Opt-in exact near shell.  The default stays the plain mesh.
        NearShellOptions nearShell;
        //! Velocity-bin policy for multi-velocity bunches (see
        //! ``VelocityBinning``).  ``Auto`` (default) sizes the boxes from the
        //! error model and ``tolerance``; ``Explicit`` runs a fixed count.
        VelocityBinningOptions velocityBinning;
        //! Velocity-moment spectral decomposition policy (see
        //! ``VelocityMomentsOptions``).  ``Off`` (default) keeps the binned path
        //! bit-for-bit; ``Auto``/``Explicit`` replace the per-box FFT chain with a
        //! one-inverse-transform moment sum for multi-velocity bunches.
        VelocityMomentsOptions velocityMoments;
        /*! Fixed Taylor expansion centre for the moment path.
         *
         * Set by the near shell so the global mesh and every per-tile
         * subtraction use the *same* truncated kernel operator.  Recomputing
         * the centre on a near subset would make the subtraction cancel a
         * different approximation and leave a residual inside the shell.
         * Unset means "derive from the bunch" (the ordinary moment path).
         */
        std::optional<Vec3> velocityMomentsCentre;
        //! Precomputed binning to use verbatim instead of deriving it from the
        //! bunch.  The near shell sets this on every per-tile local mesh so
        //! the subtraction cancels the global mesh bin for bin.
        std::optional<VelocityBinning> fixedBinning;
        //! Particle shape for the charge deposit (default TSC).
        Shape shape = Shape::TSC;
        //! FFT decomposition policy.  The replicated path stays the default for small boxes.
        FftMode fftMode = FftMode::Auto;
        /*! Padded-node count per rank above which ``Auto`` selects the distributed FFT.
         *
         * Defaults to the existing 2^28 padded-node guard, i.e. the replicated
         * path runs while one rank can still hold the historical work set.
         */
        std::size_t distributedFftMinNodes = std::size_t{1u} << 28;
        /*! Chunked solve selection.  ``InCore`` keeps every historical path
         *  bit-for-bit; ``Chunked`` requires a single process and a sink (or
         *  accumulates into the returned ``Field`` for the checkpoint writer).
         */
        SolveMode solveMode = SolveMode::InCore;
        /*! Sticky marker that ``solveMode`` is already final.
         *
         * ``false`` lets the compute entry point resolve ``Auto`` once.  The
         * particle-chunked pipeline sets it when it forces ``InCore`` so the
         * automatic policy can never fire inside that path.
         */
        bool solveModeResolved = false;
        //! Cells (not bytes) in the largest chunk buffer; must hold one x-y plane.
        std::size_t chunkCells = std::size_t{1u} << 22;
        /*! Explicit number of virtual ranks for the staged solve.  Zero
         * derives it from ``chunkCells``.  The staged path uses the same slab
         * split and index arithmetic as the MPI distributed path, so this is
         * the count a real multi-rank run would use.
         */
        int virtualRanks = 0;
        //! Empty selects ``<output-or-temp>/picbunch-staging-<pid>``.
        std::string stagingDirectory;
        //! Keep the staging files after the solve (diagnostics only).
        bool keepStaging = false;
        /*! Optional output sink.  When set, a chunked solve streams each output
         * z-slab here instead of returning a full ``Field``.  The sink is
         * called with the six components in order, each covering all planes
         * exactly once.  It is not used by the in-core paths.
         */
        FieldChunkSink chunkSink;
    };

    //! openPMD record names and component order: E then B.
    inline constexpr std::array<char const*, 6> kComponentNames{
        "E_x",
        "E_y",
        "E_z",
        "B_x",
        "B_y",
        "B_z"};

    //! Yee offsets in units of the PIC cell, same order as kComponentNames.
    //! Frozen [component][axis] table for the openPMD ``position`` attribute;
    //! the C array keeps the 6x3 storage order explicit.
    inline constexpr double kYeeOffset[6][3] = {
        {0.5, 0.0, 0.0},
        {0.0, 0.5, 0.0},
        {0.0, 0.0, 0.5},
        {0.0, 0.5, 0.5},
        {0.5, 0.0, 0.5},
        {0.5, 0.5, 0.0}};

    //! 1/(4 pi eps0) in N m^2/C^2.
    inline constexpr double kCoulombConstant = 8.9875517873681764e9;
    //! speed of light in m/s.
    inline constexpr double kSpeedOfLight = 299792458.0;
    //! elementary charge in C (magnitude).
    inline constexpr double kElementaryCharge = 1.602176634e-19;
    //! electron rest mass in kg.
    inline constexpr double kElectronMass = 9.1093837139e-31;

    /** Geometry and time metadata of an openPMD mesh iteration.
     *
     * All physical vectors are returned in ``x, y, z`` order regardless of the
     * axis-label order stored in the file.
     */
    struct MeshGeometry
    {
        Idx3 shape{0, 0, 0};       //!< nodes per x, y, z
        Vec3 gridSpacing{1.0, 1.0, 1.0}; //!< physical spacing per axis
        Vec3 gridGlobalOffset{0.0, 0.0, 0.0}; //!< physical offset per axis
        Vec3 gridUnitSI{1.0, 1.0, 1.0}; //!< raw gridUnitSI per axis
        std::array<std::string, 3> axisLabels{"x", "y", "z"};
        std::string geometry{"cartesian"};
        std::string dataOrder{"C"};
        double time = 0.0;
        double timeUnitSI = 1.0;
        std::uint64_t iteration = 0;
    };

    //! One particle iteration read from an openPMD series.
    struct BunchData
    {
        Bunch bunch;
        double time = 0.0;
        double timeUnitSI = 1.0;
        std::uint64_t iteration = 0;
    };

} // namespace picBunch
