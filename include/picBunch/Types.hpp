// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/Parallel.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace picBunch
{
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
     * free-space convolution: the FFT grid has ``M = fftPadding * n`` nodes per axis.
     * It is normally selected from ``ComputeOptions::tolerance`` (never below 2);
     * a value of 1 disables zero padding and is a test/diagnostic override only.
     */
    struct Grid
    {
        std::array<std::size_t, 3> n{48, 48, 48};
        std::array<double, 3> spacing{1.0, 1.0, 1.0}; //!< physical node spacing
        std::array<double, 3> origin{0.0, 0.0, 0.0}; //!< physical position of node (0,0,0)
        unsigned fftPadding{2};
        //! Deposit shape; the near shell currently supports TSC only.
        Shape shape = Shape::TSC;

        //! Total number of output nodes.
        [[nodiscard]] std::size_t size() const
        {
            return n[0] * n[1] * n[2];
        }
        //! Number of nodes per axis of the zero-padded FFT box.
        [[nodiscard]] std::array<std::size_t, 3> paddedSize() const
        {
            return {n[0] * fftPadding, n[1] * fftPadding, n[2] * fftPadding};
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
        [[nodiscard]] std::array<std::array<double, 3>, 2> bounds() const;
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

    /*! One particle patch region of a generated species.
     *
     * ``offset`` is the lower cell index relative to the PIConGPU domain
     * origin and ``extent`` the number of cells covered.  Regions may overlap
     * by one cell; particle ownership is disjoint and is carried separately by
     * the cumulative ``numParticlesOffset`` written next to these arrays.
     */
    struct ParticlePatch
    {
        std::array<std::int64_t, 3> offset{0, 0, 0};
        std::array<std::uint64_t, 3> extent{1, 1, 1};
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
        std::size_t tileNodes = 4;
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

    /*! Assumed constant of the velocity-bin error model.
     *
     * The model (see ``VelocityBinning``) estimates the relative error of the
     * binned mesh field as ``modelConstant * modelSum``.  The shipped value is
     * an assumption, not a refitted calibration: the original 1, 2, 4, 8 and
     * 16 box rows and their model sums from run ``pr24-20261102`` (host
     * jureca-h100, CUDA 13.4) were not archived in the repository.  The
     * recorded 16- and 64-box accuracy points are in
     * ``docs/measurements/velocity_bin_calibration.csv``; the 5% energy-spread
     * case measures 1.12e-3 at 16 boxes and 2.72e-4 at 64 (meeting 1e-3 only
     * at 64), while the 5 degree angular-spread case does not reach 1e-3
     * (0.421 at 16 boxes).  Re-running the 1024-source 5% energy-spread case,
     * the node-bounded energy term gives 3.9e-3 at 16 boxes and 9.2e-4 at 64
     * (3.4x conservative against the archived points, down from 27x/26x) and
     * selects 64 boxes at a 1e-3 tolerance instead of failing above the cap;
     * the pure angular case is unchanged.  The CLI exposes
     * ``--bin-model-constant`` as a diagnostic override.
     */
    inline constexpr double kDefaultVelocityModelConstant = 1.0;

    /*! Upper bound of the auto binning policy (box count).
     *
     * Every box costs one deposit plus 13 FFTs on the padded grid, so the cap
     * is a cost guard, not an accuracy limit: the policy reports the box
     * count the model would need and fails when it exceeds the cap.
     */
    inline constexpr std::size_t kDefaultMaxVelocityBins = 256u;

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
     * pattern, of order ``gamma * Delta_theta`` in relative terms, and an
     * energy shift, of order ``(2/gamma) * Delta_gamma / gamma`` at a node.
     * With ``q_b`` the charge in box ``b``, ``Q`` the total absolute charge,
     * ``gamma_b`` the largest ``gamma`` in the box, ``Delta_theta_b`` the
     * largest angle between a member and the box representative, and
     * ``Delta_gamma_b`` the ``gamma`` range of the box, the estimated relative
     * error is
     *
     *   modelError = modelConstant * Sum_b (q_b / Q)
     *                 * (gamma_b*Delta_theta_b
     *                    + (2/gamma_b) * Delta_gamma_b/gamma_b)
     *
     * The angular term uses the measured worst-case angular-spread error
     * (``modelConstant = 1`` reproduces the archived 5 degree point); the
     * energy term is the node-bounded single term that replaces the former
     * ``Delta_gamma/gamma + maxGammaDev`` pair, which double-counted the same
     * effect and over-predicted the archived 5% energy-spread points by
     * ~26x.  This is a calibrated/conservative estimate, not a guarantee: the
     * energy term still over-predicts the two archived points by ~3.4x, and its
     * behaviour beyond gamma 10 and the measured box range is not validated.
     *
     * ``modelSum`` is the uncalibrated sum; ``modelError`` is the calibrated
     * estimate.  The auto policy chooses the smallest power-of-two box count
     * whose best axis split drives ``modelError`` below the requested
     * tolerance; an infeasible requirement fails with the needed count.
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
        std::array<std::size_t, 3> perAxis{1u, 1u, 1u};
        //! Orthonormal beam-frame basis, rows ``e1, e2, e3`` (lab frame).
        std::array<std::array<double, 3>, 3> basis{{{1.0, 0.0, 0.0},
                                                    {0.0, 1.0, 0.0},
                                                    {0.0, 0.0, 1.0}}};
        //! Box extent in beam-frame ``u`` units, axis ``i``: ``[uMin[i], uMax[i]]``.
        std::array<double, 3> uMin{0.0, 0.0, 0.0};
        std::array<double, 3> uMax{0.0, 0.0, 0.0};
        //! Representative ``u`` (lab frame) per box, box index order
        //! ``i1 + N1*(i2 + N2*i3)``.
        std::vector<std::array<double, 3>> representativeU;
        //! Box index of each particle; size equals the bunch size the binning
        //! was derived for.  A fixed binning applied to a sub-bunch must
        //! re-derive the assignment with ``boxIndex``.
        std::vector<std::size_t> boxOf;
        //! Uncalibrated model sum (dimensionless, charge-weighted).
        double modelSum = 0.0;
        //! Calibration constant used (``kDefaultVelocityModelConstant`` unless
        //! overridden).
        double modelConstant = kDefaultVelocityModelConstant;
        //! Calibrated estimate ``modelConstant * modelSum``.
        double modelError = 0.0;

        //! Box of a particle with beam-frame ``u`` coordinates, for re-using a
        //! fixed binning on a sub-bunch (the near shell tile mesh).
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
        std::array<double, 3> cellSize{1.0, 1.0, 1.0};
        /*! Diagnostic override of the FFT zero-padding factor; 0 selects the
         * tolerance policy.  A non-zero value (including 1 for the wraparound
         * test) bypasses the policy and must be >= 1.
         */
        unsigned fftPadding = 0;
        //! Empty shell margin around the particle bounding box, in PIC cells.
        double marginCells = 4.0;
        /*! Per-axis override of ``marginCells`` (PIC cells), used by the tolerated
         * global-domain writer.  The margin is measured from the corresponding
         * bounding-box face, so anisotropic boxes do not pay the largest axis'
         * margin on every axis.  Unset keeps the scalar ``marginCells``.
         */
        std::optional<std::array<double, 3>> marginCellsPerAxis;
        /*! Output-grid overrides supplied by the input mesh.

         * The CLI fills ``meshGridSize`` from the mesh dataset shape (already
         * mapped to x,y,z by the reader) and ``meshGridOrigin`` from the mesh
         * ``gridGlobalOffset``.  They are defaults, not overrides: ``gridSize``
         * and ``gridOrigin`` still win when set.  Leaving both unset keeps the
         * bunch-bounding-box derivation.
         */
        std::optional<std::array<std::size_t, 3>> meshGridSize;
        std::optional<std::array<double, 3>> meshGridOrigin;
        /*! Explicit output-grid overrides (metres / node counts).

         * Unset values keep the mesh default or, when there is no mesh, the
         * automatic derivation from the bunch bounding box.  When set they are
         * used verbatim: an explicit ``--grid`` or ``--origin`` never blends with
         * the mesh or derived value.  Grid node counts must be even and positive
         * so the R2C transform keeps its even last extent.
         */
        std::optional<std::array<std::size_t, 3>> gridSize;
        std::optional<std::array<double, 3>> gridOrigin;
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
        std::optional<std::array<std::size_t, 3>> globalGridSize;
        std::optional<std::array<double, 3>> globalGridOrigin;
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
        std::optional<std::array<double, 3>> velocityMomentsCentre;
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
        std::array<std::size_t, 3> shape{0, 0, 0};       //!< nodes per x, y, z
        std::array<double, 3> gridSpacing{1.0, 1.0, 1.0}; //!< physical spacing per axis
        std::array<double, 3> gridGlobalOffset{0.0, 0.0, 0.0}; //!< physical offset per axis
        std::array<double, 3> gridUnitSI{1.0, 1.0, 1.0}; //!< raw gridUnitSI per axis
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
