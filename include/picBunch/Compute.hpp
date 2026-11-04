// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/Types.hpp"

#include <array>
#include <optional>

namespace picBunch
{
    /** Resolve the physical cell size from an explicit command-line value and a mesh-derived value.

     * An explicit value always wins over the mesh-derived one.  Both are in metres;
     * nothing is inferred or rescaled.  Throws ``std::invalid_argument`` when the
     * explicit value is not finite and positive, and ``std::runtime_error`` when
     * neither source is available.
     */
    std::array<double, 3> resolveCellSize(
        std::optional<std::array<double, 3>> const& explicitCellSize,
        std::optional<std::array<double, 3>> const& meshCellSize);

    /** Physical cell size from mesh metadata, or ``std::nullopt`` when unusable.
     *
     * ``readMeshGeometry`` already returns ``gridSpacing`` and
     * ``gridGlobalOffset`` in physical SI units.  This helper therefore copies
     * the spacing verbatim and applies ``gridUnitSI`` nowhere; applying it a
     * second time in the CLI was the v0.1 regression.  It rejects non-finite or
     * non-positive spacing so the caller can fall back to ``--cell-size``.
     */
    std::optional<std::array<double, 3>> physicalCellSize(picBunch::MeshGeometry const& geometry);

    /** Physical grid origin from mesh metadata, or ``std::nullopt`` when unusable.
     *
     * ``readMeshGeometry`` already converted ``gridGlobalOffset`` to physical SI
     * metres, exactly as it did for ``gridSpacing``.  This helper returns the
     * offset verbatim and applies ``gridUnitSI`` nowhere.  Non-finite offsets are
     * rejected so the caller can fall back to the bunch bounding box.
     */
    std::optional<std::array<double, 3>> physicalGridOrigin(picBunch::MeshGeometry const& geometry);

    /** Output-grid extent from the mesh dataset shape, or ``std::nullopt`` when unusable.
     *
     * ``readMeshGeometry`` returns ``shape`` already mapped from the stored axis
     * order to ``x, y, z`` nodes and applies no ``gridUnitSI`` to a node count.
     * This helper copies the shape verbatim, so the conversion stays owned by
     * the reader.  A zero extent is rejected.
     */
    std::optional<std::array<std::size_t, 3>> physicalGridExtent(picBunch::MeshGeometry const& geometry);

    /** Tolerance-derived field tail outside a compact source region.
     *
     * The field of a finite bunch does not vanish outside its bounding box.  At
     * a distance ``d`` from the box centre the monopole part scales as
     * ``|Q| / (4 pi eps0 d^2)`` and the dipole part as ``2 |p| / (4 pi eps0 d^3)``.
     * ``deriveFieldTailMargin`` solves
     *
     *   E_tail(d) = gamma_max k [ |Q|/d^2 + 2|p|/d^3 + 24 Q_abs R^2/(d-R)^4 ]
     *              <= tolerance * fieldPeak
     *
     * for the smallest ``d`` larger than the source radius ``R``.  The last term
     * is a conservative Taylor remainder that bounds every multipole above the
     * dipole.  ``gamma_max`` is the largest ``sqrt(1+|u|^2)`` of the bunch; a
     * uniformly moving charge field is the boosted Coulomb field, whose
     * transverse magnitude is at most ``gamma`` times the rest monopole, so the
     * factor keeps the bound valid for a moving bunch.  ``marginCells`` converts ``d`` into extra cells beyond each
     * bounding-box face; every point outside that expanded box is at least
     * ``distance`` from the box centre, hence its neglected field is within the
     * target.
     *
     * ``fieldPeak`` must be the peak of the computed field in the same units as
     * the bound.  The CLI uses ``min(max|E|, c max|B|)`` (ignoring a zero B
     * peak), because ``|B_tail| <= |E_tail|/c``; this keeps both records at or
     * below their own relative tolerance.  A zero peak, zero charge and zero
     * dipole return a zero margin.  A target that cannot be reached within ``maxDistance``
     * throws ``std::runtime_error``; the caller must then enlarge the compute
     * box or explicitly acknowledge the truncation.
     */
    struct FieldTailMargin
    {
        double totalCharge = 0.0;
        double absoluteCharge = 0.0;
        double dipoleMagnitude = 0.0;
        double sourceRadius = 0.0;
        double maxGamma = 1.0;
        double fieldPeak = 0.0;
        double tolerance = 0.0;
        //! Smallest centre distance satisfying the bound (metres).
        double distance = 0.0;
        //! Value of the bound at ``distance`` (same units as ``fieldPeak``).
        double bound = 0.0;
        //! Extra cells beyond each bounding-box face.
        std::array<double, 3> marginCells{0.0, 0.0, 0.0};
    };

    FieldTailMargin deriveFieldTailMargin(
        Bunch const& bunch,
        double fieldPeak,
        double tolerance,
        std::array<double, 3> const& cellSize,
        double maxDistance = 1.0e30);

    /** Build the output grid.
     *
     * When ``options.gridSize`` is set it wins.  Otherwise
     * ``options.meshGridSize`` (the input mesh extent) wins.  Without either,
     * ``n`` is derived from the bunch bounding box plus ``marginCells``.  The
     * node spacing is ``kRefinement * options.cellSize``; the origin is
     * ``options.gridOrigin`` when set, otherwise ``options.meshGridOrigin``,
     * otherwise the derived box minimum.
     *
     * Any particle outside the final grid's physical extent
     * ``[origin, origin + n * spacing]`` is rejected with a message naming the
     * count and the bounds; the TSC deposit would otherwise drop it silently.
     *
     * The padding factor comes from the tolerance policy (padding 2 for
     * ``tolerance >= 1e-12``, padding 3 below that down to the measured
     * double-precision floor ``1e-14``) unless ``options.fftPadding`` is set
     * explicitly as a diagnostic override.  A tolerance below the floor is
     * rejected with ``std::runtime_error``.
     */
    Grid makeGrid(Bunch const& bunch, ComputeOptions const& options);

    /** Full field pipeline: TSC deposit -> moving point kernel -> free-space FFT
     *  convolution -> Yee-staggered read-out.
     *
     * Selects the device of the compiled backend (CUDA, HIP, or host CPU), runs
     * every precision-critical stage in double precision, and returns the six
     * components in SI units (V/m and T) on ``makeGrid(bunch, options)``.
     *
     * ``tolerance`` is the only accuracy knob exposed to users.  It selects the
     * FFT zero padding, the only discretisation choice not fixed by the input
     * geometry.  It does not change the TSC deposit order or ``kRefinement``, so
     * near-source accuracy stays a stated limitation of v0.1.  The contract is
     * written down in ``docs/algorithm.md``.
     *
     * ``stageTimes``, when non-null, receives a wall-clock breakdown.  Every
     * boundary waits for the device, so the numbers are comparable across backends.
     */
    Field computeField(Bunch const& bunch, ComputeOptions const& options = {}, StageTimes* stageTimes = nullptr);

} // namespace picBunch
