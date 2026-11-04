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
