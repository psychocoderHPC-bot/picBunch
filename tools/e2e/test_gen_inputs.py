#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
"""Unit tests for the cost-table drawer in ``gen_inputs.py``.

The cost table was reconstructed with an inconsistent drawer: the script
offset the stored positions by ``(grid - 20) / 2`` *and* the reader multiplies
them by ``unitSI``, so under the real ``--origin`` the bunch landed at the far
grid cell corner.  These tests pin the fix:

* the particle centroid equals the grid centroid in physical units;
* the ``unitSI`` written to openPMD round-trips bit-exactly and the physical
  coordinate is ``position * unitSI`` (there is no ``positionOffset`` record);
* the code-faithful near-shell (source, node) pair count at R=1.5 matches the
  centred-bunch expectation (~4.32e8).

Run directly with ``python3 tools/e2e/test_gen_inputs.py`` (needs numpy and
h5py).
"""
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_inputs  # noqa: E402


def _tile_edges(n, tile):
    return [(i0, min(n, i0 + tile)) for i0 in range(0, n, tile)]


def _axis_coverage(s, n, tile, radius):
    """Node count of tiles whose box [i0, i1-1] reaches s within ``radius``.

    This is the code-faithful near-set rule from ``src/NearShell.cpp`` (tile
    edges start at 0, last tile may be smaller).
    """
    acc = np.zeros(s.shape, dtype=np.int64)
    for i0, i1 in _tile_edges(n, tile):
        inside = (i0 - radius <= s) & (s <= (i1 - 1) + radius)
        acc += np.where(inside, i1 - i0, 0)
    return acc


def pair_count(positions_cells, grid, tile, radius):
    """(source, node) pairs = sum over sources of per-axis node coverage."""
    cx = _axis_coverage(positions_cells[:, 0], grid[0], tile, radius)
    cy = _axis_coverage(positions_cells[:, 1], grid[1], tile, radius)
    cz = _axis_coverage(positions_cells[:, 2], grid[2], tile, radius)
    return int(np.sum(cx * cy * cz))


def read_positions(path):
    with h5py.File(path, "r") as f:
        rec = f["data/0/particles/e/position"]
        unit = rec["x"].attrs["unitSI"]
        pos = np.stack(
            [np.asarray(rec[ax], dtype=np.float64) for ax in ("x", "y", "z")],
            axis=1,
        )
    return pos, float(unit)


def main():
    cell = gen_inputs.COST_CELL_M
    grid = np.array(gen_inputs.COST_GRID, dtype=np.float64)
    origin = np.array(gen_inputs.COST_ORIGIN_M, dtype=np.float64)

    with tempfile.TemporaryDirectory() as tmp:
        path = gen_inputs.write_cost(Path(tmp), gen_inputs.COST_N)
        stored, unit = read_positions(path)

    # 1) unitSI round-trip is exact and physical coordinate = position*unitSI.
    assert unit == cell, f"unitSI {unit!r} != cell {cell!r}"
    physical = stored * unit
    assert np.array_equal(physical, stored * cell)

    # 2) Centroid equals the grid centroid in physical units.  The statistical
    #    error of the mean for COST_N uniforms is ~1e-9 m, well below 1e-8 m.
    grid_centroid = origin + 0.5 * grid * cell
    centroid = physical.mean(axis=0)
    assert np.allclose(centroid, grid_centroid, rtol=0.0, atol=1e-8), (
        f"centroid {centroid} != grid centre {grid_centroid}"
    )

    # The bunch must sit in the grid interior, once centred, not against a wall.
    grid_index = stored - origin / cell
    assert grid_index.min() > 0.0 and grid_index.max() < grid.max(), (
        f"bunch not interior: [{grid_index.min()}, {grid_index.max()}]"
    )
    # ... and be centred to within one cell in cell units.
    assert np.allclose(grid_index.mean(axis=0), 0.5 * grid, atol=1.0)

    # 3) Code-faithful pair count at R=1.5: the centred-bunch expectation is
    #    4.3200e8; the exact RNG drawer reproduces it to <0.5%.
    pairs = pair_count(grid_index, gen_inputs.COST_GRID, tile=4, radius=1.5)
    expected = 4.32e8
    assert abs(pairs - expected) <= 0.005 * expected, (
        f"R=1.5 pair count {pairs} not within 0.5% of {expected}"
    )

    print(
        f"test_gen_inputs: OK  centroid={centroid}  "
        f"cell_centroid={grid_index.mean(axis=0)}  pairs(R=1.5)={pairs}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
