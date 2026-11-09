#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
"""Regenerate the fast-near 52-point exact-field reference set.

The fast-near validation in the paper samples three configurations on the same
geometric point set: 20 points inside the TSC support (|s| <= 1.35 cells), 16
points on a support face (one coordinate at +/-1.5 cells), and 16 points outside
the support (one coordinate at +/-[1.6, 3.6] cells).  Each configuration thus
has 52 points, and every point carries all six field components.

Point definition (copied from the original validation driver):

* ``numpy.random.default_rng(12345)``;
* interior: ``uniform(-1.35, 1.35, 3)``;
* face: ``uniform(-1.2, 1.2, 3)``, then one random axis set to +/-1.5;
* outside: ``uniform(-1.2, 1.2, 3)``, then one random axis set to
  +/-``(1.6 + uniform(0, 2))``;
* the RNG is consumed in the fixed order g1, g10-axial, g10-oblique, and within
  each case interior, face, outside.

The reference is the CPU-only, host-independent TSC-convolved continuum field
``tools/velocity-bins/indfield.py`` at order 12, refine 2, with unit cells and
the source at the origin.  Values are the normalized per-unit-charge fields
(``E_n = 4*pi*eps0*E/q``, ``B_n = 4*pi*eps0*c*B/q``).  No picBunch executable
and no GPU are involved.

Configurations:
  g1          u = (0, 0, 0)
  g10-axial   u = (0, 0, sqrt(99))
  g10-oblique u = sqrt(99) * (1, 2, 1) / sqrt(6)

Output is a stable CSV (rows sorted by case, region, point index) so repeated
runs are byte-identical.

Examples::

    tools/verify/gen_fastnear_reference.py
    tools/verify/gen_fastnear_reference.py --out /path/to/ref.csv  # custom location
    tools/verify/gen_fastnear_reference.py --self-test
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

CASES = (
    ("g1", np.zeros(3)),
    ("g10-axial", np.array([0.0, 0.0, np.sqrt(99.0)])),
    ("g10-oblique", np.sqrt(99.0) * np.array([1.0, 2.0, 1.0]) / np.sqrt(6.0)),
)
REGIONS = ("interior", "face", "outside")
COUNTS = {"interior": 20, "face": 16, "outside": 16}
ORDER = 12
REFINE = 2
CELLS = np.ones(3)
SEED = 12345

HEADER = ",".join(
    ["case", "region", "index", "x_cells", "y_cells", "z_cells",
     "Ex_norm", "Ey_norm", "Ez_norm", "Bx_norm", "By_norm", "Bz_norm"])


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))


def load_indfield(path: str | None):
    """Import the repository's independent exact-field module."""
    if path is None:
        path = os.path.join(_repo_root(), "tools", "velocity-bins")
    sys.path.insert(0, os.path.abspath(path))
    import indfield  # noqa: E402
    return indfield


def sample_points(rng):
    """Yield (case, region, index, point_cells) in the documented order."""
    for case, _ in CASES:
        for region in REGIONS:
            for index in range(COUNTS[region]):
                if region == "interior":
                    point = rng.uniform(-1.35, 1.35, 3)
                elif region == "face":
                    point = rng.uniform(-1.2, 1.2, 3)
                    point[rng.integers(3)] = rng.choice([-1.5, 1.5])
                else:
                    point = rng.uniform(-1.2, 1.2, 3)
                    axis = rng.integers(3)
                    point[axis] = rng.choice([-1.0, 1.0]) * (
                        1.6 + rng.uniform(0.0, 2.0))
                yield case, region, index, point


def reference_values(indfield, u, point_cells):
    """Six normalized field components at one point in cell units."""
    e, b = indfield.integrate_point(
        np.asarray(point_cells, dtype=float) * CELLS,
        np.zeros(3),
        np.asarray(u, dtype=float),
        CELLS[0], CELLS[1], CELLS[2],
        order=ORDER, refine=REFINE, chunk=8)
    return np.concatenate([np.asarray(e, dtype=float),
                           np.asarray(b, dtype=float)])


def generate(indfield):
    """Return rows as tuples of (case, region, index, point, values)."""
    rng = np.random.default_rng(SEED)
    rows = []
    by_case = {case: u for case, u in CASES}
    for case, region, index, point in sample_points(rng):
        values = reference_values(indfield, by_case[case], point)
        rows.append((case, region, index, np.asarray(point, dtype=float),
                     values))
    return rows


def format_rows(rows):
    lines = [HEADER]
    for case, region, index, point, values in rows:
        fields = [case, region, str(index)]
        fields += [format(float(v), ".17g") for v in point]
        fields += [format(float(v), ".17g") for v in values]
        lines.append(",".join(fields))
    return "\n".join(lines) + "\n"


def generate_text(indfield):
    rows = generate(indfield)
    counts = {}
    for case, region, _, _, _ in rows:
        counts[(case, region)] = counts.get((case, region), 0) + 1
    for case, _ in CASES:
        for region in REGIONS:
            expected = COUNTS[region]
            actual = counts.get((case, region), 0)
            if actual != expected:
                raise SystemExit(
                    f"{case}/{region}: got {actual} points, expected {expected}")
    return format_rows(rows)


def check_file(indfield, path):
    """Recompute every stored row and require an exact, bitwise match."""
    import csv
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != sum(COUNTS.values()) * len(CASES):
        print(f"FAIL: {len(rows)} rows, expected "
              f"{sum(COUNTS.values()) * len(CASES)}", file=sys.stderr)
        return 1
    for row in rows:
        stored = np.array([float(row[name]) for name in (
            "Ex_norm", "Ey_norm", "Ez_norm", "Bx_norm", "By_norm", "Bz_norm")])
        point = np.array([float(row[name]) for name in (
            "x_cells", "y_cells", "z_cells")])
        again = reference_values(
            indfield, dict(CASES)[row["case"]], point)
        if not np.array_equal(stored, again):
            print(f"FAIL: {row['case']}/{row['region']}/{row['index']} "
                  "does not reproduce", file=sys.stderr)
            return 1
    print(f"check: PASS ({len(rows)} stored values reproduce exactly; "
          "reference against itself = 0)")
    return 0


def self_test(indfield):
    """Compute the set twice and confirm exact self-consistency."""
    first = generate_text(indfield)
    second = generate_text(indfield)
    if first != second:
        print("FAIL: two in-process generations differ", file=sys.stderr)
        return 1
    rows = generate(indfield)
    for case, region, index, point, values in rows:
        again = reference_values(
            indfield, dict(CASES)[case], point)
        if not np.array_equal(values, again):
            print(f"FAIL: {case}/{region}/{index} is not self-consistent",
                  file=sys.stderr)
            return 1
    print(f"self-test: PASS ({len(rows)} points, reference against itself = 0)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate the 52-point fast-near exact-field reference "
                    "set (20 interior, 16 face, 16 outside per case).")
    parser.add_argument("--out", default=None,
                        help="output CSV (default: "
                             "docs/measurements/fastnear_reference_52pt.csv)")
    parser.add_argument("--indfield-dir", default=None,
                        help="directory containing indfield.py "
                             "(default: tools/velocity-bins)")
    parser.add_argument("--self-test", action="store_true",
                        help="generate twice in memory and check self-consistency")
    parser.add_argument("--check", default=None, metavar="CSV",
                        help="recompute an existing CSV and require an exact match")
    args = parser.parse_args()

    indfield = load_indfield(args.indfield_dir)
    if args.self_test:
        return self_test(indfield)
    if args.check:
        return check_file(indfield, args.check)

    out = args.out
    if out is None:
        out = os.path.join(_repo_root(), "docs", "measurements",
                           "fastnear_reference_52pt.csv")
    text = generate_text(indfield)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print(f"wrote {out} ({len(text.splitlines()) - 1} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
