#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Offline unit tests for the ``metrics_v2`` additions of
``compare_velocity_bins.py``.

The tests use only synthetic numpy arrays and a tiny synthetic fixture; they
need neither a C++ build nor a GPU.  Run with::

    python3 tools/velocity-bins/test_compare_metrics.py
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_velocity_bins as compare  # noqa: E402


def _ones(shape):
    return np.ones(shape, dtype=float)


def _cube_bunch(center, radius, extra=()):
    """A bunch whose particle min/max are exactly the cube corners.

    ``prepare_metrics_v2`` derives the bounding box from the particles, so the
    corners pin it to ``[center - R, center + R]`` independently of any random
    interior sample.
    """
    pts = list(extra)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                pts.append((center[0] + sx * radius,
                            center[1] + sy * radius,
                            center[2] + sz * radius))
    arr = np.asarray(pts, dtype=float)
    return {"x": arr[:, 0], "y": arr[:, 1], "z": arr[:, 2]}


#: The geometry fixed by the assignment: 32^3 cells of 0.25 um, cube centre
#: 4 um, R = 1 um.
N32 = (32, 32, 32)
ORIGIN32 = (0.0, 0.0, 0.0)
CELL32 = (2.5e-7, 2.5e-7, 2.5e-7)
CENTER32 = (4.0e-6, 4.0e-6, 4.0e-6)
R32 = 1.0e-6


def _geometry_32():
    bunch = _cube_bunch(CENTER32, R32)
    ctx = compare.prepare_metrics_v2(bunch, N32, ORIGIN32, CELL32)
    return bunch, ctx


class TestComponentErrorStats(unittest.TestCase):
    def test_exact_reference_has_zero_error(self):
        ref = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4) + 1.0
        mask = np.ones((2, 3, 4), dtype=bool)
        stats = compare.component_error_stats(ref.copy(), ref, mask)
        self.assertEqual(stats["n"], 24)
        self.assertEqual(stats["n_finite"], 24)
        self.assertEqual(stats["n_nonfinite"], 0)
        self.assertEqual(stats["max_rel"], 0.0)
        self.assertEqual(stats["max_abs"], 0.0)
        self.assertEqual(stats["p99_rel"], 0.0)
        self.assertEqual(stats["rms_rel"], 0.0)
        self.assertIsNotNone(stats["max_rel_node"])

    def test_scaled_field_has_uniform_relative_error(self):
        ref = _ones((2, 2, 2))
        mask = np.ones((2, 2, 2), dtype=bool)
        stats = compare.component_error_stats(1.5 * ref, ref, mask)
        for key in ("max_rel", "p99_rel", "rms_rel"):
            self.assertAlmostEqual(stats[key], 0.5, places=14)
        self.assertAlmostEqual(stats["max_abs"], 0.5, places=14)

    def test_single_perturbed_node_separates_max_from_p99(self):
        ref = _ones((6, 5, 4))
        mesh = ref.copy()
        mesh[0, 0, 0] = 2.0  # one node with relative error 1.0
        mask = np.ones((6, 5, 4), dtype=bool)
        stats = compare.component_error_stats(mesh, ref, mask)
        self.assertEqual(stats["max_rel"], 1.0)
        self.assertEqual(stats["max_rel_node"], [0, 0, 0])
        # 99th percentile of 119 zeros and one 1.0 is zero: the artifact is
        # visible as max >> p99 rather than being blamed on the bulk.
        self.assertEqual(stats["p99_rel"], 0.0)
        self.assertGreater(stats["max_rel"], stats["p99_rel"])

    def test_zero_reference_uses_eps_floor(self):
        ref = np.zeros((2, 2, 2))
        mask = np.ones((2, 2, 2), dtype=bool)
        stats = compare.component_error_stats(1e-31 * _ones((2, 2, 2)),
                                              ref, mask, eps=1e-30)
        self.assertAlmostEqual(stats["max_rel"], 0.1, places=14)
        stats_floor = compare.component_error_stats(1e-30 * _ones((2, 2, 2)),
                                                    ref, mask, eps=1e-30)
        self.assertAlmostEqual(stats_floor["max_rel"], 1.0, places=14)

    def test_empty_mask_returns_nulls(self):
        mask = np.zeros((2, 2, 2), dtype=bool)
        stats = compare.component_error_stats(_ones((2, 2, 2)),
                                              _ones((2, 2, 2)), mask)
        self.assertEqual(stats["n"], 0)
        for key in ("max_rel", "max_abs", "p99_rel", "rms_rel",
                    "max_rel_node", "max_abs_node"):
            self.assertIsNone(stats[key])

    def test_nan_and_inf_are_counted_and_excluded(self):
        ref = _ones((2, 5, 1))
        mesh = ref.copy()
        ref[0, 0, 0] = np.nan
        mesh[0, 1, 0] = np.inf
        mask = np.ones((2, 5, 1), dtype=bool)
        stats = compare.component_error_stats(mesh, ref, mask)
        self.assertEqual(stats["n"], 10)
        self.assertEqual(stats["n_nonfinite"], 2)
        self.assertEqual(stats["n_finite"], 8)
        # finite part is exactly equal, so the robust metrics stay clean
        for key in ("max_rel", "p99_rel", "rms_rel"):
            self.assertEqual(stats[key], 0.0)

    def test_all_nonfinite_returns_counts_and_nulls(self):
        ref = np.full((2, 2, 2), np.nan)
        mask = np.ones((2, 2, 2), dtype=bool)
        stats = compare.component_error_stats(_ones((2, 2, 2)), ref, mask)
        self.assertEqual(stats["n"], 8)
        self.assertEqual(stats["n_nonfinite"], 8)
        self.assertIsNone(stats["max_rel"])


class TestMasksAndGeometry(unittest.TestCase):
    def test_box_mask_is_inclusive(self):
        n = (4, 4, 4)
        origin = [0.0, 0.0, 0.0]
        cell = [1.0, 1.0, 1.0]
        # nodes 0..1 on x, 1..2 on y, 2..3 on z -> 8 nodes
        mask = compare.box_node_mask(n, origin, cell,
                                     [0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
        self.assertEqual(int(mask.sum()), 8)
        self.assertTrue(mask[0, 1, 2])
        self.assertTrue(mask[1, 2, 3])
        self.assertFalse(mask[2, 1, 2])
        self.assertFalse(mask[0, 0, 2])

    def test_bbox_comes_from_particles(self):
        bunch = {"x": np.array([-1.0, 2.0]),
                 "y": np.array([0.0, 1.0]),
                 "z": np.array([4.0, 4.0])}
        bbox = compare.bunch_bounding_box(bunch)
        self.assertEqual(bbox["min"], [-1.0, 0.0, 4.0])
        self.assertEqual(bbox["max"], [2.0, 1.0, 4.0])
        self.assertEqual(bbox["extent"], [3.0, 1.0, 0.0])
        self.assertEqual(bbox["R"], 1.5)

    def test_far_field_is_bbox_translated_by_two_R_on_x_only(self):
        bunch = {"x": np.array([0.0, 1e-7]),
                 "y": np.array([0.0, 1e-7]),
                 "z": np.array([0.0, 1e-7])}
        n = (8, 8, 8)
        origin = [-2e-7, -2e-7, -2e-7]
        cell = [1e-7, 1e-7, 1e-7]
        ctx = compare.prepare_metrics_v2(bunch, n, origin, cell)
        bbox = ctx["bbox"]
        offset = 2.0 * bbox["R"]
        self.assertAlmostEqual(ctx["far_field_offset"], 1e-7)
        self.assertEqual(ctx["far_field_offset_axis"], "x")
        self.assertEqual(ctx["far_field_offset_vector"], [1e-7, 0.0, 0.0])
        # x is translated, y and z are untouched
        self.assertAlmostEqual(ctx["far_field_min"][0], bbox["min"][0] + offset)
        self.assertAlmostEqual(ctx["far_field_max"][0], bbox["max"][0] + offset)
        for a in (1, 2):
            self.assertEqual(ctx["far_field_min"][a], bbox["min"][a])
            self.assertEqual(ctx["far_field_max"][a], bbox["max"][a])
        # same size on every axis: translating does not shrink the box
        for a in range(3):
            self.assertAlmostEqual(
                ctx["far_field_max"][a] - ctx["far_field_min"][a],
                bbox["extent"][a])
        self.assertEqual(ctx["in_bunch_index_bounds"],
                         {"lo": [2, 2, 2], "hi": [3, 3, 3]})
        self.assertEqual(ctx["far_field_index_bounds"],
                         {"lo": [3, 2, 2], "hi": [4, 3, 3]})
        self.assertFalse(ctx["far_field_clipped"])
        self.assertEqual(int(ctx["in_mask"].sum()), 8)
        self.assertEqual(int(ctx["far_mask"].sum()), 8)

    def test_32_cubed_geometry_ranges_and_margins(self):
        bunch, ctx = _geometry_32()
        for a in range(3):
            self.assertAlmostEqual(ctx["bbox"]["min"][a], 3e-6)
            self.assertAlmostEqual(ctx["bbox"]["max"][a], 5e-6)
        self.assertAlmostEqual(ctx["bbox"]["R"], 1e-6)
        self.assertAlmostEqual(ctx["far_field_offset"], 2e-6)
        # inclusive base-node ranges at h = 0.25 um
        self.assertEqual(ctx["in_bunch_index_bounds"],
                         {"lo": [12, 12, 12], "hi": [20, 20, 20]})
        self.assertEqual(ctx["far_field_index_bounds"],
                         {"lo": [20, 12, 12], "hi": [28, 20, 20]})
        self.assertFalse(ctx["far_field_clipped"])
        self.assertTrue(ctx["far_field_margin_ok"])
        # 9^3 nodes in the bunch box and 9^3 in the full-size offset box
        self.assertEqual(int(ctx["in_mask"].sum()), 729)
        self.assertEqual(int(ctx["far_mask"].sum()), 729)
        # at least two empty layers on every side of both boxes
        for bounds in (ctx["in_bunch_index_bounds"],
                       ctx["far_field_index_bounds"]):
            for a in range(3):
                self.assertGreaterEqual(bounds["lo"][a], 2)
                self.assertLessEqual(bounds["hi"][a], N32[a] - 1 - 2)
        self.assertEqual(ctx["far_field_margins"],
                         {"lo": [20, 12, 12], "hi": [3, 11, 11]})
        # (c) no bunch particle lies in the far-field interior: the far box
        # starts exactly at the bunch max-x plane, so one particle plane sits
        # on the shared boundary but nothing is strictly inside.
        ff_lo = ctx["far_field_min"]
        ff_hi = ctx["far_field_max"]
        tol = 1e-12 * CELL32[0]
        inside = ((bunch["x"] > ff_lo[0] + tol)
                  & (bunch["y"] >= ff_lo[1] - tol)
                  & (bunch["y"] <= ff_hi[1] + tol)
                  & (bunch["z"] >= ff_lo[2] - tol)
                  & (bunch["z"] <= ff_hi[2] + tol))
        self.assertEqual(int(np.count_nonzero(inside)), 0)
        self.assertLessEqual(float(np.max(bunch["x"])), ff_lo[0] + tol)
        # (d) TSC support (1.5 cells) of the furthest particle stays in domain
        support = 1.5 * CELL32[0]
        for axis, pos in enumerate(("x", "y", "z")):
            self.assertGreaterEqual(float(np.min(bunch[pos])) - support,
                                    ORIGIN32[axis] - tol)
            self.assertLessEqual(float(np.max(bunch[pos])) + support,
                                 ORIGIN32[axis] + N32[axis] * CELL32[axis] + tol)

    def test_domain_too_small_raises_required_extent(self):
        # The integration defect: 24 x 16 x 16 at the old cube centre.
        bunch = _cube_bunch((3e-6, 2e-6, 2e-6), 1e-6)
        with self.assertRaises(compare.FarFieldGeometryError) as cm:
            compare.prepare_metrics_v2(bunch, (24, 16, 16), [0.0] * 3,
                                       [2.5e-7] * 3)
        msg = str(cm.exception)
        self.assertIn("Required domain extent", msg)
        self.assertIn("27x15x15", msg)
        self.assertIn("x[16..24]", msg)
        # The same call is fine when clipping is requested explicitly.
        ctx = compare.prepare_metrics_v2(bunch, (24, 16, 16), [0.0] * 3,
                                         [2.5e-7] * 3,
                                         far_field_policy="clip")
        self.assertTrue(ctx["far_field_clipped"])

    def test_clip_policy_reports_clipping_and_pre_clip_bounds(self):
        bunch = _cube_bunch((3e-6, 2e-6, 2e-6), 1e-6)
        n = (24, 16, 16)
        ctx = compare.prepare_metrics_v2(bunch, n, [0.0] * 3, [2.5e-7] * 3,
                                         far_field_policy="clip")
        self.assertTrue(ctx["far_field_clipped"])
        self.assertFalse(ctx["far_field_margin_ok"])
        self.assertEqual(ctx["far_field_pre_clip_bounds"],
                         {"lo": [16, 4, 4], "hi": [24, 12, 12]})
        # clipped to the domain: x 16..23, y 4..12, z 4..12 -> 8 * 9 * 9
        self.assertEqual(ctx["far_field_index_bounds"],
                         {"lo": [16, 4, 4], "hi": [23, 12, 12]})
        self.assertEqual(int(ctx["far_mask"].sum()), 648)
        result = compare.compute_metrics_v2(
            np.zeros((6,) + n), np.ones((6,) + n), ctx)
        geom = result["geometry"]
        self.assertTrue(geom["far_field_clipped"])
        self.assertEqual(geom["far_field_policy"], "clip")
        self.assertEqual(geom["far_field_pre_clip_bounds"],
                         {"lo": [16, 4, 4], "hi": [24, 12, 12]})
        self.assertEqual(geom["far_field_offset_axis"], "x")
        for a, expected in enumerate([2e-6, 0.0, 0.0]):
            self.assertAlmostEqual(geom["far_field_offset_vector"][a], expected)
        # The in-bunch box is unchanged by the far-field policy.
        self.assertEqual(geom["in_bunch_index_bounds"],
                         {"lo": [8, 4, 4], "hi": [16, 12, 12]})
        self.assertEqual(result["in_bunch"]["n_nodes"], 729)
        self.assertEqual(result["far_field"]["n_nodes"], 648)

    def test_empty_shell_is_null_with_zero_nodes(self):
        # Bunch between node planes: both boxes contain no base node, but the
        # geometry is well inside a large domain, so no clipping is involved.
        bunch = {"x": np.array([2.1, 2.4]),
                 "y": np.array([2.1, 2.4]),
                 "z": np.array([2.1, 2.4])}
        ctx = compare.prepare_metrics_v2(bunch, (10, 10, 10), [0.0] * 3,
                                         [1.0] * 3)
        self.assertFalse(ctx["far_field_clipped"])
        self.assertIsNone(ctx["far_field_index_bounds"])
        self.assertEqual(int(ctx["far_mask"].sum()), 0)
        result = compare.compute_metrics_v2(
            np.zeros((6, 10, 10, 10)), np.ones((6, 10, 10, 10)), ctx)
        self.assertEqual(result["far_field"]["n_nodes"], 0)
        self.assertIsNone(result["far_field"]["worst_rel"])
        self.assertIsNone(
            result["far_field"]["per_component"]["Bz"]["max_rel"])

    def test_degenerate_single_particle_does_not_crash(self):
        bunch = {"x": np.array([0.0]),
                 "y": np.array([0.0]),
                 "z": np.array([0.0])}
        n = (3, 3, 3)
        # R = 0, so the far box is the same node; a 3-node domain is too tight
        # for the 2-cell margin, hence the explicit clip policy.
        ctx = compare.prepare_metrics_v2(bunch, n, [-1.0] * 3, [1.0] * 3,
                                         far_field_policy="clip")
        self.assertEqual(ctx["bbox"]["R"], 0.0)
        self.assertEqual(int(ctx["in_mask"].sum()), 1)  # node at origin
        self.assertEqual(int(ctx["far_mask"].sum()), 1)  # R = 0
        self.assertFalse(ctx["far_field_clipped"])  # inside, only margin-short
        self.assertFalse(ctx["far_field_margin_ok"])
        mesh = np.zeros((6, 3, 3, 3))
        result = compare.compute_metrics_v2(mesh, np.ones((6, 3, 3, 3)), ctx)
        self.assertEqual(result["schema"], compare.METRICS_V2_SCHEMA)
        self.assertEqual(result["in_bunch"]["n_nodes"], 1)

    def test_zero_extent_axis_does_not_crash(self):
        bunch = {"x": np.array([0.0, 1.0]),
                 "y": np.array([0.0, 1.0]),
                 "z": np.array([0.5, 0.5])}  # delta in z
        ctx = compare.prepare_metrics_v2(bunch, (3, 3, 3),
                                         [0.0] * 3, [1.0] * 3)
        self.assertEqual(ctx["bbox"]["extent"][2], 0.0)
        # no node sits exactly at z = 0.5 -> empty in-bunch box, no error
        self.assertEqual(int(ctx["in_mask"].sum()), 0)
        result = compare.compute_metrics_v2(
            np.zeros((6, 3, 3, 3)), np.ones((6, 3, 3, 3)), ctx)
        self.assertEqual(result["in_bunch"]["n_nodes"], 0)
        self.assertIsNone(result["in_bunch"]["per_component"]["Ex"]["max_rel"])

    def test_empty_far_field_inside_tiny_domain_is_reported_as_clipped(self):
        bunch = {"x": np.array([0.0, 1.0]),
                 "y": np.array([0.0, 1.0]),
                 "z": np.array([0.0, 1.0])}
        # far box [1, 2]^3 lies past the grid nodes 0 and 0.5
        ctx = compare.prepare_metrics_v2(bunch, (2, 2, 2),
                                         [0.0] * 3, [0.5] * 3,
                                         far_field_policy="clip")
        self.assertTrue(ctx["far_field_clipped"])
        self.assertEqual(int(ctx["far_mask"].sum()), 0)
        result = compare.compute_metrics_v2(
            np.zeros((6, 2, 2, 2)), np.ones((6, 2, 2, 2)), ctx)
        self.assertEqual(result["far_field"]["n_nodes"], 0)
        self.assertIsNone(result["far_field"]["worst_rel"])
        self.assertIsNone(
            result["far_field"]["per_component"]["Bz"]["max_rel"])
        self.assertTrue(result["geometry"]["far_field_clipped"])


class TestEndToEnd32(unittest.TestCase):
    """Mesh equal to the reference, then one perturbed far-field node."""

    def test_zero_error_then_far_field_only_perturbation(self):
        bunch, ctx = _geometry_32()
        n = N32
        print("32^3 in-bunch index bounds:", ctx["in_bunch_index_bounds"],
              "nodes:", int(ctx["in_mask"].sum()))
        print("32^3 far-field index bounds:", ctx["far_field_index_bounds"],
              "nodes:", int(ctx["far_mask"].sum()),
              "margins:", ctx["far_field_margins"])
        ref = np.ones((6,) + n, dtype=float)
        mesh = ref.copy()
        clean = compare.compute_metrics_v2(mesh, ref, ctx)
        self.assertEqual(clean["in_bunch"]["n_nodes"], 729)
        self.assertEqual(clean["far_field"]["n_nodes"], 729)
        self.assertEqual(clean["in_bunch"]["worst_rel"], 0.0)
        self.assertEqual(clean["far_field"]["worst_rel"], 0.0)

        node = (24, 16, 16)
        self.assertTrue(ctx["far_mask"][node])
        self.assertFalse(ctx["in_mask"][node])
        perturbed = mesh.copy()
        perturbed[:, node[0], node[1], node[2]] = 3.0
        moved = compare.compute_metrics_v2(perturbed, ref, ctx)
        self.assertAlmostEqual(moved["far_field"]["worst_rel"], 2.0)
        self.assertEqual(moved["far_field"]["per_component"]["Ex"]["max_rel_node"],
                         list(node))
        self.assertEqual(moved["in_bunch"]["worst_rel"], 0.0)


def _write_fixture(tmpdir):
    """Tiny synthetic bunch/field fixture for the backward-compatibility test."""
    prefix = os.path.join(tmpdir, "case")
    n = (4, 4, 4)
    origin = [-2e-7, -2e-7, -2e-7]
    cell = [1e-7, 1e-7, 1e-7]
    particles = [
        (0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 1.0, -1.0, 0),
        (1e-7, 1e-7, 1e-7, 0.2, 0.1, 0.0, 1.0, -1.0, 1),
    ]
    with open(f"{prefix}_bunch.txt", "w") as f:
        f.write("# x y z ux uy uz w q box\n")
        for row in particles:
            f.write(" ".join(repr(float(v)) if i < 8 else str(int(v))
                             for i, v in enumerate(row)) + "\n")
    meta = {
        "grid": "4 4 4",
        "origin": " ".join(repr(v) for v in origin),
        "cell": " ".join(repr(v) for v in cell),
        "gamma": "1.0",
        "energy_spread": "0.01",
        "angle_degrees": "0.0",
        "bins_used": "2",
        "model_sum": "1e-12",
        "model_error": "1e-4",
        "per_axis": "2 2 2",
        "representative_u": "0.15 0.05 0.0 0.15 0.05 0.0",
        "box_of": "0 1",
    }
    with open(f"{prefix}_meta_B2.txt", "w") as f:
        for key, value in meta.items():
            f.write(f"{key} {value}\n")
    size = 6 * n[0] * n[1] * n[2]
    data = (np.arange(size, dtype=np.float64) + 1.0) * 1e-15
    data.tofile(f"{prefix}_field_B2.bin")
    return prefix


def _run_main(prefix, extra):
    argv = ["compare_velocity_bins.py", prefix, "--bins", "2",
            "--workers", "1", "--order", "2", "--refine", "0",
            "--bands", "0-1,1-2,16+"] + extra
    with mock.patch.object(sys, "argv", argv):
        compare.main()
    with open(f"{prefix}_compare.json") as f:
        return json.load(f)


def _legacy_canonical(payload):
    """bunch + runs with metrics_v2 removed, canonical JSON bytes."""
    runs = {key: {k: v for k, v in run.items() if k != "metrics_v2"}
            for key, run in payload["runs"].items()}
    legacy = {"bunch": payload["bunch"], "runs": runs}
    return json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode()


#: SHA-256 of the legacy JSON produced by the pre-change script (bfb2a04) on
#: the fixture above with ``--order 2 --refine 0 --workers 1``.
FROZEN_LEGACY_DIGEST = \
    "1889ff98c38a19b5622080f37a05b8f581f62de840a62ce27f0c4df97dd3bfbd"

#: A few exact pre-change values, kept for readable failures.
FROZEN_LEGACY_SCALARS = {
    "mesh_vs_T": {"window_nodes": 64, "worst_rel": 1.0,
                  "worst_abs": 1.6384065634418365e24},
    "Tb_vs_T": {"window_nodes": 64, "worst_rel": 8.74755061732023,
                "worst_abs": 9.890703141254724e21},
    "mesh_vs_Tb": {"window_nodes": 64, "worst_rel": 1.0,
                   "worst_abs": 1.6304741298157554e24},
}


class TestBackwardCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.prefix = _write_fixture(cls.tmp.name)
        cls.json_off = _run_main(cls.prefix, ["--no-metrics-v2"])
        # The tiny fixture is deliberately too small for the offset shell, so
        # the metrics-on run asks for clipping; the legacy keys are unaffected.
        cls.json_on = _run_main(cls.prefix, ["--far-field-policy", "clip"])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_no_metrics_flag_omits_key(self):
        self.assertNotIn("metrics_v2", self.json_off["runs"]["2"])

    def test_default_emits_metrics_v2(self):
        metrics = self.json_on["runs"]["2"]["metrics_v2"]
        self.assertEqual(metrics["schema"], compare.METRICS_V2_SCHEMA)
        self.assertEqual(metrics["comparison"], "mesh_vs_T")
        self.assertEqual(metrics["eps"], 1e-30)
        self.assertIn("in_bunch", metrics)
        self.assertIn("far_field", metrics)
        geom = metrics["geometry"]
        self.assertEqual(geom["far_field_offset_axis"], "x")
        self.assertTrue(geom["far_field_clipped"])
        self.assertIn("far_field_pre_clip_bounds", geom)

    def test_legacy_keys_identical_on_and_off(self):
        legacy_off = json.loads(_legacy_canonical(self.json_off))
        legacy_on = json.loads(_legacy_canonical(self.json_on))
        self.assertEqual(legacy_on, legacy_off)
        self.assertEqual(self.json_on["bunch"], self.json_off["bunch"])

    def test_legacy_payload_matches_frozen_baseline(self):
        digest = hashlib.sha256(_legacy_canonical(self.json_off)).hexdigest()
        self.assertEqual(digest, FROZEN_LEGACY_DIGEST)

    def test_frozen_scalars(self):
        run = self.json_off["runs"]["2"]
        self.assertEqual(run["bins_used"], 2)
        self.assertEqual(run["per_axis"], ["2", "2", "2"])
        for key in ("mesh_vs_T", "Tb_vs_T", "mesh_vs_Tb"):
            stats = run[key]
            self.assertIn("window_nodes", stats)
            self.assertIn("bands", stats)
            self.assertIn("dominant_band", stats)
            self.assertIn("rel", stats)
            self.assertIn("abs", stats)
            self.assertIn("worst_rel", stats)
            self.assertIn("worst_abs", stats)
        self.assertEqual(run["mesh_vs_T"]["rel"],
                         self.json_on["runs"]["2"]["mesh_vs_T"]["rel"])
        for key, expected in FROZEN_LEGACY_SCALARS.items():
            for field, value in expected.items():
                self.assertEqual(run[key][field], value,
                                 msg=f"{key}.{field} moved versus bfb2a04")

    def test_default_policy_fails_loudly_on_small_domain(self):
        argv = ["compare_velocity_bins.py", self.prefix, "--bins", "2",
                "--workers", "1", "--order", "2", "--refine", "0"]
        stderr = io.StringIO()
        with mock.patch.object(sys, "argv", argv), \
                contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as cm:
                compare.main()
        self.assertNotEqual(cm.exception.code, 0)
        msg = stderr.getvalue()
        self.assertIn("Required domain extent", msg)
        self.assertIn("far-field", msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
