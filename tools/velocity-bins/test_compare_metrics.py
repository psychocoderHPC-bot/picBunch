#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Offline unit tests for the ``metrics_v2`` additions of
``compare_velocity_bins.py``.

The tests use only synthetic numpy arrays and a tiny synthetic fixture; they
need neither a C++ build nor a GPU.  Run with::

    python3 tools/velocity-bins/test_compare_metrics.py
"""

from __future__ import annotations

import hashlib
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

    def test_far_field_is_bbox_translated_by_two_R(self):
        bunch = {"x": np.array([0.0, 1e-7]),
                 "y": np.array([0.0, 1e-7]),
                 "z": np.array([0.0, 1e-7])}
        n = (4, 4, 4)
        origin = [-2e-7, -2e-7, -2e-7]
        cell = [1e-7, 1e-7, 1e-7]
        ctx = compare.prepare_metrics_v2(bunch, n, origin, cell)
        self.assertAlmostEqual(ctx["far_field_offset"], 1e-7)
        for a in range(3):
            self.assertAlmostEqual(ctx["far_field_min"][a],
                                   ctx["bbox"]["min"][a] + 1e-7)
        # bbox [0, 1e-7] includes the base nodes at 0 and 1e-7; the far box
        # [1e-7, 2e-7] includes only the node at 1e-7 (per axis).
        self.assertEqual(int(ctx["in_mask"].sum()), 8)
        self.assertEqual(int(ctx["far_mask"].sum()), 1)

    def test_degenerate_single_particle_does_not_crash(self):
        bunch = {"x": np.array([0.0]),
                 "y": np.array([0.0]),
                 "z": np.array([0.0])}
        n = (3, 3, 3)
        ctx = compare.prepare_metrics_v2(bunch, n, [-1.0] * 3, [1.0] * 3)
        self.assertEqual(ctx["bbox"]["R"], 0.0)
        self.assertEqual(int(ctx["in_mask"].sum()), 1)  # node at origin
        self.assertEqual(int(ctx["far_mask"].sum()), 1)  # R = 0
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

    def test_empty_far_field_is_null(self):
        bunch = {"x": np.array([0.0, 1.0]),
                 "y": np.array([0.0, 1.0]),
                 "z": np.array([0.0, 1.0])}
        # far box [1, 2]^3 is outside the grid nodes 0 and 0.5
        ctx = compare.prepare_metrics_v2(bunch, (2, 2, 2),
                                         [0.0] * 3, [0.5] * 3)
        self.assertEqual(int(ctx["far_mask"].sum()), 0)
        result = compare.compute_metrics_v2(
            np.zeros((6, 2, 2, 2)), np.ones((6, 2, 2, 2)), ctx)
        self.assertEqual(result["far_field"]["n_nodes"], 0)
        self.assertIsNone(result["far_field"]["worst_rel"])
        self.assertIsNone(
            result["far_field"]["per_component"]["Bz"]["max_rel"])


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
        cls.json_on = _run_main(cls.prefix, [])

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
