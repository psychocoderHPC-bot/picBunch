#!/usr/bin/env python3
"""Prove the ingestion gate with synthetic campaign outputs in temp dirs.

The three cases required by the task:

* a complete synthetic source set ingests and passes the strict provenance gate;
* one absent source fails loudly and names the case, leaving its legacy target
  untouched;
* no synthetic data is written outside the temporary directories (the checked-in
  ``docs/paper/data`` and ``docs/paper/provenance/records`` are byte-identical
  before and after).

Run::

    python3 docs/paper/scripts/test_ingest_measurements.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(os.path.dirname(PAPER_DIR))
sys.path.insert(0, HERE)
import axis  # noqa: E402
import cases  # noqa: E402
import check_provenance  # noqa: E402
import ingest_measurements as ingest  # noqa: E402

MAP = os.path.join(HERE, "measurement_map.json")
CHECK = os.path.join(HERE, "check_provenance.py")


def tree_hash(path):
    digest = hashlib.sha256()
    if not os.path.isdir(path):
        return digest.hexdigest()
    for base, dirs, files in sorted(os.walk(path)):
        dirs.sort()
        for name in sorted(files):
            full = os.path.join(base, name)
            digest.update(os.path.relpath(full, path).encode())
            with open(full, "rb") as fh:
                digest.update(fh.read())
    return digest.hexdigest()


def _write_csv(path, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        fh.write(",".join(columns) + "\n")
        for row in (1, 2):
            values = []
            for column in columns:
                if column in ("kind", "source", "host", "gpu_model"):
                    values.append("synthetic")
                elif column in ("case_id",):
                    values.append("h100-mpi-scaling")
                else:
                    values.append(str(row + len(column) % 3))
            fh.write(",".join(values) + "\n")


def _field_source(path, cfg):
    style = cfg["style"]
    dataset = cfg["dataset"]
    n = np.array([16, 16, 16] if style == "hires" else [8, 8, 8], dtype=np.int64)
    cell = np.array([5e-8, 5e-8, 5e-8] if style == "hires"
                    else [1e-7, 1e-7, 1e-7])
    arrays = {"n": n, "origin": np.zeros(3)}
    if style == "hires":
        arrays["cell_m"] = np.asarray(5e-8)
    else:
        arrays["cell"] = cell
    if style == "large1":
        arrays["centre"] = n * cell / 2.0
    if "sources" in cfg.get("copy_metadata", []) or style == "bdata2":
        arrays["sources"] = np.arange(12, dtype=float).reshape(4, 3)
    for key in cfg.get("copy_metadata", []):
        if key == "sources":
            continue
        arrays[key] = np.asarray({
            "case": "g10", "beta": 0.9, "gamma": 10.0, "uy": 10.0,
            "particle_pos_m": [1e-6, 1e-6, 1e-6], "host": "jureca-h100",
            "picbunch": "synthetic test binary", "reference": "synthetic reference",
            "err_definition": "synthetic |pic-ref|/peak",
        }.get(key, 0.0))
    for plane in cfg["planes"]:
        shape = axis.expected_stored_shape(dataset, plane, n)
        base = np.arange(shape[0] * shape[1], dtype=float).reshape(shape) + 1.0
        for comp in ingest.COMP:
            arrays[f"{plane}_{comp}_pic"] = base * (1.0 + ingest.COMP.index(comp))
            arrays[f"{plane}_{comp}_ref"] = base * (1.0 + ingest.COMP.index(comp)) * 0.99
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)


def _h5_field_source(path, cfg, root):
    import h5py

    n = [int(v) for v in cfg["n"]]
    nz, ny, nx = n[2], n[1], n[0]
    comp_node = {"Ex": ("E", "x"), "Ey": ("E", "y"), "Ez": ("E", "z"),
                 "Bx": ("B", "x"), "By": ("B", "y"), "Bz": ("B", "z")}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as fh:
        g = fh.require_group("data/0/meshes")
        for comp, (part, axis_name) in comp_node.items():
            base = (np.arange(nx * ny * nz, dtype=float).reshape(nz, ny, nx)
                    + 1.0)
            if cfg["style"] == "large1" and comp == "Bx":
                base[:] = 0.0
            if cfg["style"] == "large1" and comp in ("By", "Bz"):
                base[:] = base * 1e-4
            g.require_group(part).create_dataset(
                axis_name, data=base * (1.0 + ingest.COMP.index(comp)))
    ref = cfg["reference"]
    ref_path = ingest._resolve(root, ref["path"])
    os.makedirs(os.path.dirname(ref_path), exist_ok=True)
    if ref["kind"] == "bin6":
        shape = (nz, ny, nx)
        blocks = np.ones((6,) + shape, dtype=np.float64)
        for i, comp in enumerate(ingest.COMP):
            blocks[i] *= 1.0 + i
        if cfg["style"] == "large1":
            blocks[3] = 0.0
        blocks.tofile(ref_path)
    elif ref["kind"] == "npz":
        arrays = {}
        for plane in cfg["planes"]:
            shape = axis.expected_stored_shape(cfg["dataset"], plane, n)
            for comp in ingest.COMP:
                arrays["%s_%s_ref" % (plane, comp)] = (
                    np.arange(shape[0] * shape[1], dtype=float).reshape(shape)
                    + 1.0)
        for name in cfg.get("copy_metadata", []):
            if name == "sources":
                arrays[name] = np.arange(12, dtype=float).reshape(4, 3)
        np.savez_compressed(ref_path, **arrays)


def _source_for(entry, root):
    source = entry["source"]
    path = ingest._resolve(root, source["path"])
    if os.path.exists(path):
        return
    if source.get("root", "measurement") == "repo":
        # An in-repo reduced table: the synthetic campaign root carries a
        # byte-identical copy of the repo file, so the mapping sha256
        # (pinned to the repo copy) verifies in the temp root too.
        repo_file = os.path.join(ingest.REPO_ROOT, source["path"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(repo_file, "rb") as fh:
            data = fh.read()
        with open(path, "wb") as fh:
            fh.write(data)
        return
    tr = entry["transform"]
    if tr["kind"] == "csv_copy":
        _write_csv(path, tr.get("columns") or ingest.required_csv_columns(
            entry["target"]))
    elif tr["kind"] == "npz_copy":
        arrays = {}
        for target_key in tr["keys"]:
            arrays[target_key] = (np.arange(32, dtype=float).reshape(4, 8)
                                  if target_key == "sources"
                                  else np.arange(64, dtype=float).reshape(8, 8))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **arrays)
    elif tr["kind"] == "json_mpi_medians":
        data = {}
        for ranks in tr["expected_ranks"]:
            data[str(ranks)] = {"total": 10.0 / ranks, "fft": 2.0 / ranks,
                                "writer": 0.7 / ranks,
                                "total_spread": 0.1, "fft_spread": 0.01,
                                "writer_spread": 0.02, "n": 3}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh)
    elif tr["kind"] == "field_export":
        _field_source(path, tr)
    elif tr["kind"] == "field_h5":
        _h5_field_source(path, tr, root)
    else:
        raise AssertionError("no synthetic generator for %s" % tr["kind"])


class IngestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ingest-selftest-")
        self.root = os.path.join(self.tmp, "measurements")
        self.data = os.path.join(self.tmp, "data")
        self.records = os.path.join(self.tmp, "records")
        self.figs = os.path.join(self.tmp, "figs")
        os.makedirs(self.root)
        os.makedirs(self.data)
        os.makedirs(self.records)
        os.makedirs(self.figs)
        with open(MAP) as fh:
            self.mapping = json.load(fh)
        # Keep the synthetic HDF5 cases small: shrink the real grid to 8^3.
        for entry in self.mapping["entries"]:
            tr = entry["transform"]
            if tr["kind"] != "field_h5":
                continue
            tr["n"] = [8, 8, 8]
            for plane, spec in tr["plane_index"].items():
                spec["index"] = 4
            if tr["reference"]["kind"] == "bin6":
                tr["reference"]["shape"] = [8, 8, 8]
        for entry in self.mapping["entries"]:
            _source_for(entry, self.root)
        for cid, meta in self.mapping["cases"].items():
            config = meta["provenance"].get("config")
            if config and "path" in config:
                if config.get("root", "measurement") == "measurement":
                    path = ingest._resolve(self.root, config["path"])
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "w") as fh:
                        fh.write("synthetic config\n")
                else:
                    # A repo-rooted config is pinned by digest; the temp
                    # root carries a byte-identical copy of the repo file.
                    path = ingest._resolve(self.root, config["path"])
                    repo_file = os.path.join(ingest.REPO_ROOT, config["path"])
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(repo_file, "rb") as fh:
                        data = fh.read()
                    with open(path, "wb") as fh:
                        fh.write(data)
                    config["sha256"] = ingest.sha256_file(path)
            for figure in cases.case(cid).get("figures", []):
                open(os.path.join(self.figs, figure + ".pdf"), "wb").close()
        for entry in self.mapping["entries"]:
            source = entry["source"]
            if source.get("root", "measurement") == "repo":
                # In-repo sources verify against the canonical campaign
                # digest; the synthetic root carries a byte-identical copy of
                # the campaign artifact (the repo file, whose rows are
                # unchanged), so no re-pinning is needed.
                continue
            path = ingest._resolve(self.root, source["path"])
            source["sha256"] = ingest.sha256_file(path)
        self.map = os.path.join(self.tmp, "map.json")
        with open(self.map, "w") as fh:
            json.dump(self.mapping, fh)
        self.repo_data_before = tree_hash(os.path.join(PAPER_DIR, "data"))
        self.repo_records_before = tree_hash(
            os.path.join(PAPER_DIR, "provenance", "records"))

    def tearDown(self):
        self.assertEqual(
            tree_hash(os.path.join(PAPER_DIR, "data")), self.repo_data_before,
            "synthetic run modified docs/paper/data")
        self.assertEqual(
            tree_hash(os.path.join(PAPER_DIR, "provenance", "records")),
            self.repo_records_before,
            "synthetic run modified docs/paper/provenance/records")
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self):
        return subprocess.run(
            [sys.executable, os.path.join(HERE, "ingest_measurements.py"),
             "--measurement-root", self.root, "--map", self.map,
             "--data-dir", self.data, "--records-dir", self.records],
            capture_output=True, text=True, cwd=REPO_ROOT)

    def test_complete_set_passes_strict_gate(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        for entry in self.mapping["entries"]:
            target = os.path.join(self.data, entry["target"])
            self.assertTrue(os.path.isfile(target), target)
            if target.endswith(".npz"):
                with np.load(target, allow_pickle=True) as d:
                    missing = [k for k in ingest.required_npz_keys(
                        entry["target"]) if k not in d.files]
                    self.assertEqual(missing, [],
                                     "%s missing %s" % (target, missing))
        errors, summary = check_provenance.run(
            self.records, self.figs,
            check_provenance.DEFAULT_SCHEMA,
            check_provenance.DEFAULT_NOT_REPRODUCED,
            check_provenance.DEFAULT_NON_DATA, True,
            check_provenance.DEFAULT_CASES)
        self.assertEqual(errors, [], "\n".join(errors))
        self.assertEqual(summary["records"], len(self.mapping["cases"]))
        self.assertEqual(summary["figures_missing"], 0)

    def test_missing_source_fails_loudly_and_keeps_legacy(self):
        entry = [e for e in self.mapping["entries"]
                 if e["case_id"] == "large1-g10"][0]
        victim = ingest._resolve(self.root, entry["source"]["path"])
        os.remove(victim)
        target = os.path.join(self.data, entry["target"])
        os.makedirs(os.path.dirname(target), exist_ok=True)
        sentinel = b"legacy reduced data must survive"
        with open(target, "wb") as fh:
            fh.write(sentinel)
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("large1-g10", result.stderr)
        self.assertIn(entry["source"]["path"], result.stderr)
        with open(target, "rb") as fh:
            self.assertEqual(fh.read(), sentinel)
        self.assertFalse(os.path.exists(
            os.path.join(self.records, "large1-g10.json")))

    def test_repo_sourced_matrix_ingests_and_renders(self):
        # The scaling-cost case reads its source from the repo (root: repo),
        # pinned to the canonical campaign digest.  The synthetic root holds a
        # stand-in whose digest is re-pinned, so the ingest writes the
        # reduced table with the drop_columns applied.
        entry = [e for e in self.mapping["entries"]
                 if e["case_id"] == "scaling-cost"][0]
        self.assertEqual(entry["source"].get("root"), "repo")
        self.assertEqual(
            entry["transform"].get("drop_columns"),
            ["row_class", "total_iqr_ms"])
        source_path = ingest._resolve(self.root, entry["source"]["path"])
        target = os.path.join(self.data, entry["target"])
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(os.path.isfile(target),
                        "repo-sourced reduced table was not written")
        with open(target, newline="") as fh:
            header = fh.readline().strip().split(",")
        self.assertNotIn("row_class", header)
        self.assertNotIn("total_iqr_ms", header)
        self.assertIn("total_median_ms", header)
        # The provenance record must be emitted.
        self.assertTrue(os.path.exists(
            os.path.join(self.records, "scaling-cost.json")))

    def test_file_checksum_mismatch_is_fatal(self):
        # A campaign-rooted case, not a repo-sourced one: a tampered temp
        # copy must fail the digest check.
        entry = [e for e in self.mapping["entries"]
                 if e["transform"]["kind"] == "csv_copy"
                 and e["source"].get("root", "measurement") == "measurement"][0]
        victim = ingest._resolve(self.root, entry["source"]["path"])
        with open(victim, "a") as fh:
            fh.write("# tampered\n")
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checksum mismatch", result.stderr)
        self.assertIn(entry["source"]["path"], result.stderr)
        self.assertFalse(os.path.exists(
            os.path.join(self.data, entry["target"])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
