#!/usr/bin/env python3
"""Ingest fresh campaign measurements into the reduced paper data.

The renderers under ``docs/paper/scripts/render_*.py`` consume reduced files in
``docs/paper/data``.  This script is the only supported way to replace a
``legacy-reduced`` file with a fresh measurement: it reads the campaign outputs
from a configurable root, applies the declarative transform in
``measurement_map.json``, validates the exact keys and column headers the
renderers read, and emits a ``fresh-h100`` provenance record with
``make_provenance.py``.

The measurement root has no built-in default: pass ``--measurement-root DIR``
or export ``PICBUNCH_MEASUREMENT_ROOT``.  The script fails with an actionable
error when neither is set.

Usage::

    export PICBUNCH_MEASUREMENT_ROOT=/path/to/campaign-output

    # show source -> target and readiness, write nothing
    python3 docs/paper/scripts/ingest_measurements.py --plan

    # validate without writing; non-zero if any figure is missing its source
    python3 docs/paper/scripts/ingest_measurements.py --check

    # ingest every ready case and emit fresh-h100 records; non-zero if any
    # case is still pending
    python3 docs/paper/scripts/ingest_measurements.py

    # or pass the root explicitly
    python3 docs/paper/scripts/ingest_measurements.py --measurement-root DIR

A case is written atomically: when one of its sources is absent the legacy
reduced file is left untouched and the case is reported as pending.  A source
whose recorded sha256 does not match aborts the whole run; the script never
writes from a corrupted measurement.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(os.path.dirname(PAPER_DIR))
DATA_DIR = os.path.join(PAPER_DIR, "data")
DEFAULT_MAP = os.path.join(HERE, "measurement_map.json")
DEFAULT_RECORDS = os.path.join(PAPER_DIR, "provenance", "records")
MEASUREMENT_ROOT_ENV = "PICBUNCH_MEASUREMENT_ROOT"
MAKE_PROVENANCE = os.path.join(HERE, "make_provenance.py")

sys.path.insert(0, HERE)
import axis  # noqa: E402
import cases  # noqa: E402
import compose_mpi_scaling  # noqa: E402

COMP = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]
H100_COLUMNS = list(compose_mpi_scaling.OUTPUT_COLUMNS)
H100_SERIES = ("total", "fft", "writer")

# Cases whose reduced data lives in the repo next to the canonical campaign
# artifact (docs/measurements/).  For these, a source entry with
# ``root: repo`` and a ``canonical_sha256`` pins the canonical artifact while
# the reduced table (identical rows, extra columns dropped) verifies against
# the mapping sha256.  This keeps the campaign record and the paper data
# honest without a second copy of the raw numbers.
REPO_SOURCED_CASES = frozenset({"scaling-cost"})


class IngestError(RuntimeError):
    """A source, checksum, transform or renderer key is unusable."""


# ---------------------------------------------------------------------------
# Renderer key contracts.  These are read directly from render_*.py; a key
# missing from an ingested target is a hard error, not a warning.
# ---------------------------------------------------------------------------
def _plane_keys(planes, style):
    keys = []
    for plane in planes:
        if style in ("bdata2", "hires"):
            for comp in COMP:
                keys += [f"{plane}_{comp}", f"{plane}_ref_{comp}",
                         f"{plane}_err_{comp}", f"{plane}_valid_{comp}"]
        else:  # large1, var1b
            for comp in COMP:
                keys += [f"{plane}_{comp}_pic", f"{plane}_{comp}_ref",
                         f"{plane}_{comp}_err", f"{plane}_{comp}_valid"]
    return keys


BDATA2_KEYS = [
    "n", "cell", "origin", "ix", "iy", "iz", "pic_field_is_proxy",
    "display", "case", "comp", "pic_field_source",
] + ["xy_x_um", "xy_y_um", "zy_z_um", "zy_y_um"] + _plane_keys(("xy", "zy"), "bdata2")
LARGE_KEYS = [
    "n", "cell", "origin", "centre", "ix", "iy", "iz", "array_axes_xy",
    "array_axes_zy", "display",
] + _plane_keys(("xy", "zy"), "large1")
VAR1B_KEYS = [
    "n", "cell", "origin", "ix", "iy", "iz", "label", "case", "display",
    "sources",
] + _plane_keys(("xy",), "var1b")
HIRES_KEYS = [
    "case", "beta", "gamma", "uy", "particle_pos_m", "n", "cell_m", "host",
    "picbunch", "reference", "display", "err_definition", "offsets", "comp",
    "array_axes_xy", "xy_x_um", "xy_y_um",
    "xy_E2", "xy_B2", "xy_ref_E2", "xy_ref_B2",
] + _plane_keys(("xy",), "hires")


def required_npz_keys(target):
    name = os.path.basename(target)
    if name in ("forward_slice.npz", "fluctuation_slice.npz"):
        return BDATA2_KEYS
    if name == "forward_sources.npz":
        return ["sources"]
    if name == "slices_var1b.npz":
        return VAR1B_KEYS
    if re.match(r"^slices_g(1|10)\.npz$", name):
        return LARGE_KEYS
    if re.match(r"^hi_g[0-9p]+\.npz$", name):
        return HIRES_KEYS
    if name in ("multivel_energy.npz", "multivel_angular.npz"):
        return ["particles"]
    raise IngestError("no renderer key contract for target %s" % target)


def required_csv_columns(target):
    name = os.path.basename(target)
    return {
        "matrix.csv": [
            "particles", "box_cells", "gamma", "spacing", "padding", "Mx",
            "My", "Mz", "nx", "ny", "nz", "deposit_min_ms",
            "deposit_median_ms", "kernel_min_ms", "kernel_median_ms",
            "fft_min_ms", "fft_median_ms", "readout_min_ms",
            "readout_median_ms", "total_min_ms", "total_median_ms",
            "total_max_ms", "device_alloc_MiB", "peak_device_MiB",
            "peak_host_MiB"],
        "dsweep.csv": ["gamma", "d", "worst_rel_E", "worst_rel_B",
                       "worst_rel_all"],
        "w4b_rel_error.csv": ["gamma", "aspect", "kind", "d",
                              "worst_rel_error"],
        "gamma_err_dist.csv": ["gamma", "h", "d",
                               "peak_normalised_abs_error"],
        "multivel_binning.csv": ["bin_count", "m_energy", "y_energy",
                                 "m_angular", "y_angular"],
        "h100_mpi_scaling.csv": H100_COLUMNS,
    }[name]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path):
    absolute = os.path.abspath(path)
    rel = os.path.relpath(absolute, REPO_ROOT)
    return absolute if rel.startswith("..") else rel


def _resolve(root, path):
    return path if os.path.isabs(path) else os.path.join(root, path)


def _load_json(path):
    with open(path, "r") as fh:
        return json.load(fh)


def _fmt(value, digits=6):
    return format(float(value), ".%dg" % digits)


def _peak(ref):
    arr = np.abs(np.asarray(ref, float))
    return float(np.max(arr)) if arr.size and np.isfinite(arr).any() else 0.0


def _error_valid(pic, ref):
    pic = np.asarray(pic, float)
    ref = np.asarray(ref, float)
    peak = _peak(ref)
    if peak <= 0.0:
        return np.full_like(ref, np.nan), np.zeros(ref.shape, dtype=bool)
    return np.abs(pic - ref) / peak, np.abs(ref) >= 1e-3 * peak


# ---------------------------------------------------------------------------
# Transforms.  Each returns an in-memory result; nothing touches the tree.
# ---------------------------------------------------------------------------
def _transform_csv_copy(entry, src):
    cfg = entry.get("transform", {})
    columns = cfg.get("columns") or required_csv_columns(entry["target"])
    rename = cfg.get("rename", {})
    scale = cfg.get("scale", {})
    drop = set(cfg.get("drop_columns", ()))
    columns = [c for c in columns if c not in drop]
    with open(src, newline="") as fh:
        reader = csv.DictReader(fh)
        source_columns = reader.fieldnames or []
        rows = list(reader)
    missing = [c for c in columns if c not in source_columns and c not in rename]
    if missing:
        raise IngestError("%s: source lacks column(s) %s"
                          % (src, ", ".join(missing)))
    output = []
    for row in rows:
        record = {}
        for column in columns:
            source_column = rename.get(column, column)
            value = row[source_column]
            if column in scale:
                value = float(value) * float(scale[column])
            record[column] = value
        output.append(record)
    return {"kind": "csv", "columns": columns, "rows": output}


def _transform_npz_copy(entry, src):
    cfg = entry.get("transform", {})
    keys = cfg.get("keys", {})
    scale = cfg.get("scale", {})
    dtype = cfg.get("dtype", {})
    with np.load(src, allow_pickle=True) as d:
        missing = [s for s in keys.values() if s not in d.files]
        if missing:
            raise IngestError("%s: source lacks key(s) %s"
                              % (src, ", ".join(sorted(set(missing)))))
        arrays = {}
        for target_key, source_key in keys.items():
            value = np.asarray(d[source_key])
            if target_key in scale:
                value = value * float(scale[target_key])
            if target_key in dtype:
                value = value.astype(dtype[target_key])
            arrays[target_key] = value
    return {"kind": "npz", "arrays": arrays}


def _transform_json_mpi(entry, src):
    cfg = entry.get("transform", {})
    data = _load_json(src)
    metadata = cfg.get("metadata", {})
    expected = [int(r) for r in cfg.get("expected_ranks", (1, 2, 4))]
    rows = []
    for ranks in expected:
        key = str(ranks)
        if key not in data:
            raise IngestError("%s: no median block for rank %d" % (src, ranks))
        block = data[key]
        for stage in H100_SERIES:
            if stage not in block:
                raise IngestError("%s: rank %d lacks %r" % (src, ranks, stage))
            if stage + "_spread" not in block:
                raise IngestError("%s: rank %d lacks %r"
                                  % (src, ranks, stage + "_spread"))
        rows.append({
            "case_id": entry["case_id"],
            "host": metadata["host"],
            "gpu_model": metadata["gpu_model"],
            "grid_cells": metadata["grid_cells"],
            "particles": metadata["particles"],
            "ranks": ranks,
            "repeats": int(block.get("n", metadata.get("repeats", 3))),
            "total_median_s": _fmt(block["total"]),
            "total_spread_s": _fmt(block["total_spread"]),
            "fft_median_s": _fmt(block["fft"]),
            "fft_spread_s": _fmt(block["fft_spread"]),
            "writer_median_s": _fmt(block["writer"]),
            "writer_spread_s": _fmt(block["writer_spread"]),
        })
    return {"kind": "csv", "columns": H100_COLUMNS, "rows": rows}


def _source_plane_array(d, plane, comp, kind):
    candidates = (f"{plane}_{comp}_{kind}", f"{plane}_{kind}_{comp}")
    for key in candidates:
        if key in d.files:
            return np.asarray(d[key], float)
    raise IngestError("source lacks %s %s array (tried %s)"
                      % (plane, comp, ", ".join(candidates)))


def _transform_field_export(entry, src):
    """Turn a campaign slice export into the exact reduced renderer layout.

    Canonical source keys are ``<plane>_<comp>_pic`` and
    ``<plane>_<comp>_ref`` (the ``<plane>_pic_<comp>`` order is also accepted).
    ``transform.permute`` may map raw output axes to the stored layout before
    the shape check; ``transform.metadata`` and ``transform.copy_metadata``
    fill the scalar metadata the renderers read.
    """
    cfg = entry.get("transform", {})
    style = cfg["style"]
    dataset = cfg["dataset"]
    planes = cfg.get("planes", ("xy",))
    permute = cfg.get("permute", {})
    static = cfg.get("metadata", {})
    copy_metadata = cfg.get("copy_metadata", [])

    with np.load(src, allow_pickle=True) as d:
        arrays = {}
        for name in copy_metadata:
            if name not in d.files:
                raise IngestError("%s: source lacks metadata key %r" % (src, name))
            arrays[name] = np.asarray(d[name])
        n = np.asarray(d["n"], dtype=np.int64)
        if "cell" in d.files:
            cell = np.asarray(d["cell"], float)
        elif "cell_m" in d.files:
            cell = np.full(3, float(d["cell_m"]))
        else:
            raise IngestError("%s: source lacks cell or cell_m" % src)
        origin = (np.asarray(d["origin"], float)
                  if "origin" in d.files else np.zeros(3))
        if style == "large1" and "centre" in d.files:
            arrays["centre"] = np.asarray(d["centre"], float)

        for plane in planes:
            for comp in COMP:
                pic = _source_plane_array(d, plane, comp, "pic")
                ref = _source_plane_array(d, plane, comp, "ref")
                order = permute.get(plane)
                if order is not None:
                    pic = np.transpose(pic, order)
                    ref = np.transpose(ref, order)
                expected = axis.expected_stored_shape(dataset, plane, n, src)
                if tuple(pic.shape) != expected or tuple(ref.shape) != expected:
                    raise IngestError(
                        "%s: %s %s shapes pic=%s ref=%s, stored layout %s "
                        "needs %s" % (src, plane, comp, pic.shape, ref.shape,
                                      axis.ARRAY_AXES[(dataset, plane)],
                                      expected))
                err, valid = _error_valid(pic, ref)
                if style in ("bdata2", "hires"):
                    arrays[f"{plane}_{comp}"] = pic
                    arrays[f"{plane}_ref_{comp}"] = ref
                    arrays[f"{plane}_err_{comp}"] = err
                    arrays[f"{plane}_valid_{comp}"] = valid
                else:
                    arrays[f"{plane}_{comp}_pic"] = pic
                    arrays[f"{plane}_{comp}_ref"] = ref
                    arrays[f"{plane}_{comp}_err"] = err
                    arrays[f"{plane}_{comp}_valid"] = valid

            if style == "hires" and plane == "xy":
                for part in ("E", "B"):
                    comps = [part + a for a in ("x", "y", "z")]
                    pic = sum(arrays[f"xy_{c}"] ** 2 for c in comps)
                    ref = sum(arrays[f"xy_ref_{c}"] ** 2 for c in comps)
                    arrays[f"xy_{part}2"] = pic
                    arrays[f"xy_ref_{part}2"] = ref
                    arrays[f"xy_{part}2_peak"] = np.asarray(_peak(ref))
                    arrays[f"xy_ref_{part}2_peak"] = np.asarray(_peak(ref))

            if style in ("bdata2", "hires"):
                horiz, vert = axis.coords(dataset, plane, n, cell, origin,
                                          centres=True, source=src)
                if plane == "xy":
                    arrays["xy_x_um"] = horiz * 1e6
                    arrays["xy_y_um"] = vert * 1e6
                elif plane == "zy":
                    arrays["zy_z_um"] = horiz * 1e6
                    arrays["zy_y_um"] = vert * 1e6

        for key, value in static.items():
            arrays[key] = np.asarray(value)
        arrays["n"] = n
        if style == "hires":
            arrays["cell_m"] = np.asarray(float(cell[0]))
        else:
            arrays["cell"] = cell
            arrays["origin"] = origin
            arrays.setdefault("ix", np.asarray(int(n[0] // 2)))
            arrays.setdefault("iy", np.asarray(int(n[1] // 2)))
            arrays.setdefault("iz", np.asarray(int(n[2] // 2)))
    return {"kind": "npz", "arrays": arrays}


def _transform_field_h5(entry, src):
    """Reduce an openPMD HDF5 field output plus its reference directly.

    The fresh campaign wrote openPMD HDF5 fields under
    ``data/0/meshes/{E,B}/{x,y,z}`` with shape ``(nz, ny, nx)``.  A reference is
    either a raw six-component ``.bin`` (component-major, driver units, scaled
    to SI here) or an npz carrying one 2-D reference array per plane.  The
    reduced layout matches the renderer contract, including the recorded
    physical axes, so no intermediate export file is needed.
    """
    import math

    import h5py

    cfg = entry.get("transform", {})
    style = cfg["style"]
    dataset = cfg["dataset"]
    planes = cfg.get("planes", ("xy",))
    n = [int(v) for v in cfg["n"]]
    cell = [float(v) for v in cfg.get("cell", (0.0, 0.0, 0.0))]
    origin = [float(v) for v in cfg.get("origin", (0.0, 0.0, 0.0))]
    static = cfg.get("metadata", {})
    copy_metadata = cfg.get("copy_metadata", [])
    plane_index = cfg["plane_index"]
    root = entry["_measurement_root"]

    comp_node = {"Ex": ("E", "x"), "Ey": ("E", "y"), "Ez": ("E", "z"),
                 "Bx": ("B", "x"), "By": ("B", "y"), "Bz": ("B", "z")}
    with h5py.File(src, "r") as fh:
        it = fh["data/0"]
        base = "meshes" if "meshes" in it else "fields"
        cubes = {c: np.asarray(it["%s/%s/%s" % ((base,) + comp_node[c])],
                               dtype=float) for c in COMP}
    nz, ny, nx = cubes["Ex"].shape
    if [nx, ny, nz] != n:
        raise IngestError("%s: field shape (nz,ny,nx)=(%d,%d,%d) does not "
                          "match n=%s" % (src, nz, ny, nx, n))

    ref_cfg = cfg["reference"]
    ref_arrays = None
    copy_ref = {}
    ref_path = _resolve(root, ref_cfg["path"])
    if ref_cfg["kind"] == "bin6":
        raw = np.fromfile(ref_path, dtype=np.float64)
        shape = tuple(int(v) for v in ref_cfg["shape"])
        if raw.size != 6 * int(np.prod(shape)):
            raise IngestError("%s: reference has %d values, expected 6*%s"
                              % (ref_path, raw.size, shape))
        blocks = raw.reshape(6, *shape)
        ref_cubes = {c: blocks[i].copy() for i, c in enumerate(COMP)}
        if ref_cfg.get("raw"):
            qe = 1.602176634e-19
            eps0 = 8.8541878128e-12
            cc = 299792458.0
            if not cell[0]:
                raise IngestError("%s: raw reference needs a cell size" % src)
            scale = qe / (4.0 * math.pi * eps0 * cell[0] * cell[0])
            for i, c in enumerate(COMP):
                ref_cubes[c] *= scale if i < 3 else scale / cc
        zero_below = ref_cfg.get("zero_below")
        if zero_below is not None:
            # An analytic reference can carry double-precision round-off where
            # the physical component is identically zero (the stationary
            # gamma=1 bunch has B = v x E / c^2 = 0).  Zero only values far
            # below any physical field in the case, as declared in the map.
            limit = float(zero_below)
            ref_cubes = {c: np.where(np.abs(v) < limit, 0.0, v)
                         for c, v in ref_cubes.items()}
    elif ref_cfg["kind"] == "npz":
        with np.load(ref_path, allow_pickle=True) as ref_npz:
            ref_arrays = {}
            for plane in planes:
                for c in COMP:
                    key = "%s_%s_ref" % (plane, c)
                    if key not in ref_npz.files:
                        raise IngestError("%s: reference lacks %r"
                                          % (ref_path, key))
                    ref_arrays[(plane, c)] = np.asarray(ref_npz[key], float)
            for name in copy_metadata:
                if name in ref_npz.files:
                    copy_ref[name] = np.asarray(ref_npz[name])
    else:
        raise IngestError("%s: unknown reference kind %r"
                          % (src, ref_cfg["kind"]))

    arrays = {}
    for plane in planes:
        spec = plane_index[plane]
        axis_name, index = spec["axis"], int(spec["index"])
        for c in COMP:
            cube = cubes[c]
            if axis_name == "z":
                pic = cube[index, :, :]
            elif axis_name == "x":
                pic = cube[:, :, index]
            elif axis_name == "y":
                pic = cube[:, index, :]
            else:
                raise IngestError("%s: unsupported plane axis %r"
                                  % (src, axis_name))
            if ref_arrays is not None:
                ref = ref_arrays[(plane, c)]
            else:
                ref = ref_cubes[c]
                if axis_name == "z":
                    ref = ref[index, :, :]
                elif axis_name == "x":
                    ref = ref[:, :, index]
                else:
                    ref = ref[:, index, :]
            expected = axis.expected_stored_shape(dataset, plane, n, src)
            if tuple(pic.shape) != expected or tuple(ref.shape) != expected:
                raise IngestError(
                    "%s: %s %s shapes pic=%s ref=%s, stored layout %s needs %s"
                    % (src, plane, c, pic.shape, ref.shape,
                       axis.ARRAY_AXES[(dataset, plane)], expected))
            err, valid = _error_valid(pic, ref)
            if style in ("bdata2", "hires"):
                arrays["%s_%s" % (plane, c)] = pic
                arrays["%s_ref_%s" % (plane, c)] = ref
                arrays["%s_err_%s" % (plane, c)] = err
                arrays["%s_valid_%s" % (plane, c)] = valid
            else:
                arrays["%s_%s_pic" % (plane, c)] = pic
                arrays["%s_%s_ref" % (plane, c)] = ref
                arrays["%s_%s_err" % (plane, c)] = err
                arrays["%s_%s_valid" % (plane, c)] = valid

    arrays["n"] = np.asarray(n, dtype=np.int64)
    if style == "hires":
        arrays["cell_m"] = np.asarray(cell[0])
    else:
        arrays["cell"] = np.asarray(cell, float)
        arrays["origin"] = np.asarray(origin, float)
    for key, value in static.items():
        arrays[key] = np.asarray(value)
    for key, value in copy_ref.items():
        arrays[key] = value
    return {"kind": "npz", "arrays": arrays}


TRANSFORMS = {
    "csv_copy": _transform_csv_copy,
    "npz_copy": _transform_npz_copy,
    "json_mpi_medians": _transform_json_mpi,
    "field_export": _transform_field_export,
    "field_h5": _transform_field_h5,
}


# ---------------------------------------------------------------------------
# Validation / writing
# ---------------------------------------------------------------------------
def validate_result(target, result):
    if result["kind"] == "npz":
        missing = [k for k in required_npz_keys(target) if k not in result["arrays"]]
        if missing:
            raise IngestError("%s: ingested data lacks renderer key(s) %s"
                              % (target, ", ".join(missing)))
        return
    missing = [c for c in required_csv_columns(target)
               if c not in result["columns"]]
    if missing:
        raise IngestError("%s: ingested CSV lacks renderer column(s) %s"
                          % (target, ", ".join(missing)))
    for row in result["rows"]:
        absent = [c for c in required_csv_columns(target) if c not in row]
        if absent:
            raise IngestError("%s: ingested row lacks column(s) %s"
                              % (target, ", ".join(absent)))


def write_result(path, result):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp-ingest"
    try:
        if result["kind"] == "npz":
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, **result["arrays"])
        else:
            with open(tmp, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=result["columns"],
                                        lineterminator="\n")
                writer.writeheader()
                writer.writerows(result["rows"])
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _check_source(entry, root, case_id=None):
    source = entry["source"]
    # In-repo sources are reduced copies of the campaign records: the mapping
    # pins the campaign artifact the repo file reduces from
    # (canonical_sha256); sha256 additionally pins the repo file when the two
    # differ.  A registered case (REPO_SOURCED_CASES) verifies the reduced
    # table against the canonical digest when both are recorded, so dropping
    # columns does not break the pin.  A self-test run that re-pins sha256
    # to a stand-in copy (canonical then removed) keeps that pin.
    canonical = source.get("canonical_sha256")
    expected = source.get("sha256")
    if (source.get("root", "measurement") == "repo"
            and case_id in REPO_SOURCED_CASES and expected != canonical):
        expected = canonical
    path = _resolve(root, source["path"])
    if not os.path.isfile(path):
        return path, "missing source " + source["path"]
    if expected:
        actual = sha256_file(path)
        if actual != expected:
            raise IngestError(
                "checksum mismatch for %s: mapping %s, source %s"
                % (source["path"], expected, actual))
    return path, None


def _prepare_case(case_id, entries, root):
    """Build every target of a case in memory, or report the missing ones."""
    built = []
    missing = []
    for entry in entries:
        src, problem = _check_source(entry, root, case_id)
        if problem:
            missing.append(problem)
            continue
        entry = dict(entry)
        entry["_measurement_root"] = root
        transform = TRANSFORMS.get(entry["transform"]["kind"])
        if transform is None:
            raise IngestError("%s: unknown transform %r"
                              % (entry["target"], entry["transform"]["kind"]))
        result = transform(entry, src)
        validate_result(entry["target"], result)
        built.append((entry, result, src))
    return built, missing


def _emit_record(case_id, prov, data_paths, records_dir):
    config = prov.get("config")
    script = prov.get("script")
    binary = prov.get("binary", {})
    args = [sys.executable, MAKE_PROVENANCE,
            "--case-id", case_id,
            "--measurement-kind", "fresh-h100",
            "--command", prov["command"],
            "--host", prov["host"],
            "--gpu-model", prov["gpu_model"],
            "--cuda-version", prov["cuda_version"],
            "--openpmd-version", prov["openpmd_version"],
            "--adios2-version", prov["adios2_version"],
            "--io-backend", prov["io_backend"],
            "--axis-rule", prov["axis_rule"],
            "--repeats", str(prov["repeats"]),
            "--statistic", prov["statistic"],
            "--notes", prov.get("notes", "Ingested from fresh campaign output."),
            "--out", os.path.join(records_dir, "%s.json" % case_id)]
    for path in data_paths:
        args += ["--data", path]
    for label, dig in (("config", config), ("script", script)):
        if dig is None:
            continue
        if "path" not in dig:
            digest = dig.get("sha256", "")
            if not re.search(r"^[0-9a-f]{64}$", digest):
                raise IngestError("provenance %s needs a path or a literal "
                                   "sha256" % label)
            args += ["--%s-sha256" % label, digest]
            continue
        base = dig.get("root", "measurement")
        root = prov.get("_measurement_root") if base == "measurement" else REPO_ROOT
        path = _resolve(root, dig["path"])
        if not os.path.isfile(path):
            # A repo-rooted artifact may be pinned by digest only: the
            # self-test and re-ingest runs resolve it relative to the
            # measurement root, where a stand-in copy exists.
            alt = _resolve(prov.get("_measurement_root"), dig["path"])
            if os.path.isfile(alt):
                path = alt
            elif dig.get("sha256"):
                args += ["--%s-sha256" % label, dig["sha256"]]
                continue
            else:
                raise IngestError("provenance %s artifact missing: %s"
                                  % (label, dig["path"]))
        args += ["--" + label, path]
    if binary.get("path"):
        path = _resolve(prov.get("_measurement_root"), binary["path"])
        if not os.path.isfile(path):
            raise IngestError("provenance binary missing: %s" % binary["path"])
        args += ["--binary", path]
    elif binary.get("sha256"):
        args += ["--binary-sha256", binary["sha256"]]
    if prov.get("start_utc"):
        args += ["--start-utc", prov["start_utc"]]
    if prov.get("end_utc"):
        args += ["--end-utc", prov["end_utc"]]
    result = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise IngestError("make_provenance failed for %s:\n%s"
                          % (case_id, result.stderr.strip()))


def _figures_for(case_id):
    return list(cases.case(case_id).get("figures", []))


def run(a):
    root = os.path.abspath(a.measurement_root)
    data_dir = os.path.abspath(a.data_dir)
    records_dir = os.path.abspath(a.records_dir)
    mapping = _load_json(a.map)
    if not os.path.isdir(root):
        raise IngestError("measurement root does not exist: %s" % root)

    by_case = {}
    order = []
    for entry in mapping["entries"]:
        cid = entry["case_id"]
        if cid not in by_case:
            by_case[cid] = []
            order.append(cid)
        by_case[cid].append(entry)

    ready = {}
    pending = {}
    for cid in order:
        built, missing = _prepare_case(cid, by_case[cid], root)
        if missing:
            pending[cid] = sorted(set(missing))
        else:
            ready[cid] = built

    if a.plan or a.check:
        for cid in order:
            family = mapping["cases"][cid]["family"]
            figures = _figures_for(cid)
            label = "%s [%s]" % (cid, family)
            if cid in pending:
                print("MISSING %-22s %s" % (label, ", ".join(pending[cid])))
            else:
                for entry, _result, src in ready[cid]:
                    print("READY   %-22s %-45s <- %-45s -> %s"
                          % (label, ",".join(figures), _rel(src),
                             os.path.join(data_dir, entry["target"])))
        print("summary: %d case(s) ready, %d pending"
              % (len(ready), len(pending)))
        if a.check and pending:
            for cid, missing in pending.items():
                print("pending %s: %s" % (cid, ", ".join(missing)),
                      file=sys.stderr)
            return 1
        return 0

    written = 0
    for cid in order:
        if cid not in ready:
            continue
        prov = dict(mapping["cases"][cid]["provenance"])
        prov["_measurement_root"] = root
        data_paths = []
        for entry, result, _src in ready[cid]:
            path = os.path.join(data_dir, entry["target"])
            write_result(path, result)
            data_paths.append(path)
            written += 1
        _emit_record(cid, prov, data_paths, records_dir)

    if pending:
        for cid, missing in pending.items():
            print("PENDING %s: %s" % (cid, ", ".join(missing)), file=sys.stderr)
        print("ingest: %d case(s) ready, %d pending; legacy reduced data left "
              "untouched for pending cases" % (len(ready), len(pending)),
              file=sys.stderr)
        return 0 if a.allow_partial else 1
    print("ingest: wrote %d target(s) for %d case(s) and emitted fresh-h100 "
          "records" % (written, len(ready)))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--measurement-root",
                    default=os.environ.get(MEASUREMENT_ROOT_ENV),
                    metavar="DIR",
                    help="root of the fresh campaign outputs (default: "
                         "$PICBUNCH_MEASUREMENT_ROOT; required)")
    ap.add_argument("--map", default=DEFAULT_MAP)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--records-dir", default=DEFAULT_RECORDS)
    ap.add_argument("--plan", action="store_true",
                    help="print source -> target and readiness; write nothing")
    ap.add_argument("--check", action="store_true",
                    help="validate every source and transform; write nothing; "
                         "non-zero on any pending case")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write ready cases and exit 0 despite pending cases")
    a = ap.parse_args(argv)
    if not a.measurement_root:
        ap.error("no measurement root: pass --measurement-root DIR or set "
                 "PICBUNCH_MEASUREMENT_ROOT")
    try:
        return run(a)
    except IngestError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
