#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
"""Compare an HDF5 restart checkpoint with a picBunch field output.

The checkpoint is the HDF5 file written by picBunch's PIConGPU restart path
(``fields/`` under ``data/<step>``).  The output is either an HDF5 or ADIOS2
BP5 series written by the normal field path (``meshes/`` under the first
iteration).  Both are compared at the openPMD level:

* payload: every ``{E,B}/{x,y,z}`` array is compared bit for bit (same shape,
  same dtype, same bytes, so NaN payloads and signed zeros must also match);
* metadata: iteration ``time``/``dt``/``timeUnitSI``/``timeOffset`` and the
  per-record and per-component attributes (``axisLabels``, ``gridSpacing``,
  ``gridGlobalOffset``, ``gridUnitSI``, ``dataOrder``, ``geometry``,
  ``unitDimension``, ``unitSI``, ``position``) must be identical.

The HDF5 reader uses h5py only.  BP5 needs ``openpmd_api``; when that module is
missing the tool says so instead of silently skipping the payload check.

Examples::

    tools/validate/compare_checkpoints.py checkpoints/checkpoint_000008.h5 out.h5
    tools/validate/compare_checkpoints.py checkpoints/checkpoint_000008.h5 out.bp
    tools/validate/compare_checkpoints.py --self-test

Exit codes: 0 equal, 1 mismatch, 2 unreadable input.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - reported at run time
    h5py = None

GROUPS = (("E", "x"), ("E", "y"), ("E", "z"),
          ("B", "x"), ("B", "y"), ("B", "z"))

RECORD_ATTRS = ("axisLabels", "gridSpacing", "gridGlobalOffset", "gridUnitSI",
                "dataOrder", "geometry", "unitDimension")
COMPONENT_ATTRS = ("unitSI", "position")
ITERATION_ATTRS = ("time", "dt", "timeUnitSI", "timeOffset")


class Unreadable(Exception):
    """The input cannot be opened or does not have the expected layout."""


def _decode(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.bytes_):
        return value.tobytes().decode()
    return value


def _same_meta(a, b) -> bool:
    a, b = _decode(a), _decode(b)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return [_decode(v) for v in a] == [_decode(v) for v in b]
    return a == b


def _same_payload(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return a.tobytes(order="C") == b.tobytes(order="C")


def _first_iteration(group):
    data = group.get("data")
    if data is None:
        raise Unreadable("missing /data group")
    keys = [k for k in data.keys() if k.lstrip("-").isdigit()]
    if not keys:
        raise Unreadable("no numeric iteration under /data")
    return data[sorted(keys, key=int)[0]]


def _read_hdf5(path):
    if h5py is None:
        raise Unreadable("h5py is not installed")
    if not os.path.isfile(path):
        raise Unreadable(f"not a file: {path}")
    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise Unreadable("missing /data group")
        iteration = _first_iteration(handle)
        base = "meshes" if "meshes" in iteration else "fields"
        if base not in iteration:
            raise Unreadable("iteration has neither meshes/ nor fields/")
        out = {"payload": {}, "records": {}, "components": {},
               "iteration": {}}
        for name in ITERATION_ATTRS:
            if name in iteration.attrs:
                out["iteration"][name] = iteration.attrs[name]
        for rec, comp in GROUPS:
            rgroup = iteration[base].get(rec)
            if rgroup is None:
                raise Unreadable(f"missing record {rec}")
            if comp not in rgroup:
                raise Unreadable(f"missing component {rec}/{comp}")
            ds = rgroup[comp]
            out["payload"][(rec, comp)] = np.asarray(ds[...])
            out["records"][rec] = {name: rgroup.attrs[name]
                                   for name in RECORD_ATTRS
                                   if name in rgroup.attrs}
            out["components"][(rec, comp)] = {name: ds.attrs[name]
                                               for name in COMPONENT_ATTRS
                                               if name in ds.attrs}
        return out


def _read_bp5(path):
    try:
        import openpmd_api as io
    except ImportError as error:
        raise Unreadable(
            "BP5 input needs openpmd_api (python -m pip install openpmd-api)"
        ) from error
    series = io.Series(path, io.Access.read_only)
    try:
        iterations = list(series.iterations.values())
        if not iterations:
            raise Unreadable("BP5 series has no iteration")
        iteration = iterations[0]
        out = {"payload": {}, "records": {}, "components": {},
               "iteration": {}}
        for name in ITERATION_ATTRS:
            if hasattr(iteration, name):
                out["iteration"][name] = getattr(iteration, name)
        for rec, comp in GROUPS:
            rgroup = iteration.meshes[rec]
            ds = rgroup[comp]
            out["payload"][(rec, comp)] = np.asarray(ds.load_chunk())
            out["records"][rec] = {
                name: getattr(rgroup, name)
                for name in RECORD_ATTRS if hasattr(rgroup, name)
            }
            out["components"][(rec, comp)] = {
                name: getattr(ds, name)
                for name in COMPONENT_ATTRS if hasattr(ds, name)
            }
        series.flush()
        return out
    finally:
        series.flush()
        del series


def read_series(path):
    """Read a checkpoint/output into the comparison dictionary."""
    if os.path.isdir(path) or path.endswith(".bp"):
        return _read_bp5(path)
    return _read_hdf5(path)


def compare(reference, candidate):
    """Return a list of human-readable differences (empty when identical)."""
    problems = []
    for name, ref in reference["iteration"].items():
        got = candidate["iteration"].get(name, "<missing>")
        if not _same_meta(ref, got):
            problems.append(f"iteration {name}: {ref!r} != {got!r}")

    for rec in ("E", "B"):
        for name, ref in reference["records"].get(rec, {}).items():
            got = candidate["records"].get(rec, {}).get(name, "<missing>")
            if not _same_meta(ref, got):
                problems.append(f"record {rec} {name}: {ref!r} != {got!r}")

    for rec, comp in GROUPS:
        key = (rec, comp)
        if key not in reference["payload"] or key not in candidate["payload"]:
            problems.append(f"missing payload {rec}/{comp}")
            continue
        ref = reference["payload"][key]
        got = candidate["payload"][key]
        if not _same_payload(ref, got):
            detail = ""
            if ref.shape != got.shape:
                detail = f" shape {ref.shape} != {got.shape}"
            elif ref.dtype != got.dtype:
                detail = f" dtype {ref.dtype} != {got.dtype}"
            else:
                diff = np.abs(ref.astype(np.float64) - got.astype(np.float64))
                detail = f" max|diff|={float(np.nanmax(diff)):.3e}"
            problems.append(f"payload {rec}/{comp}: not bitwise equal{detail}")
        for name, ref_value in reference["components"].get(key, {}).items():
            got_value = candidate["components"].get(key, {}).get(
                name, "<missing>")
            if not _same_meta(ref_value, got_value):
                problems.append(
                    f"component {rec}/{comp} {name}: "
                    f"{ref_value!r} != {got_value!r}")
    return problems


def _write_h5(path, base, arrays, records, components):
    with h5py.File(path, "w") as handle:
        iteration = handle.create_group("data/0")
        iteration.attrs["time"] = np.float64(2.5e-13)
        iteration.attrs["dt"] = np.float64(1.0e-15)
        iteration.attrs["timeUnitSI"] = np.float64(1.0)
        iteration.attrs["timeOffset"] = np.float64(0.0)
        for rec in ("E", "B"):
            rgroup = iteration.create_group(f"{base}/{rec}")
            for name, value in records[rec].items():
                rgroup.attrs[name] = value
            for comp in "xyz":
                ds = rgroup.create_dataset(comp, data=arrays[(rec, comp)])
                for name, value in components[(rec, comp)].items():
                    ds.attrs[name] = value


def _synthetic(directory, perturb=False):
    record_attrs = {
        "axisLabels": np.array([b"z", b"y", b"x"]),
        "gridSpacing": np.array([2.0e-7, 2.0e-7, 2.0e-7]),
        "gridGlobalOffset": np.array([0.0, 0.0, 0.0]),
        "gridUnitSI": np.float64(1.0),
        "dataOrder": np.bytes_(b"C"),
        "geometry": np.bytes_(b"cartesian"),
        "unitDimension": np.array([1.0, 1.0, -3.0, -1.0, 0.0, 0.0, 0.0]),
    }
    positions = {
        ("E", "x"): [0.0, 0.0, 0.5], ("E", "y"): [0.0, 0.5, 0.0],
        ("E", "z"): [0.5, 0.0, 0.0], ("B", "x"): [0.5, 0.5, 0.0],
        ("B", "y"): [0.5, 0.0, 0.5], ("B", "z"): [0.0, 0.5, 0.5],
    }
    arrays = {}
    components = {}
    for ci, (rec, comp) in enumerate(GROUPS):
        values = (np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
                  + np.float32(0.125 * ci))
        arrays[(rec, comp)] = values
        components[(rec, comp)] = {
            "unitSI": np.float64(1.0 + ci),
            "position": np.array(positions[(rec, comp)]),
        }
    checkpoint = os.path.join(directory, "checkpoint.h5")
    _write_h5(checkpoint, "fields", arrays, {"E": record_attrs, "B": record_attrs},
              components)
    if perturb:
        out_arrays = {k: v.copy() for k, v in arrays.items()}
        out_arrays[("E", "y")][1, 2, 3] = np.float32(12345.0)
    else:
        out_arrays = arrays
    output = os.path.join(directory, "output.h5")
    _write_h5(output, "meshes", out_arrays, {"E": record_attrs, "B": record_attrs},
              components)
    return checkpoint, output


def self_test() -> int:
    print("compare_checkpoints self-test")
    with tempfile.TemporaryDirectory(prefix="picbunch-compare-") as directory:
        checkpoint, same = _synthetic(directory)
        problems = compare(read_series(checkpoint), read_series(same))
        if problems:
            print("  FAIL: identical pair reported differences:", file=sys.stderr)
            for problem in problems:
                print(f"    {problem}", file=sys.stderr)
            return 1
        print("  identical pair: PASS (0 differences)")

        probe = os.path.join(directory, "changed.h5")
        perturbed_dir = os.path.join(directory, "perturbed")
        os.makedirs(perturbed_dir)
        _, changed = _synthetic(perturbed_dir, perturb=True)
        os.replace(changed, probe)
        problems = compare(read_series(checkpoint), read_series(probe))
        if not problems:
            print("  FAIL: perturbed pair was not detected", file=sys.stderr)
            return 1
        if not any("payload E/y" in problem for problem in problems):
            print("  FAIL: perturbation was not reported as a payload diff",
                  file=sys.stderr)
            for problem in problems:
                print(f"    {problem}", file=sys.stderr)
            return 1
        print(f"  perturbed pair: PASS ({len(problems)} difference(s) found)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bitwise payload and metadata comparison of an HDF5 "
                    "restart checkpoint against an HDF5 or BP5 field output.")
    parser.add_argument("checkpoint", nargs="?",
                        help="HDF5 checkpoint (fields/ under data/<step>)")
    parser.add_argument("output", nargs="?",
                        help="field output, HDF5 or BP5 (meshes/)")
    parser.add_argument("--self-test", action="store_true",
                        help="create and delete a tiny synthetic pair")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if not args.checkpoint or not args.output:
        parser.error("give checkpoint and output, or --self-test")

    try:
        reference = read_series(args.checkpoint)
        candidate = read_series(args.output)
    except (Unreadable, OSError) as error:
        print(f"compare_checkpoints: {error}", file=sys.stderr)
        return 2

    problems = compare(reference, candidate)
    if problems:
        for problem in problems:
            print(f"DIFF {problem}")
        print(f"compare_checkpoints: {len(problems)} difference(s)")
        return 1
    print("compare_checkpoints: bitwise payload and metadata identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
