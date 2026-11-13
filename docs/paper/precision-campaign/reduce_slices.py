#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
"""Reduce one picBunch field plus its independent reference to paper npz.

The script reads the openPMD/HDF5 field that picBunch wrote, the sample-mode
output of ``reference/exact_field_driver`` (component-major raw doubles) and
emits two artefacts:

* ``--out``: the reduced npz in exactly the schema the paper renderers consume
  (one of the ``hires``/``bdata2``/``large1``/``var1b`` layouts, validated by
  ``ingest_measurements.required_npz_keys``);
* ``--ingest``: the source export that the existing
  ``docs/paper/scripts/ingest_measurements.py`` map expects for the
  ``field_export`` cases (hires, bdata2).  For ``large1``/``var1b`` the map
  uses a ``field_h5`` source, so the caller copies the run file and the sparse
  six-component reference instead.

All heavy work (plane extraction, error/valid masks, E^2/B^2) happens on the
remote node; only the small 2-D arrays and metadata leave it.

The reference driver writes Slots in its normalised convention
``E_n = 4*pi*eps0*E/q`` and ``B_n = 4*pi*eps0*c*B/q``; multiplying by
``q/(4*pi*eps0*dx^2)`` (and ``/c`` for B) gives SI V/m and T.  The driver
assumes cubic cells, which every precision case uses.
"""
import argparse
import json
import math
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(os.path.dirname(HERE), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import axis as axis_convention  # noqa: E402

COMP = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]
BUNCH_KEYS = ("bunch_pos_m", "bunch_q_C", "bunch_axis",
              "bunch_rho_cell_m", "bunch_gamma")
E_CHARGE = 1.602176634e-19
EPS0 = 8.8541878128e-12
C_LIGHT = 299792458.0
REL_FLOOR = 1e-3


def load_field(path):
    """Return ``comp -> (nz, ny, nx)`` float64 arrays in SI units."""
    out = {}
    with h5py.File(path, "r") as handle:
        meshes = handle["data/0/meshes"]
        for part in ("E", "B"):
            for axis in "xyz":
                dataset = meshes[part][axis]
                out[part + axis] = (np.asarray(dataset[:], dtype=np.float64)
                                    * float(dataset.attrs.get("unitSI", 1.0)))
    return out


def load_sources(path, cell_m, columns):
    """Read particle positions from the openPMD input, in cell units."""
    with h5py.File(path, "r") as handle:
        species = handle["data/0/particles/e"]
        comps = []
        for axis in "xyz":
            node = species["position"][axis]
            comps.append(np.asarray(node[:], dtype=np.float64)
                         * float(node.attrs.get("unitSI", 1.0)))
        pos = np.stack(comps, axis=1)
        if columns == 7:
            mom = []
            for axis in "xyz":
                node = species["momentum"][axis]
                mom.append(np.asarray(node[:], dtype=np.float64)
                           * float(node.attrs.get("unitSI", 1.0))
                           / (9.1093837015e-31 * C_LIGHT))
            weight = species["weighting"]["value"]
            w = np.asarray(weight[:], dtype=np.float64) * float(weight.attrs.get("unitSI", 1.0))
            return np.column_stack([pos / np.asarray(cell_m)[None, :],
                                    np.stack(mom, axis=1), w])
    return pos / np.asarray(cell_m)[None, :]


def load_bunch(path):
    """Return the stored-order particle states of the run input.

    ``pos_m`` is ``(N, 3)`` float64 in metres and ``q_C`` is ``(N,)`` float64
    signed charge in coulomb.  The order is the on-disk order of the
    generated openPMD input, so the arrays are index aligned and unsorted.
    """
    with h5py.File(path, "r") as handle:
        species = handle["data/0/particles/e"]
        pos = []
        for axis in "xyz":
            node = species["position"][axis]
            pos.append(np.asarray(node[:], dtype=np.float64)
                       * float(node.attrs.get("unitSI", 1.0)))
        weight = species["weighting"]["value"]
        w = np.asarray(weight[:], dtype=np.float64) * float(
            weight.attrs.get("unitSI", 1.0))
        charge = species["charge"]["value"]
        c = np.asarray(charge[:], dtype=np.float64) * float(
            charge.attrs.get("unitSI", 1.0))
    pos = np.stack(pos, axis=1).astype(np.float64)
    q = np.asarray(w * c, dtype=np.float64)
    if pos.shape[0] != q.shape[0]:
        raise ValueError("%s: %d positions but %d charges"
                         % (path, pos.shape[0], q.shape[0]))
    return pos, q


def bunch_block(spec, dataset, gamma, cell):
    """Additive bunch-form keys for the particles that produced the field.

    ``bunch_rho_cell_m`` must be the field-grid cell size, so the helper
    refuses a non-cubic cell vector instead of guessing.
    """
    cell = np.asarray(cell, dtype=float)
    if cell.size != 3 or not np.allclose(cell, cell[0], rtol=0.0, atol=0.0):
        raise ValueError("%s: bunch keys need a cubic field cell, got %s"
                         % (spec.get("case"), list(cell)))
    pos, q = load_bunch(spec["input_h5"])
    return {
        "bunch_pos_m": pos,
        "bunch_q_C": q,
        "bunch_axis": np.asarray(
            axis_convention.propagation_axis(dataset, gamma)),
        "bunch_rho_cell_m": np.asarray(float(cell[0])),
        "bunch_gamma": np.asarray(float(gamma)),
    }


def reference_cube(ref_npz, n, cell, origin):
    """Scatter the sample-mode reference into a sparse ``comp -> [nz,ny,nx]`` cube."""
    data = np.load(ref_npz, allow_pickle=False)
    ijk = data["ijk"]
    comps = data["comp"]
    vals = data["vals"].astype(np.float64)
    scale_e = E_CHARGE / (4.0 * math.pi * EPS0 * float(cell[0]) ** 2)
    cube = {c: np.zeros((n[2], n[1], n[0]), dtype=np.float64) for c in COMP}
    scaled = np.where(comps < 3, vals * scale_e, vals * scale_e / C_LIGHT)
    for c, name in enumerate(COMP):
        mask = comps == c
        if not np.any(mask):
            continue
        index = ijk[mask]
        cube[name][index[:, 2], index[:, 1], index[:, 0]] = scaled[mask]
    return cube


def part_valid(refs):
    peaks = [float(np.max(np.abs(r))) if r.size else 0.0 for r in refs]
    peak = max(peaks) if peaks else 0.0
    if peak <= 0.0:
        return np.zeros(np.asarray(refs[0]).shape, dtype=bool)
    mag2 = np.zeros_like(np.asarray(refs[0], dtype=float))
    for ref in refs:
        mag2 = mag2 + np.asarray(ref, dtype=float) ** 2
    return np.sqrt(np.maximum(mag2, 0.0)) >= REL_FLOOR * peak


def component_valid(ref):
    peak = float(np.max(np.abs(ref))) if np.asarray(ref).size else 0.0
    if peak <= 0.0:
        return np.zeros(np.asarray(ref).shape, dtype=bool)
    return np.abs(np.asarray(ref, dtype=float)) >= REL_FLOOR * peak


def error(pic, ref):
    peak = float(np.max(np.abs(ref))) if np.asarray(ref).size else 0.0
    if peak <= 0.0:
        return np.full(np.asarray(ref).shape, np.nan)
    return np.abs(np.asarray(pic, dtype=float) - np.asarray(ref, dtype=float)) / peak


def planes_of(cubes, style, plane, index):
    """Extract one plane in the *stored* layout recorded by ``axis.py``."""
    out = {}
    for name in COMP:
        cube = cubes[name]
        if plane == "xy":
            out[name] = np.array(cube[index, :, :], dtype=np.float64)
        elif style == "large1":
            out[name] = np.array(cube[:, :, index], dtype=np.float64)   # (z, y)
        else:
            out[name] = np.array(cube[:, :, index].T, dtype=np.float64)  # (y, z)
    return out


def _savez(path, arrays):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(path + ".tmp", path)


def build(spec):
    n = [int(v) for v in spec["n"]]
    cell = [float(v) for v in spec["cell"]]
    origin = [float(v) for v in spec.get("origin", [0.0, 0.0, 0.0])]
    style = spec["style"]
    comp = np.array(COMP, dtype="U3")
    out = {"n": np.asarray(n, dtype=np.int64),
           "cell": np.asarray(cell, dtype=float),
           "origin": np.asarray(origin, dtype=float),
           "comp": comp}
    if style == "hires":
        out["cell_m"] = np.asarray(float(cell[0]))
        out["array_axes_xy"] = np.asarray("y,x")
        out["offsets"] = np.asarray([(0.5, 0.0, 0.0), (0.0, 0.5, 0.0),
                                     (0.0, 0.0, 0.5), (0.0, 0.5, 0.5),
                                     (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)])
        out["display"] = np.asarray(
            "xy: rows=y cols=x; coords cell-centred um; per-component Yee "
            "offsets in 'offsets'")
        out["case"] = np.asarray(spec["case"])
        out["gamma"] = np.asarray(float(spec["gamma"]))
        beta = math.sqrt(max(0.0, 1.0 - 1.0 / (float(spec["gamma"]) ** 2)))
        out["beta"] = np.asarray(beta)
        out["uy"] = np.asarray(beta * float(spec["gamma"]))
        out["particle_pos_m"] = np.asarray(spec["particle_pos_m"], dtype=float)
        out["host"] = np.asarray(spec.get("host", ""))
        out["picbunch"] = np.asarray(spec.get("picbunch", ""))
        out["reference"] = np.asarray(spec.get("reference", ""))
        out["err_definition"] = np.asarray(
            "|pic-ref|/peak_ref(component,plane), valid where "
            "|ref|>=1e-3*peak_ref")
    else:
        out["case"] = np.asarray(spec["case"])
        out["display"] = np.asarray(spec.get("display", ""))
        out["ix"] = np.asarray(int(spec["plane_indices"].get("zy", n[0] // 2)),
                               dtype=np.int64)
        out["iy"] = np.asarray(int(spec.get("iy", n[1] // 2)), dtype=np.int64)
        out["iz"] = np.asarray(int(spec["plane_indices"].get("xy", n[2] // 2)),
                               dtype=np.int64)
    if style == "large1":
        out["array_axes_xy"] = np.asarray("y,x")
        out["array_axes_zy"] = np.asarray("z,y")
        out["centre"] = np.asarray(spec.get("centre", [0.0, 0.0, 0.0]), dtype=float)
    if style == "var1b":
        out["label"] = np.asarray(spec.get("label", ""))
    if style == "bdata2":
        out["pic_field_is_proxy"] = np.asarray(int(spec.get("proxy", 0)))
        out["pic_field_source"] = np.asarray(spec.get("pic_field_source", ""))

    pic = load_field(spec["pic"])
    ref = reference_cube(spec["ref"], n, cell, origin)
    if spec.get("zero_below") is not None:
        # An analytic reference can carry round-off where the physical
        # component is identically zero (stationary gamma=1 bunch).
        limit = float(spec["zero_below"])
        for name in COMP:
            ref[name][np.abs(ref[name]) < limit] = 0.0

    ingest = {"n": out["n"], "origin": out["origin"], "comp": comp}
    ingest["cell_m" if style == "hires" else "cell"] = (
        out["cell_m"] if style == "hires" else out["cell"])

    coords = {}
    for plane in spec["planes"]:
        index = int(spec["plane_indices"][plane])
        pic_plane = planes_of(pic, style, plane, index)
        ref_plane = planes_of(ref, style, plane, index)
        if style == "hires":
            valid = {name: component_valid(ref_plane[name]) for name in COMP}
        else:
            valid = {}
            for part in ("E", "B"):
                mask = part_valid([ref_plane[part + axis] for axis in "xyz"])
                for axis in "xyz":
                    valid[part + axis] = mask
        for name in COMP:
            pic_arr = pic_plane[name]
            ref_arr = ref_plane[name]
            err = error(pic_arr, ref_arr)
            if style in ("bdata2", "hires"):
                out[f"{plane}_{name}"] = pic_arr
                out[f"{plane}_ref_{name}"] = ref_arr
                out[f"{plane}_err_{name}"] = err
                out[f"{plane}_valid_{name}"] = valid[name]
                ingest[f"{plane}_{name}_pic"] = pic_arr
                ingest[f"{plane}_{name}_ref"] = ref_arr
            else:
                out[f"{plane}_{name}_pic"] = pic_arr
                out[f"{plane}_{name}_ref"] = ref_arr
                out[f"{plane}_{name}_err"] = err
                out[f"{plane}_{name}_valid"] = valid[name]
                ingest[f"{plane}_{name}_pic"] = pic_arr
                ingest[f"{plane}_{name}_ref"] = ref_arr
        if style == "hires" and plane == "xy":
            for part in ("E", "B"):
                names = [part + axis for axis in "xyz"]
                pic2 = sum(out[f"xy_{name}"] ** 2 for name in names)
                ref2 = sum(out[f"xy_ref_{name}"] ** 2 for name in names)
                out[f"xy_{part}2"] = pic2
                out[f"xy_ref_{part}2"] = ref2
                out[f"xy_{part}2_peak"] = np.asarray(float(np.max(ref2)))
                out[f"xy_ref_{part}2_peak"] = np.asarray(float(np.max(ref2)))
        if style in ("bdata2", "hires"):
            vert_axis, horiz_axis = (("y", "x") if plane == "xy" else ("y", "z"))
            axis_size = {"x": n[0], "y": n[1], "z": n[2]}
            horiz = (np.arange(axis_size[horiz_axis]) + 0.5) * cell[0] * 1e6
            vert = (np.arange(axis_size[vert_axis]) + 0.5) * cell[0] * 1e6
            if plane == "xy":
                out["xy_x_um"] = horiz
                out["xy_y_um"] = vert
            else:
                out["zy_z_um"] = horiz
                out["zy_y_um"] = vert

    if style == "hires":
        for key in ("case", "beta", "gamma", "uy", "particle_pos_m", "host",
                    "picbunch", "reference", "err_definition"):
            if key in out:
                ingest[key] = out[key]
    if style == "bdata2":
        ingest["sources"] = load_sources(spec["input_h5"], cell, 7)
    elif style == "var1b":
        out["sources"] = load_sources(spec["input_h5"], cell, 3)
        ingest["sources"] = out["sources"]
    elif style == "large1" and spec.get("centre") is None:
        pass

    gamma = float(spec.get("gamma", 10.0))
    block = bunch_block(spec, style, gamma, cell)
    out.update(block)
    ingest.update(block)

    _savez(spec["out"], out)
    if spec.get("ingest"):
        _savez(spec["ingest"], ingest)
    summary = {"case": spec.get("case"), "style": style,
               "planes": spec["planes"], "out": spec["out"],
               "ingest": spec["ingest"],
               "keys": sorted(out.keys())}
    if spec.get("summary"):
        os.makedirs(os.path.dirname(os.path.abspath(spec["summary"])), exist_ok=True)
        with open(spec["summary"], "w") as handle:
            json.dump(summary, handle, indent=1)
    print(json.dumps(summary))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    with open(args.spec) as handle:
        spec = json.load(handle)
    build(spec)


if __name__ == "__main__":
    main()
