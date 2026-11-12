#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
"""Reduce the line-family precision cases (F_w4b, F_gamma, F_multivel).

Three modes, selected by ``spec["mode"]``:

``w4b``
    The paper metric on a single-electron mesh: worst
    ``|ref-mesh|/max(|ref|,|mesh|)`` over the electric components, per-component
    floor ``|ref| >= 1e-3 peak``, at the ring distances.
``gamma``
    Per-ring worst peak-normalised absolute error ``|mesh-ref|/peak(comp)``.
``multivel``
    Field difference of every bin count against the ``B=256`` run, plus the
    model columns and the reduced field/particle arrays.

The reference values come from the sample-mode exact driver, already scaled
back to SI here.  Heavy arrays stay on the remote node: only the CSV/panel/
reduced arrays are written.
"""
import argparse
import csv
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
YEE = np.asarray([(0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5),
                  (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)])
REL_FLOOR = 1e-3


def load_field(path):
    out = {}
    with h5py.File(path, "r") as handle:
        meshes = handle["data/0/meshes"]
        for part in ("E", "B"):
            for axis in "xyz":
                dataset = meshes[part][axis]
                out[part + axis] = (np.asarray(dataset[:], dtype=np.float64)
                                    * float(dataset.attrs.get("unitSI", 1.0)))
    return out


def load_ref(path):
    data = np.load(path, allow_pickle=False)
    return (np.asarray(data["point_m"], dtype=float),
            np.asarray(data["comp"], dtype=int),
            np.asarray(data["d_cells"], dtype=float),
            np.asarray(data["vals"], dtype=float),
            int(data["order"]) if "order" in data.files else -1)


def load_bunch(path):
    """Stored-order particle positions (m) and signed charges (C)."""
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


def bunch_block(input_h5, dataset, gamma, cell):
    """Additive bunch keys for the particles that produced the field."""
    cell = np.asarray(cell, dtype=float)
    if cell.size != 3 or not np.allclose(cell, cell[0], rtol=0.0, atol=0.0):
        raise ValueError("bunch keys need a cubic field cell, got %s"
                         % list(cell))
    pos, q = load_bunch(input_h5)
    return {
        "bunch_pos_m": pos,
        "bunch_q_C": q,
        "bunch_axis": np.asarray(
            axis_convention.propagation_axis(dataset, gamma)),
        "bunch_rho_cell_m": np.asarray(float(cell[0])),
        "bunch_gamma": np.asarray(float(gamma)),
    }


def sample_mesh(mesh, point_m, comp, cell):
    """Evaluate the picBunch staggered mesh at the reference sample points."""
    scale_e = E_CHARGE / (4.0 * math.pi * EPS0 * float(cell[0]) ** 2)
    node = (point_m - YEE[comp] * np.asarray(cell)[None, :]) / np.asarray(cell)[None, :]
    index = np.rint(node).astype(np.int64)
    got = np.full(point_m.shape[0], np.nan)
    for c, name in enumerate(COMP):
        mask = comp == c
        if not np.any(mask):
            continue
        ijk = index[mask]
        cube = mesh[name]
        nz, ny, nx = cube.shape
        ok = ((ijk[:, 0] >= 0) & (ijk[:, 0] < nx)
              & (ijk[:, 1] >= 0) & (ijk[:, 1] < ny)
              & (ijk[:, 2] >= 0) & (ijk[:, 2] < nz))
        idx = np.where(mask)[0]
        got[idx[ok]] = cube[ijk[ok, 2], ijk[ok, 1], ijk[ok, 0]]
    return got, scale_e


def ref_si(vals, comp):
    scale_e = E_CHARGE / (4.0 * math.pi * EPS0 * 1e-14)
    return vals


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def mode_w4b(spec):
    rows = []
    for unit in spec["units"]:
        mesh = load_field(unit["pic"])
        point_m, comp, d_cells, vals, _ = load_ref(unit["ref"])
        cell = np.asarray(unit["cell"], dtype=float)
        scale_e = E_CHARGE / (4.0 * math.pi * EPS0 * float(cell[0]) ** 2)
        si = np.where(comp < 3, vals * scale_e, vals * scale_e / C_LIGHT)
        got, _ = sample_mesh(mesh, point_m, comp, cell)
        peaks = np.asarray([np.max(np.abs(si[comp == c])) if np.any(comp == c) else 0.0
                            for c in range(6)])
        for ring in unit["rings"]:
            mask = np.isclose(d_cells, ring, atol=0.35)
            worst = 0.0
            for c in (0, 1, 2):
                if peaks[c] <= 0.0:
                    continue
                sel = (mask & (comp == c) & np.isfinite(got)
                       & (np.abs(si) >= REL_FLOOR * peaks[c]))
                if not np.any(sel):
                    continue
                rel = np.abs(got[sel] - si[sel]) / np.maximum(np.abs(si[sel]),
                                                              np.abs(got[sel]))
                worst = max(worst, float(np.max(rel)))
            rows.append((unit["gamma"], unit["aspect"], "single", int(ring), worst,
                         "precision campaign: paper metric |ref-mesh|/max, ExEyEz, "
                         "per-comp floor, exact driver order %d"
                         % unit.get("order", 12)))
    header = ["gamma", "aspect", "kind", "d", "worst_rel_error", "source"]
    write_csv(spec["out"], header, rows)
    write_csv(spec["ingest"], header, rows)
    return {"rows": len(rows)}


def mode_gamma(spec):
    rows = []
    for unit in spec["units"]:
        mesh = load_field(unit["pic"])
        point_m, comp, d_cells, vals, _ = load_ref(unit["ref"])
        cell = np.asarray(unit["cell"], dtype=float)
        scale_e = E_CHARGE / (4.0 * math.pi * EPS0 * float(cell[0]) ** 2)
        si = np.where(comp < 3, vals * scale_e, vals * scale_e / C_LIGHT)
        got, _ = sample_mesh(mesh, point_m, comp, cell)
        peaks = np.asarray([np.max(np.abs(si[comp == c])) if np.any(comp == c) else 0.0
                            for c in range(6)])
        pmax = float(np.max(peaks))
        for ring in unit["rings"]:
            mask = np.isclose(d_cells, ring, atol=0.35)
            worst = 0.0
            for c in range(6):
                if peaks[c] <= 1e-9 * pmax:
                    continue
                sel = mask & (comp == c) & np.isfinite(got)
                if not np.any(sel):
                    continue
                worst = max(worst, float(np.max(np.abs(got[sel] - si[sel]) / peaks[c])))
            rows.append((unit["gamma"], unit["h"], int(ring), worst))
    header = ["gamma", "h", "d", "peak_normalised_abs_error"]
    write_csv(spec["out"], header, rows)
    write_csv(spec["ingest"], header, rows)
    return {"rows": len(rows)}


def bunch_stats(path):
    with h5py.File(path, "r") as handle:
        species = handle["data/0/particles/e"]
        u = []
        for axis in "xyz":
            node = species["momentum"][axis]
            u.append(np.asarray(node[:], dtype=np.float64)
                     * float(node.attrs.get("unitSI", 1.0))
                     / (9.1093837015e-31 * C_LIGHT))
        weight = species["weighting"]["value"]
        w = np.asarray(weight[:], dtype=np.float64) * float(weight.attrs.get("unitSI", 1.0))
        charge = species["charge"]["value"]
        c = np.asarray(charge[:], dtype=np.float64) * float(charge.attrs.get("unitSI", 1.0))
    u = np.stack(u, axis=1)
    gamma = np.sqrt(1.0 + np.sum(u * u, axis=1))
    upar = u[:, 1]
    theta = np.arctan2(np.sqrt(u[:, 0] ** 2 + u[:, 2] ** 2), u[:, 1])
    return {"u_parallel_rel_std": float(np.std(upar) / np.mean(upar)),
            "gamma_mean": float(np.mean(gamma)),
            "theta_std_rad": float(np.std(theta)),
            "n": int(len(upar)),
            "total_charge_C": float(np.sum(w * c))}


def mode_multivel(spec):
    ref_bins = int(spec["ref_bins"])
    fields = {
        "e5": {int(b): load_field(p) for b, p in spec["fields_e5"].items()},
        "a5": {int(b): load_field(p) for b, p in spec["fields_a5"].items()},
    }
    stats = {"e5": bunch_stats(spec["input_e5"]),
             "a5": bunch_stats(spec["input_a5"])}
    gamma = float(spec.get("gamma", 10.0))
    cell_vec = np.full(3, float(spec["cell_m"]))
    bunch = {case: bunch_block(spec["input_" + case], "multivel", gamma,
                               cell_vec)
             for case in ("e5", "a5")}
    peaks = {}
    y = {}
    for case in ("e5", "a5"):
        ref = fields[case][ref_bins]
        ref_peak = {c: float(np.max(np.abs(ref[c]))) for c in COMP}
        den = max(ref_peak.values())
        peaks[case] = {str(ref_bins): ref_peak}
        y[case] = {}
        for b in spec["bins"]:
            field = fields[case][b]
            peak = {c: float(np.max(np.abs(field[c]))) for c in COMP}
            peaks[case][str(b)] = peak
            num = max(float(np.max(np.abs(field[c] - ref[c]))) for c in COMP)
            y[case][b] = num / den if den > 0 else 0.0
            if b in (16, 64):
                out = os.path.join(spec["fields_out"],
                                   "multivel_fields_%s_b%d.npz" % (case, b))
                os.makedirs(os.path.dirname(out), exist_ok=True)
                np.savez_compressed(
                    out,
                    Ex=field["Ex"], Ey=field["Ey"], Ez=field["Ez"],
                    Bx=field["Bx"], By=field["By"], Bz=field["Bz"],
                    array_axes="z,y,x", cell_m=float(spec["cell_m"]),
                    grid=np.asarray(spec["grid"]), case=case, bins=int(b),
                    charge_C=stats[case]["total_charge_C"], beam_axis="+y",
                    metric="field B vs B=256 peak-normalised",
                    **bunch[case])
    rows = []
    for b in spec["bins"]:
        rows.append((b,
                     stats["e5"]["u_parallel_rel_std"] / b,
                     y["e5"][b],
                     stats["a5"]["gamma_mean"] * stats["a5"]["theta_std_rad"] / math.sqrt(b),
                     y["a5"][b]))
    header = ["bin_count", "m_energy", "y_energy", "m_angular", "y_angular"]
    write_csv(os.path.join(spec["ingest_dir"], "multivel_binning.csv"), header, rows)
    write_csv(spec["out"], header, rows)

    peaks_rows = []
    for case in ("e5", "a5"):
        for b in spec["bins"] + [ref_bins]:
            p = peaks[case][str(b)]
            peaks_rows.append((case, b, p["Bz"], p["By"], y[case].get(b, 0.0)))
    write_csv(os.path.join(spec["ingest_dir"], "multivel_peaks.csv"),
              ["case", "bin_count", "peak_Bz_T", "peak_By_T", "y"], peaks_rows)

    for case in ("e5", "a5"):
        data = np.load(spec["panel_" + case], allow_pickle=False)
        targets = [os.path.join(spec["ingest_dir"],
                                "multivel_energy.npz" if case == "e5"
                                else "multivel_angular.npz"),
                   spec["panel_out_" + case]]
        for target in targets:
            os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
            np.savez(target, particles=data["particles"],
                     cell_m=float(spec["cell_m"]), **bunch[case])
    return {"rows": len(rows), "panels": 2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    with open(args.spec) as handle:
        spec = json.load(handle)
    mode = spec["mode"]
    if mode == "w4b":
        result = mode_w4b(spec)
    elif mode == "gamma":
        result = mode_gamma(spec)
    elif mode == "multivel":
        result = mode_multivel(spec)
    else:
        raise SystemExit("unknown mode " + mode)
    if spec.get("summary"):
        os.makedirs(os.path.dirname(os.path.abspath(spec["summary"])), exist_ok=True)
        with open(spec["summary"], "w") as handle:
            json.dump(result, handle, indent=1)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
