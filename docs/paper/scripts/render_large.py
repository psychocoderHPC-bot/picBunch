#!/usr/bin/env python3
"""Large-domain slices (LARGE1, 128x128x64), figure family ``large_*``.

LARGE1 is the axial-x data set: gamma=10 propagates along +x, gamma=1 is
stationary.  The x-y cut is therefore plotted with the beam axis x horizontal
and y vertical, and the z-y cut with y horizontal and z vertical (the beam
axis is out of that plane).  This is the orientation of the committed paper
figures.  :mod:`axis` records the stored layouts and the correct display
mapping, including the fact that LARGE1 stores the z-y slice as ``[z, y]``
(unlike BDATA2/HIRES, which store ``[y, z]``).

Each dataset is one figure with two panels stacked in the electronic
appendix: the x-y cut and the z-y cut.  Every panel carries the electric
block first (field row, error row) and the magnetic block second (field row,
error row), matching the per-gamma ``gp_*`` convention.  The stationary
gamma=1 case shows the magnetic field row (identically zero, blank panels)
and omits the magnetic error row, whose relative error is undefined.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import axis  # noqa: E402
import cases  # noqa: E402
import figstyle as fs  # noqa: E402
import ingest_measurements as ingest  # noqa: E402

US = 1e6
DATASET = "large1"
COMP = {"E": ["Ex", "Ey", "Ez"], "B": ["Bx", "By", "Bz"]}


def _items(d, plane, part, source, gamma, magnitude=True):
    comps = COMP[part]
    items = []
    refs = [axis.to_display(d[f"{plane}_{c}_ref"], DATASET, plane, source)
            for c in comps]
    valid = ingest.part_valid(refs)
    # An identically-zero component is blank: along the +x beam axis for
    # gamma=10 (B_x) or everywhere for the stationary gamma=1 magnetic field.
    reason = "beam axis" if gamma == 10.0 else "stationary"
    for c in comps:
        axis.check_stored_shape(DATASET, plane, d[f"{plane}_{c}_pic"], d["n"],
                                source=source)
        ref = axis.to_display(d[f"{plane}_{c}_ref"], DATASET, plane, source)
        arr = axis.to_display(d[f"{plane}_{c}_pic"], DATASET, plane, source)
        it = {
            "arr": arr,
            "err": np.abs(axis.to_display(d[f"{plane}_{c}_err"], DATASET,
                                          plane, source)),
            "valid": valid,
            "title": rf"${c[0]}_{{{c[-1]}}}$",
            "symbol": f"{c[0]}_{{{c[-1]}}}",
        }
        if float(np.max(np.abs(ref))) == 0.0:
            # The measured array may hold round-off at the 1e-19 T level for a
            # physically zero component.  Draw a blank panel, not that noise.
            it["arr"] = np.zeros_like(arr)
            it["blank"] = True
            it["blank_text"] = rf"${it['symbol']}\equiv0$"
            it["annotation"] = (
                rf"$\mathbf{{{it['symbol']}}}\equiv0$"
                f" ({reason}); relative error undefined")
        items.append(it)
    if magnitude:
        pic2 = sum(axis.to_display(d[f"{plane}_{c}_pic"], DATASET, plane,
                                   source) ** 2 for c in comps)
        ref2 = sum(axis.to_display(d[f"{plane}_{c}_ref"], DATASET, plane,
                                   source) ** 2 for c in comps)
        arr = np.maximum(pic2, 0.0)
        peak = float(np.max(ref2)) if np.max(ref2) > 0 else 0.0
        if peak > 0:
            err = np.abs(pic2 - ref2) / peak
        else:
            err = np.full_like(arr, np.nan)
        sq = {"arr": arr, "err": err, "valid": valid,
              "title": rf"${part}^2$"}
        if peak == 0.0:
            sq["arr"] = np.zeros_like(arr)
            sq["blank"] = True
            sq["blank_text"] = rf"${part}^2\equiv0$"
        items.append(sq)
    return items


def _signature(d, plane, source):
    """Magnetostatic signature: Bx==0 => +x; all B==0 => stationary."""
    def mx(c):
        return float(np.max(np.abs(d[f"{plane}_{c}_ref"])))
    return {"Bx_max": mx("Bx"), "By_max": mx("By"), "Bz_max": mx("Bz"),
            "Ex_max": mx("Ex"), "Ey_max": mx("Ey")}


def _magnitude(d, plane, source):
    e2 = sum(axis.to_display(d[f"{plane}_{c}_pic"], DATASET, plane,
                             source) ** 2 for c in COMP["E"])
    return np.sqrt(np.maximum(e2, 0.0))


def build(outdir, case, plane, source):
    d = np.load(source, allow_pickle=True)
    n = [int(v) for v in d["n"]]
    cell = np.asarray(d["cell"], float)
    origin = np.asarray(d["origin"], float)
    horiz_e, vert_e = axis.coords(DATASET, plane, n, cell, origin,
                                  centres=False, source=source)
    gamma = 10.0 if case == "g10" else 1.0
    rows = []
    for part in ("E", "B"):
        items = _items(d, plane, part, source, gamma)
        rows.append({"kind": "field", "part": part, "items": items})
        if part == "B" and gamma == 1.0:
            # B is identically zero at beta=0, so the relative B error is
            # undefined and the whole error row would be fully masked.  The
            # per-gamma figures omit it for the same reason.
            continue
        err_items = [dict(it) for it in items]
        for it in err_items:
            it["title"] = it["title"] + " err"
        rows.append({"kind": "error", "part": part, "items": err_items})
    name = f"large_s{plane}_{case}"
    xl, yl = axis.axis_labels(DATASET, plane, source)
    c = np.asarray(d["centre"], float) * US
    beam_position = {"x": float(c[0]), "y": float(c[1]), "z": float(c[2])}
    vert_name, horiz_name = axis.display_axes(DATASET, plane, source)
    man = {"source_data": os.path.relpath(source, cases.REPO_ROOT),
           "case": case, "dataset": DATASET, "plane": plane,
           "parts": ["E", "B"],
           "block_order": "E field, E error, B field, B error"
                          " (B error omitted for the stationary case)",
           "n": n, "cell_m": [float(v) for v in d["cell"]],
           "includegraphics_width": 1.0,
           "axis_convention": {"propagation_axis": axis.propagation_axis(
                                   DATASET, gamma),
                               "horizontal": horiz_name, "vertical": vert_name,
                               "stored_axes": list(axis.stored_axes(
                                   DATASET, plane, source)),
                               "display_axes": list(axis.display_axes(
                                   DATASET, plane, source))}}
    fs.render_split_field_rows(
        name, outdir, rows, horiz_e * US, vert_e * US,
        xlabel=xl, ylabel=yl, width_frac=1.0,
        shading="flat", compact=True, manifest=man)
    horiz_c, vert_c = axis.coords(DATASET, plane, n, cell, origin,
                                  centres=True, source=source)
    signature = _signature(d, plane, source)
    if gamma == 10.0:
        if not (signature["Bx_max"] == 0.0 and signature["By_max"] > 0.0
                and signature["Bz_max"] > 0.0):
            raise axis.AxisConventionError(
                f"{name}: expected the axial-x signature Bx==0, By,Bz>0; "
                f"got {signature}")
    else:
        if not all(signature[k] == 0.0 for k in ("Bx_max", "By_max", "Bz_max")):
            raise axis.AxisConventionError(
                f"{name}: expected a stationary signature (all B==0); "
                f"got {signature}")
    return axis.assert_beam_axis(
        DATASET, plane, _magnitude(d, plane, source), horiz_c * US,
        vert_c * US, n=n, cell=cell, origin=origin, gamma=gamma,
        centres=True, source=source, figure=name, label="|E|",
        beam_position=beam_position, signature=signature)


def run(outdir):
    results = []
    for case in ("g10", "g1"):
        source = cases.data("large1", f"slices_{case}.npz")
        for plane in ("xy", "zy"):
            results.append(build(outdir, case, plane, source))
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    for r in run(a.out_dir):
        print("axis-check", r["figure"], "PASS" if r["pass"] else "FAIL")
