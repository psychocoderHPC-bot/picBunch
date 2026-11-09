#!/usr/bin/env python3
"""Validation slices: BDATA2 forward/fluctuation and the VAR1b 1 mrad case.

Figure family: ``val_*``.  The beam axis (+y) is horizontal, x (x-y plane) or
z (z-y plane) is vertical; see :mod:`axis`.  No numeric value is changed from
the legacy FIGS_REPLOT/rebuild_valid.py renderer.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import axis  # noqa: E402
import cases  # noqa: E402
import figstyle as fs  # noqa: E402

US = 1e6
COMP = {"E": ["Ex", "Ey", "Ez"], "B": ["Bx", "By", "Bz"]}
ALL = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]


def _plane_data(dataset, d, plane, source):
    horiz, vert = axis.coords(dataset, plane, d["n"], d["cell"], d["origin"],
                              centres=True, source=source)
    out = {"horiz": horiz * US, "vert": vert * US, "plane": plane,
           "n": [int(v) for v in d["n"]],
           "cell": [float(v) for v in d["cell"]],
           "origin": [float(v) for v in d["origin"]]}
    for c in ALL:
        axis.check_stored_shape(dataset, plane, d[f"{plane}_{c}"], d["n"],
                                source=source)
        out[c] = {
            "pic": axis.to_display(d[f"{plane}_{c}"], dataset, plane, source),
            "ref": axis.to_display(d[f"{plane}_ref_{c}"], dataset, plane, source),
            "err": axis.to_display(d[f"{plane}_err_{c}"], dataset, plane, source),
            "valid": axis.to_display(d[f"{plane}_valid_{c}"], dataset, plane,
                                     source).astype(bool),
        }
    return out


def load_bdata(path):
    d = np.load(path, allow_pickle=False)
    return d, {p: _plane_data("bdata2", d, p, path) for p in ("xy", "zy")}


def sources_bdata(path):
    return np.load(path, allow_pickle=False)["sources"][:, :3]


def support_mask(plane, arr_shape, sources, plane_coord):
    nv, nh = arr_shape
    vc = np.arange(nv) + 0.5
    hc = np.arange(nh) + 0.5
    Hc, Vc = np.meshgrid(hc, vc)
    if plane == "xy":
        dx = sources[:, 0][:, None, None] - Vc[None]
        dy = sources[:, 1][:, None, None] - Hc[None]
        dz = sources[:, 2][:, None, None] - plane_coord
    else:
        dx = sources[:, 0][:, None, None] - plane_coord
        dy = sources[:, 1][:, None, None] - Hc[None]
        dz = sources[:, 2][:, None, None] - Vc[None]
    return np.any(dx * dx + dy * dy + dz * dz < 1.5 ** 2, axis=0)


def _magnitude(plane_data, use_ref):
    key = "ref" if use_ref else "pic"
    e2 = sum(plane_data[c][key] ** 2 for c in COMP["E"])
    return np.sqrt(np.maximum(e2, 0.0))


def panels(plane_data, part, field_mode):
    comps = COMP[part]
    items = []
    fields = []
    for c in comps:
        rec = plane_data[c]
        f = rec["ref"] if field_mode == "ref" else rec["pic"]
        fields.append(f)
        items.append({"arr": f, "err": rec["err"], "valid": rec["valid"],
                      "title": rf"${c[0]}_{{{c[1]}}}$",
                      "symbol": f"{c[0]}_{{{c[1]}}}"})
    f2 = sum(f * f for f in fields)
    r2 = sum(plane_data[c]["ref"] ** 2 for c in comps)
    arr = np.maximum(f2, 0.0)
    peak = float(np.max(r2)) if np.max(r2) > 0 else 0.0
    if peak > 0:
        err = np.abs(f2 - r2) / peak
        valid = r2 >= 1e-3 * peak
    else:
        err = np.full_like(arr, np.nan)
        valid = np.zeros_like(arr, dtype=bool)
    items.append({"arr": arr, "err": err, "valid": valid,
                  "title": rf"${part}^2$"})
    return items


def horizontal_axis_record(dataset, plane, source):
    vert, horiz = axis.display_axes(dataset, plane, source)
    return {"propagation_axis": axis.propagation_axis(dataset, 10.0),
            "horizontal": horiz, "vertical": vert,
            "stored_axes": list(axis.stored_axes(dataset, plane, source)),
            "display_axes": [vert, horiz]}


def make(outdir, name, plane_data, part, plane, sources, coord, has_error,
         dataset, source, beam_centroid, use_ref):
    field_items = panels(plane_data, part, "ref" if use_ref else "pic")
    rows = [{"kind": "field", "part": part, "items": field_items}]
    smask = None
    if has_error:
        rows.append({"kind": "error", "part": part,
                     "items": [dict(it) for it in field_items]})
        smask = support_mask(plane, field_items[0]["arr"].shape, sources, coord)
    xl, yl = axis.axis_labels(dataset, plane, source)
    man = {"source_data": os.path.relpath(source, cases.REPO_ROOT),
           "dataset": dataset, "plane": plane, "part": part,
           "has_error": has_error, "includegraphics_width": 0.92,
           "axis_convention": horizontal_axis_record(dataset, plane, source)}
    fs.render_split_field_rows(
        name, outdir, rows, plane_data["horiz"], plane_data["vert"],
        xlabel=xl, ylabel=yl + ("" if not has_error else "\n"),
        width_frac=0.92, fig_height=3.25 if has_error else 1.85,
        support=smask, manifest=man)
    return axis.assert_beam_axis(
        dataset, plane, _magnitude(plane_data, use_ref),
        plane_data["horiz"], plane_data["vert"], n=plane_data["n"],
        cell=plane_data["cell"], origin=plane_data["origin"], gamma=10.0,
        centres=True, source=source, figure=name, label="|E|",
        beam_position=beam_centroid)


def _beam_centroid_um(sources_cells, cell):
    c = np.asarray(sources_cells, float).mean(axis=0) * np.asarray(cell) * US
    return {"x": float(c[0]), "y": float(c[1]), "z": float(c[2])}


def run(outdir):
    """Render every val_* figure and return the per-figure axis assertions."""
    results = []
    src_dir = cases.data("bdata2")
    fwd_src = os.path.join(src_dir, "forward_slice.npz")
    flc_src = os.path.join(src_dir, "fluctuation_slice.npz")
    fwd_d, fwd = load_bdata(fwd_src)
    _, flc = load_bdata(flc_src)
    src = sources_bdata(os.path.join(src_dir, "forward_sources.npz"))
    cell = np.asarray(fwd_d["cell"], float)

    for plane in ("xy", "zy"):
        bc = _beam_centroid_um(src, cell)
        for part in ("E", "B"):
            results.append(make(
                outdir, f"val_forward_{plane}_{part}", fwd[plane], part, plane,
                src, 32.0, True, "bdata2", fwd_src, bc, False))
            results.append(make(
                outdir, f"val_fluct_{plane}_{part}", flc[plane], part, plane,
                src, 32.0, False, "bdata2", flc_src, bc, True))

    var_src = cases.data("var1b", "slices_var1b.npz")
    d = np.load(var_src, allow_pickle=False)
    n = [int(v) for v in d["n"]]
    cell = np.asarray(d["cell"], float)
    origin = np.asarray(d["origin"], float)
    horiz, vert = axis.coords("var1b", "xy", n, cell, origin, centres=True,
                              source=var_src)
    xy = {"horiz": horiz * US, "vert": vert * US, "plane": "xy",
          "n": n, "cell": cell, "origin": origin}
    for c in ALL:
        axis.check_stored_shape("var1b", "xy", d[f"xy_{c}_pic"], n,
                                source=var_src)
        xy[c] = {
            "pic": axis.to_display(d[f"xy_{c}_pic"], "var1b", "xy", var_src),
            "ref": axis.to_display(d[f"xy_{c}_ref"], "var1b", "xy", var_src),
            "err": axis.to_display(d[f"xy_{c}_err"], "var1b", "xy", var_src),
            "valid": axis.to_display(d[f"xy_{c}_valid"], "var1b", "xy",
                                     var_src).astype(bool),
        }
    srcv = np.asarray(d["sources"], float)
    # The VAR1b bunch sits at the upper edge of the 32-cell window, so its
    # particle centroid is not inside the plotted y range; the axis proof for
    # this case is the structural/field assertion, not the particle centroid.
    for part in ("E", "B"):
        results.append(make(
            outdir, f"val_1mrad_xy_{part}", xy, part, "xy", srcv, 8.0, True,
            "var1b", var_src, None, False))
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    for r in run(a.out_dir):
        print("axis-check", r["figure"], "PASS" if r["pass"] else "FAIL")
