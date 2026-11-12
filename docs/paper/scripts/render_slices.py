#!/usr/bin/env python3
"""Validation slices: BDATA2 forward/fluctuation and the VAR1b 1 mrad case.

Figure family: ``val_*``.  The beam axis (+y) is horizontal, x (x-y plane) or
z (z-y plane) is vertical; see :mod:`axis`.  No numeric value is changed from
the legacy FIGS_REPLOT/rebuild_valid.py renderer.

The forward and fluctuation cases each render one figure per plane with the
electric block first and the magnetic block second, matching the ``gp_*`` and
``large_*`` convention; the VAR1b 1 mrad case keeps its separate electric and
magnetic panels stacked in one figure.
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
import render_bunch  # noqa: E402

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
    refs = [plane_data[c]["ref"] for c in comps]
    valid = ingest.part_valid(refs)
    for c in comps:
        rec = plane_data[c]
        f = rec["ref"] if field_mode == "ref" else rec["pic"]
        fields.append(f)
        it = {"arr": f, "err": rec["err"], "valid": valid,
              "floor_valid": rec["valid"],
              "title": rf"${c[0]}_{{{c[1]}}}$",
              "symbol": f"{c[0]}_{{{c[1]}}}"}
        if float(np.max(np.abs(rec["ref"]))) == 0.0:
            # Identically zero reference component: keep the box in the grid
            # and draw it as a neutral masked panel.
            it["arr"] = np.zeros_like(np.asarray(f, float))
            it["blank"] = True
            it["blank_text"] = fs.zero_field_note(it["symbol"], "beam axis")
            it["error_blank_text"] = fs.zero_error_note(it["symbol"],
                                                        "beam axis")
        items.append(it)
    f2 = sum(f * f for f in fields)
    r2 = sum(plane_data[c]["ref"] ** 2 for c in comps)
    arr = np.maximum(f2, 0.0)
    peak = float(np.max(r2)) if np.max(r2) > 0 else 0.0
    if peak > 0:
        err = np.abs(f2 - r2) / peak
    else:
        err = np.full_like(arr, np.nan)
    sq = {"arr": arr, "err": err, "valid": valid, "floor_valid": valid,
          "title": rf"${part}^2$"}
    if peak == 0.0:
        sq["arr"] = np.zeros_like(arr)
        sq["blank"] = True
        sq["blank_text"] = fs.zero_field_note(f"{part}^2", "stationary")
        sq["error_blank_text"] = fs.zero_error_note(f"{part}^2",
                                                    "stationary")
    items.append(sq)
    return items


def horizontal_axis_record(dataset, plane, source):
    vert, horiz = axis.display_axes(dataset, plane, source)
    return {"propagation_axis": axis.propagation_axis(dataset, 10.0),
            "horizontal": horiz, "vertical": vert,
            "stored_axes": list(axis.stored_axes(dataset, plane, source)),
            "display_axes": [vert, horiz]}


def _part_rows(plane_data, part, plane, sources, coord, has_error, use_ref):
    """Field (and optional error) rows for one field part."""
    field_items = panels(plane_data, part, "ref" if use_ref else "pic")
    rows = [{"kind": "field", "part": part, "items": field_items}]
    smask = None
    if has_error:
        err_items = []
        for it in field_items:
            e = dict(it)
            # it["symbol"] is the LaTeX form (E_{x}, E^2); the raw
            # component names are Ex..Ez, Bx..Bz.
            sym = (it["symbol"].replace("_{", "").replace("_", "")
                   .replace("}", "").replace("^2", "2")
                   if "symbol" in it
                   else part + "2")
            if sym.startswith(part) and sym != part + "2" and float(
                    np.max(np.abs(plane_data[sym]["ref"]))) == 0.0:
                # Identically-zero reference component: keep the box as a
                # neutral masked panel and say why.
                e["undefined"] = True
                e["blank_text"] = fs.zero_error_note(e["symbol"],
                                                     "beam axis")
                e["annotation"] = e["blank_text"]
            elif it.get("blank") and sym == part + "2":
                e["undefined"] = True
                e["blank_text"] = it.get(
                    "error_blank_text",
                    fs.zero_error_note(f"{part}^2", "stationary"))
                e["annotation"] = e["blank_text"]
            err_items.append(e)
        rows.append({"kind": "error", "part": part, "items": err_items})
        smask = support_mask(plane, field_items[0]["arr"].shape, sources, coord)
    return rows, smask


def _assert_axis(name, plane_data, dataset, plane, source, beam_centroid,
                 use_ref):
    return axis.assert_beam_axis(
        dataset, plane, _magnitude(plane_data, use_ref),
        plane_data["horiz"], plane_data["vert"], n=plane_data["n"],
        cell=plane_data["cell"], origin=plane_data["origin"], gamma=10.0,
        centres=True, source=source, figure=name, label="|E|",
        beam_position=beam_centroid)


def make(outdir, name, plane_data, part, plane, sources, coord, has_error,
         dataset, source, beam_centroid, use_ref, bunch=None):
    rows, smask = _part_rows(plane_data, part, plane, sources, coord,
                             has_error, use_ref)
    xl, yl = axis.axis_labels(dataset, plane, source)
    man = {"source_data": os.path.relpath(source, cases.REPO_ROOT),
           "dataset": dataset, "plane": plane, "part": part,
           "has_error": has_error, "includegraphics_width": 0.92,
           "axis_convention": horizontal_axis_record(dataset, plane, source)}
    extra = _bunch_layout_kw(bunch)
    fs.render_split_field_rows(
        name, outdir, rows, plane_data["horiz"], plane_data["vert"],
        xlabel=xl, ylabel=yl + ("" if not has_error else "\n"),
        width_frac=1.0, support=smask, manifest=man, **extra)
    return _assert_axis(name, plane_data, dataset, plane, source,
                        beam_centroid, use_ref)


def make_combined(outdir, name, plane_data, plane, sources, coord, has_error,
                  dataset, source, beam_centroid, use_ref, bunch=None):
    """One figure per plane with the electric block first, magnetic second.

    ``val_forward`` carries both error rows (E field, E error, B field,
    B error); the reference-only ``val_fluct`` figure has no error rows and
    keeps its two field rows (E panels then B panels).  Layout parameters
    follow the large-domain figures so the four-row block fits the appendix
    page at the 9 pt print size.
    """
    rows = []
    smask = None
    for part in ("E", "B"):
        part_rows, part_mask = _part_rows(plane_data, part, plane, sources,
                                          coord, has_error, use_ref)
        rows.extend(part_rows)
        if part_mask is not None:
            smask = part_mask
    xl, yl = axis.axis_labels(dataset, plane, source)
    block_order = ("E field, E error, B field, B error" if has_error
                   else "E field, B field")
    man = {"source_data": os.path.relpath(source, cases.REPO_ROOT),
           "dataset": dataset, "plane": plane, "parts": ["E", "B"],
           "block_order": block_order, "has_error": has_error,
           "includegraphics_width": 1.0,
           "axis_convention": horizontal_axis_record(dataset, plane, source)}
    extra = _bunch_layout_kw(bunch)
    fs.render_split_field_rows(
        name, outdir, rows, plane_data["horiz"], plane_data["vert"],
        xlabel=xl, ylabel=yl + ("" if not has_error else "\n"),
        width_frac=1.0, compact=True,
        support=smask, manifest=man, **extra)
    return _assert_axis(name, plane_data, dataset, plane, source,
                        beam_centroid, use_ref)


def _bunch_layout_kw(bunch):
    """Compact field layout used only when the bunch row is present.

    The appendix slice plates place a long caption under a tall field block.
    ``panel_height_scale`` squeezes only the panel height (the panel width,
    the placed ``\\linewidth`` and every font token stay fixed), and the
    trimmed top/row gaps remove whitespace that carried no information.  At
    the default of the shared style module the layout is unchanged.
    """
    if bunch is None:
        return {}
    return {"bunch": bunch, "panel_scale": 1.0,
            "panel_height_scale": fs.APPENDIX_PANEL_HEIGHT_SCALE, "top_in": 0.10,
            "bottom_in": 0.58, "row_gap_in": 0.30}


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
    flc_d, flc = load_bdata(flc_src)
    src = sources_bdata(os.path.join(src_dir, "forward_sources.npz"))
    cell = np.asarray(fwd_d["cell"], float)
    sources_path = os.path.join(src_dir, "forward_sources.npz")
    src_full = np.load(sources_path, allow_pickle=False)["sources"]
    bdata = render_bunch.Bunch(
        np.asarray(src_full[:, :3], float) * cell,
        -np.asarray(src_full[:, 6], float) * render_bunch.E_CHARGE,
        "y", float(cell[0]), 10.0,
        os.path.relpath(sources_path, cases.REPO_ROOT),
        "derived from forward_sources.npz: positions sources[:, :3] (cell "
        "indices) x cell; charge -sources[:,6]*e = -5e-13 C each; the npz "
        "carries no bunch_* keys", pending_fresh_data=True)

    for plane in ("xy", "zy"):
        bc = _beam_centroid_um(src, cell)
        fwd_row = render_bunch.build_row(
            bdata, "bdata2", f"val_forward_{plane}", plane,
            origin=fwd_d["origin"], n=fwd_d["n"], cell=fwd_d["cell"],
            source=fwd_src)
        flc_row = render_bunch.build_row(
            bdata, "bdata2", f"val_fluct_{plane}", plane,
            origin=flc_d["origin"], n=flc_d["n"], cell=flc_d["cell"],
            source=flc_src)
        results.append(make_combined(
            outdir, f"val_forward_{plane}", fwd[plane], plane, src, 32.0,
            True, "bdata2", fwd_src, bc, False, bunch=fwd_row))
        results.append(make_combined(
            outdir, f"val_fluct_{plane}", flc[plane], plane, src, 32.0,
            False, "bdata2", flc_src, bc, True, bunch=flc_row))

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
    cellv = np.asarray(d["cell"], float)
    if "bunch_pos_m" in d.files and "bunch_q_C" in d.files:
        pos_m = np.asarray(d["bunch_pos_m"], float)
        q_C = np.asarray(d["bunch_q_C"], float)
        bunch_source = ("bunch_pos_m (metres) and bunch_q_C from the fresh "
                        "reduced file")
    else:
        # Historical var1b sources are stored in the archived input's unit
        # (unitSI = 1e-7 m), NOT in cell indices.  Convert to metres once.
        pos_m = srcv * 1e-7
        q_C = np.full(srcv.shape[0], -32.0e-18 / srcv.shape[0])
        bunch_source = ("sources[:, :3] * 1e-7 m and -32 aC / N; the file "
                        "carries no bunch_* keys")
    # Cell indices for the support outline and the crop.  The density row is
    # binned on the field lattice (cell = 2e-7 m), so the same metre positions
    # must be divided by the field cell exactly once.
    src_cells = pos_m / cellv
    bvar = render_bunch.Bunch(
        pos_m, q_C, "y", float(cellv[0]), 10.0,
        os.path.relpath(var_src, cases.REPO_ROOT),
        "derived from slices_var1b.npz: %s; cell indices = pos_m / cell" %
        bunch_source, pending_fresh_data=False)
    for part in ("E", "B"):
        row = render_bunch.build_row(
            bvar, "var1b", f"val_1mrad_xy_{part}", "xy",
            origin=d["origin"], n=d["n"], cell=d["cell"], source=var_src)
        results.append(make(
            outdir, f"val_1mrad_xy_{part}", xy, part, "xy", src_cells, 8.0,
            True, "var1b", var_src, None, False, bunch=row))
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    for r in run(a.out_dir):
        print("axis-check", r["figure"], "PASS" if r["pass"] else "FAIL")
