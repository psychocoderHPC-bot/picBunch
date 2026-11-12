#!/usr/bin/env python3
"""RENDER5: final per-gamma single-particle figures from the HIRES series.

One figure per velocity case.  In every field row the three signed components
(E_x,E_y,E_z or B_x,B_y,B_z) share ONE symmetric-log scale and ONE colour bar;
the squared magnitude (E^2 or B^2) has a DISTINCT linear scale and its own
colour bar.  Error rows repeat the same split with logarithmic scales.

Data: the seven HIRES hi_*.npz 128^3/5e-8 m slices only.  Output names are
gp_g<gamma> with "." replaced by "p" so the paper can include gp_g1p0.pdf ...
gp_g100.pdf.  The six old 64^3 gp_* figures are superseded.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import axis  # noqa: E402
import cases  # noqa: E402
import figstyle  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.text as mtext  # noqa: E402

COMP = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]
CI = {c: i for i, c in enumerate(COMP)}


def _gamma_of(path):
    with np.load(path, allow_pickle=False) as d:
        return float(d["gamma"])


def find_cases():
    hires = glob.glob(os.path.join(cases.data("hires"), "hi_*.npz"))
    if len(hires) != 7:
        raise SystemExit(f"RENDER5 requires the seven HIRES slices, found {len(hires)}")
    return [("hires", p) for p in sorted(hires, key=_gamma_of)]


def comp_array(d, plane, key):
    a = np.asarray(d[f"{plane}_{key}"], dtype=float)
    if a.ndim == 3:
        if a.shape[0] != 6:
            raise ValueError(f"{plane}_{key} unexpected shape {a.shape}")
        return a[CI[key]]
    return a


def field(d, plane, key):
    if key in ("E2", "B2"):
        return np.asarray(d[f"{plane}_{key}"], dtype=float)
    return comp_array(d, plane, key)


def ref_field(d, plane, key):
    """Reference field for one component or the squared magnitude."""
    if key in ("E2", "B2"):
        return np.asarray(d[f"{plane}_ref_{key}"], dtype=float)
    return comp_array(d, plane, "ref_" + key)


def error(d, plane, key):
    if key in ("E2", "B2"):
        pic = field(d, plane, key)
        ref = np.asarray(d[f"{plane}_ref_{key}"], dtype=float)
        peak = float(np.max(np.abs(ref))) if np.isfinite(ref).any() else 0.0
        valid = np.abs(ref) >= 1e-3 * peak if peak > 0 else np.zeros_like(ref, bool)
        err = (np.where(valid, np.abs(pic - ref) / peak, np.nan)
               if peak > 0 else np.full_like(ref, np.nan))
        return err, valid
    e = np.asarray(d[f"{plane}_err_{key}"], dtype=float)
    v = np.asarray(d[f"{plane}_valid_{key}"], dtype=bool)
    if e.ndim == 3:
        e = e[CI[key]]
    if v.ndim == 3:
        v = v[CI[key]]
    return e, v


def load(path):
    d = np.load(path, allow_pickle=False)
    plane = "xy"
    x = np.asarray(d[f"{plane}_x_um"], float)
    y = np.asarray(d[f"{plane}_y_um"], float)
    out = {
        "path": path, "source": path, "case": str(d["case"]),
        "gamma": float(d["gamma"]), "beta": float(d["beta"]),
        "n": int(np.asarray(d["n"]).ravel()[0]),
        "cell_m": float(d["cell_m"]),
        "pos_m": np.asarray(d["particle_pos_m"], float),
        "x": x, "y": y,
        "field": {k: field(d, plane, k) for k in COMP + ["E2", "B2"]},
        "ref": {k: ref_field(d, plane, k) for k in COMP + ["E2", "B2"]},
        "err": {}, "valid": {}, "undef": {},
    }
    for k in COMP + ["E2", "B2"]:
        e, v = error(d, plane, k)
        out["err"][k] = e
        out["valid"][k] = v
    # A reference component that is exactly zero everywhere (stationary
    # bunch: all B; beam along +y: By) has an undefined relative error: its
    # error panel renders fully masked and carries an in-panel annotation
    # instead of the bare "no valid pixels" text.
    for part in ("E", "B"):
        if float(np.max(np.abs(out["ref"][part + "2"]))) == 0.0:
            continue
        for j, c in enumerate((f"{part}x", f"{part}y", f"{part}z")):
            if float(np.max(np.abs(out["ref"][c]))) == 0.0:
                out["undef"][c] = (
                    rf"$\mathbf{{{c[0]}}}_{{{'xyz'[j]}}}\equiv0$"
                    " (beam axis); relative error undefined")
    for part in ("E", "B"):
        e2 = out["field"][part + "2"]
        rec = sum(out["field"][c] ** 2 for c in (f"{part}x", f"{part}y", f"{part}z"))
        assert np.allclose(e2, rec, rtol=1e-8, atol=1e-12), f"{out['case']} {part}2 mismatch"
        # A physically zero component can carry roundoff-level noise, where
        # sqrt(rec) is below the allclose atol and the magnitude check would
        # reject a correctly squared field.  Only test meaningful amplitudes.
        if float(np.max(rec)) > 1e-18:
            assert not np.allclose(e2, np.sqrt(np.maximum(rec, 0)), rtol=1e-6,
                                   atol=1e-9), f"{out['case']} {part}2 is the magnitude"
    return out


def add_cbar(fig, cax, image, ticks, labels, title):
    """Delegate to the one canonical colour-bar style."""
    return figstyle.draw_colorbar(fig, image, cax=cax, ticks=ticks,
                                  ticklabels=labels, label=title)


def _display(case, arr, dtype=float):
    return axis.to_display(np.asarray(arr, dtype), "hires", "xy", case["source"])


def _item(case, part, c, j, log):
    """One raster item in display orientation (the shared renderer draws it)."""
    if log:
        e = _display(case, case["err"][c])
        v = _display(case, case["valid"][c].astype(float)).astype(bool)
        item = {"err": e, "valid": v,
                "title": f"${part}_{{{'xyz'[j]}}}$",
                "symbol": f"{part}_{{{'xyz'[j]}}}"}
        if not np.any(v):
            item["annotation"] = case["undef"].get(c, "no valid pixels")
        return item
    zero = float(np.max(np.abs(case["ref"][c]))) == 0.0
    arr = np.zeros_like(np.asarray(case["field"][c], float)) if zero \
        else _display(case, case["field"][c])
    return {"arr": arr, "title": f"${part}_{{{'xyz'[j]}}}$",
            "symbol": f"{part}_{{{'xyz'[j]}}}",
            "note": (rf"${part}_{{{'xyz'[j]}}}\equiv0$" if zero else None)}


def _sq_item(case, part, log):
    key = part + "2"
    if log:
        e = _display(case, case["err"][key])
        v = _display(case, case["valid"][key].astype(float)).astype(bool)
        return {"err": e, "valid": v, "title": f"${part}^2$",
                "annotation": "no valid pixels" if not np.any(v) else None}
    zero = float(np.max(np.abs(case["ref"][key]))) == 0.0
    arr = np.zeros_like(np.asarray(case["field"][key], float)) if zero \
        else _display(case, case["field"][key])
    return {"arr": arr, "title": f"${part}^2$",
            "note": (rf"${part}^2\equiv0$" if zero else None)}


def draw_case(case, outdir):
    # N==1 guard: the HIRES cases are single point charges.  A charge-density
    # "bunch form" row is physically meaningless for one particle, so this
    # family never draws it and its output stays byte-identical.  (The
    # distribution families add the row through render_bunch instead.)
    n_particles = int(np.asarray(case["pos_m"]).reshape(-1, 3).shape[0])
    if n_particles != 1:
        raise ValueError(
            f"render_pergamma expects N==1 point charges, got N={n_particles} "
            f"for {case['case']}")
    figstyle.apply_paper_style(kind="field")
    # The B error row is omitted only when all three reference magnetic
    # components vanish (stationary bunch, beta=0).
    bref = [float(np.max(np.abs(case["ref"][c]))) for c in ("Bx", "By", "Bz")]
    berr_omitted = all(v == 0.0 for v in bref)
    assert not berr_omitted or float(
        np.max(np.abs(case["ref"]["B2"]))) == 0.0, (
        f"{case['case']}: B error row omitted but the reference B2 is nonzero")

    rows = []
    for part in ("E", "B"):
        comps = [part + "x", part + "y", part + "z"]
        fld = [_item(case, part, c, j, False) for j, c in enumerate(comps)]
        fld.append(_sq_item(case, part, False))
        rows.append({"kind": "field", "part": part, "items": fld})
        if part == "B" and berr_omitted:
            continue
        err = [_item(case, part, c, j, True) for j, c in enumerate(comps)]
        err.append(_sq_item(case, part, True))
        rows.append({"kind": "error", "part": part, "items": err})

    # One rotated row label per square row, in the reflowed order.
    flat = figstyle._reflow_field_rows(rows)
    labels = []
    for r in flat:
        g = float(case["gamma"])
        head = (f"{r['part']} field" if r["role"] == "comp"
                and r["kind"] == "field" else
                f"{r['part']} error" if r["role"] == "comp" else
                f"{r['part']}$^2$ value/err")
        labels.append(f"{head}\n"
                      rf"$\gamma$={case['gamma']:.4g}, "
                      rf"$\beta$={case['beta']:.4g}"
                      f"\nN=1, {case['n']}$^3$")

    xlabel, ylabel = axis.axis_labels("hires", "xy", case["source"])
    collected = []
    info = {"case": case["case"], "gamma": case["gamma"],
            "beta": case["beta"], "n": case["n"],
            "cell_m": case["cell_m"]}
    g = float(case["gamma"])
    slug = (f"g{g:.1f}" if g < 10 else f"g{g:.0f}").replace(".", "p")
    name = f"gp_{slug}"
    man = {"case": case["case"], "gamma": case["gamma"],
           "beta": case["beta"], "split_scale": True,
           "data": os.path.relpath(case["path"], cases.REPO_ROOT),
           "raster_panel_mm": figstyle.RASTER_PANEL_MM}
    bottom_note = None
    if berr_omitted:
        bottom_note = ("B error row omitted: the reference B is identically "
                       "zero at beta=0, so the relative B error is undefined")
    figstyle.render_split_field_rows(
        name, outdir, rows, case["y"], case["x"],
        xlabel=xlabel, ylabel=ylabel, width_frac=1.0,
        top_in=figstyle.FIELD_TOP_NOTE_IN,
        bottom_in=(figstyle.FIELD_BOTTOM_NOTE_IN if berr_omitted
                   else figstyle.FIELD_BOTTOM_ERR_IN),
        fixed_ticks=([0, 2, 4, 6], [0, 2, 4, 6]),
        row_labels=labels,
        top_note=("Error rows: grey = masked / no finite error; "
                  "the colour ramp spans finite errors only"),
        bottom_note=bottom_note,
        peak_annotations=False,
        collect_rows=collected,
        manifest=man)

    info["rows"] = collected
    info["checks"] = {
        "bars_per_row": max(r["panels"] for r in collected),
        "min_font_pt": figstyle.TICK_PT,
        "masked_colour": figstyle.MASKED_COLOR,
    }
    pos_um = np.asarray(case["pos_m"], float) * 1e6
    nn = int(case["n"])
    axis_result = axis.assert_beam_axis(
        "hires", "xy",
        axis.to_display(case["field"]["E2"], "hires", "xy", case["source"]),
        case["y"], case["x"], n=[nn, nn, nn],
        cell=[case["cell_m"]] * 3, origin=[0.0, 0.0, 0.0], gamma=g,
        centres=True, source=case["source"], figure=name, label="E^2",
        beam_position={"x": float(pos_um[0]), "y": float(pos_um[1]),
                       "z": float(pos_um[2])},
        require_longitudinal_contraction=bool(g >= 10.0))
    info["axis_assertion"] = axis_result
    info["figure"] = name
    return info, name, axis_result


def main(outdir):
    found_cases = find_cases()
    rendered, details, axis_results = [], [], []
    for kind, path in found_cases:
        c = load(path)
        info, name, axis_result = draw_case(c, outdir)
        details.append(info)
        axis_results.append(axis_result)
        rendered.append((c["gamma"], c["beta"], name, path))
    have = [f"{g:.4g}" for g, _, _, _ in rendered]
    want = ["1.0", "1.1", "1.5", "2.0", "10", "50", "100"]
    missing = [w for w in want if not any(abs(float(h) - float(w)) < 0.02 for h in have)]
    hi = glob.glob(os.path.join(cases.data("hires"), "hi_*.npz"))
    report = {"rendered": [{"gamma": g, "beta": b, "figure": n,
                            "data": os.path.basename(p)}
                           for g, b, n, p in rendered],
              "hires_npz_found": len(hi),
              "hires_series_available": bool(hi),
              "fallback_used": False,
              "target_hires_gamma": want,
              "missing_hires_gamma": missing,
              "details": details}
    with open(os.path.join(outdir, "render5_assertions.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    lines = ["RENDER5 final per-gamma split-scale assertions", ""]
    lines.append(f"HIRES hi_*.npz found: {len(hi)}; all seven HIRES slices rendered")
    lines.append("rendered: " + ", ".join(f"gamma={g:.4g} beta={b:.4g} -> {n}"
                                            for g, b, n, _ in rendered))
    lines.append("missing high-resolution target gamma: " + ", ".join(missing))
    lines.append("")
    for d in details:
        c = d["checks"]
        lines.append(f"{d['figure']} ({d['case']}): "
                     f"rows={len(d['rows'])} "
                     f"max_panels_per_row={c['bars_per_row']} "
                     f"min_font_pt={c['min_font_pt']} "
                     f"panel_mm=32.0")
        for r in d["rows"]:
            scales = ", ".join(f"[{lo:.6g}, {hi:.6g}]" for lo, hi in r["norms"])
            lines.append(f"  {r['role']:3s} {r['part']} {r['kind']:9s} "
                         f"panels={r['panels']} scales={scales}")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(outdir, "assertions_render5.txt"), "w") as fh:
        fh.write(text)
    print(text)
    return axis_results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    main(a.out_dir)
