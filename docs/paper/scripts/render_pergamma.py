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


def draw_case(case, outdir):
    figstyle.apply_paper_style(kind="field")
    # The B error row is omitted only when all three reference magnetic
    # components vanish (stationary bunch, beta=0).  A moving bunch keeps
    # B_x and B_z nonzero even though its axial B component is zero, so the
    # row and their error panels must stay.  The decision uses the reference
    # field, not the measured field, which can carry round-off near 1e-18 T.
    bref = [float(np.max(np.abs(case["ref"][c]))) for c in ("Bx", "By", "Bz")]
    berr_omitted = all(v == 0.0 for v in bref)
    assert not berr_omitted or float(
        np.max(np.abs(case["ref"]["B2"]))) == 0.0, (
        f"{case['case']}: B error row omitted but the reference B2 is nonzero")
    rowkeys = ["Efield", "Eerr", "Bfield"] + (["Berr"] if not berr_omitted
                                              else [])
    # Shared square-panel geometry: the same canvas width, panel size and
    # gaps as render_slices/render_large.  Only the height depends on rows.
    geo = figstyle.field_layout(
        len(rowkeys), width_frac=1.0, top_in=figstyle.FIELD_TOP_NOTE_IN,
        bottom_in=(figstyle.FIELD_BOTTOM_NOTE_IN if berr_omitted
                   else figstyle.FIELD_BOTTOM_ERR_IN),
        row_gap_in=figstyle.FIELD_ROW_GAP_IN)
    fig = figstyle.figure(1.0, geo["fig_h_in"])
    fig.text(0.5, 0.985,
             "Error rows: grey = masked / no finite error; "
             "the colour ramp spans finite errors only",
             ha="center", va="center")
    panel_x = geo["panel_x"]
    pw, rh = geo["pw"], geo["rh"]
    cb1, cb2, bw = geo["cb1"], geo["cb2"], geo["bar_width"]
    rows = {key: geo["row_y"][i] for i, key in enumerate(rowkeys)}
    x = case["x"]

    x = case["x"]
    yc = case["y"]
    xlabel, ylabel = axis.axis_labels("hires", "xy", case["source"])
    xticks = [0, 2, 4, 6]
    yticks = [0, 2, 4, 6]

    cmap_comp = figstyle.cmap_for("signed_field")
    cmap_sq = figstyle.cmap_for("squared_field", zero_floor=True)
    cmap_err = figstyle.cmap_for("error")
    masked_min_distance = figstyle.finite_min_distance(cmap_err)
    assert masked_min_distance > 0.10, (
        f"finite error ramp reaches the masked colour "
        f"(min distance {masked_min_distance:.3f})")

    info = {"case": case["case"], "gamma": case["gamma"], "beta": case["beta"],
            "n": case["n"], "cell_m": case["cell_m"], "rows": {}}
    caxes = []
    cbars = []

    def draw_row(rowkey, part, log):
        y0 = rows[rowkey]
        comps = [f"{part}x", f"{part}y", f"{part}z"]
        sq = part + "2"
        if not log:
            zero = {c: float(np.max(np.abs(case["ref"][c]))) == 0.0
                    for c in comps}
            cvals = [np.abs(case["field"][c]) for c in comps if not zero[c]]
            cvmax = (max(float(np.nanmax(v)) for v in cvals)
                     if cvals else 0.0)
            if not np.isfinite(cvmax) or cvmax <= 0:
                cvmax = 1.0
            cnorm = mcolors.SymLogNorm(
                linthresh=max(cvmax * figstyle.SYMLOG_LINTHRESH_FRAC, 1e-30),
                vmin=-cvmax, vmax=cvmax, base=10)
            sq_zero = float(np.max(np.abs(case["ref"][sq]))) == 0.0
            svmax = (0.0 if sq_zero
                     else float(np.nanmax(case["field"][sq])))
            if np.isfinite(svmax) and svmax > 0:
                svmin = max(svmax * figstyle.SQ_FLOOR_RATIO, np.finfo(float).tiny)
                snorm = mcolors.LogNorm(vmin=svmin, vmax=svmax, clip=True)
                sticks = [svmin, np.sqrt(svmin * svmax), svmax]
                slabels = [figstyle.fmt_sci(t) for t in sticks]
                sq_scale = "log"
                sl = (f"{part}$^2$ ({'V$^2$/m$^2$' if part == 'E' else 'T$^2$'}"
                      ", log)")
            else:
                snorm = mcolors.Normalize(vmin=0.0, vmax=1.0)
                sticks = [0.0, 0.5, 1.0]
                slabels = ["0", figstyle.fmt_sci(0.5), figstyle.fmt_sci(1.0)]
                sq_scale = "linear"
                sl = f"{part}$^2$ ({'V$^2$/m$^2$' if part == 'E' else 'T$^2$'})"
            cticks = [-cvmax, 0.0, cvmax]
            clabels = [figstyle.fmt_sci(-cvmax), "0", figstyle.fmt_sci(cvmax)]
            cl = f"{part} ({'V/m' if part == 'E' else 'T'})"
        else:
            ce = [case["err"][c] for c in comps]
            cvmin, cvmax = figstyle.error_limits(ce)
            cnorm = mcolors.LogNorm(vmin=cvmin, vmax=cvmax, clip=True)
            svmin, svmax = figstyle.error_limits([case["err"][sq]])
            snorm = mcolors.LogNorm(vmin=svmin, vmax=svmax, clip=True)
            sq_scale = "log"
            cticks, clabels = figstyle.error_ticks(cvmin, cvmax)
            sticks, slabels = figstyle.error_ticks(svmin, svmax)
            figstyle.assert_masked_distinct(cnorm, cmap_err,
                                             f"{part} error components")
            figstyle.assert_masked_distinct(snorm, cmap_err,
                                             f"{part}$^2$ error")
            cl, sl = f"{part} err", f"{part}$^2$ err"

        imgs = []
        for j, c in enumerate(comps):
            ax = fig.add_axes([panel_x[j], y0, pw, rh])
            arr = case["field"][c] if not log else case["err"][c]
            if not log and zero[c]:
                # A reference component that is exactly zero everywhere is
                # blank: the measured field may still hold round-off at the
                # 1e-18 T level, which must not be drawn as a real field.
                arr = np.zeros_like(arr)
            if log:
                valid = case["valid"][c]
                Z = axis.to_display(
                    np.ma.masked_where((~valid) | (~np.isfinite(arr)), arr),
                    "hires", "xy", case["source"])
                im = figstyle.draw_raster_panel(ax, yc, x, Z, "error",
                                                cnorm, cmap_err)
                try:
                    ax.contour(yc, x, Z, levels=[figstyle.ERROR_GATE],
                               colors=[figstyle.GATE_STYLE["color"]],
                               linewidths=figstyle.GATE_STYLE["linewidth"],
                               linestyles=figstyle.GATE_STYLE["linestyle"])
                except Exception:
                    pass
            else:
                im = figstyle.draw_raster_panel(
                    ax, yc, x,
                    axis.to_display(np.asarray(arr), "hires", "xy",
                                    case["source"]),
                    "signed_field", cnorm, cmap_comp)
            imgs.append(im)
            if not log and zero[c]:
                figstyle.panel_note(ax, rf"${part}_{'xyz'[j]}\equiv0$")
            if log and not np.any(valid):
                note = case["undef"].get(c, "no valid pixels")
                # The caption carries the full reason; keep the in-panel note
                # to the short component identity so it stays inside the axes.
                note = note.split(" (")[0]
                figstyle.panel_note(ax, note)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            if rowkey.endswith("err"):
                ax.set_xlabel(xlabel, labelpad=1)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(ylabel, labelpad=1)
            else:
                ax.set_yticklabels([])
            if rowkey.endswith("field"):
                ax.set_title(f"${part}_{'xyz'[j]}$" if j < 3 else f"${part}^2$",
                             pad=2)
            else:
                ax.set_title("", pad=0)

        axq = fig.add_axes([panel_x[3], y0, pw, rh])
        arrq = case["field"][sq] if not log else case["err"][sq]
        if not log and sq_zero:
            arrq = np.zeros_like(arrq)
        if log:
            valid = case["valid"][sq]
            Z = axis.to_display(
                np.ma.masked_where((~valid) | (~np.isfinite(arrq)), arrq),
                "hires", "xy", case["source"])
            imq = figstyle.draw_raster_panel(axq, yc, x, Z, "error",
                                             snorm, cmap_err)
            try:
                axq.contour(yc, x, Z, levels=[figstyle.ERROR_GATE],
                            colors=[figstyle.GATE_STYLE["color"]],
                            linewidths=figstyle.GATE_STYLE["linewidth"],
                            linestyles=figstyle.GATE_STYLE["linestyle"])
            except Exception:
                pass
        else:
            imq = figstyle.draw_raster_panel(
                axq, yc, x,
                axis.to_display(np.asarray(arrq), "hires", "xy",
                                case["source"]),
                "squared_field", snorm, cmap_sq)
        if not log and sq_zero:
            figstyle.panel_note(axq, rf"${part}^2\equiv0$")
        if log and not np.any(case["valid"][sq]):
            figstyle.panel_note(axq, "no valid pixels")
        axq.set_xticks(xticks)
        axq.set_yticks(yticks)
        if rowkey.endswith("err"):
            axq.set_xlabel(xlabel, labelpad=1)
        else:
            axq.set_xticklabels([])
        axq.set_yticklabels([])
        if rowkey.endswith("field"):
            sq_title = (f"${part}^2$ (log)" if sq_scale == "log"
                        else f"${part}^2$")
            axq.set_title(sq_title, pad=2)
        else:
            axq.set_title("", pad=0)

        assert all(im.norm is cnorm for im in imgs)
        assert imq.norm is snorm and snorm is not cnorm

        no_valid = log and not any(
            np.any(case["valid"][k] & np.isfinite(case["err"][k]))
            for k in comps + [sq])
        if not no_valid:
            cax1 = fig.add_axes([cb1, y0, bw, rh])
            cax2 = fig.add_axes([cb2, y0, bw, rh])
            caxes.extend([cax1, cax2])
            cb1o = add_cbar(fig, cax1, imgs[0], cticks, clabels, cl)
            cb2o = add_cbar(fig, cax2, imq, sticks, slabels, sl)
            cbars.extend([cb1o, cb2o])
            assert cb1o.mappable.norm is cnorm
            assert cb2o.mappable.norm is snorm
            assert cb1o.mappable.norm is not cb2o.mappable.norm

        # row label: gamma, beta, particle count
        rl = (f"{part} {'error' if log else 'field'}\n"
              rf"$\gamma$={case['gamma']:.4g}, $\beta$={case['beta']:.4g}"
              f"\nN=1, {case['n']}$^3$")
        fig.text(0.0315, y0 + rh / 2, rl, rotation=90, ha="center",
                 va="center", linespacing=1.15)

        info["rows"][rowkey] = {
            "comp": {"vmin": float(cnorm.vmin), "vmax": float(cnorm.vmax),
                     "ticks": [float(t) for t in cticks],
                     "norm": "symlog" if not log else "log",
                     "linthresh": float(getattr(cnorm, "linthresh", 0.0))},
            "sq": {"vmin": float(snorm.vmin), "vmax": float(snorm.vmax),
                   "ticks": [float(t) for t in sticks],
                   "norm": sq_scale,
                   "floor_ratio": (figstyle.SQ_FLOOR_RATIO if sq_scale == "log"
                                   and not log else None)},
            "masked_colour": figstyle.MASKED_COLOR,
            "finite_min_masked_distance": masked_min_distance,
            "shared_components": True,
            "bar_labels": [cl, sl],
            "no_valid_pixels": bool(no_valid),
        }

    for key in rowkeys:
        draw_row(key, key[0], key.endswith("err"))
    if berr_omitted:
        fig.text(0.5, 0.035,
                 "B error row omitted: the reference B is identically zero "
                 "at beta=0, so the relative B error is undefined",
                 ha="center", va="center")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # --- assertions ---
    checks = {}
    texts = []
    for t in fig.findobj(mtext.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if t.get_fontsize() > figstyle.BODY_TEXT_PT + 1e-6:
            raise AssertionError(f"font larger than the 10 pt body text: "
                                 f"{t.get_text()!r} {t.get_fontsize()} pt")
        texts.append(t)
    boxes = [(t.get_text(), t.get_window_extent(renderer)) for t in texts]
    overlaps = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            ix = min(a.x1, b.x1) - max(a.x0, b.x0)
            iy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ix > 0.5 and iy > 0.5:
                overlaps.append((boxes[i][0], boxes[j][0], round(ix,1), round(iy,1), [round(v,1) for v in (a.x0,a.y0,a.x1,a.y1)], [round(v,1) for v in (b.x0,b.y0,b.x1,b.y1)]))
    if overlaps:
        raise AssertionError(f"text overlap in {case['case']}: {overlaps[:8]}")

    # colour-bar gaps and tick counts
    panels = []
    for rowkey in rows:
        y0 = rows[rowkey]
        for j, xx in enumerate(panel_x):
            panels.append((rowkey, j, xx, y0, xx + pw, y0 + rh))
    for cax, cb in zip(caxes, cbars):
        b = cax.get_position()
        left_panels = [p for p in panels if p[4] <= b.x0 + 1e-9]
        gap = b.x0 - max(p[4] for p in left_panels)
        assert gap >= 0.008, f"bar too close to panel: {gap}"
        nt = len([t for t in cb.ax.get_yticklabels() if t.get_text().strip()])
        assert 2 <= nt <= 3, f"colour bar has {nt} ticks, expected 2-3"
    checks["no_text_overlap"] = True
    checks["components_share_norm_and_bar"] = True
    checks["squared_scale_distinct_with_own_bar"] = True
    checks["bars_per_row"] = 2
    checks["hidden_bar_rows"] = [
        k for k, v in info["rows"].items() if v.get("no_valid_pixels")]
    checks["min_font_pt"] = min(t.get_fontsize() for t in texts)
    checks["max_bar_ticks"] = max(
        len([t for t in cb.ax.get_yticklabels() if t.get_text().strip()])
        for cb in cbars)
    checks["bar_gap_fraction"] = 0.008
    checks["masked_colour"] = figstyle.MASKED_COLOR
    checks["masked_colour_distinct_from_finite_errors"] = True
    checks["finite_min_masked_distance"] = masked_min_distance
    info["checks"] = checks

    g = float(case["gamma"])
    slug = (f"g{g:.1f}" if g < 10 else f"g{g:.0f}").replace(".", "p")
    name = f"gp_{slug}"
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
    man = {"case": case["case"], "gamma": case["gamma"],
           "beta": case["beta"], "split_scale": True,
           "assertions": checks,
           "data": os.path.relpath(case["path"], cases.REPO_ROOT)}
    man.update(figstyle._layout_manifest(geo))
    figstyle.save(fig, outdir, name, man, tight=False)
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
                     f"no_text_overlap={c['no_text_overlap']} "
                     f"shared_norm_and_bar={c['components_share_norm_and_bar']} "
                     f"sq_distinct={c['squared_scale_distinct_with_own_bar']} "
                     f"bars_per_row={c['bars_per_row']} "
                     f"min_font_pt={c['min_font_pt']} "
                     f"max_bar_ticks={c['max_bar_ticks']}")
        for k, v in d["rows"].items():
            lines.append(f"  {k:6s} comp [{v['comp']['vmin']:.6g}, "
                         f"{v['comp']['vmax']:.6g}] "
                         f"ticks={len(v['comp']['ticks'])} | sq "
                         f"[{v['sq']['vmin']:.6g}, {v['sq']['vmax']:.6g}] "
                         f"ticks={len(v['sq']['ticks'])}")
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
