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
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.text as mtext  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

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
        "err": {}, "valid": {},
    }
    for k in COMP + ["E2", "B2"]:
        e, v = error(d, plane, k)
        out["err"][k] = e
        out["valid"][k] = v
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


def fmt(v):
    if v == 0:
        return "0"
    if not np.isfinite(v):
        return "n/a"
    e = int(np.floor(np.log10(abs(v))))
    m = v / 10.0 ** e
    if -2 <= e <= 2:
        return f"{v:.3g}"
    return f"{m:.2g}e{e}"


def log_ticks(vmin, vmax, nmax=3):
    """Exactly three decade-rounded ticks, as in the shared style module."""
    vmin = float(max(vmin, 1e-300))
    vmax = float(max(vmax, vmin * 10.0))
    lo = 10.0 ** np.floor(np.log10(vmin))
    hi = 10.0 ** np.ceil(np.log10(vmax))
    if hi <= lo:
        hi = lo * 10.0
    return [lo, float(np.sqrt(lo * hi)), hi]


def add_cbar(fig, cax, image, ticks, labels, title):
    cb = fig.colorbar(image, cax=cax)
    cb.set_ticks(ticks)
    cb.set_ticklabels(labels)
    cb.ax.tick_params(labelsize=9, pad=1.5, length=2)
    cb.outline.set_linewidth(0.6)
    cb.set_label(title, fontsize=9, labelpad=3)
    return cb


def draw_case(case, outdir):
    figstyle.apply_style(9.0)
    # exact A4 text width: 210 mm - 2 x 25.4 mm = 159.2 mm
    W = 159.2 / 25.4
    H = 7.85
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")

    # horizontal geometry (figure fractions)
    left = 0.125
    pwg = 0.008          # panel-to-panel gap
    bg = 0.014           # panel-to-bar gap
    bw = 0.012           # bar width
    ba = 0.080           # bar-to-next-panel gap (tick labels live here)
    right = 0.995
    fixed = left + 2 * (bg + bw + ba) + (right - right)
    pw = (right - left - 2 * (bg + bw + ba) - 3 * pwg) / 4.0
    xs = [left]
    xs.append(xs[-1] + pw + pwg)
    xs.append(xs[-1] + pw + pwg)
    cb1 = xs[-1] + pw + bg
    p4 = cb1 + bw + ba
    cb2 = p4 + pw + bg
    assert cb2 + bw <= right + 1e-9
    panel_x = [xs[0], xs[1], xs[2], p4]

    # vertical geometry: four rows, extra gap between the E and B blocks
    top, bottom = 0.948, 0.060
    g1, g2, g3 = 0.045, 0.075, 0.045
    rh = (top - bottom - g1 - g2 - g3) / 4.0
    rows = {}
    y = top - rh
    rows["Efield"] = y
    y -= g1 + rh
    rows["Eerr"] = y
    y -= g2 + rh
    rows["Bfield"] = y
    y -= g3 + rh
    rows["Berr"] = y

    x = case["x"]
    yc = case["y"]
    dx, dy = x[1] - x[0], yc[1] - yc[0]
    ext = [yc[0] - dy / 2, yc[-1] + dy / 2, x[0] - dx / 2, x[-1] + dx / 2]
    xc = np.linspace(ext[0], ext[1], len(yc))
    yv = np.linspace(ext[2], ext[3], len(x))
    xticks = [0, 2, 4, 6]
    yticks = [0, 2, 4, 6]

    cmap_comp = plt.get_cmap("RdBu_r").copy()
    cmap_comp.set_bad("white")
    cmap_sq = plt.get_cmap("viridis").copy()
    cmap_sq.set_bad("white")
    cmap_err = plt.get_cmap("magma").copy()
    cmap_err.set_bad("white")

    info = {"case": case["case"], "gamma": case["gamma"], "beta": case["beta"],
            "n": case["n"], "cell_m": case["cell_m"], "rows": {}}
    caxes = []
    cbars = []

    def draw_row(rowkey, part, log):
        y0 = rows[rowkey]
        comps = [f"{part}x", f"{part}y", f"{part}z"]
        sq = part + "2"
        if not log:
            cvals = [np.abs(case["field"][c]) for c in comps]
            cvmax = max(float(np.nanmax(v)) for v in cvals)
            if not np.isfinite(cvmax) or cvmax <= 0:
                cvmax = 1.0
            cnorm = mcolors.SymLogNorm(linthresh=max(cvmax * 1e-3, 1e-30),
                                       vmin=-cvmax, vmax=cvmax, base=10)
            svmax = float(np.nanmax(case["field"][sq]))
            if not np.isfinite(svmax) or svmax <= 0:
                svmax = 1.0
            snorm = mcolors.Normalize(vmin=0.0, vmax=svmax)
            cticks = [-cvmax, 0.0, cvmax]
            clabels = [fmt(-cvmax), "0", fmt(cvmax)]
            sticks = [0.0, svmax * 0.5, svmax]
            slabels = ["0", fmt(svmax * 0.5), fmt(svmax)]
            cl, sl = f"{part} ({'V/m' if part == 'E' else 'T'})", \
                     f"{part}$^2$ ({'V$^2$/m$^2$' if part == 'E' else 'T$^2$'})"
        else:
            ce = [case["err"][c] for c in comps]
            parts = [a[np.isfinite(a) & (a > 0)] for a in ce]
            good = np.concatenate(parts) if any(p.size for p in parts) else np.array([])
            if good.size:
                cvmin = max(float(good.min()), 1e-16)
                cvmax = max(float(good.max()), cvmin * 10)
            else:
                cvmin, cvmax = 1e-12, 1.0
            cnorm = mcolors.LogNorm(vmin=cvmin, vmax=cvmax)
            se = case["err"][sq]
            good = se[np.isfinite(se) & (se > 0)]
            if good.size:
                svmin = max(float(good.min()), 1e-16)
                svmax = max(float(good.max()), svmin * 10)
            else:
                svmin, svmax = 1e-12, 1.0
            snorm = mcolors.LogNorm(vmin=svmin, vmax=svmax)
            cticks = log_ticks(cvmin, cvmax)
            sticks = log_ticks(svmin, svmax)
            clabels = [fmt(t) for t in cticks]
            slabels = [fmt(t) for t in sticks]
            cl, sl = f"{part} err", f"{part}$^2$ err"

        imgs = []
        for j, c in enumerate(comps):
            ax = fig.add_axes([panel_x[j], y0, pw, rh])
            arr = case["field"][c] if not log else case["err"][c]
            if log:
                valid = case["valid"][c]
                Z = axis.to_display(
                    np.ma.masked_where((~valid) | (~np.isfinite(arr)), arr),
                    "hires", "xy", case["source"])
                im = ax.imshow(Z, origin="lower", extent=ext, aspect="auto",
                               cmap=cmap_err, norm=cnorm, interpolation="nearest")
                zz = Z
                try:
                    ax.contour(xc, yv, zz, levels=[1e-3],
                               colors=[figstyle.GATE_STYLE["color"]],
                               linewidths=figstyle.GATE_STYLE["linewidth"],
                               linestyles=figstyle.GATE_STYLE["linestyle"])
                except Exception:
                    pass
            else:
                im = ax.imshow(axis.to_display(np.asarray(arr), "hires", "xy",
                                               case["source"]),
                               origin="lower", extent=ext, aspect="auto",
                               cmap=cmap_comp, norm=cnorm,
                               interpolation="nearest")
            imgs.append(im)
            if not log and float(np.nanmax(np.abs(arr))) == 0.0:
                ax.text(0.5, 0.5, rf"${part}_{'xyz'[j]}\equiv0$",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=9, color="0.25")
            if log and not np.any(valid):
                ax.text(0.5, 0.5, "no valid pixels", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="0.25")
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.tick_params(labelsize=9, length=2, pad=1.5)
            if rowkey.endswith("err"):
                ax.set_xlabel(r"$y$ ($\mu$m)", fontsize=9, labelpad=1)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(r"$x$ ($\mu$m)", fontsize=9, labelpad=1)
            else:
                ax.set_yticklabels([])
            if rowkey.endswith("field"):
                ax.set_title(f"${part}_{'xyz'[j]}$" if j < 3 else f"${part}^2$",
                             fontsize=9, pad=2)
            else:
                ax.set_title("", pad=0)

        axq = fig.add_axes([panel_x[3], y0, pw, rh])
        arrq = case["field"][sq] if not log else case["err"][sq]
        if log:
            valid = case["valid"][sq]
            Z = axis.to_display(
                np.ma.masked_where((~valid) | (~np.isfinite(arrq)), arrq),
                "hires", "xy", case["source"])
            imq = axq.imshow(Z, origin="lower", extent=ext, aspect="auto",
                             cmap=cmap_err, norm=snorm, interpolation="nearest")
            try:
                axq.contour(xc, yv, Z, levels=[1e-3],
                            colors=[figstyle.GATE_STYLE["color"]],
                            linewidths=figstyle.GATE_STYLE["linewidth"],
                            linestyles=figstyle.GATE_STYLE["linestyle"])
            except Exception:
                pass
        else:
            imq = axq.imshow(axis.to_display(np.asarray(arrq), "hires", "xy",
                                             case["source"]),
                             origin="lower", extent=ext, aspect="auto",
                             cmap=cmap_sq, norm=snorm,
                             interpolation="nearest")
        if not log and float(np.nanmax(np.abs(arrq))) == 0.0:
            axq.text(0.5, 0.5, rf"${part}^2\equiv0$", transform=axq.transAxes,
                     ha="center", va="center", fontsize=9, color="0.25")
        if log and not np.any(case["valid"][sq]):
            axq.text(0.5, 0.5, "no valid pixels", transform=axq.transAxes,
                     ha="center", va="center", fontsize=9, color="0.25")
        axq.set_xticks(xticks)
        axq.set_yticks(yticks)
        axq.tick_params(labelsize=9, length=2, pad=1.5)
        if rowkey.endswith("err"):
            axq.set_xlabel(r"$y$ ($\mu$m)", fontsize=9, labelpad=1)
        else:
            axq.set_xticklabels([])
        axq.set_yticklabels([])
        if rowkey.endswith("field"):
            axq.set_title(f"${part}^2$", fontsize=9, pad=2)
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
        fig.text(0.033, y0 + rh / 2, rl, rotation=90, ha="center", va="center",
                 fontsize=9, linespacing=1.15)

        info["rows"][rowkey] = {
            "comp": {"vmin": float(cnorm.vmin), "vmax": float(cnorm.vmax),
                     "ticks": [float(t) for t in cticks],
                     "norm": "symlog" if not log else "log",
                     "linthresh": float(getattr(cnorm, "linthresh", 0.0))},
            "sq": {"vmin": float(snorm.vmin), "vmax": float(snorm.vmax),
                   "ticks": [float(t) for t in sticks],
                   "norm": "linear" if not log else "log"},
            "shared_components": True,
            "bar_labels": [cl, sl],
            "no_valid_pixels": bool(no_valid),
        }

    draw_row("Efield", "E", False)
    draw_row("Eerr", "E", True)
    draw_row("Bfield", "B", False)
    draw_row("Berr", "B", True)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # --- assertions ---
    checks = {}
    texts = []
    for t in fig.findobj(mtext.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if t.get_fontsize() < 9.0 - 1e-6:
            raise AssertionError(f"font below 9 pt: {t.get_text()!r} "
                                 f"{t.get_fontsize()} pt")
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
        assert nt == 3, f"colour bar has {nt} ticks, expected 3"
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
    figstyle.save(fig, outdir, name,
                  {"case": case["case"], "gamma": case["gamma"],
                   "beta": case["beta"], "split_scale": True,
                   "assertions": checks, "data": os.path.relpath(case["path"], cases.REPO_ROOT)})
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
