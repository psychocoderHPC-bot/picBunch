"""Shared style module for the paper figures.

Single source of truth for canvas width, fonts, the Okabe-Ito palette, and
the field/error renderers.  Output is deterministic: fixed backend, fixed PDF
metadata date, no RNG.  Audit JSON is written under ``<outdir>/audit/``.

Every generator imports this module.  It fixes the canvas width to the width at
which the figure is included in the paper, so a 9 pt font in the PDF is a 9 pt
font in print (no hidden shrink factor).  It also provides the only colour-bar
helper: one bar per row, at the row's right edge.
"""
import datetime as _dt
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np

# A4 (210 mm) minus 2 x 25.4 mm geometry margins.
TEXT_BLOCK_IN = 6.268
FIXED_DATE = "D:20261102000000+00'00'"
FIXED_DATE_OBJ = _dt.datetime(2026, 11, 2, tzinfo=_dt.timezone.utc)
HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT_DIR = None  # set per save() call: <outdir>/audit

OKABE = {
    "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
    "purple": "#CC79A7", "grey": "#565656", "black": "#000000",
}
# One fixed Okabe-Ito hue per Lorentz factor, used by every figure that
# encodes gamma as colour.
GAMMA_COLORS = {
    1: OKABE["blue"],
    10: OKABE["vermillion"],
    50: OKABE["purple"],
    100: OKABE["orange"],
}
# Cross-figure line semantics: references are dark grey dashed; the 1e-3
# acceptance gate is a single dark grey dash-dot style (no second grey).
REF_COLOR = "#333333"
REF_STYLE = {"color": REF_COLOR, "linestyle": "--", "linewidth": 0.9}
GATE_STYLE = {"color": REF_COLOR, "linestyle": "-.", "linewidth": 1.0}
UNITS = {"E": "V/m", "B": "T"}


def width_in(fraction):
    """Inches occupied by ``\\includegraphics[width=<fraction>\\linewidth]``."""
    return TEXT_BLOCK_IN * float(fraction)


def apply_style(base=9.0):
    """Print-size rcParams: no text below ``base`` points."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": base,
        "axes.titlesize": base + 0.5,
        "axes.labelsize": base,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "legend.fontsize": base,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save(fig, outdir, name, manifest):
    """Save PDF+PNG at the exact canvas size (no bbox_inches) and audit JSON."""
    os.makedirs(outdir, exist_ok=True)
    audit_dir = os.path.join(outdir, "audit")
    os.makedirs(audit_dir, exist_ok=True)
    pdf = os.path.join(outdir, name + ".pdf")
    png = os.path.join(outdir, name + ".png")
    meta_pdf = {"CreationDate": FIXED_DATE_OBJ, "ModDate": FIXED_DATE_OBJ}
    meta_png = {"CreationDate": FIXED_DATE, "ModDate": FIXED_DATE}
    fig.savefig(pdf, metadata=meta_pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png, dpi=300, metadata=meta_png, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    manifest = dict(manifest)
    manifest.update({
        "figure": name,
        "pdf_width_in": round(float(fig.get_figwidth()), 4),
        "pdf_height_in": round(float(fig.get_figheight()), 4),
        "min_font_pt": min(float(v) for v in (
            plt.rcParams["font.size"], plt.rcParams["axes.labelsize"],
            plt.rcParams["xtick.labelsize"], plt.rcParams["ytick.labelsize"],
            plt.rcParams["legend.fontsize"])),
        "generator": "figstyle.save",
    })
    with open(os.path.join(audit_dir, name + ".json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    return pdf


def _cbar(fig, cax, norm, cmap, label, ticks, ticklabels):
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(label, fontsize=plt.rcParams["axes.labelsize"], labelpad=3)
    cb.ax.tick_params(labelsize=plt.rcParams["xtick.labelsize"], pad=1.5)
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticklabels)
    cb.outline.set_linewidth(0.6)
    return cb


def _fmt_sci(v):
    if v == 0:
        return "0"
    e = int(np.floor(np.log10(abs(v))))
    m = v / 10.0 ** e
    return f"{m:.2g}e{e}"


def render_field_rows(name, outdir, rows, x, y, xlabel, ylabel, width_frac,
                      fig_height, shading="nearest", support=None,
                      suptitle=None, manifest=None):
    """Rows of field/error panels with exactly one shared colour bar per row.

    rows : list of dicts
        {"kind": "field"|"error", "items": [{"arr","err","valid","title"}],
         "part": "E"|"B", "log": bool}
    x, y : coordinate centres/edges along horizontal / vertical axes.
    support : optional bool mask; drawn as a white outline on error panels.
    """
    apply_style()
    ncol = max(len(r["items"]) for r in rows)
    nrow = len(rows)
    fig = plt.figure(figsize=(width_in(width_frac), fig_height))
    left = 0.105
    right = 0.878
    bottom = 0.125
    top = 0.905 if suptitle else 0.91
    gs = fig.add_gridspec(
        nrow, ncol + 1, width_ratios=[1] * ncol + [0.08],
        left=left, right=right, bottom=bottom, top=top,
        wspace=0.16, hspace=0.46,
    )
    field_cmap = plt.get_cmap("RdBu_r").copy()
    field_cmap.set_bad("white")
    err_cmap = plt.get_cmap("magma").copy()
    err_cmap.set_bad("white")

    if shading == "flat":
        xlo, xhi = float(x[0]), float(x[-1])
        ylo, yhi = float(y[0]), float(y[-1])
    else:
        dx = float(x[1] - x[0]); dy = float(y[1] - y[0])
        xlo, xhi = float(x[0] - dx / 2), float(x[-1] + dx / 2)
        ylo, yhi = float(y[0] - dy / 2), float(y[-1] + dy / 2)
    bar_rows = []
    for ri, row in enumerate(rows):
        items = row["items"]
        kind = row["kind"]
        part = row["part"]
        # ---- one normalisation for the whole row ----
        if kind == "field":
            vals = [np.abs(np.asarray(it["arr"], float)) for it in items]
            finite = [v[np.isfinite(v)] for v in vals]
            vmax = max((float(v.max()) for v in finite if v.size), default=1.0)
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0
            norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
            cmap = field_cmap
            cbar_label = f"{part} [{UNITS[part]}]"
            ticks = [-vmax, 0.0, vmax]
            ticklabels = [_fmt_sci(-vmax), "0", _fmt_sci(vmax)]
        else:
            evals = []
            for it in items:
                e = np.asarray(it["err"], float)
                v = np.asarray(it["valid"], bool)
                good = e[v & np.isfinite(e) & (e > 0)]
                if good.size:
                    evals.append(good)
            if evals:
                cat = np.concatenate(evals)
                vmin = 10.0 ** np.floor(np.log10(float(cat.min())))
                vmax = 10.0 ** np.ceil(np.log10(float(cat.max())))
                if vmax <= vmin:
                    vmax = vmin * 10
            else:
                vmin, vmax = 1e-4, 1.0
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            cmap = err_cmap
            cbar_label = "relative error (dimensionless)"
            ticks = [vmin, np.sqrt(vmin * vmax), vmax]
            ticklabels = [_fmt_sci(t) for t in ticks]

        for ci, it in enumerate(items):
            ax = fig.add_subplot(gs[ri, ci])
            if kind == "field":
                arr = np.asarray(it["arr"], float)
                if np.isfinite(arr).any() and np.nanmax(np.abs(arr)) > 0:
                    ax.pcolormesh(x, y, arr, cmap=cmap, norm=norm,
                                  shading=shading)
                else:
                    ax.set_facecolor("0.90")
                    ax.text(0.5, 0.5, rf"${it.get('symbol', '')}\equiv0$",
                            ha="center", va="center",
                            fontsize=plt.rcParams["font.size"],
                            transform=ax.transAxes, color="0.25")
            else:
                arr = np.asarray(it["err"], float)
                valid = np.asarray(it["valid"], bool)
                masked = np.ma.masked_where(~valid | ~np.isfinite(arr), arr)
                if np.any(valid) and np.isfinite(arr[valid]).any():
                    ax.pcolormesh(x, y, masked, cmap=cmap, norm=norm,
                                  shading=shading)
                    xc = 0.5 * (x[:-1] + x[1:]) if shading == "flat" else x
                    yc = 0.5 * (y[:-1] + y[1:]) if shading == "flat" else y
                    try:
                        ax.contour(xc, yc, np.where(valid & np.isfinite(arr),
                                                    arr, np.nan),
                                   levels=[1e-3], colors=[GATE_STYLE["color"]],
                                   linewidths=GATE_STYLE["linewidth"],
                                   linestyles=GATE_STYLE["linestyle"])
                    except Exception:
                        pass
                else:
                    ax.set_facecolor("white")
            if support is not None and kind == "error":
                xc = 0.5 * (x[:-1] + x[1:]) if shading == "flat" else x
                yc = 0.5 * (y[:-1] + y[1:]) if shading == "flat" else y
                try:
                    ax.contour(xc, yc, support.astype(float), levels=[0.5],
                               colors=["white"], linewidths=0.9)
                except Exception:
                    pass
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(ylo, yhi)
            ax.set_aspect("equal")
            ax.set_title(it["title"], pad=2.5,
                         fontsize=plt.rcParams["axes.titlesize"])
            ax.tick_params(length=2.5, pad=1.5)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(3, prune=None))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(3, prune=None))
            if ri == nrow - 1:
                ax.set_xlabel(xlabel)
            else:
                ax.tick_params(labelbottom=False)
            if ci == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.tick_params(labelleft=False)
        cax = fig.add_subplot(gs[ri, ncol])
        _cbar(fig, cax, norm, cmap, cbar_label, ticks, ticklabels)
        bar_rows.append({"row": kind, "colourbars": 1,
                         "scale": [float(norm.vmin), float(norm.vmax)]})
    if suptitle:
        fig.text(0.5, 0.975, suptitle, ha="center", va="top",
                 fontsize=plt.rcParams["axes.titlesize"])
    man = dict(manifest or {})
    man["colorbars_per_row"] = bar_rows
    man["layout"] = "one shared colour bar per row at row right edge"
    return save(fig, outdir, name, man)


def _fmt_tick(v):
    return _fmt_sci(v)


def _bar_title(fig, cax, image, ticks, ticklabels, title):
    """Compact colour bar for the split layout: label above the bar."""
    cb = fig.colorbar(image, cax=cax)
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticklabels)
    cb.ax.tick_params(labelsize=plt.rcParams["xtick.labelsize"], pad=1.5,
                      length=2)
    cb.outline.set_linewidth(0.6)
    cb.set_label(title, fontsize=plt.rcParams["axes.labelsize"], labelpad=3)
    return cb


def render_split_field_rows(name, outdir, rows, x, y, xlabel, ylabel,
                            width_frac, fig_height, shading="nearest",
                            support=None, suptitle=None, manifest=None):
    """Field/error rows with a component triad and a separate squared panel.

    Every row has four items: three signed components sharing one colour bar,
    then the squared magnitude on its own scale with its own bar and unit
    label.  Layout follows the per-gamma figures so the same three-panel plus
    bar, gap, squared-panel plus bar geometry is used everywhere.
    """
    apply_style()
    nrow = len(rows)
    fig = plt.figure(figsize=(width_in(width_frac), fig_height))
    left, right = 0.105, 0.995
    pwg, bg, bw, ba = 0.008, 0.014, 0.012, 0.080
    pw = (right - left - 2 * (bg + bw + ba) - 3 * pwg) / 4.0
    xs = [left, left + pw + pwg, left + 2 * (pw + pwg)]
    cb1 = xs[-1] + pw + bg
    p4 = cb1 + bw + ba
    cb2 = p4 + pw + bg
    assert cb2 + bw <= right + 1e-9
    panel_x = [xs[0], xs[1], xs[2], p4]

    top, bottom = 0.950, 0.105
    gap = 0.075 if nrow > 1 else 0.0
    rh = (top - bottom - gap * (nrow - 1)) / nrow
    row_y = []
    yy = top - rh
    for _ in range(nrow):
        row_y.append(yy)
        yy -= gap + rh

    if shading == "flat":
        xlo, xhi = float(x[0]), float(x[-1])
        ylo, yhi = float(y[0]), float(y[-1])
        xc, yc = 0.5 * (x[:-1] + x[1:]), 0.5 * (y[:-1] + y[1:])
    else:
        dx = float(x[1] - x[0]); dy = float(y[1] - y[0])
        xlo, xhi = float(x[0] - dx / 2), float(x[-1] + dx / 2)
        ylo, yhi = float(y[0] - dy / 2), float(y[-1] + dy / 2)
        xc, yc = x, y

    field_cmap = plt.get_cmap("RdBu_r").copy(); field_cmap.set_bad("white")
    sq_cmap = plt.get_cmap("viridis").copy(); sq_cmap.set_bad("white")
    err_cmap = plt.get_cmap("magma").copy(); err_cmap.set_bad("white")
    bar_rows = []

    for ri, row in enumerate(rows):
        y0 = row_y[ri]
        items = row["items"]
        if len(items) != 4:
            raise ValueError("render_split_field_rows needs three components plus the squared panel")
        comps, sqitem = items[:3], items[3]
        kind, part = row["kind"], row["part"]
        if kind == "field":
            cvmax = max(float(np.nanmax(np.abs(np.asarray(it["arr"], float))))
                        for it in comps)
            if not np.isfinite(cvmax) or cvmax <= 0:
                cvmax = 1.0
            cnorm = mcolors.Normalize(vmin=-cvmax, vmax=cvmax)
            svmax = float(np.nanmax(np.abs(np.asarray(sqitem["arr"], float))))
            if not np.isfinite(svmax) or svmax <= 0:
                svmax = 1.0
            snorm = mcolors.Normalize(vmin=0.0, vmax=svmax)
            ccmap, scmap = field_cmap, sq_cmap
            cticks = [-cvmax, 0.0, cvmax]
            sticks = [0.0, svmax * 0.5, svmax]
            clabel = f"{part} ({UNITS[part]})"
            unit2 = "V$^2$/m$^2$" if part == "E" else "T$^2$"
            slabel = f"{part}$^2$ ({unit2})"
        else:
            def err_span(its):
                good = []
                for it in its:
                    e = np.asarray(it["err"], float)
                    v = np.asarray(it["valid"], bool)
                    e = e[v & np.isfinite(e) & (e > 0)]
                    if e.size:
                        good.append(e)
                if not good:
                    return 1e-4, 1.0
                cat = np.concatenate(good)
                lo = 10.0 ** np.floor(np.log10(float(cat.min())))
                hi = 10.0 ** np.ceil(np.log10(float(cat.max())))
                return (lo, hi if hi > lo else lo * 10.0)
            cvmin, cvmax = err_span(comps)
            svmin, svmax = err_span([sqitem])
            cnorm = mcolors.LogNorm(vmin=cvmin, vmax=cvmax)
            snorm = mcolors.LogNorm(vmin=svmin, vmax=svmax)
            ccmap = scmap = err_cmap
            cticks = [cvmin, np.sqrt(cvmin * cvmax), cvmax]
            sticks = [svmin, np.sqrt(svmin * svmax), svmax]
            clabel, slabel = f"{part} err", f"{part}$^2$ err"

        clabels = [_fmt_tick(t) for t in cticks]
        slabels = [_fmt_tick(t) for t in sticks]

        def panel(ax, it, norm, cmap, title):
            arr = np.asarray(it["arr"], float)
            if kind == "field":
                ax.pcolormesh(x, y, arr, cmap=cmap, norm=norm,
                              shading=shading)
            else:
                e = np.asarray(it["err"], float)
                v = np.asarray(it["valid"], bool)
                z = np.ma.masked_where(~v | ~np.isfinite(e), e)
                ax.pcolormesh(x, y, z, cmap=cmap, norm=norm, shading=shading)
                try:
                    ax.contour(xc, yc, np.where(v & np.isfinite(e), e, np.nan),
                               levels=[1e-3], colors=[GATE_STYLE["color"]],
                               linewidths=GATE_STYLE["linewidth"],
                               linestyles=GATE_STYLE["linestyle"])
                except Exception:
                    pass
                if support is not None:
                    try:
                        ax.contour(xc, yc, support.astype(float), levels=[0.5],
                                   colors=["white"], linewidths=0.9)
                    except Exception:
                        pass
            ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
            ax.set_aspect("auto")
            ax.set_title(title, pad=2.5,
                         fontsize=plt.rcParams["axes.titlesize"])
            ax.tick_params(length=2, pad=1.5)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(3, prune=None))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(3, prune=None))

        for j, it in enumerate(comps):
            ax = fig.add_axes([panel_x[j], y0, pw, rh])
            panel(ax, it, cnorm, ccmap, it.get("title", ""))
            if ri == nrow - 1:
                ax.set_xlabel(xlabel)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.tick_params(labelleft=False)
        axq = fig.add_axes([panel_x[3], y0, pw, rh])
        panel(axq, sqitem, snorm, scmap, sqitem.get("title", ""))
        if ri == nrow - 1:
            axq.set_xlabel(xlabel)
        else:
            axq.tick_params(labelbottom=False)
        axq.tick_params(labelleft=False)

        cax1 = fig.add_axes([cb1, y0, bw, rh])
        cax2 = fig.add_axes([cb2, y0, bw, rh])
        im1 = plt.cm.ScalarMappable(norm=cnorm, cmap=ccmap)
        im2 = plt.cm.ScalarMappable(norm=snorm, cmap=scmap)
        _bar_title(fig, cax1, im1, cticks, clabels, clabel)
        _bar_title(fig, cax2, im2, sticks, slabels, slabel)
        bar_rows.append({"row": kind, "part": part,
                         "colourbars": 2,
                         "labels": [clabel, slabel],
                         "ticks": [cticks, sticks],
                         "fourth_column": f"{part}$^2$",
                         "scales": [[float(cnorm.vmin), float(cnorm.vmax)],
                                    [float(snorm.vmin), float(snorm.vmax)]]})

    if suptitle:
        fig.text(0.5, 0.975, suptitle, ha="center", va="top",
                 fontsize=plt.rcParams["axes.titlesize"])
    man = dict(manifest or {})
    man["colorbars_per_row"] = bar_rows
    man["layout"] = "component triad bar, squared panel with its own bar"
    return save(fig, outdir, name, man)
