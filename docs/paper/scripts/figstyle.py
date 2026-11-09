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
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.text as mtext
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
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
SUPPORT_STYLE = {"color": REF_COLOR, "linestyle": "-", "linewidth": 0.9}
UNITS = {"E": "V/m", "B": "T"}

# Error-panel presentation contract shared by the slice renderers.  The masked
# colour is a dedicated light grey and the finite error colormap is truncated
# below its near-white top, so "no data" cannot be confused with the worst
# finite error.
MASKED_COLOR = "#dddddd"
ERR_CMAP_TOP = 0.90
ERR_FLOOR = 1e-6
ERR_GATE = 1e-3
SQ_LOG_DECADES = 6.0


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


_LAYOUT_RECORDS = {}


def _axes_kind(ax):
    if getattr(ax, "_colorbar", None) is not None:
        return "colourbar"
    if ax.get_label() in ("<colorbar>",):
        return "colourbar"
    return "panel"


def _text_kind(t):
    if t.axes is None:
        return "figure"
    if getattr(t.axes, "_colorbar", None) is not None:
        return "colourbar"
    return "panel"


def _box(b):
    return [round(float(v), 2) for v in (b.x0, b.y0, b.x1, b.y1)]


def layout_collisions(fig, name):
    """Return every text/axes and text/text overlap of the drawn figure.

    Every visible text box is tested: tick labels, axis labels and titles,
    colour-bar labels, in-panel annotations, and figure text.  A text is
    allowed to sit inside its own axes rectangle only; it must stay clear of
    every other axes rectangle and of every other text box.  The record also
    stores the minimum signed clearance (positive = gap, negative = overlap)
    for axis text, so the layout margins are auditable per figure.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes = [a for a in fig.axes if a.get_visible()]
    ax_boxes = [(a, a.get_window_extent(renderer)) for a in axes]

    tick_ids = set()
    label_ids = set()
    for a in axes:
        for t in list(a.get_xticklabels()) + list(a.get_yticklabels()):
            tick_ids.add(id(t))
        for t in (a.title, a.xaxis.label, a.yaxis.label):
            label_ids.add(id(t))

    texts = [t for t in fig.findobj(mtext.Text)
             if t.get_visible() and t.get_text().strip()]
    collisions = []

    def role(t):
        if id(t) in tick_ids:
            return "tick_label"
        if id(t) in label_ids:
            return "axis_label"
        if t.axes is None:
            return "figure_text"
        return "annotation"

    def is_axis_text(t):
        return role(t) in ("tick_label", "axis_label")

    def overlap(a, b):
        return (min(a.x1, b.x1) - max(a.x0, b.x0),
                min(a.y1, b.y1) - max(a.y0, b.y0))

    def clearance(a, b):
        dx = max(a.x0 - b.x1, b.x0 - a.x1)
        dy = max(a.y0 - b.y1, b.y0 - a.y1)
        if dx >= 0 or dy >= 0:
            return max(dx, dy) * 72.0 / fig.dpi
        return -min(min(a.x1, b.x1) - max(a.x0, b.x0),
                    min(a.y1, b.y1) - max(a.y0, b.y0)) * 72.0 / fig.dpi

    def detail(kind, t, other, other_role, a, b, ix, iy):
        return {
            "kind": kind,
            "text": t.get_text(),
            "text_role": role(t),
            "text_owner": _text_kind(t),
            "other": other,
            "other_role": other_role,
            "overlap_px": [round(ix, 2), round(iy, 2)],
            "overlap_pt": [round(ix * 72.0 / fig.dpi, 2),
                           round(iy * 72.0 / fig.dpi, 2)],
            "text_bbox": _box(a),
            "other_bbox": _box(b),
        }

    minima = {
        "axis_text_to_foreign_axes": None,
        "axis_text_to_foreign_text": None,
        "any_text_to_foreign_axes": None,
        "any_text_to_text": None,
    }

    def remember(key, value):
        if minima[key] is None or value < minima[key]:
            minima[key] = value

    tboxes = [(t, t.get_window_extent(renderer)) for t in texts]
    for t, tb in tboxes:
        own = t.axes
        for a, ab in ax_boxes:
            if own is a:
                continue
            s = clearance(tb, ab)
            remember("any_text_to_foreign_axes", s)
            if is_axis_text(t):
                remember("axis_text_to_foreign_axes", s)
            ix, iy = overlap(tb, ab)
            if ix > 0.5 and iy > 0.5:
                collisions.append(detail("text/axes", t, _axes_kind(a),
                                         _axes_kind(a), tb, ab, ix, iy))
    for i in range(len(tboxes)):
        for j in range(i + 1, len(tboxes)):
            ta, a = tboxes[i]
            tb, b = tboxes[j]
            s = clearance(a, b)
            remember("any_text_to_text", s)
            if is_axis_text(ta) and is_axis_text(tb):
                remember("axis_text_to_foreign_text", s)
            ix, iy = overlap(a, b)
            if ix > 0.5 and iy > 0.5:
                collisions.append(detail(
                    "text/text", ta, tb.get_text(), role(tb), a, b, ix, iy))

    minima = {k: (None if v is None else round(float(v), 2))
              for k, v in minima.items()}
    return {"figure": name, "n_axes": len(axes), "n_texts": len(texts),
            "collisions": collisions, "min_clearance_pt": minima}


def _record_layout(outdir, record):
    _LAYOUT_RECORDS[(outdir, record["figure"])] = record
    rows = [r for (od, _), r in sorted(_LAYOUT_RECORDS.items())
            if od == outdir]
    path = os.path.join(outdir, "layout_assertions.json")
    with open(path, "w") as fh:
        json.dump({"figures": rows}, fh, indent=1, sort_keys=True)
    return path


def assert_no_layout_collisions(fig, outdir, name):
    """Fail loudly (and persist the evidence) on any layout collision."""
    record = layout_collisions(fig, name)
    path = _record_layout(outdir, record)
    if record["collisions"]:
        first = record["collisions"][0]
        raise AssertionError(
            f"layout collision in {name}: {len(record['collisions'])} "
            f"pair(s); first {first['kind']} {first['text']!r} vs "
            f"{first['other']!r} overlap {first['overlap_px']} px "
            f"({first['overlap_pt']} pt); details in {path}")
    return record


def save(fig, outdir, name, manifest):
    """Save PDF+PNG at the exact canvas size (no bbox_inches) and audit JSON."""
    assert_no_layout_collisions(fig, outdir, name)
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


def _truncated_cmap(name, top=ERR_CMAP_TOP, n=256):
    """Finite colormap truncated below its near-white top end."""
    base = plt.get_cmap(name)
    rgba = base(np.linspace(0.0, float(top), n))
    return mcolors.ListedColormap(rgba, name=f"{name}_trunc")


def _assert_masked_distinct(norm, cmap, label="error"):
    """Fail if a finite value maps to the masked grey or a white/grey top."""
    if isinstance(norm, mcolors.LogNorm):
        vals = np.geomspace(float(norm.vmin), float(norm.vmax), 257)
    else:
        vals = np.linspace(float(norm.vmin), float(norm.vmax), 257)
    cols = np.asarray(cmap(norm(vals)))[:, :3]
    bad = np.asarray(mcolors.to_rgba(MASKED_COLOR))[:3]
    dist = np.max(np.abs(cols - bad), axis=1)
    if not np.all(dist > 0.02):
        i = int(np.argmin(dist))
        raise AssertionError(
            f"{label}: finite value {vals[i]:.3g} maps to the masked colour "
            f"{tuple(np.round(cols[i], 3))}")
    top = np.asarray(cmap(1.0))[:3]
    if float(top.min()) > 0.90 or float(np.ptp(top)) < 0.08:
        raise AssertionError(
            f"{label}: finite colormap top {tuple(np.round(top, 3))} is "
            "near-white or grey")


def _error_limits(items):
    """Error range floored at ERR_FLOOR, ceiling at a decade (or the gate)."""
    good = []
    for it in items:
        e = np.asarray(it["err"], float)
        v = np.asarray(it["valid"], bool)
        e = e[v & np.isfinite(e) & (e > 0)]
        if e.size:
            good.append(e)
    hi = 1.0
    if good:
        cat = np.concatenate(good)
        hi = 10.0 ** np.ceil(np.log10(float(cat.max())))
    vmin = ERR_FLOOR
    vmax = max(float(hi), ERR_GATE)
    if vmax <= vmin:
        vmax = vmin * 10.0
    return vmin, vmax


def _error_ticks(vmin, vmax):
    """Lower floor, the marked 1e-3 gate, and the ceiling."""
    ticks = {float(vmin), float(vmax)}
    if float(vmin) <= ERR_GATE <= float(vmax):
        ticks.add(float(ERR_GATE))
    ticks = sorted(ticks)
    labels = ["1e-3\ngate" if np.isclose(t, ERR_GATE, rtol=1e-9, atol=0.0)
              else _fmt_sci(t) for t in ticks]
    return ticks, labels


def _sq_ticks(peak, decades=SQ_LOG_DECADES):
    """Decade ticks from peak*10^-decades up to the peak."""
    ticks = [float(peak) * 10.0 ** d for d in np.arange(-int(decades), 1)]
    return ticks, [_fmt_sci(t) for t in ticks]


def _peak_fraction(peak, shared):
    if not np.isfinite(peak) or shared <= 0:
        return "peak\n0%\nof max"
    return f"peak\n{100.0 * float(peak) / float(shared):.3g}%\nof max"


def _masked_note(fig):
    handle = Patch(facecolor=MASKED_COLOR, edgecolor="0.4", label="masked")
    fig.legend(handles=[handle], loc="lower center",
               bbox_to_anchor=(0.5, 0.002), frameon=False, handlelength=1.2,
               handletextpad=0.5, borderaxespad=0.0,
               fontsize=plt.rcParams["legend.fontsize"])


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
    field_cmap.set_bad(MASKED_COLOR)
    err_cmap = _truncated_cmap("magma")
    err_cmap.set_bad(MASKED_COLOR)

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
            norm = mcolors.SymLogNorm(linthresh=max(vmax * ERR_GATE, 1e-30),
                                      vmin=-vmax, vmax=vmax, base=10)
            cmap = field_cmap
            cbar_label = f"{part} [{UNITS[part]}] (symlog)"
            ticks = [-vmax, 0.0, vmax]
            ticklabels = [_fmt_sci(-vmax), "0", _fmt_sci(vmax)]
        else:
            vmin, vmax = _error_limits(items)
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            cmap = err_cmap
            cbar_label = "relative error (dimensionless)"
            ticks, ticklabels = _error_ticks(vmin, vmax)
            _assert_masked_distinct(norm, cmap, f"{part} error row")

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
                               colors=[SUPPORT_STYLE["color"]],
                               linewidths=SUPPORT_STYLE["linewidth"],
                               linestyles=SUPPORT_STYLE["linestyle"])
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
    if any(r["kind"] == "error" for r in rows):
        _masked_note(fig)
    man = dict(manifest or {})
    man["colorbars_per_row"] = bar_rows
    man["layout"] = "one shared colour bar per row at row right edge"
    man["masked_color"] = MASKED_COLOR
    man["error_floor"] = ERR_FLOOR
    return save(fig, outdir, name, man)


def _fmt_tick(v):
    return _fmt_sci(v)


def _bar_title(fig, cax, image, ticks, ticklabels, title):
    """Compact colour bar: numeric ticks left, rotated quantity label right.

    The two text columns are separate so each can be sized and kept out of the
    neighbouring image panels; putting both on the same side overran the
    squared-field panel in the old layout.
    """
    cb = fig.colorbar(image, cax=cax)
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticklabels)
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.yaxis.set_label_position("right")
    cb.ax.tick_params(labelsize=plt.rcParams["xtick.labelsize"],
                      labelleft=True, labelright=False, pad=1.5, length=2)
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
    # Slots in figure fractions (figure width tuned for 9 pt text):
    #   pwg  panel-to-panel gap for the horizontal tick labels;
    #   bg   panel-to-bar gap for the colour-bar tick labels (left side);
    #   ba   bar-to-next-panel gap for the rotated colour-bar label (right);
    #   R    final-bar-to-edge gap for its rotated label.
    pwg, bg, bw, ba, R = 0.040, 0.098, 0.012, 0.050, 0.035
    pw = (right - left - 2 * pwg - 2 * (bg + bw) - ba - R) / 4.0
    xs = [left, left + pw + pwg, left + 2 * (pw + pwg)]
    cb1 = xs[-1] + pw + bg
    p4 = cb1 + bw + ba
    cb2 = p4 + pw + bg
    assert cb2 + bw + R <= right + 1e-9
    panel_x = [xs[0], xs[1], xs[2], p4]

    top = 0.950
    # Error rows carry the shared "masked" legend at the very bottom; reserve
    # enough margin that it cannot touch the x tick labels or axis labels.
    bottom = 0.170 if any(r["kind"] == "error" for r in rows) else 0.105
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

    field_cmap = plt.get_cmap("RdBu_r").copy()
    field_cmap.set_bad(MASKED_COLOR)
    sq_cmap = plt.get_cmap("viridis").copy()
    # Zero is the physical log floor of the squared panel, not a mask: show it
    # with the colormap's own lowest colour.
    sq_cmap.set_bad(sq_cmap(0.0))
    sq_cmap.set_under(sq_cmap(0.0))
    err_cmap = _truncated_cmap("magma")
    err_cmap.set_bad(MASKED_COLOR)
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
            # One shared symmetric-log bar for the triad: a real but suppressed
            # component (for example the longitudinal E_x at gamma=10) keeps
            # visible structure instead of collapsing onto the white centre.
            cnorm = mcolors.SymLogNorm(linthresh=max(cvmax * ERR_GATE, 1e-30),
                                       vmin=-cvmax, vmax=cvmax, base=10)
            svmax = float(np.nanmax(np.abs(np.asarray(sqitem["arr"], float))))
            if not np.isfinite(svmax) or svmax <= 0:
                svmax = 1.0
            sfloor = max(svmax * 10.0 ** (-SQ_LOG_DECADES), 1e-30)
            snorm = mcolors.LogNorm(vmin=sfloor, vmax=svmax)
            ccmap, scmap = field_cmap, sq_cmap
            cticks = [-cvmax, 0.0, cvmax]
            clabels = [_fmt_sci(-cvmax), "0", _fmt_sci(cvmax)]
            sticks, slabels = _sq_ticks(svmax)
            clabel = f"{part} ({UNITS[part]}) (symlog)"
            unit2 = "V$^2$/m$^2$" if part == "E" else "T$^2$"
            slabel = f"{part}$^2$ ({unit2}) (log)"
        else:
            cvmin, cvmax = _error_limits(comps)
            svmin, svmax = _error_limits([sqitem])
            cnorm = mcolors.LogNorm(vmin=cvmin, vmax=cvmax)
            snorm = mcolors.LogNorm(vmin=svmin, vmax=svmax)
            ccmap = scmap = err_cmap
            cticks, clabels = _error_ticks(cvmin, cvmax)
            sticks, slabels = _error_ticks(svmin, svmax)
            clabel = slabel = "relative error"
            _assert_masked_distinct(cnorm, ccmap, f"{part} error components")
            _assert_masked_distinct(snorm, scmap, f"{part}$^2$ error")

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
                if not bool(np.any(v & np.isfinite(e))):
                    # A fully masked error panel: say why, so the grey cannot
                    # be read as missing data.  An identically-zero reference
                    # (peak 0) makes the relative error undefined; a zero
                    # component among live ones is named in the item
                    # annotation.
                    note = it.get("annotation",
                                  "fully masked: relative error undefined")
                    # Keep the note inside its own panel: a long single line
                    # would spill across the neighbouring axes and trip the
                    # layout guard.  The caption carries the full reason, so
                    # the in-panel note is the short component identity.
                    short = note.split(" (")[0]
                    note = short if len(short) <= 30 else "fully masked"
                    ax.text(0.5, 0.5, note, transform=ax.transAxes,
                            ha="center", va="center",
                            fontsize=min(plt.rcParams["font.size"], 6.5),
                            color="0.25")
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
                                   colors=[SUPPORT_STYLE["color"]],
                                   linewidths=SUPPORT_STYLE["linewidth"],
                                   linestyles=SUPPORT_STYLE["linestyle"])
                    except Exception:
                        pass
            ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
            ax.set_aspect("auto")
            ax.set_title(title, pad=2.5,
                         fontsize=plt.rcParams["axes.titlesize"])
            ax.tick_params(length=2, pad=1.5)
            # MaxNLocator(4, prune="both") keeps ticks strictly inside the
            # visible limits and away from the panel corners.  The old
            # MaxNLocator(3) emitted a 7.5 tick beyond a 6.4-wide x range whose
            # label spilled into the next panel, and edge ticks collided with
            # the first tick label of the neighbouring axis.
            ax.xaxis.set_major_locator(mticker.MaxNLocator(4, prune="both"))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(4, prune="both"))

        for j, it in enumerate(comps):
            ax = fig.add_axes([panel_x[j], y0, pw, rh])
            panel(ax, it, cnorm, ccmap, it.get("title", ""))
            if kind == "field":
                arr_c = np.asarray(it["arr"], float)
                peak_c = (float(np.nanmax(np.abs(arr_c)))
                          if np.isfinite(arr_c).any() else 0.0)
                ax.text(0.03, 0.965, _peak_fraction(peak_c, cvmax),
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=plt.rcParams["font.size"], color="0.10",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                  ec="none", alpha=0.72))
            if ri == nrow - 1:
                ax.set_xlabel(xlabel)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.tick_params(labelleft=False)
        axq = fig.add_axes([panel_x[3], y0, pw, rh])
        sq_title = sqitem.get("title", "")
        if kind == "field":
            sq_title = sq_title + " (log)"
        panel(axq, sqitem, snorm, scmap, sq_title)
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
    if any(r["kind"] == "error" for r in rows):
        _masked_note(fig)
    man = dict(manifest or {})
    man["colorbars_per_row"] = bar_rows
    man["layout"] = "component triad bar, squared panel with its own bar"
    man["component_norm"] = "symlog, linthresh=1e-3*shared component maximum"
    man["component_peak_annotation"] = "each panel states its peak as % of shared max"
    man["squared_norm"] = "log, 1e-6*peak..peak (field rows)"
    man["masked_color"] = MASKED_COLOR
    man["error_floor"] = ERR_FLOOR
    man["error_gate_tick"] = ERR_GATE
    return save(fig, outdir, name, man)
