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

# ---- Shared font ladder -------------------------------------------------
# The body text is 10 pt.  No figure text may exceed it; axis labels stay at
# 9 pt and everything secondary is smaller, so the data panels get the room.
BODY_TEXT_PT = 10.0
FIG_TEXT_PT = 8.0       # figure-level notes / row labels
AXIS_LABEL_PT = 9.0     # x/y axis labels and colour-bar quantity labels
TITLE_PT = 8.0          # in-panel titles
TICK_PT = 7.5           # axis and colour-bar tick labels
ANNOT_PT = 7.0          # in-panel annotations (peak labels, blank notes)
LEGEND_PT = 7.5

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


# ---- Shared field-figure geometry --------------------------------------
# Every field family (gp_*, val_*, large_*) uses this one grid: four square
# data panels per row (three signed components plus the squared magnitude),
# two colour bars per row, and fixed inch margins/gaps.  The horizontal
# layout is fixed in figure fractions, so the panel width in inches is fixed
# by the canvas width.  The figure height is then solved from the number of
# rows so that every panel has the same physical width and height (aspect
# ratio exactly 1).  Only the height grows with the row count.
FIELD_LEFT = 0.105          # left margin: y label + row labels
FIELD_RIGHT = 0.995
FIELD_PANEL_GAP = 0.008     # panel-to-panel gap (horizontal tick labels)
FIELD_BAR_GAP = 0.098       # panel-to-colour-bar gap (bar tick labels)
FIELD_BAR_WIDTH = 0.012
FIELD_BAR_LABEL_GAP = 0.032  # bar-to-next-panel / bar-to-edge (rotated label)
FIELD_EDGE_GAP = 0.032

FIELD_TOP_IN = 0.30          # above the first row (panel titles)
FIELD_BOTTOM_IN = 0.46       # below the last row (ticks + x label)
FIELD_BOTTOM_ERR_IN = 0.66   # extra room for the shared "masked" legend
FIELD_ROW_GAP_IN = 0.48      # enough for title, ticks and a row x label
FIELD_TOP_NOTE_IN = 0.55     # per-gamma top note
FIELD_BOTTOM_NOTE_IN = 0.86  # per-gamma bottom note (omitted B error row)

FIELD_NCOL = 4


def field_panel_width_frac():
    """Width of one of the four data panels, as a figure fraction."""
    reserved = (FIELD_LEFT + 2 * FIELD_PANEL_GAP
                + 2 * (FIELD_BAR_GAP + FIELD_BAR_WIDTH)
                + FIELD_BAR_LABEL_GAP + FIELD_EDGE_GAP
                + (1.0 - FIELD_RIGHT))
    return (1.0 - reserved) / FIELD_NCOL


def field_layout(nrows, width_frac=1.0, top_in=FIELD_TOP_IN,
                 bottom_in=FIELD_BOTTOM_IN, row_gap_in=FIELD_ROW_GAP_IN):
    """Unified square-panel layout for a field figure.

    Returns a dict with the canvas size, the four panel x positions, the
    panel width/height (both in figure fractions and in inches), the colour
    bar positions and the y origin of every row.  ``panel_width_in`` equals
    ``panel_height_in`` by construction, independent of ``nrows``.
    """
    nrows = int(nrows)
    if nrows < 1:
        raise ValueError("field_layout needs at least one row")
    fig_w_in = width_in(width_frac)
    pw = field_panel_width_frac()
    panel_in = pw * fig_w_in
    fig_h_in = (float(top_in) + nrows * panel_in
                + (nrows - 1) * float(row_gap_in) + float(bottom_in))
    top = 1.0 - float(top_in) / fig_h_in
    bottom = float(bottom_in) / fig_h_in
    rh = panel_in / fig_h_in
    gap = float(row_gap_in) / fig_h_in

    xs = [FIELD_LEFT,
          FIELD_LEFT + pw + FIELD_PANEL_GAP,
          FIELD_LEFT + 2 * (pw + FIELD_PANEL_GAP)]
    cb1 = xs[-1] + pw + FIELD_BAR_GAP
    p4 = cb1 + FIELD_BAR_WIDTH + FIELD_BAR_LABEL_GAP
    cb2 = p4 + pw + FIELD_BAR_GAP
    if cb2 + FIELD_BAR_WIDTH + FIELD_EDGE_GAP > FIELD_RIGHT + 1e-9:
        raise AssertionError("field_layout: colour bars overrun the canvas")
    panel_x = [xs[0], xs[1], xs[2], p4]

    row_y = []
    yy = top - rh
    for _ in range(nrows):
        row_y.append(yy)
        yy -= gap + rh
    return {
        "fig_w_in": fig_w_in, "fig_h_in": fig_h_in,
        "width_frac": float(width_frac),
        "panel_x": panel_x, "pw": pw, "rh": rh,
        "panel_width_in": panel_in, "panel_height_in": rh * fig_h_in,
        "panel_aspect": (rh * fig_h_in) / panel_in,
        "cb1": cb1, "cb2": cb2, "bar_width": FIELD_BAR_WIDTH,
        "row_y": row_y, "top": top, "bottom": bottom, "row_gap": gap,
        "nrows": nrows,
    }


def _layout_manifest(geo):
    """Record the measured square-panel geometry and the font ladder."""
    return {
        "panel_width_in": round(geo["panel_width_in"], 4),
        "panel_height_in": round(geo["panel_height_in"], 4),
        "panel_aspect": round(geo["panel_aspect"], 4),
        "font_ladder_pt": {
            "figure_text": FIG_TEXT_PT, "axis_label": AXIS_LABEL_PT,
            "title": TITLE_PT, "tick": TICK_PT, "annotation": ANNOT_PT,
            "legend": LEGEND_PT, "body_text": BODY_TEXT_PT,
        },
    }


def draw_field_panel(ax, x, y, arr, cmap, norm, shading="nearest"):
    """Draw a field raster with no white seams between adjacent pixels.

    ``pcolormesh`` anti-aliases cell edges by default; on a fine raster this
    leaves hair-width white seams at 300 dpi.  Cell edges are disabled and
    anti-aliasing is turned off so neighbouring cells share an exact edge.
    """
    mesh = ax.pcolormesh(x, y, arr, cmap=cmap, norm=norm, shading=shading,
                         antialiased=False, edgecolors="none",
                         linewidth=0.0)
    mesh.set_edgecolor("none")
    mesh.set_antialiased(False)
    return mesh


def apply_style(base=9.0):
    """Deterministic rcParams for the non-field (line/bar) figures.

    The field families use :func:`apply_field_style`; the line/bar figures
    keep their established 9 pt look.
    """
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


def apply_field_style():
    """Apply the shared field-figure font ladder (the one style for gp/val/large).

    The body text is 10 pt; axis labels are 9 pt, everything else 7-8 pt, so
    no figure text can exceed the body text and the square data panels get
    the room.
    """
    apply_style(base=FIG_TEXT_PT)
    plt.rcParams.update({
        "font.size": FIG_TEXT_PT,
        "axes.titlesize": TITLE_PT,
        "axes.labelsize": AXIS_LABEL_PT,
        "xtick.labelsize": TICK_PT,
        "ytick.labelsize": TICK_PT,
        "legend.fontsize": LEGEND_PT,
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


def save(fig, outdir, name, manifest, tight=True):
    """Save PDF+PNG and audit JSON.

    Field figures pass ``tight=False`` so the PDF canvas is exactly the shared
    figure width.  LaTeX then includes it at ``1:1`` scale and every family
    prints the same physical square panel.  Other figures keep the historical
    tight crop.
    """
    assert_no_layout_collisions(fig, outdir, name)
    os.makedirs(outdir, exist_ok=True)
    audit_dir = os.path.join(outdir, "audit")
    os.makedirs(audit_dir, exist_ok=True)
    pdf = os.path.join(outdir, name + ".pdf")
    png = os.path.join(outdir, name + ".png")
    meta_pdf = {"CreationDate": FIXED_DATE_OBJ, "ModDate": FIXED_DATE_OBJ}
    meta_png = {"CreationDate": FIXED_DATE, "ModDate": FIXED_DATE}
    if tight:
        fig.savefig(pdf, metadata=meta_pdf, bbox_inches="tight",
                    pad_inches=0.03)
        fig.savefig(png, dpi=300, metadata=meta_png, bbox_inches="tight",
                    pad_inches=0.03)
    else:
        fig.savefig(pdf, metadata=meta_pdf)
        fig.savefig(png, dpi=300, metadata=meta_png)
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


# Size of the in-panel peak annotation in points at the final print size.  The
# combined slice panels are much smaller than the per-gamma ones, so a 9 pt
# label covered most of the panel.  5.5 pt keeps the compact two-line label
# below ~22 % of the panel width (checked at render time).
PEAK_ANNOT_PT = ANNOT_PT
PEAK_ANNOT_MAX_WIDTH_FRAC = 0.30


def _peak_fraction(peak, shared):
    """Compact two-line peak label: ``peak`` over the percentage of shared max.

    The percentage is rounded to at most one decimal; peaks below 1 % use one
    significant figure with no leading zero (``.04%``, ``.5%``) so a
    nonzero-but-tiny component is never printed as ``0%`` and the label stays
    inside the panel-width budget.  The panel that sets the shared maximum
    reads ``max`` (``100%`` would widen the label past that budget).
    """
    if not np.isfinite(peak) or shared <= 0:
        return "peak\n0%"
    pct = 100.0 * float(peak) / float(shared)
    if pct == 0.0:
        value = "0%"
    elif pct >= 99.5:
        value = "max"
    elif pct >= 10.0:
        value = f"{pct:.0f}%"
    elif pct >= 1.0:
        value = f"{pct:.1f}%"
    else:
        # One significant figure, no leading zero (".04%", ".5%"), so the
        # label of a faint component still fits the panel-width budget.
        value = f"{pct:.1g}%"
        if value.startswith("0."):
            value = value[1:]
        if value in ("%", ".0%"):
            value = "<.1%"
    return f"peak\n{value}"


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
    apply_field_style()
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
                    draw_field_panel(ax, x, y, arr, cmap, norm, shading)
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
                    draw_field_panel(ax, x, y, masked, cmap, norm, shading)
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
                            width_frac, shading="nearest",
                            support=None, suptitle=None, manifest=None,
                            compact=False, row_gap_in=FIELD_ROW_GAP_IN,
                            bottom_in=None):
    """Field/error rows with a component triad and a separate squared panel.

    Every row has four items: three signed components sharing one colour bar,
    then the squared magnitude on its own scale with its own bar and unit
    label.  Geometry comes from :func:`field_layout`, so every field family
    gets the same square panels; only the figure height changes with the row
    count.  ``compact`` shortens the colour-bar labels and the squared-panel
    log ticks.  ``row_gap_in`` and ``bottom_in`` are inches.
    """
    apply_field_style()
    nrow = len(rows)
    has_error = any(r["kind"] == "error" for r in rows)
    if bottom_in is None:
        bottom_in = FIELD_BOTTOM_ERR_IN if has_error else FIELD_BOTTOM_IN
    geo = field_layout(nrow, width_frac=width_frac, top_in=FIELD_TOP_IN,
                       bottom_in=bottom_in, row_gap_in=row_gap_in)
    fig = plt.figure(figsize=(geo["fig_w_in"], geo["fig_h_in"]))
    panel_x = geo["panel_x"]
    pw, rh, bw = geo["pw"], geo["rh"], geo["bar_width"]
    cb1, cb2 = geo["cb1"], geo["cb2"]
    row_y = geo["row_y"]

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
    peak_texts = []

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
            if compact:
                sfloor_ticks = [snorm.vmin,
                                float(np.sqrt(snorm.vmin * svmax)), svmax]
                sticks = sfloor_ticks
                slabels = [_fmt_sci(t) for t in sticks]
            else:
                sticks, slabels = _sq_ticks(svmax)
            unit2 = "V$^2$/m$^2$" if part == "E" else "T$^2$"
            if compact:
                clabel = f"{part} ({UNITS[part]})"
                slabel = f"{part}$^2$ {unit2}"
            else:
                clabel = f"{part} ({UNITS[part]}) (symlog)"
                slabel = f"{part}$^2$ ({unit2}) (log)"
        else:
            cvmin, cvmax = _error_limits(comps)
            svmin, svmax = _error_limits([sqitem])
            cnorm = mcolors.LogNorm(vmin=cvmin, vmax=cvmax)
            snorm = mcolors.LogNorm(vmin=svmin, vmax=svmax)
            ccmap = scmap = err_cmap
            cticks, clabels = _error_ticks(cvmin, cvmax)
            sticks, slabels = _error_ticks(svmin, svmax)
            clabel = slabel = "rel. error" if compact else "relative error"
            _assert_masked_distinct(cnorm, ccmap, f"{part} error components")
            _assert_masked_distinct(snorm, scmap, f"{part}$^2$ error")

        def panel(ax, it, norm, cmap, title):
            arr = np.asarray(it["arr"], float)
            if kind == "field":
                if it.get("blank"):
                    # An identically-zero reference component: draw the panel
                    # blank and say which quantity vanishes.  The caption
                    # carries the physical reason (beam axis or stationary).
                    ax.set_facecolor("0.90")
                    ax.text(0.5, 0.5, it.get("blank_text", ""),
                            transform=ax.transAxes, ha="center", va="center",
                            fontsize=plt.rcParams["font.size"], color="0.25")
                else:
                    draw_field_panel(ax, x, y, arr, cmap, norm, shading)
            else:
                e = np.asarray(it["err"], float)
                v = np.asarray(it["valid"], bool)
                z = np.ma.masked_where(~v | ~np.isfinite(e), e)
                draw_field_panel(ax, x, y, z, cmap, norm, shading)
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
                            fontsize=ANNOT_PT, color="0.25")
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
            if kind == "field" and not it.get("blank"):
                arr_c = np.asarray(it["arr"], float)
                peak_c = (float(np.nanmax(np.abs(arr_c)))
                          if np.isfinite(arr_c).any() else 0.0)
                peak_texts.append((ax.text(
                    0.03, 0.965, _peak_fraction(peak_c, cvmax),
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=PEAK_ANNOT_PT, color="0.10",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec="none", alpha=0.72)), ax))
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

    # The compact peak label must stay a small fraction of the panel.  Measure
    # the drawn text against its own axes and fail loudly if it grows past the
    # budget (for example after a font or format change).
    fig.canvas.draw()
    _renderer = fig.canvas.get_renderer()
    peak_fracs = []
    for text, tax in peak_texts:
        tw = float(text.get_window_extent(_renderer).width)
        aw = float(tax.get_window_extent(_renderer).width)
        frac = tw / aw if aw > 0 else 0.0
        peak_fracs.append(frac)
        if frac > PEAK_ANNOT_MAX_WIDTH_FRAC + 1e-9:
            raise AssertionError(
                f"{name}: peak annotation {text.get_text()!r} occupies "
                f"{frac:.3f} of the panel width "
                f"(budget {PEAK_ANNOT_MAX_WIDTH_FRAC})")
    man = dict(manifest or {})
    man["colorbars_per_row"] = bar_rows
    man["layout"] = "component triad bar, squared panel with its own bar"
    man["component_norm"] = "symlog, linthresh=1e-3*shared component maximum"
    man["component_peak_annotation"] = (
        "compact two-line peak label: 'peak' over the percentage of shared max "
        "(one significant figure below 1%, 'max' for the shared maximum)")
    man["peak_annotation_font_pt"] = PEAK_ANNOT_PT
    man["peak_annotation_max_panel_frac"] = (
        round(max(peak_fracs), 4) if peak_fracs else None)
    man["squared_norm"] = "log, 1e-6*peak..peak (field rows)"
    man["masked_color"] = MASKED_COLOR
    man["error_floor"] = ERR_FLOOR
    man["error_gate_tick"] = ERR_GATE
    man.update(_layout_manifest(geo))
    return save(fig, outdir, name, man, tight=False)
