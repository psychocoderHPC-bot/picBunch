"""Canonical style module for every paper figure.

This is the single source of truth for canvas geometry, the font ladder, the
Okabe-Ito palette, colormaps and normalisation classes, the raster policy, the
colour-bar style, and the deterministic save path.  Renderers must not set
style locally: :func:`apply_paper_style` is the only rcParam entry point, and
:func:`draw_raster_panel`, :func:`cmap_for`, :func:`draw_colorbar`,
:func:`figure`, :func:`subplots`, :func:`blank_panel` and :func:`panel_note`
are the only style-carrying draw helpers.  ``check_figstyle_overrides.py``
enforces this.

Output is deterministic: fixed backend, fixed PDF metadata date, no RNG.
Audit JSON is written under ``<outdir>/audit/``.

The canvas width is the width at which the figure is included in the paper, so
a 9 pt font in the PDF is a 9 pt font in print (no hidden shrink factor).
"""
import datetime as _dt
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib._mathtext as _matplotlib_mathtext
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.text as mtext
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np

# A4 (210 mm) minus 2 x 25.4 mm geometry margins = 159.2 mm.  LaTeX reports
# this text block as 452.9679 TeX pt; a TeX point is 1/72.27 in, not 1/72 in,
# so the canvas is 452.9679 / 72.27 = 6.2677 in = 159.2 mm and matches the
# \linewidth at which every figure is included (no hidden shrink factor).
TEXT_BLOCK_IN = 452.9679 / 72.27

# ---- Canonical font ladder ---------------------------------------------
# One ladder for every figure family.  The body text is 10 pt; no figure text
# may exceed it.  Axis labels are 9 pt, titles/notes 8 pt, ticks and legends
# 7.5 pt, in-panel annotations 7 pt.
BODY_TEXT_PT = 10.0
FIG_TEXT_PT = 8.0       # figure-level notes / row labels
AXIS_LABEL_PT = 9.0     # x/y axis labels and colour-bar quantity labels
TITLE_PT = 8.0          # in-panel titles
TICK_PT = 7.5           # axis and colour-bar tick labels
ANNOT_PT = 7.0          # in-panel annotations (peak labels, blank notes)
LEGEND_PT = 7.5
FONT_FAMILY = "DejaVu Sans"

# ---- Mathtext script scale ---------------------------------------------
# Matplotlib draws sub/superscripts at 0.7x the base size (its private
# ``SHRINK_FACTOR``).  The font ladder is correct, but at the smallest tokens a
# 0.7x script falls under the 6 pt legibility floor: a 7 pt annotation in a
# figure placed at 0.98x prints at 4.8 pt, and a 7.5 pt log-tick exponent in
# the 0.951x F_multivel canvas prints at 5.0 pt.  A 0.9 script scale lifts the
# worst case to 7.0 * 0.98 * 0.9 = 6.17 pt while every non-script token keeps
# its declared ladder size, so the canvas and the layout budget are unchanged.
# Matplotlib has no rcParam for this; the module constant is read at draw time.
# Fail loudly if the hook ever disappears instead of silently regressing.
MATH_SCRIPT_SCALE = 0.9
if not hasattr(_matplotlib_mathtext, "SHRINK_FACTOR"):
    raise RuntimeError(
        "matplotlib._mathtext.SHRINK_FACTOR is missing: the paper's 6 pt "
        "script floor cannot be enforced")
_matplotlib_mathtext.SHRINK_FACTOR = MATH_SCRIPT_SCALE

# ---- Canonical line/axis tokens ----------------------------------------
LINEWIDTH = 0.8
TICK_DIRECTION = "out"
TICK_LENGTH = 2.5
TICK_WIDTH = 0.6
TICK_PAD = 1.5
GRID = False
TEXT_COLOR = "black"
ANNOT_COLOR = "0.25"

# ---- Deterministic canvas / save contract ------------------------------
SCREEN_DPI = 120
SAVEFIG_DPI = 300
SAVEFIG_BBOX = "tight"
SAVEFIG_PAD_IN = 0.03
SAVEFIG_TRANSPARENT = False
PDF_FONT_TYPE = 42
SVG_FONT_TYPE = "none"
FIXED_DATE = "D:20261102000000+00'00'"
FIXED_DATE_OBJ = _dt.datetime(2026, 11, 2, tzinfo=_dt.timezone.utc)
HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT_DIR = None  # set per save() call: <outdir>/audit

# ---- Canonical colour-bar contract -------------------------------------
# One layout for every vertical bar: ticks on the left, rotated quantity label
# on the right, 0.6 pt outline, 2 pt ticks.
CBAR_TICK_PT = TICK_PT
CBAR_LABEL_PT = AXIS_LABEL_PT
CBAR_LABEL_PAD = 3.0
CBAR_TICK_PAD = 1.5
CBAR_TICK_LENGTH = 2.0
CBAR_OUTLINE_WIDTH = 0.6
CBAR_TICK_SIDE = "left"
CBAR_LABEL_SIDE = "right"
CBAR_PAD = 0.02
CBAR_FRACTION = 0.05
# Attached bars are inset from the donor axes so the lowest/highest tick label
# cannot touch the donor's corner tick labels.
CBAR_SHRINK = 0.96
CBAR_ANCHOR = (0.5, 1.0)

# ---- Canonical raster policy -------------------------------------------
# ONE policy for every raster field, in every figure family.  A field array is
# drawn as an image with nearest-neighbour interpolation at pixel centres; the
# extent is built from the centres plus/minus half a cell, so pixel edges align
# exactly with the physical cell edges.  Image pixels share exact edges, so no
# per-cell antialiasing seam can appear at 300 dpi; nearest-neighbour
# interpolation adds no resampling artefacts.  ``pcolormesh`` was the source of
# hairline white seams because cell edges were anti-aliased.
RASTER_INTERPOLATION = "nearest"
RASTER_RESAMPLE = False
RASTER_ORIGIN = "lower"
RASTER_ASPECT = "auto"

# ---- Canonical colormaps per quantity class ----------------------------
# Signed field components: diverging, white at zero.
CMAP_FIELD_SIGNED = "RdBu_r"
# Squared magnitude: sequential, perceptually uniform.
CMAP_FIELD_SQUARED = "viridis"
# Relative error: sequential, top truncated below near-white so the worst
# finite error cannot read as "no data".
CMAP_ERROR_BASE = "magma"
ERR_CMAP_TOP = 0.90
# Charge density (projection panels): perceptually uniform and distinct from
# the squared-field viridis so the two quantity classes never look alike.
CMAP_DENSITY = "cividis"
# 2D particle-density histograms.
CMAP_HIST = "viridis"

# ONE grey for every masked / invalid / blank pixel.  Renderers must not
# define a second grey.
MASKED_COLOR = "#dddddd"
BLANK_COLOR = MASKED_COLOR
MASKED_LEGEND_EDGE = "0.4"

# ---- Canonical norm / floor rules --------------------------------------
# Error panels use a DATA-DRIVEN lower limit, not a fixed decimal floor.
# The diagnosis of the vanished roundoff star: the fixed 1e-6 floor clamped
# every error below it onto the darkest colour (93 % of valid pixels in the
# gp/val/large slices, 7-10 decades collapsed).  The single policy is now
#     vmin = clamp(round_down_2sf(smallest positive finite error),
#                  ERROR_FLOOR_MIN, ERROR_FLOOR_MAX)
# so a panel whose smallest error is 6.3e-14 gets vmin = 6.3e-14, while the
# floor can never exceed 1e-6 and never drops below 1e-14.  vmax stays the
# row peak (with the 1e-3 gate always representable).  Below vmin is valid
# data and takes the colormap's darkest finite colour; grey is only a mask.
ERROR_FLOOR_MAX = 1e-6
ERROR_FLOOR_MIN = 1e-14
ERROR_FLOOR = ERROR_FLOOR_MAX          # upper bound, kept for manifests
ERROR_GATE = 1e-3
SQ_FLOOR_RATIO = 1e-6
SQ_LOG_DECADES = 6.0
SYMLOG_LINTHRESH_FRAC = 1e-3     # of the shared component maximum

# ---- Canonical panel-letter contract -----------------------------------
# Multi-panel figures prefix the title with "(a) ", left aligned, same pad as
# every in-panel title.  Field families use row labels instead of letters.
PANEL_LETTER_FMT = "({})"
PANEL_TITLE_LOC = "left"
PANEL_TITLE_PAD = 2.5

# ---- Palette -----------------------------------------------------------
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


def width_in(fraction):
    """Inches occupied by ``\\includegraphics[width=<fraction>\\linewidth]``."""
    return TEXT_BLOCK_IN * float(fraction)


# ---- Shared field-figure geometry --------------------------------------
# ONE square raster size for the whole paper: field, error and density-row
# maps all print at RASTER_PANEL_MM.  Every row is identical: four panels
# (three signed axis slices followed by the squared-magnitude value plot) and
# one shared colour bar at the row's right edge.
#
# The width is fixed by the 159.2 mm text block:
#   LEFT 20 + 4*27.8 + 3*GAP 2.0 + BARGAP 14 + BAR 2 + LABEL 6 = 159.20 mm.
# The assert in field_layout() enforces that identity and the right edge.
RASTER_PANEL_MM = 27.80         # == BUNCH_MAP_MM: the density-row map size
RASTER_PANEL_IN = RASTER_PANEL_MM / 25.4
RASTER_LEFT_MM = 20.0           # y tick labels + rotated y label
RASTER_GAP_MM = 2.0             # between component panels
RASTER_BAR_GAP_MM = 14.0        # panel to colour bar (bar tick labels)
RASTER_BAR_MM = 2.0             # colour-bar width
RASTER_BAR_LABEL_MM = 6.0       # colour bar to its rotated quantity label
RASTER_NCOL = 4
RASTER_TEXT_WIDTH_MM = 159.20

FIELD_TOP_IN = 0.30          # above the first row (panel titles)
FIELD_BOTTOM_IN = 0.46       # below the last row (ticks + x label)
FIELD_BOTTOM_ERR_IN = 0.66   # extra room for the shared "masked" legend
FIELD_ROW_GAP_IN = 0.42      # title + ticks between two square rows
FIELD_TOP_NOTE_IN = 0.55     # per-gamma top note
FIELD_BOTTOM_NOTE_IN = 0.86  # per-gamma bottom note

FIELD_NCOL = RASTER_NCOL

# Retained for callers that still pass the height-budget knob.  The paper
# keeps square 27.8 mm panels; a panel that would not fit is split into a
# continued float instead of being squeezed.
APPENDIX_PANEL_HEIGHT_SCALE = 1.0


def field_panel_width_frac(width_frac=1.0):
    """Square panel width as a figure fraction (the shared 27.8 mm constant)."""
    return RASTER_PANEL_IN / width_in(width_frac)


def field_layout(row_panels, width_frac=1.0, top_in=FIELD_TOP_IN,
                 bottom_in=FIELD_BOTTOM_IN, row_gap_in=FIELD_ROW_GAP_IN,
                 panel_scale=1.0, panel_height_scale=1.0):
    """Uniform four-panel grid: four 27.8 mm squares and one shared bar.

    ``row_panels`` has one entry per row and every entry must be
    ``RASTER_NCOL`` (four).  There is no 3-panel or 2-panel reflow: a row is
    three signed components plus the squared-magnitude value, and every row
    uses the same size, gaps and bar placement.  The canvas height is solved
    from the row count, so only the height grows.
    """
    row_panels = [int(n) for n in row_panels]
    nrows = len(row_panels)
    if nrows < 1:
        raise ValueError("field_layout needs at least one row")
    for n in row_panels:
        if n != RASTER_NCOL:
            raise ValueError(
                f"a row must hold {RASTER_NCOL} panels, got {n}")
    panel_scale = float(panel_scale)
    if not (0.5 <= panel_scale <= 1.0):
        raise ValueError("panel_scale must be in [0.5, 1.0]")
    panel_height_scale = float(panel_height_scale)
    if not (0.5 <= panel_height_scale <= 1.0):
        raise ValueError("panel_height_scale must be in [0.5, 1.0]")
    fig_w_in = width_in(width_frac)
    panel_in = RASTER_PANEL_IN * panel_scale
    panel_h_in = panel_in * panel_height_scale
    fig_h_in = (float(top_in) + nrows * panel_h_in
                + (nrows - 1) * float(row_gap_in) + float(bottom_in))
    top = 1.0 - float(top_in) / fig_h_in
    rh = panel_h_in / fig_h_in
    gap = float(row_gap_in) / fig_h_in

    panel_f = panel_in / fig_w_in
    left_f = (RASTER_LEFT_MM / 25.4) / fig_w_in
    gap_f = (RASTER_GAP_MM / 25.4) / fig_w_in
    bar_gap_f = (RASTER_BAR_GAP_MM / 25.4) / fig_w_in
    bar_f = (RASTER_BAR_MM / 25.4) / fig_w_in
    label_f = (RASTER_BAR_LABEL_MM / 25.4) / fig_w_in
    xs = [left_f + i * (panel_f + gap_f) for i in range(RASTER_NCOL)]
    bar = xs[-1] + panel_f + bar_gap_f
    right = bar + bar_f + label_f
    # The row arithmetic is the operator contract: one bar per row, exactly
    # the 159.2 mm text width at the shared 27.8 mm panel size.
    total_mm = (RASTER_LEFT_MM + RASTER_NCOL * RASTER_PANEL_MM
                + (RASTER_NCOL - 1) * RASTER_GAP_MM
                + RASTER_BAR_GAP_MM + RASTER_BAR_MM + RASTER_BAR_LABEL_MM)
    if abs(total_mm - RASTER_TEXT_WIDTH_MM) > 1e-6:
        raise AssertionError(
            f"field_layout: row arithmetic {total_mm:.4f} mm != "
            f"{RASTER_TEXT_WIDTH_MM:.2f} mm")
    if right > 1.0 + 1e-9:
        raise AssertionError(
            f"field_layout: the row overruns the canvas (right={right:.6f})")

    row_x = [list(xs) for _ in range(nrows)]
    row_bars = [[bar] for _ in range(nrows)]

    row_y = []
    yy = top - rh
    for _ in range(nrows):
        row_y.append(yy)
        yy -= gap + rh
    return {
        "fig_w_in": fig_w_in, "fig_h_in": fig_h_in,
        "width_frac": float(width_frac),
        "row_panels": row_panels, "row_x": row_x, "row_bars": row_bars,
        "row_y": row_y, "pw": panel_f, "rh": rh,
        "panel_width_in": panel_in, "panel_height_in": rh * fig_h_in,
        "panel_aspect": (rh * fig_h_in) / panel_in,
        "bar_width": bar_f, "bar_gap": bar_gap_f,
        "bar_label_gap": label_f, "left": left_f, "panel_gap": gap_f,
        "top": top, "bottom": float(bottom_in) / fig_h_in,
        "row_gap": gap, "nrows": nrows,
        "panel_scale": panel_scale,
        "panel_height_scale": panel_height_scale,
    }


def _layout_manifest(geo):
    """Record the measured square-panel geometry and the font ladder."""
    return {
        "panel_width_in": round(geo["panel_width_in"], 4),
        "panel_height_in": round(geo["panel_height_in"], 4),
        "panel_width_mm": round(geo["panel_width_in"] * 25.4, 3),
        "panel_height_mm": round(geo["panel_height_in"] * 25.4, 3),
        "panel_aspect": round(geo["panel_aspect"], 4),
        "row_panels": list(geo["row_panels"]),
        "raster_panel_mm": RASTER_PANEL_MM,
        "font_ladder_pt": {
            "figure_text": FIG_TEXT_PT, "axis_label": AXIS_LABEL_PT,
            "title": TITLE_PT, "tick": TICK_PT, "annotation": ANNOT_PT,
            "legend": LEGEND_PT, "body_text": BODY_TEXT_PT,
        },
    }


# ---- Bunch-form first row ----------------------------------------------
# One compact row above an existing field figure: three square |rho|
# projection maps, one marginal-profile axis and one shared colour bar.  The
# sizes are in millimetres, so they are physical at the print scale.  The
# marginal axis absorbs the width left over after the maps, the two label
# gaps, the bar and the outer margins, capped at the 38 mm design width.  The
# planned 170 mm canvas yields the full 38 mm marginal; the repo's actual
# A4 + 1 in-margin text width is 159.2 mm, where the row resolves to the shared panel size
# maps and a ~23 mm marginal with every 7.5 pt tick label unclipped.
BUNCH_MAP_MM = RASTER_PANEL_MM     # one shared size with the field rasters
BUNCH_MARGINAL_MM = 38.0
BUNCH_MARGINAL_MIN_MM = 18.0
BUNCH_BAR_MM = 3.0
BUNCH_GAP_MM = 1.6          # between the maps
# Room for the marginal axis's log y tick labels.  The mathtext script scale
# (MATH_SCRIPT_SCALE) widened the longest ``10^{-16}`` label; 11.0 mm keeps its
# left edge clear of the third map panel at every bunch-figure width.
BUNCH_MARGINAL_GAP_MM = 11.0  # before the marginal axis (its y tick labels)
BUNCH_BAR_GAP_MM = 9.2       # before the bar (its decade tick labels)
BUNCH_LEFT_MM = 10.5        # first map y ticks + rotated y label
BUNCH_RIGHT_MM = 5.0        # rotated colour-bar label
BUNCH_TITLE_IN = 0.16       # panel title above the maps
BUNCH_LABEL_IN = 0.34       # x tick labels + axis label below the maps
# Between the bunch labels and the field titles.  The larger mathtext scripts
# made the marginal axis's bottom log tick label overlap the squared-panel
# title; 0.22 in lifts the bunch row clear of it at every placed scale.
BUNCH_INTER_GAP_IN = 0.22   # between the bunch labels and the field titles
_MM_IN = 1.0 / 25.4


def bunch_row_geometry(fig_w_in):
    """Axis x positions and vertical reserve (inches) for the bunch row.

    Three maps and the marginal axis are square panel height; the colour bar
    shares that height.  The marginal width absorbs the leftover canvas width
    so the row always fits the actual canvas, and it widens towards the 38 mm
    design value on a wider (170 mm) text block.
    """
    fig_w_in = float(fig_w_in)
    map_in = BUNCH_MAP_MM * _MM_IN
    bar_in = BUNCH_BAR_MM * _MM_IN
    gap_in = BUNCH_GAP_MM * _MM_IN
    marg_gap_in = BUNCH_MARGINAL_GAP_MM * _MM_IN
    bar_gap_in = BUNCH_BAR_GAP_MM * _MM_IN
    left_in = BUNCH_LEFT_MM * _MM_IN
    right_in = BUNCH_RIGHT_MM * _MM_IN
    gaps = (2.0 * gap_in + marg_gap_in + bar_gap_in)
    avail = (fig_w_in - left_in - right_in - 3.0 * map_in - bar_in - gaps)
    marg_in = min(BUNCH_MARGINAL_MM * _MM_IN,
                  max(BUNCH_MARGINAL_MIN_MM * _MM_IN, avail))
    x = [left_in]
    x.append(x[-1] + map_in + gap_in)
    x.append(x[-1] + map_in + gap_in)
    x.append(x[-1] + map_in + marg_gap_in)   # marginal axis
    x.append(x[-1] + marg_in + bar_gap_in)   # colour bar
    x_frac = [v / fig_w_in for v in x]
    w_frac = [map_in / fig_w_in] * 3 + [marg_in / fig_w_in,
                                        bar_in / fig_w_in]
    reserve_in = (BUNCH_INTER_GAP_IN + BUNCH_LABEL_IN + map_in
                  + BUNCH_TITLE_IN)
    return {
        "x": x_frac, "width": w_frac,
        "map_in": map_in, "marg_in": marg_in, "bar_in": bar_in,
        "gap_in": gap_in, "marg_gap_in": marg_gap_in,
        "bar_gap_in": bar_gap_in, "left_in": left_in, "right_in": right_in,
        "title_in": BUNCH_TITLE_IN, "label_in": BUNCH_LABEL_IN,
        "inter_gap_in": BUNCH_INTER_GAP_IN, "reserve_in": reserve_in,
    }


def figure_size(width_frac=1.0, height_in=None):
    """Canvas size in inches from the included width fraction and a height.

    ``height_in`` is explicit for line/bar figures.  Field figures pass the
    height solved by :func:`field_layout`, so their square-panel aspect is set
    by the row count rather than by a renderer constant.
    """
    if height_in is None or float(height_in) <= 0:
        raise ValueError("figure_size needs a positive height_in")
    return (width_in(width_frac), float(height_in))


def figure(width_frac=1.0, height_in=None, **kwargs):
    """Create a figure on the canonical canvas (single style entry point)."""
    return plt.figure(figsize=figure_size(width_frac, height_in), **kwargs)


def subplots(width_frac=1.0, height_in=None, nrows=1, ncols=1, **kwargs):
    """Create subplots on the canonical canvas (single style entry point)."""
    return plt.subplots(nrows, ncols,
                        figsize=figure_size(width_frac, height_in), **kwargs)


_APPLIED_KIND = "line"


def apply_paper_style(kind="line"):
    """Apply the one paper rcParam ladder; the only style entry point.

    Parameters
    ----------
    kind : {"line", "field"}
        ``"line"`` covers line, bar, scatter and histogram figures;
        ``"field"`` covers raster field figures.  Both share the same font
        ladder, line widths, tick geometry, grid state and colormap policy.
        The kind selects only the raster rcParams (``image.*``), which are
        inert for line figures; keeping them set unconditionally means an
        accidental raster call cannot escape the policy.

    Returns
    -------
    dict
        The resolved token dictionary, for audit manifests.
    """
    global _APPLIED_KIND
    if kind not in ("line", "field"):
        raise ValueError(f"unknown paper style kind {kind!r}")
    _APPLIED_KIND = kind
    plt.rcParams.update({
        "figure.dpi": SCREEN_DPI,
        "savefig.dpi": SAVEFIG_DPI,
        "savefig.bbox": SAVEFIG_BBOX,
        "savefig.pad_inches": SAVEFIG_PAD_IN,
        "savefig.transparent": SAVEFIG_TRANSPARENT,
        "font.family": FONT_FAMILY,
        "mathtext.fontset": "dejavusans",
        "font.size": FIG_TEXT_PT,
        "axes.titlesize": TITLE_PT,
        "axes.labelsize": AXIS_LABEL_PT,
        "xtick.labelsize": TICK_PT,
        "ytick.labelsize": TICK_PT,
        "legend.fontsize": LEGEND_PT,
        "axes.linewidth": LINEWIDTH,
        "axes.grid": GRID,
        "axes.edgecolor": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "xtick.direction": TICK_DIRECTION,
        "ytick.direction": TICK_DIRECTION,
        "xtick.major.size": TICK_LENGTH,
        "ytick.major.size": TICK_LENGTH,
        "xtick.major.width": TICK_WIDTH,
        "ytick.major.width": TICK_WIDTH,
        "xtick.minor.size": TICK_LENGTH * 0.6,
        "ytick.minor.size": TICK_LENGTH * 0.6,
        "xtick.minor.width": TICK_WIDTH * 0.75,
        "ytick.minor.width": TICK_WIDTH * 0.75,
        "xtick.major.pad": TICK_PAD,
        "ytick.major.pad": TICK_PAD,
        "lines.linewidth": LINEWIDTH,
        "patch.linewidth": LINEWIDTH,
        "legend.frameon": True,
        "image.interpolation": RASTER_INTERPOLATION,
        "image.resample": RASTER_RESAMPLE,
        "image.origin": RASTER_ORIGIN,
        "image.aspect": RASTER_ASPECT,
        "pdf.fonttype": PDF_FONT_TYPE,
        "ps.fonttype": PDF_FONT_TYPE,
        "svg.fonttype": SVG_FONT_TYPE,
    })
    return style_tokens(kind)


def style_tokens(kind=None):
    """Return the resolved style token dictionary for an audit manifest."""
    kind = _APPLIED_KIND if kind is None else kind
    return {
        "kind": kind,
        "font_family": FONT_FAMILY,
        "font_pt": {"body": BODY_TEXT_PT, "figure_text": FIG_TEXT_PT,
                    "axis_label": AXIS_LABEL_PT, "title": TITLE_PT,
                    "tick": TICK_PT, "annotation": ANNOT_PT,
                    "legend": LEGEND_PT, "cbar_tick": CBAR_TICK_PT,
                    "cbar_label": CBAR_LABEL_PT},
        "math_script_scale": MATH_SCRIPT_SCALE,
        "linewidth_pt": LINEWIDTH,
        "tick": {"direction": TICK_DIRECTION, "length_pt": TICK_LENGTH,
                 "width_pt": TICK_WIDTH, "pad_pt": TICK_PAD},
        "grid": GRID,
        "dpi": {"screen": SCREEN_DPI, "savefig": SAVEFIG_DPI},
        "raster": {"interpolation": RASTER_INTERPOLATION,
                   "resample": RASTER_RESAMPLE,
                   "origin": RASTER_ORIGIN, "aspect": RASTER_ASPECT},
        "colormaps": {"signed_field": CMAP_FIELD_SIGNED,
                      "squared_field": CMAP_FIELD_SQUARED,
                      "error": f"{CMAP_ERROR_BASE} (top {ERR_CMAP_TOP})",
                      "density": CMAP_DENSITY, "hist": CMAP_HIST},
        "masked_color": MASKED_COLOR,
        "error_floor_max": ERROR_FLOOR_MAX,
        "error_floor_min": ERROR_FLOOR_MIN,
        "error_gate": ERROR_GATE,
        "sq_floor_ratio": SQ_FLOOR_RATIO,
        "colorbar": {"tick_side": "left (field cax) / right (attached)",
                     "label_side": CBAR_LABEL_SIDE,
                     "pad": CBAR_PAD, "fraction": CBAR_FRACTION,
                     "shrink": CBAR_SHRINK},
        "savefig": {"bbox": SAVEFIG_BBOX, "pad_inches": SAVEFIG_PAD_IN,
                    "transparent": SAVEFIG_TRANSPARENT},
    }


def cmap_for(quantity, zero_floor=False):
    """Return a copy of the canonical colormap for a quantity class.

    ``quantity`` is one of ``signed_field``, ``squared_field``, ``error``,
    ``density`` or ``hist``.  Every true mask uses the one canonical grey.
    ``zero_floor`` is for squared-magnitude panels where zero is a physical
    log floor, not missing data: the *under* colour becomes the colormap's
    own lowest colour.  The *bad* colour stays the one mask grey, so the
    validity mask is visible in the squared panels exactly as in the slice
    and error panels.
    """
    if quantity == "error":
        cmap = _truncated_cmap(CMAP_ERROR_BASE)
    elif quantity == "signed_field":
        cmap = plt.get_cmap(CMAP_FIELD_SIGNED).copy()
    elif quantity == "squared_field":
        cmap = plt.get_cmap(CMAP_FIELD_SQUARED).copy()
    elif quantity == "density":
        cmap = plt.get_cmap(CMAP_DENSITY).copy()
    elif quantity == "hist":
        cmap = plt.get_cmap(CMAP_HIST).copy()
    else:
        raise ValueError(f"unknown quantity class {quantity!r}")
    cmap.set_bad(MASKED_COLOR)
    # Values below vmin are valid data (a physical zero floor or
    # roundoff-small errors), not missing cells: use the darkest finite
    # colour, never the mask grey.
    if zero_floor or quantity == "error":
        cmap.set_under(cmap(0.0))
    else:
        cmap.set_under(MASKED_COLOR)
    return cmap


def _cell_edges(coord, name):
    """Cell edges from uniformly spaced pixel centres."""
    coord = np.asarray(coord, float)
    if coord.ndim != 1 or coord.size < 2:
        raise ValueError(f"{name} needs a 1-D vector of at least 2 centres")
    step = np.diff(coord)
    if not np.allclose(step, step[0], rtol=1e-6, atol=0.0):
        raise ValueError(f"{name} centres are not uniformly spaced")
    half = 0.5 * float(step[0])
    return float(coord[0] - half), float(coord[-1] + half)


def draw_raster_panel(ax, x, y, arr, quantity, norm=None, cmap=None):
    """Draw one field raster with the single canonical policy.

    ``x`` and ``y`` are the **pixel centres** along the horizontal and vertical
    axis (lengths equal the array's column and row count).  The extent is built
    from the centres plus/minus half a cell, so the outer pixel edges coincide
    exactly with the physical cell edges.  ``imshow`` with nearest-neighbour
    interpolation and no resampling draws every cell as an exact block with
    shared edges, which is seam-free at 300 dpi and introduces no interpolation
    artefacts.  This is the only raster path in the paper.
    """
    # Keep the mask: np.asarray() strips it, so a masked error raster would be
    # drawn from the underlying data and its invalid far field would fall
    # through to the colormap's under colour (near-black) instead of the
    # canonical mask grey.  np.ma.asarray() preserves the mask all the way to
    # imshow, where ``set_bad(MASKED_COLOR)`` is applied in memory and the
    # grey reaches the PDF palette as a real image colour.
    arr = np.ma.asarray(arr)
    if arr.ndim != 2:
        raise ValueError("draw_raster_panel needs a 2-D array")
    xlo, xhi = _cell_edges(x, "x")
    ylo, yhi = _cell_edges(y, "y")
    if arr.shape != (np.asarray(y).size, np.asarray(x).size):
        raise ValueError(
            f"raster shape {arr.shape} does not match "
            f"(len(y), len(x)) = {(np.asarray(y).size, np.asarray(x).size)}")
    cmap = cmap_for(quantity) if cmap is None else cmap
    return ax.imshow(arr, origin=RASTER_ORIGIN,
                     extent=(xlo, xhi, ylo, yhi), aspect=RASTER_ASPECT,
                     cmap=cmap, norm=norm,
                     interpolation=RASTER_INTERPOLATION,
                     resample=RASTER_RESAMPLE)


def hist2d(ax, x, y, bins, norm=None, cmin=None):
    """Draw a 2-D density histogram with the canonical histogram colormap."""
    return ax.hist2d(x, y, bins=bins, cmin=cmin, norm=norm,
                     cmap=cmap_for("hist"))


def blank_panel(ax, text):
    """Draw a blank/fully-masked panel with the canonical grey and note.

    This is the ONLY path for a panel with no valid pixel.  Sending an
    all-masked/NaN array to ``imshow`` can fall through to the under colour in
    the PDF backend and print the panel near-black; the grey axes face cannot
    be bypassed.  The frame uses the annotation grey so the box stays neutral.
    """
    ax.set_facecolor(BLANK_COLOR)
    for spine in ax.spines.values():
        spine.set_color(ANNOT_COLOR)
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=ANNOT_PT,
            color=ANNOT_COLOR, transform=ax.transAxes)


def panel_note(ax, text, x=0.5, y=0.5, ha="center", va="center"):
    """Draw an in-panel annotation on the canonical annotation ladder."""
    return ax.text(x, y, text, ha=ha, va=va, fontsize=ANNOT_PT,
                   color=ANNOT_COLOR, transform=ax.transAxes)


def zero_field_note(symbol, reason):
    """Two-line note for an identically-zero field box."""
    return rf"${symbol}\equiv0$" "\n" f"({reason})"


def zero_error_note(symbol, reason):
    """Four-line 7 pt note for a fully masked undefined-error box.

    The panel is 27.8 mm wide, so the sentence is broken into short lines to
    stay inside it at the canonical 7 pt annotation size.
    """
    return (rf"${symbol}\equiv0$" "\n" f"({reason})"
            "\nrelative error\nundefined")


def panel_title(ax, letter, title, pad=PANEL_TITLE_PAD):
    """Set a panel title with the canonical ``(a)`` letter prefix, left aligned."""
    label = PANEL_LETTER_FMT.format(letter)
    full = f"{label} {title}".strip()
    return ax.set_title(full, loc=PANEL_TITLE_LOC, pad=pad,
                        fontsize=TITLE_PT)


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
    common = {"transparent": SAVEFIG_TRANSPARENT}
    if tight:
        fig.savefig(pdf, metadata=meta_pdf, bbox_inches=SAVEFIG_BBOX,
                    pad_inches=SAVEFIG_PAD_IN, **common)
        fig.savefig(png, dpi=SAVEFIG_DPI, metadata=meta_png,
                    bbox_inches=SAVEFIG_BBOX,
                    pad_inches=SAVEFIG_PAD_IN, **common)
    elif (manifest or {}).get("raster_panel_mm") is not None:
        # A field-raster figure: save the exact canvas.  rcParams carry
        # savefig.bbox="tight", and bbox_inches=None means "use the rcParam",
        # so switch it off for this save.  LaTeX then includes the PDF at
        # scale 1.0 and every raster panel prints at its native millimetre
        # size.  Figures without a raster panel keep the historical crop.
        _bbox, _pad = plt.rcParams["savefig.bbox"], plt.rcParams["savefig.pad_inches"]
        plt.rcParams["savefig.bbox"] = None
        plt.rcParams["savefig.pad_inches"] = 0.0
        try:
            fig.savefig(pdf, metadata=meta_pdf, **common)
            fig.savefig(png, dpi=SAVEFIG_DPI, metadata=meta_png, **common)
        finally:
            plt.rcParams["savefig.bbox"] = _bbox
            plt.rcParams["savefig.pad_inches"] = _pad
    else:
        fig.savefig(pdf, metadata=meta_pdf, **common)
        fig.savefig(png, dpi=SAVEFIG_DPI, metadata=meta_png, **common)
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
        "style_tokens": style_tokens(),
    })
    with open(os.path.join(audit_dir, name + ".json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    return pdf


def draw_colorbar(fig, mappable, cax=None, ax=None, ticks=None,
                  ticklabels=None, label=None, tick_side=None):
    """Draw a vertical colour bar with the one canonical style.

    Ticks sit on the left of the bar, the quantity label on the right, the
    outline is 0.6 pt and tick geometry comes from the shared ladder.  Field
    figures pass an explicit ``cax`` from :func:`field_layout`; line figures
    pass ``ax`` and get the shared pad/fraction placement.
    """
    if cax is not None:
        cb = fig.colorbar(mappable, cax=cax)
    elif ax is not None:
        cb = fig.colorbar(mappable, ax=ax, pad=CBAR_PAD,
                          fraction=CBAR_FRACTION, shrink=CBAR_SHRINK,
                          anchor=CBAR_ANCHOR)
    else:
        raise ValueError("draw_colorbar needs cax or ax")
    if tick_side is None:
        # Field bars sit between panels, so their ticks face left into the
        # reserved gap; an attached bar on the right edge keeps its ticks on
        # the right so they cannot collide with the donor axes.
        tick_side = CBAR_TICK_SIDE if cax is not None else "right"
    if tick_side not in ("left", "right"):
        raise ValueError(f"unknown colour-bar tick side {tick_side!r}")
    cb.ax.yaxis.set_ticks_position(tick_side)
    cb.ax.yaxis.set_label_position(CBAR_LABEL_SIDE)
    cb.ax.tick_params(labelsize=CBAR_TICK_PT, pad=CBAR_TICK_PAD,
                      length=CBAR_TICK_LENGTH,
                      labelleft=tick_side == "left",
                      labelright=tick_side == "right")
    cb.outline.set_linewidth(CBAR_OUTLINE_WIDTH)
    if ticks is not None:
        cb.set_ticks(ticks)
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)
    elif isinstance(getattr(mappable, "norm", None), mcolors.LogNorm):
        # Log colour bars use plain scientific notation ("1e-10"), the same
        # convention as the field-family bars.  The default LogFormatter emits
        # mathtext ``10^{-n}`` whose exponent is a sub-size script.
        cb.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _pos: fmt_sci(value)))
    if label is not None:
        cb.set_label(label, fontsize=CBAR_LABEL_PT, labelpad=CBAR_LABEL_PAD)
    return cb


def _cbar(fig, cax, norm, cmap, label, ticks, ticklabels):
    """Internal adapter: ScalarMappable colour bar at an explicit cax."""
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    return draw_colorbar(fig, sm, cax=cax, ticks=ticks,
                         ticklabels=ticklabels, label=label)


def fmt_sci(v):
    """Compact two-significant-figure scientific label (``4.2e6``)."""
    v = float(v)
    if v == 0.0:
        return "0"
    e = int(np.floor(np.log10(abs(v))))
    m = v / 10.0 ** e
    return f"{m:.2g}e{e}"


_fmt_sci = fmt_sci


def _truncated_cmap(name, top=ERR_CMAP_TOP, n=1024):
    """Finite colormap truncated below its near-white top end.

    1024 entries (not 256) so a data-driven logarithmic error range of ~18
    decades still resolves >20 distinct levels in one decade; the roundoff
    star is then visible instead of being quantised into a few blocks.
    """
    base = plt.get_cmap(name)
    rgba = base(np.linspace(0.0, float(top), n))
    return mcolors.ListedColormap(rgba, name=f"{name}_trunc")


def finite_min_distance(cmap, colour=MASKED_COLOR):
    """Smallest RGB distance between the finite ramp and the masked grey."""
    masked = np.asarray(mcolors.to_rgba(colour))[:3]
    lut = np.asarray(cmap(np.linspace(0.0, 1.0, 256)))[:, :3]
    return float(np.min(np.sqrt(np.sum((lut - masked) ** 2, axis=1))))


def assert_masked_distinct(norm, cmap, label="error"):
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


_assert_masked_distinct = assert_masked_distinct


def _round_down_2sf(value):
    """Largest number with two significant figures <= ``value``."""
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        return ERROR_FLOOR_MIN
    exp = int(np.floor(np.log10(v)))
    step = 10.0 ** (exp - 1)
    return float(np.floor(v / step) * step)


def error_floor(data_min):
    """The one documented error-floor rule."""
    return min(ERROR_FLOOR_MAX,
               max(ERROR_FLOOR_MIN, _round_down_2sf(data_min)))


def error_limits(items):
    """Data-driven error range: clamped smallest positive error .. row peak.

    ``items`` may be arrays or dictionaries with ``err``/``valid`` entries.
    ``vmin = clamp(round_down_2sf(smallest finite positive relative error),
    ERROR_FLOOR_MIN, ERROR_FLOOR_MAX)``; ``vmax`` is the row peak.  The
    floor uses ``floor_err``/``floor_valid`` when present: the stored measured
    error and the per-component validity mask that the diagnosis used.  One
    policy for every error panel, no per-figure override.  The 1e-3
    acceptance gate stays tick-marked and drawn as a dash-dot contour.
    """
    floor_cells, peak_cells = [], []
    for it in items:
        if isinstance(it, dict):
            v = np.asarray(it["valid"], bool)
            e = np.asarray(it["err"], float)
            peak_cells.append(e[v & np.isfinite(e) & (e > 0)])
            fv = np.asarray(it.get("floor_valid", v), bool)
            fe = np.asarray(it.get("floor_err", it["err"]), float)
            floor_cells.append(fe[fv & np.isfinite(fe) & (fe > 0)])
        else:
            e = np.asarray(it, float)
            e = e[np.isfinite(e) & (e > 0)]
            floor_cells.append(e)
            peak_cells.append(e)
    peak_cells = [e for e in peak_cells if e.size]
    floor_cells = [e for e in floor_cells if e.size]
    if not peak_cells or not floor_cells:
        return ERROR_FLOOR_MAX, max(1.0, ERROR_GATE)
    vmin = error_floor(float(np.concatenate(floor_cells).min()))
    vmax = max(float(np.concatenate(peak_cells).max()), ERROR_GATE)
    if vmax <= vmin:
        vmax = vmin * 10.0
    return vmin, vmax


_error_limits = error_limits


def error_ticks(vmin, vmax):
    """Lower floor, the marked 1e-3 gate, and the ceiling."""
    ticks = {float(vmin), float(vmax)}
    if float(vmin) <= ERROR_GATE <= float(vmax):
        ticks.add(float(ERROR_GATE))
    ticks = sorted(ticks)
    # Single-line labels: a two-line "gate" label collides with the next
    # decade tick when the data-driven range puts the gate close to the peak.
    labels = [fmt_sci(t) for t in ticks]
    return ticks, labels


_error_ticks = error_ticks


def sq_limits(values):
    """Canonical squared-magnitude log range: peak*SQ_FLOOR_RATIO .. peak."""
    arr = np.asarray(values, float)
    finite = arr[np.isfinite(arr)]
    peak = float(finite.max()) if finite.size else 1.0
    if not np.isfinite(peak) or peak <= 0.0:
        peak = 1.0
    floor = max(peak * SQ_FLOOR_RATIO, np.finfo(float).tiny)
    return floor, peak


def sq_ticks(peak, decades=SQ_LOG_DECADES):
    """Decade ticks from peak*10^-decades up to the peak."""
    ticks = [float(peak) * 10.0 ** d for d in np.arange(-int(decades), 1)]
    return ticks, [fmt_sci(t) for t in ticks]


_sq_ticks = sq_ticks


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
    handle = Patch(facecolor=MASKED_COLOR, edgecolor=MASKED_LEGEND_EDGE,
                   label="masked")
    fig.legend(handles=[handle], loc="lower center",
               bbox_to_anchor=(0.5, 0.002), frameon=False, handlelength=1.2,
               handletextpad=0.5, borderaxespad=0.0,
               fontsize=LEGEND_PT)


def render_field_rows(name, outdir, rows, x, y, xlabel, ylabel, width_frac,
                      fig_height, support=None, suptitle=None, manifest=None):
    """Rows of field/error panels with exactly one shared colour bar per row.

    rows : list of dicts
        {"kind": "field"|"error", "items": [{"arr","err","valid","title"}],
         "part": "E"|"B", "log": bool}
    x, y : pixel **centres** along the horizontal / vertical axis, shared by
        every row and every family.  The raster policy is fixed by
        :func:`draw_raster_panel`.
    support : optional bool mask; drawn as a dark outline on error panels.
    """
    apply_paper_style(kind="field")
    ncol = max(len(r["items"]) for r in rows)
    nrow = len(rows)
    fig = figure(width_frac, fig_height)
    left = 0.105
    right = 0.878
    bottom = 0.125
    top = 0.905 if suptitle else 0.91
    gs = fig.add_gridspec(
        nrow, ncol + 1, width_ratios=[1] * ncol + [0.08],
        left=left, right=right, bottom=bottom, top=top,
        wspace=0.16, hspace=0.46,
    )
    field_cmap = cmap_for("signed_field")
    err_cmap = cmap_for("error")
    xlo, xhi = _cell_edges(x, "x")
    ylo, yhi = _cell_edges(y, "y")

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
            norm = mcolors.SymLogNorm(
                linthresh=max(vmax * SYMLOG_LINTHRESH_FRAC, 1e-30),
                vmin=-vmax, vmax=vmax, base=10)
            cmap = field_cmap
            cbar_label = f"{part} [{UNITS[part]}] (symlog)"
            ticks = [-vmax, 0.0, vmax]
            ticklabels = [fmt_sci(-vmax), "0", fmt_sci(vmax)]
        else:
            vmin, vmax = error_limits(items)
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            cmap = err_cmap
            cbar_label = "relative error (dimensionless)"
            ticks, ticklabels = error_ticks(vmin, vmax)
            assert_masked_distinct(norm, cmap, f"{part} error row")

        for ci, it in enumerate(items):
            ax = fig.add_subplot(gs[ri, ci])
            if kind == "field":
                arr = np.asarray(it["arr"], float)
                if np.isfinite(arr).any() and np.nanmax(np.abs(arr)) > 0:
                    draw_raster_panel(ax, x, y, arr, "signed_field",
                                      norm=norm, cmap=cmap)
                else:
                    blank_panel(ax, rf"${it.get('symbol', '')}\equiv0$")
            else:
                arr = np.asarray(it["err"], float)
                valid = np.asarray(it["valid"], bool)
                masked = np.ma.masked_where(~valid | ~np.isfinite(arr), arr)
                if np.any(valid) and np.isfinite(arr[valid]).any():
                    draw_raster_panel(ax, x, y, masked, "error",
                                      norm=norm, cmap=cmap)
                    try:
                        ax.contour(x, y, np.where(valid & np.isfinite(arr),
                                                  arr, np.nan),
                                   levels=[ERROR_GATE],
                                   colors=[GATE_STYLE["color"]],
                                   linewidths=GATE_STYLE["linewidth"],
                                   linestyles=GATE_STYLE["linestyle"])
                    except Exception:
                        pass
                else:
                    blank_panel(ax, "")
            if support is not None and kind == "error":
                try:
                    ax.contour(x, y, support.astype(float), levels=[0.5],
                               colors=[SUPPORT_STYLE["color"]],
                               linewidths=SUPPORT_STYLE["linewidth"],
                               linestyles=SUPPORT_STYLE["linestyle"])
                except Exception:
                    pass
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(ylo, yhi)
            ax.set_aspect("equal")
            ax.set_title(it["title"], pad=PANEL_TITLE_PAD, fontsize=TITLE_PT)
            ax.tick_params(length=TICK_LENGTH, pad=TICK_PAD)
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
                 fontsize=TITLE_PT)
    if any(r["kind"] == "error" for r in rows):
        _masked_note(fig)
    man = dict(manifest or {})
    man["colorbars_per_row"] = bar_rows
    man["layout"] = "one shared colour bar per row at row right edge"
    man["masked_color"] = MASKED_COLOR
    man["error_floor"] = ERROR_FLOOR
    return save(fig, outdir, name, man)


def _fmt_tick(v):
    return fmt_sci(v)


def _bar_title(fig, cax, image, ticks, ticklabels, title):
    """Internal alias for the canonical colour bar at an explicit cax."""
    return draw_colorbar(fig, image, cax=cax, ticks=ticks,
                         ticklabels=ticklabels, label=title)


def render_split_field_rows(name, outdir, rows, x, y, xlabel, ylabel,
                            width_frac, support=None, suptitle=None,
                            manifest=None, compact=False,
                            row_gap_in=FIELD_ROW_GAP_IN, bottom_in=None,
                            bunch=None, panel_scale=1.0, top_in=FIELD_TOP_IN,
                            panel_height_scale=1.0, fixed_ticks=None,
                            row_labels=None, top_note=None, bottom_note=None,
                            peak_annotations=True, collect_rows=None,
                            row_label_x=0.0315):
    """Uniform four-panel rows at the shared density-map size.

    Every input row already has four items: three signed components plus the
    squared magnitude (for a field row) or three component errors plus the
    squared-magnitude relative error (for an error row).  Each row prints at
    once: four ``RASTER_PANEL_MM`` squares and one shared colour bar at the
    right edge.  There is no 3-panel or 2-panel reflow.

    ``fixed_ticks`` overrides the automatic locators (per-gamma figures use
    it).  ``row_labels`` draws one rotated label per output row.
    ``top_note``/``bottom_note`` are figure-level notes.  ``collect_rows``
    receives the per-row normalisation record.
    """
    apply_paper_style(kind="field")
    nrow = len(rows)
    for r in rows:
        if len(r["items"]) != RASTER_NCOL:
            raise ValueError(
                f"{name}: every row needs {RASTER_NCOL} panels, "
                f"row {r.get('part')} {r.get('kind')} has {len(r['items'])}")
    has_error = any(r["kind"] == "error" for r in rows)
    if bottom_in is None:
        bottom_in = FIELD_BOTTOM_ERR_IN if has_error else FIELD_BOTTOM_IN
    geo = field_layout([RASTER_NCOL] * nrow,
                       width_frac=width_frac, top_in=top_in,
                       bottom_in=bottom_in, row_gap_in=row_gap_in,
                       panel_scale=panel_scale,
                       panel_height_scale=panel_height_scale)
    bunch_geo = (bunch.geometry(geo["fig_w_in"])
                 if bunch is not None else None)
    fig_h_in = geo["fig_h_in"] + (bunch_geo["reserve_in"]
                                  if bunch_geo is not None else 0.0)
    fig = figure(width_frac, fig_h_in)
    pw = geo["pw"]
    rh = geo["panel_height_in"] / fig_h_in
    bw = geo["bar_width"]
    if bunch_geo is None:
        row_y = list(geo["row_y"])
    else:
        # The field block is anchored to the bottom; the bunch row adds its
        # reserve above it.
        row_y = [ry * geo["fig_h_in"] / fig_h_in for ry in geo["row_y"]]

    xlo, xhi = _cell_edges(x, "x")
    ylo, yhi = _cell_edges(y, "y")

    field_cmap = cmap_for("signed_field")
    sq_cmap = cmap_for("squared_field", zero_floor=True)
    err_cmap = cmap_for("error")
    bar_rows = []
    peak_texts = []

    def _draw_panel(ax, kind, it, norm, cmap, title, quantity):
        if it.get("undefined"):
            # Identically-zero reference: keep the box, draw it as a neutral
            # masked panel with the canonical grey face and a 7 pt note.
            blank_panel(ax, it.get("blank_text")
                        or it.get("annotation") or "")
        elif kind == "field":
            arr = np.asarray(it["arr"], float)
            v = it.get("valid")
            if it.get("blank"):
                blank_panel(ax, it.get("blank_text") or it.get("note") or "")
            elif v is not None and not bool(np.any(np.asarray(v, bool))):
                # No evaluated pixel: use the grey blank-panel path.  An
                # imshow of a fully masked/NaN array can fall through to the
                # under colour in the PDF backend, which prints near-black.
                blank_panel(ax, it.get("blank_text")
                            or it.get("note") or "not evaluated")
            else:
                # The SAME validity mask as the squared and error panels, so
                # the grey footprint marks exactly the cells that were not
                # evaluated (reference field below the validity threshold).
                if v is not None:
                    arr = np.ma.masked_where(
                        ~np.asarray(v, bool) | ~np.isfinite(arr), arr)
                draw_raster_panel(ax, x, y, arr, quantity, norm=norm,
                                  cmap=cmap)
                if it.get("note"):
                    panel_note(ax, it["note"])
        else:
            e = np.asarray(it["err"], float)
            v = np.asarray(it["valid"], bool)
            if not bool(np.any(v & np.isfinite(e))):
                # Fully invalid error panel: never imshow NaN data.  The
                # grey blank_panel path is the one that is byte-correct in
                # the PDF backend (set_bad is bypassed for an all-NaN image).
                note = (it.get("blank_text") or it.get("annotation")
                        or "fully masked: relative error undefined")
                if not it.get("blank_text"):
                    short = note.split(" (")[0]
                    note = short if len(short) <= 30 else "fully masked"
                blank_panel(ax, note)
            else:
                z = np.ma.masked_where(~v | ~np.isfinite(e), e)
                draw_raster_panel(ax, x, y, z, "error", norm=norm, cmap=cmap)
                try:
                    ax.contour(x, y, np.where(v & np.isfinite(e), e, np.nan),
                               levels=[ERROR_GATE],
                               colors=[GATE_STYLE["color"]],
                               linewidths=GATE_STYLE["linewidth"],
                               linestyles=GATE_STYLE["linestyle"])
                except Exception:
                    pass
                if support is not None:
                    try:
                        ax.contour(x, y, support.astype(float), levels=[0.5],
                                   colors=[SUPPORT_STYLE["color"]],
                                   linewidths=SUPPORT_STYLE["linewidth"],
                                   linestyles=SUPPORT_STYLE["linestyle"])
                    except Exception:
                        pass
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_aspect("auto")
        ax.set_title(title, pad=PANEL_TITLE_PAD, fontsize=TITLE_PT)
        ax.tick_params(length=TICK_LENGTH, pad=TICK_PAD)
        if fixed_ticks is not None:
            ax.set_xticks(fixed_ticks[0])
            ax.set_yticks(fixed_ticks[1])
        else:
            ax.xaxis.set_major_locator(mticker.MaxNLocator(4, prune="both"))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(4, prune="both"))

    for ri, r in enumerate(rows):
        y0 = row_y[ri]
        px = geo["row_x"][ri]
        bars = geo["row_bars"][ri]
        kind, part = r["kind"], r["part"]
        last = (ri == nrow - 1)
        comps = r["items"][:3]
        sq = r["items"][3]
        panel_specs = []          # (ax, kind, item, norm, cmap, quantity)
        bar_specs = []            # (norm, cmap, ticks, labels, label)

        if kind == "field":
            cvmax = 0.0
            for it in comps:
                if it.get("undefined") or it.get("blank"):
                    continue
                a = np.asarray(it["arr"], float)
                if np.isfinite(a).any():
                    cvmax = max(cvmax, float(np.nanmax(np.abs(a))))
            if not np.isfinite(cvmax) or cvmax <= 0:
                cvmax = 1.0
            cnorm = mcolors.SymLogNorm(
                linthresh=max(cvmax * SYMLOG_LINTHRESH_FRAC, 1e-30),
                vmin=-cvmax, vmax=cvmax, base=10)
            cticks = [-cvmax, 0.0, cvmax]
            clabels = [fmt_sci(-cvmax), "0", fmt_sci(cvmax)]
            clabel = (f"{part} ({UNITS[part]})" if compact
                      else f"{part} ({UNITS[part]}) (symlog)")
            for j, it in enumerate(comps):
                ax = fig.add_axes([px[j], y0, pw, rh])
                panel_specs.append((ax, "field", it, cnorm, field_cmap,
                                    "signed_field"))
            # Fourth panel: squared-magnitude value plot on its own six-decade
            # logarithmic scale.  The row carries ONE bar (the component bar);
            # the squared scale is stated in the caption.
            sarr = np.asarray(sq["arr"], float)
            sfloor, svmax = sq_limits(sarr)
            snorm = mcolors.LogNorm(vmin=sfloor, vmax=svmax)
            ax = fig.add_axes([px[3], y0, pw, rh])
            panel_specs.append((ax, "field", sq, snorm, sq_cmap,
                                "squared_field"))
            bar_specs.append((cnorm, field_cmap, cticks, clabels, clabel))
        else:
            cvmin, cvmax = error_limits(r["items"])
            cnorm = mcolors.LogNorm(vmin=cvmin, vmax=cvmax)
            cticks, clabels = error_ticks(cvmin, cvmax)
            clabel = "rel. error" if compact else "relative error"
            assert_masked_distinct(cnorm, err_cmap,
                                   f"{part} error row")
            for j, it in enumerate(r["items"]):
                ax = fig.add_axes([px[j], y0, pw, rh])
                panel_specs.append((ax, "error", it, cnorm, err_cmap,
                                    "squared_field" if j == 3
                                    else "signed_field"))
            bar_specs.append((cnorm, err_cmap, cticks, clabels, clabel))

        for j, (ax, pkind, it, norm, cmap, quantity) in enumerate(panel_specs):
            _draw_panel(ax, pkind, it, norm, cmap, it.get("title", ""),
                        quantity)
            if (pkind == "field" and j < 3 and peak_annotations
                    and not it.get("blank") and not it.get("undefined")):
                arr_c = np.asarray(it["arr"], float)
                peak_c = (float(np.nanmax(np.abs(arr_c)))
                          if np.isfinite(arr_c).any() else 0.0)
                peak_texts.append((ax.text(
                    0.03, 0.965, _peak_fraction(peak_c, cvmax),
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=PEAK_ANNOT_PT, color="0.10",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec="none", alpha=0.72)), ax))
            if last:
                ax.set_xlabel(xlabel)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.tick_params(labelleft=False)

        for bx, (norm, cmap, ticks, labels, label) in zip(bars, bar_specs):
            cax = fig.add_axes([bx, y0, bw, rh])
            im = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            _bar_title(fig, cax, im, ticks, labels, label)
        bar_rows.append({
            "row": kind, "part": part,
            "panels": len(panel_specs), "colourbars": len(bars),
            "labels": [b[4] for b in bar_specs],
            "ticks": [b[2] for b in bar_specs],
            "scales": [[float(b[0].vmin), float(b[0].vmax)]
                       for b in bar_specs],
        })
        if collect_rows is not None:
            collect_rows.append({
                "kind": kind, "part": part,
                "panels": len(panel_specs),
                "colourbars": len(bars),
                "norms": [[float(b[0].vmin), float(b[0].vmax)]
                          for b in bar_specs],
                "labels": [b[4] for b in bar_specs],
            })

    if row_labels is not None:
        if len(row_labels) != nrow:
            raise ValueError(
                f"{name}: {len(row_labels)} row labels for {nrow} rows")
        for ri, text in enumerate(row_labels):
            fig.text(row_label_x, row_y[ri] + rh / 2.0, text,
                     rotation=90, ha="center", va="center", linespacing=1.15)

    if bunch is not None:
        y_bottom_in = (geo["fig_h_in"] + bunch_geo["inter_gap_in"]
                       + bunch_geo["label_in"])
        bunch.draw(fig, geo["fig_w_in"], fig_h_in, y_bottom_in)
    if top_note:
        fig.text(0.5, 0.985, top_note, ha="center", va="center",
                 fontsize=FIG_TEXT_PT)
    if suptitle:
        fig.text(0.5, 0.975, suptitle, ha="center", va="top",
                 fontsize=TITLE_PT)
    if bottom_note:
        fig.text(0.5, 0.035, bottom_note, ha="center", va="center",
                 fontsize=FIG_TEXT_PT)
    if has_error:
        _masked_note(fig)

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

    # Split between the field parts (the E block and the B block) so the
    # caller can place two floats from one canvas without shrinking a panel.
    parts = [r["part"] for r in rows]
    split_row = None
    for k, p in enumerate(parts):
        if p != parts[0]:
            split_row = k
            break
    split_y_in = None
    if split_row is not None:
        y_lo = row_y[split_row - 1]
        y_hi = row_y[split_row] + rh
        split_y_in = 0.5 * (y_lo + y_hi) * fig_h_in

    man = dict(manifest or {})
    man["colorbars_per_row"] = bar_rows
    man["layout"] = ("uniform 27.8 mm square panels; four per row "
                     "(three signed components and the squared-magnitude "
                     "value) with one shared colour bar")
    man["row_roles"] = [r["kind"] for r in rows]
    man["panel_counts"] = [len(r["items"]) for r in rows]
    man["panels_per_row"] = RASTER_NCOL
    man["colorbars_per_row_count"] = [len(r["items"]) and 1 for r in rows]
    man["raster_panel_mm"] = RASTER_PANEL_MM
    man["canvas_width_mm"] = round(fig.get_figwidth() * 25.4, 3)
    man["masked_color"] = MASKED_COLOR
    man["error_floor_max"] = ERROR_FLOOR_MAX
    man["error_floor_min"] = ERROR_FLOOR_MIN
    man["error_gate_tick"] = ERROR_GATE
    man["peak_annotation_font_pt"] = PEAK_ANNOT_PT
    man["peak_annotation_max_panel_frac"] = (
        round(max(peak_fracs), 4) if peak_fracs else None)
    man.update(_layout_manifest(geo))
    man["figure_height_mm"] = round(fig_h_in * 25.4, 2)
    if split_y_in is not None:
        man["split"] = {
            "row": split_row,
            "y_in": round(split_y_in, 4),
            "trim_bottom_bp": round(split_y_in * 72.0, 2),
            "trim_top_bp": round((fig_h_in - split_y_in) * 72.0, 2),
            "note": ("include the same PDF twice: first float trims "
                     "trim_bottom_bp from the bottom, continued float trims "
                     "trim_top_bp from the top"),
        }
    if bunch is not None:
        man["bunch_form"] = bunch.manifest(geo["fig_w_in"])
    return save(fig, outdir, name, man, tight=False)
