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
# Error colourbars span [ERROR_FLOOR, decade ceiling] and always mark the
# 1e-3 acceptance gate.  Squared magnitudes use a log scale floored at
# SQ_FLOOR_RATIO times the panel peak (six decades).
ERROR_FLOOR = 1e-6
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
                 bottom_in=FIELD_BOTTOM_IN, row_gap_in=FIELD_ROW_GAP_IN,
                 panel_scale=1.0):
    """Unified square-panel layout for a field figure.

    Returns a dict with the canvas size, the four panel x positions, the
    panel width/height (both in figure fractions and in inches), the colour
    bar positions and the y origin of every row.  ``panel_width_in`` equals
    ``panel_height_in`` by construction, independent of ``nrows``.

    ``panel_scale`` shrinks every square panel by that factor and re-centres
    the block horizontally.  It is a height-budget knob; at the default 1.0
    the layout is byte-identical to the canonical field grid.
    """
    nrows = int(nrows)
    if nrows < 1:
        raise ValueError("field_layout needs at least one row")
    panel_scale = float(panel_scale)
    if not (0.5 <= panel_scale <= 1.0):
        raise ValueError("panel_scale must be in [0.5, 1.0]")
    fig_w_in = width_in(width_frac)
    pw = field_panel_width_frac() * panel_scale
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
    right_edge = cb2 + FIELD_BAR_WIDTH
    # The canonical layout leaves FIELD_EDGE_GAP at the right edge; at
    # panel_scale == 1.0 this makes shift exactly zero (byte-identical output
    # for every figure that does not request a shrink).
    avail = (FIELD_RIGHT - FIELD_EDGE_GAP) - FIELD_LEFT
    block_w = right_edge - FIELD_LEFT
    if block_w > avail + 1e-9:
        raise AssertionError("field_layout: colour bars overrun the canvas")
    shift = (avail - block_w) / 2.0
    if shift > 1e-12:
        xs = [x + shift for x in xs]
        cb1 += shift
        p4 += shift
        cb2 += shift
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
        "nrows": nrows, "panel_scale": panel_scale,
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


# ---- Bunch-form first row ----------------------------------------------
# One compact row above an existing field figure: three square |rho|
# projection maps, one marginal-profile axis and one shared colour bar.  The
# sizes are in millimetres, so they are physical at the print scale.  The
# marginal axis absorbs the width left over after the maps, the two label
# gaps, the bar and the outer margins, capped at the 38 mm design width.  The
# planned 170 mm canvas yields the full 38 mm marginal; the repo's actual
# A4 + 1 in-margin text width is 159.2 mm, where the row resolves to 32 mm
# maps and a ~23 mm marginal with every 7.5 pt tick label unclipped.
BUNCH_MAP_MM = 32.0
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
        "error_floor": ERROR_FLOOR,
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
    ``zero_floor`` is for squared-magnitude panels where zero is a physical log
    floor, not missing data: the bad/under colour becomes the colormap's own
    lowest colour so zero is not confused with a mask.
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
    if zero_floor:
        cmap.set_bad(cmap(0.0))
        cmap.set_under(cmap(0.0))
    else:
        cmap.set_bad(MASKED_COLOR)
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
    arr = np.asarray(arr)
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
    """Draw a blank (identically zero) panel with the canonical grey and note."""
    ax.set_facecolor(BLANK_COLOR)
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=ANNOT_PT,
            color=ANNOT_COLOR, transform=ax.transAxes)


def panel_note(ax, text, x=0.5, y=0.5, ha="center", va="center"):
    """Draw an in-panel annotation on the canonical annotation ladder."""
    return ax.text(x, y, text, ha=ha, va=va, fontsize=ANNOT_PT,
                   color=ANNOT_COLOR, transform=ax.transAxes)


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


def _truncated_cmap(name, top=ERR_CMAP_TOP, n=256):
    """Finite colormap truncated below its near-white top end."""
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


def error_limits(items):
    """Canonical error range: floor at ERROR_FLOOR, decade ceiling, gate shown.

    ``items`` may be arrays or dictionaries with ``err``/``valid`` entries.
    The range never drops below the acceptance gate, so the gate contour and
    its tick are always representable.
    """
    good = []
    for it in items:
        if isinstance(it, dict):
            e = np.asarray(it["err"], float)
            v = np.asarray(it["valid"], bool)
            e = e[v & np.isfinite(e) & (e > 0)]
        else:
            e = np.asarray(it, float)
            e = e[np.isfinite(e) & (e > 0)]
        if e.size:
            good.append(e)
    hi = 1.0
    if good:
        cat = np.concatenate(good)
        hi = 10.0 ** np.ceil(np.log10(float(cat.max())))
    vmin = ERROR_FLOOR
    vmax = max(float(hi), ERROR_GATE)
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
    labels = ["1e-3\ngate" if np.isclose(t, ERROR_GATE, rtol=1e-9, atol=0.0)
              else fmt_sci(t) for t in ticks]
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
                            width_frac,
                            support=None, suptitle=None, manifest=None,
                            compact=False, row_gap_in=FIELD_ROW_GAP_IN,
                            bottom_in=None, bunch=None, panel_scale=1.0,
                            top_in=FIELD_TOP_IN):
    """Field/error rows with a component triad and a separate squared panel.

    Every row has four items: three signed components sharing one colour bar,
    then the squared magnitude on its own scale with its own bar and unit
    label.  Geometry comes from :func:`field_layout`, so every field family
    gets the same square panels; only the figure height changes with the row
    count.  ``compact`` shortens the colour-bar labels and the squared-panel
    log ticks.  ``row_gap_in`` and ``bottom_in`` are inches.  ``x`` and ``y``
    are pixel **centres** shared by every field family; the raster policy is
    fixed by :func:`draw_raster_panel`.

    ``bunch`` is an optional pre-built bunch-form row (see
    :mod:`render_bunch`).  It is drawn as the FIRST row of this same figure:
    the field block keeps its physical size and position at the bottom, the
    canvas grows by the row's reserve, and ``panel_scale`` shrinks the square
    field panels when the height budget needs it.
    """
    apply_paper_style(kind="field")
    nrow = len(rows)
    has_error = any(r["kind"] == "error" for r in rows)
    if bottom_in is None:
        bottom_in = FIELD_BOTTOM_ERR_IN if has_error else FIELD_BOTTOM_IN
    geo = field_layout(nrow, width_frac=width_frac, top_in=top_in,
                       bottom_in=bottom_in, row_gap_in=row_gap_in,
                       panel_scale=panel_scale)
    bunch_geo = (bunch.geometry(geo["fig_w_in"])
                 if bunch is not None else None)
    fig_h_in = geo["fig_h_in"] + (bunch_geo["reserve_in"]
                                  if bunch_geo is not None else 0.0)
    fig = figure(width_frac, fig_h_in)
    panel_x = geo["panel_x"]
    pw = geo["pw"]
    rh = geo["panel_width_in"] / fig_h_in
    bw = geo["bar_width"]
    cb1, cb2 = geo["cb1"], geo["cb2"]
    if bunch_geo is None:
        row_y = geo["row_y"]
    else:
        # The field block is anchored to the bottom; the extra reserve is
        # added above it for the bunch row.
        row_y = [ry * geo["fig_h_in"] / fig_h_in for ry in geo["row_y"]]

    xlo, xhi = _cell_edges(x, "x")
    ylo, yhi = _cell_edges(y, "y")

    field_cmap = cmap_for("signed_field")
    # Zero is the physical log floor of the squared panel, not a mask: use the
    # colormap's own lowest colour so zero is never the mask grey.
    sq_cmap = cmap_for("squared_field", zero_floor=True)
    err_cmap = cmap_for("error")
    bar_rows = []
    peak_texts = []

    for ri, row in enumerate(rows):
        y0 = row_y[ri]
        items = row["items"]
        if len(items) != 4:
            raise ValueError(
                "render_split_field_rows needs three components "
                "plus the squared panel")
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
            cnorm = mcolors.SymLogNorm(
                linthresh=max(cvmax * SYMLOG_LINTHRESH_FRAC, 1e-30),
                vmin=-cvmax, vmax=cvmax, base=10)
            sfloor, svmax = sq_limits(np.asarray(sqitem["arr"], float))
            snorm = mcolors.LogNorm(vmin=sfloor, vmax=svmax)
            ccmap, scmap = field_cmap, sq_cmap
            cticks = [-cvmax, 0.0, cvmax]
            clabels = [fmt_sci(-cvmax), "0", fmt_sci(cvmax)]
            if compact:
                sticks = [snorm.vmin,
                          float(np.sqrt(snorm.vmin * svmax)), svmax]
                slabels = [fmt_sci(t) for t in sticks]
            else:
                sticks, slabels = sq_ticks(svmax)
            unit2 = "V$^2$/m$^2$" if part == "E" else "T$^2$"
            if compact:
                clabel = f"{part} ({UNITS[part]})"
                slabel = f"{part}$^2$ {unit2}"
            else:
                clabel = f"{part} ({UNITS[part]}) (symlog)"
                slabel = f"{part}$^2$ ({unit2}) (log)"
        else:
            cvmin, cvmax = error_limits(comps)
            svmin, svmax = error_limits([sqitem])
            cnorm = mcolors.LogNorm(vmin=cvmin, vmax=cvmax)
            snorm = mcolors.LogNorm(vmin=svmin, vmax=svmax)
            ccmap = scmap = err_cmap
            cticks, clabels = error_ticks(cvmin, cvmax)
            sticks, slabels = error_ticks(svmin, svmax)
            clabel = slabel = "rel. error" if compact else "relative error"
            assert_masked_distinct(cnorm, ccmap, f"{part} error components")
            assert_masked_distinct(snorm, scmap, f"{part}$^2$ error")

        def panel(ax, it, norm, cmap, title, quantity):
            arr = np.asarray(it["arr"], float)
            if kind == "field":
                if it.get("blank"):
                    # An identically-zero reference component: draw the panel
                    # blank and say which quantity vanishes.  The caption
                    # carries the physical reason (beam axis or stationary).
                    blank_panel(ax, it.get("blank_text", ""))
                else:
                    draw_raster_panel(ax, x, y, arr, quantity,
                                      norm=norm, cmap=cmap)
            else:
                e = np.asarray(it["err"], float)
                v = np.asarray(it["valid"], bool)
                z = np.ma.masked_where(~v | ~np.isfinite(e), e)
                draw_raster_panel(ax, x, y, z, "error",
                                  norm=norm, cmap=cmap)
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
                    panel_note(ax, note)
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
            ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
            ax.set_aspect("auto")
            ax.set_title(title, pad=PANEL_TITLE_PAD, fontsize=TITLE_PT)
            ax.tick_params(length=TICK_LENGTH, pad=TICK_PAD)
            # MaxNLocator(4, prune="both") keeps ticks strictly inside the
            # visible limits and away from the panel corners.  The old
            # MaxNLocator(3) emitted a 7.5 tick beyond a 6.4-wide x range whose
            # label spilled into the next panel, and edge ticks collided with
            # the first tick label of the neighbouring axis.
            ax.xaxis.set_major_locator(mticker.MaxNLocator(4, prune="both"))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(4, prune="both"))

        for j, it in enumerate(comps):
            ax = fig.add_axes([panel_x[j], y0, pw, rh])
            panel(ax, it, cnorm, ccmap, it.get("title", ""), "signed_field")
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
        panel(axq, sqitem, snorm, scmap, sq_title, "squared_field")
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

    if bunch is not None:
        y_bottom_in = (geo["fig_h_in"] + bunch_geo["inter_gap_in"]
                       + bunch_geo["label_in"])
        bunch.draw(fig, geo["fig_w_in"], fig_h_in, y_bottom_in)
    if suptitle:
        fig.text(0.5, 0.975, suptitle, ha="center", va="top",
                 fontsize=TITLE_PT)
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
    man["component_norm"] = (
        f"symlog, linthresh={SYMLOG_LINTHRESH_FRAC:g}*shared component maximum")
    man["component_peak_annotation"] = (
        "compact two-line peak label: 'peak' over the percentage of shared max "
        "(one significant figure below 1%, 'max' for the shared maximum)")
    man["peak_annotation_font_pt"] = PEAK_ANNOT_PT
    man["peak_annotation_max_panel_frac"] = (
        round(max(peak_fracs), 4) if peak_fracs else None)
    man["squared_norm"] = (
        f"log, {SQ_FLOOR_RATIO:g}*peak..peak (field rows)")
    man["masked_color"] = MASKED_COLOR
    man["error_floor"] = ERROR_FLOOR
    man["error_gate_tick"] = ERROR_GATE
    man.update(_layout_manifest(geo))
    man["figure_height_mm"] = round(fig_h_in * 25.4, 2)
    if bunch is not None:
        man["bunch_form"] = bunch.manifest(geo["fig_w_in"])
    return save(fig, outdir, name, man, tight=False)
