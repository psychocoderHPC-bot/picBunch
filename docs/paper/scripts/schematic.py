#!/usr/bin/env python3
"""SCHEM1: schematic of *where* each algorithm evaluates the field (run
pr24-20261102, picBunch v0.5.0).

Two panels in one figure:
  (a) real space around one source: TSC support / near shell / mesh regions,
      the two example R_shell radii, and the six Yee stagger offsets.
  (b) measured per-component error vs distance for gamma 1 and 10, with the
      same three regions shaded so the reader sees the error jump where the
      algorithm switches.

Style matches the existing paper figures (PF/make_figures.py): Okabe-Ito
palette, >= 8 pt fonts, TrueType in the PDF, fixed metadata date for stable
byte output.

Every radius drawn and every point plotted comes from a named data file; the
mapping is listed in PROVENANCE.md. The only non-data numbers here are the
two example R_shell radii (12 and 5 cells, gamma 1 / gamma 10 at h=1, aspect 1,
from the S1 3e-4 crossover table), the 1.5-node TSC-support boundary, the
gate/threshold levels (3e-4, 1e-3), and axis limits.

Deterministic: Agg backend, no RNG, fixed metadata date.
  python3 make_schematic.py --data-dir data --out-dir figs
"""

import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Circle
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Shared, colour-blind-safe style (Okabe-Ito + blue), same as PF/make_figures.py
# ---------------------------------------------------------------------------
C_BLUE   = "#0072B2"   # gamma 1 data
C_RED    = "#D55E00"   # gamma 10 data
C_GREEN  = "#009E73"
C_ORANGE = "#E69F00"
C_PURPLE = "#CC79A7"
C_GREY   = "#565656"
C_GATE   = "#333333"

# Three algorithm regions -> colour-blind-safe fills (distinct hue + the mesh
# region carries a hatch so it never relies on colour alone).
ALG_TSC = ("TSC exact",      C_PURPLE, "white")
ALG_SHELL = ("Near shell",   C_GREEN,  "white")
ALG_MESH = ("Mesh (FFT)",    C_ORANGE, "//")

# Gate / threshold levels (spec-defined, not data).
GATE_3E4 = 3e-4
GATE_1E3 = 1e-3

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linewidth": 0.5,
    "pdf.fonttype": 42,      # TrueType in PDF (editable text)
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# Fixed metadata date so PDF output is byte-stable across runs.
_FIXED_DATE = "D:20261102000000+00'00'"

# The 3e-4 crossover radius R_shell = R_cross(gamma, h, aspect) for the two
# example gammas at h=1, aspect 1, from S1-algorithm-spec.md §1.4 (which
# documents next.md:4). Rounded to the integers the task specifies.
RSHELL = {
    1:  {"h": 1.0, "cells": 11.8, "spec": "measured 3e-4 crossover, h=1, aspect 1"},
    10: {"h": 1.0, "cells": 8.8,  "spec": "measured 3e-4 crossover, h=1, aspect 1"},
}
# The four h-dependent values are carried in the provenance note, not drawn.

# TSC support radius in mesh nodes (the point kernel is singular; the
# TSC-convolved field is finite inside it).
TSC_RADIUS = 1.5

# The six Yee stagger offsets in cell units (S1-algorithm-spec.md §1.5,
# VERIFIED in meshconv.cu kOffsets / convkernel.py OFFSETS / YeeCell.hpp):
#   Ex (0.5,0,0)   Ey (0,0.5,0)   Ez (0,0,0.5)
#   Bx (0,0.5,0.5) By (0.5,0,0.5) Bz (0.5,0.5,0)
# The left panel is the in-plane (x, y) cut (z -> 0); the four resulting
# in-plane positions and their component groups are computed in panel_real.


# ---------------------------------------------------------------------------
# CSV reader (same convention as PF/make_figures.py)
# ---------------------------------------------------------------------------
def _f(x):
    return float(x)


def read_dsweep(path):
    """dsweep.csv has two blocks (gamma 1 then gamma 10) sharing one header row.
    Return a list of dicts with gamma, d, worst_rel_E, worst_rel_B, worst_rel_all."""
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                g = int(_f(r["gamma"]))
            except (ValueError, KeyError, TypeError):
                continue  # repeated header / blank
            out.append({
                "gamma": g,
                "d": int(_f(r["d"])),
                "worst_rel_E": _f(r["worst_rel_E"]),
                "worst_rel_B": _f(r["worst_rel_B"]),
                "worst_rel_all": _f(r["worst_rel_all"]),
            })
    return out


def save(fig, outdir, name):
    pdf = os.path.join(outdir, name + ".pdf")
    png = os.path.join(outdir, name + ".png")
    fig.savefig(pdf, bbox_inches="tight",
                metadata={"CreationDate": _FIXED_DATE, "ModDate": _FIXED_DATE})
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    print("wrote", pdf)
    print("wrote", png)


# ---------------------------------------------------------------------------
# Panel (a): real space around one source
# ---------------------------------------------------------------------------
def panel_real(ax, meta):
    """2D cut through the mesh. Units are mesh nodes, equal aspect, source at
    the centre. Grid nodes are faint dots; the three regions are shaded and
    labelled; the two example R_shell radii are annotated; the six Yee stagger
    offsets are marked on the cell corners/edges/centre of one reference cell."""
    # Draw the far extent: the outer ring is the mesh region.
    rmax = 14.0
    # Outer mesh region (annulus beyond R_shell(gamma1)). Draw it first, faint.
    mesh_fill = Rectangle((-rmax, -rmax), 2 * rmax, 2 * rmax,
                          facecolor=ALG_MESH[1], edgecolor="none",
                          hatch=ALG_MESH[2], alpha=0.30, zorder=0)
    ax.add_patch(mesh_fill)

    # Near shell region: ring from TSC_RADIUS to R_shell(gamma1) for gamma 1
    # (the larger of the two examples, so the shell is visible around the TSC
    # core). For gamma 10 the shell is smaller (5) and drawn as a second ring.
    R1 = RSHELL[1]["cells"]     # 12 (gamma 1)
    R10 = RSHELL[10]["cells"]   # 5  (gamma 10)

    shell1 = Circle((0, 0), R1, facecolor=ALG_SHELL[1], edgecolor="none",
                    alpha=0.40, zorder=1)
    ax.add_patch(shell1)

    # TSC core (gamma-1 TSC support is the same 1.5 nodes for both gammas).
    tsc = Circle((0, 0), TSC_RADIUS, facecolor=ALG_TSC[1], edgecolor="none",
                 alpha=0.90, zorder=3)
    ax.add_patch(tsc)

    # Grid nodes as faint dots (every integer node in the square window).
    xs = list(range(-13, 14))
    for gx in xs:
        for gy in xs:
            ax.plot([gx], [gy], marker=".", color=C_GREY, markersize=3.0,
                    alpha=0.40, zorder=4,
                    clip_on=False, linestyle="none")

    # The six Yee stagger offsets (S1-algorithm-spec.md §1.5, cell units):
    #   Ex (0.5,0,0)   Ey (0,0.5,0)   Ez (0,0,0.5)
    #   Bx (0,0.5,0.5) By (0.5,0,0.5) Bz (0.5,0.5,0)
    # This panel is the in-plane (x, y) cut (z -> 0), so a component sits at the
    # in-plane position of its (x, y) offset; the z half-cell is out of plane.
    # That gives four distinct staggered positions, and the source (0,0,0) is
    # not a Yee point.
    cell = Rectangle((-0.5, -0.5), 1.0, 1.0, facecolor="none",
                     edgecolor=C_GREY, linewidth=0.7, linestyle=":", zorder=5)
    ax.add_patch(cell)
    stagger_groups = {
        (0.5, 0.0): ["Ex", "By"],
        (0.0, 0.5): ["Ey", "Bx"],
        (0.5, 0.5): ["Bz"],
        (0.0, 0.0): ["Ez (z, out of plane)"],
    }
    # Mark each staggered position with a cross; labels are fanned out on
    # separate leader lines (the 1-node cell is only ~35 px wide at this zoom,
    # so in-cell labels would overlap). Each label sits in its own clear band.
    label_pos = {
        (0.5, 0.0): ((5.8, 5.8), "Ex   By"),
        (0.0, 0.5): ((-5.8, 5.8), "Ey   Bx"),
        (0.5, 0.5): ((5.8, 1.6), "Bz"),
    }
    for (sx, sy), comps in stagger_groups.items():
        if sx == 0.0 and sy == 0.0:
            continue  # the source occupies the cell centre
        ax.plot([sx], [sy], marker="x", color=C_GATE, markersize=4.5,
                markeredgewidth=1.0, zorder=6)
        (lx, ly), lab = label_pos[(sx, sy)]
        ax.annotate(lab, (sx, sy), xytext=(lx, ly), fontsize=9, color=C_GATE,
                    zorder=7,
                    arrowprops=dict(arrowstyle="-", color=C_GREY, lw=0.5,
                                    shrinkB=2))

    # Source marker at the centre; label below, in an empty band of the shell.
    ax.plot([0], [0], marker="*", color="black", markersize=11, zorder=8)
    ax.text(0.0, -2.4, "source  (at (0,0,0), not on a Yee point)", ha="center",
            va="top", fontsize=9, color="black", zorder=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                      alpha=0.85))

    # Region boundary circles (no fill, for the radii).
    ax.add_patch(Circle((0, 0), TSC_RADIUS, fill=False, edgecolor=ALG_TSC[1],
                        linewidth=1.0, zorder=7))
    ax.add_patch(Circle((0, 0), R10, fill=False, edgecolor=ALG_SHELL[1],
                        linewidth=1.0, linestyle="--", zorder=7))
    ax.add_patch(Circle((0, 0), R1, fill=False, edgecolor=C_GATE,
                        linewidth=1.0, linestyle="-", zorder=7))

    # Radius annotations (label on a leader line to the ring, kept inside the
    # -14..14 axis so the text does not clip). The R1 label uses the top band;
    # the R10 and TSC labels use the bottom band (the top is already taken by
    # the Yee labels).
    ax.annotate(r"$R_{\rm shell}(\gamma{=}1)=11.8$ nodes (tolerance $3\times10^{-4}$)",
                xy=(R1 * 0.7071, R1 * 0.7071),
                xytext=(-7.6, 11.2), ha="center", fontsize=9, color=C_GATE,
                zorder=8,
                arrowprops=dict(arrowstyle="-", color=C_GATE, lw=0.7,
                                shrinkB=2))
    ax.annotate(r"$R_{\rm shell}(\gamma{=}10)=8.8$ nodes (tolerance $3\times10^{-4}$)",
                xy=(R10 * 0.7071, -R10 * 0.7071),
                xytext=(6.2, -8.2), ha="center", fontsize=9, color=C_GATE,
                zorder=8,
                arrowprops=dict(arrowstyle="-", color=C_GATE, lw=0.7,
                                shrinkB=2))
    ax.annotate("TSC  (0 to 1.5 nodes)",
                xy=(-TSC_RADIUS, 0.0),
                xytext=(-8.0, -7.2), ha="center", fontsize=9,
                color=ALG_TSC[1], zorder=8,
                arrowprops=dict(arrowstyle="-", color=ALG_TSC[1], lw=0.7,
                                shrinkB=2))

    # Region labels placed in empty bands (white-backed, so they read over the
    # hatched mesh region).
    ax.text(9.2, -12.4, "mesh (FFT)", ha="center", va="center", fontsize=9,
            color="black", zorder=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                      alpha=0.9))
    ax.text(-11.5, -7.5, "near shell", ha="center", va="center", fontsize=9,
            color="black", zorder=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                      alpha=0.9))
    ax.text(0.0, 7.6, "TSC exact", ha="center", va="center", fontsize=9,
            color="white", zorder=8)

    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-rmax, rmax)
    ax.set_aspect("equal")
    ax.set_xlabel("distance from source (mesh nodes)")
    ax.set_ylabel("distance from source (mesh nodes)")
    ax.set_title("(a) Real space around one source (2D cut, h=1, aspect 1)")
    meta["panel_a"] = {
        "tsc_radius_nodes": TSC_RADIUS,
        "rshell_gamma1_cells": R1,
        "rshell_gamma10_cells": R10,
        "yee_inplane_groups": {f"({a},{b})": c for (a, b), c in stagger_groups.items()},
    }


# ---------------------------------------------------------------------------
# Panel (b): measured per-component error vs distance
# ---------------------------------------------------------------------------
def panel_err(ax, rows):
    """Plot worst_rel_all vs d for gamma 1 and 10 from dsweep.csv. Shade the
    three regions using the gamma-1 radii (the larger, so the bands are wide
    enough to read); the gamma-10 shell is narrower and noted in the legend.
    Draw the 3e-4 and 1e-3 gate lines."""
    g1 = sorted([r for r in rows if r["gamma"] == 1], key=lambda r: r["d"])
    g10 = sorted([r for r in rows if r["gamma"] == 10], key=lambda r: r["d"])

    R1 = RSHELL[1]["cells"]
    R10 = RSHELL[10]["cells"]

    # Region shading (vertical bands), drawn first so the data sits on top.
    # d=2 is the first measured point, so the TSC band is narrow; draw it from
    # 0 (the true support edge is at 1.5).
    ax.axvspan(0, TSC_RADIUS, color=ALG_TSC[1], alpha=0.18, zorder=0)
    ax.axvspan(TSC_RADIUS, R1, color=ALG_SHELL[1], alpha=0.18, zorder=0)
    ax.axvspan(R1, 48.5, facecolor=ALG_MESH[1], alpha=0.14, zorder=0,
               hatch="//", edgecolor="none")

    # Region boundary lines.
    for r, c, ls in ((TSC_RADIUS, ALG_TSC[1], "-"), (R10, ALG_SHELL[1], "--"),
                     (R1, C_GATE, "-")):
        ax.axvline(r, color=c, linestyle=ls, linewidth=0.8, zorder=1)

    for grp, col, lbl in ((g1, C_BLUE, r"$\gamma$=1"),
                          (g10, C_RED, r"$\gamma$=10")):
        d = [r["d"] for r in grp]
        e = [r["worst_rel_all"] for r in grp]
        ax.plot(d, e, color=col, marker="o", markersize=3.5, label=lbl,
                zorder=3)

    # Gate lines.
    ax.axhline(GATE_3E4, color=C_GATE, linestyle="--", linewidth=0.9, zorder=2,
               label="gate 3e-4")
    ax.axhline(GATE_1E3, color=C_GATE, linestyle=":", linewidth=0.9, zorder=2,
               label="gate 1e-3")

    # Region labels: small horizontal boxes in the empty lower-left of the plot
    # (both curves fall below 1e-3 well before d=12). One per band, spaced so
    # they do not collide on the log axis. The TSC band is only 1.5 nodes wide,
    # so its label sits in the near-shell band just right of it (a short arrow
    # would re-enter the clipped TSC band); the coloured band + the legend make
    # the assignment unambiguous.
    ax.text(2.6, 1.6e-5, "TSC", ha="left", va="center", fontsize=9,
            color=ALG_TSC[1], zorder=4,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                      alpha=0.85))
    ax.text(8.0, 1.6e-5, "near shell", ha="center", va="center", fontsize=9,
            color="black", zorder=4,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                      alpha=0.85))
    ax.text(17.5, 1.6e-5, "mesh (FFT)", ha="center", va="center", fontsize=9,
            color="black", zorder=4,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                      alpha=0.85))

    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 3.0)
    ax.set_xlim(1, 48.5)
    ax.set_xlabel("source-target distance d (mesh nodes)")
    ax.set_ylabel("worst per-component relative error")
    ax.set_title("(b) Measured per-component error vs distance (h=1, aspect 1)")
    ax.legend(loc="upper right", framealpha=0.9)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--data-dir", default=os.path.join(here, "data"))
    ap.add_argument("--out-dir", default=os.path.join(here, "figs"))
    args = ap.parse_args(argv)

    data_dir = args.data_dir
    outdir = args.out_dir
    os.makedirs(outdir, exist_ok=True)

    dsweep_path = os.path.join(data_dir, "dsweep.csv")
    if not os.path.exists(dsweep_path):
        print("ERROR: missing %s" % dsweep_path, file=sys.stderr)
        return 1

    rows = read_dsweep(dsweep_path)
    meta = {"data_file": os.path.abspath(dsweep_path)}

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8))
    panel_real(axes[0], meta)
    panel_err(axes[1], rows)

    # Shared legend: name the three algorithms.
    handles = [
        mpatches.Patch(facecolor=ALG_TSC[1], edgecolor="none", label=ALG_TSC[0] +
                       "  (0 to 1.5 nodes)"),
        mpatches.Patch(facecolor=ALG_SHELL[1], edgecolor="none", label=ALG_SHELL[0] +
                       "  (1.5 to R_shell)"),
        mpatches.Patch(facecolor=ALG_MESH[1], hatch=ALG_MESH[2], edgecolor="none",
                       label=ALG_MESH[0] + "  (beyond R_shell)"),
        plt.Line2D([], [], color=C_BLUE, marker="o", markersize=4,
                   label=r"$\gamma$=1 (measured)"),
        plt.Line2D([], [], color=C_RED, marker="o", markersize=4,
                   label=r"$\gamma$=10 (measured)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=True,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save(fig, outdir, "SCHEM1_field_evaluation_schematic")

    # Dumper for provenance: the radii used and the data columns read.
    meta["panel_b"] = {
        "columns": ["d", "worst_rel_all"],
        "gates": [GATE_3E4, GATE_1E3],
        "rshell_gamma1_cells": RSHELL[1]["cells"],
        "rshell_gamma10_cells": RSHELL[10]["cells"],
    }
    meta["rshell_spec_source"] = "S1-algorithm-spec.md §1.4 (3e-4 crossover table; next.md:4)"
    meta["rshell_values"] = RSHELL
    print("\n=== SCHEM1 metadata ===")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
