#!/usr/bin/env python3
"""Algorithm schematic ``SCHEM1_field_evaluation_schematic``.

Panel (a) is the real-space region sketch; panel (b) is the measured
per-component error from ``docs/paper/data/scaling/dsweep.csv``.  The layout
and the post-processing follow the legacy FIGS_REPLOT/rebuild_schem.py.
"""
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.text import Text

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cases  # noqa: E402
import figstyle as fs  # noqa: E402
import schematic as ms  # noqa: E402


def min_text_font_pt(fig):
    sizes = [float(t.get_fontsize()) for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip()]
    if not sizes:
        raise RuntimeError("no text artists found in the figure")
    return min(sizes)


def run(outdir):
    path = cases.data("scaling", "dsweep.csv")
    rows = ms.read_dsweep(path)
    meta = {"data_file": os.path.relpath(path, cases.REPO_ROOT)}
    fs.apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(fs.width_in(0.98), 7.35))
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.105, top=0.975,
                        hspace=0.42)
    ms.panel_real(axes[0], meta)
    ms.panel_err(axes[1], rows)

    for txt in list(axes[0].texts):
        if txt.get_text().strip() in {"TSC exact", "near shell", "mesh (FFT)"}:
            txt.remove()
    for txt in axes[0].texts:
        if "TSC" in txt.get_text() and "1.5" in txt.get_text():
            txt.remove()
        if "8.8" in txt.get_text() and "nodes" in txt.get_text():
            txt.set_text(r"$R_{\rm shell}(\gamma{=}10)=8.8$ nodes")
            txt.set_position((10.8, -6.0))
            txt.set_ha("left")
    if axes[1].get_legend() is not None:
        axes[1].get_legend().remove()
    for txt in list(axes[1].texts):
        if txt.get_text().strip() in {"TSC", "near shell", "mesh (FFT)"}:
            txt.remove()

    for ln in axes[1].lines:
        y = list(ln.get_ydata())
        if len(y) == 2 and abs(y[0] - 3e-4) < 1e-12 and abs(y[1] - 3e-4) < 1e-12:
            ln.set_color(fs.REF_COLOR); ln.set_linestyle("--"); ln.set_linewidth(0.9)
        elif len(y) == 2 and abs(y[0] - 1e-3) < 1e-12 and abs(y[1] - 1e-3) < 1e-12:
            ln.set_color(fs.GATE_STYLE["color"])
            ln.set_linestyle(fs.GATE_STYLE["linestyle"])
            ln.set_linewidth(fs.GATE_STYLE["linewidth"])

    region_handles = [
        mpatches.Patch(facecolor=ms.ALG_TSC[1], edgecolor="none",
                       label=ms.ALG_TSC[0] + "  (0 to 1.5 nodes)"),
        mpatches.Patch(facecolor=ms.ALG_SHELL[1], edgecolor="none",
                       label=ms.ALG_SHELL[0] + "  (1.5 to $R_{\\rm shell}$)"),
        mpatches.Patch(facecolor=ms.ALG_MESH[1], hatch=ms.ALG_MESH[2],
                       edgecolor="none",
                       label=ms.ALG_MESH[0] + "  (beyond $R_{\\rm shell}$)"),
    ]
    axes[0].legend(handles=region_handles, loc="lower center", ncol=1,
                   framealpha=0.92, handlelength=1.4, borderpad=0.3,
                   fontsize=fs.plt.rcParams["legend.fontsize"])
    line_handles = [
        Line2D([], [], color=fs.GAMMA_COLORS[1], marker="o", markersize=4,
               label=r"$\gamma$=1 (measured)"),
        Line2D([], [], color=fs.GAMMA_COLORS[10], marker="o", markersize=4,
               label=r"$\gamma$=10 (measured)"),
        Line2D([], [], color=fs.REF_COLOR, ls="--", lw=0.9,
               label=r"reference $3\times10^{-4}$"),
        Line2D([], [], color=fs.GATE_STYLE["color"],
               ls=fs.GATE_STYLE["linestyle"], lw=fs.GATE_STYLE["linewidth"],
               label=r"gate $10^{-3}$"),
    ]
    axes[1].legend(handles=line_handles, loc="lower left", ncol=2,
                   framealpha=0.92, handlelength=1.5, borderpad=0.3,
                   columnspacing=0.9,
                   fontsize=fs.plt.rcParams["legend.fontsize"])

    min_pt = min_text_font_pt(fig)
    if min_pt < 9.0 - 1e-6:
        raise RuntimeError(f"minimum font size {min_pt:.2f} pt is below 9 pt")
    fs.save(fig, outdir, "SCHEM1_field_evaluation_schematic",
            {"figure_id": "schem", "includegraphics_width": 0.98,
             "source_data": os.path.relpath(path, cases.REPO_ROOT),
             "min_text_font_pt": min_pt,
             "legend": "three region patches in panel a; series and gates in panel b"})
    return []


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    run(a.out_dir)
