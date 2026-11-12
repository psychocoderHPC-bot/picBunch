#!/usr/bin/env python3
"""Scaling and distribution figures: ``F_a``, ``F_b``, ``F_w4b_rel_error``,
``F_gamma_err_dist``, ``F_multivel_dist`` and ``F_mpi_rank_scaling_h100``.

All numbers come from files under ``docs/paper/data/scaling`` and
``docs/paper/data/line``.  The legacy rebuild_line.py kept the MPI and gamma
values as Python literals; they now live in CSV files so a re-measurement is a
data change, not a code change.  No value was altered.

The V100 ``F_mpi_rank_scaling`` renderer (:func:`mpi`) is kept working but is
not in the default run: the manuscript no longer includes it.  The H100
figure (:func:`mpi_h100`) reads the committed reduced table
``data/line/h100_mpi_scaling.csv``; if it is absent the function raises
:class:`PendingData`.
"""
import csv
import os
import sys

import numpy as np
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cases  # noqa: E402
import figstyle as fs  # noqa: E402
import render_bunch  # noqa: E402

STAGES = ["deposit", "kernel", "fft", "readout"]
STAGE_STYLE = {
    "deposit": dict(color=fs.OKABE["blue"], marker="o", linestyle="none"),
    "kernel": dict(color=fs.OKABE["orange"], marker="s", linestyle="none"),
    "fft": dict(color=fs.OKABE["green"], marker="^", linestyle="none"),
    "readout": dict(color=fs.OKABE["purple"], marker="D", linestyle="none"),
}


def _f(x):
    return float(x)


def read_matrix(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def select(rows, **eq):
    def ok(r):
        for k, v in eq.items():
            rv = r.get(k)
            if rv is None:
                return False
            try:
                if abs(_f(rv) - _f(v)) > 1e-12:
                    return False
            except (ValueError, TypeError):
                if str(rv) != str(v):
                    return False
        return True
    return [r for r in rows if ok(r)]


def _save(fig, outdir, name, man, tight=True):
    man.update({"style_module": "docs/paper/scripts/figstyle.py"})
    fs.save(fig, outdir, name, man, tight=tight)


# ---------------------------------------------------------------- F_a / F_b
def _row_points(r):
    """Plot positions and per-point IQR half-widths for one matrix row.

    Quotable rows plot the total median over the kept repeats with half the
    IQR of bar; bimodal rows plot both mode totals so the split is visible
    instead of hidden by one median (bar half-width is half the IQR of the
    per-mode stage cluster medians).  Rows missing the classification columns
    fall back to the raw per-row min/max bars, so the renderer still works on
    the pre-2026-11-09 table.
    """
    if r.get("row_class") == "bimodal":
        low = _f(r["mode_total_low_ms"])
        high = _f(r["mode_total_high_ms"])
        half = 0.5 * _f(r["mode_total_iqr_ms"])
        return [(low, half), (high, half)]
    med = _f(r["total_median_ms"])
    if r.get("row_class") == "quotable" and "total_iqr_ms" in r:
        return [(med, 0.5 * _f(r["total_iqr_ms"]))]
    return [(med, max(med - _f(r["total_min_ms"]), 0.0)),
            (med, _f(r["total_max_ms"]) - med)]


def _plot_totals(ax, xs, rows, rec, pooled=None):
    """Draw the total series (blue) and its bars from ``rec``/``pooled``."""
    total_pts = []
    for x in xs:
        r = rec[x]
        if "row_class" not in r:
            # Legacy rows: the pooled independent repeats bound the total.
            group = pooled[x]
            tot = [_f(g["total_median_ms"]) for g in group]
            lo = min(_f(g["total_min_ms"]) for g in group)
            hi = max(_f(g["total_max_ms"]) for g in group)
            total_pts += [(x, m, m - lo, hi - m) for m in tot]
        else:
            total_pts += [(x, v, h, h) for v, h in _row_points(r)]
    for x, m, lo, hi in total_pts:
        if lo > 0 or hi > 0:
            ax.errorbar([x], [m], yerr=[[lo], [hi]], fmt="none",
                        ecolor=fs.OKABE["grey"], elinewidth=0.8, capsize=2.5,
                        zorder=2)
    ax.plot([p[0] for p in total_pts], [p[1] for p in total_pts],
            color=fs.OKABE["blue"], marker="o", markersize=4.2,
            label="total", zorder=3)


def _plot_stages(ax, xs, rec):
    for s in STAGES:
        vals = [_f(rec[x][s + "_median_ms"]) for x in xs]
        st = STAGE_STYLE[s]
        ax.plot(list(xs), vals, color=st["color"], marker=st["marker"],
                markersize=3.6, linewidth=0, label=s)


def fig_a(outdir):
    path = cases.data("scaling", "matrix.csv")
    rows = read_matrix(path)
    sub = select(rows, box_cells=20, gamma=10, spacing=1, padding=2)
    ns = sorted({int(_f(r["particles"])) for r in sub})
    rec = {int(_f(r["particles"])): r for r in sub}
    pooled = {}
    for r in sub:
        pooled.setdefault(int(_f(r["particles"])), []).append(r)
    fs.apply_paper_style(kind="line")
    fig, ax = fs.subplots(0.62, 3.45)
    fig.subplots_adjust(left=0.185, right=0.975, bottom=0.165, top=0.965)
    _plot_totals(ax, ns, sub, rec, pooled)
    _plot_stages(ax, ns, rec)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("particle count $N$ (log)")
    ax.set_ylabel("time per stage (ms, log)")
    shown = [n for n in (1000, 100000, 2000000) if n in ns]
    ax.set_xticks(shown)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v):.0e}"))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.legend(loc="lower left", ncol=2, framealpha=0.92, handlelength=1.4,
              columnspacing=0.9, borderpad=0.3)
    ax.grid(alpha=0.3, linewidth=0.5)
    _save(fig, outdir, "F_a_scaling_vs_N",
          {"figure_id": "Fa", "source_data": os.path.relpath(path, cases.REPO_ROOT),
           "axis_limits": "autoscale; log-log"})


def fig_b(outdir):
    path = cases.data("scaling", "matrix.csv")
    rows = read_matrix(path)
    sub = select(rows, particles=2000000, gamma=10, spacing=1, padding=2)
    boxes = sorted({int(_f(r["box_cells"])) for r in sub})
    rec = {}
    for r in sub:
        rec.setdefault(int(_f(r["box_cells"])), r)
    fs.apply_paper_style(kind="line")
    fig, ax = fs.subplots(0.62, 3.45)
    fig.subplots_adjust(left=0.185, right=0.975, bottom=0.165, top=0.965)
    _plot_totals(ax, boxes, sub, rec)
    _plot_stages(ax, boxes, rec)
    for b in boxes:
        r = rec[b]
        pts = _row_points(r)
        top = max(v for v, _ in pts)
        ax.annotate(f"M={int(_f(r['Mx']))}", (b, top),
                    xytext=(0, 9), textcoords="offset points", ha="center",
                    color=fs.OKABE["grey"])
    ax.set_yscale("log")
    ax.set_xlabel("source box size (cells)")
    ax.set_ylabel("time per stage (ms, log)")
    ax.set_xticks(boxes)
    ax.legend(loc="lower right", ncol=2, framealpha=0.92, handlelength=1.4,
              columnspacing=0.9, borderpad=0.3)
    ax.grid(alpha=0.3, linewidth=0.5)
    _save(fig, outdir, "F_b_scaling_vs_box",
          {"figure_id": "Fb", "source_data": os.path.relpath(path, cases.REPO_ROOT),
           "axis_limits": "autoscale; log y"})


# ------------------------------------------------------------------ MPI
def mpi(outdir):
    """Legacy V100 rank scaling.  Not in the default set (unreferenced)."""
    path = cases.data("line", "mpi_scaling.csv")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    by = {}
    for r in rows:
        by.setdefault(r["case"], []).append(r)
    style = {
        "near": dict(color=fs.OKABE["blue"], marker="o", label="near shell"),
        "mesh": dict(color=fs.OKABE["orange"], marker="s", label="mesh"),
        "write": dict(color=fs.OKABE["green"], marker="D", label="write"),
    }
    fs.apply_paper_style(kind="line")
    fig, (axa, axb) = fs.subplots(0.58, 5.35, nrows=2, ncols=1,
                                  gridspec_kw={"hspace": 0.32})
    fig.subplots_adjust(left=0.215, right=0.975, bottom=0.095, top=0.965)
    for key, c in style.items():
        rr = sorted(by[key], key=lambda r: int(r["rank"]))
        ranks = [int(r["rank"]) for r in rr]
        w = np.array([_f(r["wall_median_s"]) for r in rr])
        runs = np.array([[_f(r["run1"]), _f(r["run2"]), _f(r["run3"])]
                         for r in rr])
        lo, hi = runs.min(axis=1), runs.max(axis=1)
        axa.errorbar(ranks, w, yerr=np.vstack([w - lo, hi - w]), fmt="none",
                     ecolor=c["color"], elinewidth=0.8, capsize=2.2, zorder=3)
        axa.plot(ranks, w, color=c["color"], marker=c["marker"], ms=4.2,
                 lw=1.0, label=c["label"], zorder=4)
        s = np.array([_f(r["speedup"]) for r in rr])
        axb.plot(ranks, s, color=c["color"], marker=c["marker"], ms=4.2, lw=1.0)
        off = {"near": 8, "mesh": -13, "write": 8}[key]
        for r in rr[1:]:
            axb.annotate(f"{int(_f(r['eff_pct']))}%",
                         (int(r["rank"]), _f(r["speedup"])),
                         textcoords="offset points", xytext=(0, off),
                         ha="center", va="bottom" if off > 0 else "top",
                         color=c["color"], fontweight="bold")
    ranks = [1, 2, 4]
    axa.plot(ranks, [36.61 / n for n in ranks], "k--", lw=0.8, alpha=0.7)
    axa.set_xscale("log", base=2); axa.set_yscale("log")
    axa.set_xlim(0.9, 4.9); axa.set_ylim(0.3, 100)
    axa.set_yticks([0.5, 1, 2, 5, 10, 20, 50])
    axa.set_ylabel("wall time (s, log)")
    axa.legend(loc="lower right", framealpha=0.92, handlelength=1.5,
               borderpad=0.3)
    axa.grid(alpha=0.3, linewidth=0.5)
    fs.panel_title(axa, "a", "wall time vs ranks")
    axb.plot(ranks, ranks, "k--", lw=0.8, alpha=0.7, label="ideal $S=n$")
    axb.set_xscale("log", base=2)
    axb.set_xlim(0.9, 4.9); axb.set_ylim(0, 5)
    axb.set_yticks([0, 1, 2, 3, 4, 5])
    axb.set_xlabel("ranks (log$_2$)")
    axb.set_ylabel("speedup $t_1/t_n$")
    axb.legend(loc="lower right", framealpha=0.92, handlelength=1.5,
               borderpad=0.3)
    axb.grid(alpha=0.3, linewidth=0.5)
    fs.panel_title(axb, "b", "speedup vs ranks")
    for ax in (axa, axb):
        ax.set_xticks([1, 2, 4]); ax.set_xticklabels(["1", "2", "4"])
    _save(fig, outdir, "F_mpi_rank_scaling",
          {"figure_id": "mpi", "source_data": os.path.relpath(path, cases.REPO_ROOT),
           "axis_limits": "x 0.9..4.9; y 0.3..100 (log) and 0..5"})


# ------------------------------------------------------------------ W4b
def w4b(outdir):
    path = cases.data("line", "w4b_rel_error.csv")
    data = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["kind"] != "single":
                continue
            g, a = int(r["gamma"]), float(r["aspect"])
            data.setdefault((g, a), []).append(
                (int(r["d"]), float(r["worst_rel_error"])))
    fs.apply_paper_style(kind="line")
    fig, ax = fs.subplots(0.95, 4.15)
    fig.subplots_adjust(left=0.145, right=0.975, bottom=0.155, top=0.94)
    for (g, a), pts in sorted(data.items()):
        pts.sort()
        col = fs.GAMMA_COLORS.get(g, fs.OKABE["black"])
        ls = "-" if a == 1.0 else (0, (2, 1))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col,
                linestyle=ls, marker="o", markersize=3.8,
                label=f"$\\gamma={g}$, aspect {int(a)}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1.8, 40); ax.set_ylim(1e-7, 3e0)
    ax.set_xticks([2, 4, 8, 12, 16, 24, 32])
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("source-target distance $d$ (cells, log)")
    ax.set_ylabel("worst per-component relative error (log)")
    ax.legend(loc="lower left", framealpha=0.92, handlelength=1.6,
              borderpad=0.3)
    ax.grid(alpha=0.3, linewidth=0.5)
    _save(fig, outdir, "F_w4b_rel_error",
          {"figure_id": "Fw4b", "source_data": os.path.relpath(path, cases.REPO_ROOT),
           "axis_limits": "x 1.8..40, y 1e-7..3e0 (log-log)"})


# ------------------------------------------------------------------ gamma
def gamma(outdir):
    path = cases.data("line", "gamma_err_dist.csv")
    D = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            D[(int(r["gamma"]), float(r["h"]), int(r["d"]))] = \
                float(r["peak_normalised_abs_error"])
    C = {str(g): fs.GAMMA_COLORS[g] for g in (10, 50, 100)}
    fs.apply_paper_style(kind="line")
    fig, ax = fs.subplots(0.62, 3.35)
    fig.subplots_adjust(left=0.20, right=0.975, bottom=0.16, top=0.965)
    ds = [2, 8, 48]
    for g in (10, 50, 100):
        for h, ls in ((1.0, "-"), (0.25, "--")):
            ax.plot(ds, [D[(g, h, d)] for d in ds], color=C[str(g)], ls=ls,
                    marker="o", ms=3.5)
    handles = [Line2D([], [], color=C[str(g)], label=rf"$\gamma={g}$")
               for g in (10, 50, 100)]
    handles += [Line2D([], [], color=fs.REF_COLOR, ls="-", label=r"$h=1$"),
                Line2D([], [], color=fs.REF_COLOR, ls="--", label=r"$h=1/4$")]
    ax.legend(handles=handles, loc="upper right", framealpha=0.92,
              handlelength=1.4, borderpad=0.3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1.5, 60); ax.set_ylim(1e-4, 1e3)
    ax.set_xticks([2, 8, 48]); ax.set_xticklabels(["2", "8", "48"])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("source-target distance $d$ (cells, log)")
    ax.set_ylabel("peak-normalised abs. error (log)")
    ax.grid(alpha=0.3, linewidth=0.5)
    _save(fig, outdir, "F_gamma_err_dist",
          {"figure_id": "gamma", "source_data": os.path.relpath(path, cases.REPO_ROOT),
           "axis_limits": "x 1.5..60, y 1e-4..1e3 (log-log)"})


# ------------------------------------------------------------------ multivel
def _multivel_bunch():
    """512 sampled multi-velocity particles: positions (m) and charge (C)."""
    epath = cases.data("line", "multivel_energy.npz")
    fpath = cases.data("line", "multivel_fields_e5_b16.npz")
    p = np.load(epath)["particles"]
    cell = float(np.load(fpath)["cell_m"])
    return render_bunch.Bunch(
        p[:, :3], p[:, 6] * p[:, 7], "y", cell, 10.0,
        os.path.relpath(epath, cases.REPO_ROOT),
        "derived from multivel_energy.npz particles[:, :3] (m); charge "
        "q = particles[:,6]*particles[:,7] = -e each; deposit cell = "
        "cell_m read from multivel_fields_e5_b16.npz; the npz carries no "
        "bunch_* keys", pending_fresh_data=True)


def multivel(outdir):
    epath = cases.data("line", "multivel_energy.npz")
    apath = cases.data("line", "multivel_angular.npz")
    bpath = cases.data("line", "multivel_binning.csv")
    E = np.load(epath)["particles"]
    A = np.load(apath)["particles"]
    Ez, Euz = E[:, 2], E[:, 5]
    Eg = np.sqrt(1.0 + E[:, 3] ** 2 + E[:, 4] ** 2 + E[:, 5] ** 2)
    Aux, Auy, Auz = A[:, 3], A[:, 4], A[:, 5]
    Au = np.sqrt(Aux ** 2 + Auy ** 2)
    Apar, Aperp = Auz, Au
    Atheta = np.arctan2(Aperp, Apar)
    gmean, gstd = Eg.mean(), Eg.std(ddof=1)
    theta_med = np.median(np.abs(Atheta)) * 1e3
    sig_rad = np.deg2rad(5.0)
    B = 16
    uedges = np.linspace(Euz.min(), Euz.max(), B + 1)
    gedges = np.sqrt(1.0 + uedges ** 2)
    with open(bpath, newline="") as fh:
        bins = list(csv.DictReader(fh))
    Bv = np.array([int(r["bin_count"]) for r in bins])
    m_e = np.array([float(r["m_energy"]) for r in bins])
    y_e = np.array([float(r["y_energy"]) for r in bins])
    m_a = np.array([float(r["m_angular"]) for r in bins])
    y_a = np.array([float(r["y_angular"]) for r in bins])
    c_e = float(np.sum(m_e * y_e) / np.sum(m_e ** 2))
    c_a = float(np.sum(m_a * y_a) / np.sum(m_a ** 2))
    fs.apply_paper_style(kind="line")
    fig_w_in = fs.width_in(0.98)
    base_h = 4.95
    bunch = _multivel_bunch()
    row = render_bunch.build_row(bunch, "multivel", "F_multivel_dist", "xy",
                                 source=epath)
    row_geo = row.geometry(fig_w_in)
    fig_h = base_h + row_geo["reserve_in"]
    fig = fs.figure(0.98, fig_h)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], hspace=0.62,
                          wspace=0.30, left=0.095, right=0.985,
                          top=(base_h * 0.965) / fig_h,
                          bottom=(base_h * 0.115) / fig_h)
    axa = fig.add_subplot(gs[0, 0])
    axa.hist(Eg, bins=45, color=fs.GAMMA_COLORS[10], alpha=0.35,
             edgecolor=fs.GAMMA_COLORS[10], linewidth=0.5)
    for ge in gedges:
        axa.axvline(ge, color=fs.REF_COLOR, lw=0.55, ls=":", alpha=0.8)
    axa.axvline(gmean, color=fs.REF_COLOR, lw=1.0, ls="--",
                label=r"mean $\gamma$")
    axa.axvline(gedges[0], color=fs.REF_COLOR, lw=0.55, ls=":", alpha=0.8,
                label=f"adaptive edges, $B={B}$")
    axa.set_xlabel(r"$\gamma$ (dimensionless)")
    axa.set_ylabel("counts per bin")
    fs.panel_title(axa, "a", r"energy spread 5%, seed 100")
    axa.text(0.03, 0.74,
             rf"$\gamma={gmean:.3f}\pm{gstd:.3f}$" + "\n" + rf"$N={len(Eg)}$",
             transform=axa.transAxes, va="top", ha="left")
    axa.legend(loc="upper right", framealpha=0.92, handlelength=1.3,
               borderpad=0.3)
    axb = fig.add_subplot(gs[0, 1])
    hb = fs.hist2d(axb, Apar, Aperp, bins=[34, 34], cmin=1,
                   norm=LogNorm(vmin=1, vmax=None))
    fs.draw_colorbar(fig, hb[3], ax=axb,
                     label="particles per bin (dimensionless)")
    xline = np.linspace(Apar.min(), Apar.max(), 50)
    axb.plot(xline, np.tan(sig_rad) * xline, color=fs.REF_COLOR, lw=1.1,
             ls="--")
    axb.plot(xline, np.tan(np.deg2rad(theta_med / 1e3)) * xline,
             color=fs.REF_COLOR, lw=1.1, ls=":")
    axb.set_xlabel(r"$u_\parallel$ ($m_ec$)")
    axb.set_ylabel(r"$u_\perp$ ($m_ec$)")
    fs.panel_title(axb, "b", r"angular spread 5$^\circ$, seed 103")
    axb.text(0.03, 0.04,
             rf"$\sigma_\theta=5^\circ$ (dashed)" + "\n" +
             rf"median $|\theta|={theta_med:.1f}$ mrad (dotted)",
             transform=axb.transAxes, va="bottom", ha="left",
             color=fs.TEXT_COLOR,
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                       alpha=0.85))
    axc = fig.add_subplot(gs[1, :])
    axc.axhline(1e-3, **fs.GATE_STYLE)
    axc.text(1.15, 1.25e-3, r"$10^{-3}$", color=fs.REF_COLOR, va="bottom")
    axc.plot(Bv, y_e, "o-", color=fs.GAMMA_COLORS[10], ms=4.0, lw=1.0,
             label=r"energy 5%: measured $T_b-T$")
    axc.plot(Bv, c_e * m_e, "--", color=fs.GAMMA_COLORS[10], lw=1.0,
             label=rf"energy 5%: model $\times {c_e:.3f}$")
    axc.plot(Bv, y_a, "s-", color=fs.GAMMA_COLORS[10], ms=4.0, lw=1.0,
             label=r"angular 5$^\circ$: measured $T_b-T$")
    axc.plot(Bv, c_a * m_a, ":", color=fs.GAMMA_COLORS[10], lw=1.0,
             label=rf"angular 5$^\circ$: model $\times {c_a:.3f}$")
    axc.set_xscale("log"); axc.set_yscale("log")
    axc.set_xlim(0.8, 90); axc.set_ylim(1.5e-4, 3.0)
    axc.set_xticks(Bv); axc.set_xticklabels([str(b) for b in Bv])
    axc.xaxis.set_minor_locator(mticker.NullLocator())
    axc.set_xlabel("bin count $B$ (log)")
    axc.set_ylabel(r"$T_b-T$, peak-normalised, $8\leq d\leq16$ cells (log)")
    fs.panel_title(axc, "c", "measured binning error and model, both cases")
    axc.legend(loc="upper right", framealpha=0.95, handlelength=1.6,
               borderpad=0.3)
    row.draw(fig, fig_w_in, fig_h,
             base_h + row_geo["inter_gap_in"] + row_geo["label_in"])
    man = {"figure_id": "multivel",
           "source_data": [os.path.relpath(p, cases.REPO_ROOT)
                           for p in (epath, apath, bpath)],
           "axis_limits": "panel c x 0.8..90, y 1.5e-4..3 (log-log)",
           "reference_lines": "dark grey, dashed sigma / dotted median",
           "figure_height_mm": round(fig_h * 25.4, 2),
           "bunch_form": row.manifest(fig_w_in)}
    _save(fig, outdir, "F_multivel_dist", man, tight=False)


# -------------------------------------------------------------- H100 MPI
H100_FIGURE = "F_mpi_rank_scaling_h100"
H100_CASE = "h100-mpi-scaling"
H100_TABLE = ("line", "h100_mpi_scaling.csv")


class PendingData(Exception):
    """A renderer exists but the campaign has not produced its reduced table."""

    def __init__(self, figure, case_id, path):
        self.figure = figure
        self.case_id = case_id
        self.path = path
        super().__init__(
            "%s needs the reduced table %s (campaign case %s)"
            % (figure, os.path.relpath(path, cases.REPO_ROOT), case_id))


def read_h100_table(path):
    """Read the reduced table written by compose_mpi_scaling.py."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError("%s: empty reduced MPI-scaling table" % path)
    rows.sort(key=lambda r: int(r["ranks"]))
    ranks = [int(r["ranks"]) for r in rows]
    if len(set(ranks)) != len(ranks):
        raise ValueError("%s: duplicate rank row" % path)
    return rows


def mpi_h100(outdir, data_path=None):
    """Distributed-mode H100 rank scaling: total, FFT and writer vs ranks.

    The reduced table ``data/line/h100_mpi_scaling.csv`` is committed.  When
    it is absent the renderer raises :class:`PendingData`, so ``run_all.py``
    can name the missing figure and case instead of skipping.
    """
    path = data_path or cases.data(*H100_TABLE)
    if not os.path.isfile(path):
        raise PendingData(H100_FIGURE, H100_CASE, path)
    rows = read_h100_table(path)
    ranks = [int(r["ranks"]) for r in rows]

    style = {
        "total": dict(color=fs.OKABE["blue"], marker="o", label="total"),
        "fft": dict(color=fs.OKABE["green"], marker="^", label="FFT"),
        "writer": dict(color=fs.OKABE["orange"], marker="s", label="writer/IO"),
    }
    fs.apply_paper_style(kind="line")
    fig, ax = fs.subplots(0.58, 3.45)
    fig.subplots_adjust(left=0.20, right=0.975, bottom=0.165, top=0.955)
    for stage, st in style.items():
        med = np.array([_f(r[stage + "_median_s"]) for r in rows])
        spread = np.array([_f(r[stage + "_spread_s"]) for r in rows])
        ax.errorbar(ranks, med, yerr=spread, fmt="none",
                    ecolor=fs.OKABE["grey"], elinewidth=0.8, capsize=2.5,
                    zorder=2)
        ax.plot(ranks, med, color=st["color"], marker=st["marker"],
                markersize=4.2, linewidth=1.0, label=st["label"], zorder=3)

    host = rows[0].get("host") or "H100 host not recorded"
    gpu = rows[0].get("gpu_model")
    # JURECA hosts record the node name inside the partition token
    # (e.g. "<partition> (<node>), 4x H100 PCIe 80 GB"); the figure shows
    # only the partition and the GPU count, never the node name.
    if host.startswith("jureca"):
        part, _, gpu_count = host.partition(",")
        part = part.split("(")[0].strip()
        host_line = "JURECA %s, %s" % (part.split("-")[-1].upper(),
                                       gpu_count.strip()) if gpu_count.strip() \
            else "JURECA %s" % part.split("-")[-1].upper()
    else:
        host_line = "%s (%s)" % (host, gpu)
    ax.text(0.03, 0.97, host_line, transform=ax.transAxes, va="top",
            ha="left", color=fs.OKABE["grey"])
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlim(0.9, 4.9); ax.set_ylim(0.4, 40)
    ax.set_xticks([1, 2, 4]); ax.set_xticklabels(["1", "2", "4"])
    ax.set_yticks([0.5, 1, 2, 5, 10, 20, 40])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: "%g" % v))
    ax.set_xlabel("ranks (log$_2$)")
    ax.set_ylabel("wall time (s, log)")
    ax.legend(loc="lower right", framealpha=0.92, handlelength=1.5,
              borderpad=0.3)
    ax.grid(alpha=0.3, linewidth=0.5)
    _save(fig, outdir, H100_FIGURE,
          {"figure_id": "mpi_h100",
           "source_data": os.path.relpath(path, cases.REPO_ROOT),
           "case_id": H100_CASE,
           "host": host,
           "gpu_model": gpu,
           "axis_limits": "x 0.9..4.9 (log2); y 0.4..40 (log)"})
    return path


def run(outdir, include_unreferenced=False):
    """Render the scaling family.

    ``mpi`` (the V100 figure) is kept callable but is not rendered by default:
    the manuscript no longer includes it.  ``mpi_h100`` is always attempted and
    returns a pending entry if its reduced table is absent.
    """
    fig_a(outdir); fig_b(outdir)
    if include_unreferenced:
        mpi(outdir)
    w4b(outdir); gamma(outdir); multivel(outdir)
    pending = []
    try:
        mpi_h100(outdir)
    except PendingData as exc:
        pending.append({"figure": exc.figure, "case_id": exc.case_id,
                        "path": os.path.relpath(exc.path, cases.REPO_ROOT),
                        "reason": "reduced MPI-scaling table absent"})
    return pending


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    for item in run(a.out_dir):
        print("PENDING %s (case %s): %s"
              % (item["figure"], item["case_id"], item["path"]))
