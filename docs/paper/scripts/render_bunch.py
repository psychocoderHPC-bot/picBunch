#!/usr/bin/env python3
"""Bunch-form first row for the distribution figures.

The row is three charge-density projection maps (|rho|, log, one shared
colour bar), a marginal-profile axis with the three projected charge curves,
and the panel letters a-d in reading order.  It is drawn on the canonical
:mod:`figstyle` ladder; this module only supplies data, geometry-free drawing
calls and the audit numbers.

Data are particle positions (m) and signed charges (C).  The renderer bins
them with a TSC deposit onto the same cell-centred grid as the field panels
(extended only when a recorded bunch reaches past the stored slice window),
projects the 3-D density onto the three planes and validates the result:

* projected sigma vs particle sigma within 0.75 cell on every axis,
* projected centroid vs particle centroid within 0.5 cell,
* exact charge conservation ``sum(rho)*cell_volume == sum(q)``,
* :func:`axis.assert_beam_axis` passes on every map.

No family's npz carries the additive ``bunch_*`` keys yet, so every figure is
recorded as ``pending_fresh_data`` with the derivation used here.
"""
import os
import sys

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import axis  # noqa: E402
import cases  # noqa: E402
import figstyle as fs  # noqa: E402

US = 1e6
E_CHARGE = 1.602176634e-19
_IDX = {"x": 0, "y": 1, "z": 2}
_SIGMA_TOL_CELL = 0.75
_CENTROID_TOL_CELL = 0.5


class Bunch:
    """Particle positions (m) and signed charges (C) for one figure."""

    def __init__(self, pos_m, q_C, axis_name, cell_m, gamma, source,
                 derivation, pending_fresh_data=True):
        self.pos_m = np.asarray(pos_m, dtype=float)
        self.q_C = np.asarray(q_C, dtype=float)
        if self.pos_m.ndim != 2 or self.pos_m.shape[1] != 3:
            raise ValueError("bunch positions must be (N, 3)")
        if self.q_C.shape != (self.pos_m.shape[0],):
            raise ValueError("bunch charge must be (N,) aligned with positions")
        if axis_name not in ("x", "y", "stationary"):
            raise ValueError(f"unknown bunch axis {axis_name!r}")
        self.axis = axis_name
        self.cell_m = float(cell_m)
        self.gamma = float(gamma)
        self.source = source
        self.derivation = derivation
        self.pending_fresh_data = bool(pending_fresh_data)

    @property
    def n(self):
        return int(self.pos_m.shape[0])


def is_single_particle(bunch):
    """Explicit N==1 guard: a point charge has no bunch form."""
    return bunch.n == 1


# --------------------------------------------------------------- grid / deposit
def _cell_vector(cell):
    cell = np.asarray(cell, dtype=float)
    if cell.ndim == 0:
        cell = np.full(3, float(cell))
    if cell.size == 1:
        cell = np.full(3, float(cell.ravel()[0]))
    return cell.astype(float)


def grid_from_particles(pos_m, cell, margin_cells=2):
    """Cell-centred grid covering every particle plus a TSC margin."""
    cell = _cell_vector(cell)
    lo = pos_m.min(axis=0) - margin_cells * cell
    hi = pos_m.max(axis=0) + margin_cells * cell
    n = np.maximum(np.ceil((hi - lo) / cell).astype(int), 1)
    return lo, [int(v) for v in n], cell


def grid_from_field(pos_m, origin, n, cell, margin_cells=2):
    """Recorded field grid, extended only when the bunch reaches past it."""
    origin = np.asarray(origin, dtype=float)
    n = np.asarray(n, dtype=int)
    cell = _cell_vector(cell)
    lo = origin.copy()
    hi = origin + n * cell
    lo = np.minimum(lo, pos_m.min(axis=0) - margin_cells * cell)
    hi = np.maximum(hi, pos_m.max(axis=0) + margin_cells * cell)
    n = np.maximum(np.ceil((hi - lo) / cell).astype(int), 1)
    return lo, [int(v) for v in n], cell


def _tsc_weights(f):
    return (0.5 * (0.5 - f) ** 2,
            0.75 - f * f,
            0.5 * (0.5 + f) ** 2)


def deposit_tsc(bunch, origin, n, cell):
    """TSC-deposit the signed charge onto a cell-centred 3-D grid."""
    rho = np.zeros(tuple(int(v) for v in n), dtype=float)
    origin = np.asarray(origin, dtype=float)
    cell = np.asarray(cell, dtype=float)
    for p, q in zip(bunch.pos_m, bunch.q_C):
        nodes = []
        weights = []
        for d in range(3):
            s = (p[d] - origin[d]) / cell[d] - 0.5
            i = int(np.floor(s))
            f = s - i
            nodes.append((i - 1, i, i + 1))
            weights.append(_tsc_weights(f))
        for a in range(3):
            ia = nodes[0][a]
            if ia < 0 or ia >= n[0]:
                continue
            wa = weights[0][a]
            if wa == 0.0:
                continue
            for b in range(3):
                ib = nodes[1][b]
                if ib < 0 or ib >= n[1]:
                    continue
                wb = weights[1][b]
                if wb == 0.0:
                    continue
                for c in range(3):
                    ic = nodes[2][c]
                    if ic < 0 or ic >= n[2]:
                        continue
                    wc = weights[2][c]
                    if wc == 0.0:
                        continue
                    rho[ia, ib, ic] += q * wa * wb * wc
    # Charge per voxel -> charge density (C/m^3), so sum(rho)*cell_volume
    # equals the deposited charge exactly.
    rho /= float(np.prod(cell))
    return rho


def _coords(origin, n, cell):
    origin = np.asarray(origin, dtype=float)
    cell = np.asarray(cell, dtype=float)
    return [origin[d] + (np.arange(int(n[d])) + 0.5) * cell[d]
            for d in range(3)]


def _plane_map(rho, coords, cell, horiz_name, vert_name):
    """Project the 3-D density to a surface density in (vert, horiz) order."""
    h = _IDX[horiz_name]
    v = _IDX[vert_name]
    out = 3 - h - v
    arr0 = rho.sum(axis=out) * cell[out]
    rest = [a for a in range(3) if a != out]
    arr = arr0 if rest == [v, h] else arr0.T
    return arr, coords[h], coords[v]


def _marginal(rho, coords, cell, name):
    a = _IDX[name]
    others = [b for b in range(3) if b != a]
    lam = rho.sum(axis=tuple(others)) * cell[others[0]] * cell[others[1]]
    return lam, coords[a]


def _weighted_moments(values, coord, weights):
    s = np.sum(weights)
    if s == 0.0:
        return None, None
    mean = float(np.sum(weights * coord) / s)
    var = float(np.sum(weights * (coord - mean) ** 2) / s)
    return mean, float(np.sqrt(max(var, 0.0)))


# ------------------------------------------------------------------- views
def _views_for_axis(beam):
    if beam == "y":
        return [("xy", "y", "x"), ("zy", "y", "z"), ("xz", "x", "z")]
    # axial-x beam and the stationary gamma=1 case share the same geometry
    return [("xy", "x", "y"), ("zy", "y", "z"), ("xz", "x", "z")]


def _ordered_views(views, figure_plane):
    if figure_plane == "zy":
        order = ["zy", "xz", "xy"]
    else:
        order = ["xy", "zy", "xz"]
    by_plane = {v[0]: v for v in views}
    return [by_plane[p] for p in order]


# ------------------------------------------------------------------ compute
def compute(bunch, dataset, figure_name, figure_plane, origin=None, n=None,
            cell=None, source=""):
    """Deposit, project, draw order and validate one bunch-form row."""
    if origin is None or n is None:
        origin, n, cell = grid_from_particles(bunch.pos_m, bunch.cell_m)
    else:
        origin, n, cell = grid_from_field(bunch.pos_m, origin, n, cell)
    cell = np.asarray(cell, dtype=float)
    rho = deposit_tsc(bunch, origin, n, cell)
    coords = _coords(origin, n, cell)

    views = _views_for_axis(bunch.axis)
    looked = _ordered_views(views, figure_plane)
    letters = "abc"
    maps = []
    for letter, (plane, horiz_name, vert_name) in zip(letters, looked):
        arr, hcoord, vcoord = _plane_map(rho, coords, cell, horiz_name,
                                         vert_name)
        arr = np.abs(arr)
        maps.append({
            "plane": plane, "horiz_name": horiz_name,
            "vert_name": vert_name, "letter": letter,
            "arr": arr,
            "horiz_um": hcoord * US, "vert_um": vcoord * US,
            "xlim": (float(hcoord[0] - 0.5 * cell[_IDX[horiz_name]]) * US,
                     float(hcoord[-1] + 0.5 * cell[_IDX[horiz_name]]) * US),
            "ylim": (float(vcoord[0] - 0.5 * cell[_IDX[vert_name]]) * US,
                     float(vcoord[-1] + 0.5 * cell[_IDX[vert_name]]) * US),
        })

    marginals = []
    for name in ("x", "y", "z"):
        lam, coord = _marginal(rho, coords, cell, name)
        marginals.append({"name": name,
                          "charge_per_len": np.abs(lam),
                          "coord_um": coord * US})

    peak = max(float(np.max(m["arr"])) for m in maps)
    if not np.isfinite(peak) or peak <= 0.0:
        peak = 1.0
    floor = max(peak * fs.SQ_FLOOR_RATIO, np.finfo(float).tiny)
    density_norm = mcolors.LogNorm(vmin=floor, vmax=peak)
    density_cmap = fs.cmap_for("density")
    mpeak = max(float(np.max(m["charge_per_len"])) for m in marginals)
    if not np.isfinite(mpeak) or mpeak <= 0.0:
        mpeak = 1.0
    mfloor = max(mpeak * fs.SQ_FLOOR_RATIO, np.finfo(float).tiny)

    validation = _validate(bunch, maps, marginals, rho, cell, dataset,
                           figure_name, origin, n)

    return {
        "maps": maps, "marginals": marginals,
        "norm": density_norm, "cmap": density_cmap,
        "map_peak": peak, "map_floor": floor,
        "marg_peak": mpeak, "marg_floor": mfloor,
        "validation": validation, "n": n, "origin": list(origin),
        "cell": list(cell), "source": source,
    }


def _validate(bunch, maps, marginals, rho, cell, dataset, figure_name,
              origin, n):
    qabs = np.abs(bunch.q_C)
    particle = {}
    for name in ("x", "y", "z"):
        d = _IDX[name]
        mean, sigma = _weighted_moments(bunch.pos_m[:, d] * US,
                                        bunch.pos_m[:, d] * US, qabs)
        particle[name] = {
            "centroid_um": mean,
            "sigma_um": sigma,
            "centroid_cell": None if mean is None else mean / (cell[d] * US),
            "sigma_cell": None if sigma is None else sigma / (cell[d] * US),
        }

    rho_moments = {}
    per_axis = {}
    ok = True
    for m in marginals:
        name = m["name"]
        d = _IDX[name]
        coord_um = m["coord_um"]
        lam = m["charge_per_len"]
        mean, sigma = _weighted_moments(coord_um, coord_um, lam)
        rho_moments[name] = {
            "centroid_um": mean,
            "sigma_um": sigma,
            "centroid_cell": None if mean is None else mean / (cell[d] * US),
            "sigma_cell": None if sigma is None else sigma / (cell[d] * US),
        }
        p = particle[name]
        dsig = abs(rho_moments[name]["sigma_cell"] - p["sigma_cell"])
        dcen = abs(rho_moments[name]["centroid_cell"] - p["centroid_cell"])
        axis_ok = bool(dsig <= _SIGMA_TOL_CELL
                       and dcen <= _CENTROID_TOL_CELL)
        per_axis[name] = {
            "particle_sigma_cell": p["sigma_cell"],
            "rho_sigma_cell": rho_moments[name]["sigma_cell"],
            "sigma_delta_cell": dsig,
            "particle_centroid_cell": p["centroid_cell"],
            "rho_centroid_cell": rho_moments[name]["centroid_cell"],
            "centroid_delta_cell": dcen,
            "pass": axis_ok,
        }
        ok &= axis_ok

    cell_vol = float(np.prod(cell))
    sum_rho = float(np.sum(rho) * cell_vol)
    sum_q = float(np.sum(bunch.q_C))
    denom = max(abs(sum_q), np.finfo(float).tiny)
    charge_rel = abs(sum_rho - sum_q) / denom
    charge_ok = bool(charge_rel <= 1e-12)

    axis_results = []
    for m in maps:
        res = axis.assert_beam_axis(
            dataset, m["plane"], m["arr"], m["horiz_um"], m["vert_um"],
            n=n, cell=cell, origin=origin, gamma=bunch.gamma,
            centres=True, source=figure_name, figure=figure_name,
            label=m["letter"])
        axis_results.append(res)

    if not ok:
        raise AssertionError(
            f"{figure_name}: bunch-form moment check failed: {per_axis}")
    if not charge_ok:
        raise AssertionError(
            f"{figure_name}: charge not conserved: sum(rho)*V={sum_rho!r} "
            f"vs sum(q)={sum_q!r} (rel {charge_rel:.3e})")
    return {
        "sigma_tolerance_cell": _SIGMA_TOL_CELL,
        "centroid_tolerance_cell": _CENTROID_TOL_CELL,
        "particle": particle,
        "rho": rho_moments,
        "per_axis": per_axis,
        "pass": True,
        "charge": {"sum_rho_C": sum_rho, "sum_q_C": sum_q,
                   "relative_error": charge_rel, "pass": charge_ok},
        "axis_assertions": axis_results,
    }


# --------------------------------------------------------------------- draw
_AXIS_COLOR = {"x": fs.OKABE["blue"], "y": fs.OKABE["vermillion"],
               "z": fs.OKABE["green"]}


def _cue(ax, m, beam):
    """Beam-axis arrow on in-plane maps, out-of-plane note elsewhere."""
    if beam == "stationary":
        fs.panel_note(ax, "stationary ($\\gamma=1$)\nno beam axis",
                      x=0.5, y=0.10, ha="center")
        return
    if beam in (m["horiz_name"], m["vert_name"]):
        ax.annotate("", xy=(0.92, 0.075), xytext=(0.58, 0.075),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=fs.TEXT_COLOR,
                                    linewidth=fs.LINEWIDTH))
        fs.panel_note(ax, r"beam $\rightarrow$", x=0.5, y=0.16, ha="center")
    else:
        fs.panel_note(ax, f"beam axis ({beam})\nout of plane",
                      x=0.5, y=0.10, ha="center")


def draw_row(fig, fig_w_in, fig_h_in, y_bottom_in, bunch, computed, dataset,
             figure_name):
    """Draw the row at ``y_bottom_in`` (inches from the figure bottom)."""
    geo = fs.bunch_row_geometry(fig_w_in)
    yb = float(y_bottom_in) / float(fig_h_in)
    mh = geo["map_in"] / float(fig_h_in)

    for i, m in enumerate(computed["maps"]):
        ax = fig.add_axes([geo["x"][i], yb, geo["width"][i], mh])
        fs.draw_raster_panel(ax, m["horiz_um"], m["vert_um"], m["arr"],
                             "density", computed["norm"], computed["cmap"])
        ax.set_xlim(m["xlim"])
        ax.set_ylim(m["ylim"])
        ax.set_aspect("auto")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(3, prune="both"))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(3, prune="both"))
        xl, yl = axis.axis_labels(dataset, m["plane"], computed["source"])
        ax.set_xlabel(xl)
        if i == 0:
            ax.set_ylabel(yl)
        else:
            ax.tick_params(labelleft=False)
        fs.panel_title(ax, m["letter"],
                       f"${m['horiz_name']}$–${m['vert_name']}$")
        _cue(ax, m, bunch.axis)

    axm = fig.add_axes([geo["x"][3], yb, geo["width"][3], mh])
    for marg in computed["marginals"]:
        name = marg["name"]
        axm.plot(marg["coord_um"], marg["charge_per_len"],
                 color=_AXIS_COLOR[name], label=f"${name}$")
    axm.set_yscale("log")
    axm.set_ylim(computed["marg_floor"], computed["marg_peak"])
    axm.set_xlabel(r"position [$\mu$m]")
    axm.xaxis.set_major_locator(mticker.MaxNLocator(3, prune="both"))
    axm.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=4))
    axm.yaxis.set_minor_locator(mticker.NullLocator())
    axm.legend(loc="upper right", frameon=False, handlelength=1.0,
               borderpad=0.2, labelspacing=0.25)
    fs.panel_title(axm, "d", r"$p$ [C/m]")

    cax = fig.add_axes([geo["x"][4], yb, geo["width"][4], mh])
    sm = plt.cm.ScalarMappable(computed["norm"], computed["cmap"])
    lo_exp = int(np.floor(np.log10(computed["map_floor"])))
    hi_exp = int(np.floor(np.log10(computed["map_peak"])))
    exps = list(range(lo_exp, hi_exp + 1))
    if len(exps) < 2:
        exps = [lo_exp, max(hi_exp, lo_exp + 1)]
    ticks = [10.0 ** e for e in exps]
    labels = [r"$10^{%d}$" % e for e in exps]
    fs.draw_colorbar(fig, sm, cax=cax, ticks=ticks, ticklabels=labels,
                     label=r"$|\rho|$ [C/m$^2$] (log)")
    return geo


def manifest(bunch, computed, geo):
    """Audit fields added to the figure's per-figure audit JSON."""
    v = computed["validation"]
    return {
        "pending_fresh_data": bunch.pending_fresh_data,
        "bunch_data_source": bunch.derivation,
        "bunch_axis": bunch.axis,
        "bunch_gamma": bunch.gamma,
        "bunch_cell_m": bunch.cell_m,
        "bunch_particles": bunch.n,
        "bunch_row": {
            "layout": "first row: three |rho| maps, marginals, one colour bar",
            "map_mm": round(geo["map_in"] * 25.4, 2),
            "marginal_mm": round(geo["marg_in"] * 25.4, 2),
            "bar_mm": round(geo["bar_in"] * 25.4, 2),
            "gap_mm": round(geo["gap_in"] * 25.4, 2),
            "map_order": [m["plane"] for m in computed["maps"]],
            "map_floor": computed["map_floor"],
            "map_peak": computed["map_peak"],
            "marginal_floor": computed["marg_floor"],
            "marginal_peak": computed["marg_peak"],
        },
        "bunch_validation": v,
    }


def layout_kwargs(bunch, panel_scale=1.0, top_in=0.14, bottom_in=0.58,
                  row_gap_in=0.40):
    """Keyword arguments that add the row and tighten the field block."""
    if bunch is None:
        return {}
    return {"bunch": bunch, "panel_scale": panel_scale, "top_in": top_in,
            "bottom_in": bottom_in, "row_gap_in": row_gap_in}


class BunchRow:
    """A computed bunch-form row ready to be drawn into a field figure."""

    def __init__(self, bunch, computed, dataset, figure_name):
        self.bunch = bunch
        self.computed = computed
        self.dataset = dataset
        self.figure_name = figure_name

    def geometry(self, fig_w_in):
        return fs.bunch_row_geometry(fig_w_in)

    def reserve_in(self, fig_w_in):
        return self.geometry(fig_w_in)["reserve_in"]

    def draw(self, fig, fig_w_in, fig_h_in, y_bottom_in):
        return draw_row(fig, fig_w_in, fig_h_in, y_bottom_in, self.bunch,
                        self.computed, self.dataset, self.figure_name)

    def manifest(self, fig_w_in):
        return manifest(self.bunch, self.computed, self.geometry(fig_w_in))


def build_row(bunch, dataset, figure_name, figure_plane, origin=None,
              n=None, cell=None, source=""):
    """Compute and wrap one bunch-form row."""
    if is_single_particle(bunch):
        raise ValueError(
            f"{figure_name}: a single point charge (N==1) has no bunch form; "
            "the field figures are left unchanged")
    computed = compute(bunch, dataset, figure_name, figure_plane,
                       origin=origin, n=n, cell=cell, source=source)
    return BunchRow(bunch, computed, dataset, figure_name)
