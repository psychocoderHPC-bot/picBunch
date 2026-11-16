#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
#
# Compare the binned mesh fields against the TSC-convolved continuum reference
# for a mixed-velocity bunch (tools/velocity-bins/indfield.py is the arbiter,
# the same module the paper's N2b/W4b measurements used).
#
# For a bunch with per-particle velocities u_p the continuum target is
#
#   T(x) = sum_p q_p w_p I(x - x_p; u_p),   I = int S(s) K(x - s*cell) ds^3
#
# with the TSC shape S and the moving-point kernel K.  The binned mesh
# replaces u_p by the box representative u_b, so the pure binning error is
#   Tb(x) - T(x) = sum_b (sum_{p in b} q_p w_p) I(x - x_p; u_b) - T(x)
# and the mesh error decomposes as
#   M_B - T = (M_B - Tb) + (Tb - T)   (discretization + binning).
#
# Usage:
#   compare_velocity_bins.py PREFIX [--bins 1,2,4,8,16] [--workers 32]
#       [--order 12] [--refine 1] [--refine100 2] [--fit] [--bands 0-1,1-2,2-4,4-8,8-16,16+]
#
# Reads PREFIX_bunch.txt, PREFIX_field_B<B>.bin, PREFIX_meta_B<B>.txt and
# writes PREFIX_compare.md / PREFIX_compare.json.  ``--fit`` reports the
# per-box-count ratio measured/modelSum (the calibration constant).

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import indfield  # noqa: E402

K_COULOMB = 8.9875517873681764e9
C_LIGHT = 299792458.0
YEE = indfield.OFFSETS  # (6, 3), cell units
COMPONENTS = ("Ex", "Ey", "Ez", "Bx", "By", "Bz")
UNITS = ("V/m", "V/m", "V/m", "T", "T", "T")

#: Version of the ``metrics_v2`` payload (new keys only; legacy keys untouched).
METRICS_V2_SCHEMA = "velocity-bin-metrics/2.0"
#: Relative-error floor, identical to the one already used by ``error_stats``.
REL_EPS = 1e-30

#: The far-field shell is the bunch box translated by ``+2 R`` along one axis.
#: ``+x`` matches the spec's "box of the same size centred on the bunch but
#: offset by 2 R"; translating all three axes produced a corner sliver.
FAR_FIELD_AXIS = 0
FAR_FIELD_AXIS_NAME = "x"
#: Empty node layers required between the far-field box and every domain face.
FAR_FIELD_MARGIN_CELLS = 2
FAR_FIELD_POLICIES = ("error", "clip")


class FarFieldGeometryError(ValueError):
    """The domain cannot hold the far-field box as requested."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prefix")
    p.add_argument("--bins", default="1,2,4,8,16")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--order", type=int, default=12)
    p.add_argument("--refine", type=int, default=1)
    p.add_argument("--refine100", type=int, default=2,
                   help="refine level used when gamma >= 100 (beamed integrands)")
    p.add_argument("--fit", action="store_true")
    p.add_argument("--bands", default="0-1,1-2,2-4,4-8,8-16,16+")
    p.add_argument("--metrics-v2", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="emit the bunch-box / far-field metrics under 'metrics_v2' "
                        "(default: on; --no-metrics-v2 reproduces the legacy JSON)")
    p.add_argument("--far-field-policy", choices=FAR_FIELD_POLICIES,
                   default="error",
                   help="what to do when the domain cannot hold the far-field box "
                        "with the required margin: 'error' (default) fails loudly, "
                        "'clip' clips explicitly and records far_field_clipped")
    return p.parse_args()


def parse_bands(text: str):
    bands = []
    for part in text.split(","):
        part = part.strip()
        if part.endswith("+"):
            lo = float(part[:-1])
            bands.append((lo, math.inf, f"[{lo:g}, inf)"))
        else:
            lo, hi = (float(v) for v in part.split("-"))
            bands.append((lo, hi, f"[{lo:g}, {hi:g})"))
    return bands


def read_meta(prefix: str, B: int) -> dict:
    meta: dict = {}
    with open(f"{prefix}_meta_B{B}.txt") as f:
        for line in f:
            tok = line.split()
            meta[tok[0]] = tok[1:]
    return meta


def read_bunch(prefix: str):
    x, y, z, ux, uy, uz, w, q, box = [], [], [], [], [], [], [], [], []
    with open(f"{prefix}_bunch.txt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            v = line.split()
            x.append(float(v[0])); y.append(float(v[1])); z.append(float(v[2]))
            ux.append(float(v[3])); uy.append(float(v[4])); uz.append(float(v[5]))
            w.append(float(v[6])); q.append(float(v[7])); box.append(int(v[8]))
    return {k: np.asarray(v) for k, v in
            dict(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, w=w, q=q).items()}


def read_field(prefix: str, B: int, n) -> np.ndarray:
    data = np.fromfile(f"{prefix}_field_B{B}.bin", dtype=np.float64)
    return data.reshape(6, n[0], n[1], n[2]).copy()


def node_windows(bunch: dict, n, origin, cell, margin: float = 2.0):
    """Per particle, the (i, j, k) node tuples whose Yee positions are within
    the TSC support (|s| < 1.5 per axis, plus the 0.5-cell Yee shift)."""
    xs = bunch["x"]; ys = bunch["y"]; zs = bunch["z"]
    windows = []
    for p in range(len(xs)):
        lo = []
        hi = []
        for pos, c, o, nn in ((xs[p], cell[0], origin[0], n[0]),
                              (ys[p], cell[1], origin[1], n[1]),
                              (zs[p], cell[2], origin[2], n[2])):
            a = math.floor((pos - margin * c - o) / c) - 1
            b = math.ceil((pos + margin * c - o) / c) + 1
            lo.append(max(0, a))
            hi.append(min(nn, b + 1))
        triples = [(i, j, k) for i in range(lo[0], hi[0])
                   for j in range(lo[1], hi[1]) for k in range(lo[2], hi[2])]
        windows.append(triples)
    return windows


def worker(args):
    (p_lo, p_hi, bunch, u_use, n, origin, cell, order, refine) = args
    n = tuple(n)
    origin = np.asarray(origin); cell = np.asarray(cell)
    acc = np.zeros((6, n[0], n[1], n[2]))
    xs = bunch["x"]; ys = bunch["y"]; zs = bunch["z"]
    ws = bunch["w"]; qs = bunch["q"]
    for p in range(p_lo, p_hi):
        x_obs_src = np.array([xs[p], ys[p], zs[p]])
        wq = ws[p] * qs[p]
        if wq == 0.0:
            continue
        lo = []
        hi = []
        for pos, c, o, nn in ((xs[p], cell[0], origin[0], n[0]),
                              (ys[p], cell[1], origin[1], n[1]),
                              (zs[p], cell[2], origin[2], n[2])):
            a = math.floor((pos - 2.0 * c - o) / c) - 1
            b = math.ceil((pos + 2.0 * c - o) / c) + 1
            lo.append(max(0, a))
            hi.append(min(nn, b + 1))
        for i in range(lo[0], hi[0]):
            for j in range(lo[1], hi[1]):
                for k in range(lo[2], hi[2]):
                    node = origin + np.array([i, j, k]) * cell
                    for c in range(6):
                        x_obs = node + YEE[c] * cell
                        e, b = indfield.integrate_point(
                            x_obs, x_obs_src, u_use[p], cell[0], cell[1], cell[2],
                            order=order, refine=refine)
                        val = e[c] if c < 3 else b[c - 3]
                        acc[c, i, j, k] += wq * (K_COULOMB if c < 3 else K_COULOMB / C_LIGHT) * val
    return acc


def compute_reference(bunch, u_use, n, origin, cell, order, refine, workers):
    nparts = len(bunch["x"])
    chunks = max(1, nparts // workers)
    jobs = []
    for lo in range(0, nparts, chunks):
        jobs.append((lo, min(nparts, lo + chunks), bunch, u_use, tuple(n),
                     [origin[0], origin[1], origin[2]], [cell[0], cell[1], cell[2]],
                     order, refine))
    if workers <= 1:
        return sum(worker(j) for j in jobs)
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        return sum(ex.map(worker, jobs))


def nearest_distance(bunch, n, origin, cell, dmax: float):
    """Distance (cells) from every node within dmax of some particle; -1 beyond."""
    d = np.full((n[0], n[1], n[2]), -1.0)
    xs = bunch["x"]; ys = bunch["y"]; zs = bunch["z"]
    cell = np.asarray(cell)
    origin = np.asarray(origin)
    nodes = np.stack(np.meshgrid(
        np.arange(n[0]), np.arange(n[1]), np.arange(n[2]), indexing="ij"), axis=-1)
    pos = origin + nodes * cell  # (n0, n1, n2, 3)
    best = np.full(pos.shape[:-1], np.inf)
    for p in range(len(xs)):
        src = np.array([xs[p], ys[p], zs[p]])
        diff = pos - src
        dist = np.sqrt(np.sum(diff * diff, axis=-1)) / cell[0]
        best = np.minimum(best, dist)
    mask = best <= dmax
    d[mask] = best[mask]
    return d


def band_index(dist, bands):
    idx = np.full(dist.shape, -1, dtype=int)
    for bi, (lo, hi, _label) in enumerate(bands):
        sel = (dist >= lo) & (dist < hi)
        idx[sel] = bi
    return idx


def error_stats(mesh, ref, dist, bands, band_idx):
    """Worst per-component relative (unfloored, over the reference window) and
    absolute error, overall and per distance band."""
    window = dist >= 0
    rel_all = np.zeros(6)
    abs_all = np.zeros(6)
    for c in range(6):
        m = mesh[c][window]
        r = ref[c][window]
        denom = np.maximum(np.abs(r), 1e-30)
        rel_all[c] = np.max(np.abs(m - r) / denom)
        abs_all[c] = np.max(np.abs(m - r))
    per_band = []
    for bi, (lo, hi, label) in enumerate(bands):
        sel = band_idx == bi
        if not sel.any():
            per_band.append({"band": label, "n": 0, "rel": [np.nan] * 6,
                             "abs": [np.nan] * 6, "worst_rel": np.nan, "worst_abs": np.nan})
            continue
        rel = np.zeros(6)
        absv = np.zeros(6)
        for c in range(6):
            m = mesh[c][sel]
            r = ref[c][sel]
            denom = np.maximum(np.abs(r), 1e-30)
            rel[c] = np.max(np.abs(m - r) / denom)
            absv[c] = np.max(np.abs(m - r))
        per_band.append({"band": label, "n": int(sel.sum()), "rel": rel.tolist(),
                         "abs": absv.tolist(), "worst_rel": float(np.max(rel)),
                         "worst_abs": float(np.max(absv))})
    return {
        "window_nodes": int(window.sum()),
        "rel": rel_all.tolist(),
        "abs": abs_all.tolist(),
        "worst_rel": float(np.max(rel_all)),
        "worst_abs": float(np.max(abs_all)),
        "worst_rel_component": int(np.argmax(rel_all)),
        "bands": per_band,
        "dominant_band": per_band[int(np.argmax([b["worst_rel"] for b in per_band]))]["band"]
        if any(b["n"] for b in per_band) else None,
    }


def bunch_bounding_box(bunch: dict) -> dict:
    """Axis-aligned bounding box of the actual particle positions.

    The box is inclusive on every axis; ``R`` is half of the largest extent,
    which is the radius used for the far-field translation (``2 R``).
    """
    if len(bunch["x"]) == 0:
        nan = [math.nan] * 3
        return {"n_particles": 0, "min": nan, "max": nan,
                "extent": nan, "R": 0.0}
    lo = np.array([float(np.min(bunch[a])) for a in ("x", "y", "z")])
    hi = np.array([float(np.max(bunch[a])) for a in ("x", "y", "z")])
    extent = hi - lo
    return {"n_particles": int(len(bunch["x"])),
            "min": lo.tolist(), "max": hi.tolist(),
            "extent": extent.tolist(),
            "R": 0.5 * float(np.max(extent))}


def box_node_mask(n, origin, cell, lo, hi) -> np.ndarray:
    """Boolean ``(n0, n1, n2)`` mask of base nodes inside a closed box.

    A base node sits at ``origin + index * cell`` (the same convention used by
    :func:`nearest_distance` and by the reference accumulation in
    :func:`worker`); the Yee half-cell offsets are not applied here, so all six
    components share one node mask.  A tiny roundoff guard (``1e-12 * cell``)
    is added to the inclusive bounds.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    n = tuple(int(v) for v in n)
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        return np.zeros(n, dtype=bool)
    axis_inside = []
    for a in range(3):
        idx = np.arange(n[a], dtype=float)
        pos = float(origin[a]) + idx * float(cell[a])
        tol = 1e-12 * abs(float(cell[a]))
        axis_inside.append((pos >= lo[a] - tol) & (pos <= hi[a] + tol))
    return (axis_inside[0][:, None, None]
            & axis_inside[1][None, :, None]
            & axis_inside[2][None, None, :])


def _node_position(node, origin, cell):
    return [float(origin[a]) + float(node[a]) * float(cell[a]) for a in range(3)]


def component_error_stats(mesh_c, ref_c, mask, *, eps: float = REL_EPS,
                          origin=None, cell=None) -> dict:
    """Robust relative-error summary for one component over one node mask.

    ``rel = |mesh - ref| / max(|ref|, eps)``.  Non-finite samples are counted
    (never silently dropped) and excluded from max/percentile/RMS.  If the mask
    is empty or everything is non-finite the numeric fields are ``None``.
    """
    n = int(np.count_nonzero(mask))
    out = {
        "n": n, "n_finite": 0, "n_nonfinite": 0,
        "max_rel": None, "max_abs": None,
        "max_rel_node": None, "max_abs_node": None,
        "max_rel_position": None, "max_abs_position": None,
        "p99_rel": None, "rms_rel": None,
    }
    if n == 0:
        return out
    m = np.asarray(mesh_c)[mask]
    r = np.asarray(ref_c)[mask]
    nodes = np.argwhere(mask)
    finite = np.isfinite(m) & np.isfinite(r)
    n_finite = int(np.count_nonzero(finite))
    out["n_finite"] = n_finite
    out["n_nonfinite"] = n - n_finite
    if n_finite == 0:
        return out
    mf = m[finite]
    rf = r[finite]
    fnodes = nodes[finite]
    rel = np.abs(mf - rf) / np.maximum(np.abs(rf), eps)
    absv = np.abs(mf - rf)
    i_rel = int(np.argmax(rel))
    i_abs = int(np.argmax(absv))
    out["max_rel"] = float(rel[i_rel])
    out["max_abs"] = float(absv[i_abs])
    out["max_rel_node"] = [int(v) for v in fnodes[i_rel]]
    out["max_abs_node"] = [int(v) for v in fnodes[i_abs]]
    if origin is not None and cell is not None:
        out["max_rel_position"] = _node_position(fnodes[i_rel], origin, cell)
        out["max_abs_position"] = _node_position(fnodes[i_abs], origin, cell)
    out["p99_rel"] = float(np.percentile(rel, 99))
    out["rms_rel"] = float(np.sqrt(np.mean(rel * rel)))
    return out


def inclusive_index_bounds(lo, hi, origin, cell):
    """Inclusive base-node index range of a closed box, before domain clipping.

    Returns ``{"lo": [i, j, k], "hi": [i, j, k]}`` or ``None`` when the box
    holds no base node (or the bounds are non-finite).  The same roundoff guard
    as :func:`box_node_mask` is used so the bounds and the mask agree exactly.
    """
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        return None
    first = []
    last = []
    for a in range(3):
        c = float(cell[a])
        o = float(origin[a])
        tol = 1e-12 * abs(c)
        first.append(int(math.ceil((float(lo[a]) - o - tol) / c)))
        last.append(int(math.floor((float(hi[a]) - o + tol) / c)))
    if any(first[a] > last[a] for a in range(3)):
        return None
    return {"lo": first, "hi": last}


def _containment(pre_bounds, n, margin):
    """Classify the unclipped far-field box against the domain.

    ``contained`` means the box fits with ``margin`` empty node layers on every
    side; ``clipped`` means at least one node of the box falls outside the
    domain (the mask built by :func:`box_node_mask` silently drops it).
    """
    n = [int(v) for v in n]
    if pre_bounds is None:
        return {"contained": True, "clipped": False, "margin_ok": True,
                "pre": None, "post": None, "margins": None,
                "required_n": list(n), "lower_violation": [], "detail": []}

    first = pre_bounds["lo"]
    last = pre_bounds["hi"]
    post_lo = [max(0, first[a]) for a in range(3)]
    post_hi = [min(n[a] - 1, last[a]) for a in range(3)]
    post = None
    if all(post_lo[a] <= post_hi[a] for a in range(3)):
        post = {"lo": post_lo, "hi": post_hi}

    lower_ok = [first[a] >= margin for a in range(3)]
    upper_ok = [last[a] <= n[a] - 1 - margin for a in range(3)]
    clipped = any(first[a] < 0 or last[a] > n[a] - 1 for a in range(3))
    margins = {"lo": [first[a] for a in range(3)],
               "hi": [(n[a] - 1) - last[a] for a in range(3)]}
    # Smallest n (per axis) that would give every part of the box a >= margin
    # gap to the upper face.  A lower-face violation needs an origin shift.
    required_n = [max(last[a] + margin + 1, margin + 1) for a in range(3)]
    detail = []
    for a, name in enumerate("xyz"):
        detail.append(f"{name}[{first[a]}..{last[a]}] needs n>={required_n[a]}")
        if not lower_ok[a]:
            detail.append(
                f"{name} starts at index {first[a]} (< {margin}-cell lower "
                f"margin), so the origin must move or the axis must grow")
    return {
        "contained": all(lower_ok) and all(upper_ok),
        "clipped": clipped,
        "margin_ok": all(lower_ok) and all(upper_ok),
        "pre": pre_bounds,
        "post": post,
        "margins": margins,
        "required_n": required_n,
        "lower_violation": [not v for v in lower_ok],
        "detail": detail,
    }


def _required_extent_message(info, n, origin, cell, margin):
    req = info["required_n"]
    have = [int(v) for v in n]
    need_m = [req[a] * float(cell[a]) for a in range(3)]
    return (
        "far-field box (bunch box translated by +2R along "
        f"+{FAR_FIELD_AXIS_NAME} only) is not contained without clipping: "
        f"domain {have[0]}x{have[1]}x{have[2]} cells "
        f"(origin {list(origin)}, cell {list(cell)}) is too small. "
        f"Required domain extent is at least {req[0]}x{req[1]}x{req[2]} cells "
        f"= {need_m[0]:.6e} x {need_m[1]:.6e} x {need_m[2]:.6e} m with "
        f">= {margin} empty cell layers on every side. "
        + "; ".join(info["detail"])
        + ". Pass --far-field-policy clip to clip explicitly (the JSON then "
        "reports far_field_clipped=true and the pre-clip bounds).")


def prepare_metrics_v2(bunch: dict, n, origin, cell, *,
                       eps: float = REL_EPS,
                       far_field_policy: str = "error",
                       margin_cells: int = FAR_FIELD_MARGIN_CELLS) -> dict:
    """Precompute the bunch-box and far-field masks once per bunch.

    The in-bunch mask is the inclusive node box of the particle min/max (the
    legacy behaviour).  The far-field shell is that same box translated by
    ``+2 R`` along ``+x`` only, where ``R`` is half the largest bunch extent.
    With ``far_field_policy="error"`` a domain that cannot hold the translated
    box with ``margin_cells`` empty layers per face raises
    :class:`FarFieldGeometryError`; ``"clip"`` keeps the intersection and
    records the clipping.  No reference field is recomputed here.
    """
    if far_field_policy not in FAR_FIELD_POLICIES:
        raise ValueError(
            f"far_field_policy must be one of {FAR_FIELD_POLICIES}, "
            f"got {far_field_policy!r}")
    bbox = bunch_bounding_box(bunch)
    offset = 2.0 * bbox["R"]
    ff_lo = [bbox["min"][a] for a in range(3)]
    ff_hi = [bbox["max"][a] for a in range(3)]
    ff_lo[FAR_FIELD_AXIS] = bbox["min"][FAR_FIELD_AXIS] + offset
    ff_hi[FAR_FIELD_AXIS] = bbox["max"][FAR_FIELD_AXIS] + offset

    ff_pre = inclusive_index_bounds(ff_lo, ff_hi, origin, cell)
    containment = _containment(ff_pre, n, int(margin_cells))
    if far_field_policy == "error" and not containment["contained"]:
        raise FarFieldGeometryError(
            _required_extent_message(containment, n, origin, cell,
                                     int(margin_cells)))

    in_pre = inclusive_index_bounds(bbox["min"], bbox["max"], origin, cell)
    in_post = None
    if in_pre is not None:
        lo = [max(0, in_pre["lo"][a]) for a in range(3)]
        hi = [min(int(n[a]) - 1, in_pre["hi"][a]) for a in range(3)]
        if all(lo[a] <= hi[a] for a in range(3)):
            in_post = {"lo": lo, "hi": hi}

    return {
        "schema": METRICS_V2_SCHEMA,
        "eps": float(eps),
        "n": tuple(int(v) for v in n),
        "origin": [float(v) for v in origin],
        "cell": [float(v) for v in cell],
        "bbox": bbox,
        "far_field_offset": offset,
        "far_field_offset_axis": FAR_FIELD_AXIS_NAME,
        "far_field_offset_vector": [
            offset if a == FAR_FIELD_AXIS else 0.0 for a in range(3)],
        "far_field_min": ff_lo,
        "far_field_max": ff_hi,
        "far_field_policy": far_field_policy,
        "far_field_clipped": bool(containment["clipped"]),
        "far_field_margin_ok": bool(containment["margin_ok"]),
        "far_field_margin_cells": int(margin_cells),
        "far_field_pre_clip_bounds": containment["pre"],
        "far_field_index_bounds": containment["post"],
        "far_field_margins": containment["margins"],
        "far_field_required_domain_n": containment["required_n"],
        "in_bunch_index_bounds": in_post,
        "in_mask": box_node_mask(n, origin, cell, bbox["min"], bbox["max"]),
        "far_mask": box_node_mask(n, origin, cell, ff_lo, ff_hi),
    }


def _region_metrics(mesh, ref, mask, origin, cell, eps: float) -> dict:
    per_component = {}
    for c, name in enumerate(COMPONENTS):
        per_component[name] = component_error_stats(
            mesh[c], ref[c], mask, eps=eps, origin=origin, cell=cell)
    finite = [(name, s["max_rel"]) for name, s in per_component.items()
              if s["max_rel"] is not None]
    if finite:
        worst_name, worst_rel = max(finite, key=lambda item: item[1])
        worst_component = worst_name
        worst_rel = float(worst_rel)
    else:
        worst_component = None
        worst_rel = None
    return {
        "n_nodes": int(np.count_nonzero(mask)),
        "per_component": per_component,
        "worst_rel": worst_rel,
        "worst_component": worst_component,
    }


def compute_metrics_v2(mesh, ref, ctx: dict) -> dict:
    """Build the ``metrics_v2`` payload from already-computed mesh and reference.

    ``mesh`` and ``ref`` are ``(6, n0, n1, n2)`` arrays in ``COMPONENTS`` order;
    the reference evaluation is not repeated.  ``ref`` is the TSC-convolved
    continuum target ``T``, so the reported error is ``mesh - T``.
    """
    origin = ctx["origin"]
    cell = ctx["cell"]
    eps = ctx["eps"]
    bbox = ctx["bbox"]
    return {
        "schema": ctx["schema"],
        "comparison": "mesh_vs_T",
        "eps": eps,
        "rel_error_definition": "|F_binned - F_ref| / max(|F_ref|, eps)",
        "geometry": {
            "bbox_min": list(bbox["min"]),
            "bbox_max": list(bbox["max"]),
            "bbox_extent": list(bbox["extent"]),
            "R": float(bbox["R"]),
            "far_field_offset": float(ctx["far_field_offset"]),
            "far_field_min": list(ctx["far_field_min"]),
            "far_field_max": list(ctx["far_field_max"]),
            "bbox_n_particles": int(bbox["n_particles"]),
            "node_position_convention":
                "origin + index*cell (base node, shared by all six components)",
            # New in E1b: single-axis offset plus the mask bounds and clipping
            # state, so a downstream criterion can never mistake a sliver for
            # a full-size shell.
            "far_field_offset_axis": ctx["far_field_offset_axis"],
            "far_field_offset_vector": list(ctx["far_field_offset_vector"]),
            "far_field_policy": ctx["far_field_policy"],
            "far_field_clipped": bool(ctx["far_field_clipped"]),
            "far_field_margin_ok": bool(ctx["far_field_margin_ok"]),
            "far_field_margin_cells": int(ctx["far_field_margin_cells"]),
            "in_bunch_index_bounds": ctx["in_bunch_index_bounds"],
            "far_field_index_bounds": ctx["far_field_index_bounds"],
            "far_field_pre_clip_bounds": ctx["far_field_pre_clip_bounds"],
            "far_field_margins_cells": ctx["far_field_margins"],
            "far_field_required_domain_n": ctx["far_field_required_domain_n"],
        },
        "in_bunch": _region_metrics(mesh, ref, ctx["in_mask"], origin, cell, eps),
        "far_field": _region_metrics(mesh, ref, ctx["far_mask"], origin, cell, eps),
    }


def fmt(v, sig=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.{sig}e}"


def main() -> None:
    args = parse_args()
    prefix = args.prefix
    bins = [int(b) for b in args.bins.split(",") if b.strip()]
    bands = parse_bands(args.bands)

    bunch = read_bunch(prefix)
    first_meta = read_meta(prefix, bins[0])
    n = [int(v) for v in first_meta["grid"]]
    origin = [float(v) for v in first_meta["origin"]]
    cell = [float(v) for v in first_meta["cell"]]
    gamma = float(first_meta["gamma"][0])
    refine = args.refine100 if gamma >= 100.0 else args.refine

    u_p = np.stack([bunch["ux"], bunch["uy"], bunch["uz"]], axis=1)
    print(f"bunch: {len(bunch['x'])} particles, gamma={gamma}, grid={n}, cell={cell[0]} m, "
          f"reference order={args.order} refine={refine}, workers={args.workers}")

    # Bunch-box and far-field masks are built once and reused for every B; the
    # new metrics only consume the mesh and T arrays computed below, never a
    # second reference pass.  Doing this before the expensive reference pass
    # makes a geometry that cannot hold the shell fail fast instead of after
    # minutes of integration.
    metrics_ctx = None
    if args.metrics_v2:
        try:
            metrics_ctx = prepare_metrics_v2(
                bunch, n, origin, cell,
                far_field_policy=args.far_field_policy)
        except FarFieldGeometryError as exc:
            sys.stderr.write(f"compare_velocity_bins.py: error: {exc}\n")
            sys.exit(2)

    dist = nearest_distance(bunch, tuple(n), origin, cell, dmax=16.0)
    band_idx = band_index(dist, bands)

    # Continuum target with per-particle velocities (independent of B).
    print("computing T (per-particle continuum reference) ...", flush=True)
    T = compute_reference(bunch, u_p, n, origin, cell, args.order, refine, args.workers)

    results = {"bunch": {"sources": len(bunch["x"]), "gamma": gamma,
                         "energy_spread": float(first_meta["energy_spread"][0]),
                         "angle_degrees": float(first_meta["angle_degrees"][0]),
                         "grid": n, "cell": cell[0], "origin": origin,
                         "order": args.order, "refine": refine},
               "runs": {}}
    lines = []
    lines.append(f"# MV1 comparison: `{os.path.basename(prefix)}`")
    lines.append("")
    lines.append(f"Bunch: {len(bunch['x'])} particles, gamma = {gamma:g}, "
                 f"sigma_g/gamma = {first_meta['energy_spread'][0]}, "
                 f"sigma_theta = {first_meta['angle_degrees'][0]} deg; "
                 f"grid {n[0]}^3, cell {cell[0]:g} m.")
    lines.append(f"Reference: TSC-convolved continuum (indfield, order {args.order}, "
                 f"refine {refine}); errors unfloored, over nodes within 16 cells of a particle.")
    lines.append("")

    for B in bins:
        meta = read_meta(prefix, B)
        B_used = int(meta["bins_used"][0])
        model_sum = float(meta["model_sum"][0])
        model_err = float(meta["model_error"][0])
        per_axis = meta["per_axis"]
        rep = np.array(meta["representative_u"], dtype=float).reshape(-1, 3)
        box_of = np.array(meta["box_of"], dtype=int)
        u_b = u_p.copy()
        u_b[box_of >= 0] = rep[box_of[box_of >= 0]] if len(rep) > 1 else rep[0]

        mesh = read_field(prefix, B_used, n)
        print(f"B={B_used}: mesh vs T ...", flush=True)
        stats_T = error_stats(mesh, T, dist, bands, band_idx)
        Tb = compute_reference(bunch, u_b, n, origin, cell, args.order, refine, args.workers)
        stats_bin = error_stats(Tb, T, dist, bands, band_idx)
        # discretization part: mesh vs Tb (needs Tb as "ref"; mesh error vs Tb)
        mesh_disp = mesh.copy()
        stats_disc = error_stats(mesh_disp, Tb, dist, bands, band_idx)
        outside = dist < 0
        roundoff = float(np.max(np.abs(mesh[:, outside])) if outside.any() else 0.0)

        run_result = {
            "bins_used": B_used, "per_axis": per_axis, "model_sum": model_sum,
            "model_error": model_err,
            "mesh_vs_T": stats_T, "Tb_vs_T": stats_bin, "mesh_vs_Tb": stats_disc,
            "outside_window_roundoff": roundoff,
        }
        if metrics_ctx is not None:
            run_result["metrics_v2"] = compute_metrics_v2(mesh, T, metrics_ctx)
        results["runs"][str(B)] = run_result

        lines.append(f"## B = {B_used} (requested {B}, per-axis {per_axis[0]}x{per_axis[1]}x{per_axis[2]}, "
                     f"modelSum {model_sum:.4e}, model error {model_err:.4e})")
        lines.append("")
        lines.append("| metric | Ex | Ey | Ez | Bx | By | Bz | worst |")
        lines.append("|---|---|---|---|---|---|---|---|")
        lines.append("| mesh-T rel | " + " | ".join(fmt(v) for v in stats_T["rel"])
                     + f" | {stats_T['worst_rel']:.3e} |")
        lines.append("| mesh-T abs | " + " | ".join(fmt(v) for v in stats_T["abs"])
                     + f" | {stats_T['worst_abs']:.3e} |")
        lines.append("| Tb-T rel (binning) | " + " | ".join(fmt(v) for v in stats_bin["rel"])
                     + f" | {stats_bin['worst_rel']:.3e} |")
        lines.append("| mesh-Tb rel (discr.) | " + " | ".join(fmt(v) for v in stats_disc["rel"])
                     + f" | {stats_disc['worst_rel']:.3e} |")
        lines.append("")
        lines.append("| band (cells) | n | worst rel mesh-T | worst abs mesh-T |")
        lines.append("|---|---|---|---|")
        for b in stats_T["bands"]:
            lines.append(f"| {b['band']} | {b['n']} | {fmt(b['worst_rel'])} | {fmt(b['worst_abs'])} |")
        lines.append("")
        lines.append(f"Dominant band: {stats_T['dominant_band']}; "
                     f"outside-window max|mesh| = {roundoff:.3e}.")
        lines.append("")
        if metrics_ctx is not None:
            mv = run_result["metrics_v2"]
            lines.append("| v2 region | nodes | worst rel | worst component |")
            lines.append("|---|---|---|---|")
            for region in ("in_bunch", "far_field"):
                r = mv[region]
                lines.append(f"| {region} | {r['n_nodes']} | {fmt(r['worst_rel'])} "
                             f"| {r['worst_component'] or '-'} |")
            lines.append("")
            lines.append("| v2 in-bunch | Ex | Ey | Ez | Bx | By | Bz |")
            lines.append("|---|---|---|---|---|---|---|")
            comp = mv["in_bunch"]["per_component"]
            lines.append("| max rel | " + " | ".join(
                fmt(comp[name]["max_rel"]) for name in COMPONENTS) + " |")
            lines.append("| p99 rel | " + " | ".join(
                fmt(comp[name]["p99_rel"]) for name in COMPONENTS) + " |")
            lines.append("| rms rel | " + " | ".join(
                fmt(comp[name]["rms_rel"]) for name in COMPONENTS) + " |")
            lines.append("")

    if args.fit:
        lines.append("## Calibration fit")
        lines.append("")
        lines.append("| B | mesh-T worst rel | Tb-T worst rel | modelSum | C (mesh-T) | C (Tb-T) |")
        lines.append("|---|---|---|---|---|---|")
        c_vals_T, c_vals_B = [], []
        for B in bins:
            r = results["runs"][str(B)]
            ms = r["model_sum"]
            if ms <= 0:
                continue
            cT = r["mesh_vs_T"]["worst_rel"] / ms
            cB = r["Tb_vs_T"]["worst_rel"] / ms
            c_vals_T.append(cT)
            c_vals_B.append(cB)
            lines.append(f"| {r['bins_used']} | {r['mesh_vs_T']['worst_rel']:.3e} | "
                         f"{r['Tb_vs_T']['worst_rel']:.3e} | {ms:.4e} | {cT:.3e} | {cB:.3e} |")
        if c_vals_T:
            lines.append("")
            lines.append(f"C median: mesh-T {np.median(c_vals_T):.3e}, "
                         f"Tb-T {np.median(c_vals_B):.3e}; "
                         f"geomean: mesh-T {math.exp(np.mean(np.log(np.maximum(c_vals_T, 1e-300)))):.3e}, "
                         f"Tb-T {math.exp(np.mean(np.log(np.maximum(c_vals_B, 1e-300)))):.3e}")
        lines.append("")

    with open(f"{prefix}_compare.md", "w") as f:
        f.write("\n".join(lines))
    with open(f"{prefix}_compare.json", "w") as f:
        json.dump(results, f, indent=1, default=float)
    print(f"wrote {prefix}_compare.md / .json")


if __name__ == "__main__":
    main()
