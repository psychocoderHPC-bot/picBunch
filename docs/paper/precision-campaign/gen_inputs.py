#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
"""Deterministic openPMD 2.0 bunch inputs for the precision campaign.

Every input the precision harness needs is generated here from a fixed seed, so
a campaign rerun on another node starts from the same particles.  The layout
follows the campaign generators that produced the committed reduced data
(``gen_inputs.py`` in the reproduction run), with one correction: the
``.sources.bin`` sidecar stores positions in **computational cell units**, which
is what the independent reference driver reads (it multiplies the raw columns
by the cell size).  The openPMD HDF5 file keeps SI metres.

Each bunch is written as ``<name>.h5`` (particles) and ``<name>.sources.bin``
(``float64[N,7]``: x y z ux uy uz weight, positions in cells).  Momentum is
``u = p/(m_e c)`` stored with ``unitSI = m_e c``.

Usage::

    gen_inputs.py --out-dir DIR [--only name1,name2]

The ``mvpanel-e5``/``mvpanel-a5`` inputs additionally emit the 512-particle
``multivel_energy.npz``/``multivel_angular.npz`` particle panels.
"""
import argparse
import os

import h5py
import numpy as np

M_E = 9.1093837015e-31
C = 299792458.0
Q_E = 1.602176634e-19
DIM_POS = np.array([1.0, 0, 0, 0, 0, 0, 0])
DIM_MOM = np.array([1.0, 1.0, -1.0, 0, 0, 0, 0])
DIM_SCALAR = np.zeros(7)
DIM_CHARGE = np.array([0.0, 0.0, 1.0, 1.0, 0, 0, 0])
DIM_MASS = np.array([0.0, 1.0, 0, 0, 0, 0, 0])
WEIGHT_PAPER = 3.12075454e6  # macro weight of the -100 pC, N=200 multivel bunch


def series_attrs(handle):
    handle.attrs["openPMD"] = "2.0.0"
    handle.attrs["openPMDextension"] = np.uint32(0)
    handle.attrs["basePath"] = "/data/%T/"
    handle.attrs["meshesPath"] = "meshes/"
    handle.attrs["particlesPath"] = "particles/"
    handle.attrs["iterationEncoding"] = "groupBased"
    handle.attrs["iterationFormat"] = "/data/%T/"
    handle.attrs["software"] = "picbunch-precision-campaign-gen"
    handle.attrs["softwareVersion"] = "1.0"


def _vec(species, name, comps, unit_si, dim):
    record = species.create_group(name)
    record.attrs["unitDimension"] = dim
    record.attrs["timeOffset"] = 0.0
    for comp_name, values in comps:
        dataset = record.create_dataset(comp_name, data=np.asarray(values, dtype=np.float64))
        dataset.attrs["unitSI"] = float(unit_si)
        dataset.attrs["unitDimension"] = dim
        dataset.attrs["timeOffset"] = 0.0


def _scalar(species, name, values, unit_si, dim):
    record = species.create_group(name)
    record.attrs["unitSI"] = float(unit_si)
    record.attrs["unitDimension"] = dim
    record.attrs["timeOffset"] = 0.0
    dataset = record.create_dataset("value", data=np.asarray(values, dtype=np.float64))
    dataset.attrs["unitSI"] = float(unit_si)
    dataset.attrs["unitDimension"] = dim
    dataset.attrs["timeOffset"] = 0.0


def write_h5(path, pos_m, u, weight, charge=-Q_E, mass=M_E):
    """Write one openPMD 2.0 particle species ``e``.

    ``pos_m`` is ``(N,3)`` metres, ``u`` is ``(N,3)`` dimensionless and
    ``weight`` is the per-macro-particle weighting.
    """
    count = pos_m.shape[0]
    weight = np.broadcast_to(np.asarray(weight, dtype=np.float64), (count,))
    with h5py.File(path, "w") as handle:
        series_attrs(handle)
        iteration = handle.create_group("data/0")
        iteration.attrs["time"] = 0.0
        iteration.attrs["dt"] = 0.0
        iteration.attrs["timeUnitSI"] = 1.0
        species = iteration.create_group("particles/e")
        species.attrs["speciesType"] = np.uint32(1)
        _vec(species, "position",
             [("x", pos_m[:, 0]), ("y", pos_m[:, 1]), ("z", pos_m[:, 2])],
             1.0, DIM_POS)
        _vec(species, "momentum",
             [("x", u[:, 0]), ("y", u[:, 1]), ("z", u[:, 2])],
             M_E * C, DIM_MOM)
        _scalar(species, "weighting", weight, 1.0, DIM_SCALAR)
        # openPMD stores value*unitSI; -1 with unitSI=Q_E is one electron.
        _scalar(species, "charge", np.full(count, charge / Q_E), Q_E, DIM_CHARGE)
        _scalar(species, "mass", np.full(count, mass / M_E), M_E, DIM_MASS)


def write_sources(path, pos_m, u, weight, cell_m):
    """Write the reference-driver sidecar: positions in cell units."""
    cell = np.asarray(cell_m, dtype=np.float64)
    if cell.ndim == 0:
        cell = np.full(3, float(cell))
    arr = np.column_stack([pos_m / cell[None, :], u,
                           np.broadcast_to(np.asarray(weight, dtype=np.float64),
                                           (pos_m.shape[0],))])
    arr.astype(np.float64).tofile(path)


def gamma_to_u(gamma):
    return float(np.sqrt(max(1e-30, gamma * gamma - 1.0)))


def _unit(axis):
    vec = np.zeros(3)
    vec["xyz".index(axis)] = 1.0
    return vec


def _emit(outdir, name, pos_m, u, weight, cell_m, panel=None):
    h5 = os.path.join(outdir, name + ".h5")
    bin_ = os.path.join(outdir, name + ".sources.bin")
    write_h5(h5, pos_m, u, weight)
    write_sources(bin_, pos_m, u, weight, cell_m)
    if panel is not None:
        np.savez(os.path.join(outdir, panel + ".npz"),
                 particles=np.column_stack([pos_m, u,
                                            np.broadcast_to(np.asarray(weight, float),
                                                            (pos_m.shape[0],)),
                                            np.full(pos_m.shape[0], -Q_E)]))
    print("wrote", name, flush=True)


def gen_single(outdir, name, gamma, centre_cells, cell_m, axis):
    pos = np.asarray(centre_cells, dtype=float)[None, :] * cell_m
    u = np.zeros((1, 3))
    u[0] = gamma_to_u(gamma) * _unit(axis)
    _emit(outdir, name, pos, u, 1.0, cell_m)


def stratified_line(count, centre_cells, sigma_cells):
    """Stratified normal sample along the cube diagonal ``(t, t, t)``.

    The archived paper bunches of the committed multivel fields and of the
    bdata2 forward/fluctuation slices are a line along the diagonal, not a
    three-dimensional cloud: their per-axis width is ``sigma_cells`` but all
    three coordinates of a particle carry the same value.  A deterministic
    quantile sample keeps the per-axis width without the realisation noise of
    a 200-particle draw.  The bdata2 and multivel field sources must use this
    geometry; the var1b and large1 sources are genuine three-dimensional
    clouds and use :func:`gen_cloud` with ``line_cells=None``.
    """
    from statistics import NormalDist
    normal = NormalDist()
    centre = float(np.asarray(centre_cells, dtype=float)[0])
    t = np.asarray([centre + float(sigma_cells)
                    * normal.inv_cdf((i + 0.5) / count) for i in range(count)])
    return np.repeat(t[:, None], 3, axis=1)


def gen_cloud(outdir, name, gamma, count, seed, centre_cells, sigma_cells,
              cell_m, axis="y", divergence_rad=0.0, weight=1.0,
              clip_cells=None, line_cells=None):
    rng = np.random.default_rng(seed)
    centre = np.asarray(centre_cells, dtype=float) * cell_m
    if line_cells is not None:
        pos = stratified_line(count, centre_cells, line_cells) * cell_m
    else:
        pos = centre[None, :] + sigma_cells * cell_m * rng.standard_normal((count, 3))
    if clip_cells is not None:
        # The current reader rejects particles outside the output grid.  A
        # validation bunch whose 3-sigma tail crosses the grid boundary is
        # therefore confined to half a cell inside it.
        lo = 0.5 * cell_m
        hi = (np.asarray(clip_cells, dtype=float) - 0.5) * cell_m
        pos = np.clip(pos, lo, hi)
    u_abs = gamma_to_u(gamma)
    u = np.zeros((count, 3))
    u[:, "xyz".index(axis)] = u_abs
    if divergence_rad > 0.0:
        # transverse x spread, matching the momentum-spread validation case
        u[:, 0] = rng.normal(0.0, divergence_rad * u_abs, count)
    _emit(outdir, name, pos, u, weight, cell_m)


def gen_multivel(outdir, name, case, seed, count=200, cell_m=1e-7,
                 panel=False):
    """The E5% / A5deg multivel bunch used by F_multivel_dist.

    Two artefacts are generated from the same velocity law:

    ``panel=False`` (field run, ``mv-e5``/``mv-a5``)
        The source of the committed field peaks.  The archived paper bunch
        (``mvpaper_{e5,a5}.h5``, N=200, macro weight 5e-13 C, +y drift) walks
        a stratified normal along the cube diagonal: every particle sits at
        ``(t, t, t)`` with ``t`` a 200-point stratified sample of
        ``N(32, 6)`` cells.  The per-axis width is the paper's sigma=6 cells
        and the centroid is (32, 32, 32); the diagonal correlation of the
        archived input is what makes the committed peaks reproducible (a
        three-dimensional ``N(32, 6)`` cloud of the same per-axis width gives
        about 5.5 kT instead of 24.7 kT because the line concentrates the
        charge).  E5 spreads the parallel momentum by 5 %; A5 draws a 5 deg
        Rayleigh cone, normalised so ``|u|`` is constant.

    ``panel=True`` (``mvpanel-e5``/``mvpanel-a5``)
        The 512-source figure sample: a three-dimensional Gaussian cloud of
        sigma 3e-6 m at 2.4e-5 m (the committed panel's ~18-cell cloud at
        24 cells in 1e-6 m units), parallel momentum along +z, unit weight.
        The renderers read ``u`` only, but positions and weight are kept
        identical to the committed ``multivel_{energy,angular}.npz``.
    """
    rng = np.random.default_rng(seed)
    u_abs = gamma_to_u(10.0)
    if panel:
        pos = (2.4e-5 + 3.0e-6 * rng.standard_normal((count, 3)))
        if case == "e5":
            u_parallel = u_abs * (1.0 + 0.05 * rng.standard_normal(count))
            u = np.column_stack([np.zeros(count), np.zeros(count), u_parallel])
        else:
            sig = np.tan(np.deg2rad(5.0))
            gx = rng.normal(0.0, sig, count)
            gy = rng.normal(0.0, sig, count)
            norm = np.sqrt(1.0 + gx * gx + gy * gy)
            u = np.column_stack([u_abs * gx / norm,
                                 u_abs * gy / norm,
                                 u_abs / norm])
        _emit(outdir, name, pos, u, 1.0, cell_m, panel=name)
        return

    pos = stratified_line(count, [32.0, 32.0, 32.0], 6.0) * cell_m
    if case == "e5":
        u_parallel = u_abs * (1.0 + 0.05 * rng.standard_normal(count))
        u = np.column_stack([np.zeros(count), u_parallel, np.zeros(count)])
    else:
        sig = np.tan(np.deg2rad(5.0))
        gx = rng.normal(0.0, sig, count)
        gz = rng.normal(0.0, sig, count)
        norm = np.sqrt(1.0 + gx * gx + gz * gz)
        u = np.column_stack([u_abs * gx / norm,
                             u_abs / norm,
                             u_abs * gz / norm])
    _emit(outdir, name, pos, u, WEIGHT_PAPER, cell_m)


CASES = {}


def _register(name, func):
    CASES[name] = func


def _build_registry():
    for gamma, tag in ((1, "1"), (10, "10"), (50, "50"), (100, "100")):
        gtag = "%g" % gamma
        _register("w4b-g" + tag, lambda o, g=gamma, n="w4b-g" + gtag: gen_single(
            o, n, g, [48, 48, 48], 1e-7, "z"))
        _register("w4b-g" + tag + "-a4",
                  lambda o, g=gamma, n="w4b-g" + gtag + "-a4": gen_single(
                      o, n, g, [48, 48, 192], 1e-7, "z"))
        _register("gamma-g" + tag + "-h1",
                  lambda o, g=gamma, n="gamma-g" + gtag + "-h1": gen_single(
                      o, n, g, [48, 48, 48], 1e-7, "z"))
        _register("gamma-g" + tag + "-h25",
                  lambda o, g=gamma, n="gamma-g" + gtag + "-h25": gen_single(
                      o, n, g, [192, 192, 192], 2.5e-8, "z"))

    for tag, gamma in (("1p0", 1.0), ("1p1", 1.1), ("1p5", 1.5),
                       ("2p0", 2.0), ("10", 10.0), ("50", 50.0), ("100", 100.0)):
        _register("hires-g" + tag, lambda o, g=gamma, t=tag: gen_single(
            o, "hires-g" + t, g, [64, 64, 64], 5e-8, "y"))

    _register("bdata2-forward", lambda o: gen_cloud(
        o, "bdata2-forward", 10.0, 200, 42, [32, 32, 32], 6.0, 1e-7,
        axis="y", weight=WEIGHT_PAPER, line_cells=6.0))
    _register("bdata2-fluctuation", lambda o: gen_cloud(
        o, "bdata2-fluctuation", 10.0, 200, 42, [32, 32, 32], 6.0, 1e-7,
        axis="y", divergence_rad=1.046e-3, weight=WEIGHT_PAPER,
        line_cells=6.0))
    _register("var1b-1mrad", lambda o: gen_cloud(
        o, "var1b-1mrad", 10.0, 200, 42, [16, 16, 8], 3.0, 2e-7,
        axis="y", divergence_rad=1.046e-3, weight=WEIGHT_PAPER,
        clip_cells=[32, 32, 16]))

    _register("large-g1", lambda o: gen_cloud(
        o, "large-g1", 1.0, 100, 42, [64, 64, 32], 6.0, 2e-7, axis="x"))
    _register("large-g10", lambda o: gen_cloud(
        o, "large-g10", 10.0, 100, 42, [64, 64, 32], 6.0, 2e-7, axis="x"))

    _register("mv-e5", lambda o: gen_multivel(o, "mv-e5", "e5", 100))
    _register("mv-a5", lambda o: gen_multivel(o, "mv-a5", "a5", 103))
    # 512-particle particle panels for F_multivel_dist (a)/(b)
    _register("mvpanel-e5", lambda o: gen_multivel(o, "mvpanel-e5", "e5",
                                                   100, count=512, panel=True))
    _register("mvpanel-a5", lambda o: gen_multivel(o, "mvpanel-a5", "a5",
                                                   103, count=512, panel=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--only", default="",
                        help="comma-separated subset of registered input names")
    args = parser.parse_args()
    _build_registry()
    os.makedirs(args.out_dir, exist_ok=True)
    wanted = set(args.only.split(",")) if args.only else set(CASES)
    unknown = wanted - set(CASES)
    if unknown:
        raise SystemExit("unknown input(s): %s" % ", ".join(sorted(unknown)))
    for name in sorted(wanted):
        CASES[name](args.out_dir)


if __name__ == "__main__":
    main()
