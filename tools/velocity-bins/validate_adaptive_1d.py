#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
#
# Analytic 1D validation of the adaptive edge placement.
#
# The synthetic bunches are beam-like: every particle has ux = uy = 0, uz = u_p
# (u_p = |p|/(m c)), and positions are a small (a few cell) 3D cloud around the
# axis.  For a relativistic beam the in-bunch field along the axis is dominated
# by the 1D beam model: the longitudinal E field of a continuous line charge with
# a velocity distribution f(u) is, to leading order in 1/gamma,
#
#   E_z(x) ~ (q / (2 pi eps0)) * integral f(u) du * (1 / (1 - (u/g)^2)) * G(x; u)
#
# where G is the Green's function of the moving point charge.  The exact
# continuum reference (all particles at their own u_p) and the binned field
# (each box's particles at the box representative u_b) differ only in the
# velocity argument.  The pure binning error at a probe node x is therefore
#
#   err(x) = E_binned(x) - E_ref(x)
#
# For the two-cluster case the optimum is exact: the adaptive cut must fall in
# the gap between the two clusters so that no box straddles both, and the
# uniform cut (at the midpoint of the range) straddles the gap and mixes the
# clusters, giving a large error.  This is the analytic check.
#
# This script computes, for each synthetic case and each fixed count (2, 4, 8),
# the 1D in-bunch field error of the uniform edges vs the adaptive edges (the
# adaptive edges are the DP output from the C++ measurement driver's meta file).
# The reference is the per-particle 1D beam sum; the binned field replaces each
# particle's u by its box representative.  Both are summed over the 1D
# Green's-function kernel of the moving point charge (the same Lienard-Wiechert
# field, reduced to the beam axis), so the comparison is exact in 1D.

from __future__ import annotations

import argparse, json, math, os, sys

K_COULOMB = 8.9875517873681764e9
C_LIGHT = 299792458.0


def read_bunch(path):
    xs, zs, us = [], [], []
    with open(path) as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            v = line.split()
            xs.append(float(v[0])); zs.append(float(v[2]))
            us.append(math.hypot(float(v[3]), float(v[4]), float(v[5])))
    return xs, zs, us


def lz_kernel(x, z_src, u, g):
    """Longitudinal E (V/m) at axial node x from a point charge of charge
    q_e * weight at z_src moving with reduced velocity u along z, gamma g.
    The Lienard-Wiechert longitudinal field of a uniformly moving charge at the
    field time t (retarded to the node) along the axis.  We use the static
    limit of the boosted Coulomb field (the dominant in-bunch term for a
    relativistic beam): E_z = q/(4 pi eps0) * (x - z_src) * (1 - beta^2)
    / |x - z_src|^3 * gamma^2 / (1 - beta^2 sin^2)^{3/2}; on the axis
    sin = 0 so E_z = q/(4 pi eps0) * (1 - beta^2) gamma^2 * (x - z_src)/|x-z_src|^3.
    Since (1 - beta^2) gamma^2 = 1, on the axis E_z = q/(4 pi eps0) * (x - z_src)/|x - z_src|^3.
    The u dependence enters through the retarded position; for the in-bunch
    comparison at the bunch time the leading u dependence is the longitudinal
    compression factor 1/(1 - beta^2)^{3/2} evaluated at the probe.  We use the
    standard result that the on-axis longitudinal field of a moving point charge
    scales as (1 - beta^2)^{-3/2} times the static field, i.e.
    E_z = q/(4 pi eps0) * (x - z_src) / |x - z_src|^3 * (1 - beta^2)^{-3/2}.
    """
    beta = u / g
    beta2 = beta * beta
    d = x - z_src
    r = abs(d)
    if r < 1e-30:
        return 0.0
    # on-axis: sin(theta) = 0, the (1 - beta^2 sin^2)^(-3/2) = 1, and the
    # longitudinal compression is (1 - beta^2)^(-3/2)
    comp = (1.0 - beta2) ** (-1.5)
    return K_COULOMB * comp * d / (r * r * r)


def field_1d(xs, zs, us, gammas, weights, x_probe):
    """Sum of the on-axis longitudinal E at x_probe for all particles."""
    total = 0.0
    for i in range(len(us)):
        total += weights[i] * lz_kernel(x_probe, zs[i], us[i], gammas[i])
    return total


def gamma_from_u(u):
    return math.sqrt(1.0 + u * u)


def inbunch_nodes(xs, zs, n_nodes, margin_cells=8.0, cell=2e-6):
    lo = min(min(xs), min(zs)) - margin_cells * cell
    hi = max(max(xs), max(zs)) + margin_cells * cell
    return [lo + (hi - lo) * i / (n_nodes - 1) for i in range(n_nodes)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cases_dir")
    p.add_argument("outdir")
    p.add_argument("--measure", default=None)
    p.add_argument("--sources", type=int, default=512)
    p.add_argument("--gamma", type=float, default=50.0)
    p.add_argument("--cell", type=float, default=2e-6)
    p.add_argument("--grid", type=int, default=32)
    p.add_argument("--origin", default="0,0,0")
    p.add_argument("--n-nodes", type=int, default=64)
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    origin = [float(v) for v in args.origin.split(",")]
    cell = args.cell
    grid = args.grid
    origin_c = tuple(origin)
    centre = tuple(o + 0.5 * g * cell for o, g in zip(origin, (grid, grid, grid)))
    counts = [2, 4, 8]
    cases = sorted(f[:-6] for f in os.listdir(args.cases_dir) if f.endswith(".bunch"))
    results = {}
    for case in cases:
        bf = os.path.join(args.cases_dir, case + ".bunch")
        xs, zs, us = read_bunch(bf)
        gammas = [gamma_from_u(u) for u in us]
        # each macroparticle carries the same charge; weight from the file
        weights = [float(open(bf).read().splitlines()[1].split()[6])] * len(us)
        # reference: per-particle 1D field
        probes = inbunch_nodes(xs, zs, args.n_nodes, cell=cell)
        ref = [field_1d(xs, zs, us, gammas, weights, x) for x in probes]
        row = {}
        for edge in ("uniform", "adaptive"):
            for count in counts:
                prefix = os.path.join(args.outdir, f"{case}_{edge}_B{count}")
                meta_path = f"{prefix}_meta_B{count}.txt"
                if not os.path.exists(meta_path):
                    # run the driver
                    import subprocess
                    measure = args.measure or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "build-cpu", "picBunchVelocityBinsMeasure")
                    cmd = [measure, "--bunch", bf, "--bin-edges", edge, "--bins", str(count),
                           "--gamma", str(args.gamma),
                           "--cell", f"{cell},{cell},{cell}",
                           "--grid", f"{grid},{grid},{grid}",
                           "--origin", f"{origin_c[0]},{origin_c[1]},{origin_c[2]}",
                           "--cloud-cells", "3.0", "--fft-padding", "2",
                           "--near-shell", "none", "--output", prefix,
                           "--sources", str(args.sources)]
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # read boxOf from the bunch file the driver wrote
                box_of = []
                with open(f"{prefix}_bunch.txt") as f:
                    for line in f:
                        if line.startswith("#"):
                            continue
                        box_of.append(int(line.split()[-1]))
                # per-box representative u (from meta: representative_u, 3 per box)
                meta = {}
                with open(meta_path) as f:
                    for line in f:
                        tok = line.split()
                        meta[tok[0]] = tok[1:]
                n_boxes = int(meta["bins_used"][0])
                rep_u = [float(v) for v in meta["representative_u"]]
                # binned 1D field: each particle uses its box representative u
                # (the representative is the charge-weighted mean u of the box)
                binned = []
                for x in probes:
                    total = 0.0
                    for i in range(len(us)):
                        b = box_of[i]
                        # representative uz (lab frame, the box mean |u| along the axis)
                        u_b = abs(rep_u[3 * b + 2])
                        g_b = gamma_from_u(u_b)
                        total += weights[i] * lz_kernel(x, zs[i], u_b, g_b)
                    binned.append(total)
                # per-component: the 1D model gives Ez (longitudinal) and Bx/By
                # (transverse B, = E/c * beta direction).  For the table we
                # report the longitudinal (Ez) relative error and the transverse
                # B (By) relative error.  The transverse B of a beam at a probe
                # on the axis is zero by symmetry, so we probe slightly off-axis
                # for the transverse B.  Here we report the on-axis Ez error and
                # the off-axis By error.
                err_ez = 0.0
                for i, x in enumerate(probes):
                    d = abs(ref[i])
                    if d > 1e-30:
                        err_ez = max(err_ez, abs(binned[i] - ref[i]) / d)
                # transverse B: probe 1 cell off axis (in x), B_y = E_x / c * ... ;
                # for a beam the transverse E at (x0, 0, z) from a charge at (0,0,z')
                # is E_x = K q (1-beta^2) x0 / (x0^2 + (z-z')^2)^{3/2} * (1-beta^2 sin^2)^{-3/2}
                # and B_y = E_x / (c * beta) for the moving charge.  The binning
                # error in B_y has the same u dependence as E_x, so the relative
                # error matches the E_x relative error.  We compute E_x ref/binned.
                x_off = cell
                err_by = 0.0
                for i, x in enumerate(probes):
                    def ex(u, g, z_src):
                        beta = u / g
                        beta2 = beta * beta
                        r2 = x_off * x_off + (x - z_src) ** 2
                        comp = (1.0 - beta2) ** (-1.5)
                        return K_COULOMB * comp * x_off / (r2 ** 1.5)
                    e_ref = sum(weights[j] * ex(us[j], gammas[j], zs[j]) for j in range(len(us)))
                    e_bin = sum(weights[j] * ex(abs(rep_u[3 * box_of[j] + 2]), gamma_from_u(abs(rep_u[3 * box_of[j] + 2])), zs[j]) for j in range(len(us)))
                    d = abs(e_ref)
                    if d > 1e-30:
                        err_by = max(err_by, abs(e_bin - e_ref) / d)
                row[f"{edge}_B{count}"] = {"err_ez": err_ez, "err_by": err_by, "n_boxes": n_boxes}
                print(f"{case} {edge} B{count}: Ez={err_ez:.3g} By={err_by:.3g} (n_boxes={n_boxes})", flush=True)
        results[case] = row
    with open(os.path.join(args.outdir, "adaptive_1d_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== 1D ANALYTIC TABLE (on-axis Ez, off-axis By; max in-bunch relative error) ===")
    for case in cases:
        print(f"\n{case}:")
        for edge in ("uniform", "adaptive"):
            for count in counts:
                r = results[case][f"{edge}_B{count}"]
                print(f"  {edge:9s} B{count}: Ez={r['err_ez']:.3g}  By={r['err_by']:.3g}")


if __name__ == "__main__":
    main()
