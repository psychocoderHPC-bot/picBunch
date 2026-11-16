#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
#
# Generate the small synthetic bunches for the adaptive edge validation
# (tools/velocity-bins/measure_velocity_bins.py --bunch).  Each bunch is a beam
# along +z with a non-uniform reduced-velocity u = |u| structure: two clusters,
# a core plus an exponential tail, a strong skew, a core plus a sparse halo, and
# a near-uniform control.  Positions are a small Gaussian cloud so the in-bunch
# field is well defined.  The file format matches the measurement driver
# (--bunch): ``x y z ux uy uz weight charge`` per line (weight = macroparticle
# charge weight, charge = q_e; the driver uses weight=1 for the generator).
#
# Usage:
#   gen_synthetic_bunches.py OUTDIR [--sources 512] [--gamma 50] [--seed 1]

from __future__ import annotations

import argparse
import math
import os
import random

K_ELECTRON_CHARGE = -1.602176634e-19


def u_of_gamma(g: float) -> float:
    return math.sqrt(max(0.0, g * g - 1.0))


def write_bunch(path: str, particles: list[tuple[float, float, float, float, float, float, float]]) -> None:
    with open(path, "w") as f:
        f.write("# x y z ux uy uz weight charge boxOf\n")
        for (x, y, z, ux, uy, uz, w) in particles:
            f.write(f"{x!r} {y!r} {z!r} {ux!r} {uy!r} {uz!r} {w!r} {K_ELECTRON_CHARGE!r} 0\n")


def beam_particles(rng: random.Random, n: int, gamma: float, gamma_fn, cloud_cells: float, cell: float, origin: tuple[float, float, float], grid: tuple[int, int, int], weight: float) -> list[tuple[float, float, float, float, float, float, float]]:
    cx = origin[0] + 0.5 * grid[0] * cell
    cy = origin[1] + 0.5 * grid[1] * cell
    cz = origin[2] + 0.5 * grid[2] * cell
    out = []
    for _ in range(n):
        x = cx + rng.gauss(0.0, cloud_cells * cell)
        y = cy + rng.gauss(0.0, cloud_cells * cell)
        z = cz + rng.gauss(0.0, cloud_cells * cell)
        g = gamma_fn(rng)
        u = u_of_gamma(g)
        out.append((x, y, z, 0.0, 0.0, u, weight))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("outdir")
    p.add_argument("--sources", type=int, default=512)
    p.add_argument("--gamma", type=float, default=50.0)
    p.add_argument("--cloud-cells", type=float, default=3.0)
    p.add_argument("--cell", type=float, default=1.0e-6)
    p.add_argument("--grid", type=int, default=48)
    p.add_argument("--origin", default="0,0,0")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    origin = tuple(float(v) for v in args.origin.split(","))
    grid = (args.grid, args.grid, args.grid)
    cell = args.cell
    w = float(args.sources)  # weight so each macroparticle carries the same charge

    # The u structure is defined in gamma units (u = |u|); gamma_fn returns a
    # per-particle gamma.  The beam axis is +z, so ux = uy = 0, uz = u.

    # 1. Two clusters in u.
    def two_cluster(r: random.Random) -> float:
        return args.gamma * (1.0 + 0.20 * (1.0 if r.random() < 0.5 else -1.0)) + r.gauss(0.0, args.gamma * 0.02)

    # 2. Core + exponential tail (in the reduced velocity / gamma).
    def exp_tail(r: random.Random) -> float:
        if r.random() < 0.85:
            return args.gamma * (1.0 + r.gauss(0.0, 0.02))
        # tail: gamma above the core, exponential falloff
        t = -math.log(1.0 - r.random())
        return args.gamma * (1.0 + 0.05 + 0.15 * t)

    # 3. Strong skew (a skewed distribution in gamma).
    def skewed(r: random.Random) -> float:
        # skewed exponential: most near the core, long upper tail
        u = -math.log(1.0 - r.random())
        return args.gamma * (1.0 + 0.01 + 0.10 * u)

    # 4. Core + sparse halo (most tight, a few far out).
    def core_halo(r: random.Random) -> float:
        if r.random() < 0.90:
            return args.gamma * (1.0 + r.gauss(0.0, 0.01))
        return args.gamma * (1.0 + 0.5 + 0.5 * r.random())

    # 5. Near-uniform (control): a wide, roughly flat gamma band.
    def uniform_band(r: random.Random) -> float:
        return args.gamma * (1.0 + r.uniform(-0.10, 0.10))

    cases = {
        "two_cluster": two_cluster,
        "exp_tail": exp_tail,
        "skewed": skewed,
        "core_halo": core_halo,
        "uniform": uniform_band,
    }
    for name, fn in cases.items():
        rng = random.Random(args.seed + (hash(name) % 1000))
        parts = beam_particles(rng, args.sources, args.gamma, fn, args.cloud_cells, cell, origin, grid, w)
        path = os.path.join(args.outdir, f"{name}.bunch")
        write_bunch(path, parts)
        us = sorted(u_of_gamma(fn(random.Random(args.seed + (hash(name) % 1000) + 7))) for _ in range(args.sources))
        print(f"{name}: n={args.sources} u in [{us[0]:.4g}, {us[-1]:.4g}] median={us[len(us)//2]:.4g}")


if __name__ == "__main__":
    main()
