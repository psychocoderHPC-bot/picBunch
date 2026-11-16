#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
#
# Synthetic bunches for the adaptive velocity-bin validation.
#
# Spec mode (default) generates the six fixed cases of ``nextTests.md``
# section 3, with the exact seeds, counts and velocity structure written there:
#
#   uniform         3.1  seed 42  1000   momentum uniform in [p0-dp, p0+dp]^3
#   two_cluster     3.2  seed 43  1000   500 @ 0.95c (0.005c) + 500 @ 0.99c (0.003c)
#   core_halo       3.3  seed 44  2000   1500 core 0.98c/0.002c + 500 halo 0.98c/0.03c
#   exp_tail        3.4  seed 45  2000   1500 core 0.98c/0.002c + 500 exp tail
#   skewed          3.5  seed 46  2000   log-normal in v, median 0.97c, sigma_log 0.05
#   single_velocity 3.6  seed 47  1000   every particle at p0 = 1e9 MeV/c
#
# Positions are uniform in a cube of side 2R (R = 1e-6 m) centred at
# ``--center`` (default: the centre of the measurement driver's default
# 48^3-cell, 1 um-cell, origin-0 domain, i.e. 24 um).
#
# Unit convention.  The spec gives momenta in MeV/c; the field code reads a
# ``--bunch`` file as dimensionless u = p / (m0 c) with no conversion
# (measure_velocity_bins.cpp reads the third-to-sixth columns straight into
# bunch.ux/uy/uz).  With c = 1 the numerical value of m0 c in MeV/c equals the
# rest energy m0 c^2 in MeV, so
#
#     u = p_MeVc / mass_mev          (per component; mass_mev = m0 c^2 in MeV)
#
# and |v|/c = |u| / sqrt(1 + |u|^2).  ``--mass-mev`` defaults to the electron
# rest energy 0.51099895 MeV.
#
# ``--bunch`` file layout (8 whitespace-separated columns, one header line):
#
#     # x y z ux uy uz weight charge
#
# This is the layout accepted by both readers: measure_velocity_bins.cpp skips
# ``#`` lines and reads the first eight columns; load_fixture.cpp consumes the
# header with getline() and then reads exactly eight tokens per particle, so a
# ninth ``boxOf`` column (as the legacy generator writes) would desynchronise
# it.  boxOf is optional for the field driver and is therefore omitted here.
#
# Each case also writes ``<case>.meta.json``: the requested distribution
# parameters, the analytic moments of the *implemented* distribution (after the
# physical v < c truncation), and the measured <|v|/c>, std, min and max per
# group.  ``--verify`` regenerates in memory and checks those numbers.
#
# Legacy mode (still supported, deprecated):
#   gen_synthetic_bunches.py OUTDIR [--sources 512] [--gamma 50] [--seed 1]
# generates the five old gamma-based cases.  Passing a positional OUTDIR or any
# of the legacy options selects it.
#
# Spec-mode usage:
#   gen_synthetic_bunches.py [--out DIR] [--case NAME ...] [--mass-mev M]
#                            [--center X,Y,Z] [--dry-run] [--verify]

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from statistics import NormalDist

K_ELECTRON_CHARGE = -1.602176634e-19
#: Electron rest energy m0 c^2 in MeV.  With c = 1 this is also m0 c in
#: MeV/c, so u = p_MeVc / M_ELECTRON_C2_MEV.
M_ELECTRON_C2_MEV = 0.51099895

R_BUNCH_M = 1.0e-6
#: Cube centre.  measure_velocity_bins.cpp defaults to origin 0, grid 48 and
#: cell 1e-6 m, so the domain centre is 24 um.  Centring the 2R cube there
#: keeps every particle inside the default run geometry.
DEFAULT_CENTER_M = (24.0e-6, 24.0e-6, 24.0e-6)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "data", "synthetic_bunches")

BETA_MAX = 1.0
#: Statistical tolerance of ``--verify`` in standard errors.  The mean uses a
#: 2-sigma gate.  The sample std of a fixed seeded draw is itself noisy (its
#: standard error is sigma/sqrt(2(n-1))), so the spread uses a 4-sigma gate;
#: that still rejects any real parameter error (a 10 percent sigma change is
#: about 20 standard errors here).
N_SIGMA = 2.0
SPREAD_N_SIGMA = 4.0
#: Measured spreads use the population standard deviation (ddof = 0) so they
#: can be compared directly with the distribution sigma.
STD_DDOF = 0

_NORMAL = NormalDist()


@dataclass(frozen=True)
class Group:
    name: str
    count: int
    #: ``momentum_cube``, ``gaussian_v``, ``exponential_v``, ``lognormal_v``,
    #: ``delta_p``.
    kind: str
    params: dict


@dataclass(frozen=True)
class SpecCase:
    name: str
    section: str
    seed: int
    groups: tuple
    note: str = ""

    @property
    def count(self) -> int:
        return sum(g.count for g in self.groups)


# NOTE: nextTests.md section 3 states "total particles across all bunches stay
# under 5000", but the per-case counts listed there add up to 9000
# (3 x 1000 + 3 x 2000).  The counts are acceptance-relevant, so they are
# reproduced exactly and the contradiction is reported rather than resolved by
# silently shrinking a bunch.
CASES = (
    SpecCase(
        "uniform",
        "3.1",
        42,
        (
            Group(
                "all",
                1000,
                "momentum_cube",
                {"p0_mev_c": 1.0e9, "dp_mev_c": 1.0e8},
            ),
        ),
    ),
    SpecCase(
        "two_cluster",
        "3.2",
        43,
        (
            Group("cluster_a", 500, "gaussian_v", {"mean_over_c": 0.95, "sigma_over_c": 0.005}),
            Group("cluster_b", 500, "gaussian_v", {"mean_over_c": 0.99, "sigma_over_c": 0.003}),
        ),
    ),
    SpecCase(
        "core_halo",
        "3.3",
        44,
        (
            Group("core", 1500, "gaussian_v", {"mean_over_c": 0.98, "sigma_over_c": 0.002}),
            Group("halo", 500, "gaussian_v", {"mean_over_c": 0.98, "sigma_over_c": 0.03}),
        ),
    ),
    SpecCase(
        "exp_tail",
        "3.4",
        45,
        (
            Group("core", 1500, "gaussian_v", {"mean_over_c": 0.98, "sigma_over_c": 0.002}),
            Group(
                "tail",
                500,
                "exponential_v",
                {"v0_over_c": 0.98, "L_over_c": 0.01, "vmax_over_c": 0.999},
            ),
        ),
    ),
    SpecCase(
        "skewed",
        "3.5",
        46,
        (Group("all", 2000, "lognormal_v", {"median_over_c": 0.97, "sigma_log": 0.05}),),
    ),
    SpecCase(
        "single_velocity",
        "3.6",
        47,
        (Group("all", 1000, "delta_p", {"p0_mev_c": 1.0e9}),),
    ),
)
CASES_BY_NAME = {case.name: case for case in CASES}

# Independent literals from nextTests.md section 3.  ``--verify`` compares the
# CASES table against these so that an accidental edit to a seed, count, mean
# or spread fails loudly instead of being validated against itself.
SPEC_EXPECTED = {
    "uniform": (
        "3.1",
        42,
        (("all", 1000, "momentum_cube", {"p0_mev_c": 1.0e9, "dp_mev_c": 1.0e8}),),
    ),
    "two_cluster": (
        "3.2",
        43,
        (
            ("cluster_a", 500, "gaussian_v", {"mean_over_c": 0.95, "sigma_over_c": 0.005}),
            ("cluster_b", 500, "gaussian_v", {"mean_over_c": 0.99, "sigma_over_c": 0.003}),
        ),
    ),
    "core_halo": (
        "3.3",
        44,
        (
            ("core", 1500, "gaussian_v", {"mean_over_c": 0.98, "sigma_over_c": 0.002}),
            ("halo", 500, "gaussian_v", {"mean_over_c": 0.98, "sigma_over_c": 0.03}),
        ),
    ),
    "exp_tail": (
        "3.4",
        45,
        (
            ("core", 1500, "gaussian_v", {"mean_over_c": 0.98, "sigma_over_c": 0.002}),
            (
                "tail",
                500,
                "exponential_v",
                {"v0_over_c": 0.98, "L_over_c": 0.01, "vmax_over_c": 0.999},
            ),
        ),
    ),
    "skewed": (
        "3.5",
        46,
        (("all", 2000, "lognormal_v", {"median_over_c": 0.97, "sigma_log": 0.05}),),
    ),
    "single_velocity": (
        "3.6",
        47,
        (("all", 1000, "delta_p", {"p0_mev_c": 1.0e9}),),
    ),
}


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------


def _std_normal(rng: random.Random) -> float:
    """One standard-normal draw; deterministic for a fixed CPython/libm."""
    u = rng.random()
    while u <= 0.0:  # inv_cdf(0) is -inf; the probability of drawing 0 is 2^-53
        u = rng.random()
    return _NORMAL.inv_cdf(u)


def _u_of_beta(beta: float) -> float:
    """Dimensionless u = gamma*beta for a speed beta = |v|/c."""
    return beta / math.sqrt(1.0 - beta * beta)


def _beta_of_u(ux: float, uy: float, uz: float) -> float:
    """|v|/c from the code's u vector: beta = |u| / sqrt(1 + |u|^2)."""
    u = math.hypot(ux, uy, uz)
    return u / math.sqrt(1.0 + u * u)


def _truncated_gaussian_beta(rng: random.Random, mean: float, sigma: float) -> float:
    """Gaussian in v truncated to the physical domain 0 < v < c.

    The spec's halo (0.98c +- 0.03c) and log-normal (0.97c, sigma_log 0.05)
    put a sizable probability above c, which is unphysical and would make
    u imaginary.  Rejection sampling keeps the requested shape and records the
    resulting analytic moments in the meta file.
    """
    while True:
        value = mean + sigma * _std_normal(rng)
        if 0.0 < value < BETA_MAX:
            return value


def _truncated_exponential_beta(
    rng: random.Random, v0: float, length: float, vmax: float
) -> float:
    """Inverse-CDF draw of f(v) ~ exp(-(v-v0)/L) on [v0, vmax]."""
    tail = (vmax - v0) / length
    u = rng.random()
    return v0 - length * math.log1p(-u * (1.0 - math.exp(-tail)))


def _truncated_lognormal_beta(rng: random.Random, median: float, sigma_log: float) -> float:
    """Log-normal in v truncated to 0 < v < c (rejection sampling)."""
    while True:
        value = median * math.exp(sigma_log * _std_normal(rng))
        if 0.0 < value < BETA_MAX:
            return value


def _sample_velocity(rng: random.Random, group: Group, mass_mev: float) -> tuple:
    """Return (ux, uy, uz) for one particle of ``group``.

    Except for the momentum-cube control, the spec fixes only |v|; its velocity
    is put on the +z beam axis, which is the driver's documented convention
    ("Beam axis: +z") and what validate_adaptive_1d.py assumes.
    """
    if group.kind == "momentum_cube":
        p0 = group.params["p0_mev_c"]
        dp = group.params["dp_mev_c"]
        # p is uniform per component in [p0-dp, p0+dp] MeV/c; u = p/(m0 c).
        ux = rng.uniform(p0 - dp, p0 + dp) / mass_mev
        uy = rng.uniform(p0 - dp, p0 + dp) / mass_mev
        uz = rng.uniform(p0 - dp, p0 + dp) / mass_mev
        return ux, uy, uz
    if group.kind == "gaussian_v":
        beta = _truncated_gaussian_beta(
            rng, group.params["mean_over_c"], group.params["sigma_over_c"]
        )
    elif group.kind == "exponential_v":
        beta = _truncated_exponential_beta(
            rng,
            group.params["v0_over_c"],
            group.params["L_over_c"],
            group.params["vmax_over_c"],
        )
    elif group.kind == "lognormal_v":
        beta = _truncated_lognormal_beta(
            rng, group.params["median_over_c"], group.params["sigma_log"]
        )
    elif group.kind == "delta_p":
        # Same momentum vector for every particle: p0 along +z.
        return 0.0, 0.0, group.params["p0_mev_c"] / mass_mev
    else:
        raise ValueError(f"unknown group kind {group.kind!r}")
    return 0.0, 0.0, _u_of_beta(beta)


def _sample_case(case: SpecCase, mass_mev: float, center: tuple) -> list:
    """Draw one case.

    The sequence is fixed: shuffle the exact group labels, then for each
    particle draw x, y, z and its velocity.  Same seed + same code therefore
    produces byte-identical output.
    """
    rng = random.Random(case.seed)
    labels = [gi for gi, group in enumerate(case.groups) for _ in range(group.count)]
    rng.shuffle(labels)
    particles = []
    for gi in labels:
        x = center[0] + rng.uniform(-R_BUNCH_M, R_BUNCH_M)
        y = center[1] + rng.uniform(-R_BUNCH_M, R_BUNCH_M)
        z = center[2] + rng.uniform(-R_BUNCH_M, R_BUNCH_M)
        ux, uy, uz = _sample_velocity(rng, case.groups[gi], mass_mev)
        particles.append((x, y, z, ux, uy, uz, gi))
    return particles


# --------------------------------------------------------------------------
# analytic moments of the implemented distributions
# --------------------------------------------------------------------------


def _phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _truncated_normal_moments(mu: float, sigma: float, lo: float, hi: float) -> tuple:
    a = (lo - mu) / sigma
    b = (hi - mu) / sigma
    pa, pb = _phi(a), _phi(b)
    z = _Phi(b) - _Phi(a)
    d = (pa - pb) / z
    mean = mu + sigma * d
    var = sigma * sigma * (1.0 + (a * pa - b * pb) / z - d * d)
    return mean, math.sqrt(max(var, 0.0))


def _truncated_exponential_moments(v0: float, length: float, vmax: float) -> tuple:
    t = (vmax - v0) / length
    e = math.exp(-t)
    denom = 1.0 - e
    m1 = length * (1.0 - (t + 1.0) * e) / denom
    m2 = 2.0 * length * length * (1.0 - e * (1.0 + t + 0.5 * t * t)) / denom
    return v0 + m1, math.sqrt(max(m2 - m1 * m1, 0.0))


def _truncated_lognormal_moments(median: float, sigma_log: float, hi: float) -> tuple:
    a = (math.log(hi) - math.log(median)) / sigma_log
    z = _Phi(a)
    m1 = median * math.exp(0.5 * sigma_log * sigma_log) * _Phi(a - sigma_log) / z
    m2 = median * median * math.exp(2.0 * sigma_log * sigma_log) * _Phi(a - 2.0 * sigma_log) / z
    return m1, math.sqrt(max(m2 - m1 * m1, 0.0))


def _analytic_beta_moments(group: Group, mass_mev: float):
    """Analytic <|v|/c> and std of one group after truncation.

    Returns None for the momentum-cube control, whose |v| has no closed form;
    ``--verify`` checks that case per momentum component instead.
    """
    if group.kind == "gaussian_v":
        return _truncated_normal_moments(
            group.params["mean_over_c"], group.params["sigma_over_c"], 0.0, BETA_MAX
        )
    if group.kind == "exponential_v":
        return _truncated_exponential_moments(
            group.params["v0_over_c"],
            group.params["L_over_c"],
            group.params["vmax_over_c"],
        )
    if group.kind == "lognormal_v":
        return _truncated_lognormal_moments(
            group.params["median_over_c"], group.params["sigma_log"], BETA_MAX
        )
    if group.kind == "delta_p":
        beta = _beta_of_u(0.0, 0.0, group.params["p0_mev_c"] / mass_mev)
        return beta, 0.0
    return None


# --------------------------------------------------------------------------
# statistics, meta record and files
# --------------------------------------------------------------------------


def _moments(values: list) -> tuple:
    n = len(values)
    mean = math.fsum(values) / n
    var = math.fsum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(max(var, 0.0)), min(values), max(values)


def _bunch_text(particles: list) -> str:
    lines = ["# x y z ux uy uz weight charge"]
    for x, y, z, ux, uy, uz, _gi in particles:
        lines.append(
            f"{x!r} {y!r} {z!r} {ux!r} {uy!r} {uz!r} 1.0 {K_ELECTRON_CHARGE!r}"
        )
    return "\n".join(lines) + "\n"


def _group_record(
    group: Group, particles: list, gi: int, mass_mev: float
) -> dict:
    selected = [p for p in particles if p[6] == gi]
    betas = [_beta_of_u(p[3], p[4], p[5]) for p in selected]
    mean, std, lo, hi = _moments(betas)
    uabs = [math.hypot(p[3], p[4], p[5]) for p in selected]
    u_mean, u_std, u_lo, u_hi = _moments(uabs)
    analytic = _analytic_beta_moments(group, mass_mev)
    record = {
        "name": group.name,
        "count": len(betas),
        "kind": group.kind,
        "requested": dict(group.params),
        "measured": {
            "mean_over_c": mean,
            "std_over_c": std,
            "min_over_c": lo,
            "max_over_c": hi,
        },
        # |v|/c saturates at 1.0 in double precision when |u| ~ 1e9 (the
        # p0 = 1e9 MeV/c control and the single-velocity case), so the
        # dimensionless |u| statistics carry the resolvable spread there.
        "measured_u": {
            "abs_mean": u_mean,
            "abs_std": u_std,
            "abs_min": u_lo,
            "abs_max": u_hi,
        },
    }
    if analytic is not None:
        record["analytic_after_truncation"] = {
            "mean_over_c": analytic[0],
            "std_over_c": analytic[1],
        }
    return record


def _meta_record(
    case: SpecCase, mass_mev: float, center: tuple, particles: list, bunch_text: str
) -> dict:
    betas = [_beta_of_u(p[3], p[4], p[5]) for p in particles]
    mean, std, lo, hi = _moments(betas)
    uabs = [math.hypot(p[3], p[4], p[5]) for p in particles]
    u_mean, u_std, u_lo, u_hi = _moments(uabs)
    return {
        "schema": "picbunch-velocity-bin-spec-bunch/1",
        "case": case.name,
        "spec_section": case.section,
        "seed": case.seed,
        "particle_count": len(particles),
        "groups": [
            _group_record(group, particles, gi, mass_mev)
            for gi, group in enumerate(case.groups)
        ],
        "measured": {
            "mean_over_c": mean,
            "std_over_c": std,
            "min_over_c": lo,
            "max_over_c": hi,
            "std_ddof": STD_DDOF,
        },
        "measured_u": {
            "abs_mean": u_mean,
            "abs_std": u_std,
            "abs_min": u_lo,
            "abs_max": u_hi,
        },
        "geometry": {
            "shape": "cube",
            "centre_m": list(center),
            "side_m": 2.0 * R_BUNCH_M,
            "radius_m": R_BUNCH_M,
        },
        "units": {
            "position": "m",
            "bunch_momentum": "u = p/(m0 c), dimensionless",
            "spec_momentum": "MeV/c; u = p_MeVc / mass_mev",
            "mass_mev_c2": mass_mev,
            "velocity": "|v|/c = |u| / sqrt(1 + |u|^2)",
        },
        "truncation": (
            "Gaussian and log-normal velocity draws are rejection-truncated to "
            "0 < |v| < c; analytic_after_truncation is the exact moment of that "
            "truncated law, measured is the sample value."
        ),
        "generator": "tools/velocity-bins/gen_synthetic_bunches.py",
        "bunch_file": f"{case.name}.bunch",
        "bunch_sha256": hashlib.sha256(bunch_text.encode("utf-8")).hexdigest(),
    }


def _meta_text(meta: dict) -> str:
    return json.dumps(meta, indent=2, sort_keys=True) + "\n"


def _write_case(case: SpecCase, outdir: str, mass_mev: float, center: tuple) -> dict:
    particles = _sample_case(case, mass_mev, center)
    bunch_text = _bunch_text(particles)
    meta = _meta_record(case, mass_mev, center, particles, bunch_text)
    os.makedirs(outdir, exist_ok=True)
    bunch_path = os.path.join(outdir, f"{case.name}.bunch")
    meta_path = os.path.join(outdir, f"{case.name}.meta.json")
    with open(bunch_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(bunch_text)
    with open(meta_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(_meta_text(meta))
    return meta


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------


def _check_spec_table(failures: list) -> None:
    for case in CASES:
        expected = SPEC_EXPECTED.get(case.name)
        if expected is None:
            failures.append(f"case {case.name}: not in SPEC_EXPECTED")
            continue
        section, seed, groups = expected
        if case.section != section:
            failures.append(f"case {case.name}: section {case.section} != {section}")
        if case.seed != seed:
            failures.append(f"case {case.name}: seed {case.seed} != {seed}")
        actual = tuple((g.name, g.count, g.kind, g.params) for g in case.groups)
        if actual != groups:
            failures.append(
                f"case {case.name}: groups {actual!r} != spec {groups!r}"
            )


def _check_close(failures: list, label: str, measured: float, target: float, tol: float) -> None:
    if not (abs(measured - target) <= tol):
        failures.append(
            f"{label}: measured {measured!r} vs target {target!r} "
            f"(|diff| = {abs(measured - target):.6g} > tol {tol:.6g})"
        )


def _stat_tolerances(n: int, target_std: float) -> tuple:
    """Two-standard-error tolerance for the mean and for the sample std."""
    mean_tol = N_SIGMA * target_std / math.sqrt(max(n, 1)) + 1.0e-12
    std_tol = SPREAD_N_SIGMA * target_std / math.sqrt(max(2 * (n - 1), 1)) + 1.0e-12
    return mean_tol, std_tol


def _verify_case(
    case: SpecCase, outdir: str, mass_mev: float, center: tuple, failures: list
) -> list:
    particles = _sample_case(case, mass_mev, center)
    rows = []
    if len(particles) != case.count:
        failures.append(f"case {case.name}: {len(particles)} particles != {case.count}")

    # geometry and support
    for p in particles:
        for axis in range(3):
            if abs(p[axis] - center[axis]) > R_BUNCH_M:
                failures.append(f"case {case.name}: particle outside the cube")
                break
        beta = _beta_of_u(p[3], p[4], p[5])
        if not (0.0 < beta <= 1.0):
            failures.append(f"case {case.name}: |v|/c = {beta!r} outside (0, 1]")

    for gi, group in enumerate(case.groups):
        selected = [p for p in particles if p[6] == gi]
        if len(selected) != group.count:
            failures.append(
                f"case {case.name}/{group.name}: {len(selected)} != {group.count}"
            )
        betas = [_beta_of_u(p[3], p[4], p[5]) for p in selected]
        mean, std, lo, hi = _moments(betas)
        analytic = _analytic_beta_moments(group, mass_mev)
        status = "ok"
        if analytic is not None:
            target_mean, target_std = analytic
            mean_tol, std_tol = _stat_tolerances(len(selected), target_std)
            before = len(failures)
            _check_close(
                failures, f"{case.name}/{group.name} mean|v|/c", mean, target_mean, mean_tol
            )
            _check_close(
                failures, f"{case.name}/{group.name} std|v|/c", std, target_std, std_tol
            )
            if len(failures) != before:
                status = "FAIL"
            rows.append(
                (
                    case.name, case.seed, case.count, group.name,
                    mean, std, target_mean, target_std, status,
                )
            )
        elif group.kind == "momentum_cube":
            p0 = group.params["p0_mev_c"]
            dp = group.params["dp_mev_c"]
            comp_target_mean = p0 / mass_mev
            comp_target_std = dp / (math.sqrt(3.0) * mass_mev)
            mean_tol, std_tol = _stat_tolerances(len(selected), comp_target_std)
            before = len(failures)
            for axis, label in ((3, "ux"), (4, "uy"), (5, "uz")):
                values = [p[axis] for p in selected]
                cmean, cstd, clo, chi = _moments(values)
                _check_close(
                    failures,
                    f"{case.name}/{label} mean",
                    cmean,
                    comp_target_mean,
                    mean_tol,
                )
                _check_close(
                    failures,
                    f"{case.name}/{label} std",
                    cstd,
                    comp_target_std,
                    std_tol,
                )
                if clo < (p0 - dp) / mass_mev or chi > (p0 + dp) / mass_mev:
                    failures.append(f"{case.name}/{label} outside [p0-dp, p0+dp]")
            if len(failures) != before:
                status = "FAIL"
            rows.append(
                (
                    case.name,
                    case.seed,
                    case.count,
                    group.name,
                    mean,
                    std,
                    float("nan"),
                    float("nan"),
                    status,
                )
            )
        elif group.kind == "delta_p":
            p0 = group.params["p0_mev_c"] / mass_mev
            before = len(failures)
            for p in selected:
                if p[3] != 0.0 or p[4] != 0.0 or p[5] != p0:
                    failures.append(f"{case.name}: single_velocity momentum differs")
                    break
            if std != 0.0:
                failures.append(f"{case.name}: single_velocity std {std!r} != 0")
            rows.append(
                (
                    case.name, case.seed, case.count, group.name, mean, std, mean, 0.0,
                    "ok" if len(failures) == before else "FAIL",
                )
            )
        if hi > BETA_MAX:
            failures.append(f"case {case.name}/{group.name}: max |v|/c {hi!r} > 1")

    # cross-check an existing record on disk
    meta_path = os.path.join(outdir, f"{case.name}.meta.json")
    bunch_path = os.path.join(outdir, f"{case.name}.bunch")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        expected = _meta_record(case, mass_mev, center, particles, _bunch_text(particles))
        if meta != expected:
            for key in sorted(set(meta) | set(expected)):
                if meta.get(key) != expected.get(key):
                    failures.append(f"{case.name}: meta field {key!r} differs from regenerated")
        if os.path.exists(bunch_path):
            with open(bunch_path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            if digest != expected["bunch_sha256"]:
                failures.append(f"{case.name}: bunch file sha256 differs from regenerated")
    return rows


def verify_cases(cases: list, outdir: str, mass_mev: float, center: tuple) -> int:
    failures = []
    _check_spec_table(failures)
    rows = []
    for case in cases:
        rows.extend(_verify_case(case, outdir, mass_mev, center, failures))

    print(f"{'case':16s} {'seed':>4s} {'count':>5s} {'group':12s} "
          f"{'mean|v|/c':>16s} {'std|v|/c':>12s} {'target_mean':>16s} "
          f"{'target_std':>12s}  status")
    for name, seed, count, group, mean, std, tmean, tstd, status in rows:
        print(
            f"{name:16s} {seed:4d} {count:5d} {group:12s} "
            f"{mean:16.12f} {std:12.6e} {tmean:16.12f} {tstd:12.6e}  {status}"
        )
    print(f"total particles: {sum(c.count for c in cases)}")
    if failures:
        print(f"\nVERIFY FAILED: {len(failures)} check(s)", file=sys.stderr)
        for message in failures:
            print(f"  - {message}", file=sys.stderr)
        return 1
    print("VERIFY OK")
    return 0


# --------------------------------------------------------------------------
# spec-mode CLI
# --------------------------------------------------------------------------


def _parse_triple(text: str) -> tuple:
    parts = [float(v) for v in text.split(",")]
    if len(parts) != 3 or not all(math.isfinite(v) for v in parts):
        raise SystemExit(f"--center expects three finite values, got {text!r}")
    return tuple(parts)


def spec_main(args) -> int:
    cases = [CASES_BY_NAME[name] for name in args.case] if args.case else list(CASES)
    mass_mev = args.mass_mev if args.mass_mev is not None else M_ELECTRON_C2_MEV
    if not (math.isfinite(mass_mev) and mass_mev > 0.0):
        raise SystemExit("--mass-mev must be finite and positive")
    center = _parse_triple(args.center) if args.center is not None else DEFAULT_CENTER_M
    outdir = os.path.abspath(args.out if args.out is not None else DEFAULT_OUT_DIR)

    if args.dry_run:
        print(f"out: {outdir}")
        print(f"mass: {mass_mev} MeV/c^2, centre: {center} m, cube side {2.0 * R_BUNCH_M} m")
        for case in cases:
            groups = ", ".join(f"{g.name}:{g.count}" for g in case.groups)
            print(
                f"{case.name:16s} section {case.section} seed {case.seed} "
                f"n={case.count} groups=[{groups}] "
                f"-> {os.path.join(outdir, case.name + '.bunch')}"
            )
        print(f"total particles: {sum(c.count for c in cases)} (no files written, --dry-run)")
        return 0

    if args.verify:
        return verify_cases(cases, outdir, mass_mev, center)

    total = 0
    for case in cases:
        meta = _write_case(case, outdir, mass_mev, center)
        total += meta["particle_count"]
        print(
            f"{case.name}: seed={case.seed} n={meta['particle_count']} "
            f"mean|v|/c={meta['measured']['mean_over_c']:.12f} "
            f"std|v|/c={meta['measured']['std_over_c']:.6e} "
            f"-> {os.path.join(outdir, case.name + '.bunch')}"
        )
    print(f"total particles: {total}")
    return 0


# --------------------------------------------------------------------------
# legacy gamma-based generator (deprecated, kept for existing callers)
# --------------------------------------------------------------------------


def u_of_gamma(g: float) -> float:
    return math.sqrt(max(0.0, g * g - 1.0))


def write_bunch(path: str, particles: list) -> None:
    with open(path, "w") as f:
        f.write("# x y z ux uy uz weight charge boxOf\n")
        for (x, y, z, ux, uy, uz, w) in particles:
            f.write(f"{x!r} {y!r} {z!r} {ux!r} {uy!r} {uz!r} {w!r} {K_ELECTRON_CHARGE!r} 0\n")


def beam_particles(
    rng: random.Random,
    n: int,
    gamma: float,
    gamma_fn,
    cloud_cells: float,
    cell: float,
    origin: tuple,
    grid: tuple,
    weight: float,
) -> list:
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


def legacy_main(args) -> int:
    """Old five-case gamma-based generator.

    Deprecated: it has no single-velocity case, uses hash()-derived seeds (which
    depend on PYTHONHASHSEED) and does not match nextTests.md section 3.  It is
    kept so existing scripts that call it with a positional OUTDIR keep
    working; use spec mode for acceptance evidence.
    """
    outdir = args.outdir
    sources = args.sources if args.sources is not None else 512
    gamma = args.gamma if args.gamma is not None else 50.0
    cloud_cells = args.cloud_cells if args.cloud_cells is not None else 3.0
    cell = args.cell if args.cell is not None else 1.0e-6
    grid_size = args.grid if args.grid is not None else 48
    origin_arg = args.origin if args.origin is not None else "0,0,0"
    seed = args.seed if args.seed is not None else 1

    print("WARNING: legacy gamma-based generator is deprecated; spec mode "
          "(--case/--out) implements nextTests.md section 3", file=sys.stderr)
    os.makedirs(outdir, exist_ok=True)
    origin = tuple(float(v) for v in origin_arg.split(","))
    grid = (grid_size, grid_size, grid_size)
    w = float(sources)

    def two_cluster(r: random.Random) -> float:
        return gamma * (1.0 + 0.20 * (1.0 if r.random() < 0.5 else -1.0)) + r.gauss(
            0.0, gamma * 0.02
        )

    def exp_tail(r: random.Random) -> float:
        if r.random() < 0.85:
            return gamma * (1.0 + r.gauss(0.0, 0.02))
        t = -math.log(1.0 - r.random())
        return gamma * (1.0 + 0.05 + 0.15 * t)

    def skewed(r: random.Random) -> float:
        u = -math.log(1.0 - r.random())
        return gamma * (1.0 + 0.01 + 0.10 * u)

    def core_halo(r: random.Random) -> float:
        if r.random() < 0.90:
            return gamma * (1.0 + r.gauss(0.0, 0.01))
        return gamma * (1.0 + 0.5 + 0.5 * r.random())

    def uniform_band(r: random.Random) -> float:
        return gamma * (1.0 + r.uniform(-0.10, 0.10))

    cases = {
        "two_cluster": two_cluster,
        "exp_tail": exp_tail,
        "skewed": skewed,
        "core_halo": core_halo,
        "uniform": uniform_band,
    }
    for name, fn in cases.items():
        rng = random.Random(seed + (hash(name) % 1000))
        parts = beam_particles(
            rng, sources, gamma, fn, cloud_cells, cell, origin, grid, w
        )
        path = os.path.join(outdir, f"{name}.bunch")
        write_bunch(path, parts)
        us = sorted(
            u_of_gamma(fn(random.Random(seed + (hash(name) % 1000) + 7)))
            for _ in range(sources)
        )
        print(
            f"{name}: n={sources} u in [{us[0]:.4g}, {us[-1]:.4g}] "
            f"median={us[len(us) // 2]:.4g}"
        )
    return 0


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

LEGACY_OPTIONS = ("sources", "gamma", "cloud_cells", "cell", "grid", "origin", "seed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "outdir",
        nargs="?",
        default=None,
        help="legacy positional output directory (selects the old gamma generator)",
    )
    parser.add_argument("--out", default=None,
                        help=f"output directory for spec mode (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--case", action="append", choices=[c.name for c in CASES],
                        help="spec case to generate (repeatable; default: all six)")
    parser.add_argument("--mass-mev", type=float, default=None,
                        help="rest energy m0 c^2 in MeV (default: electron, "
                             f"{M_ELECTRON_C2_MEV})")
    parser.add_argument("--center", default=None,
                        help="cube centre x,y,z in metres (default: "
                             "24e-6,24e-6,24e-6, the measurement driver's domain centre)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan (cases, seeds, counts, paths) and write nothing")
    parser.add_argument("--verify", action="store_true",
                        help="regenerate in memory and check counts, seeds, spreads and meta files")

    parser.add_argument("--sources", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gamma", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cloud-cells", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cell", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--grid", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--origin", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    legacy = args.outdir is not None or any(
        getattr(args, option) is not None for option in LEGACY_OPTIONS
    )
    spec = (
        args.out is not None
        or args.case is not None
        or args.center is not None
        or args.mass_mev is not None
        or args.dry_run
        or args.verify
    )
    if legacy and spec:
        parser.error(
            "the legacy positional OUTDIR/--sources/--gamma options cannot be "
            "combined with spec-mode options; use --out/--case/--mass-mev/--verify"
        )
    if args.case is not None and len(set(args.case)) != len(args.case):
        parser.error("--case names must be unique")
    if legacy:
        return legacy_main(args)
    return spec_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
