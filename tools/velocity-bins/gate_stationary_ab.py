#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
"""Pre-matrix guard: stationary versus moving gamma-ratio A/B.

Runs the binned measurement driver on two bunches that differ *only* in the
velocity and checks that the transverse electric field scales with gamma.

Why this gate exists
--------------------
An earlier measurement on an ancestor commit reported a gamma-1101 beam whose
velocity-moment deposit produced a field suppressed by roughly 1e14 relative
to the stationary arm.  The source audit says the default binned path is
dimensionally correct and the velocity-moment branch is off by default, but
that has not been proven empirically.  This gate is cheap and unambiguous: it
runs before the 60-run matrix so a broken deposit stops the campaign instead
of producing 60 meaningless files.

Input format
------------
The ``--bunch`` reader in ``tools/velocity-bins/measure_velocity_bins.cpp``
(lines 263-278) skips blank lines and lines starting with ``#`` and reads eight
whitespace separated values::

    x y z ux uy uz weight charge

with positions in metres, dimensionless ``u = p / (m0 c)`` and SI charge.  The
independently compiled fixture reader ``tools/velocity-bins/load_fixture.cpp``
(lines 77-91) reads the same eight columns after discarding the first line.
The driver's own writer emits the same header plus a ninth ``boxOf`` column
(``measure_velocity_bins.cpp`` line 356); this gate emits exactly the eight
columns requested by the task, so both readers accept the file.

Field extraction
----------------
The driver does not print a peak field.  It writes, per box count B,
``<prefix>_field_B<B>.bin`` as ``6 x n0*n1*n2`` little-endian doubles with x
fastest (``measure_velocity_bins.cpp`` lines 13 and 114-122); the grid is in
``<prefix>_meta_B<B>.txt``.  The component order is ``Ex Ey Ez Bx By Bz``:
``src/OpenPMDWriteOpenPMD.cpp`` lines 1895-1896 maps components 0..2 to the E
mesh and ``c % 3`` to x/y/z, and ``docs/algorithm.md`` line 76 fixes the Yee
staggering as Ex(.5,0,0) Ey(0,.5,0) Ez(0,0,.5).  For a particle moving along
+z, the transverse components are ``Ex`` and ``Ey``; ``Ez`` is longitudinal.
The gamma ratio therefore uses ``max(|Ex|, |Ey|)`` over the whole output grid
to locate the probe, then compares the two arms at that same node.

TSC correction (documented, not fitted)
---------------------------------------
A moving transverse kernel at gamma >> 1 is nonzero only when the observer and
the deposit node share a z plane; a deposit node one cell off in z is
suppressed by ~1/gamma^2.  TSC spreads a particle that sits on a z node to
z-1, z, z+1 with weights 0.125/0.75/0.125, while the stationary arm sees all
three planes.  The raw same-node ratio is therefore ``gamma * f`` with
``f = E_stationary(z-plane) / E_stationary(all planes)``, about 0.964 for this
geometry.  The gate computes ``f`` from the TSC weights and the Coulomb kernel
(``tsc_z0_fraction``) and reports the raw ratio, ``f`` and the corrected ratio.
The PASS/FAIL test uses the corrected ratio; a 1e-14 collapse stays a 1e-14
collapse after the division, so the failure signatures are untouched.

Pair C uses the L2 norm of (Ex, Ey), which stays O(sqrt(gamma)) and is a robust
"the field did not collapse" statistic; it needs no single-particle correction.

Moment switch
-------------
The measurement driver's off switch is ``--moments 0`` (default 0, and the
moment path only runs when ``momentOrder > 0``; see
``measure_velocity_bins.cpp`` lines 144 and 489).  ``--velocity-moments off``
belongs to the ``picBunchApp`` CLI (``src/main.cpp`` line 517), not to this
driver.  The gate passes ``--moments 0`` explicitly in both arms and records
that deviation in its output.

Exit codes: 0 for PASS and for ``--dry-run``/``--self-test``/``--emit-bunches``;
1 for a FAIL_* verdict or a failed self-test; 2 for ERROR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

# ---------------------------------------------------------------------------
# Constants from the repository sources
# ---------------------------------------------------------------------------

#: Elementary charge in coulombs (measure_velocity_bins.cpp line 36).
ELEMENTARY_CHARGE = 1.602176634e-19
#: Electron rest energy in MeV (m0 c^2; p0 / (m0 c) needs m0 c in MeV).
ELECTRON_MASS_MEV = 0.51099895
#: Spec momentum for pair B: p0 = 1e9 MeV/c.
P0_MEV_C = 1.0e9

#: Pair A: u = (0, 0, 1e9); the real-valued gamma is 1.0000000005e9.
U_A = 1.0e9
#: Pair B: u = p0 / (m0 c).  Computed, never hardcoded, so gamma follows the
#: value actually written to the bunch file.  This is 1.956951183559e9, which
#: rounds to the spec's 1.9569510e9.
U_B = P0_MEV_C / ELECTRON_MASS_MEV

#: Grid geometry shared by every arm.  The particle lives on a z node (see
#: below); 32^3 with a 1 um cell keeps the source away from the periodic wrap.
GRID = (32, 32, 32)
CELL_M = (1.0e-6, 1.0e-6, 1.0e-6)
ORIGIN_M = (0.0, 0.0, 0.0)
CENTER_NODE = tuple(GRID[a] // 2 for a in range(3))
CENTER_M = tuple(CENTER_NODE[a] * CELL_M[a] for a in range(3))

#: The repo has no literal "case 3.2" label.  The closest documented case is
#: tests/test_velocity_bins.cpp:493 (MonoenergeticHighGammaUsesOneBoxWithoutFlags):
#: gamma 1101, a 3 mrad ring plus a 1-3 mrad tilt, deterministic in p.  Pair C
#: reuses that construction for 32 particles.
CASE32_GAMMA = 1101.0
CASE32_U = math.sqrt(CASE32_GAMMA * CASE32_GAMMA - 1.0)
CASE32_THETA_BASE = 3.0e-3
CASE32_THETA_STEP = 1.5e-3
CASE32_PARTICLES = 32

#: Verdict thresholds from the task.
GAMMA_REL_TOL = 1.0e-3
COLLAPSE_RATIO = 1.0e-6
UNIT_RATIO_LO = 1.0e12
UNIT_RATIO_HI = 1.0e15
C_RATIO_LO = 1.0e-6
C_RATIO_HI = 1.0e3

#: Measurement-driver off switch for the velocity-moment path.
MOMENT_ARGS = ("--moments", "0")

HEADER = "# x y z ux uy uz weight charge"
COMPONENT_NAMES = ("Ex", "Ey", "Ez", "Bx", "By", "Bz")
#: Yee staggering in cell units (docs/algorithm.md line 76).
YEE_OFFSETS = (
    (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5),
    (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0),
)

BUNCH_FILES = {
    "A_stationary": "pairA_stationary.txt",
    "A_moving": "pairA_moving.txt",
    "B_stationary": "pairB_stationary.txt",
    "B_moving": "pairB_moving.txt",
    "C_stationary": "pairC_stationary.txt",
    "C_moving": "pairC_moving.txt",
}


class GateError(RuntimeError):
    """An ERROR verdict: missing binary, non-zero exit, unparseable output."""

    def __init__(self, message: str, command: Optional[Sequence[str]] = None,
                 stderr: str = "") -> None:
        super().__init__(message)
        self.command = list(command) if command else []
        self.stderr = stderr


# ---------------------------------------------------------------------------
# Bunch construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Particle:
    x: float
    y: float
    z: float
    ux: float
    uy: float
    uz: float
    weight: float
    charge: float

    def columns(self) -> tuple[float, ...]:
        return (self.x, self.y, self.z, self.ux, self.uy, self.uz,
                self.weight, self.charge)


def gamma_from_u(u: Sequence[float]) -> float:
    """gamma = sqrt(1 + |u|^2) from the u actually written."""
    return math.sqrt(1.0 + sum(float(c) * float(c) for c in u))


def _particle(pos: Sequence[float], u: Sequence[float]) -> Particle:
    return Particle(float(pos[0]), float(pos[1]), float(pos[2]),
                    float(u[0]), float(u[1]), float(u[2]),
                    1.0, -ELEMENTARY_CHARGE)


def pair_a_stationary() -> list[Particle]:
    return [_particle(CENTER_M, (0.0, 0.0, 0.0))]


def pair_a_moving() -> list[Particle]:
    return [_particle(CENTER_M, (0.0, 0.0, U_A))]


def pair_b_stationary() -> list[Particle]:
    return [_particle(CENTER_M, (0.0, 0.0, 0.0))]


def pair_b_moving() -> list[Particle]:
    return [_particle(CENTER_M, (0.0, 0.0, U_B))]


def case32_positions() -> list[tuple[float, float, float]]:
    """32 positions on the inner 3x3x3 node block around the grid centre.

    The z coordinate is always an integer node: the moving pancake only
    intersects an observation plane at the source z plane.  A cell-centred z
    would put the pancake between nodes and suppress the moving field by
    ~1/gamma^2, which would masquerade as the collapse this gate hunts for.
    """
    positions = []
    for p in range(CASE32_PARTICLES):
        ix = CENTER_NODE[0] + (p % 3) - 1
        iy = CENTER_NODE[1] + ((p // 3) % 3) - 1
        iz = CENTER_NODE[2] + ((p // 9) % 3) - 1
        positions.append((ix * CELL_M[0], iy * CELL_M[1], iz * CELL_M[2]))
    return positions


def case32_velocities() -> list[tuple[float, float, float]]:
    """Deterministic gamma-1101 velocities from tests/test_velocity_bins.cpp:493."""
    velocities = []
    for p in range(CASE32_PARTICLES):
        theta = CASE32_THETA_BASE + CASE32_THETA_STEP * (p // 16 % 2)
        phi = 2.0 * math.pi * (
            (2.0 * float(p % 16) + 1.0) / 32.0 + (p // 16 % 2) * 0.02
        )
        velocities.append((
            CASE32_U * math.sin(theta) * math.cos(phi),
            CASE32_U * math.sin(theta) * math.sin(phi),
            CASE32_U * math.cos(theta),
        ))
    return velocities


def pair_c_stationary() -> list[Particle]:
    return [_particle(pos, (0.0, 0.0, 0.0)) for pos in case32_positions()]


def pair_c_moving() -> list[Particle]:
    return [_particle(pos, u)
            for pos, u in zip(case32_positions(), case32_velocities())]


def case32_representative_u() -> tuple[float, float, float]:
    """Charge-weighted mean u; weights are equal, so this is the plain mean.

    The one-box path uses the absolute-charge-weighted mean momentum as the
    box representative (README.md line 575), so this is the gamma the binned
    pair-C arm actually sees.
    """
    velocities = case32_velocities()
    return tuple(
        sum(u[a] for u in velocities) / float(len(velocities)) for a in range(3)
    )


def all_bunches() -> dict[str, list[Particle]]:
    return {
        "A_stationary": pair_a_stationary(),
        "A_moving": pair_a_moving(),
        "B_stationary": pair_b_stationary(),
        "B_moving": pair_b_moving(),
        "C_stationary": pair_c_stationary(),
        "C_moving": pair_c_moving(),
    }


def bunch_bytes(particles: Sequence[Particle]) -> bytes:
    """Serialise exactly the 8 columns the reader parses.

    ``.17g`` round-trips an IEEE double, so the file is byte-reproducible and
    the reader recovers the same u values the gate used to compute gamma.
    """
    lines = [HEADER]
    for particle in particles:
        lines.append(" ".join(format(value, ".17g")
                              for value in particle.columns()))
    return ("\n".join(lines) + "\n").encode("ascii")


def write_bunches(directory: Path) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, particles in all_bunches().items():
        path = directory / BUNCH_FILES[key]
        path.write_bytes(bunch_bytes(particles))
        paths[key] = path
    return paths


# ---------------------------------------------------------------------------
# Measurement-driver invocation
# ---------------------------------------------------------------------------


def measurement_command(binary: str, bunch_path: Path, prefix: Path) -> list[str]:
    """Build one arm's command line.

    ``--bins 1 --bin-edges uniform`` is the driver's only way to request an
    explicit one-box uniform binning: ``--bin-edges`` accepts the rule names
    ``uniform|quantile|adaptive`` (measure_velocity_bins.cpp lines 197-209),
    there is no option that takes numeric edges.  Both arms use the same grid,
    cell, origin, box count and edge rule; only the bunch file differs.
    """
    return [
        binary,
        "--bunch", str(bunch_path),
        "--bins", "1",
        "--bin-edges", "uniform",
        *MOMENT_ARGS,
        "--grid", ",".join(str(n) for n in GRID),
        "--cell", ",".join(format(v, ".17g") for v in CELL_M),
        "--origin", ",".join(format(v, ".17g") for v in ORIGIN_M),
        "--output", str(prefix),
    ]


# ---------------------------------------------------------------------------
# Field binary parsing
# ---------------------------------------------------------------------------


@dataclass
class FieldData:
    path: Path
    grid: tuple[int, int, int]
    n: int
    components: list[list[float]]
    finite: bool
    max_abs: tuple[float, float, float, float, float, float]
    max_trans: float
    max_trans_component: int
    max_trans_index: int
    sum_sq_trans: float

    def component(self, index: int) -> list[float]:
        return self.components[index]

    def summary(self) -> dict:
        return {
            "path": str(self.path),
            "grid": list(self.grid),
            "samples": self.n,
            "finite": self.finite,
            "max_abs": {name: value for name, value in
                        zip(COMPONENT_NAMES, self.max_abs)},
            "max_transverse": self.max_trans,
            "max_transverse_component": COMPONENT_NAMES[self.max_trans_component],
            "max_transverse_index": self.max_trans_index,
            "sum_sq_transverse": self.sum_sq_trans,
        }


def read_meta(path: Path) -> dict:
    if not path.is_file():
        raise GateError("missing metadata file " + str(path))
    meta: dict = {"grid": None, "bins_used": None, "per_axis": None,
                  "model_sum": None, "cell": None, "origin": None}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "grid" and len(parts) >= 4:
            meta["grid"] = tuple(int(parts[1 + a]) for a in range(3))
        elif parts[0] == "bins_used" and len(parts) >= 2:
            meta["bins_used"] = int(parts[1])
        elif parts[0] == "per_axis" and len(parts) >= 4:
            meta["per_axis"] = tuple(int(parts[1 + a]) for a in range(3))
        elif parts[0] == "cell" and len(parts) >= 4:
            meta["cell"] = tuple(float(parts[1 + a]) for a in range(3))
        elif parts[0] == "origin" and len(parts) >= 4:
            meta["origin"] = tuple(float(parts[1 + a]) for a in range(3))
        elif parts[0] == "model_sum" and len(parts) >= 2:
            try:
                meta["model_sum"] = float(parts[1])
            except ValueError:
                pass
    if meta["grid"] is None:
        raise GateError("metadata " + str(path) + " has no grid line")
    return meta


def read_field(path: Path, grid: Sequence[int]) -> FieldData:
    if not path.is_file():
        raise GateError("missing field file " + str(path))
    n = int(grid[0]) * int(grid[1]) * int(grid[2])
    expected = 6 * n * 8
    raw = path.read_bytes()
    if len(raw) != expected:
        raise GateError(
            "field file {} has {} bytes, expected {} = 6 * {} * 8 "
            "(6 x n0*n1*n2 little-endian doubles, see "
            "measure_velocity_bins.cpp:13)".format(path, len(raw), expected, n)
        )
    values = struct.unpack("<{}d".format(6 * n), raw)
    components = [list(values[c * n:(c + 1) * n]) for c in range(6)]
    finite = all(math.isfinite(value) for value in values)

    max_abs = []
    for component in components:
        peak = max(abs(value) for value in component)
        max_abs.append(peak)

    trans = components[0], components[1]
    max_trans = 0.0
    max_trans_component = 0
    max_trans_index = 0
    for c, component in enumerate(trans):
        for index, value in enumerate(component):
            magnitude = abs(value)
            if magnitude > max_trans:
                max_trans = magnitude
                max_trans_component = c
                max_trans_index = index

    sum_sq_trans = sum(value * value for component in trans
                       for value in component)

    return FieldData(
        path=path,
        grid=tuple(int(v) for v in grid),
        n=n,
        components=components,
        finite=finite,
        max_abs=tuple(max_abs),
        max_trans=max_trans,
        max_trans_component=max_trans_component,
        max_trans_index=max_trans_index,
        sum_sq_trans=sum_sq_trans,
    )


# ---------------------------------------------------------------------------
# TSC deposit correction for the single-particle gamma test
# ---------------------------------------------------------------------------


def tsc_weight(argument: float) -> float:
    """TSC assignment weight, argument in cell units (Kernel.hpp:39-48)."""
    ax = abs(argument)
    if ax < 0.5:
        return 0.75 - ax * ax
    if ax < 1.5:
        return 0.5 * (1.5 - ax) * (1.5 - ax)
    return 0.0


def point_field_stationary(R: Sequence[float], component: int) -> float:
    """Coulomb point field, same normalization as pointFieldNormalized."""
    r2 = R[0] * R[0] + R[1] * R[1] + R[2] * R[2]
    if r2 == 0.0:
        return 0.0
    scale = 1.0 / (r2 * math.sqrt(r2))
    return scale * R[component] if component < 3 else 0.0


def tsc_z0_fraction(particle: Particle, observer_flat_index: int,
                    grid: Sequence[int], cell: Sequence[float],
                    component: int,
                    origin: Sequence[float] = (0.0, 0.0, 0.0)) -> Optional[float]:
    """Fraction of the stationary field from the observer's own z plane.

    The binned moving kernel is transverse, so at gamma >> 1 only deposit
    nodes with ``n_z == observer_z`` survive (a node one cell off in z is
    suppressed by ~1/gamma^2).  The TSC deposit, however, spreads a particle
    at a z node to z-1, z, z+1 with weights 0.125/0.75/0.125.  The raw
    moving/stationary peak ratio is therefore gamma times this z-plane
    fraction, not gamma.  Dividing it out is a correction for a known property
    of the shared deposit, not a fitted fudge: both the weights and the kernel
    are the source formulas above.

    Returns ``None`` when the measured peak is not within two cells of the
    single particle, so a corrupted field is never silently corrected.
    """
    nx, ny, nz = (int(v) for v in grid)
    if len(cell) != 3:
        return None
    ix = observer_flat_index % nx
    iy = (observer_flat_index // nx) % ny
    iz = observer_flat_index // (nx * ny)
    if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
        return None
    observer = (ix, iy, iz)

    position = ((particle.x - origin[0]) / cell[0],
                (particle.y - origin[1]) / cell[1],
                (particle.z - origin[2]) / cell[2])
    centre = tuple(int(round(value)) for value in position)
    fraction = tuple(position[a] - centre[a] for a in range(3))
    if max(abs(observer[a] - centre[a]) for a in range(3)) > 2:
        return None

    total = 0.0
    same_plane = 0.0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                weight = (tsc_weight(dx - fraction[0])
                          * tsc_weight(dy - fraction[1])
                          * tsc_weight(dz - fraction[2]))
                if weight == 0.0:
                    continue
                node = (centre[0] + dx, centre[1] + dy, centre[2] + dz)
                displacement = tuple(observer[a] - node[a] for a in range(3))
                R = tuple((displacement[a] + YEE_OFFSETS[component][a])
                          * cell[a] for a in range(3))
                value = weight * point_field_stationary(R, component)
                total += value
                if node[2] == observer[2]:
                    same_plane += value
    if total == 0.0 or not math.isfinite(total) or not math.isfinite(same_plane):
        return None
    result = same_plane / total
    if not math.isfinite(result) or result <= 0.0:
        return None
    return result


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def classify_ab(ratio: Optional[float], gamma: float, finite: bool) -> tuple[Optional[str], str]:
    """Classify one gamma-ratio pair.

    Returns ``(label, reason)``; ``label is None`` means this arm passes.
    """
    if not finite:
        return "FAIL_COLLAPSE", "a sampled field value is NaN or Inf"
    if ratio is None or not math.isfinite(ratio):
        return "FAIL_COLLAPSE", "ratio is not finite"
    if ratio < COLLAPSE_RATIO:
        return "FAIL_COLLAPSE", (
            "ratio {:.6e} < {:g}: the ~1e14 collapse signature".format(
                ratio, COLLAPSE_RATIO)
        )
    if UNIT_RATIO_LO <= ratio <= UNIT_RATIO_HI:
        return "FAIL_UNITS", (
            "ratio {:.6e} in [{:g}, {:g}]: unit mis-assignment".format(
                ratio, UNIT_RATIO_LO, UNIT_RATIO_HI)
        )
    relative = abs(ratio - gamma) / gamma
    if relative < GAMMA_REL_TOL:
        return None, "ratio agrees with gamma to {:.3e}".format(relative)
    if ratio < 2.0:
        return "FAIL_GAMMA_SCALING", (
            "ratio {:.6e} is near unity: gamma amplification is missing".format(
                ratio)
        )
    # The task leaves no bucket for "amplification present but magnitude off".
    # Keep the machine label inside the allowed set and spell the number out.
    return "FAIL_GAMMA_SCALING", (
        "ratio {:.6e} is {:.3e} relative away from gamma {:.6e}: "
        "gamma scaling present but wrong".format(ratio, relative, gamma)
    )


def classify_c(ratio: Optional[float], finite: bool) -> tuple[Optional[str], str]:
    """Classify pair C's multi-particle sanity ratio."""
    if not finite:
        return "FAIL_COLLAPSE", "a sampled field value is NaN or Inf"
    if ratio is None or not math.isfinite(ratio):
        return "FAIL_COLLAPSE", "ratio is not finite"
    if ratio < C_RATIO_LO:
        return "FAIL_COLLAPSE", (
            "ratio {:.6e} < {:g}: multi-particle field collapsed".format(
                ratio, C_RATIO_LO)
        )
    if UNIT_RATIO_LO <= ratio <= UNIT_RATIO_HI:
        return "FAIL_UNITS", (
            "ratio {:.6e} in [{:g}, {:g}]: unit mis-assignment".format(
                ratio, UNIT_RATIO_LO, UNIT_RATIO_HI)
        )
    if ratio > C_RATIO_HI:
        return "FAIL_GAMMA_SCALING", (
            "ratio {:.6e} > {:g}: multi-particle enhancement outside the "
            "sanity band".format(ratio, C_RATIO_HI)
        )
    return None, "ratio {:.6e} is inside [{:g}, {:g}]".format(
        ratio, C_RATIO_LO, C_RATIO_HI)


def combine_verdict(labels: Iterable[Optional[str]]) -> str:
    present = [label for label in labels if label]
    for label in ("FAIL_COLLAPSE", "FAIL_UNITS", "FAIL_GAMMA_SCALING"):
        if label in present:
            return label
    return "PASS"


# ---------------------------------------------------------------------------
# Running the arms
# ---------------------------------------------------------------------------


@dataclass
class ArmResult:
    tag: str
    prefix: Path
    command: list[str]
    meta: dict
    field: FieldData
    stdout: str
    stderr: str

    def summary(self) -> dict:
        return {
            "tag": self.tag,
            "prefix": str(self.prefix),
            "command": self.command,
            "meta": {
                "grid": list(self.meta["grid"]),
                "bins_used": self.meta["bins_used"],
                "per_axis": list(self.meta["per_axis"])
                if self.meta["per_axis"] else None,
                "model_sum": self.meta["model_sum"],
                "cell": list(self.meta["cell"]) if self.meta["cell"] else None,
                "origin": list(self.meta["origin"]) if self.meta["origin"] else None,
            },
            "field": self.field.summary(),
        }


def run_arm(binary: str, bunch_path: Path, tag: str, run_dir: Path,
            timeout_s: float) -> ArmResult:
    prefix = run_dir / tag
    command = measurement_command(binary, bunch_path, prefix)
    try:
        process = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(run_dir),
        )
    except FileNotFoundError as exc:
        raise GateError("binary not found: " + binary, command, str(exc))
    except PermissionError as exc:
        raise GateError("binary not executable: " + binary, command, str(exc))
    except subprocess.TimeoutExpired as exc:
        raise GateError("binary timed out after {:g} s".format(timeout_s),
                        command, str(exc))

    (run_dir / (tag + "_stdout.txt")).write_text(process.stdout,
                                                  encoding="utf-8")
    (run_dir / (tag + "_stderr.txt")).write_text(process.stderr,
                                                  encoding="utf-8")
    if process.returncode != 0:
        tail = "\n".join(process.stderr.strip().splitlines()[-20:])
        raise GateError(
            "{} exited with status {}; command: {}; stderr:\n{}".format(
                tag, process.returncode, " ".join(command), tail or "<empty>"),
            command,
            process.stderr,
        )

    meta_path = Path(str(prefix) + "_meta_B1.txt")
    field_path = Path(str(prefix) + "_field_B1.bin")
    try:
        meta = read_meta(meta_path)
        if meta["bins_used"] != 1:
            raise GateError(
                "{}: expected bins_used=1, got {}".format(
                    tag, meta["bins_used"]))
        field = read_field(field_path, meta["grid"])
    except (GateError, ValueError, struct.error) as exc:
        raise GateError(
            "{}: cannot parse the driver output: {}".format(tag, exc),
            command,
            process.stderr,
        ) from exc
    return ArmResult(tag, prefix, command, meta, field, process.stdout,
                     process.stderr)


@dataclass
class PairResult:
    name: str
    gamma: float
    u_moving: tuple[float, float, float]
    stationary: ArmResult
    moving: ArmResult
    ratio: float
    ratio_peak_raw: float
    ratio_node_raw: float
    l2_ratio: float
    tsc_z0_fraction: Optional[float] = None
    correction_applied: bool = False
    label: Optional[str] = None
    reason: str = ""
    metric: str = "max|Ex,Ey|"

    def summary(self) -> dict:
        return {
            "name": self.name,
            "u_moving": list(self.u_moving),
            "gamma": self.gamma,
            "metric": self.metric,
            "ratio": self.ratio,
            "ratio_peak_raw": self.ratio_peak_raw,
            "ratio_node_raw": self.ratio_node_raw,
            "tsc_z0_fraction": self.tsc_z0_fraction,
            "tsc_correction_applied": self.correction_applied,
            "l2_ratio": self.l2_ratio,
            "relative_error_vs_gamma": (
                abs(self.ratio - self.gamma) / self.gamma
                if self.name in ("A", "B") else None
            ),
            "label": self.label or "PASS",
            "reason": self.reason,
            "stationary": self.stationary.summary(),
            "moving": self.moving.summary(),
        }


def make_pair(name: str, u_moving: Sequence[float], stationary: ArmResult,
              moving: ArmResult, metric: str) -> PairResult:
    gamma = gamma_from_u(u_moving)
    stationary_peak = stationary.field.max_trans
    if stationary_peak == 0.0:
        peak_ratio = 0.0
    else:
        peak_ratio = moving.field.max_trans / stationary_peak

    # Fixed-point ratio: re-use the stationary arm's argmax node and component
    # in the moving arm.  This is a direct same-node A/B number; the peak ratio
    # above is what the task's max|E| wording asks for.
    component = stationary.field.max_trans_component
    index = stationary.field.max_trans_index
    if stationary_peak == 0.0:
        node_ratio = 0.0
    else:
        node_ratio = abs(moving.field.component(component)[index]) / stationary_peak

    if stationary.field.sum_sq_trans == 0.0:
        l2 = 0.0
    else:
        l2 = math.sqrt(moving.field.sum_sq_trans
                       / stationary.field.sum_sq_trans)

    return PairResult(name, gamma, tuple(float(v) for v in u_moving),
                      stationary, moving, peak_ratio, peak_ratio, node_ratio,
                      l2, metric=metric)


def apply_tsc_correction(pair: PairResult, particle: Particle) -> None:
    """Turn the same-node raw ratio into the gamma ratio.

    The raw ratio is ``gamma * f`` where ``f`` is the stationary field
    fraction from the observer's own z deposit plane.  ``f`` is computed from
    the TSC weights and the Coulomb kernel, then divided out.  A 1e-14 collapse
    stays a 1e-14 collapse after the division, so the failure signatures are
    unchanged.
    """
    meta = pair.stationary.meta
    cell = meta.get("cell") or CELL_M
    origin = meta.get("origin") or ORIGIN_M
    fraction = tsc_z0_fraction(
        particle,
        pair.stationary.field.max_trans_index,
        pair.stationary.field.grid,
        cell,
        pair.stationary.field.max_trans_component,
        origin,
    )
    pair.tsc_z0_fraction = fraction
    if fraction is None or not math.isfinite(fraction) or fraction <= 0.0:
        pair.correction_applied = False
        pair.ratio = pair.ratio_node_raw
        return
    pair.correction_applied = True
    pair.ratio = pair.ratio_node_raw / fraction


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test_field(tmp: Path, values: Sequence[float], grid=(2, 2, 1)) -> FieldData:
    path = tmp / "synthetic_field.bin"
    path.write_bytes(struct.pack("<{}d".format(len(values)), *values))
    return read_field(path, grid)


def run_self_test(verbose: bool = True) -> int:
    failures: list[str] = []

    def check(condition: bool, description: str) -> None:
        if condition:
            if verbose:
                print("[PASS] " + description)
        else:
            failures.append(description)
            print("[FAIL] " + description)

    # gamma values come from the u that is written.
    gamma_a = gamma_from_u((0.0, 0.0, U_A))
    gamma_b = gamma_from_u((0.0, 0.0, U_B))
    check(gamma_a >= 1.0e9 and abs(gamma_a - U_A) / U_A < 1e-9,
          "pair A gamma from written u ({:.17g})".format(gamma_a))
    check(abs(gamma_b - U_B) / U_B < 1e-12,
          "pair B gamma from written u ({:.17g})".format(gamma_b))
    check(abs(U_B - 1.9569510e9) / U_B < 1e-6,
          "pair B u rounds to the spec's 1.9569510e9"
          " ({:.17g})".format(U_B))

    # Verdict function: PASS, collapse, gamma-missing, unit-error.
    cases = [
        ("PASS", gamma_a, True, None),
        ("PASS within tolerance", gamma_a * (1.0 + 5.0e-4), True, None),
        ("collapse 1e-14", 1.0e-14, True, "FAIL_COLLAPSE"),
        ("collapse NaN", None, False, "FAIL_COLLAPSE"),
        ("gamma missing (ratio~1)", 1.0, True, "FAIL_GAMMA_SCALING"),
        ("unit error 1e12", 1.0e12, True, "FAIL_UNITS"),
        ("unit error 1e15", 1.0e15, True, "FAIL_UNITS"),
        ("residual magnitude error", gamma_a * 1.1, True,
         "FAIL_GAMMA_SCALING"),
    ]
    for description, ratio, finite, expected in cases:
        label, reason = classify_ab(ratio, gamma_a, finite)
        check(label == expected,
              "classify_ab {} -> {} (expected {}); {}".format(
                  description, label, expected, reason))

    c_cases = [
        ("C in band 33", 33.0, True, None),
        ("C collapse 1e-14", 1.0e-14, True, "FAIL_COLLAPSE"),
        ("C unit error 1e12", 1.0e12, True, "FAIL_UNITS"),
        ("C above band", 2.0e3, True, "FAIL_GAMMA_SCALING"),
    ]
    for description, ratio, finite, expected in c_cases:
        label, reason = classify_c(ratio, finite)
        check(label == expected,
              "classify_c {} -> {} (expected {}); {}".format(
                  description, label, expected, reason))

    check(combine_verdict([None, "FAIL_UNITS", "FAIL_COLLAPSE"])
          == "FAIL_COLLAPSE", "verdict priority puts collapse first")
    check(combine_verdict([None, None, None]) == "PASS",
          "all-pass labels combine to PASS")

    # Bunch files are byte-reproducible and have the 8-column reader layout.
    first = {key: bunch_bytes(particles) for key, particles in all_bunches().items()}
    second = {key: bunch_bytes(particles) for key, particles in all_bunches().items()}
    for key in first:
        check(first[key] == second[key],
              "bunch file {} is byte-reproducible".format(BUNCH_FILES[key]))
        lines = first[key].decode("ascii").splitlines()
        check(lines[0] == HEADER, "{} header is exactly 8 columns".format(key))
        parsed = [line.split() for line in lines[1:]]
        check(all(len(columns) == 8 for columns in parsed),
              "{} every row has 8 columns".format(key))
        check(all(math.isfinite(float(value))
                  for columns in parsed for value in columns),
              "{} every value parses as a finite float".format(key))

    # Pair A/B must differ only in velocity, never in position.
    a_stat = pair_a_stationary()[0]
    a_mov = pair_a_moving()[0]
    b_stat = pair_b_stationary()[0]
    b_mov = pair_b_moving()[0]
    check((a_stat.x, a_stat.y, a_stat.z) == (a_mov.x, a_mov.y, a_mov.z),
          "pair A positions are identical across arms")
    check((b_stat.x, b_stat.y, b_stat.z) == (b_mov.x, b_mov.y, b_mov.z),
          "pair B positions are identical across arms")
    check(a_stat.ux == a_stat.uy == a_stat.uz == 0.0
          and b_stat.ux == b_stat.uy == b_stat.uz == 0.0,
          "stationary arms have u = (0,0,0)")
    check(a_mov.uz == U_A and a_mov.ux == a_mov.uy == 0.0,
          "pair A writes u = (0,0,1e9)")
    check(b_mov.uz == U_B and b_mov.ux == b_mov.uy == 0.0,
          "pair B writes u = (0,0,p0/(m0 c))")
    check(all(pos[2] / CELL_M[2] == round(pos[2] / CELL_M[2])
              for pos in case32_positions()),
          "pair C positions are on z grid nodes")

    # Field parser: known maxima, component order, and NaN detection.
    n = 4
    values = [0.0] * (6 * n)
    values[0] = 3.0      # Ex[0]
    values[1 * n + 1] = 4.0  # Ey[1]
    values[2 * n + 2] = -5.0  # Ez[2]
    with tempfile.TemporaryDirectory(prefix="tg-selftest-") as tmp_name:
        tmp = Path(tmp_name)
        field = _self_test_field(tmp, values)
        check(field.max_trans == 4.0 and field.max_trans_component == 1
              and field.max_trans_index == 1,
              "field parser reports max|Ex,Ey| and its component/index")
        check(field.max_abs[2] == 5.0,
              "field parser keeps Ex,Ey,Ez,Bx,By,Bz order")
        check(field.finite, "field parser accepts finite values")
        nan_values = list(values)
        nan_values[3 * n] = float("nan")
        field_nan = _self_test_field(tmp, nan_values)
        check(not field_nan.finite, "field parser detects NaN")
        try:
            path = tmp / "short.bin"
            path.write_bytes(b"\x00" * 8)
            read_field(path, (2, 2, 1))
            check(False, "field parser rejects a short file")
        except GateError:
            check(True, "field parser rejects a short file")

    # TSC z-plane correction: the known factor for a node-centred single
    # particle at the Ex/Ey peak node, and the collapse/guard behaviour.
    single = pair_a_moving()[0]
    nx, ny, _ = GRID
    ex_index = ((CENTER_NODE[0] - 1)
                + CENTER_NODE[1] * nx + CENTER_NODE[2] * nx * ny)
    ey_index = (CENTER_NODE[0]
                + (CENTER_NODE[1] - 1) * nx + CENTER_NODE[2] * nx * ny)
    fraction_ex = tsc_z0_fraction(single, ex_index, GRID, CELL_M, 0)
    fraction_ey = tsc_z0_fraction(single, ey_index, GRID, CELL_M, 1)
    check(fraction_ex is not None
          and abs(fraction_ex - 0.964263564144275) < 1e-12,
          "TSC z0 fraction at the Ex peak ({!r})".format(fraction_ex))
    check(fraction_ey is not None
          and abs(fraction_ey - 0.964263564144275) < 1e-12,
          "TSC z0 fraction at the Ey peak ({!r})".format(fraction_ey))
    check(tsc_z0_fraction(single, 0, GRID, CELL_M, 0) is None,
          "TSC correction refuses a peak far from the particle")
    raw = gamma_a * fraction_ex
    check(abs(raw / fraction_ex - gamma_a) / gamma_a < 1e-12,
          "TSC correction recovers gamma from a raw ratio of gamma*f")

    if failures:
        print("SELF_TEST=FAIL ({} failure(s))".format(len(failures)))
        return 1
    print("SELF_TEST=PASS")
    return 0


# ---------------------------------------------------------------------------
# Dry run and emit
# ---------------------------------------------------------------------------


def planned_run_dir(args: argparse.Namespace) -> Path:
    if args.workdir:
        return Path(args.workdir).expanduser().resolve() / "tg_stationary_ab"
    return Path(tempfile.gettempdir()) / "tg_stationary_ab_dry_run"


def print_dry_run(args: argparse.Namespace) -> int:
    run_dir = planned_run_dir(args)
    binary = args.binary or "<BINARY>"
    print("GATE: stationary versus moving gamma-ratio A/B")
    print("binary: {} (not required for --dry-run)".format(binary))
    print("run dir: {}".format(run_dir))
    print("grid={} cell={} origin={} bins=1 bin-edges=uniform".format(
        GRID, CELL_M, ORIGIN_M))
    print("moment switch: {} (the measurement driver's off switch; "
          "--velocity-moments off belongs to picBunchApp)".format(
              " ".join(MOMENT_ARGS)))
    print("")
    print("Pair A: u=0 versus u=({:.17g},{:.17g},{:.17g})".format(0.0, 0.0, U_A))
    print("  gamma = sqrt(1+u^2) = {:.17g}".format(gamma_from_u((0, 0, U_A))))
    print("  expected ratio = gamma = {:.17g}".format(gamma_from_u((0, 0, U_A))))
    print("Pair B: u=0 versus u=({:.17g},{:.17g},{:.17g})".format(0.0, 0.0, U_B))
    print("  u = p0/(m0 c) = {:g}/{:g}".format(P0_MEV_C, ELECTRON_MASS_MEV))
    print("  gamma = sqrt(1+u^2) = {:.17g}".format(gamma_from_u((0, 0, U_B))))
    print("  expected ratio = gamma = {:.17g}".format(gamma_from_u((0, 0, U_B))))
    print("Pair C: {} particles at gamma {:.0f} (tests/test_velocity_bins.cpp:493)"
          " versus the same positions at u=0".format(
              CASE32_PARTICLES, CASE32_GAMMA))
    print("  expected L2(|Ex,Ey|) ratio about sqrt(gamma) ~ {:.1f}, "
          "sanity band [{:g}, {:g}]".format(
              math.sqrt(CASE32_GAMMA), C_RATIO_LO, C_RATIO_HI))
    print("")
    for tag, filename in BUNCH_FILES.items():
        bunch = run_dir / filename
        prefix = run_dir / tag
        print("planned command [{}]:".format(tag))
        print("  " + " ".join(measurement_command(binary, bunch, prefix)))
        print("  outputs: {}_field_B1.bin, {}_meta_B1.txt, {}_report.txt".format(
            prefix, prefix, prefix))
    print("")
    print("verdict rules: PASS if both A and B agree with gamma within {:g} "
          "and C in [{:g}, {:g}]; FAIL_COLLAPSE if ratio < {:g} or NaN/Inf; "
          "FAIL_UNITS if ratio in [{:g}, {:g}]; FAIL_GAMMA_SCALING otherwise."
          .format(GAMMA_REL_TOL, C_RATIO_LO, C_RATIO_HI, COLLAPSE_RATIO,
                  UNIT_RATIO_LO, UNIT_RATIO_HI))
    print("DRY_RUN=PASS")
    return 0


def emit_bunches(directory: Path) -> int:
    paths = write_bunches(directory)
    print("wrote {} bunch files to {}".format(len(paths), directory))
    for key, path in paths.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print("  {}  {}  {}".format(key, path.name, digest))
    print("EMIT_BUNCHES=PASS")
    return 0


# ---------------------------------------------------------------------------
# Main gate
# ---------------------------------------------------------------------------


def write_json(path: Optional[Path], payload: dict) -> None:
    if path is None:
        return
    path = path.expanduser()
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gate_stationary_ab.py",
        description="Pre-matrix guard: stationary versus moving gamma-ratio A/B "
                    "for the velocity-binned deposit.",
        epilog="The measurement driver's velocity-moment off switch is "
               "'--moments 0' (default off).  '--velocity-moments off' is a "
               "picBunchApp flag and is not accepted by "
               "picBunchVelocityBinsMeasure.  See the module docstring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--binary", help="path to picBunchVelocityBinsMeasure")
    parser.add_argument("--workdir",
                        help="base directory; outputs go in "
                             "<DIR>/tg_stationary_ab")
    parser.add_argument("--json", type=Path, help="write the verdict JSON here")
    parser.add_argument("--keep", action="store_true",
                        help="keep the auto-created temp work directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="print planned commands and expected ratios; "
                             "does not need the binary")
    parser.add_argument("--self-test", action="store_true",
                        help="exercise the verdict function and bunch/field "
                             "parsing on synthetic data")
    parser.add_argument("--emit-bunches", type=Path, metavar="DIR",
                        help="write the six bunch files and exit (for a "
                             "byte-reproducibility check)")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="per-arm timeout in seconds (default 600)")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()
    if args.emit_bunches is not None:
        return emit_bunches(args.emit_bunches.resolve())
    if args.dry_run:
        return print_dry_run(args)

    run_dir: Optional[Path] = None
    keep = False
    payload: dict = {
        "gate": "TG-stationary-vs-moving-gamma-ratio",
        "verdict": "ERROR",
        "binary": args.binary,
        "workdir": None,
        "thresholds": {
            "gamma_relative_tolerance": GAMMA_REL_TOL,
            "collapse_ratio": COLLAPSE_RATIO,
            "unit_ratio": [UNIT_RATIO_LO, UNIT_RATIO_HI],
            "pair_c_ratio": [C_RATIO_LO, C_RATIO_HI],
        },
        "components": {
            "order": list(COMPONENT_NAMES),
            "transverse": ["Ex", "Ey"],
            "longitudinal": "Ez",
            "motion_axis": "+z",
        },
        "pairs": {},
    }

    try:
        if not args.binary:
            raise GateError("--binary is required unless --dry-run or "
                            "--self-test is used")
        binary_path = Path(args.binary).expanduser().resolve()
        binary = str(binary_path)
        if not binary_path.is_file():
            raise GateError("binary not found: " + binary)
        if not os.access(binary_path, os.X_OK):
            raise GateError("binary is not executable: " + binary)

        if args.workdir:
            run_dir = Path(args.workdir).expanduser().resolve() / "tg_stationary_ab"
            run_dir.mkdir(parents=True, exist_ok=True)
            keep = True
        else:
            run_dir = Path(tempfile.mkdtemp(prefix="tg_stationary_ab_"))
            keep = args.keep
        payload["workdir"] = str(run_dir)

        paths = write_bunches(run_dir)
        arms = {
            tag: run_arm(binary, paths[tag], tag, run_dir, args.timeout)
            for tag in BUNCH_FILES
        }

        pair_a = make_pair(
            "A", (0.0, 0.0, U_A),
            arms["A_stationary"], arms["A_moving"],
            "TSC-corrected same-node |Ex,Ey| ratio")
        pair_b = make_pair(
            "B", (0.0, 0.0, U_B),
            arms["B_stationary"], arms["B_moving"],
            "TSC-corrected same-node |Ex,Ey| ratio")
        pair_c = make_pair(
            "C", case32_representative_u(),
            arms["C_stationary"], arms["C_moving"], "L2(|Ex,Ey|) ratio")

        apply_tsc_correction(pair_a, pair_a_moving()[0])
        apply_tsc_correction(pair_b, pair_b_moving()[0])
        # Pair C is a multi-particle sanity check; its verdict uses the L2
        # ratio, which does not need the single-particle TSC correction.
        pair_c.ratio = pair_c.l2_ratio

        pair_a.label, pair_a.reason = classify_ab(
            pair_a.ratio, pair_a.gamma, pair_a.stationary.field.finite
            and pair_a.moving.field.finite)
        pair_b.label, pair_b.reason = classify_ab(
            pair_b.ratio, pair_b.gamma, pair_b.stationary.field.finite
            and pair_b.moving.field.finite)
        pair_c.label, pair_c.reason = classify_c(
            pair_c.l2_ratio, pair_c.stationary.field.finite
            and pair_c.moving.field.finite)

        verdict = combine_verdict(
            [pair_a.label, pair_b.label, pair_c.label])
        payload["verdict"] = verdict
        payload["pairs"] = {
            "A": pair_a.summary(),
            "B": pair_b.summary(),
            "C": pair_c.summary(),
        }
        payload["commands"] = {
            tag: arm.command for tag, arm in arms.items()
        }
        payload["notes"] = [
            "Pair C uses the L2 ratio of (Ex,Ey); its particles span a "
            "small gamma range around 1101.",
            "Pair A/B 'ratio' is the same-node raw ratio divided by the "
            "TSC z-plane fraction; both raw numbers and the fraction are in "
            "the pair dictionaries.",
            "The measurement driver prints no peak field; max|E| is read from "
            "the documented 6-component field binary.",
            "The driver's off switch is --moments 0, not --velocity-moments off.",
        ]

        for pair in (pair_a, pair_b, pair_c):
            print("PAIR {}: u_moving={} gamma={:.17g} ratio={:.6e} "
                  "node_raw={:.6e} L2={:.6e}".format(
                      pair.name, pair.u_moving, pair.gamma, pair.ratio,
                      pair.ratio_node_raw, pair.l2_ratio))
            print("  max|Ex,Ey| stationary={:.6e} moving={:.6e} "
                  "peak_raw={:.6e} node_raw={:.6e} tsc_z0={}".format(
                      pair.stationary.field.max_trans,
                      pair.moving.field.max_trans,
                      pair.ratio_peak_raw, pair.ratio_node_raw,
                      "{:.6f}".format(pair.tsc_z0_fraction)
                      if pair.tsc_z0_fraction is not None else "n/a"))
            if pair.name in ("A", "B"):
                print("  corrected_ratio={:.6e} relative_error={:.3e}".format(
                    pair.ratio, abs(pair.ratio - pair.gamma) / pair.gamma))
            print("  -> {}: {}".format(pair.label or "PASS", pair.reason))

        print("VERDICT: {}".format(verdict))
        write_json(args.json, payload)
        if keep and run_dir is not None:
            print("WORKDIR_KEPT={}".format(run_dir))
        print("GATE_VERDICT={}".format(verdict))
        if verdict != "PASS":
            return 1
        return 0

    except GateError as exc:
        payload["verdict"] = "ERROR"
        payload["error"] = str(exc)
        payload["commands"] = [exc.command] if exc.command else []
        payload["stderr"] = exc.stderr
        print("ERROR: {}".format(exc))
        if exc.command:
            print("COMMAND: {}".format(" ".join(exc.command)))
        if exc.stderr:
            print("STDERR:")
            print(exc.stderr.rstrip())
        print("VERDICT: ERROR")
        write_json(args.json, payload)
        if keep and run_dir is not None:
            print("WORKDIR_KEPT={}".format(run_dir))
        print("GATE_VERDICT=ERROR")
        return 2

    except Exception as exc:  # noqa: BLE001 - the gate must emit ERROR, not a traceback
        payload["verdict"] = "ERROR"
        payload["error"] = "{}: {}".format(type(exc).__name__, exc)
        print("ERROR: {}: {}".format(type(exc).__name__, exc))
        print("VERDICT: ERROR")
        write_json(args.json, payload)
        if keep and run_dir is not None:
            print("WORKDIR_KEPT={}".format(run_dir))
        print("GATE_VERDICT=ERROR")
        return 2

    finally:
        if run_dir is not None and not keep:
            shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
