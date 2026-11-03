#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
"""Independent openPMD/HDF5 consumer check.

Runs picBunchWriteSample, then opens the resulting file with h5py only.  The
point is that no openPMD-api code participates in the read: a wrong HDF5 layout,
axis order, attribute name or unit would survive the C++ round-trip test but
fail here.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import h5py
import numpy as np

NX, NY, NZ = 4, 5, 6
SPACING_ZYX = np.array([2.5, 1.5, 0.5])
OFFSET_ZYX = np.array([3.0, 2.0, 1.0])
POSITIONS = {
    ("E", "x"): [0.0, 0.0, 0.5],
    ("E", "y"): [0.0, 0.5, 0.0],
    ("E", "z"): [0.5, 0.0, 0.0],
    ("B", "x"): [0.5, 0.5, 0.0],
    ("B", "y"): [0.5, 0.0, 0.5],
    ("B", "z"): [0.0, 0.5, 0.5],
}
UNIT_DIMENSION = {
    "E": [1.0, 1.0, -3.0, -1.0, 0.0, 0.0, 0.0],
    "B": [0.0, 1.0, -2.0, -1.0, 0.0, 0.0, 0.0],
}


def expected(component: int, i: int, j: int, k: int) -> float:
    flat = i + NX * (j + NY * k)
    return 0.125 * component + 1.0e-3 * flat + 1.0e-12 * (component * 7 + flat)


def fail(message: str) -> None:
    raise AssertionError(message)


def decode(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.bytes_):
        return value.tobytes().decode()
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--writer", required=True, help="picBunchWriteSample executable")
    parser.add_argument("--workdir", default="/tmp", help="directory for the generated file")
    args = parser.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    path = os.path.join(args.workdir, f"picbunch_openpmd_check_{os.getpid()}.h5")
    if os.path.exists(path):
        os.remove(path)

    try:
        subprocess.run([args.writer, path], check=True)
        with h5py.File(path, "r") as handle:
            root = handle.attrs
            if "openPMD" not in root:
                fail("missing series attribute openPMD")
            if decode(root["iterationEncoding"]) != "groupBased":
                fail(f"iterationEncoding is {decode(root['iterationEncoding'])!r}")
            if decode(root["meshesPath"]) != "meshes/":
                fail(f"meshesPath is {decode(root['meshesPath'])!r}")
            base = decode(root["basePath"])
            iteration_path = base.replace("%T", "7").lstrip("/")
            if iteration_path not in handle:
                fail(f"missing iteration group {iteration_path!r}")
            iteration = handle[iteration_path]

            if abs(float(iteration.attrs["time"]) - 2.5e-13) > 0.0:
                fail(f"iteration time is {iteration.attrs['time']!r}")
            if float(iteration.attrs["timeUnitSI"]) != 1.0:
                fail("iteration timeUnitSI is not 1.0")
            if float(iteration.attrs["timeOffset"]) != 0.0:
                fail("iteration timeOffset is not 0.0")

            for record, first in (("E", 0), ("B", 3)):
                if record not in iteration["meshes"]:
                    fail(f"missing mesh record {record}")
                mesh = iteration["meshes"][record]
                if [decode(v) for v in mesh.attrs["axisLabels"]] != ["z", "y", "x"]:
                    fail(f"{record}: axisLabels are {mesh.attrs['axisLabels']!r}")
                if decode(mesh.attrs["dataOrder"]) != "C":
                    fail(f"{record}: dataOrder is {decode(mesh.attrs['dataOrder'])!r}")
                if decode(mesh.attrs["geometry"]) != "cartesian":
                    fail(f"{record}: geometry is {decode(mesh.attrs['geometry'])!r}")
                if not np.allclose(mesh.attrs["gridSpacing"], SPACING_ZYX, rtol=0.0, atol=0.0):
                    fail(f"{record}: gridSpacing is {mesh.attrs['gridSpacing']!r}")
                if not np.allclose(mesh.attrs["gridGlobalOffset"], OFFSET_ZYX, rtol=0.0, atol=0.0):
                    fail(f"{record}: gridGlobalOffset is {mesh.attrs['gridGlobalOffset']!r}")
                if float(mesh.attrs["gridUnitSI"]) != 1.0:
                    fail(f"{record}: gridUnitSI is {mesh.attrs['gridUnitSI']!r}")
                if not np.allclose(
                    mesh.attrs["unitDimension"], UNIT_DIMENSION[record], rtol=0.0, atol=0.0
                ):
                    fail(f"{record}: unitDimension is {mesh.attrs['unitDimension']!r}")

                for axis in "xyz":
                    if axis not in mesh:
                        fail(f"{record}: missing component {axis}")
                    dataset = mesh[axis]
                    if dataset.shape != (NZ, NY, NX):
                        fail(f"{record}/{axis}: shape is {dataset.shape!r}, expected {(NZ, NY, NX)}")
                    if float(dataset.attrs["unitSI"]) != 1.0:
                        fail(f"{record}/{axis}: unitSI is {dataset.attrs['unitSI']!r}")
                    position = [float(v) for v in dataset.attrs["position"]]
                    if position != POSITIONS[(record, axis)]:
                        fail(f"{record}/{axis}: position is {position!r}")
                    component = first + "xyz".index(axis)
                    reference = np.array(
                        [
                            [[expected(component, i, j, k) for i in range(NX)] for j in range(NY)]
                            for k in range(NZ)
                        ]
                    )
                    stored = dataset[()]
                    if not np.array_equal(stored, reference):
                        worst = float(np.max(np.abs(stored - reference)))
                        fail(f"{record}/{axis}: values differ, max|diff|={worst:.3e}")

        leftovers = [
            name
            for name in os.listdir(args.workdir)
            if ".picbunch-tmp-" in name and name.startswith(os.path.basename(path))
        ]
        if leftovers:
            fail(f"writer left temporary files: {leftovers}")
    finally:
        if os.path.exists(path):
            os.remove(path)

    print("openPMD/HDF5 independent consumer check: OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as error:
        print(f"check_openpmd: FAIL: {error}", file=sys.stderr)
        sys.exit(1)
