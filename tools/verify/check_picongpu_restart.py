#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
"""Independent h5py-only check of the PIConGPU restart checkpoint writer.

Runs picBunchWriteRestartSample, then reads the produced checkpoint with h5py
only.  No openPMD-api code participates, so a wrong HDF5 layout, attribute name,
axis order, dtype or unit normalization cannot hide behind a C++ round-trip.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import h5py
import numpy as np

STEP = 2
NX, NY, NZ = 4, 5, 6
STEP_UNIT = 5.7735e-8
E_UNIT = 8.850761452125e12
B_UNIT = 29522.962355927582
STORED_E = [0.25, 0.5, -0.75]
STORED_B = [0.125, -0.25, 0.5]
# position attribute is written in x,y,z order, exactly as PIConGPU does.
POSITIONS = {
    ("E", "x"): [0.5, 0.0, 0.0],
    ("E", "y"): [0.0, 0.5, 0.0],
    ("E", "z"): [0.0, 0.0, 0.5],
    ("B", "x"): [0.0, 0.5, 0.5],
    ("B", "y"): [0.5, 0.0, 0.5],
    ("B", "z"): [0.5, 0.5, 0.0],
}
UNIT_DIMENSION = {
    "E": [1.0, 1.0, -3.0, -1.0, 0.0, 0.0, 0.0],
    "B": [0.0, 1.0, -2.0, -1.0, 0.0, 0.0, 0.0],
}


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
    parser.add_argument("--writer", required=True, help="picBunchWriteRestartSample executable")
    parser.add_argument("--workdir", default="/tmp", help="directory for the generated files")
    args = parser.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    subprocess.run([args.writer, args.workdir], check=True)
    directory = os.path.join(args.workdir, "checkpoints")
    path = os.path.join(directory, f"checkpoint_{STEP:06d}.h5")
    if not os.path.isfile(path):
        fail(f"missing checkpoint file {path}")
    if os.path.exists(os.path.join(directory, "checkpoints.txt")):
        with open(os.path.join(directory, "checkpoints.txt"), encoding="utf-8") as handle:
            if handle.read() != f"{STEP}\n":
                fail("checkpoints.txt does not contain exactly the step line")

    try:
        with h5py.File(path, "r") as handle:
            root = handle.attrs
            expected_root = {
                "openPMD": "1.1.0",
                "iterationEncoding": "fileBased",
                "iterationFormat": "checkpoint_%06T",
                "meshesPath": "fields/",
                "particlesPath": "particles/",
            }
            for name, value in expected_root.items():
                if name not in root:
                    fail(f"missing series attribute {name}")
                if decode(root[name]) != value:
                    fail(f"{name} is {decode(root[name])!r}, expected {value!r}")
            if int(root["picongpuIOVersionMajor"]) != 5:
                fail(f"picongpuIOVersionMajor is {root['picongpuIOVersionMajor']!r}")
            if int(root["picongpuIOVersionMinor"]) != 0:
                fail(f"picongpuIOVersionMinor is {root['picongpuIOVersionMinor']!r}")
            if "picBunchRestartWarning" in root:
                fail(f"unexpected restart warning: {decode(root['picBunchRestartWarning'])!r}")

            iteration = handle[f"data/{STEP}"]
            if int(iteration.attrs["iteration"]) != STEP:
                fail(f"iteration is {iteration.attrs['iteration']!r}")
            if int(iteration.attrs["sim_slides"]) != 0:
                fail(f"sim_slides is {iteration.attrs['sim_slides']!r}")
            if abs(float(iteration.attrs["time"]) - 2.5e-13) > 0.0:
                fail(f"time is {iteration.attrs['time']!r}")
            if float(iteration.attrs["timeUnitSI"]) != 1.0:
                fail(f"timeUnitSI is {iteration.attrs['timeUnitSI']!r}")
            if abs(float(iteration.attrs["dt"]) - 1.0e-15) > 0.0:
                fail(f"dt is {iteration.attrs['dt']!r}")

            for record, first, unit, stored in (
                ("E", 0, E_UNIT, STORED_E),
                ("B", 3, B_UNIT, STORED_B),
            ):
                mesh = iteration[f"fields/{record}"]
                if [decode(v) for v in mesh.attrs["axisLabels"]] != ["z", "y", "x"]:
                    fail(f"{record}: axisLabels are {mesh.attrs['axisLabels']!r}")
                if decode(mesh.attrs["dataOrder"]) != "C":
                    fail(f"{record}: dataOrder is {decode(mesh.attrs['dataOrder'])!r}")
                if decode(mesh.attrs["geometry"]) != "cartesian":
                    fail(f"{record}: geometry is {decode(mesh.attrs['geometry'])!r}")
                expected_spacing = np.array([2.5, 1.5, 0.5]) / STEP_UNIT
                if not np.array_equal(mesh.attrs["gridSpacing"], expected_spacing):
                    fail(f"{record}: gridSpacing is {mesh.attrs['gridSpacing']!r}")
                expected_offset = np.array([3.0, 2.0, 1.0]) / STEP_UNIT
                if not np.array_equal(mesh.attrs["gridGlobalOffset"], expected_offset):
                    fail(f"{record}: gridGlobalOffset is {mesh.attrs['gridGlobalOffset']!r}")
                if float(mesh.attrs["gridUnitSI"]) != STEP_UNIT:
                    fail(f"{record}: gridUnitSI is {mesh.attrs['gridUnitSI']!r}")
                if not np.array_equal(
                    mesh.attrs["unitDimension"], UNIT_DIMENSION[record]
                ):
                    fail(f"{record}: unitDimension is {mesh.attrs['unitDimension']!r}")

                for axis in "xyz":
                    dataset = mesh[axis]
                    if dataset.shape != (NZ, NY, NX):
                        fail(f"{record}/{axis}: shape is {dataset.shape!r}")
                    if dataset.dtype != np.float32:
                        fail(f"{record}/{axis}: dtype is {dataset.dtype!r}")
                    if float(dataset.attrs["unitSI"]) != unit:
                        fail(f"{record}/{axis}: unitSI is {dataset.attrs['unitSI']!r}")
                    position = [float(v) for v in dataset.attrs["position"]]
                    if position != POSITIONS[(record, axis)]:
                        fail(f"{record}/{axis}: position is {position!r}")
                    expected = np.float32(stored["xyz".index(axis)])
                    values = dataset[()]
                    if not np.all(values == expected):
                        fail(
                            f"{record}/{axis}: normalization wrong, expected {expected!r}, "
                            f"got range {values.min()!r}..{values.max()!r}"
                        )

            provider = iteration["fields/picongpu_idProvider"]
            for name, expected in (("startId", 0), ("nextId", 1)):
                dataset = provider[name]
                if dataset.dtype != np.uint64:
                    fail(f"idProvider/{name}: dtype is {dataset.dtype!r}")
                if int(dataset[()].reshape(-1)[0]) != expected:
                    fail(f"idProvider/{name} is {dataset[()]!r}")

            species = iteration["particles/electrons"]
            if decode(species.attrs["sourceMarker"]) != "sample":
                fail("particle species attribute was not preserved")
            if "particlePatches" not in species:
                fail("particlePatches was not copied")
            if species["position/x"].shape != (1,):
                fail(f"position/x shape is {species['position/x'].shape!r}")

        leftovers = [
            name
            for name in os.listdir(directory)
            if ".picbunch-tmp-" in name
        ]
        if leftovers:
            fail(f"writer left temporary files: {leftovers}")
    finally:
        pass

    print("PIConGPU restart-format independent consumer check: OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as error:
        print(f"check_picongpu_restart: FAIL: {error}", file=sys.stderr)
        sys.exit(1)
