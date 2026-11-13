#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
"""End-to-end check of the ordered species list, empty species and field sum.

The config path reads one bunch per ``species`` entry, places each with the
same ``[placement]`` block, computes a field per species and writes the sum.
The field is linear in charge, so the two-species checkpoint must equal the sum
of the two single-species checkpoints.  ``empty_species`` must add a valid
zero-particle group without changing the field, and the generated ids must be
disjoint across the list.
"""

import argparse
import os
import subprocess
import sys

import h5py
import numpy as np

CONFIG = """\
[picongpu]
delta_t = 1.0e-15
cell_size = [1.0e-6, 1.0e-6, 1.0e-6]
base_density = 1.0e24
typical_particles_per_cell = 2
step = 0
precision = "float32"
species = {species}
{empty}

[domain]
cells = [16, 16, 16]
origin = [0.0, 0.0, 0.0]
cell_size = [1.0e-6, 1.0e-6, 1.0e-6]

[placement]
patch = "domain"
center = "none"

[accuracy]
tolerance = 1.0e-3
near_shell = "none"
"""

LENGTH = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
MOMENTUM = [1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0]
DIMENSIONLESS = [0.0] * 7
ELECTRON_MASS = 9.1093837015e-31
SPEED_OF_LIGHT = 299792458.0


def string_attribute(group, name, value):
    group.attrs[name] = np.bytes_(value)


def write_species(iteration, name, x0):
    species = iteration.create_group("particles/" + name)
    count = 6
    species.attrs["numParticles"] = np.int64(count)

    def record(parent, record_name, unit_dimension, macro_weighted, weighting_power=0.0):
        group = parent.create_group(record_name)
        group.attrs["unitDimension"] = np.asarray(unit_dimension)
        group.attrs["macroWeighted"] = np.int32(macro_weighted)
        group.attrs["weightingPower"] = np.float64(weighting_power)
        group.attrs["timeOffset"] = np.float64(0.0)
        return group

    xs = np.linspace(x0 - 2.0e-7, x0 + 2.0e-7, count)
    values = {
        "x": xs,
        "y": np.full(count, 8.0e-6) + np.linspace(-1.0e-7, 1.0e-7, count),
        "z": np.full(count, 8.0e-6)}
    position = record(species, "position", LENGTH, 0)
    offset = record(species, "positionOffset", LENGTH, 0)
    for axis, data in values.items():
        dataset = position.create_dataset(axis, data=np.asarray(data, dtype=np.float64))
        dataset.attrs["unitSI"] = np.float64(1.0)
        cell = offset.create_dataset(axis, data=np.zeros(count, dtype=np.float64))
        cell.attrs["unitSI"] = np.float64(1.0e-6)

    momentum = record(species, "momentum", MOMENTUM, 0)
    for axis in "xyz":
        dataset = momentum.create_dataset(axis, data=np.zeros(count, dtype=np.float64))
        dataset.attrs["unitSI"] = np.float64(ELECTRON_MASS * SPEED_OF_LIGHT)

    weighting = record(species, "weighting", DIMENSIONLESS, 1, 1.0)
    dataset = weighting.create_dataset("value", data=np.ones(count, dtype=np.float64))
    dataset.attrs["unitSI"] = np.float64(1.0)


def write_input(path):
    with h5py.File(path, "w") as handle:
        string_attribute(handle, "openPMD", "1.1.0")
        string_attribute(handle, "iterationEncoding", "groupBased")
        string_attribute(handle, "iterationFormat", "%T")
        string_attribute(handle, "basePath", "/data/%T/")
        string_attribute(handle, "meshesPath", "meshes/")
        string_attribute(handle, "particlesPath", "particles/")
        handle.attrs["openPMDextension"] = np.uint32(1)
        iteration = handle.create_group("data/0")
        iteration.attrs["dt"] = np.float64(1.0e-15)
        iteration.attrs["time"] = np.float64(0.0)
        iteration.attrs["timeUnitSI"] = np.float64(1.0)
        write_species(iteration, "s1", 4.0e-6)
        write_species(iteration, "s2", 12.0e-6)


def run(app, workdir, name, input_path, species, empty=()):
    config_path = os.path.join(workdir, name + ".toml")
    with open(config_path, "w") as handle:
        handle.write(CONFIG.format(
            species=list(species),
            empty="empty_species = " + str(list(empty)) if empty else ""))
    destination = os.path.join(workdir, name)
    subprocess.run(
        [app, "--input", input_path, "--destination", destination,
         "--config", config_path, "--velocity-bins", "1"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return os.path.join(destination, "checkpoint_000000.h5")


def fields(path):
    result = {}
    with h5py.File(path, "r") as handle:
        group = handle["data/0/fields"]
        for record in ("E", "B"):
            for component in "xyz":
                result[(record, component)] = group[record][component][...].astype(np.float64)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", required=True)
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()
    workdir = args.workdir
    os.makedirs(workdir, exist_ok=True)

    input_path = os.path.join(workdir, "species_input.h5")
    write_input(input_path)

    first = fields(run(args.app, workdir, "only_s1", input_path, ["s1"]))
    second = fields(run(args.app, workdir, "only_s2", input_path, ["s2"]))
    combined_path = run(args.app, workdir, "both", input_path, ["s1", "s2"], ["e0"])
    combined = fields(combined_path)

    worst = 0.0
    for key in combined:
        peak = np.abs(combined[key]).max()
        difference = np.abs(combined[key] - (first[key] + second[key])).max()
        if peak > 0.0:
            worst = max(worst, difference / peak)
    if worst > 1.0e-6:
        print(f"field sum mismatch: relative difference {worst:.3e}", file=sys.stderr)
        return 1

    with h5py.File(combined_path, "r") as handle:
        particles = handle["data/0/particles"]
        if sorted(particles.keys()) != ["e0", "s1", "s2"]:
            print(f"unexpected species: {sorted(particles.keys())}", file=sys.stderr)
            return 1
        # An empty species is written as a constant record with shape [0]
        # (the PIConGPU/openPMD representation), not as a zero-length dataset.
        # Accept either so the test covers both writer layouts.
        empty_position = particles["e0/position/x"]
        if isinstance(empty_position, h5py.Dataset):
            empty_shape = list(empty_position.shape)
        else:
            empty_shape = np.asarray(empty_position.attrs["shape"]).tolist()
        if empty_shape != [0]:
            print(f"empty species is not empty: {empty_shape}", file=sys.stderr)
            return 1
        patches = particles["e0/particlePatches/numParticles"][...]
        if patches.tolist() != [0]:
            print(f"empty patch count {patches}", file=sys.stderr)
            return 1
        ids1 = particles["s1/id"][...]
        ids2 = particles["s2/id"][...]
        total = int(handle["data/0/fields/picongpu_idProvider/nextId"][...].ravel()[0])
        merged = np.concatenate([ids1, ids2])
        if len(np.unique(merged)) != len(merged):
            print("generated ids overlap", file=sys.stderr)
            return 1
        if total < len(merged) or ids2.min() < ids1.max() + 1:
            print(f"id range inconsistent: nextId={total}", file=sys.stderr)
            return 1
        print(f"species sum relative difference {worst:.3e}; "
              f"ids s1=[{ids1.min()},{ids1.max()}] s2=[{ids2.min()},{ids2.max()}] "
              f"nextId={total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
