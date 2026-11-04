#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
"""Create an openPMD 2.0 particle file whose scalar records are groups.

openPMD-api 0.17 does not write this layout on demand, so the reader test uses
h5py to build the two forms that exist in the wild:

* ``weighting`` is a group with a child dataset named ``value`` (the standard
  scalar-record representation);
* ``charge`` is a group with a child dataset named after the record (the form
  PIConGPU writes), and ``mass`` uses ``value`` again.

Attributes live on the record group, or on the child for ``charge``, to cover
both attribute locations the reader must accept.
"""

import argparse
import h5py
import numpy as np


def string_attribute(group, name, value):
    group.attrs[name] = np.bytes_(value)


def scalar_group(parent, name, child, values, unit_si, unit_dimension,
                 macro_weighted, weighting_power, unit_on_child=False):
    record = parent.create_group(name)
    record.attrs["macroWeighted"] = np.int32(macro_weighted)
    record.attrs["weightingPower"] = np.float64(weighting_power)
    record.attrs["unitDimension"] = np.asarray(unit_dimension, dtype=np.float64)
    record.attrs["timeOffset"] = np.float64(0.0)
    record.attrs["unitSI"] = np.float64(unit_si)
    dataset = record.create_dataset(child, data=np.asarray(values, dtype=np.float64))
    dataset.attrs["unitSI"] = np.float64(unit_si)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    args = parser.parse_args()

    with h5py.File(args.output, "w") as handle:
        string_attribute(handle, "openPMD", "2.0.0")
        string_attribute(handle, "iterationEncoding", "groupBased")
        string_attribute(handle, "iterationFormat", "%T")
        string_attribute(handle, "basePath", "/data/%T/")
        string_attribute(handle, "meshesPath", "meshes/")
        string_attribute(handle, "particlesPath", "particles/")
        string_attribute(handle, "software", "picBunch scalar-record fixture")
        string_attribute(handle, "softwareVersion", "0.1.0")
        handle.attrs["openPMDextension"] = np.uint32(1)

        iteration = handle.create_group("data/0")
        iteration.attrs["dt"] = np.float64(1.0)
        iteration.attrs["time"] = np.float64(0.0)
        iteration.attrs["timeUnitSI"] = np.float64(1.0)

        species = iteration.create_group("particles/electrons")
        species.attrs["numParticles"] = np.int32(2)

        length_dimension = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        momentum_dimension = [1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0]
        mass_dimension = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        charge_dimension = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        dimensionless = [0.0] * 7

        position = species.create_group("position")
        position.attrs["macroWeighted"] = np.int32(0)
        position.attrs["weightingPower"] = np.float64(0.0)
        position.attrs["unitDimension"] = np.asarray(length_dimension)
        position.attrs["timeOffset"] = np.float64(0.0)
        for axis, values in {"x": [1.0, 2.0], "y": [3.0, 4.0], "z": [5.0, 6.0]}.items():
            dataset = position.create_dataset(axis, data=np.asarray(values, dtype=np.float64))
            dataset.attrs["unitSI"] = np.float64(1.0e-6)

        momentum = species.create_group("momentum")
        momentum.attrs["macroWeighted"] = np.int32(0)
        momentum.attrs["weightingPower"] = np.float64(0.0)
        momentum.attrs["unitDimension"] = np.asarray(momentum_dimension)
        momentum.attrs["timeOffset"] = np.float64(0.0)
        electron_mass = 9.1093837015e-31
        speed_of_light = 299792458.0
        momentum_unit = electron_mass * speed_of_light
        for axis, values in {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [1.0, 2.0]}.items():
            dataset = momentum.create_dataset(axis, data=np.asarray(values, dtype=np.float64))
            dataset.attrs["unitSI"] = np.float64(momentum_unit)

        offset = species.create_group("positionOffset")
        offset.attrs["macroWeighted"] = np.int32(0)
        offset.attrs["weightingPower"] = np.float64(0.0)
        offset.attrs["unitDimension"] = np.asarray(length_dimension)
        offset.attrs["timeOffset"] = np.float64(0.0)
        # Both record and component carry unitSI, as openPMD requires.
        offset.attrs["unitSI"] = np.float64(1.0e-7)
        for axis, values in {"x": [7.0, 8.0], "y": [9.0, 10.0], "z": [11.0, 12.0]}.items():
            dataset = offset.create_dataset(axis, data=np.asarray(values, dtype=np.float64))
            dataset.attrs["unitSI"] = np.float64(1.0e-7)

        scalar_group(species, "weighting", "value", [2.0, 3.0], 1.0,
                     dimensionless, 1, 0.0)
        scalar_group(species, "charge", "charge", [-2.0e-19, -3.0e-19], 1.0,
                     charge_dimension, 0, 1.0, unit_on_child=True)
        scalar_group(species, "mass", "value", [electron_mass, electron_mass], 1.0,
                     mass_dimension, 0, 0.0)


if __name__ == "__main__":
    main()
