// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Deterministic PIConGPU-restart sample used by
// tools/verify/check_picongpu_restart.py.  It builds a synthetic particle input
// with patch metadata and writes a checkpoint whose stored (internal) field
// values are known constants: E/unitE and B/unitB per component.

#include "picBunch/OpenPMDWriter.hpp"
#include "picBunch/Types.hpp"

#include <hdf5.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>

namespace
{
    hid_t create(hid_t loc, char const* name, hid_t type, hsize_t count)
    {
        hid_t space = H5Screate_simple(1, &count, nullptr);
        hid_t dataset = H5Dcreate2(loc, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Sclose(space);
        return dataset;
    }

    void writeStringAttribute(hid_t object, char const* name, std::string const& value)
    {
        hid_t type = H5Tcopy(H5T_C_S1);
        H5Tset_size(type, H5T_VARIABLE);
        hid_t space = H5Screate(H5S_SCALAR);
        hid_t attribute = H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
        char const* pointer = value.c_str();
        H5Awrite(attribute, type, &pointer);
        H5Aclose(attribute);
        H5Sclose(space);
        H5Tclose(type);
    }

    void writeDoubleAttribute(hid_t object, char const* name, double value)
    {
        hid_t space = H5Screate(H5S_SCALAR);
        hid_t attribute = H5Acreate2(object, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attribute, H5T_NATIVE_DOUBLE, &value);
        H5Aclose(attribute);
        H5Sclose(space);
    }

    void writeUInt32Attribute(hid_t object, char const* name, uint32_t value)
    {
        hid_t space = H5Screate(H5S_SCALAR);
        hid_t attribute = H5Acreate2(object, name, H5T_NATIVE_UINT32, space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attribute, H5T_NATIVE_UINT32, &value);
        H5Aclose(attribute);
        H5Sclose(space);
    }

    void writeParticleInput(std::filesystem::path const& path)
    {
        hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        writeStringAttribute(file, "openPMD", "2.0.0");
        writeStringAttribute(file, "iterationEncoding", "groupBased");
        writeStringAttribute(file, "meshesPath", "meshes/");
        writeStringAttribute(file, "particlesPath", "particles/");
        writeStringAttribute(file, "basePath", "/data/%T/");
        writeStringAttribute(file, "iterationFormat", "%T");
        writeStringAttribute(file, "software", "picBunch-sample");
        writeStringAttribute(file, "softwareVersion", "0.1.0");
        writeUInt32Attribute(file, "openPMDextension", 1);
        H5Gcreate2(file, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t iteration = H5Gcreate2(file, "data/0", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        writeDoubleAttribute(iteration, "dt", 1.0);
        writeDoubleAttribute(iteration, "time", 0.0);
        writeDoubleAttribute(iteration, "timeUnitSI", 1.0);
        hid_t particles = H5Gcreate2(iteration, "particles", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t species = H5Gcreate2(particles, "electrons", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        writeStringAttribute(species, "sourceMarker", "sample");
        for(char const* record : {"position", "momentum", "positionOffset"})
        {
            hid_t group = H5Gcreate2(species, record, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            for(char const* axis : {"x", "y", "z"})
            {
                bool const integer = std::string(record) == "positionOffset";
                hid_t dataset = create(
                    group, axis, integer ? H5T_STD_I32LE : H5T_IEEE_F32LE, 1);
                if(integer)
                {
                    int32_t const value = 0;
                    H5Dwrite(dataset, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
                }
                else
                {
                    float const value = std::string(record) == "position" ? 0.5f : 0.0f;
                    H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
                }
                H5Dclose(dataset);
            }
            H5Gclose(group);
        }
        hid_t weighting = create(species, "weighting", H5T_IEEE_F32LE, 1);
        float const weight = 1.0f;
        H5Dwrite(weighting, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &weight);
        H5Dclose(weighting);

        hid_t patches = H5Gcreate2(species, "particlePatches", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t count = create(patches, "numParticles", H5T_STD_U64LE, 1);
        uint64_t const one = 1;
        H5Dwrite(count, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, &one);
        H5Dclose(count);
        hid_t offsetCount = create(patches, "numParticlesOffset", H5T_STD_U64LE, 1);
        uint64_t const zero = 0;
        H5Dwrite(offsetCount, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, &zero);
        H5Dclose(offsetCount);
        H5Gclose(patches);

        H5Gclose(species);
        H5Gclose(particles);
        H5Gclose(iteration);
        H5Fclose(file);
    }
} // namespace

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::fprintf(stderr, "usage: %s <workdir>\n", argv[0]);
        return 2;
    }
    std::filesystem::path const workdir(argv[1]);
    std::filesystem::path const directory = workdir / "checkpoints";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(workdir);
    std::filesystem::path const input = workdir / "restart_sample_input.h5";
    writeParticleInput(input);

    double const unitLength = 5.7735e-8;
    double const unitE = 8.850761452125e12;
    double const unitB = 29522.962355927582;
    std::array<double, 3> const storedE{0.25, 0.5, -0.75};
    std::array<double, 3> const storedB{0.125, -0.25, 0.5};

    picBunch::Field field;
    field.grid.n = {4, 5, 6};
    field.grid.spacing = {0.5, 1.5, 2.5};
    field.grid.origin = {1.0, 2.0, 3.0};
    field.grid.fftPadding = 2;
    for(int c = 0; c < 6; ++c)
    {
        double const unit = (c < 3) ? unitE : unitB;
        double const stored = (c < 3) ? storedE[static_cast<std::size_t>(c)]
                                      : storedB[static_cast<std::size_t>(c - 3)];
        field.data[c].assign(field.grid.size(), stored * unit);
    }

    picBunch::PicOnGPUWriteOptions options;
    options.step = 2;
    options.directory = directory.string();
    options.inputFile = input.string();
    options.inputIteration = 0;
    options.species = "electrons";
    options.time = 2.5e-13;
    options.timeUnitSI = 1.0;
    options.dt = 1.0e-15;
    options.unitLength = unitLength;
    options.unitEField = unitE;
    options.unitBField = unitB;
    options.nextId = 1;
    try
    {
        picBunch::writePicOnGPUCheckpoint(field, options);
    }
    catch(std::exception const& error)
    {
        std::fprintf(stderr, "write restart sample: %s\n", error.what());
        return 1;
    }
    return 0;
}
