// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group: convolutional PML auxiliary state of the PIConGPU checkpoint writer.
// With absorber = "pml" the writer adds the eight slab records a PIConGPU PML
// deck loads on restart (zero initial state, main-field geometry, parent
// units and Yee positions); the default and "exponential" keep the previous
// tree.  Verified against the raw HDF5 layout.

#include "picBunch/Compute.hpp"
#include "picBunch/Config.hpp"
#include "picBunch/OpenPMD.hpp"
#include "picBunch/OpenPMDWriter.hpp"
#include "picBunch/Types.hpp"

#include <hdf5.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

namespace
{
    std::filesystem::path temporaryDirectory()
    {
        return std::filesystem::temp_directory_path()
            / ("picbunch_pml_" + std::to_string(static_cast<long long>(getpid())));
    }

    picBunch::Field makeField()
    {
        picBunch::Field field;
        field.grid.n = {4, 5, 6};
        field.grid.spacing = {0.5, 1.5, 2.5};
        field.grid.origin = {1.0, 2.0, 3.0};
        field.grid.fftPadding = 2;
        for(int c = 0; c < 6; ++c)
        {
            field.data[c].resize(field.grid.size());
            for(std::size_t l = 0; l < field.grid.size(); ++l)
                field.data[c][l] = 0.0625 * static_cast<double>(c);
        }
        return field;
    }

    void writeStringAttr(hid_t object, char const* name, std::string const& value)
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

    void writeDoubleAttr(hid_t object, char const* name, double value)
    {
        hid_t space = H5Screate(H5S_SCALAR);
        hid_t attribute = H5Acreate2(object, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attribute, H5T_NATIVE_DOUBLE, &value);
        H5Aclose(attribute);
        H5Sclose(space);
    }

    hid_t createDataset(hid_t loc, char const* name, hid_t type, hsize_t count)
    {
        hid_t space = H5Screate_simple(1, &count, nullptr);
        hid_t dataset = H5Dcreate2(loc, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Sclose(space);
        return dataset;
    }

    //! Particle input in the exact format that the openPMD writer reads back
    //! (mirrors tests/write_restart_sample.cpp): series attributes, one species
    //! with position, momentum, positionOffset groups and a scalar weighting,
    //! plus minimal particlePatches metadata.
    void writeParticleInput(std::filesystem::path const& path)
    {
        hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        writeStringAttr(file, "openPMD", "2.0.0");
        writeStringAttr(file, "iterationEncoding", "groupBased");
        writeStringAttr(file, "meshesPath", "meshes/");
        writeStringAttr(file, "particlesPath", "particles/");
        writeStringAttr(file, "basePath", "/data/%T/");
        writeStringAttr(file, "iterationFormat", "%T");
        writeDoubleAttr(file, "openPMDextension", 1.0);
        H5Gcreate2(file, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t iteration = H5Gcreate2(file, "data/0", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        writeDoubleAttr(iteration, "dt", 1.0e-15);
        writeDoubleAttr(iteration, "time", 0.0);
        writeDoubleAttr(iteration, "timeUnitSI", 1.0);
        hid_t particles = H5Gcreate2(iteration, "particles", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t species = H5Gcreate2(particles, "electrons", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        writeStringAttr(species, "sourceMarker", "pml-test");
        for(char const* record : {"position", "momentum", "positionOffset"})
        {
            hid_t group = H5Gcreate2(species, record, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            for(char const* axis : {"x", "y", "z"})
            {
                bool const integer = std::string(record) == "positionOffset";
                hid_t dataset = createDataset(
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
        hid_t weighting = createDataset(species, "weighting", H5T_IEEE_F32LE, 1);
        float const weight = 1.0f;
        H5Dwrite(weighting, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &weight);
        H5Dclose(weighting);
        hid_t patches = H5Gcreate2(species, "particlePatches", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t count = createDataset(patches, "numParticles", H5T_STD_U64LE, 1);
        std::uint64_t const one = 1;
        H5Dwrite(count, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, &one);
        H5Dclose(count);
        hid_t offsetCount = createDataset(patches, "numParticlesOffset", H5T_STD_U64LE, 1);
        std::uint64_t const zero = 0;
        H5Dwrite(offsetCount, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, &zero);
        H5Dclose(offsetCount);
        H5Gclose(patches);
        H5Gclose(species);
        H5Gclose(particles);
        H5Gclose(iteration);
        H5Fclose(file);
    }

    std::vector<double> readDoubleArrayAttribute(hid_t object, char const* name)
    {
        std::vector<double> values;
        hid_t attribute = H5Aopen(object, name, H5P_DEFAULT);
        if(attribute < 0)
            return values;
        hid_t space = H5Aget_space(attribute);
        hsize_t maxdims[8] = {0};
        int const rank = H5Sget_simple_extent_dims(space, maxdims, nullptr);
        hsize_t count = 1;
        for(int r = 0; r < rank; ++r)
            count *= maxdims[r];
        values.resize(count);
        H5Aread(attribute, H5T_NATIVE_DOUBLE, values.data());
        H5Aclose(attribute);
        H5Sclose(space);
        return values;
    }

    template<typename T>
    T readScalarAttribute(hid_t object, char const* name, hid_t type)
    {
        T value{};
        hid_t attribute = H5Aopen(object, name, H5P_DEFAULT);
        if(attribute < 0)
            return value;
        H5Aread(attribute, type, &value);
        H5Aclose(attribute);
        return value;
    }

    std::vector<float> readFloatDataset(hid_t file, char const* path)
    {
        hid_t dataset = H5Dopen2(file, path, H5P_DEFAULT);
        if(dataset < 0)
            return {};
        hid_t space = H5Dget_space(dataset);
        hsize_t maxdims[8] = {0};
        int const rank = H5Sget_simple_extent_dims(space, maxdims, nullptr);
        hsize_t count = 1;
        for(int r = 0; r < rank; ++r)
            count *= maxdims[r];
        std::vector<float> values(count);
        H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
        H5Sclose(space);
        H5Dclose(dataset);
        return values;
    }
} // namespace

// ---------------------------------------------------------------------------
// Convolutional PML auxiliary state: absorber = "pml" writes the eight slab
// records a PIConGPU PML deck loads on restart; the default and "exponential"
// keep the previous tree.

TEST(Restart, PmlAbsorberWritesZeroSlabs)
{
    auto const directory = temporaryDirectory() / "checkpoints_pml_write";
    std::filesystem::remove_all(temporaryDirectory());
    std::filesystem::create_directories(temporaryDirectory());
    auto const input = temporaryDirectory() / "input.h5";
    writeParticleInput(input);

    auto const field = makeField();
    picBunch::PicOnGPUWriteOptions options;
    options.step = 2;
    options.directory = directory.string();
    options.inputFile = input.string();
    options.inputIteration = 0;

    options.time = 2.5e-13;
    options.timeUnitSI = 1.0;
    options.dt = 1.0e-15;
    options.unitLength = 1.0;
    options.unitEField = 2.0;
    options.unitBField = 4.0;
    options.nextId = 1;
    options.absorber = "pml";
    picBunch::writePicOnGPUCheckpointOpenPMD(field, options);

    auto const path = directory / "checkpoint_000002.h5";
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    ASSERT_GE(file, 0);
    char const* const slabs[6] = {"xneg", "xpos", "yneg", "ypos", "zneg", "zpos"};
    char const* const components[6] = {"xy", "xz", "yx", "yz", "zx", "zy"};
    double const expectedUnit[2] = {2.0, 4.0};
    double const expectedUnitDimension[2][7] = {
        {1.0, 1.0, -3.0, -1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, -2.0, -1.0, 0.0, 0.0, 0.0}};
    // Parent Yee positions per field: the PML cross components inherit the
    // parent component's Yee offset (x -> xy,xz; y -> yx,yz; z -> zx,zy).
    // E: Ex={0.5,0,0} Ey={0,0.5,0} Ez={0,0,0.5}
    // B: Bx={0,0.5,0.5} By={0.5,0,0.5} Bz={0.5,0.5,0}
    double const parentPosition[2][3][3] = {
        {{0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}, {0.0, 0.0, 0.5}},
        {{0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}}};
    std::uint64_t const expectedPoints = field.grid.size();
    for(int r = 0; r < 2; ++r)
    {
        char const* const record =
            r == 0 ? "Convolutional PML E" : "Convolutional PML B";
        for(char const* slab : slabs)
        {
            std::string const recordPath =
                std::string("data/2/fields/") + record + "_" + slab;
            hid_t group = H5Gopen2(file, recordPath.c_str(), H5P_DEFAULT);
            ASSERT_GE(group, 0) << recordPath;
            // The slab adopts the main-field geometry.
            EXPECT_EQ(
                readDoubleArrayAttribute(group, "gridSpacing"),
                (std::vector<double>{
                    field.grid.spacing[2], field.grid.spacing[1],
                    field.grid.spacing[0]}));
            EXPECT_EQ(
                readDoubleArrayAttribute(group, "gridGlobalOffset"),
                (std::vector<double>{
                    field.grid.origin[2], field.grid.origin[1],
                    field.grid.origin[0]}));
            EXPECT_DOUBLE_EQ(
                readScalarAttribute<double>(group, "gridUnitSI", H5T_NATIVE_DOUBLE),
                1.0);
            auto const unitDimension = readDoubleArrayAttribute(group, "unitDimension");
            ASSERT_EQ(unitDimension.size(), 7u);
            for(std::size_t a = 0; a < 7; ++a)
                EXPECT_DOUBLE_EQ(unitDimension[a], expectedUnitDimension[r][a])
                    << recordPath;
            H5Gclose(group);

            for(int c = 0; c < 6; ++c)
            {
                std::string const componentPath = recordPath + "/" + components[c];
                // Read-back: every node of every slab component is zero.
                auto const values = readFloatDataset(file, componentPath.c_str());
                ASSERT_EQ(values.size(), expectedPoints) << componentPath;
                for(float const v : values)
                    EXPECT_EQ(v, 0.0f) << componentPath;
                hid_t dataset = H5Dopen2(file, componentPath.c_str(), H5P_DEFAULT);
                ASSERT_GE(dataset, 0);
                hid_t space = H5Dget_space(dataset);
                hsize_t dimensions[3] = {0, 0, 0};
                int const rank = H5Sget_simple_extent_dims(space, dimensions, nullptr);
                EXPECT_EQ(rank, 3);
                EXPECT_EQ(dimensions[0], field.grid.n[2]);
                EXPECT_EQ(dimensions[1], field.grid.n[1]);
                EXPECT_EQ(dimensions[2], field.grid.n[0]);
                H5Sclose(space);
                hid_t type = H5Dget_type(dataset);
                EXPECT_EQ(H5Tget_class(type), H5T_FLOAT);
                EXPECT_EQ(H5Tget_size(type), 4u);
                H5Tclose(type);
                EXPECT_DOUBLE_EQ(
                    readScalarAttribute<double>(dataset, "unitSI", H5T_NATIVE_DOUBLE),
                    expectedUnit[r]);
                auto const position = readDoubleArrayAttribute(dataset, "position");
                ASSERT_EQ(position.size(), 3u);
                for(std::size_t a = 0; a < 3; ++a)
                    EXPECT_DOUBLE_EQ(position[a], parentPosition[r][c / 2][a])
                        << componentPath;
                H5Dclose(dataset);
            }
        }
    }
    // Exactly 8 slab records: 6 existing (E, B, picongpu_idProvider) + 2 E/B
    // record groups are split per slab, so count the Convolutional PML groups.
    EXPECT_EQ(H5Lexists(file, "data/2/fields/Convolutional PML E_xneg", H5P_DEFAULT), 1);
    EXPECT_EQ(H5Lexists(file, "data/2/fields/Convolutional PML B_zpos", H5P_DEFAULT), 1);
    H5Fclose(file);
    std::filesystem::remove_all(temporaryDirectory());
}

TEST(Restart, PmlAbsorberDefaultLeavesTreeUnchanged)
{
    for(char const* absorber : {"none", "exponential"})
    {
        auto const directory = temporaryDirectory() / "checkpoints_pml_default";
        std::filesystem::remove_all(temporaryDirectory());
        std::filesystem::create_directories(temporaryDirectory());
        auto const input = temporaryDirectory() / "input.h5";
        writeParticleInput(input);

        auto const field = makeField();
        picBunch::PicOnGPUWriteOptions options;
        options.step = 2;
        options.directory = directory.string();
        options.inputFile = input.string();
        options.inputIteration = 0;
    
        options.time = 2.5e-13;
        options.timeUnitSI = 1.0;
        options.dt = 1.0e-15;
        options.unitLength = 1.0;
        options.unitEField = 1.0;
        options.unitBField = 1.0;
        options.nextId = 1;
        options.absorber = absorber;
        picBunch::writePicOnGPUCheckpointOpenPMD(field, options);

        auto const path = directory / "checkpoint_000002.h5";
        hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        ASSERT_GE(file, 0) << absorber;
        char const* const records[8] = {
            "Convolutional PML E_xneg", "Convolutional PML E_xpos",
            "Convolutional PML E_yneg", "Convolutional PML E_ypos",
            "Convolutional PML E_zneg", "Convolutional PML E_zpos",
            "Convolutional PML B_xneg", "Convolutional PML B_zpos"};
        for(char const* record : records)
            EXPECT_EQ(
                H5Lexists(
                    file,
                    (std::string("data/2/fields/") + record).c_str(),
                    H5P_DEFAULT),
                0)
                << absorber << record;
        H5Fclose(file);
        std::filesystem::remove_all(temporaryDirectory());
    }
}

TEST(Restart, PmlAbsorberFloat64UsesDoubleSlabs)
{
    auto const directory = temporaryDirectory() / "checkpoints_pml_f64";
    std::filesystem::remove_all(temporaryDirectory());
    std::filesystem::create_directories(temporaryDirectory());
    auto const input = temporaryDirectory() / "input.h5";
    writeParticleInput(input);

    auto const field = makeField();
    picBunch::PicOnGPUWriteOptions options;
    options.step = 2;
    options.directory = directory.string();
    options.inputFile = input.string();
    options.inputIteration = 0;

    options.time = 2.5e-13;
    options.timeUnitSI = 1.0;
    options.dt = 1.0e-15;
    options.unitLength = 1.0;
    options.unitEField = 1.0;
    options.unitBField = 1.0;
    options.nextId = 1;
    options.absorber = "pml";
    options.float64 = true;
    picBunch::writePicOnGPUCheckpointOpenPMD(field, options);

    auto const path = directory / "checkpoint_000002.h5";
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    ASSERT_GE(file, 0);
    char const* const componentPath
        = "data/2/fields/Convolutional PML B_zpos/zy";
    hid_t dataset = H5Dopen2(file, componentPath, H5P_DEFAULT);
    ASSERT_GE(dataset, 0);
    hid_t type = H5Dget_type(dataset);
    EXPECT_EQ(H5Tget_class(type), H5T_FLOAT);
    EXPECT_EQ(H5Tget_size(type), 8u);
    H5Tclose(type);
    H5Dclose(dataset);
    H5Fclose(file);
    std::filesystem::remove_all(temporaryDirectory());
}
