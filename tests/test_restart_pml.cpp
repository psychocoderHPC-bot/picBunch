// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group: convolutional PML auxiliary state of the PIConGPU checkpoint writer.
// With absorber = "pml" the writer adds the twelve slab records a PIConGPU PML
// deck loads on restart (zero initial state, main-field geometry, parent
// units and Yee positions); only the face rim is stored, while the dataset
// extent stays the full domain.  The default and "exponential" keep the
// previous tree.  Verified against the raw HDF5 layout and the openPMD
// slab-view read path.

#include "picBunch/Compute.hpp"
#include "picBunch/Config.hpp"
#include "picBunch/OpenPMD.hpp"
#include "picBunch/OpenPMDWriter.hpp"
#include "picBunch/Types.hpp"

#include <openPMD/openPMD.hpp>

#include <hdf5.h>

#include <gtest/gtest.h>

#include <algorithm>
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
    //! Larger grid so every face rim leaves a non-empty interior.
    picBunch::Field makeSparseField()
    {
        picBunch::Field field;
        field.grid.n = {24, 20, 16}; // nx, ny, nz
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

    //! PIConGPU PML face geometry for slab 0..5 (xneg..zpos).
    struct TestFace
    {
        std::size_t axis;
        std::size_t begin;
        std::size_t size;
    };

    TestFace testFace(
        std::size_t slab,
        std::size_t thickness,
        std::size_t nz,
        std::size_t ny,
        std::size_t nx)
    {
        std::size_t const axis = slab < 2u ? 2u : (slab < 4u ? 1u : 0u);
        std::size_t const n = axis == 2u ? nx : (axis == 1u ? ny : nz);
        std::size_t const size = std::min(thickness, n);
        return {axis, slab % 2u == 0u ? 0u : n - size, size};
    }

    //! Global [z, y, x] begin and extent of one face.
    void faceBox(
        TestFace const& face,
        std::size_t nz,
        std::size_t ny,
        std::size_t nx,
        std::array<std::size_t, 3>& begin,
        std::array<std::size_t, 3>& extent)
    {
        begin = {0u, 0u, 0u};
        extent = {nz, ny, nx};
        begin[face.axis] = face.begin;
        extent[face.axis] = face.size;
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

// ---------------------------------------------------------------------------
// Sparse-chunk layout: the extent stays (nz, ny, nx), but only the rim chunk of
// each face is written.  The interior is left unwritten and reads as HDF5's
// default fill value zero; a PIConGPU-style slab-view load returns zeros.
TEST(Restart, PmlAbsorberWritesSparseRimChunks)
{
    auto const directory = temporaryDirectory() / "checkpoints_pml_sparse";
    std::filesystem::remove_all(temporaryDirectory());
    std::filesystem::create_directories(temporaryDirectory());
    auto const input = temporaryDirectory() / "input.h5";
    writeParticleInput(input);

    auto const field = makeSparseField();
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
    std::size_t constexpr thickness = 3u;
    options.pmlThickness = {thickness, thickness, thickness, thickness, thickness, thickness};
    picBunch::writePicOnGPUCheckpointOpenPMD(field, options);

    std::size_t const nx = field.grid.n[0];
    std::size_t const ny = field.grid.n[1];
    std::size_t const nz = field.grid.n[2];
    auto const path = directory / "checkpoint_000002.h5";
    char const* const slabs[6] = {"xneg", "xpos", "yneg", "ypos", "zneg", "zpos"};
    char const* const components[6] = {"xy", "xz", "yx", "yz", "zx", "zy"};
    char const* const records[2] = {"Convolutional PML E", "Convolutional PML B"};

    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    ASSERT_GE(file, 0);
    for(int r = 0; r < 2; ++r)
        for(int s = 0; s < 6; ++s)
        {
            auto const face = testFace(static_cast<std::size_t>(s), thickness, nz, ny, nx);
            std::array<std::size_t, 3> chunkBegin{};
            std::array<std::size_t, 3> chunkExtent{};
            faceBox(face, nz, ny, nx, chunkBegin, chunkExtent);
            // The HDF5 chunk is one rim plane; its grid aligns with the dataset
            // origin, so a positive face allocates only rim planes.
            std::array<std::size_t, 3> expectedChunk{nz, ny, nx};
            expectedChunk[face.axis] = 1u;
            std::string const base
                = std::string("data/2/fields/") + records[r] + "_" + slabs[s];
            for(int c = 0; c < 6; ++c)
            {
                std::string const componentPath = base + "/" + components[c];
                hid_t dataset = H5Dopen2(file, componentPath.c_str(), H5P_DEFAULT);
                ASSERT_GE(dataset, 0) << componentPath;
                hid_t space = H5Dget_space(dataset);
                hsize_t dims[3] = {0, 0, 0};
                ASSERT_EQ(H5Sget_simple_extent_dims(space, dims, nullptr), 3);
                // (a) extent is still the full global domain.
                EXPECT_EQ(dims[0], nz) << componentPath;
                EXPECT_EQ(dims[1], ny) << componentPath;
                EXPECT_EQ(dims[2], nx) << componentPath;
                // (b) only rim planes are allocated; one chunk per plane.
                hid_t plist = H5Dget_create_plist(dataset);
                hsize_t chunkDims[3] = {0, 0, 0};
                ASSERT_EQ(H5Pget_chunk(plist, 3, chunkDims), 3) << componentPath;
                EXPECT_EQ(chunkDims[0], expectedChunk[0]) << componentPath;
                EXPECT_EQ(chunkDims[1], expectedChunk[1]) << componentPath;
                EXPECT_EQ(chunkDims[2], expectedChunk[2]) << componentPath;
                hsize_t chunks = 0;
                ASSERT_GE(H5Dget_num_chunks(dataset, space, &chunks), 0);
                EXPECT_EQ(chunks, face.size) << componentPath;
                hsize_t bytes = 0;
                for(hsize_t chunk = 0; chunk < chunks; ++chunk)
                {
                    hsize_t offset[3] = {0, 0, 0};
                    unsigned filterMask = 0;
                    haddr_t address = 0;
                    hsize_t size = 0;
                    ASSERT_GE(
                        H5Dget_chunk_info(
                            dataset, space, chunk, offset, &filterMask, &address, &size),
                        0);
                    // Every allocated plane lies inside the face rim.
                    EXPECT_GE(offset[face.axis], chunkBegin[face.axis]) << componentPath;
                    EXPECT_LT(offset[face.axis], chunkBegin[face.axis] + face.size)
                        << componentPath;
                    for(std::size_t axis = 0; axis < 3u; ++axis)
                    {
                        if(axis != face.axis)
                        {
                            EXPECT_EQ(offset[axis], 0u) << componentPath;
                        }
                    }
                    bytes = size;
                }
                EXPECT_EQ(
                    bytes, chunkDims[0] * chunkDims[1] * chunkDims[2] * sizeof(float))
                    << componentPath;
                H5Pclose(plist);
                H5Sclose(space);
                H5Dclose(dataset);
            }
        }
    H5Fclose(file);

    // (c) PIConGPU's slab-view call pattern and (d) the unwritten interior.
    openPMD::Series series(path.string(), openPMD::Access::READ_ONLY);
    auto iteration = series.iterations[2];
    for(int r = 0; r < 2; ++r)
        for(int s = 0; s < 6; ++s)
        {
            auto const face = testFace(static_cast<std::size_t>(s), thickness, nz, ny, nx);
            std::array<std::size_t, 3> faceBegin{};
            std::array<std::size_t, 3> faceExtent{};
            faceBox(face, nz, ny, nx, faceBegin, faceExtent);
            std::string const record
                = std::string(records[r]) + "_" + slabs[s];
            for(int c = 0; c < 6; ++c)
            {
                auto rc = iteration.meshes[record][components[c]];
                auto data = rc.loadChunk<float>(
                    openPMD::Offset{faceBegin[0], faceBegin[1], faceBegin[2]},
                    openPMD::Extent{faceExtent[0], faceExtent[1], faceExtent[2]});
                series.flush();
                std::size_t const count = faceExtent[0] * faceExtent[1] * faceExtent[2];
                for(std::size_t i = 0; i < count; ++i)
                    ASSERT_EQ(data.get()[i], 0.0f) << record << "/" << components[c];
            }
        }
    // (d) the interior is never written.  PIConGPU's RestartFieldLoader zeroes
    // its host buffer (`field.getHostBuffer().setValue(0.0)`) and then loads
    // only the per-face rim chunk.  openPMD 0.17 sets H5D_FILL_TIME_NEVER, so a
    // generic read of an unallocated chunk leaves the caller buffer untouched;
    // the consumer's zero-fill is what makes the interior zero.  Emulate that
    // exact sequence: start from a zeroed interior and read it back (the
    // unallocated chunks leave it untouched at zero).
    {
        auto rc = iteration.meshes["Convolutional PML E_xneg"]["xy"];
        std::size_t const iz = nz - 2 * thickness;
        std::size_t const iy = ny - 2 * thickness;
        std::size_t const ix = nx - 2 * thickness;
        ASSERT_GT(iz * iy * ix, 0u);
        std::vector<float> interior(iz * iy * ix, 0.0f);
        rc.loadChunkRaw(
            interior.data(),
            openPMD::Offset{thickness, thickness, thickness},
            openPMD::Extent{iz, iy, ix});
        series.flush();
        for(float const value : interior)
            ASSERT_EQ(value, 0.0f);
    }
    std::filesystem::remove_all(temporaryDirectory());
}

// ---------------------------------------------------------------------------
// Copy path: an input checkpoint with sparse, non-zero PML rim values is copied
// as rim chunks (not densified) and the values survive bit for bit.
TEST(Restart, PmlCopyPreservesSparseRim)
{
    auto const root = temporaryDirectory();
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    auto const input = root / "input.h5";
    writeParticleInput(input);

    auto const field = makeSparseField();
    picBunch::PicOnGPUWriteOptions base;
    base.step = 2;
    base.directory = (root / "source").string();
    base.inputFile = input.string();
    base.inputIteration = 0;
    base.time = 2.5e-13;
    base.timeUnitSI = 1.0;
    base.dt = 1.0e-15;
    base.unitLength = 1.0;
    base.unitEField = 2.0;
    base.unitBField = 4.0;
    base.nextId = 1;
    base.absorber = "pml";
    std::size_t constexpr thickness = 3u;
    base.pmlThickness = {thickness, thickness, thickness, thickness, thickness, thickness};
    picBunch::writePicOnGPUCheckpointOpenPMD(field, base);

    std::size_t const nx = field.grid.n[0];
    std::size_t const ny = field.grid.n[1];
    std::size_t const nz = field.grid.n[2];
    auto const sourcePath = root / "source" / "checkpoint_000002.h5";

    // Inject non-zero values into the x-rim of E_xneg/xy and the y-rim of
    // B_yneg/yz of the source checkpoint.
    auto writeSlab = [](std::filesystem::path const& path,
                        char const* record,
                        char const* component,
                        std::array<hsize_t, 3> const& start,
                        std::array<hsize_t, 3> const& count,
                        float value) {
        hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        ASSERT_GE(file, 0);
        std::string const datasetPath
            = std::string("data/2/fields/") + record + "/" + component;
        hid_t dataset = H5Dopen2(file, datasetPath.c_str(), H5P_DEFAULT);
        ASSERT_GE(dataset, 0) << datasetPath;
        hid_t space = H5Dget_space(dataset);
        ASSERT_GE(
            H5Sselect_hyperslab(
                space, H5S_SELECT_SET, start.data(), nullptr, count.data(), nullptr),
            0);
        hsize_t memoryDims[3] = {count[0], count[1], count[2]};
        hid_t memory = H5Screate_simple(3, memoryDims, nullptr);
        std::vector<float> values(
            static_cast<std::size_t>(count[0] * count[1] * count[2]), value);
        ASSERT_GE(
            H5Dwrite(
                dataset, H5T_NATIVE_FLOAT, memory, space, H5P_DEFAULT, values.data()),
            0);
        H5Sclose(memory);
        H5Sclose(space);
        H5Dclose(dataset);
        H5Fclose(file);
    };
    writeSlab(
        sourcePath, "Convolutional PML E_xneg", "xy",
        {0, 0, 0}, {nz, ny, thickness}, 1.5f);
    writeSlab(
        sourcePath, "Convolutional PML B_yneg", "yz",
        {0, 0, 0}, {nz, thickness, nx}, 2.5f);

    picBunch::PicOnGPUWriteOptions copy = base;
    copy.step = 3;
    copy.directory = (root / "copy").string();
    copy.inputFile = sourcePath.string();
    copy.inputIteration = 2;
    copy.requiredInputSpecies = {};
    copy.absorber = "none";
    picBunch::writePicOnGPUCheckpointOpenPMD(field, copy);

    auto const copyPath = root / "copy" / "checkpoint_000003.h5";
    hid_t file = H5Fopen(copyPath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    ASSERT_GE(file, 0);
    {
        std::string const path = "data/3/fields/Convolutional PML E_xneg/xy";
        hid_t dataset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
        ASSERT_GE(dataset, 0);
        hid_t space = H5Dget_space(dataset);
        hsize_t dims[3] = {0, 0, 0};
        ASSERT_EQ(H5Sget_simple_extent_dims(space, dims, nullptr), 3);
        EXPECT_EQ(dims[0], nz);
        EXPECT_EQ(dims[1], ny);
        EXPECT_EQ(dims[2], nx);
        hsize_t chunks = 0;
        ASSERT_GE(H5Dget_num_chunks(dataset, space, &chunks), 0);
        EXPECT_EQ(chunks, thickness);
        auto const values = readFloatDataset(file, path.c_str());
        ASSERT_EQ(values.size(), nz * ny * nx);
        for(std::size_t z = 0; z < nz; ++z)
            for(std::size_t y = 0; y < ny; ++y)
                for(std::size_t x = 0; x < nx; ++x)
                {
                    float const expected = x < thickness ? 1.5f : 0.0f;
                    EXPECT_EQ(values[(z * ny + y) * nx + x], expected);
                }
        H5Sclose(space);
        H5Dclose(dataset);
    }
    {
        std::string const path = "data/3/fields/Convolutional PML B_yneg/yz";
        auto const values = readFloatDataset(file, path.c_str());
        ASSERT_EQ(values.size(), nz * ny * nx);
        for(std::size_t z = 0; z < nz; ++z)
            for(std::size_t y = 0; y < ny; ++y)
                for(std::size_t x = 0; x < nx; ++x)
                {
                    float const expected = y < thickness ? 2.5f : 0.0f;
                    EXPECT_EQ(values[(z * ny + y) * nx + x], expected);
                }
    }
    H5Fclose(file);
    std::filesystem::remove_all(root);
}

// ---------------------------------------------------------------------------
// ADIOS2/BP5 sparse rim.  BP5 has no HDF5 chunk grid and no fill-value
// metadata: the interior is simply never stored.  PIConGPU's
// RestartFieldLoader zeroes its host buffer and then loads only the per-face
// rim chunk, so the unwritten interior must leave that buffer untouched.  A
// backend that materialised the interior (or returned garbage) would corrupt
// the consumer's zeroed field; this test is the regression for that claim.
TEST(Restart, PmlAbsorberWritesSparseRimChunksAdios2)
{
#if !(defined(openPMD_HAVE_ADIOS2) && openPMD_HAVE_ADIOS2)
    GTEST_SKIP() << "openPMD was built without the ADIOS2 backend";
#else
    auto const directory = temporaryDirectory() / "checkpoints_pml_sparse_bp5";
    std::filesystem::remove_all(temporaryDirectory());
    std::filesystem::create_directories(temporaryDirectory());
    auto const input = temporaryDirectory() / "input.h5";
    writeParticleInput(input);

    auto const field = makeSparseField();
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
    options.backend = picBunch::CheckpointBackend::Adios2;
    std::size_t constexpr thickness = 4u;
    options.pmlThickness = {thickness, thickness, thickness, thickness, thickness, thickness};
    picBunch::writePicOnGPUCheckpointOpenPMD(field, options);

    std::size_t const nx = field.grid.n[0];
    std::size_t const ny = field.grid.n[1];
    std::size_t const nz = field.grid.n[2];
    auto const path = directory / "checkpoint_000002.bp5";
    ASSERT_TRUE(std::filesystem::is_directory(path)) << path;

    char const* const slabs[6] = {"xneg", "xpos", "yneg", "ypos", "zneg", "zpos"};
    char const* const components[6] = {"xy", "xz", "yx", "yz", "zx", "zy"};
    char const* const records[2] = {"Convolutional PML E", "Convolutional PML B"};

    std::size_t const iz = nz - 2u * thickness;
    std::size_t const iy = ny - 2u * thickness;
    std::size_t const ix = nx - 2u * thickness;
    ASSERT_GT(iz * iy * ix, 0u);

    openPMD::Series series(path.string(), openPMD::Access::READ_ONLY);
    auto iteration = series.iterations[2];
    for(int r = 0; r < 2; ++r)
        for(int s = 0; s < 6; ++s)
        {
            auto const face = testFace(static_cast<std::size_t>(s), thickness, nz, ny, nx);
            std::array<std::size_t, 3> faceBegin{};
            std::array<std::size_t, 3> faceExtent{};
            faceBox(face, nz, ny, nx, faceBegin, faceExtent);
            std::size_t const count = faceExtent[0] * faceExtent[1] * faceExtent[2];
            std::string const record = std::string(records[r]) + "_" + slabs[s];
            for(int c = 0; c < 6; ++c)
            {
                auto rc = iteration.meshes[record][components[c]];
                auto const extent = rc.getExtent();
                ASSERT_EQ(extent.size(), 3u) << record << "/" << components[c];
                // The extent is still the full global domain.
                EXPECT_EQ(extent[0], nz) << record << "/" << components[c];
                EXPECT_EQ(extent[1], ny) << record << "/" << components[c];
                EXPECT_EQ(extent[2], nx) << record << "/" << components[c];
                // The in-memory PML state is zero; the rim chunk is stored at
                // its global offset and reads back bit for bit.
                std::vector<float> faceValues(count, 0.0f);
                rc.loadChunkRaw(
                    faceValues.data(),
                    openPMD::Offset{
                        static_cast<openPMD::Offset::value_type>(faceBegin[0]),
                        static_cast<openPMD::Offset::value_type>(faceBegin[1]),
                        static_cast<openPMD::Offset::value_type>(faceBegin[2])},
                    openPMD::Extent{
                        static_cast<openPMD::Extent::value_type>(faceExtent[0]),
                        static_cast<openPMD::Extent::value_type>(faceExtent[1]),
                        static_cast<openPMD::Extent::value_type>(faceExtent[2])});
                series.flush();
                for(std::size_t i = 0; i < count; ++i)
                    ASSERT_EQ(faceValues[i], 0.0f) << record << "/" << components[c];
                // Consumer emulation: zero the destination, then load only the
                // interior chunk.  An unwritten BP5 region must leave it zero.
                std::vector<float> interior(iz * iy * ix, 0.0f);
                rc.loadChunkRaw(
                    interior.data(),
                    openPMD::Offset{
                        static_cast<openPMD::Offset::value_type>(thickness),
                        static_cast<openPMD::Offset::value_type>(thickness),
                        static_cast<openPMD::Offset::value_type>(thickness)},
                    openPMD::Extent{
                        static_cast<openPMD::Extent::value_type>(iz),
                        static_cast<openPMD::Extent::value_type>(iy),
                        static_cast<openPMD::Extent::value_type>(ix)});
                series.flush();
                for(std::size_t i = 0; i < interior.size(); ++i)
                    ASSERT_EQ(interior[i], 0.0f) << record << "/" << components[c];
            }
        }
    std::filesystem::remove_all(temporaryDirectory());
#endif
}


// ---------------------------------------------------------------------------
// ADIOS2/BP5 copy path with non-zero rim values: a sparse input stays sparse,
// the rim chunks read back bit for bit at their global offsets, and the
// consumer's zeroed destination keeps a zero interior.  The HDF5 source is
// injected with the in-memory rim values exactly like the HDF5-only test.
TEST(Restart, PmlCopyPreservesSparseRimAdios2)
{
#if !(defined(openPMD_HAVE_ADIOS2) && openPMD_HAVE_ADIOS2)
    GTEST_SKIP() << "openPMD was built without the ADIOS2 backend";
#else
    auto const root = temporaryDirectory() / "pml_copy_bp5";
    std::filesystem::remove_all(temporaryDirectory());
    std::filesystem::create_directories(root);
    auto const input = root / "input.h5";
    writeParticleInput(input);

    auto const field = makeSparseField();
    picBunch::PicOnGPUWriteOptions base;
    base.step = 2;
    base.directory = (root / "source").string();
    base.inputFile = input.string();
    base.inputIteration = 0;
    base.time = 2.5e-13;
    base.timeUnitSI = 1.0;
    base.dt = 1.0e-15;
    base.unitLength = 1.0;
    base.unitEField = 2.0;
    base.unitBField = 4.0;
    base.nextId = 1;
    base.absorber = "pml";
    std::size_t constexpr thickness = 4u;
    base.pmlThickness = {thickness, thickness, thickness, thickness, thickness, thickness};
    picBunch::writePicOnGPUCheckpointOpenPMD(field, base);

    std::size_t const nx = field.grid.n[0];
    std::size_t const ny = field.grid.n[1];
    std::size_t const nz = field.grid.n[2];
    auto const sourcePath = root / "source" / "checkpoint_000002.h5";

    // In-memory rim values injected into the HDF5 source checkpoint.
    auto writeSlab = [](std::filesystem::path const& path,
                        char const* record,
                        char const* component,
                        std::array<hsize_t, 3> const& start,
                        std::array<hsize_t, 3> const& count,
                        float value) {
        hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        ASSERT_GE(file, 0);
        std::string const datasetPath
            = std::string("data/2/fields/") + record + "/" + component;
        hid_t dataset = H5Dopen2(file, datasetPath.c_str(), H5P_DEFAULT);
        ASSERT_GE(dataset, 0) << datasetPath;
        hid_t space = H5Dget_space(dataset);
        ASSERT_GE(
            H5Sselect_hyperslab(
                space, H5S_SELECT_SET, start.data(), nullptr, count.data(), nullptr),
            0);
        hsize_t memoryDims[3] = {count[0], count[1], count[2]};
        hid_t memory = H5Screate_simple(3, memoryDims, nullptr);
        std::vector<float> values(
            static_cast<std::size_t>(count[0] * count[1] * count[2]), value);
        ASSERT_GE(
            H5Dwrite(
                dataset, H5T_NATIVE_FLOAT, memory, space, H5P_DEFAULT, values.data()),
            0);
        H5Sclose(memory);
        H5Sclose(space);
        H5Dclose(dataset);
        H5Fclose(file);
    };
    writeSlab(
        sourcePath, "Convolutional PML E_xneg", "xy",
        {0, 0, 0}, {nz, ny, thickness}, 1.5f);
    writeSlab(
        sourcePath, "Convolutional PML B_yneg", "yz",
        {0, 0, 0}, {nz, thickness, nx}, 2.5f);

    picBunch::PicOnGPUWriteOptions copy = base;
    copy.step = 3;
    copy.directory = (root / "copy").string();
    copy.inputFile = sourcePath.string();
    copy.inputIteration = 2;
    copy.requiredInputSpecies = {};
    copy.absorber = "none";
    copy.backend = picBunch::CheckpointBackend::Adios2;
    picBunch::writePicOnGPUCheckpointOpenPMD(field, copy);

    auto const copyPath = root / "copy" / "checkpoint_000003.bp5";
    ASSERT_TRUE(std::filesystem::is_directory(copyPath)) << copyPath;

    std::size_t const iz = nz - 2u * thickness;
    std::size_t const iy = ny - 2u * thickness;
    std::size_t const ix = nx - 2u * thickness;
    ASSERT_GT(iz * iy * ix, 0u);

    openPMD::Series series(copyPath.string(), openPMD::Access::READ_ONLY);
    auto iteration = series.iterations[3];
    //! Zero the consumer's destination, load only the interior (must stay
    //! zero), then load the six face rims at their global offsets into the
    //! buffer and compare the reconstruction with the in-memory array.
    auto const verify = [&](char const* record,
                            char const* component,
                            auto const& expected) {
        std::string const name = std::string(record) + "/" + component;
        auto rc = iteration.meshes[record][component];
        auto const extent = rc.getExtent();
        ASSERT_EQ(extent.size(), 3u) << name;
        EXPECT_EQ(extent[0], nz) << name;
        EXPECT_EQ(extent[1], ny) << name;
        EXPECT_EQ(extent[2], nx) << name;
        std::vector<float> full(nz * ny * nx, 0.0f);
        // The consumer zero-filled its host field first.
        std::vector<float> interior(iz * iy * ix, 0.0f);
        rc.loadChunkRaw(
            interior.data(),
            openPMD::Offset{
                static_cast<openPMD::Offset::value_type>(thickness),
                static_cast<openPMD::Offset::value_type>(thickness),
                static_cast<openPMD::Offset::value_type>(thickness)},
            openPMD::Extent{
                static_cast<openPMD::Extent::value_type>(iz),
                static_cast<openPMD::Extent::value_type>(iy),
                static_cast<openPMD::Extent::value_type>(ix)});
        series.flush();
        for(std::size_t i = 0; i < interior.size(); ++i)
            ASSERT_EQ(interior[i], 0.0f) << name;
        for(int s = 0; s < 6; ++s)
        {
            auto const face = testFace(static_cast<std::size_t>(s), thickness, nz, ny, nx);
            std::array<std::size_t, 3> begin{};
            std::array<std::size_t, 3> ext{};
            faceBox(face, nz, ny, nx, begin, ext);
            std::size_t const count = ext[0] * ext[1] * ext[2];
            std::vector<float> chunk(count, 0.0f);
            rc.loadChunkRaw(
                chunk.data(),
                openPMD::Offset{
                    static_cast<openPMD::Offset::value_type>(begin[0]),
                    static_cast<openPMD::Offset::value_type>(begin[1]),
                    static_cast<openPMD::Offset::value_type>(begin[2])},
                openPMD::Extent{
                    static_cast<openPMD::Extent::value_type>(ext[0]),
                    static_cast<openPMD::Extent::value_type>(ext[1]),
                    static_cast<openPMD::Extent::value_type>(ext[2])});
            series.flush();
            std::size_t i = 0;
            for(std::size_t z = begin[0]; z < begin[0] + ext[0]; ++z)
                for(std::size_t y = begin[1]; y < begin[1] + ext[1]; ++y)
                    for(std::size_t x = begin[2]; x < begin[2] + ext[2]; ++x)
                        full[(z * ny + y) * nx + x] = chunk[i++];
        }
        for(std::size_t z = 0; z < nz; ++z)
            for(std::size_t y = 0; y < ny; ++y)
                for(std::size_t x = 0; x < nx; ++x)
                    ASSERT_EQ(full[(z * ny + y) * nx + x], expected(z, y, x))
                        << name << " at " << z << "," << y << "," << x;
    };
    verify("Convolutional PML E_xneg", "xy", [&](std::size_t, std::size_t, std::size_t x) {
        return x < thickness ? 1.5f : 0.0f;
    });
    verify("Convolutional PML B_yneg", "yz", [&](std::size_t, std::size_t y, std::size_t) {
        return y < thickness ? 2.5f : 0.0f;
    });
    std::filesystem::remove_all(temporaryDirectory());
#endif
}
