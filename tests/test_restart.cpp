// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group: explicit geometry resolution and the PIConGPU checkpoint writer.
// The writer is checked through the raw HDF5 layout, not through openPMD-api,
// so attribute names, dtypes, extents and the unit normalization are tested as
// they land on disk.

#include "picBunch/Compute.hpp"
#include "picBunch/OpenPMDWriter.hpp"
#include "picBunch/Types.hpp"

#include <hdf5.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace
{
    std::filesystem::path temporaryDirectory()
    {
        return std::filesystem::temp_directory_path()
            / ("picbunch_restart_" + std::to_string(static_cast<long long>(getpid())));
    }

    hid_t create(hid_t loc, char const* name, hid_t type, std::vector<hsize_t> const& dimensions)
    {
        hid_t space = H5Screate_simple(
            static_cast<int>(dimensions.size()), dimensions.data(), nullptr);
        hid_t dataset = H5Dcreate2(loc, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Sclose(space);
        return dataset;
    }

    void writeScalarDouble(hid_t loc, char const* name, double value)
    {
        hid_t dataset = create(loc, name, H5T_IEEE_F64LE, {1});
        H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
        H5Dclose(dataset);
    }

    void writeFloatComponent(hid_t parent, char const* name, float value)
    {
        hid_t dataset = create(parent, name, H5T_IEEE_F32LE, {1});
        H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
        hid_t space = H5Screate(H5S_SCALAR);
        hid_t attribute = H5Acreate2(dataset, "unitSI", H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
        double const unit = 1.0;
        H5Awrite(attribute, H5T_NATIVE_DOUBLE, &unit);
        H5Aclose(attribute);
        H5Sclose(space);
        H5Dclose(dataset);
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

    //! Minimal openPMD/HDF5 particle input with (optionally) patch metadata.
    void writeParticleInput(std::filesystem::path const& path, bool withPatches)
    {
        hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        H5Gcreate2(file, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t iteration = H5Gcreate2(file, "data/0", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t particles = H5Gcreate2(iteration, "particles", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t species = H5Gcreate2(particles, "electrons", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        writeStringAttribute(species, "sourceMarker", "preserved");
        writeScalarDouble(species, "particleShape", 1.0);

        char const* axes = "xyz";
        hid_t position = H5Gcreate2(species, "position", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t momentum = H5Gcreate2(species, "momentum", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t offset = H5Gcreate2(species, "positionOffset", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for(int a = 0; a < 3; ++a)
        {
            std::string const axis(1, axes[a]);
            writeFloatComponent(position, axis.c_str(), 0.25f + 0.1f * static_cast<float>(a));
            writeFloatComponent(momentum, axis.c_str(), 0.0f);
            hid_t dataset = create(offset, axis.c_str(), H5T_STD_I32LE, {1});
            int32_t const stored = a;
            H5Dwrite(dataset, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, &stored);
            H5Dclose(dataset);
        }
        H5Gclose(position);
        H5Gclose(momentum);
        H5Gclose(offset);
        writeFloatComponent(species, "weighting", 1.0f);

        if(withPatches)
        {
            hid_t patches = H5Gcreate2(species, "particlePatches", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            hid_t numParticles = create(patches, "numParticles", H5T_STD_U64LE, {1});
            uint64_t const one = 1;
            H5Dwrite(numParticles, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, &one);
            H5Dclose(numParticles);
            hid_t numOffset = create(patches, "numParticlesOffset", H5T_STD_U64LE, {1});
            uint64_t const zero = 0;
            H5Dwrite(numOffset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, &zero);
            H5Dclose(numOffset);
            H5Gclose(patches);
        }

        H5Gclose(species);
        H5Gclose(particles);
        H5Gclose(iteration);
        H5Fclose(file);
    }

    std::string readStringAttribute(hid_t object, char const* name)
    {
        hid_t attribute = H5Aopen(object, name, H5P_DEFAULT);
        EXPECT_GE(attribute, 0) << "missing attribute " << name;
        hid_t type = H5Aget_type(attribute);
        char* value = nullptr;
        H5Aread(attribute, type, &value);
        std::string result = value != nullptr ? value : "";
        if(value != nullptr)
            H5free_memory(value);
        H5Tclose(type);
        H5Aclose(attribute);
        return result;
    }

    template<typename T>
    T readScalarAttribute(hid_t object, char const* name, hid_t type)
    {
        hid_t attribute = H5Aopen(object, name, H5P_DEFAULT);
        EXPECT_GE(attribute, 0) << "missing attribute " << name;
        T value{};
        H5Aread(attribute, type, &value);
        H5Aclose(attribute);
        return value;
    }

    std::vector<double> readDoubleArrayAttribute(hid_t object, char const* name)
    {
        hid_t attribute = H5Aopen(object, name, H5P_DEFAULT);
        hid_t space = H5Aget_space(attribute);
        hsize_t count = 0;
        H5Sget_simple_extent_dims(space, &count, nullptr);
        std::vector<double> values(count);
        H5Aread(attribute, H5T_NATIVE_DOUBLE, values.data());
        H5Sclose(space);
        H5Aclose(attribute);
        return values;
    }

    std::vector<float> readFloatDataset(hid_t file, char const* path)
    {
        hid_t dataset = H5Dopen2(file, path, H5P_DEFAULT);
        EXPECT_GE(dataset, 0) << "missing dataset " << path;
        hid_t space = H5Dget_space(dataset);
        hsize_t dimensions[3] = {0, 0, 0};
        int const rank = H5Sget_simple_extent_dims(space, dimensions, nullptr);
        EXPECT_EQ(rank, 3);
        std::vector<float> values(static_cast<std::size_t>(dimensions[0] * dimensions[1] * dimensions[2]));
        H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
        H5Sclose(space);
        H5Dclose(dataset);
        return values;
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
                field.data[c][l] = 0.0625 * static_cast<double>(c) + 1.0e-2 * static_cast<double>(l)
                    + 1.0e-9 * static_cast<double>(c * 11 + l);
        }
        return field;
    }

    picBunch::Bunch twoParticles()
    {
        picBunch::Bunch bunch;
        bunch.x = {0.0, 1.0};
        bunch.y = {0.0, 1.0};
        bunch.z = {0.0, 1.0};
        bunch.ux = {0.0, 0.0};
        bunch.uy = {0.0, 0.0};
        bunch.uz = {0.0, 0.0};
        bunch.weight = {1.0, 1.0};
        bunch.charge = {-1.0, -1.0};
        return bunch;
    }
} // namespace

TEST(Restart, CellSizeExplicitOverridesMesh)
{
    std::array<double, 3> const explicitSize{1.0, 2.0, 3.0};
    std::array<double, 3> const meshSize{4.0, 5.0, 6.0};
    EXPECT_EQ(picBunch::resolveCellSize(explicitSize, meshSize), explicitSize);
    EXPECT_EQ(picBunch::resolveCellSize(std::nullopt, meshSize), meshSize);
    EXPECT_THROW(picBunch::resolveCellSize(std::nullopt, std::nullopt), std::runtime_error);
    EXPECT_THROW(
        picBunch::resolveCellSize(std::array<double, 3>{0.0, 1.0, 1.0}, meshSize),
        std::invalid_argument);
    EXPECT_THROW(
        picBunch::resolveCellSize(std::array<double, 3>{1.0, -1.0, 1.0}, meshSize),
        std::invalid_argument);
}

TEST(Restart, GridAndOriginOverridesReplaceDerivedValues)
{
    picBunch::ComputeOptions options;
    options.cellSize = {0.25, 0.5, 0.75};
    options.marginCells = 4.0;
    auto const derived = picBunch::makeGrid(twoParticles(), options);

    options.gridSize = std::array<std::size_t, 3>{16, 20, 24};
    options.gridOrigin = std::array<double, 3>{-1.0, -2.0, -3.0};
    auto const overridden = picBunch::makeGrid(twoParticles(), options);
    EXPECT_EQ(overridden.n, (std::array<std::size_t, 3>{16, 20, 24}));
    EXPECT_EQ(overridden.origin, (std::array<double, 3>{-1.0, -2.0, -3.0}));
    // Spacing still comes from the cell size; only n and origin are overridden.
    EXPECT_EQ(overridden.spacing, derived.spacing);
    EXPECT_NE(overridden.n, derived.n);

    options.gridSize = std::array<std::size_t, 3>{15, 20, 24};
    EXPECT_THROW(picBunch::makeGrid(twoParticles(), options), std::invalid_argument);
}

TEST(Restart, WriterEmitsPIConGPUCheckpointContract)
{
    auto const directory = temporaryDirectory() / "checkpoints";
    std::filesystem::remove_all(temporaryDirectory());
    std::filesystem::create_directories(temporaryDirectory());
    auto const input = temporaryDirectory() / "input.h5";
    writeParticleInput(input, true);

    auto const field = makeField();
    picBunch::PicOnGPUWriteOptions options;
    options.step = 2;
    options.directory = directory.string();
    options.inputFile = input.string();
    options.inputIteration = 0;
    options.species = "electrons";
    options.time = 2.5e-13;
    options.timeUnitSI = 1.0;
    options.dt = 1.0e-15;
    options.unitLength = 5.7735e-8;
    options.unitEField = 8.850761452125e12;
    options.unitBField = 29522.962355927582;
    options.nextId = 1;

    picBunch::writePicOnGPUCheckpoint(field, options);

    auto const path = directory / "checkpoint_000002.h5";
    ASSERT_TRUE(std::filesystem::exists(path));
    {
        std::ifstream text(directory / "checkpoints.txt");
        std::string content;
        std::getline(text, content);
        EXPECT_EQ(content, "2");
        std::string extra;
        EXPECT_FALSE(std::getline(text, extra));
    }

    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    ASSERT_GE(file, 0);

    EXPECT_EQ(readStringAttribute(file, "openPMD"), "1.1.0");
    EXPECT_EQ(readStringAttribute(file, "iterationEncoding"), "fileBased");
    EXPECT_EQ(readStringAttribute(file, "iterationFormat"), "checkpoint_%06T");
    EXPECT_EQ(readStringAttribute(file, "meshesPath"), "fields/");
    EXPECT_EQ(readStringAttribute(file, "particlesPath"), "particles/");
    EXPECT_EQ(readScalarAttribute<uint32_t>(file, "picongpuIOVersionMajor", H5T_NATIVE_UINT32), 5u);
    EXPECT_EQ(readScalarAttribute<uint32_t>(file, "picongpuIOVersionMinor", H5T_NATIVE_UINT32), 0u);
    EXPECT_EQ(H5Aexists(file, "picBunchRestartWarning"), 0);

    hid_t iteration = H5Gopen2(file, "data/2", H5P_DEFAULT);
    ASSERT_GE(iteration, 0);
    EXPECT_EQ(readScalarAttribute<uint64_t>(iteration, "iteration", H5T_NATIVE_UINT64), 2u);
    EXPECT_EQ(readScalarAttribute<uint32_t>(iteration, "sim_slides", H5T_NATIVE_UINT32), 0u);
    EXPECT_DOUBLE_EQ(readScalarAttribute<double>(iteration, "time", H5T_NATIVE_DOUBLE), 2.5e-13);
    EXPECT_DOUBLE_EQ(readScalarAttribute<double>(iteration, "timeUnitSI", H5T_NATIVE_DOUBLE), 1.0);
    EXPECT_DOUBLE_EQ(readScalarAttribute<double>(iteration, "dt", H5T_NATIVE_DOUBLE), 1.0e-15);

    hid_t fields = H5Gopen2(iteration, "fields", H5P_DEFAULT);
    ASSERT_GE(fields, 0);
    struct Record
    {
        char const* name;
        std::size_t first;
        double unit;
        std::array<double, 7> dimension;
    };
    std::array<double, 7> const dimensionE{1, 1, -3, -1, 0, 0, 0};
    std::array<double, 7> const dimensionB{0, 1, -2, -1, 0, 0, 0};
    Record const records[] = {
        {"E", 0, options.unitEField, dimensionE},
        {"B", 3, options.unitBField, dimensionB}};
    for(auto const& spec : records)
    {
        hid_t record = H5Gopen2(fields, spec.name, H5P_DEFAULT);
        ASSERT_GE(record, 0);
        EXPECT_EQ(readDoubleArrayAttribute(record, "gridSpacing"),
                  (std::vector<double>{
                      field.grid.spacing[2] / options.unitLength,
                      field.grid.spacing[1] / options.unitLength,
                      field.grid.spacing[0] / options.unitLength}));
        EXPECT_EQ(readDoubleArrayAttribute(record, "gridGlobalOffset"),
                  (std::vector<double>{
                      field.grid.origin[2] / options.unitLength,
                      field.grid.origin[1] / options.unitLength,
                      field.grid.origin[0] / options.unitLength}));
        EXPECT_DOUBLE_EQ(
            readScalarAttribute<double>(record, "gridUnitSI", H5T_NATIVE_DOUBLE), options.unitLength);
        EXPECT_EQ(readDoubleArrayAttribute(record, "unitDimension"),
                  (std::vector<double>(spec.dimension.begin(), spec.dimension.end())));
        char const* const components[3] = {"x", "y", "z"};
        for(std::size_t a = 0; a < 3; ++a)
        {
            std::string const path = std::string("data/2/fields/") + spec.name + "/" + components[a];
            hid_t dataset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
            ASSERT_GE(dataset, 0);
            hid_t type = H5Dget_type(dataset);
            EXPECT_EQ(H5Tget_class(type), H5T_FLOAT);
            EXPECT_EQ(H5Tget_size(type), 4u);
            hid_t space = H5Dget_space(dataset);
            hsize_t dimensions[3] = {0, 0, 0};
            H5Sget_simple_extent_dims(space, dimensions, nullptr);
            EXPECT_EQ(dimensions[0], field.grid.n[2]);
            EXPECT_EQ(dimensions[1], field.grid.n[1]);
            EXPECT_EQ(dimensions[2], field.grid.n[0]);
            H5Sclose(space);
            H5Tclose(type);

            auto const position = readDoubleArrayAttribute(dataset, "position");
            std::size_t const component = spec.first + a;
            EXPECT_EQ(
                position,
                (std::vector<double>{
                    picBunch::kYeeOffset[component][0],
                    picBunch::kYeeOffset[component][1],
                    picBunch::kYeeOffset[component][2]}));
            EXPECT_DOUBLE_EQ(
                readScalarAttribute<double>(dataset, "unitSI", H5T_NATIVE_DOUBLE), spec.unit);
            H5Dclose(dataset);

            auto const stored = readFloatDataset(file, path.c_str());
            ASSERT_EQ(stored.size(), field.data[component].size());
            for(std::size_t l = 0; l < stored.size(); ++l)
                EXPECT_EQ(stored[l], static_cast<float>(field.data[component][l] / spec.unit))
                    << spec.name << "/" << components[a] << " index " << l;
        }
        H5Gclose(record);
    }

    for(char const* name : {"startId", "nextId"})
    {
        hid_t idGroup = H5Gopen2(fields, "picongpu_idProvider", H5P_DEFAULT);
        ASSERT_GE(idGroup, 0);
        hid_t dataset = H5Dopen2(idGroup, name, H5P_DEFAULT);
        ASSERT_GE(dataset, 0);
        hid_t type = H5Dget_type(dataset);
        EXPECT_EQ(H5Tget_class(type), H5T_INTEGER);
        EXPECT_EQ(H5Tget_size(type), 8u);
        H5Tclose(type);
        H5Dclose(dataset);
        H5Gclose(idGroup);
    }
    // H5Ocopy preserved the source species attributes and the patch metadata.
    hid_t particles = H5Gopen2(iteration, "particles", H5P_DEFAULT);
    ASSERT_GE(particles, 0);
    hid_t species = H5Gopen2(particles, "electrons", H5P_DEFAULT);
    ASSERT_GE(species, 0);
    EXPECT_EQ(readStringAttribute(species, "sourceMarker"), "preserved");
    EXPECT_EQ(H5Lexists(species, "particlePatches", H5P_DEFAULT), 1);
    H5Gclose(species);
    H5Gclose(particles);

    H5Gclose(fields);
    H5Gclose(iteration);
    H5Fclose(file);
    std::filesystem::remove_all(temporaryDirectory());
}

TEST(Restart, WriterReportsMissingParticlePatches)
{
    auto const directory = temporaryDirectory() / "checkpoints";
    std::filesystem::remove_all(temporaryDirectory());
    std::filesystem::create_directories(temporaryDirectory());
    auto const input = temporaryDirectory() / "input_nopatches.h5";
    writeParticleInput(input, false);

    auto const field = makeField();
    picBunch::PicOnGPUWriteOptions options;
    options.step = 0;
    options.directory = directory.string();
    options.inputFile = input.string();
    options.inputIteration = 0;
    options.species = "electrons";
    options.unitLength = 1.0;
    options.unitEField = 1.0;
    options.unitBField = 1.0;
    options.nextId = 1;
    picBunch::writePicOnGPUCheckpoint(field, options);

    auto const path = directory / "checkpoint_000000.h5";
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    ASSERT_GE(file, 0);
    ASSERT_EQ(H5Aexists(file, "picBunchRestartWarning"), 1);
    auto const warning = readStringAttribute(file, "picBunchRestartWarning");
    EXPECT_NE(warning.find("particlePatches"), std::string::npos);
    H5Fclose(file);
    std::filesystem::remove_all(temporaryDirectory());
}

TEST(Restart, WriterRejectsMissingUnits)
{
    auto const field = makeField();
    picBunch::PicOnGPUWriteOptions options;
    options.inputFile = "unused.h5";
    options.unitEField = 0.0;
    EXPECT_THROW(picBunch::writePicOnGPUCheckpoint(field, options), std::invalid_argument);
    options.unitEField = 1.0;
    options.unitBField = -1.0;
    EXPECT_THROW(picBunch::writePicOnGPUCheckpoint(field, options), std::invalid_argument);
}
