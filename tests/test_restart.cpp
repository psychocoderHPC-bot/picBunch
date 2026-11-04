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
#include <cstring>
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

    void writeStringArrayAttribute(hid_t object, char const* name, std::vector<std::string> const& values)
    {
        hid_t type = H5Tcopy(H5T_C_S1);
        H5Tset_size(type, H5T_VARIABLE);
        hsize_t dimensions[1] = {static_cast<hsize_t>(values.size())};
        hid_t space = H5Screate_simple(1, dimensions, nullptr);
        hid_t attribute = H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
        std::vector<char const*> pointers;
        pointers.reserve(values.size());
        for(auto const& value : values)
            pointers.push_back(value.c_str());
        H5Awrite(attribute, type, pointers.data());
        H5Aclose(attribute);
        H5Sclose(space);
        H5Tclose(type);
    }

    void writeDoubleArrayAttribute(hid_t object, char const* name, std::array<double, 3> const& values)
    {
        hsize_t dimensions[1] = {3};
        hid_t space = H5Screate_simple(1, dimensions, nullptr);
        hid_t attribute = H5Acreate2(object, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attribute, H5T_NATIVE_DOUBLE, values.data());
        H5Aclose(attribute);
        H5Sclose(space);
    }

    void writeUnitDimension(hid_t object, std::array<double, 7> const& values)
    {
        hsize_t dimensions[1] = {7};
        hid_t space = H5Screate_simple(1, dimensions, nullptr);
        hid_t attribute = H5Acreate2(object, "unitDimension", H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attribute, H5T_NATIVE_DOUBLE, values.data());
        H5Aclose(attribute);
        H5Sclose(space);
    }

    void writeDoubleComponent(hid_t parent, char const* name, double value)
    {
        hid_t dataset = create(parent, name, H5T_IEEE_F64LE, {1});
        H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
        hid_t space = H5Screate(H5S_SCALAR);
        hid_t attribute = H5Acreate2(dataset, "unitSI", H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
        double const unit = 1.0;
        H5Awrite(attribute, H5T_NATIVE_DOUBLE, &unit);
        H5Aclose(attribute);
        H5Sclose(space);
        H5Dclose(dataset);
    }

    //! One PML slab record with the six cross components PIConGPU stores.
    void writePmlRecord(hid_t fields, char const* name, bool electric, float seed)
    {
        hid_t record = H5Gcreate2(fields, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        writeStringArrayAttribute(record, "axisLabels", {"z", "y", "x"});
        writeStringAttribute(record, "dataOrder", "C");
        writeStringAttribute(record, "geometry", "cartesian");
        writeStringAttribute(record, "fieldSmoothing", "none");
        writeDoubleArrayAttribute(record, "gridSpacing", {2.5, 1.5, 0.5});
        writeDoubleArrayAttribute(record, "gridGlobalOffset", {3.0, 2.0, 1.0});
        double const gridUnit = 1.0;
        hid_t scalarSpace = H5Screate(H5S_SCALAR);
        hid_t unitAttribute = H5Acreate2(
            record, "gridUnitSI", H5T_NATIVE_DOUBLE, scalarSpace, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(unitAttribute, H5T_NATIVE_DOUBLE, &gridUnit);
        H5Aclose(unitAttribute);
        float const timeOffset = 0.0f;
        hid_t timeAttribute = H5Acreate2(
            record, "timeOffset", H5T_NATIVE_FLOAT, scalarSpace, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(timeAttribute, H5T_NATIVE_FLOAT, &timeOffset);
        H5Aclose(timeAttribute);
        H5Sclose(scalarSpace);
        std::array<double, 7> const dimension
            = electric ? std::array<double, 7>{1, 1, -3, -1, 0, 0, 0}
                       : std::array<double, 7>{0, 1, -2, -1, 0, 0, 0};
        writeUnitDimension(record, dimension);

        char const* const components[6] = {"xy", "xz", "yx", "yz", "zx", "zy"};
        double const positions[6][3] = {
            {0.5, 0.0, 0.0},
            {0.5, 0.0, 0.0},
            {0.0, 0.5, 0.0},
            {0.0, 0.5, 0.0},
            {0.0, 0.0, 0.5},
            {0.0, 0.0, 0.5}};
        std::vector<float> values(6 * 5 * 4);
        for(std::size_t i = 0; i < values.size(); ++i)
            values[i] = seed + static_cast<float>(i) * 1.0e-3f;
        for(std::size_t c = 0; c < 6; ++c)
        {
            hid_t dataset = create(record, components[c], H5T_IEEE_F32LE, {6, 5, 4});
            H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
            double const position[3] = {positions[c][0], positions[c][1], positions[c][2]};
            hid_t space = H5Screate_simple(1, std::array<hsize_t, 1>{3}.data(), nullptr);
            hid_t attribute = H5Acreate2(dataset, "position", H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
            H5Awrite(attribute, H5T_NATIVE_DOUBLE, position);
            H5Aclose(attribute);
            H5Sclose(space);
            double const unit = electric ? 1.5 : 2.5;
            space = H5Screate(H5S_SCALAR);
            attribute = H5Acreate2(dataset, "unitSI", H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
            H5Awrite(attribute, H5T_NATIVE_DOUBLE, &unit);
            H5Aclose(attribute);
            H5Sclose(space);
            H5Dclose(dataset);
        }
        H5Gclose(record);
    }

    //! Minimal openPMD/HDF5 particle input with (optionally) patch metadata.
    void writeParticleInput(
        std::filesystem::path const& path,
        bool withPatches,
        bool withPml = false,
        bool doubleParticles = false,
        std::vector<std::string> const& speciesNames = std::vector<std::string>{"electrons"})
    {
        hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        H5Gcreate2(file, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t iteration = H5Gcreate2(file, "data/0", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t particles = H5Gcreate2(iteration, "particles", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for(auto const& speciesName : speciesNames)
        {
            hid_t species = H5Gcreate2(particles, speciesName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            writeStringAttribute(species, "sourceMarker", "preserved");
            writeScalarDouble(species, "particleShape", 1.0);

            char const* axes = "xyz";
            hid_t position = H5Gcreate2(species, "position", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            hid_t momentum = H5Gcreate2(species, "momentum", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            hid_t offset = H5Gcreate2(species, "positionOffset", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            for(int a = 0; a < 3; ++a)
            {
                std::string const axis(1, axes[a]);
                if(doubleParticles)
                {
                    writeDoubleComponent(position, axis.c_str(), 0.25 + 0.1 * static_cast<double>(a));
                    writeDoubleComponent(momentum, axis.c_str(), 0.0);
                    hid_t dataset = create(offset, axis.c_str(), H5T_IEEE_F64LE, {1});
                    double const stored = static_cast<double>(a);
                    H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &stored);
                    H5Dclose(dataset);
                }
                else
                {
                    writeFloatComponent(position, axis.c_str(), 0.25f + 0.1f * static_cast<float>(a));
                    writeFloatComponent(momentum, axis.c_str(), 0.0f);
                    hid_t dataset = create(offset, axis.c_str(), H5T_STD_I32LE, {1});
                    int32_t const stored = a;
                    H5Dwrite(dataset, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, &stored);
                    H5Dclose(dataset);
                }
            }
            H5Gclose(position);
            H5Gclose(momentum);
            H5Gclose(offset);
            if(doubleParticles)
            {
                hid_t weighting = H5Gcreate2(species, "weighting", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                writeDoubleComponent(weighting, "weighting", 1.0);
                H5Gclose(weighting);
            }
            else
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
        }

        if(withPml)
        {
            hid_t fields = H5Gcreate2(iteration, "fields", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            writePmlRecord(fields, "Convolutional PML E_xneg", true, 0.5f);
            writePmlRecord(fields, "Convolutional PML B_ypos", false, 1.5f);
            H5Gclose(fields);
        }

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

    std::uint64_t datasetPoints(hid_t file, char const* path)
    {
        hid_t dataset = H5Dopen2(file, path, H5P_DEFAULT);
        EXPECT_GE(dataset, 0) << "missing dataset " << path;
        hid_t space = H5Dget_space(dataset);
        hssize_t const points = H5Sget_simple_extent_npoints(space);
        H5Sclose(space);
        H5Dclose(dataset);
        return points < 0 ? 0u : static_cast<std::uint64_t>(points);
    }

    std::uint64_t readUint64Dataset(hid_t file, char const* path)
    {
        hid_t dataset = H5Dopen2(file, path, H5P_DEFAULT);
        EXPECT_GE(dataset, 0) << "missing dataset " << path;
        std::uint64_t value = 0;
        H5Dread(dataset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
        H5Dclose(dataset);
        return value;
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
    // The input has no PML records, so none may be synthesized.
    EXPECT_EQ(H5Lexists(fields, "Convolutional PML E_xneg", H5P_DEFAULT), 0);
    H5Gclose(species);
    H5Gclose(particles);

    H5Gclose(fields);
    H5Gclose(iteration);
    H5Fclose(file);
    std::filesystem::remove_all(temporaryDirectory());
}

TEST(Restart, WriterCopiesEverySpeciesAndUsesGlobalIdRange)
{
    auto const directory = temporaryDirectory() / "checkpoints_species";
    std::filesystem::remove_all(temporaryDirectory());
    std::filesystem::create_directories(temporaryDirectory());
    auto const input = temporaryDirectory() / "input_two_species.h5";
    writeParticleInput(input, true, false, false, {"electrons", "ions"});

    auto const field = makeField();
    picBunch::PicOnGPUWriteOptions options;
    options.step = 2;
    options.directory = directory.string();
    options.inputFile = input.string();
    options.inputIteration = 0;
    // Selecting one species must not drop the other one.
    options.species = "electrons";
    options.unitLength = 1.0;
    options.unitEField = 1.0;
    options.unitBField = 1.0;
    // Lower bound below the real particle count: the writer must never store it.
    options.nextId = 1;
    picBunch::writePicOnGPUCheckpoint(field, options);

    auto const path = directory / "checkpoint_000002.h5";
    ASSERT_TRUE(std::filesystem::exists(path));
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    ASSERT_GE(file, 0);

    hid_t particles = H5Gopen2(file, "data/2/particles", H5P_DEFAULT);
    ASSERT_GE(particles, 0);
    for(char const* name : {"electrons", "ions"})
    {
        hid_t species = H5Gopen2(particles, name, H5P_DEFAULT);
        ASSERT_GE(species, 0) << "missing copied species " << name;
        EXPECT_EQ(readStringAttribute(species, "sourceMarker"), "preserved");
        EXPECT_EQ(H5Lexists(species, "particlePatches", H5P_DEFAULT), 1);
        H5Gclose(species);
    }
    H5Gclose(particles);

    // One global range: two particles in total, one per species.
    std::uint64_t numberOfParticles = 0;
    for(char const* name : {"electrons", "ions"})
    {
        std::string const positionPath = std::string("data/2/particles/") + name + "/position/x";
        numberOfParticles += datasetPoints(file, positionPath.c_str());
    }
    EXPECT_EQ(numberOfParticles, 2u);
    EXPECT_EQ(readUint64Dataset(file, "data/2/fields/picongpu_idProvider/startId"), 0u);
    EXPECT_EQ(
        readUint64Dataset(file, "data/2/fields/picongpu_idProvider/nextId"),
        numberOfParticles);
    hid_t startId = H5Dopen2(file, "data/2/fields/picongpu_idProvider/startId", H5P_DEFAULT);
    ASSERT_GE(startId, 0);
    EXPECT_EQ(readScalarAttribute<uint64_t>(startId, "maxNumProc", H5T_NATIVE_UINT64), 1u);
    H5Dclose(startId);

    H5Fclose(file);
    std::filesystem::remove_all(temporaryDirectory());
}

TEST(Restart, WriterRejectsMissingRequestedSpecies)
{
    auto const directory = temporaryDirectory() / "checkpoints_missing_species";
    std::filesystem::remove_all(temporaryDirectory());
    std::filesystem::create_directories(temporaryDirectory());
    auto const input = temporaryDirectory() / "input_electrons_only.h5";
    writeParticleInput(input, true);

    auto const field = makeField();
    picBunch::PicOnGPUWriteOptions options;
    options.step = 0;
    options.directory = directory.string();
    options.inputFile = input.string();
    options.inputIteration = 0;
    options.species = "ions";
    options.unitLength = 1.0;
    options.unitEField = 1.0;
    options.unitBField = 1.0;
    try
    {
        picBunch::writePicOnGPUCheckpoint(field, options);
        FAIL() << "expected an error for the absent species";
    }
    catch(std::runtime_error const& error)
    {
        EXPECT_NE(std::string(error.what()).find("ions"), std::string::npos) << error.what();
    }
    EXPECT_FALSE(std::filesystem::exists(directory / "checkpoint_000000.h5"));
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

TEST(Restart, WriterCopiesPmlRecordsVerbatim)
{
    auto const directory = temporaryDirectory() / "checkpoints_pml";
    std::filesystem::remove_all(temporaryDirectory());
    std::filesystem::create_directories(temporaryDirectory());
    auto const input = temporaryDirectory() / "input_pml.h5";
    writeParticleInput(input, true, true);

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
    options.unitLength = 1.0;
    options.unitEField = 1.0;
    options.unitBField = 1.0;
    options.nextId = 1;
    picBunch::writePicOnGPUCheckpoint(field, options);

    auto const path = directory / "checkpoint_000002.h5";
    hid_t sourceFile = H5Fopen(input.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    ASSERT_GE(sourceFile, 0);
    ASSERT_GE(file, 0);
    char const* const records[2] = {"Convolutional PML E_xneg", "Convolutional PML B_ypos"};
    char const* const components[6] = {"xy", "xz", "yx", "yz", "zx", "zy"};
    for(char const* recordName : records)
    {
        std::string const sourceRecord = std::string("data/0/fields/") + recordName;
        std::string const outputRecord = std::string("data/2/fields/") + recordName;
        hid_t sourceGroup = H5Gopen2(sourceFile, sourceRecord.c_str(), H5P_DEFAULT);
        hid_t outputGroup = H5Gopen2(file, outputRecord.c_str(), H5P_DEFAULT);
        ASSERT_GE(sourceGroup, 0) << sourceRecord;
        ASSERT_GE(outputGroup, 0) << outputRecord;
        // The copied record adopts the main-field geometry.
        EXPECT_EQ(
            readDoubleArrayAttribute(outputGroup, "gridSpacing"),
            (std::vector<double>{
                field.grid.spacing[2], field.grid.spacing[1], field.grid.spacing[0]}));
        EXPECT_EQ(
            readDoubleArrayAttribute(outputGroup, "gridGlobalOffset"),
            (std::vector<double>{
                field.grid.origin[2], field.grid.origin[1], field.grid.origin[0]}));
        EXPECT_DOUBLE_EQ(
            readScalarAttribute<double>(outputGroup, "gridUnitSI", H5T_NATIVE_DOUBLE), 1.0);
        H5Gclose(sourceGroup);
        H5Gclose(outputGroup);

        for(char const* component : components)
        {
            std::string const sourcePath = sourceRecord + "/" + component;
            std::string const outputPath = outputRecord + "/" + component;
            auto const expected = readFloatDataset(sourceFile, sourcePath.c_str());
            auto const actual = readFloatDataset(file, outputPath.c_str());
            ASSERT_EQ(actual.size(), expected.size());
            EXPECT_TRUE(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0)
                << recordName << "/" << component;

            hid_t dataset = H5Dopen2(file, outputPath.c_str(), H5P_DEFAULT);
            ASSERT_GE(dataset, 0);
            hid_t type = H5Dget_type(dataset);
            EXPECT_EQ(H5Tget_class(type), H5T_FLOAT);
            EXPECT_EQ(H5Tget_size(type), 4u);
            H5Tclose(type);
            EXPECT_EQ(
                readDoubleArrayAttribute(dataset, "position").size(), 3u);
            H5Dclose(dataset);
        }
    }
    // RNG state is optional and is not copied with the PML slabs.
    EXPECT_EQ(H5Lexists(file, "data/2/fields/RNGProvider3XorMin", H5P_DEFAULT), 0);
    H5Fclose(sourceFile);
    H5Fclose(file);
    std::filesystem::remove_all(temporaryDirectory());
}

TEST(Restart, WriterPinsFloat32ParticleDtypes)
{
    auto const directory = temporaryDirectory() / "checkpoints_dtypes";
    std::filesystem::remove_all(temporaryDirectory());
    std::filesystem::create_directories(temporaryDirectory());
    auto const input = temporaryDirectory() / "input_double.h5";
    writeParticleInput(input, true, false, true);

    auto const field = makeField();
    picBunch::PicOnGPUWriteOptions options;
    options.step = 2;
    options.directory = directory.string();
    options.inputFile = input.string();
    options.inputIteration = 0;
    options.species = "electrons";
    options.unitLength = 1.0;
    options.unitEField = 1.0;
    options.unitBField = 1.0;
    options.nextId = 1;
    picBunch::writePicOnGPUCheckpoint(field, options);

    auto const path = directory / "checkpoint_000002.h5";
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    ASSERT_GE(file, 0);
    char const* const axes[3] = {"x", "y", "z"};
    for(char const* record : {"position", "momentum"})
        for(char const* axis : axes)
        {
            std::string const datasetPath
                = std::string("data/2/particles/electrons/") + record + "/" + axis;
            hid_t dataset = H5Dopen2(file, datasetPath.c_str(), H5P_DEFAULT);
            ASSERT_GE(dataset, 0) << datasetPath;
            hid_t type = H5Dget_type(dataset);
            EXPECT_EQ(H5Tget_class(type), H5T_FLOAT) << datasetPath;
            EXPECT_EQ(H5Tget_size(type), 4u) << datasetPath;
            H5Tclose(type);
            H5Dclose(dataset);
        }
    for(char const* axis : axes)
    {
        std::string const datasetPath
            = std::string("data/2/particles/electrons/positionOffset/") + axis;
        hid_t dataset = H5Dopen2(file, datasetPath.c_str(), H5P_DEFAULT);
        ASSERT_GE(dataset, 0) << datasetPath;
        hid_t type = H5Dget_type(dataset);
        EXPECT_EQ(H5Tget_class(type), H5T_INTEGER) << datasetPath;
        EXPECT_EQ(H5Tget_size(type), 4u) << datasetPath;
        H5Tclose(type);
        H5Dclose(dataset);
    }
    hid_t weighting = H5Dopen2(file, "data/2/particles/electrons/weighting", H5P_DEFAULT);
    ASSERT_GE(weighting, 0);
    hid_t weightingType = H5Dget_type(weighting);
    EXPECT_EQ(H5Tget_class(weightingType), H5T_FLOAT);
    EXPECT_EQ(H5Tget_size(weightingType), 4u);
    H5Tclose(weightingType);
    H5Dclose(weighting);
    H5Fclose(file);
    std::filesystem::remove_all(temporaryDirectory());
}
