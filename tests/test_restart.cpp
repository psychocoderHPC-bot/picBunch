// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group: explicit geometry resolution and the PIConGPU checkpoint writer.
// The HDF5 layout is checked through the raw HDF5 API, so attribute names,
// dtypes, extents and the unit normalization are tested as they land on disk.
// Backend-parity checks read the same tree back through openPMD-api.

#include "picBunch/Compute.hpp"
#include "picBunch/Config.hpp"
#include "picBunch/OpenPMD.hpp"
#include "picBunch/OpenPMDWriter.hpp"
#include "picBunch/Types.hpp"

#include <openPMD/config.hpp>
#include <openPMD/openPMD.hpp>

#include <hdf5.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
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

    void writeDoubleAttribute(hid_t object, char const* name, double value)
    {
        hid_t space = H5Screate(H5S_SCALAR);
        hid_t attribute = H5Acreate2(object, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attribute, H5T_NATIVE_DOUBLE, &value);
        H5Aclose(attribute);
        H5Sclose(space);
    }

    void writeUInt32Attribute(hid_t object, char const* name, std::uint32_t value)
    {
        hid_t space = H5Screate(H5S_SCALAR);
        hid_t attribute = H5Acreate2(object, name, H5T_NATIVE_UINT32, space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attribute, H5T_NATIVE_UINT32, &value);
        H5Aclose(attribute);
        H5Sclose(space);
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

    void writeDoubleArrayAttribute(hid_t object, char const* name, picBunch::Vec3 const& values)
    {
        hsize_t dimensions[1] = {3};
        hid_t space = H5Screate_simple(1, dimensions, nullptr);
        hid_t attribute = H5Acreate2(object, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attribute, H5T_NATIVE_DOUBLE, &values[0]);
        H5Aclose(attribute);
        H5Sclose(space);
    }

    void writeUnitDimension(hid_t object, std::array<double, 7> const& values)
    {
        hsize_t dimensions[1] = {7};
        hid_t space = H5Screate_simple(1, dimensions, nullptr);
        hid_t attribute = H5Acreate2(object, "unitDimension", H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attribute, H5T_NATIVE_DOUBLE, &values[0]);
        H5Aclose(attribute);
        H5Sclose(space);
    }

    void writeUnitSI(hid_t dataset, double unit)
    {
        hid_t space = H5Screate(H5S_SCALAR);
        hid_t attribute = H5Acreate2(dataset, "unitSI", H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attribute, H5T_NATIVE_DOUBLE, &unit);
        H5Aclose(attribute);
        H5Sclose(space);
    }

    void writeRecordAttributes(
        hid_t record,
        std::array<double, 7> const& unitDimension,
        bool macroWeighted,
        double weightingPower)
    {
        writeUnitDimension(record, unitDimension);
        writeDoubleAttribute(record, "weightingPower", weightingPower);
        writeUInt32Attribute(record, "macroWeighted", macroWeighted ? 1u : 0u);
        float const timeOffset = 0.0f;
        hid_t space = H5Screate(H5S_SCALAR);
        hid_t attribute = H5Acreate2(
            record, "timeOffset", H5T_NATIVE_FLOAT, space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attribute, H5T_NATIVE_FLOAT, &timeOffset);
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
        // The openPMD reader requires the series attributes; the fixture is a
        // plain openPMD 1.x tree written without openPMD-api.
        writeStringAttribute(file, "openPMD", "1.1.0");
        writeStringAttribute(file, "iterationEncoding", "groupBased");
        writeStringAttribute(file, "iterationFormat", "%T");
        writeStringAttribute(file, "basePath", "/data/%T/");
        writeStringAttribute(file, "meshesPath", "fields/");
        writeStringAttribute(file, "particlesPath", "particles/");
        writeUInt32Attribute(file, "openPMDextension", 1);
        H5Gcreate2(file, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t iteration = H5Gcreate2(file, "data/0", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        writeDoubleAttribute(iteration, "dt", 1.0);
        writeDoubleAttribute(iteration, "time", 0.0);
        writeDoubleAttribute(iteration, "timeUnitSI", 1.0);
        hid_t particles = H5Gcreate2(iteration, "particles", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for(auto const& speciesName : speciesNames)
        {
            hid_t species = H5Gcreate2(particles, speciesName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            writeStringAttribute(species, "sourceMarker", "preserved");

            char const* axes = "xyz";
            hid_t position = H5Gcreate2(species, "position", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            writeRecordAttributes(position, {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, false, 0.0);
            hid_t momentum = H5Gcreate2(species, "momentum", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            writeRecordAttributes(momentum, {1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0}, true, 1.0);
            hid_t offset = H5Gcreate2(species, "positionOffset", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            writeRecordAttributes(offset, {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, false, 0.0);
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
                    writeUnitSI(dataset, 1.0);
                    H5Dclose(dataset);
                }
                else
                {
                    writeFloatComponent(position, axis.c_str(), 0.25f + 0.1f * static_cast<float>(a));
                    writeFloatComponent(momentum, axis.c_str(), 0.0f);
                    hid_t dataset = create(offset, axis.c_str(), H5T_STD_I32LE, {1});
                    int32_t const stored = a;
                    H5Dwrite(dataset, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, &stored);
                    writeUnitSI(dataset, 1.0);
                    H5Dclose(dataset);
                }
            }
            H5Gclose(position);
            H5Gclose(momentum);
            H5Gclose(offset);
            {
                hid_t weighting = doubleParticles
                    ? create(species, "weighting", H5T_IEEE_F64LE, {1})
                    : create(species, "weighting", H5T_IEEE_F32LE, {1});
                if(doubleParticles)
                {
                    double const weight = 1.0;
                    H5Dwrite(weighting, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &weight);
                }
                else
                {
                    float const weight = 1.0f;
                    H5Dwrite(weighting, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &weight);
                }
                writeUnitSI(weighting, 1.0);
                writeRecordAttributes(
                    weighting, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, true, 1.0);
                H5Dclose(weighting);
            }

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
        std::string result;
        if(H5Tis_variable_str(type) > 0)
        {
            char* value = nullptr;
            H5Aread(attribute, type, &value);
            result = value != nullptr ? value : "";
            if(value != nullptr)
                H5free_memory(value);
        }
        else
        {
            // openPMD writes fixed-length strings; read them into a byte buffer.
            std::size_t const size = H5Tget_size(type);
            std::vector<char> value(size + 1u, '\0');
            H5Aread(attribute, type, value.data());
            result.assign(value.data());
        }
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

    std::vector<std::uint64_t> readUint64Vector(hid_t file, char const* path)
    {
        hid_t dataset = H5Dopen2(file, path, H5P_DEFAULT);
        EXPECT_GE(dataset, 0) << "missing dataset " << path;
        hid_t space = H5Dget_space(dataset);
        hssize_t const points = H5Sget_simple_extent_npoints(space);
        std::vector<std::uint64_t> values(static_cast<std::size_t>(points < 0 ? 0 : points));
        H5Dread(dataset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
        H5Sclose(space);
        H5Dclose(dataset);
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

    std::vector<std::int32_t> readInt32Vector(hid_t file, char const* path)
    {
        hid_t dataset = H5Dopen2(file, path, H5P_DEFAULT);
        EXPECT_GE(dataset, 0) << "missing dataset " << path;
        hid_t space = H5Dget_space(dataset);
        hssize_t const points = H5Sget_simple_extent_npoints(space);
        std::vector<std::int32_t> values(static_cast<std::size_t>(points < 0 ? 0 : points));
        H5Dread(dataset, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
        H5Sclose(space);
        H5Dclose(dataset);
        return values;
    }

    std::vector<double> readDoubleVector(hid_t file, char const* path)
    {
        hid_t dataset = H5Dopen2(file, path, H5P_DEFAULT);
        EXPECT_GE(dataset, 0) << "missing dataset " << path;
        hid_t space = H5Dget_space(dataset);
        hssize_t const points = H5Sget_simple_extent_npoints(space);
        std::vector<double> values(static_cast<std::size_t>(points < 0 ? 0 : points));
        H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
        H5Sclose(space);
        H5Dclose(dataset);
        return values;
    }

    //! An empty openPMD component is either a zero-length dataset or a
    //! constant record group carrying ``shape = [0]``.
    void expectEmptyComponent(hid_t file, std::string const& path)
    {
        hid_t dataset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
        if(dataset >= 0)
        {
            hid_t space = H5Dget_space(dataset);
            hssize_t const points = H5Sget_simple_extent_npoints(space);
            EXPECT_EQ(points, 0) << path;
            H5Sclose(space);
            H5Dclose(dataset);
            return;
        }
        auto const shapeOf = [&](std::string const& objectPath)
            -> std::optional<std::uint64_t> {
            hid_t object = H5Oopen(file, objectPath.c_str(), H5P_DEFAULT);
            if(object < 0)
                return std::nullopt;
            std::optional<std::uint64_t> shape;
            if(H5Aexists(object, "shape") > 0)
            {
                hid_t attribute = H5Aopen(object, "shape", H5P_DEFAULT);
                std::uint64_t value = 1;
                H5Aread(attribute, H5T_NATIVE_UINT64, &value);
                H5Aclose(attribute);
                shape = value;
            }
            H5Oclose(object);
            return shape;
        };
        auto shape = shapeOf(path);
        if(!shape)
        {
            std::size_t const slash = path.find_last_of('/');
            if(slash != std::string::npos)
                shape = shapeOf(path.substr(0, slash));
        }
        ASSERT_TRUE(shape.has_value()) << path;
        EXPECT_EQ(*shape, 0u) << path;
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
    picBunch::Vec3 const explicitSize{1.0, 2.0, 3.0};
    picBunch::Vec3 const meshSize{4.0, 5.0, 6.0};
    EXPECT_EQ(picBunch::resolveCellSize(explicitSize, meshSize), explicitSize);
    EXPECT_EQ(picBunch::resolveCellSize(std::nullopt, meshSize), meshSize);
    EXPECT_THROW(picBunch::resolveCellSize(std::nullopt, std::nullopt), std::runtime_error);
    EXPECT_THROW(
        picBunch::resolveCellSize(picBunch::Vec3{0.0, 1.0, 1.0}, meshSize),
        std::invalid_argument);
    EXPECT_THROW(
        picBunch::resolveCellSize(picBunch::Vec3{1.0, -1.0, 1.0}, meshSize),
        std::invalid_argument);
}

TEST(Restart, GridAndOriginOverridesReplaceDerivedValues)
{
    picBunch::ComputeOptions options;
    options.cellSize = {0.25, 0.5, 0.75};
    options.marginCells = 4.0;
    auto const derived = picBunch::makeGrid(twoParticles(), options);

    options.gridSize = picBunch::Idx3{16, 20, 24};
    options.gridOrigin = picBunch::Vec3{-1.0, -2.0, -3.0};
    auto const overridden = picBunch::makeGrid(twoParticles(), options);
    EXPECT_EQ(overridden.n, (picBunch::Idx3{16, 20, 24}));
    EXPECT_EQ(overridden.origin, (picBunch::Vec3{-1.0, -2.0, -3.0}));
    // Spacing still comes from the cell size; only n and origin are overridden.
    EXPECT_EQ(overridden.spacing, derived.spacing);
    EXPECT_NE(overridden.n, derived.n);

    // Odd extents are supported by the parity-correct R2C path (task W3o): the
    // even-only requirement that previously rejected {15,20,24} is lifted, so an
    // odd explicit grid is now accepted verbatim.
    options.gridSize = picBunch::Idx3{15, 20, 24};
    auto const oddGrid = picBunch::makeGrid(twoParticles(), options);
    EXPECT_EQ(oddGrid.n, (picBunch::Idx3{15, 20, 24}));
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
    options.requiredInputSpecies = {"electrons"};
    options.time = 2.5e-13;
    options.timeUnitSI = 1.0;
    options.dt = 1.0e-15;
    options.unitLength = 5.7735e-8;
    options.unitEField = 8.850761452125e12;
    options.unitBField = 29522.962355927582;
    options.nextId = 1;

    picBunch::writePicOnGPUCheckpointOpenPMD(field, options);

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

TEST(Restart, IterationUnitAttributesMatchDerivedUnits)
{
    auto const directory = temporaryDirectory() / "checkpoints_units";
    std::filesystem::remove_all(temporaryDirectory());
    std::filesystem::create_directories(temporaryDirectory());
    auto const input = temporaryDirectory() / "input.h5";
    writeParticleInput(input, true);

    picBunch::Config::PicOnGPU deck;
    deck.deltaT = 0.8e-16;
    deck.cellSize = {0.1772e-6, 0.1772e-6, 0.1772e-6};
    deck.baseDensity = 1.0e25;
    deck.typicalParticlesPerCell = 2.0;
    picBunch::derivePicOnGPUUnits(deck);

    auto const field = makeField();
    picBunch::PicOnGPUWriteOptions options;
    options.step = 2;
    options.directory = directory.string();
    options.inputFile = input.string();
    options.inputIteration = 0;
    options.requiredInputSpecies = {"electrons"};
    options.time = 2.5e-13;
    options.timeUnitSI = 1.0;
    options.dt = deck.dt;
    options.unitLength = deck.unitLength;
    options.unitEField = deck.unitEField;
    options.unitBField = deck.unitBField;
    options.unitTime = deck.unitTime;
    options.unitSpeed = deck.unitSpeed;
    options.unitEnergy = deck.unitEnergy;
    options.unitMass = deck.unitMass;
    options.unitCharge = deck.unitCharge;
    picBunch::writePicOnGPUCheckpointOpenPMD(field, options);

    hid_t file = H5Fopen(
        (directory / "checkpoint_000002.h5").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    ASSERT_GE(file, 0);
    hid_t iteration = H5Gopen2(file, "data/2", H5P_DEFAULT);
    ASSERT_GE(iteration, 0);
    struct Attribute
    {
        char const* name;
        double value;
    };
    Attribute const expected[] = {
        {"unit_time", deck.unitTime},
        {"unit_speed", deck.unitSpeed},
        {"unit_length", deck.unitLength},
        {"unit_mass", deck.unitMass},
        {"unit_charge", deck.unitCharge},
        {"unit_energy", deck.unitEnergy},
        {"unit_efield", deck.unitEField},
        {"unit_bfield", deck.unitBField}};
    for(auto const& attribute : expected)
        EXPECT_DOUBLE_EQ(
            readScalarAttribute<double>(iteration, attribute.name, H5T_NATIVE_DOUBLE),
            attribute.value)
            << attribute.name;
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
    options.requiredInputSpecies = {"electrons"};
    options.unitLength = 1.0;
    options.unitEField = 1.0;
    options.unitBField = 1.0;
    // Lower bound below the real particle count: the writer must never store it.
    options.nextId = 1;
    picBunch::writePicOnGPUCheckpointOpenPMD(field, options);

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
    options.requiredInputSpecies = {"ions"};
    options.unitLength = 1.0;
    options.unitEField = 1.0;
    options.unitBField = 1.0;
    try
    {
        picBunch::writePicOnGPUCheckpointOpenPMD(field, options);
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
    options.requiredInputSpecies = {"electrons"};
    options.unitLength = 1.0;
    options.unitEField = 1.0;
    options.unitBField = 1.0;
    options.nextId = 1;
    picBunch::writePicOnGPUCheckpointOpenPMD(field, options);

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
    EXPECT_THROW(picBunch::writePicOnGPUCheckpointOpenPMD(field, options), std::invalid_argument);
    options.unitEField = 1.0;
    options.unitBField = -1.0;
    EXPECT_THROW(picBunch::writePicOnGPUCheckpointOpenPMD(field, options), std::invalid_argument);
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
    options.requiredInputSpecies = {"electrons"};
    options.time = 2.5e-13;
    options.timeUnitSI = 1.0;
    options.dt = 1.0e-15;
    options.unitLength = 1.0;
    options.unitEField = 1.0;
    options.unitBField = 1.0;
    options.nextId = 1;
    picBunch::writePicOnGPUCheckpointOpenPMD(field, options);

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

TEST(Restart, WriterEmitsOnePatchPerRankOverlapWithDisjointOffsets)
{
    auto const directory = temporaryDirectory();
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    auto const input = directory / "input.h5";
    writeParticleInput(input, true);

    picBunch::PicOnGPUWriteOptions options;
    options.step = 7;
    options.directory = directory.string();
    options.inputFile = input.string();
    options.inputIteration = 0;
    options.requiredInputSpecies = {"electrons"};
    options.timeUnitSI = 1.0;
    options.unitLength = 1.0;
    options.unitEField = 1.0;
    options.unitBField = 1.0;

    picBunch::GeneratedParticles generated;
    generated.species = "electrons";
    generated.bunch = twoParticles();
    generated.bunch.z[1] = 3.0; // patch 1 owns this particle
    generated.domainOrigin = {0.0, 0.0, 0.0};
    generated.cellSize = {1.0, 1.0, 1.0};
    generated.patchOffset = {0, 0, 0};
    generated.patchExtent = {5, 5, 4};
    generated.unitMassSpeed = 1.0;
    generated.unitCharge = 1.0;
    generated.unitMass = 1.0;
    generated.fallbackMass = 1.0;
    // Two rank-overlap slabs; z cells 2 and 3 are in both.
    generated.patches = {
        {{0, 0, 0}, {5, 5, 3}},
        {{0, 0, 2}, {5, 5, 2}}};
    options.generatedSpecies = {generated};
    picBunch::writePicOnGPUCheckpointOpenPMD(makeField(), options);

    auto const path = directory / "checkpoint_000007.h5";
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    ASSERT_GE(file, 0);
    // Particle z=0 belongs to patch 0 (cells [0,3)); z=3 belongs to patch 1
    // (cells [2,4)).  The regions overlap on cells 2, but ownership is disjoint:
    // the writer assigns each particle to the first containing patch and writes
    // cumulative numParticlesOffset.
    auto const counts = readUint64Vector(
        file, "data/7/particles/electrons/particlePatches/numParticles");
    auto const offsets = readUint64Vector(
        file, "data/7/particles/electrons/particlePatches/numParticlesOffset");
    auto const zo = readUint64Vector(file, "data/7/particles/electrons/particlePatches/offset/z");
    auto const ze = readUint64Vector(file, "data/7/particles/electrons/particlePatches/extent/z");
    ASSERT_EQ(counts.size(), 2u);
    ASSERT_EQ(offsets.size(), 2u);
    EXPECT_EQ(offsets[0], 0u);
    EXPECT_EQ(offsets[1], counts[0]);
    EXPECT_EQ(counts[0], 1u);
    EXPECT_EQ(counts[1], 1u);
    EXPECT_EQ(counts[0] + counts[1], 2u);
    EXPECT_EQ(zo[0], 0u);
    EXPECT_EQ(zo[1], 2u);
    EXPECT_EQ(ze[0], 3u);
    EXPECT_EQ(ze[1], 2u);
    H5Fclose(file);
    std::filesystem::remove_all(directory);
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
    options.requiredInputSpecies = {"electrons"};
    options.unitLength = 1.0;
    options.unitEField = 1.0;
    options.unitBField = 1.0;
    options.nextId = 1;
    picBunch::writePicOnGPUCheckpointOpenPMD(field, options);

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

TEST(Restart, GeneratedMomentumUsesPIConGPUMacroWeighting)
{
    auto const directory = temporaryDirectory() / "checkpoints_momentum";
    std::filesystem::remove_all(temporaryDirectory());
    std::filesystem::create_directories(temporaryDirectory());
    auto const input = temporaryDirectory() / "input.h5";
    writeParticleInput(input, true);

    picBunch::PicOnGPUWriteOptions options;
    options.step = 4;
    options.directory = directory.string();
    options.inputFile = input.string();
    options.inputIteration = 0;
    options.requiredInputSpecies = {"electrons"};
    options.unitLength = 1.0;
    options.unitEField = 1.0;
    options.unitBField = 1.0;
    // Use float64 so the round-trip checks the convention, not float32
    // quantization.
    options.float64 = true;

    double const mass = picBunch::kElectronMass;
    double const charge = -picBunch::kElementaryCharge;
    picBunch::GeneratedParticles generated;
    generated.species = "electrons";
    generated.bunch.x = {0.0, 1.0};
    generated.bunch.y = {0.0, 0.0};
    generated.bunch.z = {0.0, 1.0};
    generated.bunch.ux = {0.3, -0.7};
    generated.bunch.uy = {0.0, 1.1};
    generated.bunch.uz = {2.0, -1.5};
    generated.bunch.weight = {1.0, 100.0};
    generated.bunch.charge = {charge, charge};
    generated.bunch.restMass = {mass, mass};
    generated.domainOrigin = {0.0, 0.0, 0.0};
    generated.cellSize = {1.0, 1.0, 1.0};
    generated.patchOffset = {0, 0, 0};
    generated.patchExtent = {5, 5, 4};
    generated.unitMassSpeed = mass * picBunch::kSpeedOfLight;
    generated.unitCharge = picBunch::kElementaryCharge;
    generated.unitMass = mass;
    generated.fallbackMass = mass;
    options.generatedSpecies = {generated};
    picBunch::writePicOnGPUCheckpointOpenPMD(makeField(), options);

    auto const path = directory / "checkpoint_000004.h5";
    // The PIConGPU convention is macro momentum: raw = u*m*c*w/unitMassSpeed,
    // macroWeighted = true, weightingPower = 1.  Check the raw data and
    // metadata first.
    {
        hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        ASSERT_GE(file, 0);
        hid_t momentum = H5Gopen2(file, "data/4/particles/electrons/momentum", H5P_DEFAULT);
        ASSERT_GE(momentum, 0);
        EXPECT_EQ(readScalarAttribute<std::uint32_t>(momentum, "macroWeighted", H5T_NATIVE_UINT32), 1u);
        EXPECT_DOUBLE_EQ(readScalarAttribute<double>(momentum, "weightingPower", H5T_NATIVE_DOUBLE), 1.0);
        H5Gclose(momentum);
        auto const stored = readDoubleVector(file, "data/4/particles/electrons/momentum/x");
        ASSERT_EQ(stored.size(), 2u);
        EXPECT_NEAR(stored[0], 0.3 * 1.0, 1.0e-12);
        EXPECT_NEAR(stored[1], -0.7 * 100.0, 1.0e-12);
        H5Fclose(file);
    }

    // picBunch's own reader applies macroWeighted/weightingPower, so the
    // per-particle u must survive for both weights.
    auto const roundTrip = picBunch::readBunch(path.string(), 0u);
    ASSERT_EQ(roundTrip.bunch.size(), 2u);
    EXPECT_NEAR(roundTrip.bunch.ux[0], 0.3, 1.0e-12);
    EXPECT_NEAR(roundTrip.bunch.uy[0], 0.0, 1.0e-12);
    EXPECT_NEAR(roundTrip.bunch.uz[0], 2.0, 1.0e-12);
    EXPECT_NEAR(roundTrip.bunch.ux[1], -0.7, 1.0e-12);
    EXPECT_NEAR(roundTrip.bunch.uy[1], 1.1, 1.0e-12);
    EXPECT_NEAR(roundTrip.bunch.uz[1], -1.5, 1.0e-12);
    EXPECT_DOUBLE_EQ(roundTrip.bunch.weight[0], 1.0);
    EXPECT_DOUBLE_EQ(roundTrip.bunch.weight[1], 100.0);
    std::filesystem::remove_all(temporaryDirectory());
}

TEST(Restart, GeneratedPatchesAreContiguousForInterleavedInput)
{
    auto const directory = temporaryDirectory() / "checkpoints_interleaved";
    std::filesystem::remove_all(temporaryDirectory());
    std::filesystem::create_directories(temporaryDirectory());
    auto const input = temporaryDirectory() / "input.h5";
    writeParticleInput(input, true);

    picBunch::PicOnGPUWriteOptions options;
    options.step = 5;
    options.directory = directory.string();
    options.inputFile = input.string();
    options.inputIteration = 0;
    options.requiredInputSpecies = {"electrons"};
    options.unitLength = 1.0;
    options.unitEField = 1.0;
    options.unitBField = 1.0;

    picBunch::GeneratedParticles generated;
    generated.species = "electrons";
    // Deliberately interleaved patch membership A,B,A,B.  y is unique per
    // particle, so any lost or duplicated particle is visible in the dataset.
    generated.bunch.x = {0.0, 0.0, 0.0, 0.0};
    generated.bunch.y = {0.0, 1.0, 2.0, 3.0};
    generated.bunch.z = {0.0, 2.0, 0.0, 2.0};
    generated.bunch.ux = {0.0, 0.0, 0.0, 0.0};
    generated.bunch.uy = {0.0, 0.0, 0.0, 0.0};
    generated.bunch.uz = {0.0, 0.0, 0.0, 0.0};
    generated.bunch.weight = {1.0, 1.0, 1.0, 1.0};
    generated.bunch.charge = {-1.0, -1.0, -1.0, -1.0};
    generated.domainOrigin = {0.0, 0.0, 0.0};
    generated.cellSize = {1.0, 1.0, 1.0};
    generated.patchOffset = {0, 0, 0};
    generated.patchExtent = {5, 5, 4};
    generated.unitMassSpeed = 1.0;
    generated.unitCharge = 1.0;
    generated.unitMass = 1.0;
    generated.fallbackMass = 1.0;
    generated.patches = {
        {{0, 0, 0}, {5, 5, 2}},
        {{0, 0, 2}, {5, 5, 2}}};
    options.generatedSpecies = {generated};
    picBunch::writePicOnGPUCheckpointOpenPMD(makeField(), options);

    auto const path = directory / "checkpoint_000005.h5";
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    ASSERT_GE(file, 0);
    auto const counts = readUint64Vector(
        file, "data/5/particles/electrons/particlePatches/numParticles");
    auto const offsets = readUint64Vector(
        file, "data/5/particles/electrons/particlePatches/numParticlesOffset");
    auto const patchOffsetZ = readUint64Vector(
        file, "data/5/particles/electrons/particlePatches/offset/z");
    auto const patchExtentZ = readUint64Vector(
        file, "data/5/particles/electrons/particlePatches/extent/z");
    auto const cellZ = readInt32Vector(file, "data/5/particles/electrons/positionOffset/z");
    auto const cellY = readInt32Vector(file, "data/5/particles/electrons/positionOffset/y");
    H5Fclose(file);

    ASSERT_EQ(counts.size(), 2u);
    ASSERT_EQ(offsets.size(), 2u);
    EXPECT_EQ(counts[0], 2u);
    EXPECT_EQ(counts[1], 2u);
    EXPECT_EQ(offsets[0], 0u);
    EXPECT_EQ(offsets[1], counts[0]);
    ASSERT_EQ(cellZ.size(), 4u);
    ASSERT_EQ(cellY.size(), 4u);

    // Every declared patch slice must lie in its spatial extent.
    for(std::size_t q = 0; q < 2u; ++q)
        for(std::uint64_t s = offsets[q]; s < offsets[q] + counts[q]; ++s)
        {
            EXPECT_GE(cellZ[static_cast<std::size_t>(s)],
                      static_cast<std::int32_t>(patchOffsetZ[q]));
            EXPECT_LT(cellZ[static_cast<std::size_t>(s)],
                      static_cast<std::int32_t>(patchOffsetZ[q] + patchExtentZ[q]));
        }

    // Patch 0 owns z=0 (input 0, 2), patch 1 owns z=2 (input 1, 3); the y
    // values prove that every input particle appears exactly once.
    EXPECT_EQ(cellZ, (std::vector<std::int32_t>{0, 0, 2, 2}));
    EXPECT_EQ(cellY, (std::vector<std::int32_t>{0, 2, 1, 3}));
    std::filesystem::remove_all(temporaryDirectory());
}

namespace
{
    //! One particlePatches row read back through the openPMD API.
    struct PatchRow
    {
        std::vector<std::uint64_t> numParticles;
        std::vector<std::uint64_t> numParticlesOffset;
        std::array<std::vector<std::uint64_t>, 3> offset;
        std::array<std::vector<std::uint64_t>, 3> extent;
    };

    std::vector<std::uint64_t> loadPatchComponent(
        openPMD::Series& series,
        openPMD::RecordComponent component)
    {
        auto const extent = component.getExtent();
        std::uint64_t count = 1u;
        for(auto const length : extent)
            count *= length;
        std::vector<std::uint64_t> values(static_cast<std::size_t>(count));
        if(count != 0u)
        {
            auto raw = component.loadChunk<std::uint64_t>(openPMD::Offset{0}, extent);
            series.flush();
            for(std::size_t i = 0; i < values.size(); ++i)
                values[i] = raw.get()[i];
        }
        return values;
    }

    //! Read the patch metadata of one species from either backend.
    PatchRow readPatchRow(openPMD::Series& series, std::uint64_t step, std::string const& species)
    {
        auto& patches = series.iterations[step].particles[species].particlePatches;
        PatchRow row;
        row.numParticles = loadPatchComponent(
            series, patches["numParticles"][openPMD::RecordComponent::SCALAR]);
        row.numParticlesOffset = loadPatchComponent(
            series, patches["numParticlesOffset"][openPMD::RecordComponent::SCALAR]);
        for(std::size_t a = 0; a < 3u; ++a)
        {
            std::string const axis(1, "xyz"[a]);
            row.offset[a] = loadPatchComponent(series, patches["offset"][axis]);
            row.extent[a] = loadPatchComponent(series, patches["extent"][axis]);
        }
        return row;
    }

    //! A small generated bunch with distinct positions per particle.
    picBunch::GeneratedParticles generatedBunch(std::string name, std::size_t count, double offset)
    {
        picBunch::GeneratedParticles generated;
        generated.species = std::move(name);
        generated.bunch.resize(count);
        for(std::size_t p = 0; p < count; ++p)
        {
            generated.bunch.x[p] = offset + static_cast<double>(p);
            generated.bunch.y[p] = 1.0 + static_cast<double>(p);
            generated.bunch.z[p] = 2.0 + static_cast<double>(p);
            generated.bunch.ux[p] = 0.1 * static_cast<double>(p);
            generated.bunch.uy[p] = 0.0;
            generated.bunch.uz[p] = 1.0;
            generated.bunch.weight[p] = 1.0;
            generated.bunch.charge[p] = -picBunch::kElementaryCharge;
            generated.bunch.restMass[p] = picBunch::kElectronMass;
        }
        generated.domainOrigin = {0.0, 0.0, 0.0};
        generated.cellSize = {1.0, 1.0, 1.0};
        generated.patchOffset = {0, 0, 0};
        generated.patchExtent = {64, 64, 64};
        generated.unitMassSpeed = picBunch::kElectronMass * picBunch::kSpeedOfLight;
        generated.unitCharge = picBunch::kElementaryCharge;
        generated.unitMass = picBunch::kElectronMass;
        generated.fallbackMass = picBunch::kElectronMass;
        return generated;
    }
} // namespace

TEST(Restart, WriterGivesTwoGeneratedSpeciesDisjointIdRanges)
{
    auto const directory = temporaryDirectory() / "checkpoints_two_generated";
    std::filesystem::remove_all(temporaryDirectory());
    std::filesystem::create_directories(temporaryDirectory());
    auto const input = temporaryDirectory() / "input_two.h5";
    writeParticleInput(input, true, false, false, {"a", "b"});

    picBunch::PicOnGPUWriteOptions options;
    options.step = 6;
    options.directory = directory.string();
    options.inputFile = input.string();
    options.inputIteration = 0;
    options.requiredInputSpecies = {"a", "b"};
    options.unitLength = 1.0;
    options.unitEField = 1.0;
    options.unitBField = 1.0;
    options.generatedSpecies = {
        generatedBunch("a", 2u, 0.0),
        generatedBunch("b", 3u, 10.0)};
    picBunch::writePicOnGPUCheckpointOpenPMD(makeField(), options);

    auto const path = directory / "checkpoint_000006.h5";
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    ASSERT_GE(file, 0);
    EXPECT_EQ(datasetPoints(file, "data/6/particles/a/position/x"), 2u);
    EXPECT_EQ(datasetPoints(file, "data/6/particles/b/position/x"), 3u);
    auto const idsA = readUint64Vector(file, "data/6/particles/a/id");
    auto const idsB = readUint64Vector(file, "data/6/particles/b/id");
    EXPECT_EQ(idsA, (std::vector<std::uint64_t>{0u, 1u}));
    EXPECT_EQ(idsB, (std::vector<std::uint64_t>{2u, 3u, 4u}));
    EXPECT_EQ(readUint64Dataset(file, "data/6/fields/picongpu_idProvider/nextId"), 5u);
    H5Fclose(file);
    std::filesystem::remove_all(temporaryDirectory());
}

TEST(Restart, WriterEmitsZeroParticleEmptySpeciesWithoutCharge)
{
    auto const directory = temporaryDirectory() / "checkpoints_empty_species";
    std::filesystem::remove_all(temporaryDirectory());
    std::filesystem::create_directories(temporaryDirectory());
    auto const input = temporaryDirectory() / "input_b.h5";
    writeParticleInput(input, true, false, false, {"b"});

    auto const field = makeField();
    picBunch::PicOnGPUWriteOptions options;
    options.step = 7;
    options.directory = directory.string();
    options.inputFile = input.string();
    options.inputIteration = 0;
    options.requiredInputSpecies = {"b"};
    options.unitLength = 1.0;
    options.unitEField = 1.0;
    options.unitBField = 1.0;

    picBunch::GeneratedParticles generatedB = generatedBunch("b", 2u, 0.0);
    generatedB.inputSpecies = "b";
    picBunch::GeneratedParticles empty;
    empty.species = "e";
    empty.empty = true;
    empty.domainOrigin = {0.0, 0.0, 0.0};
    empty.cellSize = {0.5, 1.5, 2.5};
    empty.patchOffset = {0, 0, 0};
    empty.patchExtent = {4, 5, 6};
    empty.unitMassSpeed = picBunch::kElectronMass * picBunch::kSpeedOfLight;
    empty.unitCharge = picBunch::kElementaryCharge;
    empty.unitMass = picBunch::kElectronMass;
    empty.fallbackMass = picBunch::kElectronMass;
    options.generatedSpecies = {generatedB, empty};
    picBunch::writePicOnGPUCheckpointOpenPMD(field, options);

    auto const path = directory / "checkpoint_000007.h5";
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    ASSERT_GE(file, 0);

    // The empty group exists and every standard record is present and empty.
    EXPECT_EQ(H5Lexists(file, "data/7/particles/e", H5P_DEFAULT), 1);
    for(char const* record : {
            "position/x", "position/y", "position/z",
            "positionOffset/x", "positionOffset/y", "positionOffset/z",
            "momentum/x", "momentum/y", "momentum/z",
            "weighting", "id", "charge/charge", "mass/mass"})
    {
        std::string const pathInFile = std::string("data/7/particles/e/") + record;
        expectEmptyComponent(file, pathInFile);
    }
    // One domain-covering patch row with zero particles.
    EXPECT_EQ(
        readUint64Vector(file, "data/7/particles/e/particlePatches/numParticles"),
        (std::vector<std::uint64_t>{0u}));
    EXPECT_EQ(
        readUint64Vector(file, "data/7/particles/e/particlePatches/numParticlesOffset"),
        (std::vector<std::uint64_t>{0u}));
    EXPECT_EQ(
        readUint64Vector(file, "data/7/particles/e/particlePatches/offset/x"),
        (std::vector<std::uint64_t>{0u}));
    EXPECT_EQ(
        readUint64Vector(file, "data/7/particles/e/particlePatches/extent/x"),
        (std::vector<std::uint64_t>{4u}));
    EXPECT_EQ(
        readUint64Vector(file, "data/7/particles/e/particlePatches/extent/z"),
        (std::vector<std::uint64_t>{6u}));
    // The momentum metadata stays the PIConGPU macro-weighting contract.
    hid_t momentum = H5Gopen2(file, "data/7/particles/e/momentum", H5P_DEFAULT);
    ASSERT_GE(momentum, 0);
    EXPECT_EQ(
        readScalarAttribute<std::uint32_t>(momentum, "macroWeighted", H5T_NATIVE_UINT32),
        1u);
    EXPECT_DOUBLE_EQ(
        readScalarAttribute<double>(momentum, "weightingPower", H5T_NATIVE_DOUBLE),
        1.0);
    H5Gclose(momentum);
    // Zero particles add no charge: the field is written verbatim.
    auto const written = readFloatDataset(file, "data/7/fields/E/x");
    ASSERT_EQ(written.size(), field.data[0].size());
    for(std::size_t i = 0; i < written.size(); ++i)
        EXPECT_EQ(written[i], static_cast<float>(field.data[0][i]));
    // nextId covers only the two real particles.
    EXPECT_EQ(readUint64Dataset(file, "data/7/fields/picongpu_idProvider/nextId"), 2u);
    H5Fclose(file);
    std::filesystem::remove_all(temporaryDirectory());
}

TEST(Restart, EmptySpeciesCollidingWithInputKeepsDomainPatchMetadata)
{
    auto const directory = temporaryDirectory() / "checkpoints_empty_collision";
    std::filesystem::remove_all(temporaryDirectory());
    std::filesystem::create_directories(temporaryDirectory());
    // The input carries a same-named "e" group with stale patch metadata.  The
    // checkpoint-only "e" must replace it, not copy it and overwrite only the
    // particle records.
    auto const input = temporaryDirectory() / "input_be.h5";
    writeParticleInput(input, true, false, false, {"b", "e"});

    auto const field = makeField();
    picBunch::PicOnGPUWriteOptions options;
    options.step = 9;
    options.directory = directory.string();
    options.inputFile = input.string();
    options.inputIteration = 0;
    options.requiredInputSpecies = {"b"};
    options.unitLength = 1.0;
    options.unitEField = 1.0;
    options.unitBField = 1.0;

    picBunch::GeneratedParticles generatedB = generatedBunch("b", 2u, 0.0);
    generatedB.inputSpecies = "b";
    picBunch::GeneratedParticles empty;
    empty.species = "e";
    empty.empty = true;
    empty.domainOrigin = {0.0, 0.0, 0.0};
    empty.cellSize = {0.5, 1.5, 2.5};
    empty.patchOffset = {0, 0, 0};
    empty.patchExtent = {4, 5, 6};
    empty.unitMassSpeed = picBunch::kElectronMass * picBunch::kSpeedOfLight;
    empty.unitCharge = picBunch::kElementaryCharge;
    empty.unitMass = picBunch::kElectronMass;
    empty.fallbackMass = picBunch::kElectronMass;
    options.generatedSpecies = {generatedB, empty};

    auto const check = [&](picBunch::CheckpointBackend backend, std::filesystem::path const& path) {
        options.backend = backend;
        std::filesystem::remove_all(path);
        picBunch::writePicOnGPUCheckpointOpenPMD(field, options);
        ASSERT_TRUE(std::filesystem::exists(path)) << path;
        openPMD::Series series(path.string(), openPMD::Access::READ_ONLY);
        auto const row = readPatchRow(series, options.step, "e");
        EXPECT_EQ(row.numParticles, (std::vector<std::uint64_t>{0u}));
        EXPECT_EQ(row.numParticlesOffset, (std::vector<std::uint64_t>{0u}));
        for(std::size_t a = 0; a < 3u; ++a)
        {
            EXPECT_EQ(row.offset[a], (std::vector<std::uint64_t>{0u})) << "axis " << a;
            EXPECT_EQ(
                row.extent[a],
                (std::vector<std::uint64_t>{static_cast<std::uint64_t>(field.grid.n[a])}))
                << "axis " << a;
        }
    };

    check(picBunch::CheckpointBackend::Hdf5, directory / "checkpoint_000009.h5");
#if defined(openPMD_HAVE_ADIOS2) && openPMD_HAVE_ADIOS2
    check(picBunch::CheckpointBackend::Adios2, directory / "checkpoint_000009.bp5");
#endif
    std::filesystem::remove_all(temporaryDirectory());
}

TEST(Restart, RenamedGeneratedSpeciesKeepsTheRawInputCopy)
{
    auto const directory = temporaryDirectory() / "checkpoints_renamed";
    std::filesystem::remove_all(temporaryDirectory());
    std::filesystem::create_directories(temporaryDirectory());
    auto const input = temporaryDirectory() / "input_b.h5";
    writeParticleInput(input, true, false, false, {"b"});

    picBunch::PicOnGPUWriteOptions options;
    options.step = 8;
    options.directory = directory.string();
    options.inputFile = input.string();
    options.inputIteration = 0;
    options.requiredInputSpecies = {"b"};
    options.unitLength = 1.0;
    options.unitEField = 1.0;
    options.unitBField = 1.0;
    picBunch::GeneratedParticles renamed = generatedBunch("e", 2u, 3.0);
    renamed.inputSpecies = "b";
    options.generatedSpecies = {renamed};
    picBunch::writePicOnGPUCheckpointOpenPMD(makeField(), options);

    auto const path = directory / "checkpoint_000008.h5";
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    ASSERT_GE(file, 0);
    // The renamed output group and the preserved raw input group both exist.
    EXPECT_EQ(H5Lexists(file, "data/8/particles/e/position/x", H5P_DEFAULT), 1);
    EXPECT_EQ(H5Lexists(file, "data/8/particles/b/position/x", H5P_DEFAULT), 1);
    EXPECT_EQ(datasetPoints(file, "data/8/particles/e/position/x"), 2u);
    EXPECT_EQ(datasetPoints(file, "data/8/particles/b/position/x"), 1u);
    // Generated ids start after the copied range [0, 1).
    EXPECT_EQ(
        readUint64Vector(file, "data/8/particles/e/id"),
        (std::vector<std::uint64_t>{1u, 2u}));
    EXPECT_EQ(readUint64Dataset(file, "data/8/fields/picongpu_idProvider/nextId"), 3u);
    H5Fclose(file);
    std::filesystem::remove_all(temporaryDirectory());
}
