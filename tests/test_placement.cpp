// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#include "picBunch/Config.hpp"
#include "picBunch/OpenPMDWriter.hpp"
#include "picBunch/Placement.hpp"
#include "picBunch/Types.hpp"

#include <gtest/gtest.h>

#include <hdf5.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    void writeMinimalInput(std::filesystem::path const& path)
    {
        hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        H5Gcreate2(file, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t iteration = H5Gcreate2(file, "data/0", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Gcreate2(iteration, "particles", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t species = H5Gcreate2(file, "data/0/particles/electrons", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Gcreate2(species, "position", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hsize_t const count = 3;
        hid_t space = H5Screate_simple(1, &count, nullptr);
        hid_t dataset = H5Dcreate2(
            file, "data/0/particles/electrons/position/x", H5T_IEEE_F64LE, space,
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        picBunch::Vec3 const values{0.0, 0.0, 0.0};
        H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &values[0]);
        H5Dclose(dataset);
        H5Sclose(space);
        H5Gclose(species);
        H5Gclose(iteration);
        H5Fclose(file);
    }

    std::vector<double> readVector(hid_t file, std::string const& path)
    {
        hid_t dataset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
        hid_t space = H5Dget_space(dataset);
        hssize_t const count = H5Sget_simple_extent_npoints(space);
        std::vector<double> values(static_cast<std::size_t>(count));
        H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
        H5Sclose(space);
        H5Dclose(dataset);
        return values;
    }

    std::uint64_t readScalar(hid_t file, std::string const& path)
    {
        hid_t dataset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
        std::uint64_t value = 0;
        H5Dread(dataset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
        H5Dclose(dataset);
        return value;
    }

    struct OnDiskType
    {
        H5T_class_t klass;
        std::size_t size;
    };

    OnDiskType readType(hid_t file, std::string const& path)
    {
        hid_t dataset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
        if(dataset < 0)
            throw std::runtime_error("cannot open dataset " + path);
        hid_t type = H5Dget_type(dataset);
        OnDiskType const result{H5Tget_class(type), H5Tget_size(type)};
        H5Tclose(type);
        H5Dclose(dataset);
        return result;
    }
} // namespace

TEST(Placement, CentroidPlacementAndPatchAreWritten)
{
    picBunch::Config config;
    config.picongpu.unitLength = 1.0;
    config.picongpu.unitEField = 1.0;
    config.picongpu.unitBField = 1.0;
    config.picongpu.unitMass = picBunch::kElectronMass;
    config.picongpu.unitSpeed = picBunch::kSpeedOfLight;
    config.picongpu.step = 3;
    config.picongpu.species = {"electrons"};
    config.domain.cells = {10, 10, 10};
    config.domain.cellSize = {1.0, 1.0, 1.0};
    config.domain.origin = {0.0, 0.0, 0.0};
    config.placement.fraction = true;
    config.placement.offset = {0.5, 0.5, 0.5};
    config.accuracy.tolerance = 1e-3;
    config.accuracy.nearShell = "none";

    picBunch::Bunch bunch;
    bunch.resize(3);
    std::array<picBunch::Vec3, 3> const positions{{{1.0, 2.0, 3.0}, {3.0, 4.0, 5.0}, {5.0, 6.0, 7.0}}};
    for(std::size_t p = 0; p < 3; ++p)
    {
        bunch.x[p] = positions[p][0];
        bunch.y[p] = positions[p][1];
        bunch.z[p] = positions[p][2];
        bunch.ux[p] = 0.0;
        bunch.uy[p] = 0.0;
        bunch.uz[p] = 0.0;
        bunch.weight[p] = 1.0;
        bunch.charge[p] = -picBunch::kElementaryCharge;
        bunch.restMass[p] = picBunch::kElectronMass;
    }

    auto const placed = picBunch::placeBunch(bunch, config);
    EXPECT_NEAR(placed.centroid[0], 3.0, 1e-12);
    EXPECT_NEAR(placed.centroid[1], 4.0, 1e-12);
    EXPECT_NEAR(placed.centroid[2], 5.0, 1e-12);
    for(int a = 0; a < 3; ++a)
        EXPECT_NEAR(placed.targetCentre[a], 5.0, 1e-12);
    EXPECT_EQ(placed.patchOffset, (picBunch::IVec3{3, 3, 3}));
    EXPECT_EQ(placed.patchExtent, (picBunch::UVec3{5, 5, 5}));

    std::filesystem::path const directory
        = std::filesystem::temp_directory_path() / "picbunch-placement-test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    auto const input = directory / "input.h5";
    writeMinimalInput(input);

    picBunch::Field field;
    field.grid.n = {10, 10, 10};
    field.grid.spacing = {1.0, 1.0, 1.0};
    field.grid.origin = {0.0, 0.0, 0.0};
    field.grid.fftPadding = 2;
    for(auto& component : field.data)
        component.assign(field.grid.size(), 0.0);

    picBunch::PicOnGPUWriteOptions options;
    options.step = 3;
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
    generated.bunch = placed.bunch;
    generated.domainOrigin = config.domain.origin;
    generated.cellSize = config.domain.cellSize;
    generated.patchOffset = placed.patchOffset;
    generated.patchExtent = placed.patchExtent;
    generated.unitMassSpeed = config.picongpu.unitMass * config.picongpu.unitSpeed;
    generated.unitCharge = config.picongpu.unitCharge;
    generated.unitMass = config.picongpu.unitMass;
    generated.fallbackMass = config.picongpu.unitMass;
    options.generatedSpecies = {generated};
    picBunch::writePicOnGPUCheckpoint(field, options);

    auto const path = directory / "checkpoint_000003.h5";
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    auto const px = readVector(file, "data/3/particles/electrons/position/x");
    auto const py = readVector(file, "data/3/particles/electrons/position/y");
    auto const pz = readVector(file, "data/3/particles/electrons/position/z");
    std::vector<std::int32_t> ox(3), oy(3), oz(3);
    std::vector<std::int32_t>* offsets[3] = {&ox, &oy, &oz};
    char const* axisNames[3] = {"x", "y", "z"};
    for(int a = 0; a < 3; ++a)
    {
        hid_t dataset = H5Dopen2(
            file,
            (std::string("data/3/particles/electrons/positionOffset/") + axisNames[a]).c_str(),
            H5P_DEFAULT);
        H5Dread(dataset, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, offsets[a]->data());
        H5Dclose(dataset);
    }
    for(std::size_t p = 0; p < 3; ++p)
    {
        double const x = (ox[p] + px[p]) * config.domain.cellSize[0] + config.domain.origin[0];
        double const y = (oy[p] + py[p]) * config.domain.cellSize[1] + config.domain.origin[1];
        double const z = (oz[p] + pz[p]) * config.domain.cellSize[2] + config.domain.origin[2];
        EXPECT_NEAR(x, placed.bunch.x[p], 1e-9);
        EXPECT_NEAR(y, placed.bunch.y[p], 1e-9);
        EXPECT_NEAR(z, placed.bunch.z[p], 1e-9);
    }
    double const meanX = ((ox[0] + px[0]) + (ox[1] + px[1]) + (ox[2] + px[2])) / 3.0;
    EXPECT_NEAR(meanX, 5.0, 1e-9);
    EXPECT_EQ(readScalar(file, "data/3/particles/electrons/particlePatches/numParticles"), 3u);
    EXPECT_EQ(readScalar(file, "data/3/particles/electrons/particlePatches/offset/x"), 3u);
    EXPECT_EQ(readScalar(file, "data/3/particles/electrons/particlePatches/extent/x"), 5u);

    // charge and mass are stored in internal units; the input bunch carries the
    // SI values, so the record is charge/unit_charge and restMass/unit_mass.
    auto const chargeRecord = readVector(file, "data/3/particles/electrons/charge/charge");
    auto const massRecord = readVector(file, "data/3/particles/electrons/mass/mass");
    ASSERT_EQ(chargeRecord.size(), 3u);
    ASSERT_EQ(massRecord.size(), 3u);
    for(std::size_t p = 0; p < 3; ++p)
    {
        EXPECT_NEAR(chargeRecord[p], bunch.charge[p] / config.picongpu.unitCharge, 1e-12);
        EXPECT_NEAR(massRecord[p], bunch.restMass[p] / config.picongpu.unitMass, 1e-12);
    }

    // On-disk dtype per record for the default float32 configuration. Every
    // floating-point field and particle property is float32; the integer
    // records keep their PIConGPU layout.
    auto const positionType = readType(file, "data/3/particles/electrons/position/x");
    EXPECT_EQ(positionType.klass, H5T_FLOAT);
    EXPECT_EQ(positionType.size, 4u);
    auto const weightingType = readType(file, "data/3/particles/electrons/weighting");
    EXPECT_EQ(weightingType.klass, H5T_FLOAT);
    EXPECT_EQ(weightingType.size, 4u);
    auto const momentumType = readType(file, "data/3/particles/electrons/momentum/x");
    EXPECT_EQ(momentumType.klass, H5T_FLOAT);
    EXPECT_EQ(momentumType.size, 4u);
    auto const offsetType = readType(file, "data/3/particles/electrons/positionOffset/x");
    EXPECT_EQ(offsetType.klass, H5T_INTEGER);
    EXPECT_EQ(offsetType.size, 4u);
    auto const idType = readType(file, "data/3/particles/electrons/id");
    EXPECT_EQ(idType.klass, H5T_INTEGER);
    EXPECT_EQ(idType.size, 8u);
    auto const chargeType = readType(file, "data/3/particles/electrons/charge/charge");
    EXPECT_EQ(chargeType.klass, H5T_FLOAT);
    EXPECT_EQ(chargeType.size, 4u);
    auto const massType = readType(file, "data/3/particles/electrons/mass/mass");
    EXPECT_EQ(massType.klass, H5T_FLOAT);
    EXPECT_EQ(massType.size, 4u);
    EXPECT_EQ(readType(file, "data/3/fields/E/x").size, 4u);
    EXPECT_EQ(readType(file, "data/3/fields/B/x").size, 4u);
    H5Fclose(file);

    // float64 applies to every floating-point field and particle property; the
    // integer records do not change, and the warning is recorded.
    options.float64 = true;
    picBunch::writePicOnGPUCheckpoint(field, options);
    file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    EXPECT_EQ(readType(file, "data/3/particles/electrons/position/x").size, 8u);
    EXPECT_EQ(readType(file, "data/3/particles/electrons/weighting").size, 8u);
    EXPECT_EQ(readType(file, "data/3/particles/electrons/momentum/x").size, 8u);
    EXPECT_EQ(readType(file, "data/3/particles/electrons/charge/charge").size, 8u);
    EXPECT_EQ(readType(file, "data/3/particles/electrons/mass/mass").size, 8u);
    EXPECT_EQ(readType(file, "data/3/particles/electrons/positionOffset/x").size, 4u);
    EXPECT_EQ(readType(file, "data/3/particles/electrons/id").size, 8u);
    EXPECT_EQ(readType(file, "data/3/fields/E/x").size, 8u);
    EXPECT_EQ(readType(file, "data/3/fields/B/x").size, 8u);
    hid_t attribute = H5Aopen(file, "picBunchPrecisionWarning", H5P_DEFAULT);
    ASSERT_GE(attribute, 0);
    char* warning = nullptr;
    hid_t attributeType = H5Aget_type(attribute);
    ASSERT_EQ(H5Aread(attribute, attributeType, &warning), 0);
    H5Tclose(attributeType);
    H5Aclose(attribute);
    std::string const warningText = warning != nullptr ? warning : "";
    H5free_memory(warning);
    EXPECT_NE(
        warningText.find("stock PIConGPU cannot load double particle records"),
        std::string::npos);
    H5Fclose(file);
    std::filesystem::remove_all(directory);
}

TEST(Placement, DomainPatchCoversTheWholeDomain)
{
    picBunch::Config config;
    config.domain.cells = {10, 12, 14};
    config.domain.cellSize = {1.0, 1.0, 1.0};
    config.domain.origin = {0.0, 0.0, 0.0};
    config.placement.fraction = true;
    config.placement.offset = {0.5, 0.5, 0.5};
    config.placement.patch = picBunch::Config::Placement::Patch::Domain;

    picBunch::Bunch bunch;
    bunch.resize(3);
    std::array<picBunch::Vec3, 3> const positions{{{1.0, 2.0, 3.0}, {3.0, 4.0, 5.0}, {5.0, 6.0, 7.0}}};
    for(std::size_t p = 0; p < 3; ++p)
    {
        bunch.x[p] = positions[p][0];
        bunch.y[p] = positions[p][1];
        bunch.z[p] = positions[p][2];
        bunch.weight[p] = 1.0;
        bunch.charge[p] = -picBunch::kElementaryCharge;
        bunch.restMass[p] = picBunch::kElectronMass;
    }

    auto const placed = picBunch::placeBunch(bunch, config);
    EXPECT_EQ(placed.patchOffset, (picBunch::IVec3{0, 0, 0}));
    EXPECT_EQ(placed.patchExtent, (picBunch::UVec3{10, 12, 14}));
    // positionOffset still indexes the global cells, unchanged by the patch mode.
    EXPECT_NEAR(placed.centroid[0], 3.0, 1e-12);
}

TEST(Placement, PatchCountSplitsTheRegionWithCumulativeCoverage)
{
    picBunch::Config config;
    config.domain.cells = {10, 12, 32};
    config.domain.cellSize = {1.0, 1.0, 1.0};
    config.domain.origin = {0.0, 0.0, 0.0};
    config.placement.fraction = true;
    config.placement.offset = {0.5, 0.5, 0.5};
    config.placement.patch = picBunch::Config::Placement::Patch::Domain;
    config.placement.patchCount = 4;

    picBunch::Bunch bunch;
    bunch.resize(3);
    std::array<picBunch::Vec3, 3> const positions{{{1.0, 2.0, 3.0}, {3.0, 4.0, 5.0}, {5.0, 6.0, 7.0}}};
    for(std::size_t p = 0; p < 3; ++p)
    {
        bunch.x[p] = positions[p][0];
        bunch.y[p] = positions[p][1];
        bunch.z[p] = positions[p][2];
        bunch.weight[p] = 1.0;
        bunch.charge[p] = -picBunch::kElementaryCharge;
        bunch.restMass[p] = picBunch::kElectronMass;
    }

    auto const placed = picBunch::placeBunch(bunch, config);
    ASSERT_EQ(placed.patches.size(), 4u);
    // The z slices tile the domain; neighbours overlap by one cell and the
    // union covers [0, 32).  x/y stay at the full domain patch.
    EXPECT_EQ(placed.patches.front().offset[2], 0);
    EXPECT_EQ(placed.patches.back().offset[2] + static_cast<std::int64_t>(
                  placed.patches.back().extent[2]), 32);
    for(std::size_t q = 0; q < placed.patches.size(); ++q)
    {
        EXPECT_EQ(placed.patches[q].offset[0], 0);
        EXPECT_EQ(placed.patches[q].extent[0], 10u);
        EXPECT_EQ(placed.patches[q].extent[1], 12u);
        if(q + 1u < placed.patches.size())
        {
            auto const& current = placed.patches[q];
            auto const& next = placed.patches[q + 1u];
            EXPECT_LE(
                current.offset[2] + static_cast<std::int64_t>(current.extent[2]),
                next.offset[2] + static_cast<std::int64_t>(next.extent[2]));
            EXPECT_LE(next.offset[2], current.offset[2] + static_cast<std::int64_t>(current.extent[2]));
        }
    }
}

TEST(Placement, OutOfDomainIsRejectedWithoutClipping)
{
    picBunch::Config config;
    config.domain.cells = {10, 10, 10};
    config.domain.cellSize = {1.0, 1.0, 1.0};
    config.domain.origin = {0.0, 0.0, 0.0};
    config.placement.fraction = false;
    config.placement.offset = {9.0, 5.0, 5.0};

    picBunch::Bunch bunch;
    bunch.resize(2);
    bunch.x = {1.0, 9.0};
    bunch.y = {4.0, 6.0};
    bunch.z = {4.0, 6.0};
    for(std::size_t p = 0; p < 2; ++p)
    {
        bunch.weight[p] = 1.0;
        bunch.charge[p] = -picBunch::kElementaryCharge;
        bunch.restMass[p] = picBunch::kElectronMass;
    }

    // charge-weighted centroid is x=5, shift=+4, so the bunch spans x=[5,13]
    // and the upper edge is 3 m (3 cells) outside the 10 m domain.
    try
    {
        (void)picBunch::placeBunch(bunch, config);
        FAIL() << "expected an out-of-domain placement error";
    }
    catch(std::exception const& error)
    {
        std::string const message = error.what();
        EXPECT_NE(message.find("axis x"), std::string::npos);
        EXPECT_NE(message.find("outside"), std::string::npos);
        EXPECT_NE(message.find("3"), std::string::npos);
    }
    // The input is untouched: no particle was clipped or moved.
    EXPECT_DOUBLE_EQ(bunch.x[1], 9.0);
    EXPECT_DOUBLE_EQ(bunch.y[1], 6.0);
}

TEST(Placement, CenterNoneKeepsAbsoluteInputCoordinates)
{
    picBunch::Config config;
    config.domain.cells = {12, 12, 12};
    config.domain.cellSize = {1.0, 1.0, 1.0};
    config.domain.origin = {0.0, 0.0, 0.0};
    // A non-zero offset must be ignored when the input positions are kept.
    config.placement.fraction = true;
    config.placement.offset = {0.9, 0.9, 0.9};
    config.placement.centre = picBunch::Config::Placement::Centre::None;

    picBunch::Bunch bunch;
    std::array<picBunch::Vec3, 3> const positions{
        {{1.0, 2.0, 3.0}, {3.0, 4.0, 5.0}, {5.0, 6.0, 7.0}}};
    bunch.resize(positions.size());
    for(std::size_t p = 0; p < positions.size(); ++p)
    {
        bunch.x[p] = positions[p][0];
        bunch.y[p] = positions[p][1];
        bunch.z[p] = positions[p][2];
        bunch.ux[p] = bunch.uy[p] = bunch.uz[p] = 0.0;
        bunch.weight[p] = 1.0;
        bunch.charge[p] = -picBunch::kElementaryCharge;
        bunch.restMass[p] = picBunch::kElectronMass;
    }

    auto const placed = picBunch::placeBunch(bunch, config);
    for(int a = 0; a < 3; ++a)
    {
        EXPECT_EQ(placed.shift[a], 0.0);
        // The reported centroid is the input centroid, not the requested one.
        EXPECT_NEAR(placed.centroid[a], 3.0 + static_cast<double>(a), 1e-15);
        EXPECT_NEAR(placed.targetCentre[a], placed.centroid[a], 1e-15);
    }
    // Positions are copied bit-identically: no centroid shift is applied.
    for(std::size_t p = 0; p < positions.size(); ++p)
    {
        EXPECT_EQ(placed.bunch.x[p], positions[p][0]);
        EXPECT_EQ(placed.bunch.y[p], positions[p][1]);
        EXPECT_EQ(placed.bunch.z[p], positions[p][2]);
    }
    // The bounding-box patch still indexes the absolute input cells.
    EXPECT_EQ(placed.patchOffset, (picBunch::IVec3{1, 2, 3}));
    EXPECT_EQ(placed.patchExtent, (picBunch::UVec3{5, 5, 5}));
}

TEST(Placement, CenterCentroidStillAppliesTheShift)
{
    picBunch::Config config;
    config.domain.cells = {12, 12, 12};
    config.domain.cellSize = {1.0, 1.0, 1.0};
    config.domain.origin = {0.0, 0.0, 0.0};
    config.placement.fraction = true;
    config.placement.offset = {0.5, 0.5, 0.5};
    config.placement.centre = picBunch::Config::Placement::Centre::Centroid;

    picBunch::Bunch bunch;
    bunch.resize(2);
    bunch.x = {1.0, 3.0};
    bunch.y = {1.0, 3.0};
    bunch.z = {1.0, 3.0};
    for(std::size_t p = 0; p < 2; ++p)
    {
        bunch.weight[p] = 1.0;
        bunch.charge[p] = -picBunch::kElementaryCharge;
        bunch.restMass[p] = picBunch::kElectronMass;
    }
    auto const placed = picBunch::placeBunch(bunch, config);
    EXPECT_NEAR(placed.centroid[0], 2.0, 1e-15);
    EXPECT_NEAR(placed.targetCentre[0], 6.0, 1e-15);
    EXPECT_NEAR(placed.shift[0], 4.0, 1e-15);
    EXPECT_NEAR(placed.bunch.x[0], 5.0, 1e-15);
    EXPECT_NEAR(placed.bunch.x[1], 7.0, 1e-15);
}

TEST(Placement, CenterNoneRejectsAbsoluteOutOfDomain)
{
    picBunch::Config config;
    config.domain.cells = {10, 10, 10};
    config.domain.cellSize = {1.0, 1.0, 1.0};
    config.domain.origin = {0.0, 0.0, 0.0};
    config.placement.centre = picBunch::Config::Placement::Centre::None;

    picBunch::Bunch bunch;
    bunch.resize(2);
    bunch.x = {5.0, 13.0};
    bunch.y = {4.0, 6.0};
    bunch.z = {4.0, 6.0};
    for(std::size_t p = 0; p < 2; ++p)
    {
        bunch.weight[p] = 1.0;
        bunch.charge[p] = -picBunch::kElementaryCharge;
        bunch.restMass[p] = picBunch::kElectronMass;
    }
    try
    {
        (void)picBunch::placeBunch(bunch, config);
        FAIL() << "expected an out-of-domain placement error";
    }
    catch(std::exception const& error)
    {
        std::string const message = error.what();
        EXPECT_NE(message.find("axis x"), std::string::npos) << message;
        EXPECT_NE(message.find("outside"), std::string::npos) << message;
    }
    // The input is untouched: no particle was clipped or moved.
    EXPECT_DOUBLE_EQ(bunch.x[1], 13.0);
}
