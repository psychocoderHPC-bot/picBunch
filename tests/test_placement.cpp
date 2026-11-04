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
        std::array<double, 3> const values{0.0, 0.0, 0.0};
        H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
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
    config.picongpu.species = "electrons";
    config.domain.cells = {10, 10, 10};
    config.domain.cellSize = {1.0, 1.0, 1.0};
    config.domain.origin = {0.0, 0.0, 0.0};
    config.placement.fraction = true;
    config.placement.offset = {0.5, 0.5, 0.5};
    config.accuracy.tolerance = 1e-3;
    config.accuracy.nearShell = "none";

    picBunch::Bunch bunch;
    bunch.resize(3);
    std::array<std::array<double, 3>, 3> const positions{{{1.0, 2.0, 3.0}, {3.0, 4.0, 5.0}, {5.0, 6.0, 7.0}}};
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
    EXPECT_EQ(placed.patchOffset, (std::array<std::int64_t, 3>{3, 3, 3}));
    EXPECT_EQ(placed.patchExtent, (std::array<std::uint64_t, 3>{5, 5, 5}));

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
    options.species = "electrons";
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
    generated.fallbackMass = config.picongpu.unitMass;
    options.generatedParticles = generated;
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
    H5Fclose(file);
    std::filesystem::remove_all(directory);
}
