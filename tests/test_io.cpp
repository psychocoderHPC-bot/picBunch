// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group 4: openPMD writer/reader round-trip.  Values must be bit-identical and
// every metadata item the writer sets must survive: unitSI, Yee position,
// gridSpacing, gridGlobalOffset, axis labels, dataOrder, geometry,
// unitDimension and the iteration index.

#include "picBunch/Compute.hpp"
#include "picBunch/OpenPMD.hpp"
#include "picBunch/OpenPMDWriter.hpp"

#include <openPMD/openPMD.hpp>
#include <openPMD/config.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace
{
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
                field.data[c][l] = 0.125 * static_cast<double>(c) + 1.0e-3 * static_cast<double>(l)
                    + 1.0e-12 * static_cast<double>(c * 7 + l);
        }
        return field;
    }

    std::filesystem::path temporaryPath()
    {
        return std::filesystem::temp_directory_path()
            / ("picbunch_io_" + std::to_string(static_cast<long long>(getpid())) + ".h5");
    }

    std::vector<double> loadComponent(
        openPMD::Series& series,
        openPMD::RecordComponent& component,
        std::array<std::size_t, 3> const& shape)
    {
        openPMD::Extent extent{
            static_cast<openPMD::Extent::value_type>(shape[2]),
            static_cast<openPMD::Extent::value_type>(shape[1]),
            static_cast<openPMD::Extent::value_type>(shape[0])};
        auto raw = component.loadChunk<double>(openPMD::Offset{0, 0, 0}, extent);
        series.flush();
        std::vector<double> values(static_cast<std::size_t>(extent[0] * extent[1] * extent[2]));
        for(std::size_t l = 0; l < values.size(); ++l)
            values[l] = raw.get()[l];
        return values;
    }
} // namespace

TEST(Io, FieldRoundTripPreservesValuesAndMetadata)
{
    auto const field = makeField();
    auto const path = temporaryPath();
    std::filesystem::remove(path);

    picBunch::FieldWriteOptions options;
    options.iteration = 7u;
    options.time = 2.5e-13;
    options.timeUnitSI = 1.0;
    options.timeOffset = 0.0;
    picBunch::writeField(field, path.string(), options);

    {
        openPMD::Series series(path.string(), openPMD::Access::READ_ONLY);
        ASSERT_TRUE(series.iterations.contains(options.iteration));
        auto& iteration = series.iterations[options.iteration];
        EXPECT_DOUBLE_EQ(iteration.time<double>(), options.time);
        EXPECT_DOUBLE_EQ(iteration.timeUnitSI(), options.timeUnitSI);
        ASSERT_TRUE(iteration.containsAttribute("timeOffset"));
        EXPECT_DOUBLE_EQ(iteration.getAttribute("timeOffset").get<double>(), 0.0);

        // Raw stored metadata: axis order z,y,x and spacing/offset reversed.
        ASSERT_TRUE(iteration.meshes.contains("E"));
        auto& meshE = iteration.meshes["E"];
        EXPECT_EQ(meshE.axisLabels(), (std::vector<std::string>{"z", "y", "x"}));
        EXPECT_EQ(meshE.dataOrder(), openPMD::Mesh::DataOrder::C);
        EXPECT_EQ(meshE.geometryString(), "cartesian");
        EXPECT_DOUBLE_EQ(meshE.gridUnitSI(), 1.0);
        auto const spacing = meshE.gridSpacing<double>();
        ASSERT_EQ(spacing.size(), 3u);
        EXPECT_DOUBLE_EQ(spacing[0], field.grid.spacing[2]);
        EXPECT_DOUBLE_EQ(spacing[1], field.grid.spacing[1]);
        EXPECT_DOUBLE_EQ(spacing[2], field.grid.spacing[0]);
        auto const offset = meshE.gridGlobalOffset();
        ASSERT_EQ(offset.size(), 3u);
        EXPECT_DOUBLE_EQ(offset[0], field.grid.origin[2]);
        EXPECT_DOUBLE_EQ(offset[1], field.grid.origin[1]);
        EXPECT_DOUBLE_EQ(offset[2], field.grid.origin[0]);

        auto const dimensionE = meshE.unitDimension();
        auto const dimensionB = iteration.meshes["B"].unitDimension();
        std::array<double, 7> const expectedE{1.0, 1.0, -3.0, -1.0, 0.0, 0.0, 0.0};
        std::array<double, 7> const expectedB{0.0, 1.0, -2.0, -1.0, 0.0, 0.0, 0.0};
        for(int i = 0; i < 7; ++i)
        {
            EXPECT_DOUBLE_EQ(dimensionE[i], expectedE[i]) << "unitDimension E index " << i;
            EXPECT_DOUBLE_EQ(dimensionB[i], expectedB[i]) << "unitDimension B index " << i;
        }

        for(int c = 0; c < 6; ++c)
        {
            std::string const recordName = (c < 3) ? "E" : "B";
            char const componentName = "xyz"[c % 3];
            auto& record = iteration.meshes[recordName];
            ASSERT_TRUE(record.contains(std::string(1, componentName)));
            auto& component = record[std::string(1, componentName)];

            // Values must match exactly.
            auto const values = loadComponent(series, component, field.grid.n);
            ASSERT_EQ(values.size(), field.data[c].size());
            for(std::size_t l = 0; l < values.size(); ++l)
                EXPECT_EQ(values[l], field.data[c][l]) << "component " << c << " value " << l;

            EXPECT_DOUBLE_EQ(component.unitSI(), 1.0);

            // position attribute in stored z,y,x order, mapped back to x,y,z.
            ASSERT_TRUE(component.containsAttribute("position"));
            auto const position = component.getAttribute("position").get<std::vector<double>>();
            ASSERT_EQ(position.size(), 3u);
            EXPECT_DOUBLE_EQ(position[0], picBunch::kYeeOffset[c][2]);
            EXPECT_DOUBLE_EQ(position[1], picBunch::kYeeOffset[c][1]);
            EXPECT_DOUBLE_EQ(position[2], picBunch::kYeeOffset[c][0]);
        }
    }

    // readMeshGeometry normalizes the stored order back to x,y,z.
    {
        auto const geometry = picBunch::readMeshGeometry(path.string(), "E", 0, options.iteration);
        EXPECT_EQ(geometry.shape, (std::array<std::size_t, 3>{4, 5, 6}));
        EXPECT_EQ(geometry.axisLabels, (std::array<std::string, 3>{"x", "y", "z"}));
        EXPECT_EQ(geometry.geometry, "cartesian");
        EXPECT_EQ(geometry.dataOrder, "C");
        for(int a = 0; a < 3; ++a)
        {
            EXPECT_DOUBLE_EQ(geometry.gridSpacing[a], field.grid.spacing[a]);
            EXPECT_DOUBLE_EQ(geometry.gridGlobalOffset[a], field.grid.origin[a]);
            EXPECT_DOUBLE_EQ(geometry.gridUnitSI[a], 1.0);
        }
        EXPECT_EQ(geometry.iteration, options.iteration);
    }

    std::filesystem::remove(path);
}

TEST(Io, MissingMacroWeightedUsesStandardDefault)
{
    // openPMD standard default is macroWeighted=false.  With weightingPower set
    // but macroWeighted absent, the momentum must remain a per-particle value:
    // the weightingPower must not be applied.  The pre-fix fallback of true
    // would divide u by weight^2.
    auto const path = temporaryPath();
    std::filesystem::remove(path);
    double const massC = picBunch::kElectronMass * picBunch::kSpeedOfLight;
    double const weightValue = 3.0;
    // The chunk buffers must outlive openPMD's deferred flush.
    std::array<double, 3> const positions{4.0, 4.0, 5.0};
    std::array<double, 3> const momenta{0.0, 0.0, 1.0};
    double const storedWeight = weightValue;
    {
        openPMD::Series series(path.string(), openPMD::Access::CREATE);
        auto& iteration = series.iterations[0];
        auto& species = iteration.particles["electrons"];
        char const* const names = "xyz";
        auto bind = [](openPMD::RecordComponent& component, double const* value, double unitSI) {
            component.resetDataset(
                openPMD::Dataset(openPMD::determineDatatype<double>(), openPMD::Extent{1}));
            component.storeChunkRaw(value, openPMD::Offset{0}, openPMD::Extent{1});
            component.setUnitSI(unitSI);
        };
        auto& position = species["position"];
        position.setUnitDimension({{openPMD::UnitDimension::L, 1.0}});
        for(int a = 0; a < 3; ++a)
            bind(position[std::string(1, names[a])], &positions[a], 1.0);

        auto& momentum = species["momentum"];
        momentum.setUnitDimension(
            {{openPMD::UnitDimension::L, 1.0},
             {openPMD::UnitDimension::M, 1.0},
             {openPMD::UnitDimension::T, -1.0}});
        for(int a = 0; a < 3; ++a)
            bind(momentum[std::string(1, names[a])], &momenta[a], massC);
        momentum.setAttribute("weightingPower", 2.0);
        // macroWeighted is deliberately absent -> standard default false.

        bind(species["weighting"], &storedWeight, 1.0);
        series.flush();
    }

    auto const data = picBunch::readBunch(path.string(), std::size_t{0});
    ASSERT_EQ(data.bunch.size(), 1u);
    EXPECT_DOUBLE_EQ(data.bunch.weight[0], weightValue);
    EXPECT_DOUBLE_EQ(data.bunch.ux[0], 0.0);
    EXPECT_DOUBLE_EQ(data.bunch.uy[0], 0.0);
    EXPECT_DOUBLE_EQ(data.bunch.uz[0], 1.0);
    // No charge record: documented default -e (a warning is emitted on stderr).
    EXPECT_DOUBLE_EQ(data.bunch.charge[0], -picBunch::kElementaryCharge);

    std::filesystem::remove(path);
}

namespace
{
    //! Write a tiny mesh whose grid metadata is stored in ``z,y,x`` order.
    void writeGeometryMesh(
        std::filesystem::path const& path,
        double gridUnitSI,
        bool setUnitSI,
        std::array<double, 3> const& spacingZYX = {0.5, 1.5, 2.5},
        std::array<double, 3> const& offsetZYX = {3.0, 2.0, 1.0})
    {
        std::array<double, 4 * 5 * 6> const values{};
        openPMD::Series series(path.string(), openPMD::Access::CREATE);
        auto& mesh = series.iterations[0].meshes["E"];
        mesh.setGeometry(openPMD::Mesh::Geometry::cartesian);
        mesh.setDataOrder(openPMD::Mesh::DataOrder::C);
        mesh.setAxisLabels(std::vector<std::string>{"z", "y", "x"});
        mesh.setGridSpacing(std::vector<double>(spacingZYX.begin(), spacingZYX.end()));
        mesh.setGridGlobalOffset(std::vector<double>(offsetZYX.begin(), offsetZYX.end()));
        if(setUnitSI)
            mesh.setGridUnitSI(gridUnitSI);
        auto& component = mesh["x"];
        component.resetDataset(openPMD::Dataset(
            openPMD::determineDatatype<double>(),
            openPMD::Extent{6, 5, 4}));
        component.storeChunkRaw(values.data(), openPMD::Offset{0, 0, 0}, openPMD::Extent{6, 5, 4});
        series.flush();
    }

    //! One particle plus an ``E`` mesh whose physical geometry is given in x,y,z.
    void writeCliOriginSample(
        std::filesystem::path const& path,
        std::array<double, 3> const& offset,
        std::array<double, 3> const& spacing,
        double gridUnitSI)
    {
        std::array<double, 3> const position{offset[0], offset[1], offset[2]};
        std::array<double, 3> const momentum{0.0, 0.0, 1.0};
        std::array<double, 1> const weight{1.0};
        std::array<double, 1> const charge{-picBunch::kElementaryCharge};
        std::array<double, 6 * 5 * 4> const meshValues{};

        openPMD::Series series(path.string(), openPMD::Access::CREATE);
        auto& iteration = series.iterations[0];

        auto& mesh = iteration.meshes["E"];
        mesh.setGeometry(openPMD::Mesh::Geometry::cartesian);
        mesh.setDataOrder(openPMD::Mesh::DataOrder::C);
        mesh.setAxisLabels(std::vector<std::string>{"z", "y", "x"});
        mesh.setGridSpacing(
            std::vector<double>{spacing[2] / gridUnitSI, spacing[1] / gridUnitSI, spacing[0] / gridUnitSI});
        mesh.setGridGlobalOffset(
            std::vector<double>{offset[2] / gridUnitSI, offset[1] / gridUnitSI, offset[0] / gridUnitSI});
        mesh.setGridUnitSI(gridUnitSI);
        auto& meshComponent = mesh["x"];
        meshComponent.resetDataset(openPMD::Dataset(
            openPMD::determineDatatype<double>(),
            openPMD::Extent{4, 5, 6}));
        meshComponent.storeChunkRaw(
            meshValues.data(),
            openPMD::Offset{0, 0, 0},
            openPMD::Extent{4, 5, 6});

        auto bindScalar = [](openPMD::RecordComponent& component, double const* value, double unitSI) {
            component.resetDataset(
                openPMD::Dataset(openPMD::determineDatatype<double>(), openPMD::Extent{1}));
            component.storeChunkRaw(value, openPMD::Offset{0}, openPMD::Extent{1});
            component.setUnitSI(unitSI);
        };

        auto& species = iteration.particles["electrons"];
        species["position"].setUnitDimension({{openPMD::UnitDimension::L, 1.0}});
        species["momentum"].setUnitDimension(
            {{openPMD::UnitDimension::L, 1.0},
             {openPMD::UnitDimension::M, 1.0},
             {openPMD::UnitDimension::T, -1.0}});
        species["charge"].setUnitDimension(
            {{openPMD::UnitDimension::T, 1.0},
             {openPMD::UnitDimension::I, 1.0}});
        char const* const names = "xyz";
        for(int a = 0; a < 3; ++a)
        {
            bindScalar(species["position"][std::string(1, names[a])], &position[a], 1.0);
            bindScalar(
                species["momentum"][std::string(1, names[a])],
                &momentum[a],
                picBunch::kElectronMass * picBunch::kSpeedOfLight);
        }
        bindScalar(species["weighting"], weight.data(), 1.0);
        bindScalar(species["charge"], charge.data(), 1.0);
        series.flush();
    }
} // namespace

TEST(Io, ReaderAppliesGridUnitSIToPhysicalSpacing)
{
    auto const path = temporaryPath();
    std::filesystem::remove(path);
    // 1e-6 is the PIConGPU convention; 8.85e12 exercises a large SI-style scale.
    for(double const unit : {1.0e-6, 8.85e12})
    {
        writeGeometryMesh(path, unit, true);
        auto const geometry = picBunch::readMeshGeometry(path.string());
        EXPECT_EQ(geometry.shape, (std::array<std::size_t, 3>{4, 5, 6}));
        EXPECT_EQ(geometry.axisLabels, (std::array<std::string, 3>{"x", "y", "z"}));
        // Stored z,y,x spacing/offset reversed to x,y,z and multiplied by unit.
        EXPECT_DOUBLE_EQ(geometry.gridUnitSI[0], unit);
        EXPECT_DOUBLE_EQ(geometry.gridUnitSI[1], unit);
        EXPECT_DOUBLE_EQ(geometry.gridUnitSI[2], unit);
        // Stored z,y,x = (0.5,1.5,2.5) becomes x,y,z = (2.5,1.5,0.5).
        EXPECT_DOUBLE_EQ(geometry.gridSpacing[0], 2.5 * unit);
        EXPECT_DOUBLE_EQ(geometry.gridSpacing[1], 1.5 * unit);
        EXPECT_DOUBLE_EQ(geometry.gridSpacing[2], 0.5 * unit);
        // Stored z,y,x = (3,2,1) becomes x,y,z = (1,2,3).
        EXPECT_DOUBLE_EQ(geometry.gridGlobalOffset[0], 1.0 * unit);
        EXPECT_DOUBLE_EQ(geometry.gridGlobalOffset[1], 2.0 * unit);
        EXPECT_DOUBLE_EQ(geometry.gridGlobalOffset[2], 3.0 * unit);
    }
    std::filesystem::remove(path);
}

TEST(Io, CliMeshGeometryIsConvertedExactlyOnce)
{
    // End-to-end through the same reader + geometry helper the CLI uses, then
    // through makeGrid, the geometry stage computeField consumes.  The v0.1 CLI
    // multiplied gridUnitSI a second time after readMeshGeometry had already
    // converted the spacing, so a physical 1e-6 m cell became 1e-12 m.
    auto const path = temporaryPath();
    std::filesystem::remove(path);

    auto const checkCase = [&](double unit, std::array<double, 3> const& expected) {
        // Expected x,y,z = (2.5, 1.5, 0.5) * unit.  The raw z,y,x vector is
        // therefore (0.5, 1.5, 2.5) in units of gridUnitSI.
        writeGeometryMesh(
            path,
            unit,
            true,
            {expected[2] / unit, expected[1] / unit, expected[0] / unit},
            {3.0 / unit, 2.0 / unit, 1.0 / unit});

        auto const geometry = picBunch::readMeshGeometry(path.string());
        auto const cell = picBunch::physicalCellSize(geometry);
        ASSERT_TRUE(cell.has_value());
        for(int a = 0; a < 3; ++a)
            EXPECT_DOUBLE_EQ((*cell)[a], expected[a]);

        picBunch::Bunch bunch;
        bunch.x = {0.0};
        bunch.y = {0.0};
        bunch.z = {0.0};
        bunch.ux = {0.0};
        bunch.uy = {0.0};
        bunch.uz = {0.0};
        bunch.weight = {1.0};
        bunch.charge = {-picBunch::kElementaryCharge};

        picBunch::ComputeOptions options;
        options.cellSize = picBunch::resolveCellSize(std::nullopt, cell);
        auto const grid = picBunch::makeGrid(bunch, options);
        for(int a = 0; a < 3; ++a)
            EXPECT_DOUBLE_EQ(grid.spacing[a], expected[a]);
    };

    // Small unit: the PIConGPU convention.  A second multiplication would give
    // 1e-12 m instead of 1e-6 m.
    checkCase(1.0e-6, {2.5e-6, 1.5e-6, 0.5e-6});
    // Large unit: 8.85e12-style SI scale; a second multiplication would give
    // 2.2e13 m instead of 2.5 m.
    checkCase(8.85e12, {2.5, 1.5, 0.5});

    // The old double-converted value must fail as a clear runtime error, not as
    // std::vector's max_size/bad_alloc message.  Two particles span 1 m.
    picBunch::Bunch spread;
    spread.x = {0.0, 1.0};
    spread.y = {0.0, 0.0};
    spread.z = {0.0, 0.0};
    spread.ux = {0.0, 0.0};
    spread.uy = {0.0, 0.0};
    spread.uz = {0.0, 0.0};
    spread.weight = {1.0, 1.0};
    spread.charge = {-picBunch::kElementaryCharge, -picBunch::kElementaryCharge};
    picBunch::ComputeOptions infeasible;
    infeasible.cellSize = {2.5e-12, 1.5e-12, 0.5e-12}; // 2.5e-6 m * 1e-6, the v0.1 bug
    try
    {
        (void) picBunch::makeGrid(spread, infeasible);
        FAIL() << "an infeasible grid was accepted";
    }
    catch(std::bad_alloc const&)
    {
        FAIL() << "infeasible grid surfaced as std::bad_alloc";
    }
    catch(std::runtime_error const& error)
    {
        EXPECT_NE(std::string(error.what()).find("infeasible"), std::string::npos)
            << error.what();
    }
    std::filesystem::remove(path);
}

TEST(Io, CliUsesMeshGridGlobalOffsetAsOrigin)
{
    // End-to-end through the production executable: the reader converts the mesh
    // gridGlobalOffset to physical metres, the CLI adopts it as the default grid
    // origin, and --origin still overrides it.  The raw offset is stored in z,y,x
    // units of gridUnitSI, so a second unit conversion would give 1e-12 m.
    auto const path = temporaryPath();
    std::filesystem::remove(path);
    std::array<double, 3> const offset{1.0e-6, 2.0e-6, 3.0e-6};
    std::array<double, 3> const spacing{1.0e-6, 1.0e-6, 1.0e-6};
    writeCliOriginSample(path, offset, spacing, 1.0e-6);

    auto run = [&](std::string const& extra) {
        std::string const command = std::string{"\""} + PICBUNCH_APP_PATH + "\" --input \""
            + path.string() + "\" " + extra + " 2>&1";
        FILE* pipe = ::popen(command.c_str(), "r");
        EXPECT_NE(pipe, nullptr);
        std::string output;
        if(pipe != nullptr)
        {
            std::array<char, 4096> buffer{};
            while(std::fgets(buffer.data(), buffer.size(), pipe) != nullptr)
                output += buffer.data();
            int const status = ::pclose(pipe);
            EXPECT_EQ(status, 0) << output;
        }
        return output;
    };
    auto parseOrigin = [](std::string const& output) {
        std::array<double, 3> origin{
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN()};
        auto const at = output.find("origin=(");
        if(at != std::string::npos)
            std::sscanf(
                output.c_str() + at,
                "origin=(%lf,%lf,%lf)",
                &origin[0],
                &origin[1],
                &origin[2]);
        return origin;
    };

    auto const defaults = parseOrigin(run(""));
    for(int a = 0; a < 3; ++a)
        EXPECT_NEAR(defaults[a], offset[a], 1.0e-12) << "mesh-default origin axis " << a;

    std::array<double, 3> const overrideOrigin{7.0e-7, 8.0e-7, 9.0e-7};
    auto const overridden = parseOrigin(run("--origin 7e-7,8e-7,9e-7"));
    for(int a = 0; a < 3; ++a)
        EXPECT_NEAR(overridden[a], overrideOrigin[a], 1.0e-12) << "--origin axis " << a;

    std::filesystem::remove(path);
}

TEST(Io, ReaderRejectsNonPositiveGridUnitSI)
{
    auto const path = temporaryPath();
    std::filesystem::remove(path);
    writeGeometryMesh(path, -1.0, true);
    EXPECT_THROW(picBunch::readMeshGeometry(path.string()), std::runtime_error);
    std::filesystem::remove(path);
}

TEST(Io, WriterRejectsMismatchedComponentsBeforeCreatingFile)
{
    auto const field = makeField();
    auto broken = field;
    broken.data[5].pop_back();
    auto const path = temporaryPath();
    std::filesystem::remove(path);
    EXPECT_THROW(picBunch::writeField(broken, path.string()), std::runtime_error);
    EXPECT_FALSE(std::filesystem::exists(path));
    std::filesystem::remove(path);
}

TEST(Io, WriterPublishesIterationPatternAtomically)
{
    auto const directory = std::filesystem::temp_directory_path()
        / ("picbunch_pattern_" + std::to_string(static_cast<long long>(getpid())));
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);

    auto const field = makeField();
    picBunch::FieldWriteOptions options;
    options.iteration = 7u;
    options.time = 2.5e-13;
    picBunch::writeField(field, (directory / "field_%T.h5").string(), options);

    std::size_t files = 0;
    for(auto const& entry : std::filesystem::directory_iterator(directory))
    {
        std::string const name = entry.path().filename().string();
        EXPECT_EQ(name.find(".picbunch-tmp-"), std::string::npos) << "temporary file left behind";
        if(name.rfind("field_", 0) == 0 && entry.path().extension() == ".h5")
        {
            ++files;
            auto const geometry = picBunch::readMeshGeometry(entry.path().string(), "E", 0, 7u);
            EXPECT_EQ(geometry.shape, (std::array<std::size_t, 3>{4, 5, 6}));
        }
    }
    EXPECT_EQ(files, 1u);
    std::filesystem::remove_all(directory);
}

TEST(Io, Adios2Bp5DirectoryIsPublishedAfterSeriesClose)
{
#if defined(openPMD_HAVE_ADIOS2) && openPMD_HAVE_ADIOS2
    // Regression: the temporary BP5 directory used to be renamed while the
    // openPMD Series was still alive.  ADIOS2 then wrote deferred metadata
    // (profiling.json) into the old temporary path at finalize time and
    // aborted.  writeFieldToDomain must close the series before the rename.
    auto const field = makeField();
    auto const directory = std::filesystem::temp_directory_path()
        / ("picbunch_bp5_" + std::to_string(static_cast<long long>(getpid())));
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    auto const path = directory / "field.bp";

    picBunch::FieldWriteOptions options;
    options.iteration = 7u;
    options.time = 2.5e-13;
    options.timeUnitSI = 1.0;
    picBunch::writeFieldToDomain(
        field, path.string(), options, field.grid.n, field.grid.origin);

    // BP5 stores one directory per series, published under the final name.
    ASSERT_TRUE(std::filesystem::is_directory(path));
    ASSERT_TRUE(std::filesystem::exists(path / "md.0"));

    {
        openPMD::Series series(path.string(), openPMD::Access::READ_ONLY);
        auto& iteration = series.iterations[options.iteration];
        for(int c = 0; c < 6; ++c)
        {
            std::string const recordName = (c < 3) ? "E" : "B";
            auto& component = iteration.meshes[recordName][std::string(1, "xyz"[c % 3])];
            auto const values = loadComponent(series, component, field.grid.n);
            ASSERT_EQ(values.size(), field.data[c].size());
            for(std::size_t l = 0; l < values.size(); ++l)
                EXPECT_EQ(values[l], field.data[c][l])
                    << "component " << c << " value " << l;
        }
    }

    for(auto const& entry : std::filesystem::directory_iterator(directory))
        EXPECT_EQ(
            entry.path().filename().string().find(".picbunch-tmp-"), std::string::npos)
            << "temporary path left behind";

    std::filesystem::remove_all(directory);
#else
    GTEST_SKIP() << "openPMD was built without the ADIOS2 backend";
#endif
}

TEST(Io, ReaderAcceptsOpenPMD2ScalarRecords)
{
    // The file is produced by tests/make_scalar_record_sample.py with h5py,
    // because openPMD-api 0.17 cannot create the 2.0 scalar-record layout on
    // demand.  Both group forms are present: ``value`` and record-named child.
    char const* file = std::getenv("PICBUNCH_SCALAR_RECORD_FILE");
    if(file == nullptr)
        GTEST_SKIP() << "PICBUNCH_SCALAR_RECORD_FILE is not set";

    auto const data = picBunch::readBunch(file, std::size_t{0});
    ASSERT_EQ(data.bunch.size(), 2u);
    EXPECT_DOUBLE_EQ(data.bunch.weight[0], 2.0);
    EXPECT_DOUBLE_EQ(data.bunch.weight[1], 3.0);
    EXPECT_DOUBLE_EQ(data.bunch.x[0], 1.0e-6 + 7.0e-7);
    EXPECT_DOUBLE_EQ(data.bunch.x[1], 2.0e-6 + 8.0e-7);
    EXPECT_DOUBLE_EQ(data.bunch.y[0], 3.0e-6 + 9.0e-7);
    EXPECT_DOUBLE_EQ(data.bunch.y[1], 4.0e-6 + 10.0e-7);
    EXPECT_DOUBLE_EQ(data.bunch.z[0], 5.0e-6 + 11.0e-7);
    EXPECT_DOUBLE_EQ(data.bunch.z[1], 6.0e-6 + 12.0e-7);
    EXPECT_DOUBLE_EQ(data.bunch.ux[0], 0.0);
    EXPECT_DOUBLE_EQ(data.bunch.uy[0], 0.0);
    EXPECT_DOUBLE_EQ(data.bunch.uz[0], 1.0);
    EXPECT_DOUBLE_EQ(data.bunch.uz[1], 2.0);
    EXPECT_DOUBLE_EQ(data.bunch.charge[0], -2.0e-19);
    EXPECT_DOUBLE_EQ(data.bunch.charge[1], -3.0e-19);
}

namespace
{
    struct RunResult
    {
        int status = -1;
        std::string output;
    };

    //! Run the production executable and capture stdout+stderr and the exit code.
    RunResult runApp(std::filesystem::path const& path, std::string const& extra)
    {
        std::string const command = std::string{"\""} + PICBUNCH_APP_PATH + "\" --input \""
            + path.string() + "\" " + extra + " 2>&1";
        RunResult result;
        FILE* pipe = ::popen(command.c_str(), "r");
        EXPECT_NE(pipe, nullptr);
        if(pipe != nullptr)
        {
            std::array<char, 4096> buffer{};
            while(std::fgets(buffer.data(), buffer.size(), pipe) != nullptr)
                result.output += buffer.data();
            int const status = ::pclose(pipe);
            result.status = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        }
        return result;
    }

    std::array<std::size_t, 3> parseGrid(std::string const& output)
    {
        std::array<std::size_t, 3> grid{
            std::numeric_limits<std::size_t>::max(),
            std::numeric_limits<std::size_t>::max(),
            std::numeric_limits<std::size_t>::max()};
        auto const at = output.find("grid: n=(");
        if(at != std::string::npos)
            std::sscanf(
                output.c_str() + at,
                "grid: n=(%zu,%zu,%zu)",
                &grid[0],
                &grid[1],
                &grid[2]);
        return grid;
    }

    std::array<double, 3> parseOrigin(std::string const& output)
    {
        std::array<double, 3> origin{
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN()};
        auto const at = output.find("origin=(");
        if(at != std::string::npos)
            std::sscanf(
                output.c_str() + at,
                "origin=(%lf,%lf,%lf)",
                &origin[0],
                &origin[1],
                &origin[2]);
        return origin;
    }

    //! Write an openPMD sample with particles and, when ``shape`` is set, an E mesh.
    //!
    //! ``shape``, ``offset`` and ``spacing`` are physical x,y,z.  The mesh is
    //! stored in the PIConGPU z,y,x axis order with raw values divided by
    //! ``gridUnitSI``, so the reader has to undo both the order and the unit.
    void writeSample(
        std::filesystem::path const& path,
        std::vector<std::array<double, 3>> const& positions,
        std::optional<std::array<std::size_t, 3>> shape,
        std::array<double, 3> const& offset,
        std::array<double, 3> const& spacing,
        double gridUnitSI)
    {
        auto const count = positions.size();
        std::vector<double> px(count);
        std::vector<double> py(count);
        std::vector<double> pz(count);
        std::vector<double> ux(count, 0.0);
        std::vector<double> uy(count, 0.0);
        std::vector<double> uz(count, 1.0);
        std::vector<double> weight(count, 1.0);
        std::vector<double> charge(count, -picBunch::kElementaryCharge);
        for(std::size_t p = 0; p < count; ++p)
        {
            px[p] = positions[p][0];
            py[p] = positions[p][1];
            pz[p] = positions[p][2];
        }
        std::vector<double> meshValues;
        if(shape)
            meshValues.assign((*shape)[0] * (*shape)[1] * (*shape)[2], 0.0);

        openPMD::Series series(path.string(), openPMD::Access::CREATE);
        auto& iteration = series.iterations[0];

        if(shape)
        {
            auto& mesh = iteration.meshes["E"];
            mesh.setGeometry(openPMD::Mesh::Geometry::cartesian);
            mesh.setDataOrder(openPMD::Mesh::DataOrder::C);
            mesh.setAxisLabels(std::vector<std::string>{"z", "y", "x"});
            mesh.setGridSpacing(std::vector<double>{
                spacing[2] / gridUnitSI,
                spacing[1] / gridUnitSI,
                spacing[0] / gridUnitSI});
            mesh.setGridGlobalOffset(std::vector<double>{
                offset[2] / gridUnitSI,
                offset[1] / gridUnitSI,
                offset[0] / gridUnitSI});
            mesh.setGridUnitSI(gridUnitSI);
            auto& component = mesh["x"];
            component.resetDataset(openPMD::Dataset(
                openPMD::determineDatatype<double>(),
                openPMD::Extent{(*shape)[2], (*shape)[1], (*shape)[0]}));
            component.storeChunkRaw(
                meshValues.data(),
                openPMD::Offset{0, 0, 0},
                openPMD::Extent{(*shape)[2], (*shape)[1], (*shape)[0]});
        }

        auto bindScalar = [count](openPMD::RecordComponent& component, double const* value, double unitSI) {
            component.resetDataset(openPMD::Dataset(
                openPMD::determineDatatype<double>(),
                openPMD::Extent{static_cast<openPMD::Extent::value_type>(count)}));
            component.storeChunkRaw(
                value,
                openPMD::Offset{0},
                openPMD::Extent{static_cast<openPMD::Extent::value_type>(count)});
            component.setUnitSI(unitSI);
        };

        auto& species = iteration.particles["electrons"];
        species["position"].setUnitDimension({{openPMD::UnitDimension::L, 1.0}});
        species["momentum"].setUnitDimension(
            {{openPMD::UnitDimension::L, 1.0},
             {openPMD::UnitDimension::M, 1.0},
             {openPMD::UnitDimension::T, -1.0}});
        species["charge"].setUnitDimension(
            {{openPMD::UnitDimension::T, 1.0},
             {openPMD::UnitDimension::I, 1.0}});
        char const* const names = "xyz";
        double* const positionData[3] = {px.data(), py.data(), pz.data()};
        double* const momentumData[3] = {ux.data(), uy.data(), uz.data()};
        for(int a = 0; a < 3; ++a)
        {
            bindScalar(species["position"][std::string(1, names[a])], positionData[a], 1.0);
            bindScalar(
                species["momentum"][std::string(1, names[a])],
                momentumData[a],
                picBunch::kElectronMass * picBunch::kSpeedOfLight);
        }
        bindScalar(species["weighting"], weight.data(), 1.0);
        bindScalar(species["charge"], charge.data(), 1.0);
        series.flush();
    }
} // namespace

TEST(Io, MeshZeroOffsetAdoptsMeshExtentWithoutFlags)
{
    // A real checkpoint mesh: zero gridGlobalOffset and a 24x24x12 extent, while
    // the particles occupy a sub-volume.  Defaults must yield exactly the mesh
    // grid, not an AABB-derived grid.
    auto const path = temporaryPath();
    std::filesystem::remove(path);
    writeSample(
        path,
        {{12.0e-6, 12.0e-6, 6.0e-6}},
        std::array<std::size_t, 3>{24, 24, 12},
        {0.0, 0.0, 0.0},
        {1.0e-6, 1.0e-6, 1.0e-6},
        1.0e-6);

    auto const result = runApp(path, "");
    ASSERT_EQ(result.status, 0) << result.output;
    EXPECT_EQ(parseGrid(result.output), (std::array<std::size_t, 3>{24, 24, 12})) << result.output;
    auto const origin = parseOrigin(result.output);
    for(int a = 0; a < 3; ++a)
        EXPECT_NEAR(origin[a], 0.0, 1.0e-15) << result.output;
    std::filesystem::remove(path);
}

TEST(Io, MeshNonZeroOffsetAdoptsMeshExtentAndOriginWithoutFlags)
{
    auto const path = temporaryPath();
    std::filesystem::remove(path);
    std::array<double, 3> const offset{1.0e-6, 2.0e-6, 3.0e-6};
    writeSample(
        path,
        {{offset[0] + 2.0e-6, offset[1] + 3.0e-6, offset[2] + 1.0e-6}},
        std::array<std::size_t, 3>{8, 10, 6},
        offset,
        {1.0e-6, 1.0e-6, 1.0e-6},
        1.0e-6);

    auto const result = runApp(path, "");
    ASSERT_EQ(result.status, 0) << result.output;
    EXPECT_EQ(parseGrid(result.output), (std::array<std::size_t, 3>{8, 10, 6})) << result.output;
    auto const origin = parseOrigin(result.output);
    for(int a = 0; a < 3; ++a)
        EXPECT_NEAR(origin[a], offset[a], 1.0e-15) << result.output;
    std::filesystem::remove(path);
}

TEST(Io, GridAndOriginOverrideMeshExtent)
{
    auto const path = temporaryPath();
    std::filesystem::remove(path);
    writeSample(
        path,
        {{1.0e-6, 1.0e-6, 1.0e-6}},
        std::array<std::size_t, 3>{24, 24, 12},
        {0.0, 0.0, 0.0},
        {1.0e-6, 1.0e-6, 1.0e-6},
        1.0e-6);

    auto const result = runApp(path, "--grid 8,8,8 --origin 0,0,0");
    ASSERT_EQ(result.status, 0) << result.output;
    EXPECT_EQ(parseGrid(result.output), (std::array<std::size_t, 3>{8, 8, 8})) << result.output;
    auto const origin = parseOrigin(result.output);
    for(int a = 0; a < 3; ++a)
        EXPECT_NEAR(origin[a], 0.0, 1.0e-15) << result.output;
    std::filesystem::remove(path);
}

TEST(Io, ParticlesOutsideMeshGridAreRejectedByNameAndCount)
{
    auto const path = temporaryPath();
    std::filesystem::remove(path);
    writeSample(
        path,
        {{1.0e-6, 1.0e-6, 1.0e-6}, {100.0e-6, 1.0e-6, 1.0e-6}},
        std::array<std::size_t, 3>{8, 8, 8},
        {0.0, 0.0, 0.0},
        {1.0e-6, 1.0e-6, 1.0e-6},
        1.0e-6);

    auto const result = runApp(path, "");
    EXPECT_NE(result.status, 0) << result.output;
    EXPECT_NE(result.output.find("1 of 2 particles lie outside the output grid bounds"), std::string::npos)
        << result.output;
    EXPECT_NE(result.output.find("grid bounds"), std::string::npos) << result.output;
    std::filesystem::remove(path);
}

TEST(Io, AnisotropicCellSpacingWarnsWithoutChangingFields)
{
    // The warning is advisory: the executable must still compute the same field
    // as the library for the same anisotropic spacing.  The mesh-derived and
    // --cell-size paths share the resolved spacing, so both must warn.
    auto const path = temporaryPath();

    auto runSpacing = [&](std::array<double, 3> const& spacing, std::string const& extra) {
        std::filesystem::remove(path);
        std::optional<std::array<std::size_t, 3>> const shape
            = extra.empty() ? std::optional<std::array<std::size_t, 3>>{
                  std::array<std::size_t, 3>{8, 8, 8}}
                            : std::nullopt;
        writeSample(path, {{4.0, 4.0, 4.0}}, shape, {0.0, 0.0, 0.0}, spacing, 1.0);
        auto const result = runApp(path, extra);
        EXPECT_EQ(result.status, 0) << result.output;
        std::filesystem::remove(path);
        return result.output;
    };

    for(double const ratio : {1.0, 1.5})
    {
        std::string const output = runSpacing({ratio, 1.0, 1.0}, "");
        EXPECT_EQ(output.find("anisotropic cell spacing"), std::string::npos) << output;
    }
    for(double const ratio : {2.0, 4.0})
    {
        std::string const output = runSpacing({ratio, 1.0, 1.0}, "");
        EXPECT_NE(output.find("anisotropic cell spacing"), std::string::npos) << output;
        EXPECT_NE(
            output.find("near-field accuracy is set by the shortest axis"),
            std::string::npos)
            << output;
        EXPECT_NE(output.find("shortest 1 m"), std::string::npos) << output;

        std::string const meshFree = runSpacing({ratio, 1.0, 1.0}, "--cell-size 1,1,4");
        EXPECT_NE(meshFree.find("anisotropic cell spacing"), std::string::npos) << meshFree;
    }

    // Same geometry through the API: the warning path must not touch the field.
    picBunch::Bunch bunch;
    bunch.resize(1);
    bunch.x[0] = 4.0;
    bunch.y[0] = 4.0;
    bunch.z[0] = 4.0;
    bunch.ux[0] = 0.0;
    bunch.uy[0] = 0.0;
    bunch.uz[0] = 1.0;
    bunch.weight[0] = 1.0;
    bunch.charge[0] = -picBunch::kElementaryCharge;
    picBunch::ComputeOptions options;
    options.cellSize = {4.0, 1.0, 1.0};
    options.gridSize = std::array<std::size_t, 3>{8, 8, 8};
    options.gridOrigin = std::array<double, 3>{0.0, 0.0, 0.0};
    picBunch::Field const field = picBunch::computeField(bunch, options);
    double maxE = 0.0;
    double maxB = 0.0;
    for(int c = 0; c < 6; ++c)
        for(double value : field.data[c])
            (c < 3 ? maxE : maxB) = std::max(c < 3 ? maxE : maxB, std::abs(value));

    std::string const warned = runSpacing({4.0, 1.0, 1.0}, "");
    double cliE = 0.0;
    double cliB = 0.0;
    ASSERT_NE(warned.find("field: max|E|="), std::string::npos) << warned;
    std::sscanf(
        warned.c_str() + warned.find("field: max|E|="),
        "field: max|E|=%lf V/m  max|B|=%lf T",
        &cliE,
        &cliB);
    EXPECT_NEAR(cliE, maxE, 1.0e-5 * std::max(maxE, 1.0e-300));
    EXPECT_NEAR(cliB, maxB, 1.0e-5 * std::max(maxB, 1.0e-300));
}

TEST(Io, NoMeshRequiresCellSizeAndUsesBunchAabb)
{
    auto const path = temporaryPath();
    std::filesystem::remove(path);
    writeSample(
        path,
        {{1.0e-6, 2.0e-6, 3.0e-6}},
        std::nullopt,
        {0.0, 0.0, 0.0},
        {1.0e-6, 1.0e-6, 1.0e-6},
        1.0e-6);

    auto const missing = runApp(path, "");
    EXPECT_NE(missing.status, 0) << missing.output;
    EXPECT_NE(missing.output.find("no cell size available"), std::string::npos) << missing.output;
    EXPECT_NE(missing.output.find("--cell-size"), std::string::npos) << missing.output;

    auto const result = runApp(path, "--cell-size 1e-6");
    ASSERT_EQ(result.status, 0) << result.output;
    // One particle at (1,2,3) um, margin 4 cells: n = ceil(8)+1 rounded even = 10.
    EXPECT_EQ(parseGrid(result.output), (std::array<std::size_t, 3>{10, 10, 10})) << result.output;
    auto const origin = parseOrigin(result.output);
    std::array<double, 3> const expectedOrigin{-3.0e-6, -2.0e-6, -1.0e-6};
    for(int a = 0; a < 3; ++a)
        EXPECT_NEAR(origin[a], expectedOrigin[a], 1.0e-15) << result.output;
    std::filesystem::remove(path);
}

TEST(Io, FarTailUsesYeeComponentPositions)
{
    // One moving charge at the centre of the exact box.  Outside the box the
    // writer evaluates the multipole tail; each written component must use its
    // own Yee-staggered observation point, matching the advertised `position`.
    picBunch::Bunch bunch;
    bunch.resize(1);
    bunch.x[0] = 3.5;
    bunch.y[0] = 3.5;
    bunch.z[0] = 3.5;
    bunch.ux[0] = 1.0;
    bunch.uy[0] = 0.0;
    bunch.uz[0] = 0.0;
    bunch.weight[0] = 1.0;
    bunch.charge[0] = picBunch::kElementaryCharge;
    auto const tail = picBunch::deriveFarTail(bunch);

    picBunch::Field field;
    field.grid.n = {4, 4, 4};
    field.grid.spacing = {1.0, 1.0, 1.0};
    field.grid.origin = {2.0, 2.0, 2.0};
    field.grid.fftPadding = 2;
    for(auto& component : field.data)
        component.assign(field.grid.size(), 0.0);

    std::array<std::size_t, 3> const globalN{16, 16, 16};
    std::array<double, 3> const globalOrigin{0.0, 0.0, 0.0};
    auto const path = temporaryPath();
    std::filesystem::remove(path);
    picBunch::FieldWriteOptions options;
    options.iteration = 1;
    picBunch::writeFieldToDomain(
        field, path.string(), options, globalN, globalOrigin, &tail);

    openPMD::Series series(path.string(), openPMD::Access::READ_ONLY);
    auto& iteration = series.iterations[1];
    std::array<std::vector<double>, 6> values;
    for(std::size_t c = 0; c < 3u; ++c)
    {
        std::string const name(1, "xyz"[c]);
        values[c] = loadComponent(series, iteration.meshes["E"][name], globalN);
        values[c + 3u] = loadComponent(series, iteration.meshes["B"][name], globalN);
    }
    series.flush();

    // A node outside the exact box [2,6)^3.
    std::array<std::size_t, 3> const node{11, 12, 13};
    std::size_t const index = node[0] + globalN[0] * (node[1] + globalN[1] * node[2]);
    for(std::size_t c = 0; c < 6u; ++c)
    {
        std::array<double, 3> position{};
        std::array<double, 3> unshifted{};
        for(int a = 0; a < 3; ++a)
        {
            unshifted[static_cast<std::size_t>(a)]
                = globalOrigin[static_cast<std::size_t>(a)]
                + static_cast<double>(node[static_cast<std::size_t>(a)])
                    * field.grid.spacing[static_cast<std::size_t>(a)];
            position[static_cast<std::size_t>(a)] = unshifted[static_cast<std::size_t>(a)]
                + picBunch::kYeeOffset[c][static_cast<std::size_t>(a)]
                    * field.grid.spacing[static_cast<std::size_t>(a)]
                    / picBunch::kRefinement;
        }
        auto const expected = c < 3u ? picBunch::evaluateFarTailE(tail, position)
                                     : picBunch::evaluateFarTailB(tail, position);
        auto const separate = c < 3u ? picBunch::evaluateFarTailE(tail, unshifted)
                                     : picBunch::evaluateFarTailB(tail, unshifted);
        std::size_t const component = c < 3u ? c : c - 3u;
        double const tolerance = 1.0e-9 * std::max(1.0, std::abs(expected[component]));
        EXPECT_NEAR(values[c][index], expected[component], tolerance)
            << "component " << c << " is not evaluated at its Yee offset";
        bool const hasOffset = picBunch::kYeeOffset[c][0] != 0.0
            || picBunch::kYeeOffset[c][1] != 0.0 || picBunch::kYeeOffset[c][2] != 0.0;
        if(hasOffset && std::abs(expected[component]) > 0.0)
            EXPECT_GT(
                std::abs(expected[component] - separate[component]),
                1.0e-6 * std::abs(expected[component]))
                << "component " << c << " offset is not observable in this setup";
    }
    series.flush();
    std::filesystem::remove(path);
}
