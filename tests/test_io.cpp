// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group 4: openPMD writer/reader round-trip.  Values must be bit-identical and
// every metadata item the writer sets must survive: unitSI, Yee position,
// gridSpacing, gridGlobalOffset, axis labels, dataOrder, geometry,
// unitDimension and the iteration index.

#include "picBunch/OpenPMD.hpp"
#include "picBunch/OpenPMDWriter.hpp"

#include <openPMD/openPMD.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
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
