// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group: ChunkedWriter.  The streaming openPMD writer must create the same
// dataset extents, offsets and per-component metadata as the in-core writer,
// and the union of the stored z-slabs must be the in-core array.

#include "picBunch/ChunkedWriter.hpp"
#include "picBunch/OpenPMDWriter.hpp"
#include "picBunch/Types.hpp"

#include <openPMD/openPMD.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <unistd.h>

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
                field.data[c][l] = 0.125 * static_cast<double>(c)
                    + 1.0e-3 * static_cast<double>(l)
                    + 1.0e-12 * static_cast<double>(c * 7 + l);
        }
        return field;
    }

    std::filesystem::path temporaryPath(std::string const& suffix)
    {
        return std::filesystem::temp_directory_path()
            / ("picbunch_chunked_writer_"
               + std::to_string(static_cast<long long>(getpid())) + suffix + ".h5");
    }

    std::vector<double> loadComponent(
        openPMD::Series& series,
        openPMD::RecordComponent& component,
        picBunch::Idx3 const& shape,
        std::uint64_t iteration)
    {
        openPMD::Extent const extent{
            static_cast<openPMD::Extent::value_type>(shape[2]),
            static_cast<openPMD::Extent::value_type>(shape[1]),
            static_cast<openPMD::Extent::value_type>(shape[0])};
        auto raw = component.loadChunk<double>(openPMD::Offset{0, 0, 0}, extent);
        series.flush();
        std::vector<double> values(
            static_cast<std::size_t>(extent[0] * extent[1] * extent[2]));
        for(std::size_t l = 0; l < values.size(); ++l)
            values[l] = raw.get()[l];
        (void) iteration;
        return values;
    }

    void writeChunked(picBunch::Field const& field, std::string const& path, std::size_t chunkPlanes)
    {
        picBunch::FieldWriteOptions options;
        options.iteration = 3u;
        options.time = 1.25e-13;
        options.timeUnitSI = 1.0;
        picBunch::ChunkedFieldWriter writer(
            path,
            options,
            field.grid.n,
            field.grid.origin,
            field.grid);
        std::size_t const plane = field.grid.n[0] * field.grid.n[1];
        // Interleave components and emit z-slabs of the requested size.
        for(std::size_t z = 0; z < field.grid.n[2]; z += chunkPlanes)
        {
            std::size_t const count = std::min(chunkPlanes, field.grid.n[2] - z);
            for(int c = 0; c < 6; ++c)
            {
                picBunch::FieldChunk chunk;
                chunk.grid = field.grid;
                chunk.component = c;
                chunk.zOffset = z;
                chunk.zCount = count;
                chunk.data.assign(
                    field.data[c].begin() + static_cast<std::ptrdiff_t>(z * plane),
                    field.data[c].begin() + static_cast<std::ptrdiff_t>((z + count) * plane));
                writer.write(chunk);
            }
        }
        writer.close();
    }
} // namespace

TEST(ChunkedWriter, DatasetExtentsAndValuesMatchInCore)
{
    auto const field = makeField();
    auto const inCorePath = temporaryPath("_incore");
    auto const chunkedPath = temporaryPath("_chunked");
    std::filesystem::remove(inCorePath);
    std::filesystem::remove(chunkedPath);

    picBunch::FieldWriteOptions options;
    options.iteration = 3u;
    options.time = 1.25e-13;
    options.timeUnitSI = 1.0;
    picBunch::writeField(field, inCorePath.string(), options);
    writeChunked(field, chunkedPath.string(), 2u);

    openPMD::Series reference(inCorePath.string(), openPMD::Access::READ_ONLY);
    openPMD::Series streamed(chunkedPath.string(), openPMD::Access::READ_ONLY);
    ASSERT_TRUE(reference.iterations.contains(3u));
    ASSERT_TRUE(streamed.iterations.contains(3u));

    std::array<char const*, 2> const records{"E", "B"};
    std::array<char const*, 3> const components{"x", "y", "z"};
    for(auto const* record : records)
    {
        auto& refMesh = reference.iterations[3u].meshes[record];
        auto& outMesh = streamed.iterations[3u].meshes[record];
        EXPECT_EQ(outMesh.axisLabels(), refMesh.axisLabels());
        EXPECT_EQ(outMesh.dataOrder(), refMesh.dataOrder());
        EXPECT_EQ(outMesh.geometryString(), refMesh.geometryString());
        EXPECT_DOUBLE_EQ(outMesh.gridUnitSI(), refMesh.gridUnitSI());
        auto const refSpacing = refMesh.gridSpacing<double>();
        auto const outSpacing = outMesh.gridSpacing<double>();
        ASSERT_EQ(refSpacing.size(), outSpacing.size());
        for(std::size_t a = 0; a < refSpacing.size(); ++a)
            EXPECT_DOUBLE_EQ(outSpacing[a], refSpacing[a]);
        auto const refOffset = refMesh.gridGlobalOffset();
        auto const outOffset = outMesh.gridGlobalOffset();
        ASSERT_EQ(refOffset.size(), outOffset.size());
        for(std::size_t a = 0; a < refOffset.size(); ++a)
            EXPECT_DOUBLE_EQ(outOffset[a], refOffset[a]);
        for(auto const* component : components)
        {
            auto refComponent = refMesh[component];
            auto outComponent = outMesh[component];
            ASSERT_EQ(
                refComponent.getExtent(),
                outComponent.getExtent());
            EXPECT_DOUBLE_EQ(refComponent.unitSI(), outComponent.unitSI());
            auto const refPosition = refComponent.getAttribute("position").get<std::vector<double>>();
            auto const outPosition = outComponent.getAttribute("position").get<std::vector<double>>();
            ASSERT_EQ(refPosition.size(), outPosition.size());
            for(std::size_t a = 0; a < refPosition.size(); ++a)
                EXPECT_DOUBLE_EQ(outPosition[a], refPosition[a]);
            auto const expected = loadComponent(
                reference, refComponent, field.grid.n, options.iteration);
            auto const actual = loadComponent(
                streamed, outComponent, field.grid.n, options.iteration);
            ASSERT_EQ(expected.size(), actual.size());
            for(std::size_t l = 0; l < expected.size(); ++l)
                EXPECT_DOUBLE_EQ(actual[l], expected[l]);
        }
    }
    std::filesystem::remove(inCorePath);
    std::filesystem::remove(chunkedPath);
}

TEST(ChunkedWriter, RejectsIncompleteCoverage)
{
    auto const field = makeField();
    auto const path = temporaryPath("_incomplete");
    std::filesystem::remove(path);
    picBunch::FieldWriteOptions options;
    picBunch::ChunkedFieldWriter writer(
        path.string(),
        options,
        field.grid.n,
        field.grid.origin,
        field.grid);
    std::size_t const plane = field.grid.n[0] * field.grid.n[1];
    picBunch::FieldChunk chunk;
    chunk.grid = field.grid;
    chunk.component = 0;
    chunk.zOffset = 0;
    chunk.zCount = 1;
    chunk.data.assign(field.data[0].begin(), field.data[0].begin() + plane);
    writer.write(chunk);
    EXPECT_THROW(writer.close(), std::runtime_error);
    std::filesystem::remove(path);
}
