// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Opt-in large-dataset writer probe.  The production field dataset is 5.03 GB
// per component, above 4 GiB, and the staged run failed at the first output.
// Enable with PICBUNCH_LARGE_WRITER_TEST=1.

#include "picBunch/ChunkedWriter.hpp"
#include "picBunch/Types.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

TEST(ChunkedWriter, LargeDatasetAboveFourGiB)
{
    if(std::getenv("PICBUNCH_LARGE_WRITER_TEST") == nullptr)
        GTEST_SKIP() << "set PICBUNCH_LARGE_WRITER_TEST=1 to run the 5 GB writer probe";
    picBunch::Grid grid;
    grid.n = {640, 1536, 640};
    grid.spacing = {1.772e-7, 1.772e-7, 1.772e-7};
    grid.origin = {0.0, 0.0, 0.0};
    grid.fftPadding = 2;

    auto const path = std::filesystem::temp_directory_path() / "picbunch_large_writer.h5";
    std::filesystem::remove(path);
    picBunch::FieldWriteOptions options;
    picBunch::ChunkedFieldWriter writer(path.string(), options, grid.n, grid.origin, grid);

    std::size_t const plane = grid.n[0] * grid.n[1];
    std::size_t const chunkPlanes = 40;
    std::vector<double> data(chunkPlanes * plane);
    for(std::size_t z = 0; z < grid.n[2]; z += chunkPlanes)
    {
        std::size_t const count = std::min(chunkPlanes, grid.n[2] - z);
        data.assign(count * plane, 0.5);
        for(int c = 0; c < 6; ++c)
        {
            picBunch::FieldChunk chunk;
            chunk.grid = grid;
            chunk.component = c;
            chunk.zOffset = z;
            chunk.zCount = count;
            chunk.data = data;
            writer.write(chunk);
        }
    }
    writer.close();
    EXPECT_TRUE(std::filesystem::exists(path));
    EXPECT_GT(std::filesystem::file_size(path), 4ull * 1024ull * 1024ull * 1024ull);
    std::filesystem::remove(path);
}
