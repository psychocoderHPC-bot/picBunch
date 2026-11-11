// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Global-domain tests: makeGrid must keep the computation on the bunch box and
// align it to the global nodes; the chunked writer must place the field at the
// right global offset and write exact zeros elsewhere.

#include "picBunch/Compute.hpp"
#include "picBunch/OpenPMDWriter.hpp"
#include "picBunch/Types.hpp"

#include <gtest/gtest.h>

#include <openPMD/openPMD.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace
{
    picBunch::Field makeField()
    {
        picBunch::Field field;
        field.grid.n = {8, 8, 8};
        field.grid.spacing = {1.0, 1.0, 1.0};
        field.grid.origin = {2.0, 3.0, 4.0};
        field.grid.fftPadding = 2;
        for(std::size_t c = 0; c < 6; ++c)
        {
            field.data[c].resize(field.grid.size());
            for(std::size_t l = 0; l < field.grid.size(); ++l)
                field.data[c][l] = 100.0 * static_cast<double>(c) + static_cast<double>(l) + 1.0;
        }
        return field;
    }
} // namespace

TEST(Domain, GridStaysOnBunchBox)
{
    picBunch::Bunch bunch;
    bunch.resize(1);
    bunch.x[0] = 5.0;
    bunch.y[0] = 5.0;
    bunch.z[0] = 5.0;
    bunch.ux[0] = bunch.uy[0] = bunch.uz[0] = 0.0;
    bunch.weight[0] = 1.0;
    bunch.charge[0] = -picBunch::kElementaryCharge;

    picBunch::ComputeOptions options;
    options.cellSize = {1.0, 1.0, 1.0};
    options.marginCells = 4.0;
    options.globalGridSize = picBunch::Idx3{1024, 1024, 1024};
    options.globalGridOrigin = picBunch::Vec3{0.0, 0.0, 0.0};
    auto const grid = picBunch::makeGrid(bunch, options);

    EXPECT_LT(grid.size(), 1024u * 1024u * 1024u);
    EXPECT_LE(grid.n[0], 32u);
    for(int a = 0; a < 3; ++a)
    {
        double const index = (grid.origin[a] - 0.0) / grid.spacing[a];
        EXPECT_DOUBLE_EQ(index, std::round(index));
        EXPECT_GE(grid.origin[a], 0.0);
    }
}

TEST(Domain, ChunkedWriteZerosAndOffset)
{
    auto const field = makeField();
    picBunch::Idx3 const globalN{16, 16, 16};
    picBunch::Vec3 const globalOrigin{0.0, 0.0, 0.0};

    auto const path = std::filesystem::temp_directory_path() / "picbunch_domain_test.h5";
    std::filesystem::remove(path);
    picBunch::FieldWriteOptions options;
    options.iteration = 3;
    picBunch::writeFieldToDomain(field, path.string(), options, globalN, globalOrigin);

    openPMD::Series series(path.string(), openPMD::Access::READ_ONLY);
    auto& mesh = series.iterations[3].meshes["E"];
    EXPECT_EQ(mesh.gridSpacing<double>(), (std::vector<double>{1.0, 1.0, 1.0}));
    EXPECT_EQ(mesh.gridGlobalOffset(), (std::vector<double>{0.0, 0.0, 0.0}));
    EXPECT_DOUBLE_EQ(mesh.gridUnitSI(), 1.0);

    auto component = mesh["x"];
    auto const extent = component.getExtent();
    EXPECT_EQ(extent, (openPMD::Extent{16, 16, 16}));
    auto data = component.loadChunk<double>();
    series.flush();

    std::size_t nonzeroOutside = 0;
    for(std::size_t k = 0; k < 16; ++k)
        for(std::size_t j = 0; j < 16; ++j)
            for(std::size_t i = 0; i < 16; ++i)
            {
                bool const inside = i >= 2 && i < 10 && j >= 3 && j < 11 && k >= 4 && k < 12;
                double const value = data.get()[
                    i + 16u * (j + 16u * k)];
                if(!inside && value != 0.0)
                    ++nonzeroOutside;
            }
    EXPECT_EQ(nonzeroOutside, 0u);

    std::size_t const fi = 5 - 2;
    std::size_t const fj = 6 - 3;
    std::size_t const fk = 7 - 4;
    std::size_t const fieldIndex = fi + 8u * (fj + 8u * fk);
    double const expected = field.data[0][fieldIndex];
    EXPECT_DOUBLE_EQ(data.get()[5 + 16u * (6 + 16u * 7)], expected);

    series.flush();
    std::filesystem::remove(path);
}

TEST(Domain, TailMarginMonopoleBound)
{
    picBunch::Bunch bunch;
    bunch.resize(1);
    bunch.x[0] = bunch.y[0] = bunch.z[0] = 0.0;
    bunch.ux[0] = bunch.uy[0] = bunch.uz[0] = 0.0;
    bunch.weight[0] = 1.0;
    bunch.charge[0] = -picBunch::kElementaryCharge;

    double const cell = 1.0;
    double const peak = picBunch::kCoulombConstant * picBunch::kElementaryCharge;
    double const tolerance = 1.0e-3;
    auto const margin = picBunch::deriveFieldTailMargin(
        bunch, peak, tolerance, picBunch::Vec3{cell, cell, cell});

    double const target = tolerance * peak;
    EXPECT_LE(margin.bound, target);
    EXPECT_LE(margin.distance, std::sqrt(picBunch::kCoulombConstant * picBunch::kElementaryCharge / target) + 1.0e-9);
    // A point charge has R=0 and a 32-cell margin at 1e-3.
    EXPECT_EQ(margin.marginCells, (picBunch::Vec3{32.0, 32.0, 32.0}));
    // Just inside the margin the neglected field is above target; this is why
    // the boundary must be at the solved distance, not at the old 4 cells.
    double const inside = margin.distance - 1.0;
    EXPECT_GT(picBunch::kCoulombConstant * picBunch::kElementaryCharge / (inside * inside), target);
}

TEST(Domain, TailMarginIncludesDipoleForNeutralBunch)
{
    picBunch::Bunch bunch;
    bunch.resize(2);
    bunch.x = {0.5, -0.5};
    bunch.y = {0.0, 0.0};
    bunch.z = {0.0, 0.0};
    bunch.ux = {0.0, 0.0};
    bunch.uy = {0.0, 0.0};
    bunch.uz = {0.0, 0.0};
    bunch.weight = {1.0, 1.0};
    bunch.charge = {picBunch::kElementaryCharge, -picBunch::kElementaryCharge};

    double const cell = 1.0;
    double const peak = picBunch::kCoulombConstant * picBunch::kElementaryCharge;
    double const tolerance = 1.0e-3;
    auto const margin = picBunch::deriveFieldTailMargin(
        bunch, peak, tolerance, picBunch::Vec3{cell, cell, cell});

    EXPECT_NEAR(margin.totalCharge, 0.0, 1.0e-30);
    EXPECT_NEAR(margin.dipoleMagnitude, picBunch::kElementaryCharge, 1.0e-30);
    EXPECT_LE(margin.bound, tolerance * peak);
    // The dipole term, not a zero monopole, sets this margin.
    EXPECT_GT(margin.marginCells[0], 0.0);
    EXPECT_LT(margin.marginCells[0], 32.0);
}

TEST(Domain, GlobalDomainEndingInsideMarginHasNoZeroHole)
{
    picBunch::Bunch bunch;
    bunch.resize(1);
    bunch.x[0] = bunch.y[0] = bunch.z[0] = 0.0;
    bunch.ux[0] = bunch.uy[0] = bunch.uz[0] = 0.0;
    bunch.weight[0] = 1.0;
    bunch.charge[0] = -picBunch::kElementaryCharge;

    picBunch::ComputeOptions options;
    options.cellSize = {1.0, 1.0, 1.0};
    options.marginCellsPerAxis = picBunch::Vec3{32.0, 32.0, 32.0};
    options.globalGridSize = picBunch::Idx3{16, 16, 16};
    options.globalGridOrigin = picBunch::Vec3{0.0, 0.0, 0.0};
    auto const grid = picBunch::makeGrid(bunch, options);

    // The requested margin exceeds the written domain, so makeGrid clips the
    // compute box to the whole domain: there is no interior zero-fill region.
    EXPECT_EQ(grid.n, (picBunch::Idx3{16, 16, 16}));
    EXPECT_EQ(grid.origin, (picBunch::Vec3{0.0, 0.0, 0.0}));
}

TEST(Domain, TailMarginIncludesBeaming)
{
    picBunch::Bunch bunch;
    bunch.resize(1);
    bunch.x[0] = bunch.y[0] = bunch.z[0] = 0.0;
    bunch.ux[0] = bunch.uy[0] = 0.0;
    bunch.uz[0] = 10.0;
    bunch.weight[0] = 1.0;
    bunch.charge[0] = -picBunch::kElementaryCharge;

    double const peak = picBunch::kCoulombConstant * picBunch::kElementaryCharge;
    double const tolerance = 1.0e-3;
    auto const margin = picBunch::deriveFieldTailMargin(
        bunch, peak, tolerance, picBunch::Vec3{1.0, 1.0, 1.0});

    EXPECT_NEAR(margin.maxGamma, std::sqrt(101.0), 1.0e-12);
    EXPECT_LE(margin.bound, tolerance * peak);
    // The boosted transverse field needs a larger margin than the rest case.
    EXPECT_GT(margin.marginCells[0], 32.0);
}
