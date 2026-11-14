// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Regression test for the PIConGPU restart field scale (W4).
//
// The checkpoint stores the mesh field as value/unitSI (unitless, the
// PIConGPU convention) and writes the component unitSI from the same option
// set.  The test pins that contract:
//   1. the component unitSI attribute equals the configured deck unit,
//   2. the stored array equals the in-memory field divided by unitSI,
//   3. stored * unitSI recovers the in-memory SI value, i.e. the value a
//      PIConGPU consumer sees when it applies unitSI on read.
// A unit double conversion in the writer (storing SI while the consumer
// multiplies by unitSI again, or dividing by the unit twice) rescales every
// value by unitEField or unitBField, which is 1e13-1e27 for realistic decks
// and would show up as the stored value being ~1e13 off.
//
// It also pins the derived PIConGPU units against the simulation.unitless
// formula so a wrong unitEField/unitBField derivation is caught.

#include "picBunch/Compute.hpp"
#include "picBunch/Config.hpp"
#include "picBunch/OpenPMDWriter.hpp"

#include <openPMD/openPMD.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace
{
    //! Deck values matching the large-run configuration: dt = 6.63137190879337e-17
    //! s, cell = 1.772e-7 m, base density 1e24 m^-3, base charge -e, 2 typical
    //! particles per cell.  The derived E unit is ~1e13 V/m for this deck.
    constexpr double kDt = 6.63137190879337e-17;
    constexpr double kCell = 1.772e-7;
    constexpr double kBaseMass = 9.1093837139e-31;
    constexpr double kBaseCharge = -1.602176634e-19;
    constexpr double kBaseDensity = 1.0e24;
    constexpr double kPpc = 2.0;

    std::filesystem::path temporaryDirectory()
    {
        return std::filesystem::temp_directory_path()
            / ("picbunch_units_" + std::to_string(static_cast<long long>(getpid())));
    }

    void independentUnits(double& unitLength, double& unitEField, double& unitBField)
    {
        double const c = 299792458.0;
        double const L = kDt * c;
        double const npp = kBaseDensity * kCell * kCell * kCell / kPpc;
        double const unitMass = kBaseMass * npp;
        double const unitCharge = -1.0 * kBaseCharge * npp;
        unitLength = L;
        unitEField = 1.0 / (kDt * kDt / unitMass / L * unitCharge);
        unitBField = unitMass / (kDt * unitCharge);
    }

    //! Build a small field with known, non-trivial values.  E and B
    //! components near 1e10 V/m / T so stored*unitSI (unitE~1e13, unitB~1e5)
    //! lands near 1e23 and 1e15, both within float32 range.  Node spacing
    //! 1000 so the stored values differ by more than float32 epsilon.
    picBunch::Field makeField()
    {
        picBunch::Field field;
        field.grid.n = {4, 5, 6};
        field.grid.spacing = {kCell, kCell, kCell};
        field.grid.origin = {0.0, 0.0, 0.0};
        field.grid.fftPadding = 2;
        for(int c = 0; c < 6; ++c)
        {
            field.data[c].resize(field.grid.size());
            double const base = 1.0e10;
            for(std::size_t l = 0; l < field.grid.size(); ++l)
                field.data[c][l]
                    = base + 1.0e6 * static_cast<double>(c) + 1.0e3 * static_cast<double>(l);
        }
        return field;
    }

    //! Write a minimal openPMD particle file with one electron so the
    //! checkpoint writer has a valid input to copy species from.
    std::filesystem::path writeMinimalInput(std::filesystem::path const& dir)
    {
        std::filesystem::path const path = dir / "input.h5";
        openPMD::Series series(path.string(), openPMD::Access::CREATE);
        auto& root = series.iterations[0];
        root.setAttribute("time", 0.0);
        root.setAttribute("timeUnitSI", 1.0);
        root.setAttribute("dt", kDt);
        auto& species = root.particles["electrons"];
        species.setAttribute("numParticles", 1u);
        auto& pos = species["position"];
        pos.setUnitDimension({{openPMD::UnitDimension::L, 1.0}});
        auto& off = species["positionOffset"];
        off.setUnitDimension({{openPMD::UnitDimension::L, 1.0}});
        auto& mom = species["momentum"];
        mom.setUnitDimension(
            {{openPMD::UnitDimension::L, 1.0},
             {openPMD::UnitDimension::M, 1.0},
             {openPMD::UnitDimension::T, -1.0}});
        auto& wt = species["weighting"];
        wt.setUnitDimension({{openPMD::UnitDimension::I, 1.0}});
        double const massC = 9.1093837139e-31 * 299792458.0;
        double const position[3] = {8.0e-6, 8.0e-6, 8.0e-6};
        char const* const axisNames[3] = {"x", "y", "z"};
        for(int a = 0; a < 3; ++a)
        {
            auto& ds = pos[axisNames[a]];
            ds.resetDataset(openPMD::Dataset(openPMD::determineDatatype<double>(),
                                             openPMD::Extent{1}));
            ds.storeChunkRaw(&position[a], openPMD::Offset{0}, openPMD::Extent{1});
            ds.setUnitSI(1.0);
            auto& ods = off[axisNames[a]];
            ods.resetDataset(openPMD::Dataset(openPMD::determineDatatype<double>(),
                                              openPMD::Extent{1}));
            double const oval = 0.0;
            ods.storeChunkRaw(&oval, openPMD::Offset{0}, openPMD::Extent{1});
            ods.setUnitSI(kCell);
            auto& mds = mom[axisNames[a]];
            mds.resetDataset(openPMD::Dataset(openPMD::determineDatatype<double>(),
                                              openPMD::Extent{1}));
            double const pval = 0.0;
            mds.storeChunkRaw(&pval, openPMD::Offset{0}, openPMD::Extent{1});
            mds.setUnitSI(massC);
        }
        auto& wds = wt["value"];
        wds.resetDataset(openPMD::Dataset(openPMD::determineDatatype<double>(),
                                          openPMD::Extent{1}));
        double const wval = 1.0;
        wds.storeChunkRaw(&wval, openPMD::Offset{0}, openPMD::Extent{1});
        wds.setUnitSI(1.0);
        species.setAttribute("weightingPower", 0.0);
        series.flush();
        return path;
    }

    picBunch::PicOnGPUWriteOptions makeOptions(
        std::filesystem::path const& dir,
        std::filesystem::path const& input)
    {
        double unitLength = 0.0, unitEField = 0.0, unitBField = 0.0;
        independentUnits(unitLength, unitEField, unitBField);
        picBunch::PicOnGPUWriteOptions options;
        options.step = 7;
        options.directory = dir.string();
        options.inputFile = input.string();
        options.inputIteration = 0;
        options.requiredInputSpecies = {"electrons"};
        options.time = 0.0;
        options.timeUnitSI = 1.0;
        options.dt = kDt;
        options.unitLength = unitLength;
        options.unitEField = unitEField;
        options.unitBField = unitBField;
        options.unitTime = kDt;
        options.unitSpeed = 299792458.0;
        options.unitEnergy = 1.0;
        options.unitMass = 1.0;
        options.unitCharge = 1.0;
        options.backend = picBunch::CheckpointBackend::Hdf5;
        picBunch::GeneratedParticles generated;
        generated.species = "electrons";
        generated.inputSpecies = "electrons";
        generated.bunch.resize(1);
        generated.bunch.x = {8.0e-6};
        generated.bunch.y = {8.0e-6};
        generated.bunch.z = {8.0e-6};
        generated.bunch.ux = {0.0};
        generated.bunch.uy = {0.0};
        generated.bunch.uz = {0.0};
        generated.bunch.weight = {1.0};
        generated.bunch.charge = {-1.602176634e-19};
        generated.bunch.restMass = {9.1093837139e-31};
        generated.domainOrigin = {0.0, 0.0, 0.0};
        generated.cellSize = {kCell, kCell, kCell};
        generated.patchOffset = {0, 0, 0};
        generated.patchExtent = {64, 64, 64};
        generated.unitMassSpeed = 9.1093837139e-31 * 299792458.0;
        generated.unitCharge = 1.602176634e-19;
        generated.unitMass = 9.1093837139e-31;
        generated.fallbackMass = 9.1093837139e-31;
        options.generatedSpecies = {generated};
        return options;
    }
} // namespace

TEST(CheckpointFieldUnits, DerivedUnitsMatchSimulationUnitless)
{
    double unitLength = 0.0, unitEField = 0.0, unitBField = 0.0;
    independentUnits(unitLength, unitEField, unitBField);
    picBunch::Config::PicOnGPU p{};
    p.deltaT = kDt;
    p.cellSize = {kCell, kCell, kCell};
    p.baseMass = kBaseMass;
    p.baseCharge = kBaseCharge;
    p.baseDensity = kBaseDensity;
    p.typicalParticlesPerCell = 2;
    picBunch::derivePicOnGPUUnits(p);
    EXPECT_NEAR(p.unitLength, unitLength, 1.0e-15);
    EXPECT_NEAR(p.unitEField, unitEField, 1.0e-6 * std::abs(unitEField));
    EXPECT_NEAR(p.unitBField, unitBField, 1.0e-6 * std::abs(unitBField));
    // The E unit must be enormous for a realistic deck; a unit double
    // conversion rescales the stored field by this factor.
    EXPECT_GT(unitEField, 1.0e13);
    EXPECT_GT(unitBField, 1.0e4);
}

TEST(CheckpointFieldUnits, StoredFieldIsUnitlessAndRoundTrips)
{
    std::filesystem::path const dir = temporaryDirectory();
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    std::filesystem::path const input = writeMinimalInput(dir);
    picBunch::Field const field = makeField();
    picBunch::PicOnGPUWriteOptions const options = makeOptions(dir, input);
    double const unitE = options.unitEField;
    double const unitB = options.unitBField;

    picBunch::writePicOnGPUCheckpointOpenPMD(field, options);

    std::filesystem::path const file = dir / (options.prefix + "_000007.h5");
    ASSERT_TRUE(std::filesystem::exists(file));
    {
        // The checkpoint is a file-based series; open it with the iteration
        // pattern so snapshots() resolves to the iteration group.
        std::string const pattern = (std::filesystem::path(dir) /
                                     (options.prefix + "_%06T.h5"))
                                        .string();
        openPMD::Series series(pattern, openPMD::Access::READ_ONLY);
        auto& iteration = series.snapshots()[static_cast<long long>(options.step)];
        auto& eMesh = iteration.meshes["E"];
        auto& bMesh = iteration.meshes["B"];
        char const* const axisNames[3] = {"x", "y", "z"};

        // The component unitSI must equal the configured deck unit.
        for(int a = 0; a < 3; ++a)
        {
            EXPECT_NEAR(eMesh[axisNames[a]].unitSI(), unitE,
                        1.0e-6 * std::abs(unitE))
                << "E comp " << a << " unitSI";
            EXPECT_NEAR(bMesh[axisNames[a]].unitSI(), unitB,
                        1.0e-6 * std::abs(unitB))
                << "B comp " << a << " unitSI";
        }

        // The stored values must be field/unitSI (unitless), and stored*unitSI
        // must recover the in-memory SI value.  Both the in-memory field (x
        // fastest, then y, then z) and the dataset ((z, y, x) axis labels, C
        // order with x fastest) use the same linear layout, so node l in
        // field.data maps to index l in the dataset.
        double const invUnitE = 1.0 / unitE;
        double const invUnitB = 1.0 / unitB;
        std::size_t const nx = 4, ny = 5, nz = 6;
        std::size_t const total = nx * ny * nz;
        for(int a = 0; a < 3; ++a)
        {
            openPMD::Extent const extent{nz, ny, nx};
            openPMD::Offset const origin{0, 0, 0};
            auto eData = eMesh[axisNames[a]].loadChunk<float>(origin, extent);
            auto bData = bMesh[axisNames[a]].loadChunk<float>(origin, extent);
            series.flush();
            float const* const ePtr = eData.get();
            float const* const bPtr = bData.get();
            ASSERT_TRUE(ePtr != nullptr);
            ASSERT_TRUE(bPtr != nullptr);
            for(std::size_t l = 0; l < total; ++l)
            {
                float const storedEf = ePtr[l];
                float const storedBf = bPtr[l];
                double const expectE = field.data[a][l] * invUnitE;
                double const expectB = field.data[a + 3][l] * invUnitB;
                // Stored value must equal the unitless field at float32
                // precision; a unit double conversion would be off by
                // unitEField (~1e13 for this deck).
                EXPECT_NEAR(storedEf, static_cast<float>(expectE),
                            1.0e-6 * std::abs(static_cast<float>(expectE)) + 1.0e-30f)
                    << "E comp " << a << " node " << l;
                EXPECT_NEAR(storedBf, static_cast<float>(expectB),
                            1.0e-6 * std::abs(static_cast<float>(expectB)) + 1.0e-30f)
                    << "B comp " << a << " node " << l;
                // The SI round-trip a PIConGPU consumer performs on read must
                // recover the in-memory value at float32 precision.
                EXPECT_NEAR(storedEf * unitE, field.data[a][l],
                            1.0e-5 * std::abs(field.data[a][l]) + 1.0e-12)
                    << "E comp " << a << " node " << l << " round-trip";
                EXPECT_NEAR(storedBf * unitB, field.data[a + 3][l],
                            1.0e-5 * std::abs(field.data[a + 3][l]) + 1.0e-12)
                    << "B comp " << a << " node " << l << " round-trip";
            }
        }
        series.close();
    }
    std::filesystem::remove_all(dir, ec);
}
