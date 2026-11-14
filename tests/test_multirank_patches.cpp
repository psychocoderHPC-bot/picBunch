// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group MultiRankPatches: the checkpoint's particlePatches table must have the
// structure PIConGPU's restart loader expects for any rank count: the records
// numParticles, numParticlesOffset, offset and extent, the vector records with
// x/y/z components, and one entry per patch covering the domain exactly once.

#include "picBunch/Mpi.hpp"
#include "picBunch/OpenPMDWriter.hpp"
#include "picBunch/Types.hpp"

#include <gtest/gtest.h>

#include <hdf5.h>

#if defined(PICBUNCH_ENABLE_MPI)
#    include <mpi.h>
#endif

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>
#include <unistd.h>

namespace
{
    //! Minimal openPMD/HDF5 input with one empty species.  The generated
    //! species overwrites it, so no particle records are needed.
    void writeMinimalInput(std::filesystem::path const& path, std::string const& species)
    {
        hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        auto stringAttribute = [&](char const* name, std::string const& value) {
            hid_t type = H5Tcopy(H5T_C_S1);
            H5Tset_size(type, H5T_VARIABLE);
            hid_t space = H5Screate(H5S_SCALAR);
            hid_t attribute
                = H5Acreate2(file, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
            char const* pointer = value.c_str();
            H5Awrite(attribute, type, &pointer);
            H5Aclose(attribute);
            H5Sclose(space);
            H5Tclose(type);
        };
        auto doubleAttribute = [](hid_t object, char const* name, double value) {
            hid_t space = H5Screate(H5S_SCALAR);
            hid_t attribute
                = H5Acreate2(object, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
            H5Awrite(attribute, H5T_NATIVE_DOUBLE, &value);
            H5Aclose(attribute);
            H5Sclose(space);
        };
        stringAttribute("openPMD", "2.0.0");
        stringAttribute("iterationEncoding", "groupBased");
        stringAttribute("iterationFormat", "%T");
        stringAttribute("basePath", "/data/%T/");
        stringAttribute("meshesPath", "fields/");
        stringAttribute("particlesPath", "particles/");
        {
            hid_t space = H5Screate(H5S_SCALAR);
            hid_t attribute = H5Acreate2(
                file, "openPMDextension", H5T_NATIVE_UINT32, space, H5P_DEFAULT, H5P_DEFAULT);
            std::uint32_t const extension = 1u;
            H5Awrite(attribute, H5T_NATIVE_UINT32, &extension);
            H5Aclose(attribute);
            H5Sclose(space);
        }
        H5Gcreate2(file, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t iteration
            = H5Gcreate2(file, "data/0", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        doubleAttribute(iteration, "dt", 1.0);
        doubleAttribute(iteration, "time", 0.0);
        doubleAttribute(iteration, "timeUnitSI", 1.0);
        hid_t particles
            = H5Gcreate2(iteration, "particles", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t sp = H5Gcreate2(
            particles, species.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // openPMD only lists a species that carries at least one extent-bearing
        // record.  One position sample keeps the group valid; the generated
        // species overwrites it and never reads these values.
        hid_t position
            = H5Gcreate2(sp, "position", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        {
            hsize_t const seven = 7;
            hid_t space = H5Screate_simple(1, &seven, nullptr);
            hid_t attribute = H5Acreate2(
                position, "unitDimension", H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
            std::array<double, 7> const unitDimension{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            H5Awrite(attribute, H5T_NATIVE_DOUBLE, unitDimension.data());
            H5Aclose(attribute);
            H5Sclose(space);
        }
        auto scalarAttribute = [](hid_t object, char const* name, hid_t type, void const* value) {
            hid_t space = H5Screate(H5S_SCALAR);
            hid_t attribute
                = H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
            H5Awrite(attribute, type, value);
            H5Aclose(attribute);
            H5Sclose(space);
        };
        double const timeOffset = 0.0;
        std::uint32_t const macroWeighted = 0u;
        double const weightingPower = 0.0;
        scalarAttribute(position, "timeOffset", H5T_NATIVE_DOUBLE, &timeOffset);
        scalarAttribute(position, "macroWeighted", H5T_NATIVE_UINT32, &macroWeighted);
        scalarAttribute(position, "weightingPower", H5T_NATIVE_DOUBLE, &weightingPower);
        for(char const* axis : {"x", "y", "z"})
        {
            hsize_t const one = 1;
            hid_t space = H5Screate_simple(1, &one, nullptr);
            hid_t dataset = H5Dcreate2(
                position, axis, H5T_IEEE_F32LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            float const value = 0.5f;
            H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
            hid_t attributeSpace = H5Screate(H5S_SCALAR);
            hid_t unit = H5Acreate2(
                dataset, "unitSI", H5T_NATIVE_DOUBLE, attributeSpace, H5P_DEFAULT, H5P_DEFAULT);
            double const unitSI = 1.0;
            H5Awrite(unit, H5T_NATIVE_DOUBLE, &unitSI);
            H5Aclose(unit);
            H5Sclose(attributeSpace);
            H5Sclose(space);
            H5Dclose(dataset);
        }
        H5Gclose(position);
        // Explicit flush: H5Fclose alone can leave the file truncated on some
        // overlay filesystems, which breaks the peers' collective open.
        H5Fflush(file, H5F_SCOPE_GLOBAL);
        H5Fclose(file);
    }

    std::vector<std::uint64_t> readUint64(
        hid_t file,
        std::string const& path,
        std::vector<hsize_t>& shape)
    {
        hid_t dataset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
        EXPECT_GE(dataset, 0) << path;
        if(dataset < 0)
            return {};
        hid_t space = H5Dget_space(dataset);
        int const rank = H5Sget_simple_extent_ndims(space);
        shape.assign(static_cast<std::size_t>(rank), 0u);
        H5Sget_simple_extent_dims(space, shape.data(), nullptr);
        H5Sclose(space);
        hid_t type = H5Dget_type(dataset);
        EXPECT_EQ(H5Tget_class(type), H5T_INTEGER) << path;
        EXPECT_EQ(H5Tget_size(type), 8u) << path;
        H5Tclose(type);
        hsize_t count = 1u;
        for(auto const length : shape)
            count *= length;
        std::vector<std::uint64_t> values(static_cast<std::size_t>(count), 0u);
        if(count != 0u)
            H5Dread(dataset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
        H5Dclose(dataset);
        return values;
    }
} // namespace

#if defined(PICBUNCH_ENABLE_MPI)
TEST(MultiRankPatches, PatchTableMatchesPIConGPUForEveryRankCount)
{
    int initialised = 0;
    MPI_Initialized(&initialised);
    if(!initialised)
        MPI_Init(nullptr, nullptr);
    int const rank = picBunch::mpiRank();
    int const size = picBunch::mpiSize();
    if(size < 2)
        GTEST_SKIP() << "run this group with mpirun -n 2 or -n 4";

    long long rootPid = rank == 0 ? static_cast<long long>(getpid()) : 0ll;
    MPI_Bcast(&rootPid, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    std::filesystem::path const directory = std::filesystem::temp_directory_path()
        / ("picbunch_mrpatches_" + std::to_string(rootPid));
    if(rank == 0)
        std::filesystem::remove_all(directory);
    MPI_Barrier(MPI_COMM_WORLD);

    auto const input = directory / "input.h5";
    if(rank == 0)
    {
        std::filesystem::create_directories(directory);
        writeMinimalInput(input, "electrons");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    constexpr std::size_t n = 8u;
    picBunch::Field field;
    field.grid.n = {n, n, n};
    field.grid.spacing = {1.0, 1.0, 1.0};
    field.grid.origin = {0.0, 0.0, 0.0};
    for(int c = 0; c < 6; ++c)
        field.data[c].assign(field.grid.size(), 0.0);

    picBunch::GeneratedParticles generated;
    generated.species = "electrons";
    generated.domainOrigin = {0.0, 0.0, 0.0};
    generated.cellSize = {1.0, 1.0, 1.0};
    generated.patchOffset = {0, 0, 0};
    generated.patchExtent = {n, n, n};
    generated.unitMassSpeed = 1.0;
    generated.unitCharge = 1.0;
    generated.unitMass = 1.0;
    generated.fallbackMass = 1.0;
    // One disjoint patch per rank, tiling z exactly once.
    for(int p = 0; p < size; ++p)
    {
        std::uint64_t const lo = (n * static_cast<std::uint64_t>(p))
            / static_cast<std::uint64_t>(size);
        std::uint64_t const hi = (n * static_cast<std::uint64_t>(p + 1))
            / static_cast<std::uint64_t>(size);
        picBunch::ParticlePatch patch;
        patch.offset = {0, 0, static_cast<std::int64_t>(lo)};
        patch.extent = {n, n, hi - lo};
        generated.patches.push_back(patch);
        generated.bunch.x.push_back(0.5);
        generated.bunch.y.push_back(0.5);
        generated.bunch.z.push_back(static_cast<double>(lo) + 0.5);
        generated.bunch.ux.push_back(0.0);
        generated.bunch.uy.push_back(0.0);
        generated.bunch.uz.push_back(0.0);
        generated.bunch.weight.push_back(1.0);
        generated.bunch.charge.push_back(-1.0);
    }

    picBunch::PicOnGPUWriteOptions options;
    options.step = 0;
    options.directory = directory.string();
    options.inputFile = input.string();
    options.inputIteration = 0;
    options.requiredInputSpecies = {"electrons"};
    options.timeUnitSI = 1.0;
    options.unitLength = 1.0;
    options.unitEField = 1.0;
    options.unitBField = 1.0;
    options.unitCharge = 1.0;
    options.unitMass = 1.0;
    options.generatedSpecies = {generated};
    picBunch::writePicOnGPUCheckpointOpenPMD(field, options);
    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0)
    {
        auto const path = directory / "checkpoint_000000.h5";
        hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        ASSERT_GE(file, 0);
        std::string const base = "data/0/particles/electrons/particlePatches/";
        std::vector<hsize_t> shape;
        auto const counts = readUint64(file, base + "numParticles", shape);
        ASSERT_EQ(shape, (std::vector<hsize_t>{static_cast<hsize_t>(size)}));
        auto const particleOffsets = readUint64(file, base + "numParticlesOffset", shape);
        ASSERT_EQ(shape.size(), 1u);
        ASSERT_EQ(static_cast<int>(shape[0]), size);
        ASSERT_EQ(counts.size(), static_cast<std::size_t>(size));
        ASSERT_EQ(particleOffsets.size(), static_cast<std::size_t>(size));
        std::uint64_t cumulative = 0u;
        for(int q = 0; q < size; ++q)
        {
            EXPECT_EQ(particleOffsets[static_cast<std::size_t>(q)], cumulative);
            cumulative += counts[static_cast<std::size_t>(q)];
        }
        EXPECT_EQ(cumulative, static_cast<std::uint64_t>(size));

        char const* const axes[3] = {"x", "y", "z"};
        std::array<std::vector<std::uint64_t>, 3> offsets;
        std::array<std::vector<std::uint64_t>, 3> extents;
        for(int a = 0; a < 3; ++a)
        {
            offsets[static_cast<std::size_t>(a)]
                = readUint64(file, base + "offset/" + axes[a], shape);
            ASSERT_EQ(static_cast<int>(shape[0]), size) << axes[a];
            extents[static_cast<std::size_t>(a)]
                = readUint64(file, base + "extent/" + axes[a], shape);
            ASSERT_EQ(static_cast<int>(shape[0]), size) << axes[a];
        }
        // x and y are full-domain; z intervals partition [0, n) exactly once.
        for(int q = 0; q < size; ++q)
        {
            EXPECT_EQ(offsets[0][static_cast<std::size_t>(q)], 0u);
            EXPECT_EQ(offsets[1][static_cast<std::size_t>(q)], 0u);
            EXPECT_EQ(extents[0][static_cast<std::size_t>(q)], n);
            EXPECT_EQ(extents[1][static_cast<std::size_t>(q)], n);
        }
        std::uint64_t covered = 0u;
        for(int q = 0; q < size; ++q)
        {
            EXPECT_EQ(offsets[2][static_cast<std::size_t>(q)], covered);
            EXPECT_GT(extents[2][static_cast<std::size_t>(q)], 0u);
            covered += extents[2][static_cast<std::size_t>(q)];
        }
        EXPECT_EQ(covered, n);
        H5Fclose(file);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0)
        std::filesystem::remove_all(directory);
}
#endif
