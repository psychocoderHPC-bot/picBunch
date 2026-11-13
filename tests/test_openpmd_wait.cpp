// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group OpenPmdWait: the host/device queue ordering at the openPMD boundary.
//
// The openPMD writer consumes host ``std::vector`` data.  A producer that fills
// those vectors from an alpaka queue must wait on that queue before it hands the
// data to the writer, otherwise the writer and openPMD read a buffer whose
// kernels are still in flight.  The first test makes that contract executable:
// it produces a field on the compiled backend's queue, waits explicitly, then
// writes, flushes and reads the file back bit for bit.
//
// The second test pins the generated-particle slot fill.  The fill reorders
// particles by patch and computes the in-cell position and cell offset with
// ``floor(cells + 1e-9)``.  It compares the on-disk records with an independent
// reference implementation bit for bit, so a later vectorisation of the loop
// cannot silently change rounding or patch order.
//
// Both the input fixture and the output checks use the HDF5 C API on purpose:
// the test pins the exact on-disk bytes, dtypes and patch order produced by the
// openPMD writer, so it must not round-trip through the abstraction it verifies.

#include "picBunch/OpenPMDWriter.hpp"
#include "picBunch/Types.hpp"

#include "TestDevice.hpp"

#include <alpaka/alpaka.hpp>
#include <openPMD/openPMD.hpp>

#include <hdf5.h>
#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

namespace
{
    std::filesystem::path temporaryDirectory()
    {
        return std::filesystem::temp_directory_path()
            / ("picbunch_openpmd_wait_" + std::to_string(static_cast<long long>(getpid())));
    }

    //! Exactly representable, component- and index-dependent value.
    //! Host and device form the same bit pattern; the kernel below calls it.
    ALPAKA_FN_HOST_ACC double producedValue(std::size_t component, std::size_t index)
    {
        std::uint64_t h = static_cast<std::uint64_t>(index) * 1000003u
                        + static_cast<std::uint64_t>(component + 1u) * 17u;
        h ^= h >> 17;
        h *= 0x9e3779b97f4a7c15ull;
        h ^= h >> 29;
        return static_cast<double>(h % 1000001ull) / 1024.0;
    }

    struct FillFieldKernel
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            double* out,
            std::uint32_t component,
            std::uint32_t count) const
        {
            for(auto [l] : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInGrid,
                    alpaka::IdxRange{count}))
            {
                // Deliberate latency: the dependent chain keeps the kernel in
                // flight long enough that a missing host wait is observable on
                // an asynchronous queue.  The stored value is still the exact
                // producedValue(); the chain only gates the branch.
                double delay = producedValue(component, l);
                for(int r = 0; r < 64; ++r)
                    delay = alpaka::math::sqrt(delay * delay + 1.0);
                out[l] = (delay < -1.0) ? -delay : producedValue(component, l);
            }
        }
    };

    picBunch::Field makeField(picBunch::Idx3 n = {3, 4, 5})
    {
        picBunch::Field field;
        field.grid.n = n;
        field.grid.spacing = {0.5, 1.5, 2.5};
        field.grid.origin = {1.0, 2.0, 3.0};
        field.grid.fftPadding = 2;
        for(int c = 0; c < 6; ++c)
            field.data[static_cast<std::size_t>(c)].resize(field.grid.size());
        return field;
    }

    char const kRecordName[2] = {'E', 'B'};
    char const kComponentName[3] = {'x', 'y', 'z'};

    void expectBitEqual(double stored, double expected, std::string const& where)
    {
        EXPECT_EQ(std::bit_cast<std::uint64_t>(stored), std::bit_cast<std::uint64_t>(expected))
            << where;
    }

    /* ---- minimal PIConGPU-style input used by the checkpoint writer ------- */

    hid_t createDataset(hid_t loc, char const* name, hid_t type, std::vector<hsize_t> const& dims)
    {
        hid_t space = H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr);
        hid_t dataset = H5Dcreate2(loc, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Sclose(space);
        return dataset;
    }

    void writeFloatComponent(hid_t parent, char const* name, float value)
    {
        hid_t dataset = createDataset(parent, name, H5T_IEEE_F32LE, {1});
        H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
        hid_t space = H5Screate(H5S_SCALAR);
        hid_t attribute
            = H5Acreate2(dataset, "unitSI", H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
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
        hsize_t dimensions[1] = {7};
        hid_t space = H5Screate_simple(1, dimensions, nullptr);
        hid_t attribute = H5Acreate2(record, "unitDimension", H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attribute, H5T_NATIVE_DOUBLE, &unitDimension[0]);
        H5Aclose(attribute);
        H5Sclose(space);
        writeDoubleAttribute(record, "weightingPower", weightingPower);
        writeUInt32Attribute(record, "macroWeighted", macroWeighted ? 1u : 0u);
        writeDoubleAttribute(record, "timeOffset", 0.0);
    }

    void writeInputSpecies(std::filesystem::path const& path)
    {
        hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        ASSERT_GE(file, 0);
        // The openPMD reader requires the series attributes; the fixture is a
        // plain openPMD 1.x tree written without openPMD-api.
        writeStringAttribute(file, "openPMD", "1.1.0");
        writeStringAttribute(file, "iterationEncoding", "groupBased");
        writeStringAttribute(file, "iterationFormat", "%T");
        writeStringAttribute(file, "basePath", "/data/%T/");
        writeStringAttribute(file, "meshesPath", "fields/");
        writeStringAttribute(file, "particlesPath", "particles/");
        writeUInt32Attribute(file, "openPMDextension", 1);
        hid_t data = H5Gcreate2(file, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t iteration = H5Gcreate2(data, "0", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        writeDoubleAttribute(iteration, "dt", 1.0);
        writeDoubleAttribute(iteration, "time", 0.0);
        writeDoubleAttribute(iteration, "timeUnitSI", 1.0);
        hid_t particles
            = H5Gcreate2(iteration, "particles", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t species
            = H5Gcreate2(particles, "electrons", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        writeStringAttribute(species, "sourceMarker", "preserved");
        hid_t position
            = H5Gcreate2(species, "position", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        writeRecordAttributes(position, {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, false, 0.0);
        hid_t momentum
            = H5Gcreate2(species, "momentum", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        writeRecordAttributes(momentum, {1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0}, true, 1.0);
        hid_t offset
            = H5Gcreate2(species, "positionOffset", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        writeRecordAttributes(offset, {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, false, 0.0);
        for(int a = 0; a < 3; ++a)
        {
            std::string const axis(1, kComponentName[a]);
            writeFloatComponent(position, axis.c_str(), 0.25f + 0.1f * static_cast<float>(a));
            writeFloatComponent(momentum, axis.c_str(), 0.0f);
            hid_t dataset = createDataset(offset, axis.c_str(), H5T_STD_I32LE, {1});
            std::int32_t const stored = a;
            H5Dwrite(dataset, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, &stored);
            writeUnitSI(dataset, 1.0);
            H5Dclose(dataset);
        }
        H5Gclose(position);
        H5Gclose(momentum);
        H5Gclose(offset);
        hid_t weighting = createDataset(species, "weighting", H5T_IEEE_F32LE, {1});
        float const weight = 1.0f;
        H5Dwrite(weighting, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &weight);
        writeUnitSI(weighting, 1.0);
        writeRecordAttributes(weighting, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, true, 1.0);
        H5Dclose(weighting);
        hid_t patches
            = H5Gcreate2(species, "particlePatches", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t count = createDataset(patches, "numParticles", H5T_STD_U64LE, {1});
        std::uint64_t const one = 1;
        H5Dwrite(count, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, &one);
        H5Dclose(count);
        hid_t countOffset = createDataset(patches, "numParticlesOffset", H5T_STD_U64LE, {1});
        std::uint64_t const zero = 0;
        H5Dwrite(countOffset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, &zero);
        H5Dclose(countOffset);
        H5Gclose(patches);
        H5Gclose(species);
        H5Gclose(particles);
        H5Gclose(iteration);
        H5Gclose(data);
        H5Fclose(file);
    }

    std::vector<double> readDoubleDataset(hid_t file, char const* path)
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

    std::vector<std::int32_t> readInt32Dataset(hid_t file, char const* path)
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
} // namespace

TEST(OpenPmdWait, QueueProducedFieldIsWaitedBeforeWriteAndReadBack)
{
    auto const directory = temporaryDirectory() / "field";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    auto const path = directory / "queue_produced.h5";

    // Cubic dimensions keep the read-back extent free of the x-fastest
    // transposition used for the small default field.
    constexpr std::size_t kSide = 64u;
    picBunch::Idx3 const dims{kSide, kSide, kSide};

    picBunchTest::runOnDevice(
        [&](auto& device, auto& queue)
        {
            picBunch::Field field = makeField(dims);
            std::size_t const count = field.grid.size();
            auto staging = alpaka::onHost::allocUnified<double>(device, count);

            for(std::uint32_t c = 0; c < 6u; ++c)
            {
                std::uint32_t const elements = static_cast<std::uint32_t>(count);
                queue.enqueue(
                    alpaka::onHost::FrameSpec{(elements + 63u) / 64u, 64u},
                    alpaka::KernelBundle{
                        FillFieldKernel{},
                        staging.data(),
                        c,
                        elements});

                // The explicit host wait is the contract this test pins: the
                // host copy below and every later openPMD call may only see the
                // completed kernel.  With the large, delayed workload above,
                // removing this wait makes the read-back compare against
                // in-flight or stale data on an asynchronous queue (CUDA, HIP
                // and the async host queue).  A queue that executes enqueues
                // synchronously cannot expose the missing wait; that is a
                // property of the backend, not of this contract.
                alpaka::onHost::wait(queue);

                for(std::size_t l = 0; l < count; ++l)
                    field.data[c][l] = staging.data()[l];
            }

            picBunch::FieldWriteOptions options;
            options.iteration = 3;
            picBunch::writeField(field, path.string(), options);
        });

    openPMD::Series series(path.string(), openPMD::Access::READ_ONLY);
    auto& iteration = series.iterations[3];
    openPMD::Extent const extent{
        static_cast<openPMD::Extent::value_type>(dims[0]),
        static_cast<openPMD::Extent::value_type>(dims[1]),
        static_cast<openPMD::Extent::value_type>(dims[2])};
    openPMD::Offset const offset{0, 0, 0};
    std::size_t const count = dims[0] * dims[1] * dims[2];
    for(std::size_t c = 0; c < 6u; ++c)
    {
        std::string const record(1, kRecordName[c / 3u]);
        std::string const component(1, kComponentName[c % 3u]);
        auto raw = iteration.meshes[record][component].loadChunk<double>(offset, extent);
        series.flush();
        for(std::size_t l = 0; l < count; ++l)
            expectBitEqual(
                raw.get()[l],
                producedValue(c, l),
                record + "/" + component + " index " + std::to_string(l));
    }
    std::filesystem::remove_all(temporaryDirectory());
}

TEST(OpenPmdWait, ParticleSlotFillIsBitIdenticalToReference)
{
    auto const directory = temporaryDirectory() / "particles";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    auto const input = directory / "input.h5";
    writeInputSpecies(input);

    picBunch::GeneratedParticles generated;
    generated.species = "electrons";
    // Interleaved patch membership plus values on both sides of a cell boundary
    // (the +1e-9 term) and exact integers.
    generated.bunch.x = {0.1, 0.85, 0.475, 1.975};
    generated.bunch.y = {-0.2, 1.05, 1.0499999999, 2.3};
    generated.bunch.z = {0.3, 1.3, 0.3, 1.8};
    generated.bunch.ux = {0.1, -0.2, 0.3, -0.4};
    generated.bunch.uy = {1.0, 2.0, 3.0, 4.0};
    generated.bunch.uz = {-1.0, 0.5, -0.25, 2.0};
    generated.bunch.weight = {1.0, 2.5, 0.25, 4.0};
    generated.bunch.charge = {-1.0, -2.0, -0.5, -8.0};
    double const mass = picBunch::kElectronMass;
    generated.bunch.restMass = {mass, 0.0, mass, -1.0}; // 0/negative use the fallback
    generated.domainOrigin = {0.1, -0.2, 0.3};
    generated.cellSize = {0.75, 1.25, 0.5};
    generated.patchOffset = {0, 0, 0};
    generated.patchExtent = {8, 8, 4};
    generated.patches = {
        {{0, 0, 0}, {8, 8, 2}},
        {{0, 0, 2}, {8, 8, 2}}};
    generated.unitMassSpeed = mass * picBunch::kSpeedOfLight;
    generated.unitCharge = picBunch::kElementaryCharge;
    generated.unitMass = mass;
    generated.fallbackMass = 2.0 * mass;

    picBunch::PicOnGPUWriteOptions options;
    options.step = 3;
    options.directory = directory.string();
    options.inputFile = input.string();
    options.inputIteration = 0;
    options.requiredInputSpecies = {"electrons"};
    options.unitLength = 1.0;
    options.unitEField = 1.0;
    options.unitBField = 1.0;
    options.float64 = true;
    options.generatedSpecies = {generated};
    picBunch::writePicOnGPUCheckpointOpenPMD(makeField(), options);

    std::size_t const count = generated.bunch.size();
    // Reference: assign each particle to the first patch that contains its
    // cell, then concatenate per patch and run the documented transform.
    std::vector<std::vector<std::size_t>> owned(generated.patches.size());
    for(std::size_t p = 0; p < count; ++p)
    {
        double const xyz[3]
            = {generated.bunch.x[p], generated.bunch.y[p], generated.bunch.z[p]};
        bool found = false;
        for(std::size_t q = 0; q < generated.patches.size() && !found; ++q)
        {
            bool inside = true;
            for(int a = 0; a < 3; ++a)
            {
                double const cells
                    = (xyz[a] - generated.domainOrigin[a]) / generated.cellSize[a];
                std::int64_t const cell
                    = static_cast<std::int64_t>(std::floor(cells + 1e-9));
                std::int64_t const lo = generated.patches[q].offset[a];
                std::int64_t const hi
                    = lo + static_cast<std::int64_t>(generated.patches[q].extent[a]);
                if(cell < lo || cell >= hi)
                    inside = false;
            }
            if(inside)
            {
                owned[q].push_back(p);
                found = true;
            }
        }
        ASSERT_TRUE(found);
    }
    std::vector<std::size_t> order;
    for(auto const& patchParticles : owned)
        order.insert(order.end(), patchParticles.begin(), patchParticles.end());

    std::array<std::vector<double>, 3> position;
    std::array<std::vector<std::int32_t>, 3> positionOffset;
    std::array<std::vector<double>, 3> momentum;
    std::vector<double> weighting(count);
    std::vector<double> charge(count);
    std::vector<double> massRecord(count);
    for(int a = 0; a < 3; ++a)
    {
        position[a].resize(count);
        positionOffset[a].resize(count);
        momentum[a].resize(count);
    }
    for(std::size_t slot = 0; slot < count; ++slot)
    {
        std::size_t const p = order[slot];
        double const xyz[3]
            = {generated.bunch.x[p], generated.bunch.y[p], generated.bunch.z[p]};
        double const u[3]
            = {generated.bunch.ux[p], generated.bunch.uy[p], generated.bunch.uz[p]};
        double const weight = generated.bunch.weight[p];
        double const restMass = generated.bunch.restMass[p] > 0.0
            ? generated.bunch.restMass[p]
            : generated.fallbackMass;
        for(int a = 0; a < 3; ++a)
        {
            double const cells
                = (xyz[a] - generated.domainOrigin[a]) / generated.cellSize[a];
            double const cellIndex = std::floor(cells + 1e-9);
            position[a][slot] = cells - cellIndex;
            positionOffset[a][slot] = static_cast<std::int32_t>(cellIndex);
            momentum[a][slot] = u[a] * restMass * weight * picBunch::kSpeedOfLight
                              / generated.unitMassSpeed;
        }
        weighting[slot] = weight;
        charge[slot] = generated.bunch.charge[p] / generated.unitCharge;
        massRecord[slot] = restMass / generated.unitMass;
    }

    auto const path = directory / "checkpoint_000003.h5";
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    ASSERT_GE(file, 0);
    std::string const prefix = "data/3/particles/electrons/";
    for(int a = 0; a < 3; ++a)
    {
        std::string const axis(1, kComponentName[a]);
        auto const storedPosition = readDoubleDataset(file, (prefix + "position/" + axis).c_str());
        auto const storedOffset
            = readInt32Dataset(file, (prefix + "positionOffset/" + axis).c_str());
        auto const storedMomentum
            = readDoubleDataset(file, (prefix + "momentum/" + axis).c_str());
        ASSERT_EQ(storedPosition.size(), count);
        ASSERT_EQ(storedOffset.size(), count);
        ASSERT_EQ(storedMomentum.size(), count);
        for(std::size_t slot = 0; slot < count; ++slot)
        {
            expectBitEqual(
                storedPosition[slot],
                position[a][slot],
                "position/" + axis + " slot " + std::to_string(slot));
            EXPECT_EQ(storedOffset[slot], positionOffset[a][slot])
                << "positionOffset/" << axis << " slot " << slot;
            expectBitEqual(
                storedMomentum[slot],
                momentum[a][slot],
                "momentum/" + axis + " slot " + std::to_string(slot));
        }
    }
    auto const storedWeighting = readDoubleDataset(file, (prefix + "weighting").c_str());
    auto const storedCharge = readDoubleDataset(file, (prefix + "charge").c_str());
    auto const storedMass = readDoubleDataset(file, (prefix + "mass").c_str());
    ASSERT_EQ(storedWeighting.size(), count);
    ASSERT_EQ(storedCharge.size(), count);
    ASSERT_EQ(storedMass.size(), count);
    for(std::size_t slot = 0; slot < count; ++slot)
    {
        expectBitEqual(storedWeighting[slot], weighting[slot], "weighting slot " + std::to_string(slot));
        expectBitEqual(storedCharge[slot], charge[slot], "charge slot " + std::to_string(slot));
        expectBitEqual(storedMass[slot], massRecord[slot], "mass slot " + std::to_string(slot));
    }
    H5Fclose(file);
    std::filesystem::remove_all(temporaryDirectory());
}
