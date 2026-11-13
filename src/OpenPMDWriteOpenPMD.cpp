// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// openPMD-Series checkpoint writer (HDF5 and ADIOS2/BP5 backends).
//
// The historical checkpoint writer (OpenPMDWrite.cpp) is built directly on the
// HDF5 C API.  It cannot target the ADIOS2/BP5 backend: a BP5 file is not an
// HDF5 file, the per-slab ``H5Sselect_hyperslab`` + ``H5Dwrite`` pattern is
// rejected by HDF5 1.10.10 with a non-zero dataset offset, and copying a species
// from a BP5 input with ``H5Ocopy`` is impossible.  This file writes the *same*
// PIConGPU-restart tree through the backend-agnostic openPMD C++ API, so one
// code path produces an identical openPMD layout for both ``.h5`` (HDF5
// backend) and ``.bp5`` (ADIOS2 backend).
//
// Each field component is one record component with the full ``(Nz, Ny, Nx)``
// shape, matching a real PIConGPU checkpoint: the values are converted to the
// configured precision, written with a single ``storeChunkRaw``, and flushed.
// This is the layout PIConGPU's restart reader expects.

#include "picBunch/Compute.hpp"
#include "picBunch/OpenPMDWriter.hpp"
#include "picBunch/Types.hpp"

#include <openPMD/Datatype.hpp>
#include <openPMD/Series.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace
{
    // Mirrors the constant in OpenPMDWrite.cpp (which is file-local there).
    char const* const kFloat64Warning
        = "float64 particle records are written; a stock PIConGPU cannot load "
          "them. Re-run with float32 (the default) for a restart-compatible "
          "checkpoint.";
} // namespace

namespace picBunch
{
    namespace
    {
        using openPMD::Access;
        using openPMD::Datatype;
        using openPMD::Dataset;
        using openPMD::Extent;
        using openPMD::Iteration;
        using openPMD::Mesh;
        using openPMD::MeshRecordComponent;
        using openPMD::Offset;
        using openPMD::ParticlePatches;
        using openPMD::ParticleSpecies;
        using openPMD::PatchRecord;
        using openPMD::PatchRecordComponent;
        using openPMD::Record;
        using openPMD::RecordComponent;
        using openPMD::Series;

        bool isBp5Path(std::string const& path)
        {
            return path.size() >= 4u && path.compare(path.size() - 4u, 4u, ".bp5") == 0;
        }

        //! Resolve the backend from the requested value and the destination
        //! path, then return the full output file path
        //! (``<dir>/<prefix>_%06T.<ext>``).  For ``Auto`` a ``.bp5`` destination
        //! selects ADIOS2, a ``.h5`` destination selects HDF5, and anything else
        //! falls back to HDF5 (the openPMD default extension).
        std::string resolveOutputPath(
            PicOnGPUWriteOptions const& options,
            std::string const& directory,
            CheckpointBackend& backend)
        {
            namespace fs = std::filesystem;
            std::string const dest = fs::path(directory).string();
            if(backend == CheckpointBackend::Auto)
                backend = isBp5Path(dest) ? CheckpointBackend::Adios2 : CheckpointBackend::Hdf5;
            char suffix[32];
            std::snprintf(suffix, sizeof(suffix), "%06llu",
                static_cast<unsigned long long>(options.step));
            std::string const extension
                = backend == CheckpointBackend::Adios2 ? "bp5" : "h5";
            return (fs::path(directory) / (options.prefix + "_" + suffix + "." + extension))
                .string();
        }

        //! The openPMD backend JSON config: ADIOS2 must be named explicitly so
        //! the backend and the file extension can never disagree; HDF5 uses the
        //! build default (empty config).
        std::string backendConfigJson(CheckpointBackend backend)
        {
            return backend == CheckpointBackend::Adios2 ? R"({"backend": "adios2"})"
                                                        : std::string{};
        }

        Datatype fieldDatatype(bool float64)
        {
            return float64 ? Datatype::DOUBLE : Datatype::FLOAT;
        }

        //! Allocate a full-shape dataset on a record component (one dataset per
        //! component, matching a real PIConGPU checkpoint).
        template<typename T_Rc>
        T_Rc& allocateFull(
            T_Rc& rc,
            Datatype type,
            std::size_t nz,
            std::size_t ny,
            std::size_t nx)
        {
            Dataset dataset(
                type,
                {static_cast<std::uint64_t>(nz),
                 static_cast<std::uint64_t>(ny),
                 static_cast<std::uint64_t>(nx)});
            rc.resetDataset(std::move(dataset));
            return rc;
        }

        //! Allocate a one-dimensional dataset.  PIConGPU's particle data and
        //! particlePatches records are all one-dimensional (one entry per
        //! particle / per MPI patch), not three-dimensional like the fields.
        template<typename T_Rc>
        T_Rc& allocate1D(T_Rc& rc, Datatype type, std::size_t length)
        {
            Dataset dataset(type, {static_cast<std::uint64_t>(length)});
            rc.resetDataset(std::move(dataset));
            return rc;
        }

        //! One particle species written from the output Series API.  ``bunch``
        //! is the placed bunch (SI metres, dimensionless momenta); the records
        //! are written in PIConGPU internal units with the supplied unit
        //! factors, matching the HDF5 writer's generated-species layout.
        void writeParticleSpecies(
            Iteration& iteration,
            std::string const& name,
            GeneratedParticles const& generated,
            bool float64,
            std::uint64_t idBase,
            std::function<void()> flushAfterStore)
        {
            Bunch const& bunch = generated.bunch;
            bool const empty = generated.empty || bunch.size() == 0u;
            std::size_t const count = empty ? 0u : bunch.size();
            if(generated.empty && bunch.size() != 0u)
                throw std::runtime_error(
                    "openPMD checkpoint writer: empty species '" + generated.species
                    + "' must not carry particles");
            if(!empty)
            {
                if(bunch.charge.size() != count)
                    throw std::runtime_error(
                        "openPMD checkpoint writer: generated species '" + generated.species
                        + "' has no per-particle charge");
                if(!(generated.unitCharge > 0.0))
                    throw std::runtime_error(
                        "openPMD checkpoint writer: generated species '" + generated.species
                        + "' needs a positive unit_charge");
                if(!(generated.unitMass > 0.0))
                    throw std::runtime_error(
                        "openPMD checkpoint writer: generated species '" + generated.species
                        + "' needs a positive unit_mass");
            }
            ParticleSpecies species = iteration.particles[name];
            auto const positionOf = [](Bunch const& b, std::string_view comp) {
                return comp == "x" ? b.x
                                   : comp == "y"
                                   ? b.y
                                   : b.z;
            };
            auto const momentumOf = [](Bunch const& b, std::string_view comp) {
                return comp == "x" ? b.ux
                                   : comp == "y"
                                   ? b.uy
                                   : b.uz;
            };
            auto const axisOf = [](std::string_view comp) {
                return comp == "x" ? 0 : comp == "y" ? 1 : 2;
            };
            // Record-level particle metadata, identical values to the HDF5
            // writer's setParticleRecordAttributes (unitDimension +
            // macroWeighted + weightingPower + timeOffset).  These are what a
            // real PIConGPU checkpoint carries; the openPMD restart loader
            // ignores them but other consumers and the byte-compatibility
            // guarantee need them present.
            auto setParticleRecordAttributes = [](
                Record& record,
                std::array<double, 7> const& unitDimension,
                bool macroWeighted,
                double weightingPower)
            {
                record.setUnitDimension(unitDimension);
                record.setAttribute(
                    "macroWeighted",
                    static_cast<std::uint32_t>(macroWeighted ? 1u : 0u));
                record.setAttribute("weightingPower", weightingPower);
                record.setAttribute("timeOffset", 0.0f);
            };

            // Patch layout: one region per rank-overlap slab when requested,
            // else the single supplied patch.  The patch metadata (offset/extent
            // in cells from the domain origin) is written identically to the
            // HDF5 writer; only the record data goes through openPMD.
            std::vector<ParticlePatch> patches = generated.patches;
            if(patches.empty())
            {
                ParticlePatch patch;
                patch.offset = generated.patchOffset;
                patch.extent = generated.patchExtent;
                patches.push_back(patch);
            }
            // Assign each particle to the first containing patch (disjoint
            // ownership, cumulative numParticlesOffset), as in the HDF5 writer.
            std::vector<std::vector<std::size_t>> owned(patches.size());
            for(std::size_t p = 0; p < count; ++p)
            {
                double const xyz[3] = {bunch.x[p], bunch.y[p], bunch.z[p]};
                std::int64_t cell[3];
                for(int a = 0; a < 3; ++a)
                    cell[a] = static_cast<std::int64_t>(
                        std::floor((xyz[a] - generated.domainOrigin[a])
                                     / generated.cellSize[a] + 1e-9));
                bool found = false;
                for(std::size_t q = 0; q < patches.size() && !found; ++q)
                {
                    bool inside = true;
                    for(int a = 0; a < 3; ++a)
                    {
                        std::int64_t const lo
                            = patches[q].offset[static_cast<std::size_t>(a)];
                        std::int64_t const hi
                            = lo + static_cast<std::int64_t>(
                                       patches[q].extent[static_cast<std::size_t>(a)]);
                        if(cell[a] < lo || cell[a] >= hi)
                            inside = false;
                    }
                    if(inside)
                    {
                        owned[q].push_back(p);
                        found = true;
                    }
                }
                if(!found)
                    throw std::runtime_error(
                        "openPMD checkpoint writer: particle " + std::to_string(p)
                        + " of species '" + generated.species
                        + "' lies outside every particle patch");
            }
            std::uint64_t running = 0;
            std::vector<std::uint64_t> patchCount(patches.size());
            std::vector<std::uint64_t> patchOffset(patches.size());
            for(std::size_t q = 0; q < patches.size(); ++q)
            {
                patchOffset[q] = running;
                patchCount[q] = static_cast<std::uint64_t>(owned[q].size());
                running += patchCount[q];
            }

            // position / positionOffset / momentum / weighting / id / charge /
            // mass records.  Positions and momenta are converted to PIConGPU
            // internal units (cell size and unit_mass*unit_speed);
            // positionOffset is the integer cell index in each patch's local
            // frame.  Record-level metadata and the (possibly zero-extent)
            // datasets are created for every species, so an empty species is a
            // valid (zero-particle) PIConGPU species, as in the HDF5 writer.
            // Data is stored only when count > 0: a zero-size storeChunkRaw
            // with a null pointer is rejected by the ADIOS2 backend.
            char const* const xyz[3] = {"x", "y", "z"};
            // Define every record component's dataset (zero-extent for empty
            // species) so the structure matches the HDF5 writer unconditionally.
            // The records are created here on first component access; the
            // record-level metadata (below) is set after the components exist,
            // so openPMD never sees a record that is written without
            // components.
            for(char const* comp : xyz)
            {
                int const axis = axisOf(comp);
                RecordComponent posRc = species["position"][comp];
                allocate1D(posRc, fieldDatatype(float64), count);
                posRc.setUnitSI(generated.cellSize[axis]);
                RecordComponent offRc = species["positionOffset"][comp];
                allocate1D(offRc, Datatype::INT, count);
                // The HDF5 writer stores positionOffset component unitSI as the
                // cell size (writeIntegerVectorRecord passes cellUnit), not 1.
                offRc.setUnitSI(generated.cellSize[axis]);
                RecordComponent momRc = species["momentum"][comp];
                allocate1D(momRc, fieldDatatype(float64), count);
                momRc.setUnitSI(generated.unitMassSpeed);
            }
            {
                RecordComponent wRc = species["weighting"][RecordComponent::SCALAR];
                allocate1D(wRc, fieldDatatype(float64), count);
                wRc.setUnitSI(1.0);
                RecordComponent idRc = species["id"][RecordComponent::SCALAR];
                allocate1D(idRc, Datatype::ULONGLONG, count);
                idRc.setUnitSI(1.0);
                RecordComponent chRc = species["charge"][RecordComponent::SCALAR];
                allocate1D(chRc, fieldDatatype(float64), count);
                chRc.setUnitSI(generated.unitCharge);
                RecordComponent maRc = species["mass"][RecordComponent::SCALAR];
                allocate1D(maRc, fieldDatatype(float64), count);
                maRc.setUnitSI(generated.unitMass);
            }
            // Record-level metadata (identical values to the HDF5 writer), set
            // now that each record has its component(s).
            setParticleRecordAttributes(
                species["position"],
                std::array<double, 7>{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                false,
                0.0);
            setParticleRecordAttributes(
                species["positionOffset"],
                std::array<double, 7>{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                false,
                0.0);
            setParticleRecordAttributes(
                species["momentum"],
                std::array<double, 7>{1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0},
                true,
                1.0);
            setParticleRecordAttributes(
                species["weighting"],
                std::array<double, 7>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                true,
                1.0);
            setParticleRecordAttributes(
                species["id"],
                std::array<double, 7>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                false,
                0.0);
            setParticleRecordAttributes(
                species["charge"],
                std::array<double, 7>{0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0},
                false,
                1.0);
            setParticleRecordAttributes(
                species["mass"],
                std::array<double, 7>{0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                false,
                1.0);
            if(count > 0)
            {
            for(char const* comp : xyz)
            {
                int const axis = axisOf(comp);
                RecordComponent posRc = species["position"][comp];
                std::vector<float> data;
                data.reserve(count);
                for(std::size_t p = 0; p < count; ++p)
                {
                    // PIConGPU stores position as the sub-cell offset
                    // (cells - cellIndex), not the absolute cell index.
                    double const cells = (positionOf(bunch, comp)[p]
                                          - generated.domainOrigin[axis])
                                         / generated.cellSize[axis];
                    data.push_back(static_cast<float>(
                        cells - std::floor(cells + 1e-9)));
                }
                posRc.storeChunkRaw(data.data(), Offset{0u}, Extent{count});
                flushAfterStore();
            }
            for(char const* comp : xyz)
            {
                int const axis = axisOf(comp);
                RecordComponent offRc = species["positionOffset"][comp];
                std::vector<std::int32_t> data(count, 0);
                for(std::size_t p = 0; p < count; ++p)
                    data[p] = static_cast<std::int32_t>(
                        std::floor((positionOf(bunch, comp)[p]
                                    - generated.domainOrigin[axis])
                                    / generated.cellSize[axis] + 1e-9)
                        - generated.patchOffset[axis]);
                offRc.storeChunkRaw(data.data(), Offset{0u}, Extent{count});
                flushAfterStore();
            }
            for(char const* comp : xyz)
            {
                RecordComponent momRc = species["momentum"][comp];
                std::vector<float> data;
                data.reserve(count);
                for(std::size_t p = 0; p < count; ++p)
                {
                    double const restMass = !bunch.restMass.empty() && bunch.restMass[p] > 0.0
                        ? bunch.restMass[p]
                        : generated.fallbackMass;
                    // PIConGPU openPMD momentum is the macro-particle momentum
                    // in units of unitMassSpeed: u*m*c*weight/unitMassSpeed.
                    data.push_back(static_cast<float>(
                        momentumOf(bunch, comp)[p] * restMass * bunch.weight[p]
                        * picBunch::kSpeedOfLight / generated.unitMassSpeed));
                }
                momRc.storeChunkRaw(data.data(), Offset{0u}, Extent{count});
                flushAfterStore();
            }
            // weighting (scalar record)
            {
                RecordComponent wRc = species["weighting"][RecordComponent::SCALAR];
                std::vector<float> data(count);
                for(std::size_t p = 0; p < count; ++p)
                    data[p] = static_cast<float>(bunch.weight[p]);
                wRc.storeChunkRaw(data.data(), Offset{0u}, Extent{count});
                flushAfterStore();
            }
            // id: global particle ids, disjoint across generated species and
            // spanning [idBase, idBase+count).  Scalar ULONGLONG record, as in
            // the HDF5 writer.
            {
                RecordComponent idRc = species["id"][RecordComponent::SCALAR];
                std::vector<std::uint64_t> ids(count);
                for(std::size_t i = 0; i < count; ++i)
                    ids[i] = idBase + static_cast<std::uint64_t>(i);
                idRc.storeChunkRaw(ids.data(), Offset{0u}, Extent{count});
                flushAfterStore();
            }
            // charge / mass: self-contained restart (values in internal units,
            // unitSI carries the TOML factor), as in the HDF5 writer.
            {
                RecordComponent chRc = species["charge"][RecordComponent::SCALAR];
                std::vector<float> data(count);
                for(std::size_t p = 0; p < count; ++p)
                    data[p] = static_cast<float>(bunch.charge[p] / generated.unitCharge);
                chRc.storeChunkRaw(data.data(), Offset{0u}, Extent{count});
                flushAfterStore();
            }
            {
                RecordComponent maRc = species["mass"][RecordComponent::SCALAR];
                std::vector<float> data(count);
                for(std::size_t p = 0; p < count; ++p)
                {
                    double const restMass = !bunch.restMass.empty() && bunch.restMass[p] > 0.0
                        ? bunch.restMass[p]
                        : generated.fallbackMass;
                    data[p] = static_cast<float>(restMass / generated.unitMass);
                }
                maRc.storeChunkRaw(data.data(), Offset{0u}, Extent{count});
                flushAfterStore();
            }
            } // if(count > 0)

            // particlePatches metadata (identical values to the HDF5 writer).
            // unitDimension lives on the record, unitSI=1 on each component;
            // the HDF5 writer puts unitDimension on the numParticles /
            // numParticlesOffset / offset / extent *groups*, not on x/y/z.
            {
                ParticlePatches pp = species.particlePatches;
                std::uint64_t const nPatches = patches.size();
                auto const patchUd
                    = std::array<double, 7>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                // PatchRecord only exposes a map-form setUnitDimension; setting
                // the attribute directly keeps the explicit [0,0,0,0,0,0,0]
                // (asMap would drop all zeros).  unitDimension belongs on the
                // record, not the x/y/z components.
                auto setRecordUd = [&patchUd](PatchRecord& rec) {
                    rec.setAttribute("unitDimension", patchUd);
                };
                PatchRecordComponent numPr = pp["numParticles"][RecordComponent::SCALAR];
                allocate1D(numPr, Datatype::ULONGLONG, nPatches);
                numPr.setUnitSI(1.0);
                std::vector<std::uint64_t> numData(patches.size());
                for(std::size_t q = 0; q < patches.size(); ++q)
                    numData[q] = patchCount[q];
                numPr.storeChunkRaw(numData.data(), Offset{0u}, Extent{nPatches});
                flushAfterStore();
                PatchRecordComponent numOffRc
                    = pp["numParticlesOffset"][RecordComponent::SCALAR];
                allocate1D(numOffRc, Datatype::ULONGLONG, nPatches);
                numOffRc.setUnitSI(1.0);
                std::vector<std::uint64_t> offData(patches.size());
                for(std::size_t q = 0; q < patches.size(); ++q)
                    offData[q] = patchOffset[q];
                numOffRc.storeChunkRaw(
                    offData.data(), Offset{0u}, Extent{nPatches});
                setRecordUd(pp["numParticles"]);
                setRecordUd(pp["numParticlesOffset"]);
                char const* const xyzc[3] = {"x", "y", "z"};
                for(char const* comp : xyzc)
                {
                    int const axis = axisOf(comp);
                    PatchRecordComponent offRc = pp["offset"][comp];
                    allocate1D(offRc, Datatype::ULONGLONG, nPatches);
                    offRc.setUnitSI(1.0);
                    std::vector<std::uint64_t> od(patches.size());
                    for(std::size_t q = 0; q < patches.size(); ++q)
                        od[q] = static_cast<std::uint64_t>(patches[q].offset[axis]);
                    offRc.storeChunkRaw(od.data(), Offset{0u}, Extent{nPatches});
                    flushAfterStore();
                    PatchRecordComponent extRc = pp["extent"][comp];
                    allocate1D(extRc, Datatype::ULONGLONG, nPatches);
                    extRc.setUnitSI(1.0);
                    std::vector<std::uint64_t> ed(patches.size());
                    for(std::size_t q = 0; q < patches.size(); ++q)
                        ed[q] = static_cast<std::uint64_t>(patches[q].extent[axis]);
                    extRc.storeChunkRaw(ed.data(), Offset{0u}, Extent{nPatches});
                    flushAfterStore();
                }
                // unitDimension belongs on the record, not the x/y/z components
                // (as in the HDF5 / PIConGPU writer).  Set it after the
                // components exist so the record is a valid non-empty record.
                setRecordUd(pp["offset"]);
                setRecordUd(pp["extent"]);
            }
        }

        //! Copy one input particle species to the output Series at the
        //! configured precision, normalizing dtypes exactly as the HDF5 writer
        //! (position/momentum float32 or float64, positionOffset int32,
        //! weighting flattened, charge/mass/id dropped).  This is the openPMD
        //! replacement for the HDF5 writer's ``H5Ocopy`` + per-record rewrite
        //! and works for both BP5 and HDF5 inputs.
        //! Read one record component from the input and write it to the output
        //! at the target dtype, returning the (double) source values.  The
        //! count is the extent of the source dataset.
        std::vector<double> loadRecordComponent(
            Series& in,
            RecordComponent& inRc,
            std::uint64_t& count)
        {
            // The particle record component is 1-D.  ``getExtent()`` returns
            // only the rank it was written with, so use the product of the
            // extents and an equally sized zero offset instead of assuming a
            // padded 3-D extent.
            Extent const ext = inRc.getExtent();
            count = 1u;
            for(auto const length : ext)
                count *= length;
            std::vector<double> buf(count);
            inRc.loadChunkRaw(buf.data(), Offset(ext.size(), 0u), ext);
            in.flush();
            return buf;
        }

        void copyInputSpecies(
            Series& in,
            long long inStep,
            Iteration& outIteration,
            std::string const& species,
            bool float64,
            std::function<void()> flushAfterStore)
        {
            ParticleSpecies inSpecies = in.snapshots()[inStep].particles[species];
            ParticleSpecies outSpecies = outIteration.particles[species];
            char const* const xyz[3] = {"x", "y", "z"};
            std::uint64_t count = 0;
            for(char const* comp : xyz)
            {
                RecordComponent inRc = inSpecies["position"][comp];
                std::vector<double> const buf = loadRecordComponent(in, inRc, count);
                RecordComponent outRc = outSpecies["position"][comp];
                allocate1D(outRc, fieldDatatype(float64), count);
                double const unitSI = inRc.getAttribute("unitSI").get<double>();
                outRc.setUnitSI(unitSI);
                outRc.setAttribute("position", std::vector<double>{0.0, 0.0, 0.0});
                std::vector<float> conv;
                conv.reserve(count);
                for(double const v : buf)
                    conv.push_back(static_cast<float>(v));
                outRc.storeChunkRaw(conv.data(), Offset{0u}, Extent{count});
                flushAfterStore();
            }
            for(char const* comp : xyz)
            {
                RecordComponent inRc = inSpecies["positionOffset"][comp];
                std::uint64_t c;
                std::vector<double> const buf = loadRecordComponent(in, inRc, c);
                RecordComponent outRc = outSpecies["positionOffset"][comp];
                allocate1D(outRc, Datatype::INT, c);
                outRc.setUnitSI(1.0);
                std::vector<std::int32_t> conv(c);
                for(std::uint64_t i = 0; i < c; ++i)
                    conv[i] = static_cast<std::int32_t>(buf[i]);
                outRc.storeChunkRaw(conv.data(), Offset{0u}, Extent{c});
                flushAfterStore();
            }
            // weighting (scalar).  An input written by h5py may store the
            // scalar as a "value" dataset inside the record group instead of
            // the openPMD SCALAR component; accept both.
            {
                Record inRecord = inSpecies["weighting"];
                RecordComponent inRc = inRecord.contains(RecordComponent::SCALAR)
                    ? inRecord[RecordComponent::SCALAR]
                    : inRecord["value"];
                std::uint64_t c;
                std::vector<double> const buf = loadRecordComponent(in, inRc, c);
                RecordComponent outRc
                    = outSpecies["weighting"][RecordComponent::SCALAR];
                allocate1D(outRc, fieldDatatype(float64), c);
                outRc.setUnitSI(1.0);
                std::vector<float> conv;
                conv.reserve(c);
                for(double const v : buf)
                    conv.push_back(static_cast<float>(v));
                outRc.storeChunkRaw(conv.data(), Offset{0u}, Extent{c});
                flushAfterStore();
            }
        }

        void publishPicOnGPUOpenPMD(
            Field const& field,
            PicOnGPUWriteOptions const& options,
            std::string const& finalName,
            CheckpointBackend backend)
        {
            if(!(options.unitLength > 0.0) || !std::isfinite(options.unitLength))
                throw std::invalid_argument("openPMD: unitLength must be finite and positive");
            if(!(options.unitEField > 0.0) || !std::isfinite(options.unitEField))
                throw std::invalid_argument("openPMD: unitEField must be finite and positive");
            if(!(options.unitBField > 0.0) || !std::isfinite(options.unitBField))
                throw std::invalid_argument("openPMD: unitBField must be finite and positive");
            if(!(options.timeUnitSI > 0.0) || !std::isfinite(options.timeUnitSI))
                throw std::invalid_argument("openPMD: timeUnitSI must be finite and positive");
            if(options.step > 999999u)
                throw std::invalid_argument("openPMD: step must fit a six-digit suffix");
            if(field.grid.size() == 0u)
                throw std::runtime_error("openPMD: grid is empty");

            namespace fs = std::filesystem;
            bool const adios = backend == CheckpointBackend::Adios2;
            fs::create_directories(fs::path(finalName).parent_path());

            // File-based iteration layout: <dir>/<prefix>_%06T.<ext>.
            std::string const pattern
                = (fs::path(finalName).parent_path()
                   / (options.prefix + "_%06T." + (adios ? "bp5" : "h5")))
                    .string();
            Series out(
                pattern,
                Access::CREATE,
                backendConfigJson(backend));
            if(options.float64)
            {
                std::cerr << "picBunch: warning: " << kFloat64Warning << '\n';
                out.setAttribute("picBunchPrecisionWarning", std::string(kFloat64Warning));
            }
            out.setIterationEncoding(openPMD::IterationEncoding::fileBased);
            out.setIterationFormat(options.prefix + "_%06T");
            out.setAttribute("meshesPath", std::string("fields/"));
            out.setAttribute("particlesPath", std::string("particles/"));
            out.setAttribute("basePath", std::string("/data/%T/"));
            out.setSoftware("picBunch", "0.1.0");
            out.setAttribute("picongpuIOVersionMajor", std::uint32_t{5});
            out.setAttribute("picongpuIOVersionMinor", std::uint32_t{0});
            // The openPMD high-level API defaults this to 0 (no extension);
            // PIConGPU checkpoints carry 1 (standard openPMD extension).  Match
            // the HDF5 writer's byte-compatible value so the file is read by
            // the same openPMD consumer path.
            out.setAttribute("openPMDextension", std::uint32_t{1});

            Iteration iteration = out.writeIterations()[static_cast<long long>(options.step)];
            iteration.setAttribute("sim_slides", std::uint32_t{0});
            iteration.setAttribute("iteration", static_cast<std::uint64_t>(options.step));
            iteration.setAttribute("dt", options.dt);
            iteration.setAttribute("time", options.time);
            iteration.setAttribute("timeUnitSI", options.timeUnitSI);
            iteration.setAttribute("unit_length", options.unitLength);
            iteration.setAttribute("unit_efield", options.unitEField);
            iteration.setAttribute("unit_bfield", options.unitBField);
            auto setUnitIfKnown = [&](char const* name, double value) {
                if(std::isfinite(value) && value != 0.0)
                    iteration.setAttribute(name, value);
            };
            setUnitIfKnown("unit_time", options.unitTime);
            setUnitIfKnown("unit_speed", options.unitSpeed);
            setUnitIfKnown("unit_energy", options.unitEnergy);
            setUnitIfKnown("unit_mass", options.unitMass);
            setUnitIfKnown("unit_charge", options.unitCharge);

            // Open the input Series (BP5 or HDF5) in read-only mode to
            // enumerate and copy species.  BP5 inputs are read through ADIOS2;
            // HDF5 inputs through the HDF5 backend.  The input extension is
            // auto-detected by openPMD from the path.
            Series in(options.inputFile, Access::READ_ONLY, std::string{});
            long long const inStep = static_cast<long long>(options.inputIteration);

            // Fields group metadata.  The openPMD meshes container maps to the
            // ``meshesPath`` ("fields/") group, so the field-group attributes
            // (identical to the HDF5 path) are set on it.
            auto& meshes = iteration.meshes;
            meshes.setAttribute("fieldSolver", std::string("Yee"));
            meshes.setAttribute("currentSmoothing", std::string("none"));
            meshes.setAttribute("BackgroundFieldIncluded", std::uint8_t{0});

            std::size_t const nx = field.grid.n[0];
            std::size_t const ny = field.grid.n[1];
            std::size_t const nz = field.grid.n[2];
            double const compUnit[6] = {
                options.unitEField, options.unitEField, options.unitEField,
                options.unitBField, options.unitBField, options.unitBField};
            std::array<double, 7> const unitE{1.0, 1.0, -3.0, -1.0, 0.0, 0.0, 0.0};
            std::array<double, 7> const unitB{0.0, 1.0, -2.0, -1.0, 0.0, 0.0, 0.0};
            struct RecordSpec
            {
                char const* name;
                std::size_t first;
                std::array<double, 7> unitDimension;
            };
            RecordSpec const records[] = {
                {"E", 0, unitE},
                {"B", 3, unitB}};
            char const* const componentNames[3] = {"x", "y", "z"};

            // Allocate all six full-shape datasets once, with metadata
            // identical to the HDF5 path.
            for(auto const& spec : records)
            {
                Mesh mesh = iteration.meshes[spec.name];
                std::vector<double> const spacing{
                    field.grid.spacing[2] / options.unitLength,
                    field.grid.spacing[1] / options.unitLength,
                    field.grid.spacing[0] / options.unitLength};
                std::vector<double> const origin{
                    field.grid.origin[2] / options.unitLength,
                    field.grid.origin[1] / options.unitLength,
                    field.grid.origin[0] / options.unitLength};
                // openPMD's default axisLabels for a freshly created mesh is
                // {"x"}; PIConGPU stores the field in z,y,x order, so set the
                // labels explicitly for structural parity.
                mesh.setAxisLabels(std::vector<std::string>{"z", "y", "x"});
                mesh.setGridSpacing(spacing);
                mesh.setGridGlobalOffset(origin);
                mesh.setGridUnitSI(options.unitLength);
                mesh.setUnitDimension(spec.unitDimension);
                for(std::size_t a = 0; a < 3; ++a)
                {
                    MeshRecordComponent rc = mesh[componentNames[a]];
                    std::vector<double> const position{
                        kYeeOffset[spec.first + a][0],
                        kYeeOffset[spec.first + a][1],
                        kYeeOffset[spec.first + a][2]};
                    rc.setPosition(position);
                    rc.setUnitSI(compUnit[spec.first + a]);
                    allocateFull(rc, fieldDatatype(options.float64), nz, ny, nx);
                }
                // PIConGPU checkpoints carry fieldSmoothing on each field mesh;
                // the HDF5 writer sets "none".  openPMD's Mesh default does not
                // emit it, so set it explicitly for structural parity.  Set
                // after the components exist (as with the particle records).
                mesh.setAttribute("fieldSmoothing", std::string("none"));
            }

            // Write the field: one full-shape chunk per component.
            // openPMD's deferred-data contract keeps every stored buffer alive
            // until the flush below.
            std::vector<std::vector<float>> fieldBuffers;
            fieldBuffers.reserve(6);
            for(int c = 0; c < 6; ++c)
            {
                Mesh& mesh = c < 3 ? iteration.meshes["E"] : iteration.meshes["B"];
                MeshRecordComponent rc = mesh[componentNames[c % 3]];
                std::vector<float> buf;
                buf.reserve(field.data[c].size());
                for(double const v : field.data[c])
                    buf.push_back(static_cast<float>(v / compUnit[c]));
                rc.storeChunkRaw(
                    buf.data(),
                    Offset{0u, 0u, 0u},
                    Extent{static_cast<std::uint64_t>(nz),
                           static_cast<std::uint64_t>(ny),
                           static_cast<std::uint64_t>(nx)});
                fieldBuffers.push_back(std::move(buf));
            }
            out.flush();

            // Enumerate the input species before the id provider so the
            // generated id range starts after every copied particle.
            std::vector<std::string> inSpeciesNames;
            auto inIteration = in.snapshots()[inStep];
            for(auto const& kv : inIteration.particles)
                inSpeciesNames.push_back(kv.first);
            std::set<std::string> generatedInputs;
            for(auto const& generated : options.generatedSpecies)
            {
                std::string inputName = generated.empty
                    ? std::string{}
                    : generated.inputSpecies.empty()
                    ? generated.species
                    : generated.inputSpecies;
                if(!inputName.empty())
                    generatedInputs.insert(inputName);
            }
            std::uint64_t copiedParticles = 0;
            for(auto const& name : inSpeciesNames)
            {
                bool isGenerated = generatedInputs.count(name) != 0u;
                for(auto const& g : options.generatedSpecies)
                    if(g.species == name)
                        isGenerated = true;
                if(isGenerated)
                    continue;
                Extent const ext
                    = inIteration.particles[name]["position"]["x"].getExtent();
                std::uint64_t count = 1u;
                for(auto const length : ext)
                    count *= length;
                copiedParticles += count;
            }
            std::uint64_t generatedParticles = 0;
            for(auto const& generated : options.generatedSpecies)
                generatedParticles += generated.empty
                    ? std::uint64_t{0}
                    : static_cast<std::uint64_t>(generated.bunch.size());
            // picongpu_idProvider: one global id range for the whole
            // checkpoint.  nextId covers copied and generated species together.
            std::uint64_t const nextId
                = std::max(options.nextId, copiedParticles + generatedParticles);
            {
                // PIConGPU reads the id provider as
                // ``meshes["picongpu_idProvider"]["startId" | "nextId"]``
                // (NDScalars.hpp), so the record components are named startId /
                // nextId, not scalar.
                Mesh idp = iteration.meshes["picongpu_idProvider"];
                idp.setGridSpacing(std::vector<double>{1.0});
                idp.setGridGlobalOffset(std::vector<double>{0.0});
                idp.setGridUnitSI(1.0);
                idp.setUnitDimension(std::array<double, 7>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
                auto const zeroPos = std::vector<double>{0.0};
                MeshRecordComponent start = idp["startId"];
                // idProvider is a 3D (1,1,1) scalar mesh for single-rank 3D,
                // read by PIConGPU via NDScalars at the local offset.
                allocateFull(start, Datatype::ULONGLONG, 1, 1, 1);
                start.setPosition(zeroPos);
                start.setUnitSI(1.0);
                std::uint64_t const startId = 0;
                start.storeChunkRaw(&startId, Offset{0u, 0u, 0u}, Extent{1u, 1u, 1u});
                start.setAttribute("maxNumProc", std::uint64_t{1});
                MeshRecordComponent next = idp["nextId"];
                allocateFull(next, Datatype::ULONGLONG, 1, 1, 1);
                next.setPosition(zeroPos);
                next.setUnitSI(1.0);
                next.storeChunkRaw(&nextId, Offset{0u, 0u, 0u}, Extent{1u, 1u, 1u});
                out.flush();
            }

            // Particles: copy the non-generated input species, then write the
            // generated/empty ones.
            for(auto const& name : inSpeciesNames)
            {
                bool isGeneratedInput = generatedInputs.count(name) != 0u;
                bool isGeneratedOut
                    = false;
                for(auto const& g : options.generatedSpecies)
                    if(g.species == name)
                        isGeneratedOut = true;
                if(isGeneratedOut)
                    continue; // written below
                if(isGeneratedInput)
                    continue; // consumed by a generated entry; not copied
                copyInputSpecies(
                    in, inStep, iteration, name, options.float64,
                    [&out]() { out.flush(); });
            }
            std::uint64_t idBase = copiedParticles;
            for(auto const& generated : options.generatedSpecies)
            {
                writeParticleSpecies(
                    iteration, generated.species, generated,
                    options.float64, idBase, [&out]() { out.flush(); });
                idBase += generated.empty
                    ? std::uint64_t{0}
                    : static_cast<std::uint64_t>(generated.bunch.size());
            }

            out.flush();
            in.close();
            out.close();
        }

    } // namespace

    void appendCheckpointStep(std::filesystem::path const& directory, std::uint64_t step)
    {
        namespace fs = std::filesystem;
        fs::path const path = directory / "checkpoints.txt";
        std::string const stepText = std::to_string(step);
        bool present = false;
        {
            std::ifstream input(path);
            std::string line;
            while(std::getline(input, line))
                if(line == stepText)
                    present = true;
        }
        if(!present)
        {
            std::ofstream output(path, std::ios::app);
            if(!output)
                throw std::runtime_error(
                    "PicOnGPU restart writer: cannot write '" + path.string() + "'");
            output << stepText << '\n';
            if(!output)
                throw std::runtime_error(
                    "PicOnGPU restart writer: failed writing '" + path.string() + "'");
        }
    }

    void writePicOnGPUCheckpointOpenPMD(
        Field const& field,
        PicOnGPUWriteOptions const& options)
    {
        namespace fs = std::filesystem;
        CheckpointBackend backend = options.backend;
        // Resolve the backend from the value and the destination directory,
        // then build the final output path.  The writer creates the file
        // directly (openPMD handles the temp-file + rename semantics per
        // backend), so there is no separate temp path.
        std::string finalName = resolveOutputPath(options, options.directory, backend);
        publishPicOnGPUOpenPMD(field, options, finalName, backend);
        appendCheckpointStep(options.directory, options.step);
    }

} // namespace picBunch
