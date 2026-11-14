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
// shape, matching a real PIConGPU checkpoint.  Multi-rank runs open one
// collective Series on MPI_COMM_WORLD: every rank creates the same datasets,
// then each rank stores only its own z-slab at its global offset, so no rank
// allocates or gathers the full domain.  The particle records are replicated
// and written by rank 0.  This is the layout PIConGPU's restart reader expects.

#include "picBunch/Compute.hpp"
#include "picBunch/Mpi.hpp"
#include "picBunch/OpenPMDWriter.hpp"
#include "picBunch/Types.hpp"

#include <openPMD/Datatype.hpp>
#include <openPMD/Series.hpp>
#include <openPMD/config.hpp>

#if defined(PICBUNCH_ENABLE_MPI)
#    include <mpi.h>
#endif

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
    // Shared by the CLI and the configured checkpoint path.
    char const* const kFloat64Warning
        = "precision=float64: stock PIConGPU cannot load double particle records; "
          "the restart fails with 'Type conversion during chunk loading not yet implemented!'. "
          "Use precision=float32 for a stock single-precision PIConGPU.";
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
        //! build default (empty config).  ``verify_homogeneous_extents`` is
        //! relaxed for the input read so a particle species whose record
        //! extents are not all present (or differ) does not abort the whole
        //! checkpoint write before the PML slabs are emitted.
        std::string backendConfigJson(CheckpointBackend backend)
        {
            return backend == CheckpointBackend::Adios2
                ? R"({"backend": "adios2", "verify_homogeneous_extents": false})"
                : R"({"verify_homogeneous_extents": false})";
        }

        Datatype fieldDatatype(bool float64)
        {
            return float64 ? Datatype::DOUBLE : Datatype::FLOAT;
        }

        //! Element count of a one- or three-dimensional dataset extent.
        std::size_t extentCount(Extent const& extent)
        {
            std::size_t count = 1u;
            for(auto const length : extent)
                count *= static_cast<std::size_t>(length);
            return count;
        }

        //! Copy a buffer into an openPMD-owned chunk.
        //!
        //! storeChunkRaw keeps only the pointer: with ADIOS2 the write can be
        //! deferred past the local flush, so a stack buffer is not safe.  The
        //! shared pointer keeps the data alive until the backend releases it.
        template<typename T>
        std::shared_ptr<T[]> ownedChunk(T const* values, std::size_t count)
        {
            auto buffer = std::shared_ptr<T[]>(new T[count == 0u ? 1u : count]);
            if(count != 0u)
                std::copy(values, values + count, buffer.get());
            return buffer;
        }

        //! True when this process opens the checkpoint as one collective series.
        bool collectiveSeries()
        {
#if defined(PICBUNCH_ENABLE_MPI) && openPMD_HAVE_MPI
            return mpiActive();
#else
            return false;
#endif
        }

        //! A multi-rank run without an MPI-capable openPMD must not fall back to
        //! one file per rank: fail before the first rank writes anything.
        void requireCollectiveSupport(int size)
        {
            if(size > 1 && !collectiveSeries())
                throw std::runtime_error(
                    "openPMD checkpoint writer: this openPMD build has no MPI support, "
                    "so " + std::to_string(size)
                    + " MPI ranks cannot write one collective checkpoint. Rebuild "
                    "openPMD-api with -DopenPMD_USE_MPI=ON and -DopenPMD_USE_ADIOS2=ON.");
        }

        //! Combine every rank's local writer error into one identical message.
        //!
        //! Every rank must call this at the same collective point.  The first
        //! failing rank's text is broadcast so all ranks throw the same reason.
        std::string collectiveCheckpointError(std::string localError)
        {
#if defined(PICBUNCH_ENABLE_MPI) && openPMD_HAVE_MPI
            if(mpiActive() && mpiSize() > 1)
            {
                int const localFailed = localError.empty() ? 0 : 1;
                int anyFailed = 0;
                MPI_Allreduce(&localFailed, &anyFailed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
                if(anyFailed == 0)
                    return {};
                int const candidate = localFailed
                    ? mpiRank()
                    : std::numeric_limits<int>::max();
                int firstFailed = 0;
                MPI_Allreduce(&candidate, &firstFailed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
                int length = mpiRank() == firstFailed
                    ? static_cast<int>(localError.size())
                    : 0;
                MPI_Bcast(&length, 1, MPI_INT, firstFailed, MPI_COMM_WORLD);
                std::string message(static_cast<std::size_t>(length), '\0');
                if(mpiRank() == firstFailed)
                    message = localError;
                MPI_Bcast(message.data(), length, MPI_CHAR, firstFailed, MPI_COMM_WORLD);
                return message;
            }
#endif
            return localError;
        }

        /*! The z plane range this rank writes.
         *
         * A distributed field already owns ``[slabOffset, slabOffset+slabCount)``.
         * A replicated field is split evenly by rank so every rank still writes
         * only its slice and no rank allocates the full domain.
         */
        std::pair<std::size_t, std::size_t> localZRange(
            Field const& field,
            int rank,
            int size,
            std::size_t nz)
        {
            std::size_t begin = 0u;
            std::size_t end = nz;
            if(field.zSlab)
            {
                begin = field.slabOffset;
                end = field.slabOffset + field.slabCount;
            }
            else if(size > 1)
            {
                begin = (static_cast<std::size_t>(rank) * nz) / static_cast<std::size_t>(size);
                end = ((static_cast<std::size_t>(rank) + 1u) * nz)
                    / static_cast<std::size_t>(size);
            }
            if(begin > end || end > nz)
                throw std::runtime_error(
                    "openPMD checkpoint writer: rank " + std::to_string(rank)
                    + " owns invalid z planes [" + std::to_string(begin) + ", "
                    + std::to_string(end) + ") of " + std::to_string(nz));
            return {begin, end};
        }

        //! Check that the rank slabs tile ``[0, nz)`` exactly once.
        void verifySlabTiling(Field const& field, std::size_t nz)
        {
            int const rank = mpiRank();
            int const size = mpiSize();
            auto const [begin, end] = localZRange(field, rank, size, nz);
#if defined(PICBUNCH_ENABLE_MPI) && openPMD_HAVE_MPI
            if(mpiActive() && size > 1)
            {
                int const localSlab = field.zSlab ? 1 : 0;
                int minSlab = 0;
                int maxSlab = 0;
                MPI_Allreduce(&localSlab, &minSlab, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
                MPI_Allreduce(&localSlab, &maxSlab, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
                if(minSlab != maxSlab)
                    throw std::runtime_error(
                        "openPMD checkpoint writer: the ranks disagree on whether the "
                        "field is distributed (some hold z-slabs, some the full grid)");
                std::uint64_t localRange[2] = {begin, end};
                std::vector<std::uint64_t> ranges(static_cast<std::size_t>(size) * 2u);
                MPI_Allgather(
                    localRange,
                    2,
                    MPI_UINT64_T,
                    ranges.data(),
                    2,
                    MPI_UINT64_T,
                    MPI_COMM_WORLD);
                std::vector<std::pair<std::uint64_t, std::uint64_t>> sorted;
                sorted.reserve(static_cast<std::size_t>(size));
                for(int r = 0; r < size; ++r)
                    sorted.emplace_back(
                        ranges[static_cast<std::size_t>(r) * 2u],
                        ranges[static_cast<std::size_t>(r) * 2u + 1u]);
                std::sort(sorted.begin(), sorted.end());
                std::uint64_t covered = 0u;
                for(auto const& [z0, z1] : sorted)
                {
                    if(z0 != covered)
                        throw std::runtime_error(
                            "openPMD checkpoint writer: the rank z-slabs have a gap or "
                            "overlap at plane " + std::to_string(z0));
                    covered = z1;
                }
                if(covered != static_cast<std::uint64_t>(nz))
                    throw std::runtime_error(
                        "openPMD checkpoint writer: the rank z-slabs cover "
                        + std::to_string(covered) + " of " + std::to_string(nz)
                        + " planes");
                return;
            }
#endif
            if(begin != 0u || end != nz)
                throw std::runtime_error(
                    "openPMD checkpoint writer: a single-rank slab does not cover the "
                    "full grid");
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
            if(length == 0u)
            {
                // resetDataset() with a zero extent makes the HDF5 backend
                // emit an empty group.  makeEmpty() emits the standard openPMD
                // constant/empty record representation (shape = [0]).
                rc.makeEmpty(type, 1u);
                return rc;
            }
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
            bool writeData,
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
            // Store particles patch by patch: every patch owns a contiguous
            // slot range, so each output record is a concatenation of the
            // owned particles in patch order.
            std::vector<std::size_t> order;
            order.reserve(count);
            for(auto const& patchParticles : owned)
                order.insert(order.end(), patchParticles.begin(), patchParticles.end());

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
            // Store a floating-point record in the configured precision.  The
            // dataset was allocated with fieldDatatype(float64), so a float
            // buffer must not be handed to a DOUBLE component.
            auto storeFloating = [&](RecordComponent& rc, std::vector<double> const& values) {
                // The converted buffer must outlive flushAfterStore().  Only the
                // checkpoint owner writes the replicated particle records; every
                // rank still participates in the collective flush.
                if(writeData && !values.empty())
                {
                    if(float64)
                    {
                        rc.storeChunk(
                            ownedChunk(values.data(), values.size()),
                            Offset{0u},
                            Extent{values.size()});
                    }
                    else
                    {
                        std::vector<float> conv(values.size());
                        for(std::size_t i = 0; i < values.size(); ++i)
                            conv[i] = static_cast<float>(values[i]);
                        rc.storeChunk(
                            ownedChunk(conv.data(), conv.size()),
                            Offset{0u},
                            Extent{conv.size()});
                    }
                }
                flushAfterStore();
            };
            if(count > 0)
            {
            for(char const* comp : xyz)
            {
                int const axis = axisOf(comp);
                RecordComponent posRc = species["position"][comp];
                std::vector<double> data;
                data.reserve(count);
                for(std::size_t slot = 0; slot < count; ++slot)
                {
                    std::size_t const p = order[slot];
                    // PIConGPU stores position as the sub-cell offset
                    // (cells - cellIndex), not the absolute cell index.
                    double const cells = (positionOf(bunch, comp)[p]
                                          - generated.domainOrigin[axis])
                                         / generated.cellSize[axis];
                    data.push_back(cells - std::floor(cells + 1e-9));
                }
                storeFloating(posRc, data);
            }
            for(char const* comp : xyz)
            {
                int const axis = axisOf(comp);
                RecordComponent offRc = species["positionOffset"][comp];
                std::vector<std::int32_t> data(count, 0);
                for(std::size_t slot = 0; slot < count; ++slot)
                {
                    std::size_t const p = order[slot];
                    data[slot] = static_cast<std::int32_t>(
                        std::floor((positionOf(bunch, comp)[p]
                                    - generated.domainOrigin[axis])
                                    / generated.cellSize[axis] + 1e-9));
                }
                if(writeData)
                    offRc.storeChunk(ownedChunk(data.data(), data.size()), Offset{0u}, Extent{count});
                flushAfterStore();
            }
            for(char const* comp : xyz)
            {
                RecordComponent momRc = species["momentum"][comp];
                std::vector<double> data;
                data.reserve(count);
                for(std::size_t slot = 0; slot < count; ++slot)
                {
                    std::size_t const p = order[slot];
                    double const restMass = !bunch.restMass.empty() && bunch.restMass[p] > 0.0
                        ? bunch.restMass[p]
                        : generated.fallbackMass;
                    // PIConGPU openPMD momentum is the macro-particle momentum
                    // in units of unitMassSpeed: u*m*c*weight/unitMassSpeed.
                    data.push_back(
                        momentumOf(bunch, comp)[p] * restMass * bunch.weight[p]
                        * picBunch::kSpeedOfLight / generated.unitMassSpeed);
                }
                storeFloating(momRc, data);
            }
            // weighting (scalar record)
            {
                RecordComponent wRc = species["weighting"][RecordComponent::SCALAR];
                std::vector<double> data(count);
                for(std::size_t slot = 0; slot < count; ++slot)
                    data[slot] = bunch.weight[order[slot]];
                storeFloating(wRc, data);
            }
            // id: global particle ids, disjoint across generated species and
            // spanning [idBase, idBase+count).  Scalar ULONGLONG record, as in
            // the HDF5 writer.
            {
                RecordComponent idRc = species["id"][RecordComponent::SCALAR];
                std::vector<std::uint64_t> ids(count);
                for(std::size_t i = 0; i < count; ++i)
                    ids[i] = idBase + static_cast<std::uint64_t>(i);
                if(writeData)
                    idRc.storeChunk(ownedChunk(ids.data(), ids.size()), Offset{0u}, Extent{count});
                flushAfterStore();
            }
            // charge / mass: self-contained restart (values in internal units,
            // unitSI carries the TOML factor), as in the HDF5 writer.
            {
                RecordComponent chRc = species["charge"][RecordComponent::SCALAR];
                std::vector<double> data(count);
                for(std::size_t slot = 0; slot < count; ++slot)
                    data[slot] = bunch.charge[order[slot]] / generated.unitCharge;
                storeFloating(chRc, data);
            }
            {
                RecordComponent maRc = species["mass"][RecordComponent::SCALAR];
                std::vector<double> data(count);
                for(std::size_t slot = 0; slot < count; ++slot)
                {
                    std::size_t const p = order[slot];
                    double const restMass = !bunch.restMass.empty() && bunch.restMass[p] > 0.0
                        ? bunch.restMass[p]
                        : generated.fallbackMass;
                    data[slot] = restMass / generated.unitMass;
                }
                storeFloating(maRc, data);
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
                if(writeData)
                    numPr.storeChunk(ownedChunk(numData.data(), numData.size()), Offset{0u}, Extent{nPatches});
                flushAfterStore();
                PatchRecordComponent numOffRc
                    = pp["numParticlesOffset"][RecordComponent::SCALAR];
                allocate1D(numOffRc, Datatype::ULONGLONG, nPatches);
                numOffRc.setUnitSI(1.0);
                std::vector<std::uint64_t> offData(patches.size());
                for(std::size_t q = 0; q < patches.size(); ++q)
                    offData[q] = patchOffset[q];
                if(writeData)
                    numOffRc.storeChunk(
                        ownedChunk(offData.data(), offData.size()),
                        Offset{0u},
                        Extent{nPatches});
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
                    if(writeData)
                        offRc.storeChunk(ownedChunk(od.data(), od.size()), Offset{0u}, Extent{nPatches});
                    flushAfterStore();
                    PatchRecordComponent extRc = pp["extent"][comp];
                    allocate1D(extRc, Datatype::ULONGLONG, nPatches);
                    extRc.setUnitSI(1.0);
                    std::vector<std::uint64_t> ed(patches.size());
                    for(std::size_t q = 0; q < patches.size(); ++q)
                        ed[q] = static_cast<std::uint64_t>(patches[q].extent[axis]);
                    if(writeData)
                        extRc.storeChunk(ownedChunk(ed.data(), ed.size()), Offset{0u}, Extent{nPatches});
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
            std::uint64_t& count,
            bool writeData)
        {
            // The particle record component is 1-D.  ``getExtent()`` returns
            // only the rank it was written with, so use the product of the
            // extents and an equally sized zero offset instead of assuming a
            // padded 3-D extent.
            Extent const ext = inRc.getExtent();
            count = extentCount(ext);
            // Only the checkpoint owner loads the replicated particle records.
            // Every rank still calls in.flush() once so a collective read flush
            // is not skipped.
            if(!writeData || count == 0u)
            {
                in.flush();
                return {};
            }
            std::vector<double> buf(count);
            Offset const offset(ext.size(), 0u);
            // loadChunkRaw requires the caller buffer to match the on-disk
            // datatype exactly; the source precision may differ from the
            // requested output precision, so convert after loading.
            // The raw buffer must stay alive until ``in.flush()`` executes the
            // queued read; flush before the switch-local buffer is destroyed.
            switch(inRc.getDatatype())
            {
            case Datatype::FLOAT:
            {
                std::vector<float> raw(count);
                inRc.loadChunkRaw(raw.data(), offset, ext);
                in.flush();
                for(std::size_t i = 0; i < count; ++i)
                    buf[i] = static_cast<double>(raw[i]);
                break;
            }
            case Datatype::DOUBLE:
                inRc.loadChunkRaw(buf.data(), offset, ext);
                in.flush();
                break;
            case Datatype::INT:
            {
                std::vector<std::int32_t> raw(count);
                inRc.loadChunkRaw(raw.data(), offset, ext);
                in.flush();
                for(std::size_t i = 0; i < count; ++i)
                    buf[i] = static_cast<double>(raw[i]);
                break;
            }
            case Datatype::UINT:
            {
                std::vector<std::uint32_t> raw(count);
                inRc.loadChunkRaw(raw.data(), offset, ext);
                in.flush();
                for(std::size_t i = 0; i < count; ++i)
                    buf[i] = static_cast<double>(raw[i]);
                break;
            }
            case Datatype::LONGLONG:
            {
                std::vector<std::int64_t> raw(count);
                inRc.loadChunkRaw(raw.data(), offset, ext);
                in.flush();
                for(std::size_t i = 0; i < count; ++i)
                    buf[i] = static_cast<double>(raw[i]);
                break;
            }
            case Datatype::ULONGLONG:
            {
                std::vector<std::uint64_t> raw(count);
                inRc.loadChunkRaw(raw.data(), offset, ext);
                in.flush();
                for(std::size_t i = 0; i < count; ++i)
                    buf[i] = static_cast<double>(raw[i]);
                break;
            }
            default:
                throw std::runtime_error(
                    "openPMD checkpoint writer: unsupported particle record datatype");
            }
            return buf;
        }

        //! Copy user attributes that are not structural openPMD metadata.
        //! Unknown or read-only keys are skipped: the goal is to preserve
        //! custom species/mesh attributes (for example PIConGPU's
        //! ``sourceMarker``), not to overwrite the writer's own metadata.
        void copyAttributes(
            openPMD::Attributable const& src,
            openPMD::Attributable& dst)
        {
            static std::set<std::string> const skip{
                "position",
                "unitSI",
                "unitDimension",
                "macroWeighted",
                "weightingPower",
                "timeOffset",
                "axisLabels",
                "dataOrder",
                "geometry",
                "gridSpacing",
                "gridGlobalOffset",
                "gridUnitSI",
                "fieldSmoothing"};
            for(auto const& key : src.attributes())
            {
                if(skip.count(key) != 0u)
                    continue;
                auto const attribute = src.getAttribute(key);
                try
                {
                    switch(attribute.dtype)
                    {
                    case Datatype::CHAR:
                        dst.setAttribute(key, attribute.get<char>());
                        break;
                    case Datatype::UCHAR:
                        dst.setAttribute(key, attribute.get<unsigned char>());
                        break;
                    case Datatype::SCHAR:
                        dst.setAttribute(key, attribute.get<signed char>());
                        break;
                    case Datatype::SHORT:
                        dst.setAttribute(key, attribute.get<short>());
                        break;
                    case Datatype::INT:
                        dst.setAttribute(key, attribute.get<int>());
                        break;
                    case Datatype::LONG:
                        dst.setAttribute(key, attribute.get<long>());
                        break;
                    case Datatype::LONGLONG:
                        dst.setAttribute(key, attribute.get<long long>());
                        break;
                    case Datatype::USHORT:
                        dst.setAttribute(key, attribute.get<unsigned short>());
                        break;
                    case Datatype::UINT:
                        dst.setAttribute(key, attribute.get<unsigned int>());
                        break;
                    case Datatype::ULONG:
                        dst.setAttribute(key, attribute.get<unsigned long>());
                        break;
                    case Datatype::ULONGLONG:
                        dst.setAttribute(key, attribute.get<unsigned long long>());
                        break;
                    case Datatype::FLOAT:
                        dst.setAttribute(key, attribute.get<float>());
                        break;
                    case Datatype::DOUBLE:
                        dst.setAttribute(key, attribute.get<double>());
                        break;
                    case Datatype::STRING:
                        dst.setAttribute(key, attribute.get<std::string>());
                        break;
                    case Datatype::VEC_FLOAT:
                        dst.setAttribute(key, attribute.get<std::vector<float>>());
                        break;
                    case Datatype::VEC_DOUBLE:
                        dst.setAttribute(key, attribute.get<std::vector<double>>());
                        break;
                    case Datatype::VEC_INT:
                        dst.setAttribute(key, attribute.get<std::vector<int>>());
                        break;
                    case Datatype::VEC_UINT:
                        dst.setAttribute(key, attribute.get<std::vector<unsigned int>>());
                        break;
                    case Datatype::VEC_ULONGLONG:
                        dst.setAttribute(
                            key, attribute.get<std::vector<unsigned long long>>());
                        break;
                    case Datatype::VEC_STRING:
                        dst.setAttribute(key, attribute.get<std::vector<std::string>>());
                        break;
                    case Datatype::BOOL:
                        dst.setAttribute(key, attribute.get<bool>());
                        break;
                    default:
                        break;
                    }
                }
                catch(std::exception const&)
                {
                    // A reserved attribute that openPMD refuses to set is
                    // already covered by the explicit metadata above.
                }
            }
        }

        //! Copy one record component byte-for-byte at its on-disk datatype.
        template<typename T_Component>
        void copyRecordComponent(
            Series& in,
            Series& out,
            T_Component& src,
            T_Component& dst,
            bool writeData)
        {
            Extent const ext = src.getExtent();
            std::uint64_t const count = extentCount(ext);
            Offset const offset(ext.size(), 0u);
            Datatype const datatype = src.getDatatype();
            dst.resetDataset(Dataset(datatype, ext));
            auto copy = [&](auto value) {
                using T = decltype(value);
                if(writeData)
                {
                    std::vector<T> buffer(static_cast<std::size_t>(count));
                    src.loadChunkRaw(buffer.data(), offset, ext);
                    in.flush();
                    dst.storeChunk(ownedChunk(buffer.data(), buffer.size()), offset, ext);
                }
                else
                {
                    in.flush();
                }
                out.flush();
            };
            switch(datatype)
            {
            case Datatype::FLOAT:
                copy(static_cast<float>(0));
                break;
            case Datatype::DOUBLE:
                copy(static_cast<double>(0));
                break;
            case Datatype::CHAR:
                copy(static_cast<char>(0));
                break;
            case Datatype::UCHAR:
                copy(static_cast<unsigned char>(0));
                break;
            case Datatype::SCHAR:
                copy(static_cast<signed char>(0));
                break;
            case Datatype::SHORT:
                copy(static_cast<short>(0));
                break;
            case Datatype::USHORT:
                copy(static_cast<unsigned short>(0));
                break;
            case Datatype::INT:
                copy(static_cast<std::int32_t>(0));
                break;
            case Datatype::UINT:
                copy(static_cast<std::uint32_t>(0));
                break;
            case Datatype::LONG:
                copy(static_cast<long>(0));
                break;
            case Datatype::ULONG:
                copy(static_cast<unsigned long>(0));
                break;
            case Datatype::LONGLONG:
                copy(static_cast<std::int64_t>(0));
                break;
            case Datatype::ULONGLONG:
                copy(static_cast<std::uint64_t>(0));
                break;
            default:
                throw std::runtime_error(
                    "openPMD checkpoint writer: unsupported copied record datatype");
            }
            copyAttributes(src, dst);
        }

        template<typename T_Record>
        void copyRecord(
            Series& in,
            Series& out,
            T_Record& src,
            T_Record& dst,
            bool writeData)
        {
            copyAttributes(src, dst);
            for(auto const& entry : src)
            {
                std::string const& key = entry.first;
                auto srcComponent = src[key];
                auto dstComponent = dst[key];
                copyRecordComponent(in, out, srcComponent, dstComponent, writeData);
            }
        }

        //! Copy one PML record component.  A field-shaped record is written as
        //! one z-slab per rank; any other shape falls back to a root-only copy.
        template<typename T_Component>
        void copyMeshComponent(
            Series& in,
            Series& out,
            T_Component& src,
            T_Component& dst,
            Idx3 const& globalN,
            bool collective)
        {
            Extent const ext = src.getExtent();
            std::uint64_t const count = extentCount(ext);
            Datatype const datatype = src.getDatatype();
            dst.resetDataset(Dataset(datatype, ext));
            bool const fieldShaped = ext.size() == 3u
                && ext[0] == static_cast<Extent::value_type>(globalN[2])
                && ext[1] == static_cast<Extent::value_type>(globalN[1])
                && ext[2] == static_cast<Extent::value_type>(globalN[0]);
            int const rank = mpiRank();
            int const size = mpiSize();
            auto copy = [&](auto value) {
                using T = decltype(value);
                if(fieldShaped && collective)
                {
                    // The input PML record covers the same global domain, so
                    // split it exactly like the main field: one z-slab per rank.
                    std::size_t const begin
                        = (static_cast<std::size_t>(rank) * globalN[2])
                        / static_cast<std::size_t>(size);
                    std::size_t const end
                        = ((static_cast<std::size_t>(rank) + 1u) * globalN[2])
                        / static_cast<std::size_t>(size);
                    std::size_t const planes = end - begin;
                    std::size_t const localCount
                        = planes * globalN[0] * globalN[1];
                    if(localCount != 0u)
                    {
                        auto buffer = std::shared_ptr<T[]>(new T[localCount]);
                        Offset const offset{
                            static_cast<Offset::value_type>(begin), 0u, 0u};
                        Extent const localExtent{
                            static_cast<Extent::value_type>(planes),
                            static_cast<Extent::value_type>(globalN[1]),
                            static_cast<Extent::value_type>(globalN[0])};
                        src.loadChunkRaw(buffer.get(), offset, localExtent);
                        in.flush();
                        dst.storeChunk(buffer, offset, localExtent);
                    }
                    else
                        in.flush();
                }
                else if(!collective || rank == 0)
                {
                    std::vector<T> buffer(static_cast<std::size_t>(count));
                    Offset const offset(ext.size(), 0u);
                    src.loadChunkRaw(buffer.data(), offset, ext);
                    in.flush();
                    dst.storeChunk(ownedChunk(buffer.data(), buffer.size()), offset, ext);
                }
                else
                    in.flush();
                out.flush();
            };
            switch(datatype)
            {
            case Datatype::FLOAT:
                copy(static_cast<float>(0));
                break;
            case Datatype::DOUBLE:
                copy(static_cast<double>(0));
                break;
            case Datatype::CHAR:
                copy(static_cast<char>(0));
                break;
            case Datatype::UCHAR:
                copy(static_cast<unsigned char>(0));
                break;
            case Datatype::SCHAR:
                copy(static_cast<signed char>(0));
                break;
            case Datatype::SHORT:
                copy(static_cast<short>(0));
                break;
            case Datatype::USHORT:
                copy(static_cast<unsigned short>(0));
                break;
            case Datatype::INT:
                copy(static_cast<std::int32_t>(0));
                break;
            case Datatype::UINT:
                copy(static_cast<std::uint32_t>(0));
                break;
            case Datatype::LONG:
                copy(static_cast<long>(0));
                break;
            case Datatype::ULONG:
                copy(static_cast<unsigned long>(0));
                break;
            case Datatype::LONGLONG:
                copy(static_cast<std::int64_t>(0));
                break;
            case Datatype::ULONGLONG:
                copy(static_cast<std::uint64_t>(0));
                break;
            default:
                throw std::runtime_error(
                    "openPMD checkpoint writer: unsupported copied record datatype");
            }
            copyAttributes(src, dst);
        }

        //! Copy one input mesh (a PML slab) including its component metadata.
        void copyMesh(
            Series& in,
            Series& out,
            Mesh& src,
            Mesh& dst,
            Idx3 const& globalN,
            bool collective)
        {
            dst.setGeometry(src.geometry());
            dst.setDataOrder(src.dataOrder());
            dst.setAxisLabels(src.axisLabels());
            dst.setGridSpacing(src.gridSpacing<double>());
            dst.setGridGlobalOffset(src.gridGlobalOffset());
            dst.setGridUnitSI(src.gridUnitSI());
            dst.setUnitDimension(src.unitDimension());
            copyAttributes(src, dst);
            for(auto const& entry : src)
            {
                std::string const& key = entry.first;
                MeshRecordComponent srcComponent = src[key];
                MeshRecordComponent dstComponent = dst[key];
                std::vector<double> const sourcePosition
                    = srcComponent.position<double>();
                double const sourceUnitSI = srcComponent.unitSI();
                copyMeshComponent(in, out, srcComponent, dstComponent, globalN, collective);
                dstComponent.setUnitSI(sourceUnitSI);
                // The first component flush installs openPMD's default scalar
                // ``position`` attribute (one-element dataspace).  Remove it
                // and recreate it from the source vector so the PML slab keeps
                // its three-component position metadata.
                dstComponent.deleteAttribute("position");
                dstComponent.setPosition(sourcePosition);
                out.flush();
            }
        }

        void copyInputSpecies(
            Series& in,
            long long inStep,
            Series& out,
            Iteration& outIteration,
            std::string const& species,
            bool float64,
            bool writeData,
            std::function<void()> flushAfterStore)
        {
            ParticleSpecies inSpecies = in.snapshots()[inStep].particles[species];
            ParticleSpecies outSpecies = outIteration.particles[species];
            copyAttributes(inSpecies, outSpecies);
            char const* const xyz[3] = {"x", "y", "z"};
            std::uint64_t count = 0;
            auto storeFloating = [&](RecordComponent& rc, std::vector<double> const& values) {
                // Every rank creates the record and calls the collective flush;
                // only the checkpoint owner stores the replicated particle data.
                if(writeData && !values.empty())
                {
                    std::vector<float> conv;
                    if(float64)
                    {
                        rc.storeChunk(
                            ownedChunk(values.data(), values.size()),
                            Offset{0u},
                            Extent{values.size()});
                    }
                    else
                    {
                        conv.reserve(values.size());
                        for(double const v : values)
                            conv.push_back(static_cast<float>(v));
                        rc.storeChunk(
                            ownedChunk(conv.data(), conv.size()),
                            Offset{0u},
                            Extent{conv.size()});
                    }
                }
                flushAfterStore();
            };
            for(char const* comp : xyz)
            {
                RecordComponent inRc = inSpecies["position"][comp];
                std::vector<double> const buf
                    = loadRecordComponent(in, inRc, count, writeData);
                RecordComponent outRc = outSpecies["position"][comp];
                allocate1D(outRc, fieldDatatype(float64), count);
                double const unitSI = inRc.getAttribute("unitSI").get<double>();
                outRc.setUnitSI(unitSI);
                outRc.setAttribute("position", std::vector<double>{0.0, 0.0, 0.0});
                storeFloating(outRc, buf);
            }
            // Momentum is part of a restart checkpoint; the legacy HDF5 writer
            // rewrote it to the configured precision.  Keep the input unitSI.
            for(char const* comp : xyz)
            {
                RecordComponent inRc = inSpecies["momentum"][comp];
                std::uint64_t c = 0;
                std::vector<double> const buf
                    = loadRecordComponent(in, inRc, c, writeData);
                RecordComponent outRc = outSpecies["momentum"][comp];
                allocate1D(outRc, fieldDatatype(float64), c);
                outRc.setUnitSI(inRc.getAttribute("unitSI").get<double>());
                storeFloating(outRc, buf);
            }
            for(char const* comp : xyz)
            {
                RecordComponent inRc = inSpecies["positionOffset"][comp];
                std::uint64_t c = 0;
                std::vector<double> const buf
                    = loadRecordComponent(in, inRc, c, writeData);
                RecordComponent outRc = outSpecies["positionOffset"][comp];
                allocate1D(outRc, Datatype::INT, c);
                outRc.setUnitSI(1.0);
                if(writeData && c != 0u)
                {
                    std::vector<std::int32_t> conv(c);
                    for(std::uint64_t i = 0; i < c; ++i)
                        conv[i] = static_cast<std::int32_t>(buf[i]);
                    outRc.storeChunk(ownedChunk(conv.data(), conv.size()), Offset{0u}, Extent{c});
                }
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
                std::uint64_t c = 0;
                std::vector<double> const buf
                    = loadRecordComponent(in, inRc, c, writeData);
                RecordComponent outRc
                    = outSpecies["weighting"][RecordComponent::SCALAR];
                allocate1D(outRc, fieldDatatype(float64), c);
                outRc.setUnitSI(1.0);
                storeFloating(outRc, buf);
            }
            // particlePatches is part of the restart contract; copy every
            // record and component of the input species verbatim.
            ParticlePatches inPatches = inSpecies.particlePatches;
            if(inPatches.numPatches() != 0u)
            {
                ParticlePatches outPatches = outSpecies.particlePatches;
                for(auto const& entry : inPatches)
                {
                    std::string const& key = entry.first;
                    PatchRecord srcRecord = inPatches[key];
                    PatchRecord dstRecord = outPatches[key];
                    copyRecord(in, out, srcRecord, dstRecord, writeData);
                }
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
            int const rank = mpiRank();
            int const size = mpiSize();
            bool const collective = collectiveSeries();
            requireCollectiveSupport(size);
            fs::create_directories(fs::path(finalName).parent_path());

            // Open the input Series before creating the output so a missing
            // required species fails without leaving a checkpoint behind.  A
            // multi-rank run opens both Series collectively on MPI_COMM_WORLD;
            // openPMD then merges the per-rank slabs into one output file.
            std::unique_ptr<Series> inputSeries;
#if defined(PICBUNCH_ENABLE_MPI) && openPMD_HAVE_MPI
            if(collective)
                inputSeries = std::make_unique<Series>(
                    options.inputFile,
                    Access::READ_ONLY,
                    MPI_COMM_WORLD,
                    std::string{});
            else
#endif
                inputSeries = std::make_unique<Series>(
                    options.inputFile, Access::READ_ONLY, std::string{});
            Series& in = *inputSeries;
            long long const inStep = static_cast<long long>(options.inputIteration);
            auto inIteration = in.snapshots()[inStep];
            std::vector<std::string> inSpeciesNames;
            std::set<std::string> inputSpecies;
            for(auto const& kv : inIteration.particles)
            {
                inSpeciesNames.push_back(kv.first);
                inputSpecies.insert(kv.first);
            }
            for(auto const& requested : options.requiredInputSpecies)
            {
                if(inputSpecies.count(requested) == 0u)
                    throw std::runtime_error(
                        "PicOnGPU restart writer: input has no particle species '"
                        + requested + "'");
            }
            for(auto const& generated : options.generatedSpecies)
            {
                if(generated.empty)
                    continue;
                std::string const inputName = generated.inputSpecies.empty()
                    ? generated.species
                    : generated.inputSpecies;
                if(inputSpecies.count(inputName) == 0u)
                    throw std::runtime_error(
                        "PicOnGPU restart writer: generated species '"
                        + generated.species + "' reads input species '" + inputName
                        + "', which is absent from the input iteration");
            }

            // File-based iteration layout: <dir>/<prefix>_%06T.<ext>.
            std::string const pattern
                = (fs::path(finalName).parent_path()
                   / (options.prefix + "_%06T." + (adios ? "bp5" : "h5")))
                    .string();
            std::unique_ptr<Series> outputSeries;
#if defined(PICBUNCH_ENABLE_MPI) && openPMD_HAVE_MPI
            if(collective)
                outputSeries = std::make_unique<Series>(
                    pattern,
                    Access::CREATE,
                    MPI_COMM_WORLD,
                    backendConfigJson(backend));
            else
#endif
                outputSeries = std::make_unique<Series>(
                    pattern,
                    Access::CREATE,
                    backendConfigJson(backend));
            Series& out = *outputSeries;

            // Rank-local data stores must not strand a peer inside the next
            // collective flush.  Record the first message but always execute
            // the following collective calls, then rethrow the first rank's
            // message at a checkpoint.
            std::string localError;
            auto attempt = [&](auto&& function) {
                try
                {
                    function();
                }
                catch(std::exception const& error)
                {
                    if(localError.empty())
                        localError = error.what();
                }
                catch(...)
                {
                    if(localError.empty())
                        localError = "unknown exception in the checkpoint writer";
                }
            };
            auto checkpoint = [&]() {
                std::string const error = collectiveCheckpointError(localError);
                if(error.empty())
                    return;
                try
                {
                    in.close();
                }
                catch(...)
                {
                }
                try
                {
                    out.close();
                }
                catch(...)
                {
                }
                throw std::runtime_error(error);
            };
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

            // Write the field: each rank stores only its own z-slab, at its
            // global offset.  Every rank allocated the same full-shape
            // datasets above; openPMD/ADIOS2 merges the disjoint chunks into
            // one record component of shape (nz, ny, nx).  No rank allocates
            // the full domain.  A small buffer per component is kept alive by
            // storeChunk() until the collective flush below.
            std::size_t zBegin = 0u;
            std::size_t zEnd = nz;
            attempt([&]() {
                verifySlabTiling(field, nz);
                auto const range = localZRange(field, rank, size, nz);
                zBegin = range.first;
                zEnd = range.second;
            });
            checkpoint();
            std::size_t const plane = nx * ny;
            std::size_t const localPlanes = zEnd - zBegin;
            std::size_t const localValues = localPlanes * plane;
            std::size_t const fieldOffset = field.zSlab ? 0u : zBegin * plane;
            openPMD::Offset const fieldChunkOffset{
                static_cast<std::uint64_t>(zBegin), 0u, 0u};
            openPMD::Extent const fieldChunkExtent{
                static_cast<std::uint64_t>(localPlanes),
                static_cast<std::uint64_t>(ny),
                static_cast<std::uint64_t>(nx)};
            attempt([&]() {
                for(int c = 0; c < 6; ++c)
                {
                    if(localValues == 0u)
                        break;
                    Mesh& mesh = c < 3 ? iteration.meshes["E"] : iteration.meshes["B"];
                    MeshRecordComponent rc = mesh[componentNames[c % 3]];
                    double const inverseUnit = 1.0 / compUnit[c];
                    if(options.float64)
                    {
                        auto buf = std::shared_ptr<double[]>(new double[localValues]);
                        for(std::size_t i = 0; i < localValues; ++i)
                            buf[i] = field.data[c][fieldOffset + i] * inverseUnit;
                        rc.storeChunk(buf, fieldChunkOffset, fieldChunkExtent);
                    }
                    else
                    {
                        auto buf = std::shared_ptr<float[]>(new float[localValues]);
                        for(std::size_t i = 0; i < localValues; ++i)
                            buf[i] = static_cast<float>(
                                field.data[c][fieldOffset + i] * inverseUnit);
                        rc.storeChunk(buf, fieldChunkOffset, fieldChunkExtent);
                    }
                }
            });

            // Convolutional PML auxiliary state (absorber = "pml" only).
            //
            // The loop below writes twelve slab records, one per record/slab
            // pair: "Convolutional PML E" and "Convolutional PML B" for the six
            // slab names xneg, xpos, yneg, ypos, zneg, zpos (getPmlSlabName in
            // the openPMD plugin's RestartFieldLoader), each with the six
            // cross components xy, xz, yx, yz, zx, zy (12 * 6 = 72 components
            // in total).  The layout matches the PIConGPU writer: full
            // domain extent (Nz,Ny,Nx) like E and B, axisLabels z,y,x,
            // dataOrder C, geometry cartesian, gridSpacing/gridGlobalOffset/
            // gridUnitSI from the main field, unitDimension of the parent E or
            // B record, per-component unitSI unitEField/unitBField and the
            // parent Yee position duplicated across the split components.
            //
            // picBunch computes a free-space field and has no PML state, so the
            // values are the physically correct initial state of a fresh deck:
            // zero.  PIConGPU resets the slab buffers at startup (Field::reset
            // zeroes both buffers), so a run that checkpointed right after
            // initialization stores zeros in these slabs.  Restoring zeros is
            // therefore the faithful initial condition for a deck whose PML
            // has never run.
            if(options.absorber == "pml")
            {
                char const* const pmlComponents[6] = {"xy", "xz", "yx", "yz", "zx", "zy"};
                char const* const pmlSlabs[6]
                    = {"xneg", "xpos", "yneg", "ypos", "zneg", "zpos"};
                for(char const* record : {"Convolutional PML E", "Convolutional PML B"})
                {
                    bool const isElectric = std::string_view{record} == "Convolutional PML E";
                    double const recordUnit = isElectric ? options.unitEField : options.unitBField;
                    std::size_t const parentFirst = isElectric ? std::size_t{0} : std::size_t{3};
                    std::array<double, 7> const recordUnitDimension
                        = isElectric ? std::array<double, 7>{1.0, 1.0, -3.0, -1.0, 0.0, 0.0, 0.0}
                                     : std::array<double, 7>{0.0, 1.0, -2.0, -1.0, 0.0, 0.0, 0.0};
                    for(char const* slab : pmlSlabs)
                    {
                        Mesh mesh = iteration.meshes[std::string{record} + "_" + slab];
                        mesh.setAxisLabels(std::vector<std::string>{"z", "y", "x"});
                        mesh.setGridSpacing(
                            std::vector<double>{
                                field.grid.spacing[2] / options.unitLength,
                                field.grid.spacing[1] / options.unitLength,
                                field.grid.spacing[0] / options.unitLength});
                        mesh.setGridGlobalOffset(
                            std::vector<double>{
                                field.grid.origin[2] / options.unitLength,
                                field.grid.origin[1] / options.unitLength,
                                field.grid.origin[0] / options.unitLength});
                        mesh.setGridUnitSI(options.unitLength);
                        mesh.setUnitDimension(recordUnitDimension);
                        for(int a = 0; a < 6; ++a)
                        {
                            // Split components inherit the parent Yee position:
                            // x -> (xy, xz), y -> (yx, yz), z -> (zx, zy).
                            MeshRecordComponent rc = mesh[pmlComponents[a]];
                            std::size_t const parentAxis = a / 2;
                            rc.setPosition(
                                std::vector<double>{
                                    kYeeOffset[parentFirst + parentAxis][0],
                                    kYeeOffset[parentFirst + parentAxis][1],
                                    kYeeOffset[parentFirst + parentAxis][2]});
                            rc.setUnitSI(recordUnit);
                            allocateFull(rc, fieldDatatype(options.float64), nz, ny, nx);
                        }
                        mesh.setAttribute("fieldSmoothing", std::string("none"));
                    }
                }
                // All 48 components are zero.  Each rank stores only its own
                // z-slab; the zero buffer is shared by all components so the
                // per-rank memory is one slab, not 48 full fields.
                attempt([&]() {
                    if(localValues == 0u)
                        return;
                    if(options.float64)
                    {
                        auto zero = std::shared_ptr<double[]>(new double[localValues]());
                        for(char const* record : {"Convolutional PML E", "Convolutional PML B"})
                            for(char const* slab : pmlSlabs)
                            {
                                Mesh& mesh
                                    = iteration.meshes[std::string{record} + "_" + slab];
                                for(char const* component : pmlComponents)
                                    mesh[component].storeChunk(
                                        zero, fieldChunkOffset, fieldChunkExtent);
                            }
                    }
                    else
                    {
                        auto zero = std::shared_ptr<float[]>(new float[localValues]());
                        for(char const* record : {"Convolutional PML E", "Convolutional PML B"})
                            for(char const* slab : pmlSlabs)
                            {
                                Mesh& mesh
                                    = iteration.meshes[std::string{record} + "_" + slab];
                                for(char const* component : pmlComponents)
                                    mesh[component].storeChunk(
                                        zero, fieldChunkOffset, fieldChunkExtent);
                            }
                    }
                });
            }

            attempt([&]() { out.flush(); });
            checkpoint();

            // A same-name generated entry overwrites the raw input group and
            // therefore skips the copy.  A renamed entry (input "b" -> output
            // "e") still copies "b" raw.  A checkpoint-only empty species also
            // skips the copy: the input may carry a same-named group whose
            // particlePatches must not survive next to the generated
            // domain-covering patch row.
            std::set<std::string> overwrittenInputs;
            for(auto const& generated : options.generatedSpecies)
            {
                if(generated.empty)
                {
                    overwrittenInputs.insert(generated.species);
                    continue;
                }
                std::string const inputName = generated.inputSpecies.empty()
                    ? generated.species
                    : generated.inputSpecies;
                if(inputName == generated.species)
                    overwrittenInputs.insert(inputName);
            }
            std::uint64_t copiedParticles = 0;
            for(auto const& name : inSpeciesNames)
            {
                if(overwrittenInputs.count(name) != 0u)
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
                start.setAttribute("maxNumProc", std::uint64_t{1});
                MeshRecordComponent next = idp["nextId"];
                allocateFull(next, Datatype::ULONGLONG, 1, 1, 1);
                next.setPosition(zeroPos);
                next.setUnitSI(1.0);
                // The id provider is a single scalar pair owned by rank 0.
                attempt([&]() {
                    if(rank != 0)
                        return;
                    start.storeChunk(
                        ownedChunk(&startId, 1u), Offset{0u, 0u, 0u}, Extent{1u, 1u, 1u});
                    next.storeChunk(
                        ownedChunk(&nextId, 1u), Offset{0u, 0u, 0u}, Extent{1u, 1u, 1u});
                });
                attempt([&]() { out.flush(); });
                checkpoint();
            }

            // Particles: copy the non-overwritten input species, then write
            // the generated/empty ones.  A renamed generated entry keeps its
            // raw input group.
            for(auto const& name : inSpeciesNames)
            {
                if(overwrittenInputs.count(name) != 0u)
                    continue; // written as a generated species below
                copyInputSpecies(
                    in, inStep, out, iteration, name, options.float64, rank == 0,
                    [&out]() { out.flush(); });
            }

            // PML state is part of a PIConGPU checkpoint and is loaded by an
            // open-boundary restart.  Copy every Convolutional PML mesh from
            // the input checkpoint instead of synthesizing it.
            for(auto const& entry : inIteration.meshes)
            {
                std::string const& name = entry.first;
                if(name.rfind("Convolutional PML ", 0u) != 0u)
                    continue;
                Mesh srcMesh = inIteration.meshes[name];
                Mesh dstMesh = iteration.meshes[name];
                copyMesh(in, out, srcMesh, dstMesh, field.grid.n, collective);
                if(rank == 0)
                    std::cerr << "picBunch: copied Convolutional PML record '" << name
                              << "'\n";
            }

            // Missing patch metadata is non-fatal but may make PIConGPU
            // reject the restart; record every affected species on the root.
            std::string restartWarning;
            for(auto const& name : inSpeciesNames)
            {
                if(overwrittenInputs.count(name) != 0u)
                    continue;
                if(inIteration.particles[name].particlePatches.numPatches() == 0u)
                {
                    if(!restartWarning.empty())
                        restartWarning += "; ";
                    restartWarning += "species '" + name
                        + "' has no particlePatches metadata";
                }
            }
            if(!restartWarning.empty())
            {
                std::cerr << "picBunch: warning: " << restartWarning
                          << "; PIConGPU may reject the restart\n";
                out.setAttribute("picBunchRestartWarning", restartWarning);
            }
            std::uint64_t idBase = copiedParticles;
            for(auto const& generated : options.generatedSpecies)
            {
                writeParticleSpecies(
                    iteration, generated.species, generated,
                    options.float64, idBase, rank == 0,
                    [&out]() { out.flush(); });
                idBase += generated.empty
                    ? std::uint64_t{0}
                    : static_cast<std::uint64_t>(generated.bunch.size());
            }

            attempt([&]() { out.flush(); });
            checkpoint();
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
        // checkpoints.txt is plain-text metadata owned by rank 0 only.
        if(mpiRank() == 0)
            appendCheckpointStep(options.directory, options.step);
    }

} // namespace picBunch
