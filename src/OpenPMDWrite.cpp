// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// openPMD writer: the six Yee-staggered field components as scalar mesh
// records.  Values are SI (V/m, T), grid metadata is SI, and every component
// carries its Yee staggering through the openPMD ``position`` attribute.
//
// The series is written under a temporary name in the destination directory and
// renamed only after a successful flush, so a failed write never leaves a
// half-written output file behind.
//
// Host/queue contract: every input of this writer is a host-resident
// std::vector.  The compute stage owns the alpaka queue and issues an explicit
// alpaka::onHost::wait() before it copies device results into these vectors, so
// this translation unit never reads device memory and owns no queue.  The
// openPMD/HDF5 side is asynchronous in its own right: series.flush() executes
// the queued reads and finalises the writes before any loaded pointer is
// dereferenced or the series is closed.

#include "picBunch/OpenPMDWriter.hpp"

#include "picBunch/Mpi.hpp"

#include <openPMD/openPMD.hpp>
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
#include <iostream>
#include <memory>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#    include <unistd.h>
#else
#    include <process.h>
#endif

namespace picBunch
{
    namespace
    {
        //! x,y,z -> z,y,x, the stored openPMD axis order.
        template<typename T>
        std::vector<T> reverse(alpaka::Vec<T, 3u> const& value)
        {
            return {value[2], value[1], value[0]};
        }

        long long processId()
        {
#if defined(__unix__) || defined(__APPLE__)
            return static_cast<long long>(::getpid());
#else
            return static_cast<long long>(::_getpid());
#endif
        }

        //! Same extension, unique marker before it, in the same directory.
        std::string temporaryName(std::string const& fileName)
        {
            std::string const marker = ".picbunch-tmp-" + std::to_string(processId());
            auto const slash = fileName.find_last_of("/\\");
            auto const dot = fileName.find_last_of('.');
            if(dot != std::string::npos && (slash == std::string::npos || dot > slash))
                return fileName.substr(0, dot) + marker + fileName.substr(dot);
            return fileName + marker;
        }

        std::string regexEscape(char c)
        {
            static std::string const special = R"(.^$|()[]{}*+?\)";
            if(special.find(c) != std::string::npos)
                return std::string("\\") + c;
            return std::string(1, c);
        }

        //! Replace every "%T" in ``name`` with ``iteration``.
        std::string expandPattern(std::string const& name, std::string const& iteration)
        {
            std::string result;
            for(std::size_t i = 0; i < name.size();)
            {
                if(name.compare(i, 2, "%T") == 0)
                {
                    result += iteration;
                    i += 2;
                }
                else
                    result += name[i++];
            }
            return result;
        }

        /*! Maps the temporary series path(s) to the final path(s).
         *
         * A concrete name is a single rename.  A "%T" pattern names one file per
         * iteration; the temporary files are found by matching the numbered
         * expansion of the temporary pattern.
         */
        std::vector<std::pair<std::filesystem::path, std::filesystem::path>> planRenames(
            std::string const& temporary,
            std::string const& finalName)
        {
            namespace fs = std::filesystem;
            if(finalName.find("%T") == std::string::npos)
                return {{fs::path(temporary), fs::path(finalName)}};

            fs::path const temporaryPath(temporary);
            fs::path const finalPath(finalName);
            fs::path const directory
                = temporaryPath.has_parent_path() ? temporaryPath.parent_path() : fs::path(".");
            std::string const pattern = temporaryPath.filename().string();

            std::string expression = "^";
            for(std::size_t i = 0; i < pattern.size();)
            {
                if(pattern.compare(i, 2, "%T") == 0)
                {
                    expression += "([0-9]+)";
                    i += 2;
                }
                else
                    expression += regexEscape(pattern[i++]);
            }
            expression += "$";
            std::regex const matcher(expression);

            std::vector<std::pair<fs::path, fs::path>> renames;
            for(auto const& entry : fs::directory_iterator(directory))
            {
                // HDF5 series are regular files; an ADIOS2/BP5 series is a
                // directory (``md.0``, ``data.0`` ...).  Both must be published.
                if(!entry.is_regular_file() && !entry.is_directory())
                    continue;
                std::smatch match;
                std::string const candidate = entry.path().filename().string();
                if(std::regex_match(candidate, match, matcher))
                    renames.emplace_back(
                        entry.path(),
                        finalPath.parent_path() / expandPattern(finalPath.filename().string(), match[1].str()));
            }
            return renames;
        }

        //! Publish one temporary series path, replacing an existing target.
        /*! ``remove_all`` also removes a non-empty ADIOS2/BP5 directory; plain
         * ``remove`` only deletes an empty directory and would make a repeated
         * write to the same ``.bp`` path fail.  The temporary path is already
         * fully finalised (Series closed) when this runs, so the replace is
         * still an atomic rename for both backends.
         */
        void publishTemporary(
            std::filesystem::path const& from,
            std::filesystem::path const& to)
        {
            std::error_code error;
            std::filesystem::remove_all(to, error);
            std::filesystem::rename(from, to);
        }

        void removeTemporary(std::string const& temporary, std::string const& finalName) noexcept
        {
            namespace fs = std::filesystem;
            std::error_code ignored;
            try
            {
                for(auto const& entry : planRenames(temporary, finalName))
                    fs::remove_all(entry.first, ignored);
            }
            catch(...)
            {
            }
        }

        void publishHdf5Series(
            Field const& field,
            std::string const& fileName,
            FieldWriteOptions const& options)
        {
            openPMD::Series series(fileName, openPMD::Access::CREATE);
            auto& iteration = series.iterations[options.iteration];
            iteration.setTime(options.time);
            iteration.setTimeUnitSI(options.timeUnitSI);
            iteration.setAttribute("timeOffset", options.timeOffset);

            // Stored axis order is z,y,x so that the fastest-varying axis of the
            // C-order dataset is x, matching Field::data.
            std::vector<std::string> const axisLabels{"z", "y", "x"};
            openPMD::Extent const extent{
                static_cast<openPMD::Extent::value_type>(field.grid.n[2]),
                static_cast<openPMD::Extent::value_type>(field.grid.n[1]),
                static_cast<openPMD::Extent::value_type>(field.grid.n[0])};
            openPMD::Offset const offset{0, 0, 0};
            openPMD::Dataset const dataset{openPMD::determineDatatype<double>(), extent};

            struct RecordSpec
            {
                char const* name;
                std::size_t firstComponent;
                std::array<double, 7> unitDimension;
            };
            // openPMD unitDimension order is (L, M, T, I, theta, N, J).
            // E is kg m s^-3 A^-1, B is kg s^-2 A^-1.
            RecordSpec const records[] = {
                {"E", 0, {1.0, 1.0, -3.0, -1.0, 0.0, 0.0, 0.0}},
                {"B", 3, {0.0, 1.0, -2.0, -1.0, 0.0, 0.0, 0.0}}};

            char const componentNames[3] = {'x', 'y', 'z'};
            for(auto const& spec : records)
            {
                auto& mesh = iteration.meshes[spec.name];
                mesh.setGeometry(openPMD::Mesh::Geometry::cartesian);
                mesh.setDataOrder(openPMD::Mesh::DataOrder::C);
                mesh.setAxisLabels(axisLabels);
                mesh.setGridSpacing(reverse(field.grid.spacing));
                mesh.setGridGlobalOffset(reverse(field.grid.origin));
                mesh.setGridUnitSI(1.0);
                mesh.setUnitDimension(
                    {{openPMD::UnitDimension::L, spec.unitDimension[0]},
                     {openPMD::UnitDimension::M, spec.unitDimension[1]},
                     {openPMD::UnitDimension::T, spec.unitDimension[2]},
                     {openPMD::UnitDimension::I, spec.unitDimension[3]},
                     {openPMD::UnitDimension::theta, spec.unitDimension[4]},
                     {openPMD::UnitDimension::N, spec.unitDimension[5]},
                     {openPMD::UnitDimension::J, spec.unitDimension[6]}});

                for(std::size_t a = 0; a < 3; ++a)
                {
                    std::string const name(1, componentNames[a]);
                    auto& component = mesh[name];
                    component.resetDataset(dataset);
                    component.storeChunkRaw(
                        field.data[spec.firstComponent + a].data(), offset, extent);
                    component.setUnitSI(1.0);
                    // position attribute in cell-size units, stored axis order z,y,x.
                    Vec3 const yee{
                        kYeeOffset[spec.firstComponent + a][0] / kRefinement,
                        kYeeOffset[spec.firstComponent + a][1] / kRefinement,
                        kYeeOffset[spec.firstComponent + a][2] / kRefinement};
                    component.setAttribute("position", reverse(yee));
                }
            }

            series.flush();
        }
    } // namespace

    void writeField(Field const& field, std::string const& fileName, FieldWriteOptions const& options)
    {
        // Validate all six components before touching the file system.
        for(std::size_t c = 0; c < field.data.size(); ++c)
            if(field.data[c].size() != field.grid.size())
                throw std::runtime_error(
                    "writeField: component " + std::to_string(c) + " has "
                    + std::to_string(field.data[c].size()) + " values, grid.size() is "
                    + std::to_string(field.grid.size()));
        if(field.grid.size() == 0u)
            throw std::runtime_error("writeField: grid is empty");

        std::string const temporary = temporaryName(fileName);
        try
        {
            publishHdf5Series(field, temporary, options);

            auto const renames = planRenames(temporary, fileName);
            if(renames.empty())
                throw std::runtime_error(
                    "writeField: the temporary series '" + temporary + "' produced no output file");
            for(auto const& [from, to] : renames)
                publishTemporary(from, to);
        }
        catch(...)
        {
            removeTemporary(temporary, fileName);
            throw;
        }
    }

    namespace
    {
        //! Check that the compute field lies on global nodes and inside the domain.
        std::array<long long, 3> globalBase(
            Grid const& fieldGrid,
            Idx3 const& globalGridSize,
            Vec3 const& globalGridOrigin)
        {
            std::array<long long, 3> base{};
            for(int a = 0; a < 3; ++a)
            {
                if(globalGridSize[a] == 0u)
                    throw std::invalid_argument("writeFieldToDomain: global extent must be positive");
                double const relative
                    = (fieldGrid.origin[a] - globalGridOrigin[a]) / fieldGrid.spacing[a];
                double const nearest = std::round(relative);
                if(std::abs(relative - nearest) > 1e-6)
                    throw std::invalid_argument(
                        "writeFieldToDomain: the compute grid is not aligned to the global grid on axis "
                        + std::to_string(a));
                if(nearest < 0.0
                   || nearest + static_cast<double>(fieldGrid.n[a])
                       > static_cast<double>(globalGridSize[a]))
                    throw std::invalid_argument(
                        "writeFieldToDomain: the compute grid is not contained in the global domain on axis "
                        + std::to_string(a));
                base[a] = static_cast<long long>(nearest);
            }
            return base;
        }

        /*! Fill one tile of the global array for one component.
         *
         * ``tile`` is x-fastest and covers global ``[zBegin,zEnd) x [0,Ny) x [0,Nx)``.
         * Nodes inside the compute box are copied.  Outside it, the multipole
         * far tail is evaluated when ``tail`` is given; without a tail the
         * remaining nodes stay zero (the diagnostic/truncation path and the
         * direct writer tests).
         */
        void fillGlobalTile(
            std::vector<double>& tile,
            Field const& field,
            int component,
            std::array<long long, 3> const& base,
            Idx3 const& globalN,
            std::size_t zBegin,
            std::size_t zEnd,
            FarTailBatch const* tail,
            Vec3 const& globalOrigin)
        {
            std::size_t const nx = globalN[0];
            std::size_t const ny = globalN[1];
            std::size_t const nz = zEnd - zBegin;
            tile.assign(nx * ny * nz, 0.0);
            if(nz == 0u)
                return;

            // For a distributed field the component arrays hold only the
            // rank's z-slab; the tile covers this rank's full global slab, so
            // clamp the copy to the locally owned planes.
            long long const fieldZBegin = base[2]
                + static_cast<long long>(field.zSlab ? field.slabOffset : 0u);
            long long const fieldZEnd = fieldZBegin
                + static_cast<long long>(field.zSlab ? field.slabCount : field.grid.n[2]);
            std::array<long long, 3> lo{
                std::max<long long>(base[0], 0),
                std::max<long long>(base[1], 0),
                std::max<long long>(base[2], std::max(
                    static_cast<long long>(zBegin), fieldZBegin))};
            std::array<long long, 3> hi{
                std::min<long long>(base[0] + static_cast<long long>(field.grid.n[0]),
                                    static_cast<long long>(globalN[0])),
                std::min<long long>(base[1] + static_cast<long long>(field.grid.n[1]),
                                    static_cast<long long>(globalN[1])),
                std::min<long long>(base[2] + static_cast<long long>(field.grid.n[2]),
                                    std::min(static_cast<long long>(zEnd), fieldZEnd))};
            bool const overlaps = lo[0] < hi[0] && lo[1] < hi[1] && lo[2] < hi[2];
            if(overlaps)
            {
            std::size_t const fnx = field.grid.n[0];
            std::size_t const fny = field.grid.n[1];
            for(long long k = lo[2]; k < hi[2]; ++k)
            {
                std::size_t const fk = static_cast<std::size_t>(k - fieldZBegin);
                for(long long j = lo[1]; j < hi[1]; ++j)
                {
                    std::size_t const fj = static_cast<std::size_t>(j - base[1]);
                    std::size_t const dstRow
                        = (static_cast<std::size_t>(k) - zBegin) * nx * ny
                        + static_cast<std::size_t>(j) * nx;
                    std::size_t const srcRow = fk * fnx * fny + fj * fnx;
                    for(long long i = lo[0]; i < hi[0]; ++i)
                    {
                        std::size_t const fi = static_cast<std::size_t>(i - base[0]);
                        tile[dstRow + static_cast<std::size_t>(i)] = field.data[component][srcRow + fi];
                    }
                }
            }
            } // overlaps
            if(tail == nullptr)
                return;
            // Outside the exact compute box: evaluate the monopole/dipole/
            // quadrupole tail instead of writing zeros.  The exact box was
            // expanded to the measured expansion switch radius, so every node
            // reached here is in the valid far zone.
            auto const& spacing = field.grid.spacing;
            for(long long k = static_cast<long long>(zBegin);
                k < static_cast<long long>(zEnd); ++k)
                for(long long j = 0; j < static_cast<long long>(ny); ++j)
                    for(long long i = 0; i < static_cast<long long>(nx); ++i)
                    {
                        bool const inside = i >= lo[0] && i < hi[0]
                            && j >= lo[1] && j < hi[1] && k >= lo[2] && k < hi[2];
                        if(inside)
                            continue;
                        // The component metadata advertises its Yee offset in
                        // PIC-cell units.  Evaluate the tail at the same
                        // staggered point as the exact compute box, whose node
                        // spacing is kRefinement * cellSize.
                        Vec3 const position{
                            globalOrigin[0] + static_cast<double>(i) * spacing[0]
                                + kYeeOffset[component][0] * spacing[0] / kRefinement,
                            globalOrigin[1] + static_cast<double>(j) * spacing[1]
                                + kYeeOffset[component][1] * spacing[1] / kRefinement,
                            globalOrigin[2] + static_cast<double>(k) * spacing[2]
                                + kYeeOffset[component][2] * spacing[2] / kRefinement};
                        double value = 0.0;
                        if(component < 3)
                            value = evaluateFarTailE(*tail, position)[component];
                        else
                            value = evaluateFarTailB(*tail, position)[component - 3];
                        tile[(static_cast<std::size_t>(k) - zBegin) * nx * ny
                             + static_cast<std::size_t>(j) * nx
                             + static_cast<std::size_t>(i)]
                            = value;
                    }
        }
    } // namespace

    void writeFieldToDomain(
        Field const& field,
        std::string const& fileName,
        FieldWriteOptions const& options,
        Idx3 const& globalGridSize,
        Vec3 const& globalGridOrigin,
        FarTail const* tail)
    {
        for(std::size_t c = 0; c < field.data.size(); ++c)
            if(field.data[c].size() != field.localSize())
                throw std::runtime_error(
                    "writeFieldToDomain: component " + std::to_string(c) + " has "
                    + std::to_string(field.data[c].size()) + " values, localSize() is "
                    + std::to_string(field.localSize()));
        if(field.grid.size() == 0u)
            throw std::runtime_error("writeFieldToDomain: grid is empty");
        auto const base = globalBase(field.grid, globalGridSize, globalGridOrigin);

        // Flatten the far-tail groups once, not per node and group: the
        // per-group boost geometry is hoisted out of the node loop and the
        // group constants become contiguous arrays.  The values are unchanged
        // (see Margin.FarTailBatchMatchesRecursiveGroups).
        std::optional<FarTailBatch> tailBatch;
        if(tail != nullptr)
            tailBatch.emplace(makeFarTailBatch(*tail));

        int const rank = mpiRank();
        int const size = mpiSize();

#if defined(PICBUNCH_ENABLE_MPI) && openPMD_HAVE_MPI
        bool const collective = mpiActive();
#else
        bool const collective = false;
#endif

        // A slab per rank, divided into tiles.  The tile z extent is chosen once
        // from the global x/y extents, so every rank issues the same number of
        // collective flushes even when its slab is shorter.
        std::size_t const nx = globalGridSize[0];
        std::size_t const ny = globalGridSize[1];
        std::size_t const nz = globalGridSize[2];
        std::size_t const maxSlab = (nz + static_cast<std::size_t>(size) - 1u)
            / static_cast<std::size_t>(size);
        constexpr std::size_t kTargetTileBytes = 128u * 1024u * 1024u;
        std::size_t const bytesPerZ = nx * ny * sizeof(double);
        std::size_t tileZ = bytesPerZ == 0u ? maxSlab : kTargetTileBytes / bytesPerZ;
        tileZ = std::max<std::size_t>(tileZ, 1u);
        tileZ = std::min<std::size_t>(tileZ, std::max<std::size_t>(maxSlab, 1u));
        std::size_t const tiles = (maxSlab + tileZ - 1u) / tileZ;

        std::size_t const zBegin = (static_cast<std::size_t>(rank) * nz)
            / static_cast<std::size_t>(size);
        std::size_t const zEnd = ((static_cast<std::size_t>(rank) + 1u) * nz)
            / static_cast<std::size_t>(size);

        // Parallel openPMD writes one shared series.  Without its MPI backend,
        // each rank writes its own file; the content is then chunk-local and a
        // warning says so.
        std::string outputName = fileName;
        if(!collective && size > 1)
        {
            char suffix[32];
            std::snprintf(suffix, sizeof(suffix), ".rank%04d", rank);
            outputName += suffix;
            if(rank == 0)
                std::fprintf(
                    stderr,
                    "picBunch: warning: openPMD has no MPI backend; writing one file per rank "
                    "(%s.rankNNNN) instead of a single collectively written series\n",
                    fileName.c_str());
        }

        std::string temporary = temporaryName(outputName);
        if(collective)
            mpiBroadcastString(temporary, 0);

        try
        {
            std::unique_ptr<openPMD::Series> series;
#if defined(PICBUNCH_ENABLE_MPI) && openPMD_HAVE_MPI
            if(collective)
                series = std::make_unique<openPMD::Series>(
                    temporary, openPMD::Access::CREATE, MPI_COMM_WORLD);
            else
#endif
                series = std::make_unique<openPMD::Series>(temporary, openPMD::Access::CREATE);

            auto& iteration = series->iterations[options.iteration];
            iteration.setTime(options.time);
            iteration.setTimeUnitSI(options.timeUnitSI);
            iteration.setAttribute("timeOffset", options.timeOffset);

            std::vector<std::string> const axisLabels{"z", "y", "x"};
            openPMD::Extent const extent{
                static_cast<openPMD::Extent::value_type>(globalGridSize[2]),
                static_cast<openPMD::Extent::value_type>(globalGridSize[1]),
                static_cast<openPMD::Extent::value_type>(globalGridSize[0])};
            openPMD::Dataset const dataset{openPMD::determineDatatype<double>(), extent};

            struct RecordSpec
            {
                char const* name;
                std::size_t firstComponent;
                std::array<double, 7> unitDimension;
            };
            RecordSpec const records[] = {
                {"E", 0, {1.0, 1.0, -3.0, -1.0, 0.0, 0.0, 0.0}},
                {"B", 3, {0.0, 1.0, -2.0, -1.0, 0.0, 0.0, 0.0}}};

            char const componentNames[3] = {'x', 'y', 'z'};
            for(auto const& spec : records)
            {
                auto& mesh = iteration.meshes[spec.name];
                mesh.setGeometry(openPMD::Mesh::Geometry::cartesian);
                mesh.setDataOrder(openPMD::Mesh::DataOrder::C);
                mesh.setAxisLabels(axisLabels);
                mesh.setGridSpacing(
                    std::vector<double>{
                        field.grid.spacing[2], field.grid.spacing[1], field.grid.spacing[0]});
                mesh.setGridGlobalOffset(
                    std::vector<double>{
                        globalGridOrigin[2], globalGridOrigin[1], globalGridOrigin[0]});
                mesh.setGridUnitSI(1.0);
                mesh.setUnitDimension(
                    {{openPMD::UnitDimension::L, spec.unitDimension[0]},
                     {openPMD::UnitDimension::M, spec.unitDimension[1]},
                     {openPMD::UnitDimension::T, spec.unitDimension[2]},
                     {openPMD::UnitDimension::I, spec.unitDimension[3]},
                     {openPMD::UnitDimension::theta, spec.unitDimension[4]},
                     {openPMD::UnitDimension::N, spec.unitDimension[5]},
                     {openPMD::UnitDimension::J, spec.unitDimension[6]}});

                for(std::size_t a = 0; a < 3; ++a)
                {
                    std::string const name(1, componentNames[a]);
                    auto& component = mesh[name];
                    component.resetDataset(dataset);
                    component.setUnitSI(1.0);
                    Vec3 const yee{
                        kYeeOffset[spec.firstComponent + a][0] / kRefinement,
                        kYeeOffset[spec.firstComponent + a][1] / kRefinement,
                        kYeeOffset[spec.firstComponent + a][2] / kRefinement};
                    component.setAttribute("position", reverse(yee));
                }
            }

            std::vector<double> tile;
            for(std::size_t t = 0; t < tiles; ++t)
            {
                std::size_t const tBegin = zBegin + t * tileZ;
                std::size_t const tEnd = std::min(zBegin + (t + 1u) * tileZ, zEnd);
                for(auto const& spec : records)
                    for(std::size_t a = 0; a < 3; ++a)
                    {
                        auto& component = iteration.meshes[spec.name][std::string(1, componentNames[a])];
                        std::size_t const thisEnd = std::min(tEnd, zEnd);
                        std::size_t const thisBegin = std::min(tBegin, thisEnd);
                        if(thisBegin < thisEnd)
                        {
                            fillGlobalTile(
                                tile,
                                field,
                                static_cast<int>(spec.firstComponent + a),
                                base,
                                globalGridSize,
                                thisBegin,
                                thisEnd,
                                tailBatch ? &*tailBatch : nullptr,
                                globalGridOrigin);
                            openPMD::Extent const chunkExtent{
                                static_cast<openPMD::Extent::value_type>(thisEnd - thisBegin),
                                static_cast<openPMD::Extent::value_type>(ny),
                                static_cast<openPMD::Extent::value_type>(nx)};
                            openPMD::Offset const chunkOffset{thisBegin, 0, 0};
                            auto buffer = std::shared_ptr<double[]>(new double[tile.size()]());
                            std::copy(tile.begin(), tile.end(), buffer.get());
                            component.storeChunk(buffer, chunkOffset, chunkExtent);
                        }
                    }
                // One collective flush per z-tile instead of one per component.
                // The old per-component flush multiplied the collective HDF5
                // metadata round-trips by six and dominated the multi-rank
                // writer; a tile's six component buffers are still released
                // here, so peak memory stays at six tile buffers per rank.
                series->flush();
            }

            // Close (and thereby finalise) the openPMD Series before the
            // directory is renamed.  With ADIOS2/BP5 the engine writes deferred
            // metadata such as ``profiling.json`` into the series directory at
            // finalize time; renaming first made ADIOS2 2.11 reopen the old
            // temporary path and abort (rc=134).  HDF5 happened to finish all
            // deferred writes inside the last flush, which is why only BP5 hit
            // the race.  Series::close() is collective for an MPI series, so
            // every rank must reach it.
            series->close();
            series.reset();

            if(collective)
                mpiBarrier();

            auto const renames = planRenames(temporary, outputName);
            if(renames.empty())
                throw std::runtime_error(
                    "writeFieldToDomain: the temporary series '" + temporary
                    + "' produced no output file");
            if(collective && rank != 0)
            {
                // Only rank 0 owns the shared temporary path; the other ranks
                // must not rename it.
            }
            else
            {
                for(auto const& [from, to] : renames)
                    publishTemporary(from, to);
            }
            // Keep the other ranks from reopening the shared temporary name
            // (for the next iteration) until rank 0 has published it.
            if(collective)
                mpiBarrier();
        }
        catch(...)
        {
            if(!collective || rank == 0)
                removeTemporary(temporary, outputName);
            throw;
        }
    }

} // namespace picBunch
