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

#include "picBunch/OpenPMDWriter.hpp"

#include "picBunch/Mpi.hpp"

#include <openPMD/openPMD.hpp>
#include <openPMD/config.hpp>

#include <hdf5.h>

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
#include <regex>
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
        std::vector<T> reverse(std::array<T, 3> const& value)
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
                if(!entry.is_regular_file())
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

        void removeTemporary(std::string const& temporary, std::string const& finalName) noexcept
        {
            namespace fs = std::filesystem;
            std::error_code ignored;
            try
            {
                for(auto const& entry : planRenames(temporary, finalName))
                    fs::remove(entry.first, ignored);
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
                    std::array<double, 3> const yee{
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
            {
                std::error_code error;
                std::filesystem::remove(to, error);
                std::filesystem::rename(from, to);
            }
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
            std::array<std::size_t, 3> const& globalGridSize,
            std::array<double, 3> const& globalGridOrigin)
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
            std::array<std::size_t, 3> const& globalN,
            std::size_t zBegin,
            std::size_t zEnd,
            FarTail const* tail,
            std::array<double, 3> const& globalOrigin)
        {
            std::size_t const nx = globalN[0];
            std::size_t const ny = globalN[1];
            std::size_t const nz = zEnd - zBegin;
            tile.assign(nx * ny * nz, 0.0);
            if(nz == 0u)
                return;

            std::array<long long, 3> lo{
                std::max<long long>(base[0], 0),
                std::max<long long>(base[1], 0),
                std::max<long long>(base[2], static_cast<long long>(zBegin))};
            std::array<long long, 3> hi{
                std::min<long long>(base[0] + static_cast<long long>(field.grid.n[0]),
                                    static_cast<long long>(globalN[0])),
                std::min<long long>(base[1] + static_cast<long long>(field.grid.n[1]),
                                    static_cast<long long>(globalN[1])),
                std::min<long long>(base[2] + static_cast<long long>(field.grid.n[2]),
                                    static_cast<long long>(zEnd))};
            bool const overlaps = lo[0] < hi[0] && lo[1] < hi[1] && lo[2] < hi[2];
            if(overlaps)
            {
            std::size_t const fnx = field.grid.n[0];
            std::size_t const fny = field.grid.n[1];
            for(long long k = lo[2]; k < hi[2]; ++k)
            {
                std::size_t const fk = static_cast<std::size_t>(k - base[2]);
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
                        std::array<double, 3> const position{
                            globalOrigin[0] + static_cast<double>(i) * spacing[0],
                            globalOrigin[1] + static_cast<double>(j) * spacing[1],
                            globalOrigin[2] + static_cast<double>(k) * spacing[2]};
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
        std::array<std::size_t, 3> const& globalGridSize,
        std::array<double, 3> const& globalGridOrigin,
        FarTail const* tail)
    {
        for(std::size_t c = 0; c < field.data.size(); ++c)
            if(field.data[c].size() != field.grid.size())
                throw std::runtime_error(
                    "writeFieldToDomain: component " + std::to_string(c) + " has "
                    + std::to_string(field.data[c].size()) + " values, grid.size() is "
                    + std::to_string(field.grid.size()));
        if(field.grid.size() == 0u)
            throw std::runtime_error("writeFieldToDomain: grid is empty");
        auto const base = globalBase(field.grid, globalGridSize, globalGridOrigin);

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
                    std::array<double, 3> const yee{
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
                                tail,
                                globalGridOrigin);
                            openPMD::Extent const chunkExtent{
                                static_cast<openPMD::Extent::value_type>(thisEnd - thisBegin),
                                static_cast<openPMD::Extent::value_type>(ny),
                                static_cast<openPMD::Extent::value_type>(nx)};
                            openPMD::Offset const chunkOffset{thisBegin, 0, 0};
                            auto buffer = std::make_shared<double[]>(tile.size());
                            std::copy(tile.begin(), tile.end(), buffer.get());
                            component.storeChunk(buffer, chunkOffset, chunkExtent);
                        }
                        // Collective flush: every rank calls it the same number
                        // of times, so the buffers above are released per tile.
                        series->flush();
                    }
            }

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
                {
                    std::error_code error;
                    std::filesystem::remove(to, error);
                    std::filesystem::rename(from, to);
                }
            }
        }
        catch(...)
        {
            removeTemporary(temporary, outputName);
            throw;
        }
    }

} // namespace picBunch

namespace picBunch
{
    namespace
    {
        [[noreturn]] void hdf5Fail(std::string const& what)
        {
            throw std::runtime_error("PicOnGPU restart writer: HDF5 " + what + " failed");
        }

        hid_t checked(hid_t id, char const* what)
        {
            if(id < 0)
                hdf5Fail(what);
            return id;
        }

        void checked(herr_t status, char const* what)
        {
            if(status < 0)
                hdf5Fail(what);
        }

        void closeQuietly(hid_t id)
        {
            if(id < 0 || H5Iis_valid(id) <= 0)
                return;
            switch(H5Iget_type(id))
            {
            case H5I_FILE:
                H5Fclose(id);
                break;
            case H5I_GROUP:
                H5Gclose(id);
                break;
            case H5I_DATASET:
                H5Dclose(id);
                break;
            case H5I_DATATYPE:
                H5Tclose(id);
                break;
            case H5I_DATASPACE:
                H5Sclose(id);
                break;
            case H5I_ATTR:
                H5Aclose(id);
                break;
            default:
                H5Idec_ref(id);
                break;
            }
        }

        void setStringAttribute(hid_t object, char const* name, std::string const& value)
        {
            hid_t type = checked(H5Tcopy(H5T_C_S1), "H5Tcopy");
            checked(H5Tset_size(type, H5T_VARIABLE), "H5Tset_size");
            hid_t space = checked(H5Screate(H5S_SCALAR), "H5Screate");
            hid_t attribute = checked(
                H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT), "H5Acreate2");
            char const* pointer = value.c_str();
            checked(H5Awrite(attribute, type, &pointer), "H5Awrite");
            closeQuietly(attribute);
            closeQuietly(space);
            closeQuietly(type);
        }

        void setStringArrayAttribute(hid_t object, char const* name, std::vector<std::string> const& values)
        {
            hid_t type = checked(H5Tcopy(H5T_C_S1), "H5Tcopy");
            checked(H5Tset_size(type, H5T_VARIABLE), "H5Tset_size");
            hsize_t const dimensions[1] = {static_cast<hsize_t>(values.size())};
            hid_t space = checked(H5Screate_simple(1, dimensions, nullptr), "H5Screate_simple");
            hid_t attribute = checked(
                H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT), "H5Acreate2");
            std::vector<char const*> pointers;
            pointers.reserve(values.size());
            for(auto const& value : values)
                pointers.push_back(value.c_str());
            checked(H5Awrite(attribute, type, pointers.data()), "H5Awrite");
            closeQuietly(attribute);
            closeQuietly(space);
            closeQuietly(type);
        }

        template<typename T>
        void setScalarAttribute(hid_t object, char const* name, hid_t type, T const& value)
        {
            hid_t space = checked(H5Screate(H5S_SCALAR), "H5Screate");
            hid_t attribute = checked(
                H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT), "H5Acreate2");
            checked(H5Awrite(attribute, type, &value), "H5Awrite");
            closeQuietly(attribute);
            closeQuietly(space);
        }

        void setDoubleArrayAttribute(hid_t object, char const* name, std::array<double, 3> const& values)
        {
            hsize_t const dimensions[1] = {3};
            hid_t space = checked(H5Screate_simple(1, dimensions, nullptr), "H5Screate_simple");
            hid_t attribute = checked(
                H5Acreate2(object, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT), "H5Acreate2");
            checked(H5Awrite(attribute, H5T_NATIVE_DOUBLE, values.data()), "H5Awrite");
            closeQuietly(attribute);
            closeQuietly(space);
        }

        void setUnitDimension(hid_t object, std::array<double, 7> const& values)
        {
            hsize_t const dimensions[1] = {7};
            hid_t space = checked(H5Screate_simple(1, dimensions, nullptr), "H5Screate_simple");
            hid_t attribute = checked(
                H5Acreate2(object, "unitDimension", H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT),
                "H5Acreate2");
            checked(H5Awrite(attribute, H5T_NATIVE_DOUBLE, values.data()), "H5Awrite");
            closeQuietly(attribute);
            closeQuietly(space);
        }

        hid_t createFieldGroup(hid_t parent, char const* name)
        {
            hid_t group = checked(
                H5Gcreate2(parent, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "H5Gcreate2");
            setStringArrayAttribute(group, "axisLabels", {"z", "y", "x"});
            setStringAttribute(group, "dataOrder", "C");
            setStringAttribute(group, "geometry", "cartesian");
            double const zero = 0.0;
            setScalarAttribute(group, "timeOffset", H5T_NATIVE_FLOAT, static_cast<float>(zero));
            setStringAttribute(group, "fieldSmoothing", "none");
            return group;
        }

        hid_t createComponent(
            hid_t parent,
            char const* name,
            hsize_t const (&dimensions)[3],
            double const* data,
            std::size_t count,
            double unit,
            std::array<double, 3> const& position,
            bool float64 = false)
        {
            hid_t space = checked(H5Screate_simple(3, dimensions, nullptr), "H5Screate_simple");
            hid_t dataset = checked(
                H5Dcreate2(
                    parent,
                    name,
                    float64 ? H5T_IEEE_F64LE : H5T_IEEE_F32LE,
                    space,
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                    H5P_DEFAULT),
                "H5Dcreate2");
            if(float64)
            {
                std::vector<double> stored(count);
                for(std::size_t i = 0; i < count; ++i)
                    stored[i] = data[i] / unit;
                checked(
                    H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, stored.data()),
                    "H5Dwrite");
            }
            else
            {
                std::vector<float> stored(count);
                for(std::size_t i = 0; i < count; ++i)
                    stored[i] = static_cast<float>(data[i] / unit);
                checked(
                    H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, stored.data()),
                    "H5Dwrite");
            }
            setDoubleArrayAttribute(dataset, "position", position);
            setScalarAttribute(dataset, "unitSI", H5T_NATIVE_DOUBLE, unit);
            closeQuietly(space);
            return dataset;
        }

        std::string expandIteration(std::string name, std::uint64_t iteration)
        {
            auto replaceAll = [&](std::string const& marker, std::string const& replacement) {
                std::string::size_type position = 0;
                while((position = name.find(marker, position)) != std::string::npos)
                {
                    name.replace(position, marker.size(), replacement);
                    position += replacement.size();
                }
            };
            char padded[32];
            std::snprintf(padded, sizeof(padded), "%06llu", static_cast<unsigned long long>(iteration));
            replaceAll("%06T", padded);
            replaceAll("%T", std::to_string(iteration));
            return name;
        }

        std::string locateInputFile(std::string const& path, std::uint64_t iteration)
        {
            namespace fs = std::filesystem;
            std::error_code ignored;
            if(fs::is_regular_file(path, ignored))
                return path;
            if(path.find("%T") != std::string::npos || path.find("%06T") != std::string::npos)
            {
                std::string const expanded = expandIteration(path, iteration);
                if(fs::is_regular_file(expanded, ignored))
                    return expanded;
            }
            throw std::runtime_error(
                "PicOnGPU restart writer: cannot locate the input HDF5 file for '"
                + path + "' at iteration " + std::to_string(iteration));
        }

        bool linkExists(hid_t file, std::string const& path)
        {
            htri_t const result = H5Lexists(file, path.c_str(), H5P_DEFAULT);
            if(result < 0)
                hdf5Fail("H5Lexists('" + path + "')");
            return result > 0;
        }

        /*! All particle species groups of the input iteration.
         *
         * PIConGPU restarts every compiled species from the checkpoint, so the
         * writer copies every species present instead of dropping the ones that
         * do not contain the bunch.  ``requested`` is only a consistency check;
         * it does not restrict the copy.
         */
        std::vector<std::string> findSpeciesGroups(
            hid_t file,
            std::uint64_t iteration,
            std::string const& requested)
        {
            std::string const particles = "data/" + std::to_string(iteration) + "/particles";
            if(!linkExists(file, particles))
                throw std::runtime_error(
                    "PicOnGPU restart writer: input has no '" + particles + "' group; "
                    "cannot copy particles");
            if(!requested.empty())
            {
                std::string const path = particles + "/" + requested;
                if(!linkExists(file, path))
                    throw std::runtime_error(
                        "PicOnGPU restart writer: input has no particle species '" + requested + "' at '" + path + "'");
            }
            hid_t group = checked(H5Gopen2(file, particles.c_str(), H5P_DEFAULT), "H5Gopen2");
            std::vector<std::string> names;
            struct Context
            {
                std::vector<std::string>* names;
            } context{&names};
            auto collect = [](hid_t, char const* name, H5L_info_t const*, void* info) -> herr_t {
                auto* data = static_cast<Context*>(info);
                data->names->emplace_back(name);
                return 0;
            };
            hsize_t index = 0;
            checked(H5Literate(group, H5_INDEX_NAME, H5_ITER_INC, &index, collect, &context), "H5Literate");
            closeQuietly(group);
            if(names.empty())
                throw std::runtime_error(
                    "PicOnGPU restart writer: input iteration has no particle species groups");
            std::vector<std::string> groups;
            groups.reserve(names.size());
            for(auto const& name : names)
                groups.push_back(particles + "/" + name);
            return groups;
        }

        std::string particleWarning(
            hid_t sourceFile,
            std::string const& sourceGroup,
            std::string const& species)
        {
            std::vector<std::string> missing;
            if(!linkExists(sourceFile, sourceGroup + "/particlePatches"))
                missing.emplace_back("particlePatches");
            for(char const* record : {"position", "positionOffset", "momentum", "weighting"})
                if(!linkExists(sourceFile, sourceGroup + "/" + record))
                    missing.emplace_back(record);
            if(missing.empty())
                return {};
            std::string message = "input particle species '" + species + "' is missing";
            for(auto const& item : missing)
                message += " " + item;
            message += "; PIConGPU restart requires them and may reject this checkpoint";
            return message;
        }

        bool readDatasetElementCount(hid_t file, std::string const& path, std::uint64_t& count)
        {
            if(!linkExists(file, path))
                return false;
            H5O_info_t info;
            checked(H5Oget_info_by_name(file, path.c_str(), &info, H5P_DEFAULT), "H5Oget_info_by_name");
            if(info.type != H5O_TYPE_DATASET)
                return false;
            hid_t dataset = checked(H5Dopen2(file, path.c_str(), H5P_DEFAULT), "H5Dopen2");
            hid_t space = checked(H5Dget_space(dataset), "H5Dget_space");
            hssize_t const points = H5Sget_simple_extent_npoints(space);
            closeQuietly(space);
            closeQuietly(dataset);
            if(points < 0)
                hdf5Fail("H5Sget_simple_extent_npoints('" + path + "')");
            count = static_cast<std::uint64_t>(points);
            return true;
        }

        bool readPatchParticleCount(hid_t file, std::string const& group, std::uint64_t& count)
        {
            std::string const path = group + "/particlePatches/numParticles";
            if(!linkExists(file, path))
                return false;
            hid_t dataset = checked(H5Dopen2(file, path.c_str(), H5P_DEFAULT), "H5Dopen2");
            hid_t space = checked(H5Dget_space(dataset), "H5Dget_space");
            hssize_t const points = H5Sget_simple_extent_npoints(space);
            if(points < 0)
            {
                closeQuietly(space);
                closeQuietly(dataset);
                hdf5Fail("H5Sget_simple_extent_npoints('" + path + "')");
            }
            std::vector<std::uint64_t> values(static_cast<std::size_t>(points));
            checked(
                H5Dread(dataset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()),
                "H5Dread");
            closeQuietly(space);
            closeQuietly(dataset);
            std::uint64_t total = 0;
            for(auto const value : values)
            {
                if(total > std::numeric_limits<std::uint64_t>::max() - value)
                    throw std::runtime_error(
                        "PicOnGPU restart writer: particle patch counts overflow uint64");
                total += value;
            }
            count = total;
            return true;
        }

        /*! Number of particles actually copied for one species.
         *
         * The position record is the authoritative per-particle extent.  Patch
         * metadata is the fallback when the record layout is not the usual
         * openPMD 2.0 ``position/x`` vector record.
         */
        std::uint64_t countSpeciesParticles(hid_t file, std::string const& group)
        {
            std::uint64_t count = 0;
            if(readDatasetElementCount(file, group + "/position/x", count))
                return count;
            if(readPatchParticleCount(file, group, count))
                return count;
            if(readDatasetElementCount(file, group + "/position", count))
                return count;
            throw std::runtime_error(
                "PicOnGPU restart writer: cannot determine the particle count of species '"
                + group.substr(group.find_last_of('/') + 1)
                + "'; the input has neither a position record nor particlePatches/numParticles");
        }

        void appendCheckpointStep(std::filesystem::path const& directory, std::uint64_t step)
        {
            std::filesystem::path const path = directory / "checkpoints.txt";
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

        void copyAttributes(hid_t source, hid_t destination, bool skipExisting = false)
        {
            H5O_info_t info;
            checked(H5Oget_info(source, &info), "H5Oget_info");
            for(hsize_t index = 0; index < info.num_attrs; ++index)
            {
                hid_t attribute = checked(
                    H5Aopen_by_idx(
                        source, ".", H5_INDEX_NAME, H5_ITER_INC, index, H5P_DEFAULT, H5P_DEFAULT),
                    "H5Aopen_by_idx");
                char name[256];
                ssize_t const length = H5Aget_name(attribute, sizeof(name), name);
                if(length < 0 || static_cast<std::size_t>(length) >= sizeof(name))
                {
                    closeQuietly(attribute);
                    throw std::runtime_error(
                        "PicOnGPU restart writer: cannot read an attribute name");
                }
                if(skipExisting && H5Aexists(destination, name) > 0)
                {
                    closeQuietly(attribute);
                    continue;
                }
                hid_t type = checked(H5Aget_type(attribute), "H5Aget_type");
                hid_t space = checked(H5Aget_space(attribute), "H5Aget_space");
                hssize_t const points = H5Sget_simple_extent_npoints(space);
                std::size_t const elementSize = H5Tget_size(type);
                std::vector<char> buffer(static_cast<std::size_t>(points) * elementSize + 1u);
                checked(H5Aread(attribute, type, buffer.data()), "H5Aread");
                hid_t created = checked(
                    H5Acreate2(destination, name, type, space, H5P_DEFAULT, H5P_DEFAULT),
                    "H5Acreate2");
                checked(H5Awrite(created, type, buffer.data()), "H5Awrite");
                closeQuietly(created);
                closeQuietly(space);
                closeQuietly(type);
                closeQuietly(attribute);
            }
        }

        template<typename T>
        void rewriteDataset(hid_t file, std::string const& path, hid_t memoryType, hid_t fileType)
        {
            if(H5Lexists(file, path.c_str(), H5P_DEFAULT) <= 0)
                return;
            H5O_info_t info;
            checked(
                H5Oget_info_by_name(file, path.c_str(), &info, H5P_DEFAULT),
                "H5Oget_info_by_name");
            if(info.type != H5O_TYPE_DATASET)
                return;
            hid_t dataset = checked(H5Dopen2(file, path.c_str(), H5P_DEFAULT), "H5Dopen2");
            hid_t space = checked(H5Dget_space(dataset), "H5Dget_space");
            int const rank = H5Sget_simple_extent_ndims(space);
            std::array<hsize_t, 8> dimensions{};
            checked(
                H5Sget_simple_extent_dims(space, dimensions.data(), nullptr),
                "H5Sget_simple_extent_dims");
            hssize_t const points = H5Sget_simple_extent_npoints(space);
            std::vector<T> buffer(static_cast<std::size_t>(points));
            checked(
                H5Dread(dataset, memoryType, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data()),
                "H5Dread");
            std::string const temporary = path + ".picbunch-convert";
            hid_t newSpace = checked(
                H5Screate_simple(rank, dimensions.data(), nullptr), "H5Screate_simple");
            hid_t newDataset = checked(
                H5Dcreate2(
                    file, temporary.c_str(), fileType, newSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                "H5Dcreate2");
            checked(
                H5Dwrite(newDataset, memoryType, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data()),
                "H5Dwrite");
            copyAttributes(dataset, newDataset);
            closeQuietly(newDataset);
            closeQuietly(newSpace);
            closeQuietly(space);
            closeQuietly(dataset);
            checked(H5Ldelete(file, path.c_str(), H5P_DEFAULT), "H5Ldelete");
            checked(
                H5Lmove(file, temporary.c_str(), file, path.c_str(), H5P_DEFAULT, H5P_DEFAULT),
                "H5Lmove");
        }

        void flattenScalarRecordToFloat(hid_t file, std::string const& path)
        {
            if(H5Lexists(file, path.c_str(), H5P_DEFAULT) <= 0)
                return;
            H5O_info_t info;
            checked(
                H5Oget_info_by_name(file, path.c_str(), &info, H5P_DEFAULT),
                "H5Oget_info_by_name");
            if(info.type == H5O_TYPE_DATASET)
            {
                rewriteDataset<float>(file, path, H5T_NATIVE_FLOAT, H5T_IEEE_F32LE);
                return;
            }
            hid_t group = checked(H5Gopen2(file, path.c_str(), H5P_DEFAULT), "H5Gopen2");
            std::vector<std::string> names;
            struct Context
            {
                std::vector<std::string>* names;
            } context{&names};
            auto collect = [](hid_t, char const* name, H5L_info_t const*, void* data) -> herr_t {
                static_cast<Context*>(data)->names->emplace_back(name);
                return 0;
            };
            hsize_t index = 0;
            H5Literate(group, H5_INDEX_NAME, H5_ITER_INC, &index, collect, &context);
            if(names.size() != 1u)
            {
                closeQuietly(group);
                throw std::runtime_error(
                    "PicOnGPU restart writer: scalar record '" + path
                    + "' must contain exactly one dataset");
            }
            std::string const child = path + "/" + names.front();
            hid_t dataset = checked(H5Dopen2(file, child.c_str(), H5P_DEFAULT), "H5Dopen2");
            hid_t space = checked(H5Dget_space(dataset), "H5Dget_space");
            int const rank = H5Sget_simple_extent_ndims(space);
            std::array<hsize_t, 8> dimensions{};
            checked(
                H5Sget_simple_extent_dims(space, dimensions.data(), nullptr),
                "H5Sget_simple_extent_dims");
            hssize_t const points = H5Sget_simple_extent_npoints(space);
            std::vector<float> buffer(static_cast<std::size_t>(points));
            checked(
                H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data()),
                "H5Dread");
            std::string const temporary = path + ".picbunch-convert";
            hid_t newSpace = checked(
                H5Screate_simple(rank, dimensions.data(), nullptr), "H5Screate_simple");
            hid_t newDataset = checked(
                H5Dcreate2(
                    file, temporary.c_str(), H5T_IEEE_F32LE, newSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                "H5Dcreate2");
            checked(
                H5Dwrite(newDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data()),
                "H5Dwrite");
            copyAttributes(group, newDataset);
            copyAttributes(dataset, newDataset, true);
            closeQuietly(newDataset);
            closeQuietly(newSpace);
            closeQuietly(space);
            closeQuietly(dataset);
            closeQuietly(group);
            checked(H5Ldelete(file, path.c_str(), H5P_DEFAULT), "H5Ldelete");
            checked(
                H5Lmove(file, temporary.c_str(), file, path.c_str(), H5P_DEFAULT, H5P_DEFAULT),
                "H5Lmove");
        }

        void removeRecordIfPresent(hid_t file, std::string const& path)
        {
            if(H5Lexists(file, path.c_str(), H5P_DEFAULT) > 0)
                checked(H5Ldelete(file, path.c_str(), H5P_DEFAULT), "H5Ldelete");
        }

        std::vector<std::string> groupNames(hid_t group)
        {
            std::vector<std::string> names;
            struct Context
            {
                std::vector<std::string>* names;
            } context{&names};
            auto collect = [](hid_t, char const* name, H5L_info_t const*, void* data) -> herr_t {
                static_cast<Context*>(data)->names->emplace_back(name);
                return 0;
            };
            hsize_t index = 0;
            checked(H5Literate(group, H5_INDEX_NAME, H5_ITER_INC, &index, collect, &context), "H5Literate");
            return names;
        }

        bool startsWith(std::string const& value, std::string const& prefix)
        {
            return value.size() >= prefix.size()
                && value.compare(0, prefix.size(), prefix) == 0;
        }

        //! Names of the ``Convolutional PML ...`` records in the source iteration.
        std::vector<std::string> pmlRecordNames(hid_t source, std::uint64_t iteration)
        {
            std::string const path = "data/" + std::to_string(iteration) + "/fields";
            if(!linkExists(source, path))
                return {};
            hid_t group = checked(H5Gopen2(source, path.c_str(), H5P_DEFAULT), "H5Gopen2");
            auto names = groupNames(group);
            closeQuietly(group);
            names.erase(
                std::remove_if(
                    names.begin(),
                    names.end(),
                    [](std::string const& name) { return !startsWith(name, "Convolutional PML "); }),
                names.end());
            return names;
        }

        void removeAttributeIfPresent(hid_t object, char const* name)
        {
            if(H5Aexists(object, name) > 0)
                checked(H5Adelete(object, name), "H5Adelete");
        }

        //! PIConGPU restart reads float_X; convert a non-float32 component once.
        void ensureFloat32Components(hid_t file, std::string const& recordPath)
        {
            hid_t group = checked(H5Gopen2(file, recordPath.c_str(), H5P_DEFAULT), "H5Gopen2");
            auto const names = groupNames(group);
            for(auto const& name : names)
            {
                std::string const childPath = recordPath + "/" + name;
                hid_t dataset = checked(H5Dopen2(file, childPath.c_str(), H5P_DEFAULT), "H5Dopen2");
                hid_t type = checked(H5Dget_type(dataset), "H5Dget_type");
                bool const isFloat32 = H5Tget_class(type) == H5T_FLOAT && H5Tget_size(type) == 4u;
                closeQuietly(type);
                closeQuietly(dataset);
                if(!isFloat32)
                    rewriteDataset<float>(file, childPath, H5T_NATIVE_FLOAT, H5T_IEEE_F32LE);
            }
            closeQuietly(group);
        }

        /*! Copy the source checkpoint's PML records without recomputing them.
         *
         * The values and their per-component ``position``/``unitSI`` attributes
         * come from the source, so the PML state stays exactly what PIConGPU
         * wrote.  The record geometry is replaced with the main-field geometry
         * so the slabs describe the same grid as E and B.
         */
        void copyPmlRecords(
            hid_t file,
            hid_t source,
            hid_t destinationFields,
            std::string const& sourceIterationPath,
            std::string const& destinationIterationPath,
            std::vector<std::string> const& names,
            Field const& field,
            PicOnGPUWriteOptions const& options)
        {
            std::array<double, 3> const spacing{
                field.grid.spacing[2] / options.unitLength,
                field.grid.spacing[1] / options.unitLength,
                field.grid.spacing[0] / options.unitLength};
            std::array<double, 3> const origin{
                field.grid.origin[2] / options.unitLength,
                field.grid.origin[1] / options.unitLength,
                field.grid.origin[0] / options.unitLength};
            hsize_t const expected[3] = {
                static_cast<hsize_t>(field.grid.n[2]),
                static_cast<hsize_t>(field.grid.n[1]),
                static_cast<hsize_t>(field.grid.n[0])};
            for(auto const& name : names)
            {
                std::string const sourcePath = sourceIterationPath + "/fields/" + name;
                checked(
                    H5Ocopy(
                        source,
                        sourcePath.c_str(),
                        destinationFields,
                        name.c_str(),
                        H5P_DEFAULT,
                        H5P_DEFAULT),
                    "H5Ocopy");
                std::string const destinationPath = destinationIterationPath + "/fields/" + name;
                hid_t record = checked(H5Gopen2(file, destinationPath.c_str(), H5P_DEFAULT), "H5Gopen2");
                auto const components = groupNames(record);
                for(auto const& component : components)
                {
                    std::string const componentPath = destinationPath + "/" + component;
                    hid_t dataset = checked(H5Dopen2(file, componentPath.c_str(), H5P_DEFAULT), "H5Dopen2");
                    hid_t space = checked(H5Dget_space(dataset), "H5Dget_space");
                    hsize_t dimensions[3] = {0, 0, 0};
                    int const rank = H5Sget_simple_extent_dims(space, dimensions, nullptr);
                    closeQuietly(space);
                    closeQuietly(dataset);
                    if(rank != 3 || dimensions[0] != expected[0] || dimensions[1] != expected[1]
                       || dimensions[2] != expected[2])
                        throw std::runtime_error(
                            "PicOnGPU restart writer: PML record '" + name + "' component '"
                            + component + "' does not match the output grid; refusing to copy "
                            "a slab with a different extent");
                }
                closeQuietly(record);
                ensureFloat32Components(file, destinationPath);

                record = checked(H5Gopen2(file, destinationPath.c_str(), H5P_DEFAULT), "H5Gopen2");
                for(char const* attribute : {
                        "axisLabels",
                        "dataOrder",
                        "geometry",
                        "gridSpacing",
                        "gridGlobalOffset",
                        "gridUnitSI"})
                    removeAttributeIfPresent(record, attribute);
                setStringArrayAttribute(record, "axisLabels", {"z", "y", "x"});
                setStringAttribute(record, "dataOrder", "C");
                setStringAttribute(record, "geometry", "cartesian");
                setDoubleArrayAttribute(record, "gridSpacing", spacing);
                setDoubleArrayAttribute(record, "gridGlobalOffset", origin);
                setScalarAttribute(record, "gridUnitSI", H5T_NATIVE_DOUBLE, options.unitLength);
                closeQuietly(record);
            }
        }

        void setDoubleVectorAttribute(
            hid_t object,
            char const* name,
            std::vector<double> const& values)
        {
            hsize_t const dimensions[1] = {static_cast<hsize_t>(values.size())};
            hid_t space = checked(H5Screate_simple(1, dimensions, nullptr), "H5Screate_simple");
            hid_t attribute = checked(
                H5Acreate2(object, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT),
                "H5Acreate2");
            checked(H5Awrite(attribute, H5T_NATIVE_DOUBLE, values.data()), "H5Awrite");
            closeQuietly(attribute);
            closeQuietly(space);
        }

        hid_t createOneDimensional(
            hid_t parent,
            char const* name,
            hsize_t count,
            hid_t fileType)
        {
            hid_t space = checked(H5Screate_simple(1, &count, nullptr), "H5Screate_simple");
            hid_t dataset = checked(
                H5Dcreate2(parent, name, fileType, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                "H5Dcreate2");
            closeQuietly(space);
            return dataset;
        }

        void setParticleRecordAttributes(
            hid_t record,
            std::array<double, 7> const& unitDimension,
            bool macroWeighted,
            double weightingPower)
        {
            setUnitDimension(record, unitDimension);
            setScalarAttribute(
                record,
                "macroWeighted",
                H5T_NATIVE_UINT32,
                static_cast<std::uint32_t>(macroWeighted ? 1u : 0u));
            setScalarAttribute(record, "weightingPower", H5T_NATIVE_DOUBLE, weightingPower);
            setScalarAttribute(record, "timeOffset", H5T_NATIVE_FLOAT, 0.0f);
        }

        void writeFloatVectorRecord(
            hid_t species,
            char const* recordName,
            std::array<std::vector<double>, 3> const& values,
            std::array<double, 3> const& unitSI,
            std::array<double, 7> const& unitDimension,
            bool macroWeighted,
            double weightingPower,
            bool float64)
        {
            hid_t record = checked(
                H5Gcreate2(species, recordName, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                "H5Gcreate2");
            setParticleRecordAttributes(record, unitDimension, macroWeighted, weightingPower);
            for(std::size_t a = 0; a < 3; ++a)
            {
                char const* componentName = a == 0 ? "x" : (a == 1 ? "y" : "z");
                hsize_t const count = static_cast<hsize_t>(values[a].size());
                hid_t dataset = createOneDimensional(
                    record, componentName, count, float64 ? H5T_IEEE_F64LE : H5T_IEEE_F32LE);
                if(float64)
                {
                    checked(
                        H5Dwrite(
                            dataset,
                            H5T_NATIVE_DOUBLE,
                            H5S_ALL,
                            H5S_ALL,
                            H5P_DEFAULT,
                            values[a].data()),
                        "H5Dwrite");
                }
                else
                {
                    std::vector<float> stored(values[a].size());
                    for(std::size_t i = 0; i < values[a].size(); ++i)
                        stored[i] = static_cast<float>(values[a][i]);
                    checked(
                        H5Dwrite(
                            dataset,
                            H5T_NATIVE_FLOAT,
                            H5S_ALL,
                            H5S_ALL,
                            H5P_DEFAULT,
                            stored.data()),
                        "H5Dwrite");
                }
                setScalarAttribute(dataset, "unitSI", H5T_NATIVE_DOUBLE, unitSI[a]);
                closeQuietly(dataset);
            }
            closeQuietly(record);
        }

        void writeIntegerVectorRecord(
            hid_t species,
            char const* recordName,
            std::array<std::vector<std::int32_t>, 3> const& values,
            std::array<double, 3> const& unitSI)
        {
            hid_t record = checked(
                H5Gcreate2(species, recordName, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                "H5Gcreate2");
            setParticleRecordAttributes(
                record, {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, false, 0.0);
            for(std::size_t a = 0; a < 3; ++a)
            {
                char const* componentName = a == 0 ? "x" : (a == 1 ? "y" : "z");
                hsize_t const count = static_cast<hsize_t>(values[a].size());
                hid_t dataset = createOneDimensional(record, componentName, count, H5T_STD_I32LE);
                checked(
                    H5Dwrite(
                        dataset,
                        H5T_NATIVE_INT32,
                        H5S_ALL,
                        H5S_ALL,
                        H5P_DEFAULT,
                        values[a].data()),
                    "H5Dwrite");
                setScalarAttribute(dataset, "unitSI", H5T_NATIVE_DOUBLE, unitSI[a]);
                closeQuietly(dataset);
            }
            closeQuietly(record);
        }

        void writeScalarRecord(
            hid_t species,
            char const* recordName,
            std::vector<double> const& values,
            double unitSI,
            std::array<double, 7> const& unitDimension,
            bool macroWeighted,
            double weightingPower,
            bool float64)
        {
            hid_t record = checked(
                H5Gcreate2(species, recordName, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                "H5Gcreate2");
            setParticleRecordAttributes(record, unitDimension, macroWeighted, weightingPower);
            hsize_t const count = static_cast<hsize_t>(values.size());
            hid_t dataset = createOneDimensional(
                record, recordName, count, float64 ? H5T_IEEE_F64LE : H5T_IEEE_F32LE);
            if(float64)
                checked(
                    H5Dwrite(
                        dataset,
                        H5T_NATIVE_DOUBLE,
                        H5S_ALL,
                        H5S_ALL,
                        H5P_DEFAULT,
                        values.data()),
                    "H5Dwrite");
            else
            {
                std::vector<float> stored(values.size());
                for(std::size_t i = 0; i < values.size(); ++i)
                    stored[i] = static_cast<float>(values[i]);
                checked(
                    H5Dwrite(
                        dataset,
                        H5T_NATIVE_FLOAT,
                        H5S_ALL,
                        H5S_ALL,
                        H5P_DEFAULT,
                        stored.data()),
                    "H5Dwrite");
            }
            setScalarAttribute(dataset, "unitSI", H5T_NATIVE_DOUBLE, unitSI);
            closeQuietly(dataset);
            closeQuietly(record);
        }

        void writeGeneratedSpecies(
            hid_t file,
            std::string const& speciesPath,
            GeneratedParticles const& generated,
            bool float64)
        {
            Bunch const& bunch = generated.bunch;
            std::size_t const count = bunch.size();
            if(count == 0u)
                throw std::runtime_error(
                    "PicOnGPU restart writer: generated species '"
                    + generated.species + "' has no particles");
            hid_t particles = checked(
                H5Gopen2(file, speciesPath.c_str(), H5P_DEFAULT), "H5Gopen2");
            // The generated group is new; there is nothing to delete first.
            std::array<std::vector<double>, 3> position;
            std::array<std::vector<double>, 3> momentum;
            std::array<std::vector<std::int32_t>, 3> positionOffset;
            std::vector<double> weighting(count, 0.0);
            std::vector<double> charge(count, 0.0);
            std::vector<double> mass(count, 0.0);
            if(bunch.charge.size() != count)
                throw std::runtime_error(
                    "PicOnGPU restart writer: generated species '"
                    + generated.species + "' has no per-particle charge; "
                    "the input bunch must carry it");
            if(!(generated.unitCharge > 0.0))
                throw std::runtime_error(
                    "PicOnGPU restart writer: generated species '"
                    + generated.species + "' needs a positive unit_charge to write the "
                    "charge record");
            if(!(generated.unitMass > 0.0))
                throw std::runtime_error(
                    "PicOnGPU restart writer: generated species '"
                    + generated.species + "' needs a positive unit_mass to write the "
                    "mass record");
            for(int a = 0; a < 3; ++a)
            {
                position[static_cast<std::size_t>(a)].resize(count);
                momentum[static_cast<std::size_t>(a)].resize(count);
                positionOffset[static_cast<std::size_t>(a)].resize(count);
            }
            for(std::size_t p = 0; p < count; ++p)
            {
                double const xyz[3] = {bunch.x[p], bunch.y[p], bunch.z[p]};
                double const u[3] = {bunch.ux[p], bunch.uy[p], bunch.uz[p]};
                double const restMass = (!bunch.restMass.empty() && bunch.restMass[p] > 0.0)
                    ? bunch.restMass[p]
                    : (generated.fallbackMass > 0.0
                           ? generated.fallbackMass
                           : generated.unitMass);
                for(int a = 0; a < 3; ++a)
                {
                    double const cells = (xyz[a] - generated.domainOrigin[a])
                        / generated.cellSize[a];
                    double const cellIndex = std::floor(cells + 1e-9);
                    if(cellIndex < 0.0
                       || cellIndex >= static_cast<double>(
                           generated.patchOffset[static_cast<std::size_t>(a)]
                           + static_cast<std::int64_t>(
                               generated.patchExtent[static_cast<std::size_t>(a)])))
                        throw std::runtime_error(
                            "PicOnGPU restart writer: particle " + std::to_string(p)
                            + " lies outside the generated patch on axis " + std::to_string(a));
                    position[static_cast<std::size_t>(a)][p] = cells - cellIndex;
                    positionOffset[static_cast<std::size_t>(a)][p]
                        = static_cast<std::int32_t>(cellIndex);
                    momentum[static_cast<std::size_t>(a)][p]
                        = u[a] * restMass * kSpeedOfLight / generated.unitMassSpeed;
                }
                weighting[p] = bunch.weight[p];
                charge[p] = bunch.charge[p] / generated.unitCharge;
                mass[p] = restMass / generated.unitMass;
            }

            std::array<double, 3> const cellUnit{
                generated.cellSize[0], generated.cellSize[1], generated.cellSize[2]};
            writeFloatVectorRecord(
                particles,
                "position",
                position,
                cellUnit,
                {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                false,
                0.0,
                true); // fractional in-cell position is always float64
            writeIntegerVectorRecord(particles, "positionOffset", positionOffset, cellUnit);
            writeFloatVectorRecord(
                particles,
                "momentum",
                momentum,
                {generated.unitMassSpeed,
                 generated.unitMassSpeed,
                 generated.unitMassSpeed},
                {1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0},
                true,
                1.0,
                float64);

            hsize_t const scalarCount = static_cast<hsize_t>(count);
            // The statistical weight can span orders of magnitude, so it is
            // always stored as float64, independent of the configured precision.
            hid_t weightingDataset = createOneDimensional(
                particles, "weighting", scalarCount, H5T_IEEE_F64LE);
            checked(
                H5Dwrite(
                    weightingDataset,
                    H5T_NATIVE_DOUBLE,
                    H5S_ALL,
                    H5S_ALL,
                    H5P_DEFAULT,
                    weighting.data()),
                "H5Dwrite");
            setParticleRecordAttributes(weightingDataset, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, true, 1.0);
            setScalarAttribute(weightingDataset, "unitSI", H5T_NATIVE_DOUBLE, 1.0);
            closeQuietly(weightingDataset);

            // Global particle ids run [0, count), matching the startId/nextId
            // range written below.  PIConGPU ignores the record for a species
            // without particleId, and loads it for a species that has one.
            std::vector<std::uint64_t> ids(count);
            for(std::size_t i = 0; i < count; ++i)
                ids[i] = static_cast<std::uint64_t>(i);
            hid_t idDataset = createOneDimensional(particles, "id", scalarCount, H5T_STD_U64LE);
            checked(
                H5Dwrite(idDataset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, ids.data()),
                "H5Dwrite");
            setParticleRecordAttributes(idDataset, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, false, 0.0);
            setScalarAttribute(idDataset, "unitSI", H5T_NATIVE_DOUBLE, 1.0);
            closeQuietly(idDataset);

            // PIConGPU can derive charge and mass from the species flags, but a
            // restart from this checkpoint is self-contained when both records
            // are present.  Values are stored in internal units and the unitSI
            // attribute carries the TOML factor.
            writeScalarRecord(
                particles,
                "charge",
                charge,
                generated.unitCharge,
                {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0},
                false,
                1.0,
                float64);
            writeScalarRecord(
                particles,
                "mass",
                mass,
                generated.unitMass,
                {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                false,
                1.0,
                float64);

            hid_t patches = checked(
                H5Gcreate2(particles, "particlePatches", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                "H5Gcreate2");
            std::uint64_t const total = static_cast<std::uint64_t>(count);
            std::uint64_t const offsetZero = 0;
            std::array<double, 7> const patchUnitDimension{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            hid_t numParticles = createOneDimensional(patches, "numParticles", 1, H5T_STD_U64LE);
            checked(
                H5Dwrite(numParticles, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, &total),
                "H5Dwrite");
            setUnitDimension(numParticles, patchUnitDimension);
            setScalarAttribute(numParticles, "unitSI", H5T_NATIVE_DOUBLE, 1.0);
            closeQuietly(numParticles);
            hid_t numParticlesOffset
                = createOneDimensional(patches, "numParticlesOffset", 1, H5T_STD_U64LE);
            checked(
                H5Dwrite(
                    numParticlesOffset,
                    H5T_NATIVE_UINT64,
                    H5S_ALL,
                    H5S_ALL,
                    H5P_DEFAULT,
                    &offsetZero),
                "H5Dwrite");
            setUnitDimension(numParticlesOffset, patchUnitDimension);
            setScalarAttribute(numParticlesOffset, "unitSI", H5T_NATIVE_DOUBLE, 1.0);
            closeQuietly(numParticlesOffset);
            hid_t offset = checked(
                H5Gcreate2(patches, "offset", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "H5Gcreate2");
            hid_t extent = checked(
                H5Gcreate2(patches, "extent", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "H5Gcreate2");
            setUnitDimension(offset, patchUnitDimension);
            setUnitDimension(extent, patchUnitDimension);
            for(std::size_t a = 0; a < 3; ++a)
            {
                char const* componentName = a == 0 ? "x" : (a == 1 ? "y" : "z");
                std::uint64_t const offsetValue
                    = static_cast<std::uint64_t>(generated.patchOffset[a]);
                std::uint64_t const extentValue = generated.patchExtent[a];
                hid_t offsetDataset = createOneDimensional(offset, componentName, 1, H5T_STD_U64LE);
                hid_t extentDataset = createOneDimensional(extent, componentName, 1, H5T_STD_U64LE);
                checked(
                    H5Dwrite(
                        offsetDataset,
                        H5T_NATIVE_UINT64,
                        H5S_ALL,
                        H5S_ALL,
                        H5P_DEFAULT,
                        &offsetValue),
                    "H5Dwrite");
                checked(
                    H5Dwrite(
                        extentDataset,
                        H5T_NATIVE_UINT64,
                        H5S_ALL,
                        H5S_ALL,
                        H5P_DEFAULT,
                        &extentValue),
                    "H5Dwrite");
                setScalarAttribute(offsetDataset, "unitSI", H5T_NATIVE_DOUBLE, 1.0);
                setScalarAttribute(extentDataset, "unitSI", H5T_NATIVE_DOUBLE, 1.0);
                closeQuietly(offsetDataset);
                closeQuietly(extentDataset);
            }
            closeQuietly(offset);
            closeQuietly(extent);
            closeQuietly(patches);
            closeQuietly(particles);
        }

        void publishPicOnGPU(
            Field const& field,
            PicOnGPUWriteOptions const& options,
            std::string const& temporary,
            std::string const& finalName)
        {
            hid_t file = checked(H5Fcreate(temporary.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), "H5Fcreate");
            hid_t source = -1;
            try
            {
                setStringAttribute(file, "openPMD", "1.1.0");
                setStringAttribute(file, "iterationEncoding", "fileBased");
                setStringAttribute(file, "iterationFormat", options.prefix + "_%06T");
                setStringAttribute(file, "meshesPath", "fields/");
                setStringAttribute(file, "particlesPath", "particles/");
                setStringAttribute(file, "basePath", "/data/%T/");
                setStringAttribute(file, "software", "picBunch");
                setStringAttribute(file, "softwareVersion", "0.1.0");
                setScalarAttribute(file, "openPMDextension", H5T_NATIVE_UINT32, std::uint32_t{1});
                setScalarAttribute(
                    file, "picongpuIOVersionMajor", H5T_NATIVE_UINT32, std::uint32_t{5});
                setScalarAttribute(
                    file, "picongpuIOVersionMinor", H5T_NATIVE_UINT32, std::uint32_t{0});

                // Locate and validate the particle source before writing the fields,
                // so a missing species fails without a half-written checkpoint.
                std::string const sourcePath = locateInputFile(options.inputFile, options.inputIteration);
                source = checked(H5Fopen(sourcePath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), "H5Fopen");
                std::vector<std::string> const sourceGroups
                    = findSpeciesGroups(source, options.inputIteration, options.species);
                std::string const generatedSpecies = options.generatedParticles
                    ? options.generatedParticles->species
                    : std::string{};
                if(!generatedSpecies.empty())
                {
                    bool found = false;
                    for(auto const& group : sourceGroups)
                        if(group.substr(group.find_last_of('/') + 1) == generatedSpecies)
                            found = true;
                    if(!found)
                        throw std::runtime_error(
                            "PicOnGPU restart writer: generated species '"
                            + generatedSpecies + "' is absent from the input iteration");
                }
                std::vector<std::string> warnings;
                std::uint64_t totalParticles = 0;
                for(auto const& sourceGroup : sourceGroups)
                {
                    std::string const species
                        = sourceGroup.substr(sourceGroup.find_last_of('/') + 1);
                    if(species == generatedSpecies)
                        continue;
                    std::string const warning = particleWarning(source, sourceGroup, species);
                    if(!warning.empty())
                        warnings.push_back(warning);
                    auto const count = countSpeciesParticles(source, sourceGroup);
                    if(totalParticles > std::numeric_limits<std::uint64_t>::max() - count)
                        throw std::runtime_error(
                            "PicOnGPU restart writer: particle count overflows uint64");
                    totalParticles += count;
                }
                if(options.generatedParticles)
                {
                    auto const count
                        = static_cast<std::uint64_t>(options.generatedParticles->bunch.size());
                    if(totalParticles > std::numeric_limits<std::uint64_t>::max() - count)
                        throw std::runtime_error(
                            "PicOnGPU restart writer: particle count overflows uint64");
                    totalParticles += count;
                }
                if(!warnings.empty())
                {
                    std::string joined;
                    for(std::size_t i = 0; i < warnings.size(); ++i)
                    {
                        if(i != 0)
                            joined += "; ";
                        joined += warnings[i];
                    }
                    std::cerr << "picBunch: warning: " << joined << '\n';
                    setStringAttribute(file, "picBunchRestartWarning", joined);
                }

                checked(
                    H5Gcreate2(file, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                    "H5Gcreate2");
                std::string const iterationPath = "data/" + std::to_string(options.step);
                hid_t iteration = checked(
                    H5Gcreate2(file, iterationPath.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                    "H5Gcreate2");
                setScalarAttribute(iteration, "iteration", H5T_NATIVE_UINT64, options.step);
                setScalarAttribute(iteration, "sim_slides", H5T_NATIVE_UINT32, std::uint32_t{0});
                setScalarAttribute(iteration, "dt", H5T_NATIVE_DOUBLE, options.dt);
                setScalarAttribute(iteration, "time", H5T_NATIVE_DOUBLE, options.time);
                setScalarAttribute(iteration, "timeUnitSI", H5T_NATIVE_DOUBLE, options.timeUnitSI);
                setScalarAttribute(iteration, "unit_length", H5T_NATIVE_DOUBLE, options.unitLength);
                setScalarAttribute(iteration, "unit_efield", H5T_NATIVE_DOUBLE, options.unitEField);
                setScalarAttribute(iteration, "unit_bfield", H5T_NATIVE_DOUBLE, options.unitBField);

                hid_t fields = checked(
                    H5Gcreate2(iteration, "fields", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "H5Gcreate2");
                setStringAttribute(fields, "fieldSolver", "Yee");
                setStringAttribute(fields, "currentSmoothing", "none");
                setScalarAttribute(fields, "BackgroundFieldIncluded", H5T_NATIVE_UINT8, std::uint8_t{0});

                hsize_t const dimensions[3] = {
                    static_cast<hsize_t>(field.grid.n[2]),
                    static_cast<hsize_t>(field.grid.n[1]),
                    static_cast<hsize_t>(field.grid.n[0])};
                struct RecordSpec
                {
                    char const* name;
                    std::size_t firstComponent;
                    std::array<double, 7> unitDimension;
                    double unit;
                };
                std::array<double, 7> const unitE{1.0, 1.0, -3.0, -1.0, 0.0, 0.0, 0.0};
                std::array<double, 7> const unitB{0.0, 1.0, -2.0, -1.0, 0.0, 0.0, 0.0};
                RecordSpec const records[] = {
                    {"E", 0, unitE, options.unitEField},
                    {"B", 3, unitB, options.unitBField}};

                char const* const componentNames[3] = {"x", "y", "z"};
                for(auto const& spec : records)
                {
                    hid_t record = createFieldGroup(fields, spec.name);
                    std::array<double, 3> const spacing{
                        field.grid.spacing[2] / options.unitLength,
                        field.grid.spacing[1] / options.unitLength,
                        field.grid.spacing[0] / options.unitLength};
                    std::array<double, 3> const origin{
                        field.grid.origin[2] / options.unitLength,
                        field.grid.origin[1] / options.unitLength,
                        field.grid.origin[0] / options.unitLength};
                    setDoubleArrayAttribute(record, "gridSpacing", spacing);
                    setDoubleArrayAttribute(record, "gridGlobalOffset", origin);
                    setScalarAttribute(record, "gridUnitSI", H5T_NATIVE_DOUBLE, options.unitLength);
                    setUnitDimension(record, spec.unitDimension);
                    for(std::size_t a = 0; a < 3; ++a)
                    {
                        std::array<double, 3> const position{
                            kYeeOffset[spec.firstComponent + a][0],
                            kYeeOffset[spec.firstComponent + a][1],
                            kYeeOffset[spec.firstComponent + a][2]};
                        hid_t dataset = createComponent(
                            record,
                            componentNames[a],
                            dimensions,
                            field.data[spec.firstComponent + a].data(),
                            field.grid.size(),
                            spec.unit,
                            position,
                            options.float64);
                        closeQuietly(dataset);
                    }
                    closeQuietly(record);
                }

                hid_t idProvider = checked(
                    H5Gcreate2(fields, "picongpu_idProvider", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                    "H5Gcreate2");
                setStringArrayAttribute(idProvider, "axisLabels", {"x"});
                setStringAttribute(idProvider, "dataOrder", "C");
                setStringAttribute(idProvider, "geometry", "cartesian");
                setDoubleVectorAttribute(idProvider, "gridGlobalOffset", {0.0});
                setDoubleVectorAttribute(idProvider, "gridSpacing", {1.0});
                setScalarAttribute(idProvider, "gridUnitSI", H5T_NATIVE_DOUBLE, 1.0);
                setScalarAttribute(idProvider, "timeOffset", H5T_NATIVE_FLOAT, 0.0f);
                setUnitDimension(idProvider, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
                hsize_t const idDimensions[3] = {1, 1, 1};
                hid_t idSpace = checked(H5Screate_simple(3, idDimensions, nullptr), "H5Screate_simple");
                // One global id range for the whole checkpoint, not one per rank
                // or per species.  The ids consumed by the copied particles are
                // [0, totalParticles), so nextId must cover all species together.
                // options.nextId is a lower bound for callers that reserve ids.
                std::uint64_t const startId = 0;
                std::uint64_t const nextId = std::max(options.nextId, totalParticles);
                hid_t startDataset = checked(
                    H5Dcreate2(
                        idProvider, "startId", H5T_STD_U64LE, idSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                    "H5Dcreate2");
                checked(H5Dwrite(startDataset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, &startId), "H5Dwrite");
                setScalarAttribute(startDataset, "maxNumProc", H5T_NATIVE_UINT64, std::uint64_t{1});
                setDoubleVectorAttribute(startDataset, "position", {0.0});
                setScalarAttribute(startDataset, "unitSI", H5T_NATIVE_DOUBLE, 1.0);
                hid_t nextDataset = checked(
                    H5Dcreate2(
                        idProvider, "nextId", H5T_STD_U64LE, idSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                    "H5Dcreate2");
                checked(H5Dwrite(nextDataset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, &nextId), "H5Dwrite");
                setDoubleVectorAttribute(nextDataset, "position", {0.0});
                setScalarAttribute(nextDataset, "unitSI", H5T_NATIVE_DOUBLE, 1.0);
                closeQuietly(startDataset);
                closeQuietly(nextDataset);
                closeQuietly(idSpace);
                closeQuietly(idProvider);

                // Copy every source particle group, then normalize each to the
                // layout and dtypes PIConGPU's particle loader expects:
                // float32 position/momentum/weighting and int32 positionOffset.
                hid_t particles = checked(
                    H5Gcreate2(iteration, "particles", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "H5Gcreate2");
                for(auto const& sourceGroup : sourceGroups)
                {
                    std::string const species
                        = sourceGroup.substr(sourceGroup.find_last_of('/') + 1);
                    if(species == generatedSpecies)
                        continue;
                    checked(
                        H5Ocopy(
                            source,
                            sourceGroup.c_str(),
                            particles,
                            species.c_str(),
                            H5P_DEFAULT,
                            H5P_DEFAULT),
                        "H5Ocopy");
                }
                if(options.generatedParticles)
                {
                    hid_t generatedGroup = checked(
                        H5Gcreate2(
                            particles,
                            options.generatedParticles->species.c_str(),
                            H5P_DEFAULT,
                            H5P_DEFAULT,
                            H5P_DEFAULT),
                        "H5Gcreate2");
                    closeQuietly(generatedGroup);
                }
                closeQuietly(particles);

                for(auto const& sourceGroup : sourceGroups)
                {
                    std::string const species
                        = sourceGroup.substr(sourceGroup.find_last_of('/') + 1);
                    if(species == generatedSpecies)
                        continue;
                    std::string const particlePath = iterationPath + "/particles/" + species;
                    for(char const* record : {"position", "momentum"})
                        for(char const* component : {"x", "y", "z"})
                            rewriteDataset<float>(
                                file,
                                particlePath + "/" + record + "/" + component,
                                H5T_NATIVE_FLOAT,
                                H5T_IEEE_F32LE);
                    for(char const* component : {"x", "y", "z"})
                        rewriteDataset<std::int32_t>(
                            file,
                            particlePath + "/positionOffset/" + component,
                            H5T_NATIVE_INT32,
                            H5T_STD_I32LE);
                    flattenScalarRecordToFloat(file, particlePath + "/weighting");
                    // PIConGPU derives charge and mass from the species flags and
                    // ignores these records; dropping them keeps the particle
                    // species extents homogeneous for the openPMD reader.
                    for(char const* record : {"charge", "mass", "id"})
                        removeRecordIfPresent(file, particlePath + "/" + record);
                }
                if(options.generatedParticles)
                    writeGeneratedSpecies(
                        file,
                        iterationPath + "/particles/" + options.generatedParticles->species,
                        *options.generatedParticles,
                        options.float64);

                // PML state is part of a PIConGPU checkpoint and is loaded
                // unconditionally by an open-boundary restart.  Copy it from the
                // input checkpoint; never synthesize it when the input has none.
                auto const pmlNames = pmlRecordNames(source, options.inputIteration);
                if(!pmlNames.empty())
                {
                    copyPmlRecords(
                        file,
                        source,
                        fields,
                        "data/" + std::to_string(options.inputIteration),
                        iterationPath,
                        pmlNames,
                        field,
                        options);
                    std::cerr << "picBunch: copied " << pmlNames.size()
                              << " Convolutional PML records from the input checkpoint\n";
                }

                closeQuietly(fields);
                closeQuietly(iteration);
                closeQuietly(source);
                source = -1;
                checked(H5Fclose(file), "H5Fclose");
                file = -1;

                namespace fs = std::filesystem;
                std::error_code ignored;
                fs::remove(finalName, ignored);
                fs::rename(temporary, finalName);
            }
            catch(...)
            {
                closeQuietly(source);
                closeQuietly(file);
                std::error_code ignored;
                std::filesystem::remove(temporary, ignored);
                throw;
            }
        }
    } // namespace

    void writePicOnGPUCheckpoint(Field const& field, PicOnGPUWriteOptions const& options)
    {
        for(std::size_t c = 0; c < field.data.size(); ++c)
            if(field.data[c].size() != field.grid.size())
                throw std::runtime_error(
                    "writePicOnGPUCheckpoint: component " + std::to_string(c) + " has "
                    + std::to_string(field.data[c].size()) + " values, grid.size() is "
                    + std::to_string(field.grid.size()));
        if(field.grid.size() == 0u)
            throw std::runtime_error("writePicOnGPUCheckpoint: grid is empty");
        if(options.prefix.empty())
            throw std::invalid_argument("writePicOnGPUCheckpoint: prefix must not be empty");
        if(options.inputFile.empty())
            throw std::invalid_argument("writePicOnGPUCheckpoint: inputFile must not be empty");
        auto const requireUnit = [](char const* name, double value) {
            if(!(value > 0.0) || !std::isfinite(value))
                throw std::invalid_argument(
                    std::string("writePicOnGPUCheckpoint: ") + name
                    + " must be finite and positive; no unit is inferred");
        };
        requireUnit("unitLength (--picongpu-unit-length)", options.unitLength);
        requireUnit("unitEField (--picongpu-unit-e)", options.unitEField);
        requireUnit("unitBField (--picongpu-unit-b)", options.unitBField);
        if(!(options.timeUnitSI > 0.0) || !std::isfinite(options.timeUnitSI))
            throw std::invalid_argument("writePicOnGPUCheckpoint: timeUnitSI must be finite and positive");
        if(options.step > 999999u)
            throw std::invalid_argument("writePicOnGPUCheckpoint: step must fit a six-digit file suffix");

        namespace fs = std::filesystem;
        fs::create_directories(options.directory);
        char suffix[32];
        std::snprintf(suffix, sizeof(suffix), "%06llu", static_cast<unsigned long long>(options.step));
        fs::path const finalPath = fs::path(options.directory) / (options.prefix + "_" + suffix + ".h5");
        fs::path const temporaryPath
            = fs::path(options.directory)
            / (options.prefix + "_" + suffix + ".picbunch-tmp-"
               + std::to_string(processId()) + ".h5");
        std::error_code ignored;
        fs::remove(temporaryPath, ignored);
        publishPicOnGPU(field, options, temporaryPath.string(), finalPath.string());
        appendCheckpointStep(options.directory, options.step);
    }
} // namespace picBunch
