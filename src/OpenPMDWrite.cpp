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

#include <openPMD/openPMD.hpp>

#include <array>
#include <cstddef>
#include <filesystem>
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

                char const componentNames[3] = {'x', 'y', 'z'};
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

} // namespace picBunch
