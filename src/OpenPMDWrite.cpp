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

#include <hdf5.h>

#include <array>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>
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
            std::array<double, 3> const& position)
        {
            hid_t space = checked(H5Screate_simple(3, dimensions, nullptr), "H5Screate_simple");
            hid_t dataset = checked(
                H5Dcreate2(
                    parent,
                    name,
                    H5T_IEEE_F32LE,
                    space,
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                    H5P_DEFAULT),
                "H5Dcreate2");
            std::vector<float> stored(count);
            for(std::size_t i = 0; i < count; ++i)
                stored[i] = static_cast<float>(data[i] / unit);
            checked(H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, stored.data()), "H5Dwrite");
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

        std::string findSpeciesGroup(hid_t file, std::uint64_t iteration, std::string const& requested)
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
                return path;
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
            H5Literate(group, H5_INDEX_NAME, H5_ITER_INC, &index, collect, &context);
            closeQuietly(group);
            if(names.size() != 1u)
                throw std::runtime_error(
                    "PicOnGPU restart writer: input iteration has "
                    + std::to_string(names.size())
                    + " particle species; pass --picongpu-species to select one");
            return particles + "/" + names.front();
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
                std::string const sourceGroup
                    = findSpeciesGroup(source, options.inputIteration, options.species);
                std::string const species = sourceGroup.substr(sourceGroup.find_last_of('/') + 1);
                std::string const warning = particleWarning(source, sourceGroup, species);
                if(!warning.empty())
                {
                    std::cerr << "picBunch: warning: " << warning << '\n';
                    setStringAttribute(file, "picBunchRestartWarning", warning);
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
                            position);
                        closeQuietly(dataset);
                    }
                    closeQuietly(record);
                }

                hid_t idProvider = checked(
                    H5Gcreate2(fields, "picongpu_idProvider", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                    "H5Gcreate2");
                hsize_t const idDimensions[3] = {1, 1, 1};
                hid_t idSpace = checked(H5Screate_simple(3, idDimensions, nullptr), "H5Screate_simple");
                std::uint64_t const startId = 0;
                std::uint64_t const nextId = options.nextId;
                hid_t startDataset = checked(
                    H5Dcreate2(
                        idProvider, "startId", H5T_STD_U64LE, idSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                    "H5Dcreate2");
                checked(H5Dwrite(startDataset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, &startId), "H5Dwrite");
                setScalarAttribute(startDataset, "maxNumProc", H5T_NATIVE_UINT64, std::uint64_t{1});
                hid_t nextDataset = checked(
                    H5Dcreate2(
                        idProvider, "nextId", H5T_STD_U64LE, idSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                    "H5Dcreate2");
                checked(H5Dwrite(nextDataset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, &nextId), "H5Dwrite");
                closeQuietly(startDataset);
                closeQuietly(nextDataset);
                closeQuietly(idSpace);
                closeQuietly(idProvider);

                // Copy the source particle group as an HDF5 object so every
                // attribute and dataset (including particlePatches) survives.
                hid_t particles = checked(
                    H5Gcreate2(iteration, "particles", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "H5Gcreate2");
                checked(
                    H5Ocopy(
                        source,
                        sourceGroup.c_str(),
                        particles,
                        species.c_str(),
                        H5P_DEFAULT,
                        H5P_DEFAULT),
                    "H5Ocopy");
                closeQuietly(particles);

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
