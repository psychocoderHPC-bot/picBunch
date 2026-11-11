// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// openPMD reader: particles (positions, momenta, weights, charge, mass) and
// mesh geometry.  No convention is inferred from unit equality: positions are
// x = position*unitSI + offset*unitSI_offset and momenta become u = p/(m c).

#include "picBunch/OpenPMD.hpp"

#include <openPMD/openPMD.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

namespace picBunch
{
    namespace
    {
        using openPMD::Datatype;
        using openPMD::Iteration;
        using openPMD::Mesh;
        using openPMD::ParticleSpecies;
        using openPMD::Record;
        using openPMD::RecordComponent;
        using openPMD::Series;

        [[noreturn]] void fail(std::string const& message)
        {
            throw std::runtime_error("picBunch openPMD reader: " + message);
        }

        void requireRecord(Record& record, std::string const& name, std::string const& where)
        {
            if(!record.contains(name))
                fail(where + ": required component '" + name + "' is missing");
        }

        /*! Read unitSI from a component, with an optional record-level fallback.
         *
         * openPMD 2.0 scalar records put the attributes on the record group and
         * the values in a child dataset; either location may carry unitSI.  The
         * openPMD 1.x form stores both on the dataset itself.
         */
        double checkedUnitSI(
            RecordComponent const& component,
            std::string const& path,
            Record const* recordFallback = nullptr)
        {
            std::optional<double> unit;
            try
            {
                unit = component.unitSI();
            }
            catch(std::exception const&)
            {
            }
            if(!unit && recordFallback != nullptr)
            {
                try
                {
                    unit = recordFallback->unitSI();
                }
                catch(std::exception const&)
                {
                }
            }
            double const value = unit.value_or(1.0);
            if(!std::isfinite(value) || !(value > 0.0))
                fail(path + ": unitSI must be finite and > 0");
            return value;
        }

        /*! The component that holds the values of a possibly scalar record.
         *
         * openPMD 2.0 writes a scalar record as a group with a single child
         * dataset.  The standard child name is ``value``; PIConGPU's files use
         * the record name itself.  A plain openPMD 1.x scalar record is a bare
         * dataset and is returned unchanged.
         */
        RecordComponent& scalarComponent(Record& record, std::string const& path)
        {
            // A plain openPMD 1.x scalar record is itself the dataset.
            if(record.getDatatype() != Datatype::UNDEFINED)
                return record;
            if(record.contains("value"))
                return record["value"];
            auto it = record.begin();
            if(it == record.end())
                return record;
            RecordComponent& first = it->second;
            ++it;
            if(it != record.end())
                fail(path + ": scalar record holds more than one dataset");
            return first;
        }

        void loadValuesInto(
            Series& series,
            RecordComponent& component,
            std::size_t offset,
            std::size_t count,
            std::string const& path,
            std::vector<double>& destination);

        /*! Read a scalar record into a reusable buffer.
         *
         * This is the streaming form of the former ``loadScalarValues``: the
         * per-component scratch buffer is the destination itself, so a chunked
         * read never holds a second full particle array next to the bunch.
         * ``destination`` is resized to ``count``.
         */
        void loadScalarValuesInto(
            Series& series,
            Record& record,
            std::size_t offset,
            std::size_t count,
            std::string const& path,
            std::vector<double>& destination)
        {
            destination.resize(count);
            // A constant scalar record may carry a ``value`` attribute instead of
            // a dataset.  Prefer a real ``value`` child when both exist.
            if(!record.contains("value") && record.containsAttribute("value"))
            {
                std::fill(destination.begin(), destination.end(), record.getAttribute("value").get<double>());
                return;
            }
            loadValuesInto(series, scalarComponent(record, path), offset, count, path, destination);
        }

        double scalarUnitSI(Record& record, std::string const& path)
        {
            return checkedUnitSI(scalarComponent(record, path), path, &record);
        }

        void checkUnitDimension(
            Record const& record,
            std::string const& path,
            std::array<double, 7> const& expected)
        {
            if(!record.containsAttribute("unitDimension"))
                return;
            auto const dimension = record.unitDimension();
            for(std::size_t i = 0; i < expected.size(); ++i)
                if(std::abs(dimension[i] - expected[i]) > 0.0)
                    fail(path + ": unitDimension does not match the openPMD convention");
        }

        bool isMacroWeighted(Record const& record, bool fallback)
        {
            if(!record.containsAttribute("macroWeighted"))
                return fallback;
            return record.getAttribute("macroWeighted").get<double>() != 0.0;
        }

        double weightingPower(Record const& record, double fallback)
        {
            if(!record.containsAttribute("weightingPower"))
                return fallback;
            return record.getAttribute("weightingPower").get<double>();
        }

        /*! Read one component range into ``destination``.
         *
         * The typed buffer returned by openPMD is converted straight into the
         * destination and released before the next component is read.  A chunked
         * read therefore peaks at one particle array per component instead of a
         * copy of every component held until the bunch is assembled.
         */
        void loadValuesInto(
            Series& series,
            RecordComponent& component,
            std::size_t offset,
            std::size_t count,
            std::string const& path,
            std::vector<double>& destination)
        {
            if(destination.size() != count)
                destination.resize(count);
            if(component.constant())
            {
                if(!component.containsAttribute("value"))
                    fail(path + ": constant component without a 'value' attribute");
                std::fill(destination.begin(), destination.end(), component.getAttribute("value").get<double>());
                return;
            }
            auto const extent = component.getExtent();
            std::size_t total = 1;
            for(auto const e : extent)
                total *= static_cast<std::size_t>(e);
            if(total < count)
                fail(path + ": dataset holds fewer values than the position extent");
            // openPMD defers non-constant reads; flush() executes the queued task.
            auto const datatype = component.getDatatype();
            openPMD::Offset const chunkOffset{static_cast<openPMD::Offset::value_type>(offset)};
            openPMD::Extent const requested{count};
            auto const convert = [&](auto const& raw) {
                for(std::size_t i = 0; i < count; ++i)
                    destination[i] = static_cast<double>(raw.get()[i]);
            };
#define PICBUNCH_LOAD(type)                                                                                      \
    {                                                                                                            \
        auto raw = component.loadChunk<type>(chunkOffset, requested);                                                 \
        series.flush();                                                                                          \
        convert(raw);                                                                                            \
        return;                                                                                                  \
    }
            switch(datatype)
            {
            case Datatype::DOUBLE:
                PICBUNCH_LOAD(double)
            case Datatype::FLOAT:
                PICBUNCH_LOAD(float)
            case Datatype::INT:
                PICBUNCH_LOAD(int)
            case Datatype::UINT:
                PICBUNCH_LOAD(unsigned int)
            case Datatype::LONG:
                PICBUNCH_LOAD(long)
            case Datatype::ULONG:
                PICBUNCH_LOAD(unsigned long)
            case Datatype::LONGLONG:
                PICBUNCH_LOAD(long long)
            case Datatype::ULONGLONG:
                PICBUNCH_LOAD(unsigned long long)
            default:
                fail(path + ": unsupported dataset datatype " + openPMD::datatypeToString(datatype));
            }
#undef PICBUNCH_LOAD
        }

        std::array<double, 3> readComponentUnits(Record& record, std::string const& prefix)
        {
            std::array<double, 3> units{};
            for(std::size_t a = 0; a < 3; ++a)
            {
                std::string const componentName = std::string(1, static_cast<char>('x' + a));
                std::string const path = prefix + "/" + componentName;
                requireRecord(record, componentName, prefix);
                units[a] = checkedUnitSI(record[componentName], path, &record);
            }
            return units;
        }

        std::pair<Iteration&, std::uint64_t> selectIteration(
            Series& series,
            std::optional<std::uint64_t> wanted)
        {
            if(series.iterations.empty())
                fail("the series contains no iterations");
            if(wanted)
            {
                auto it = series.iterations.find(*wanted);
                if(it == series.iterations.end())
                    fail("the series has no iteration " + std::to_string(*wanted));
                return {it->second, *wanted};
            }
            auto it = series.iterations.begin();
            return {it->second, static_cast<std::uint64_t>(it->first)};
        }

        std::array<double, 3> meshGeometryVector(
            std::array<std::size_t, 3> const& labelIndex,
            std::vector<double> const& values,
            std::array<double, 3> const& fallback)
        {
            if(values.size() != 3u)
                return fallback;
            std::array<double, 3> result{};
            for(std::size_t a = 0; a < 3; ++a)
                result[a] = values[labelIndex[a]];
            return result;
        }
    } // namespace

    OpenPMDInfo openPMDInfo()
    {
        OpenPMDInfo info;
        info.version = openPMD::getVersion();
#if defined(openPMD_HAVE_HDF5) && openPMD_HAVE_HDF5
        info.hdf5 = true;
#else
        info.hdf5 = false;
#endif
        return info;
    }

    namespace
    {
        /*! Fill ``result`` with one particle range.
         *
         * ``result`` may own buffers from an earlier call; ``Bunch::resize``
         * reuses their capacity and every particle field is overwritten, so a
         * chunked reader keeps a fixed pair of buffers instead of allocating a
         * full bunch per chunk.
         */
        void readBunchImpl(
            std::string const& fileName,
            BunchSelection const& selection,
            std::size_t offset,
            std::size_t count,
            BunchData& result)
        {
        Series series(fileName, openPMD::Access::READ_ONLY);
        auto [iteration, iterationIndex] = selectIteration(series, selection.iteration);
        result.iteration = iterationIndex;
        try
        {
            result.time = iteration.time<double>();
        }
        catch(std::exception const&)
        {
            result.time = 0.0;
        }
        try
        {
            result.timeUnitSI = iteration.timeUnitSI();
        }
        catch(std::exception const&)
        {
            result.timeUnitSI = 1.0;
        }

        auto& particles = iteration.particles;
        if(particles.empty())
            fail("the selected iteration contains no particle species");
        ParticleSpecies* speciesPtr = nullptr;
        std::string speciesName;
        if(selection.speciesName)
        {
            auto it = particles.find(*selection.speciesName);
            if(it == particles.end())
                fail("particle species '" + *selection.speciesName + "' is absent");
            speciesPtr = &it->second;
            speciesName = it->first;
        }
        else
        {
            std::size_t const speciesIndex = selection.speciesIndex.value_or(0);
            auto speciesIt = particles.begin();
            for(std::size_t i = 0; i < speciesIndex && speciesIt != particles.end(); ++i)
                ++speciesIt;
            if(speciesIt == particles.end())
                fail("particle species index " + std::to_string(speciesIndex) + " is out of range");
            speciesPtr = &speciesIt->second;
            speciesName = speciesIt->first;
        }
        ParticleSpecies& species = *speciesPtr;

        if(!species.contains("position"))
            fail("species '" + speciesName + "' has no 'position' record");
        Record& position = species["position"];
        checkUnitDimension(position, "position", {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

        requireRecord(position, "x", "position");
        auto const referenceExtent = position["x"].getExtent();
        if(referenceExtent.empty() || referenceExtent[0] == 0u)
            fail("position/x has no particle extent");
        std::size_t const totalCount = static_cast<std::size_t>(referenceExtent[0]);
        std::size_t const chunkCount = (count == 0u) ? (totalCount - offset) : count;
        if(offset > totalCount || chunkCount > totalCount - offset)
            fail("requested particle range is outside the species extent");

        // The bunch arrays are the only per-chunk particle storage kept alive.
        // Every component is read into one reusable scratch vector, converted,
        // and released before the next component, so the reader peak is the
        // bunch itself plus one raw component instead of a full untransformed
        // copy of all nine particle arrays.
        Bunch& bunch = result.bunch;
        bunch.resize(chunkCount);
        std::vector<double> scratch;
        std::vector<double> scratchOffset;

        auto const positionUnits = readComponentUnits(position, "position");
        Record* offsetRecord = nullptr;
        std::array<double, 3> offsetUnits{0.0, 0.0, 0.0};
        if(species.contains("positionOffset"))
        {
            offsetRecord = &species["positionOffset"];
            offsetUnits = readComponentUnits(*offsetRecord, "positionOffset");
        }
        std::array<std::vector<double>*, 3> coordinates{&bunch.x, &bunch.y, &bunch.z};
        for(std::size_t a = 0; a < 3; ++a)
        {
            std::string const componentName = std::string(1, static_cast<char>('x' + a));
            std::string const path = "position/" + componentName;
            requireRecord(position, componentName, "position");
            loadValuesInto(series, position[componentName], offset, chunkCount, path, scratch);
            std::vector<double>* offsetValues = nullptr;
            if(offsetRecord != nullptr)
            {
                requireRecord(*offsetRecord, componentName, "positionOffset");
                loadValuesInto(
                    series,
                    (*offsetRecord)[componentName],
                    offset,
                    chunkCount,
                    "positionOffset/" + componentName,
                    scratchOffset);
                offsetValues = &scratchOffset;
            }
            std::vector<double>& destination = *coordinates[a];
            for(std::size_t p = 0; p < chunkCount; ++p)
                destination[p] = scratch[p] * positionUnits[a]
                    + (offsetValues != nullptr ? (*offsetValues)[p] * offsetUnits[a] : 0.0);
        }

        if(!species.contains("momentum"))
            fail("species '" + speciesName + "' has no 'momentum' record");
        Record& momentum = species["momentum"];
        checkUnitDimension(momentum, "momentum", {1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0});
        auto const momentumUnits = readComponentUnits(momentum, "momentum");
        // openPMD standard default for macroWeighted is false; the momentum is
        // then a per-particle value and the weightingPower must not be applied.
        bool const momentumMacro = isMacroWeighted(momentum, false);
        double const momentumPower = weightingPower(momentum, 0.0);
        if(momentumMacro && !momentum.containsAttribute("weightingPower"))
            std::cerr << "picBunch openPMD reader: species '" << speciesName
                      << "': momentum has macroWeighted=true but no 'weightingPower';"
                      << " applying the standard default weightingPower=0\n";

        if(!species.contains("weighting"))
            fail("species '" + speciesName + "' has no 'weighting' record");
        Record& weightingRecord = species["weighting"];
        checkUnitDimension(weightingRecord, "weighting", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        double const weightUnitSI = scalarUnitSI(weightingRecord, "weighting");
        loadScalarValuesInto(series, weightingRecord, offset, chunkCount, "weighting", bunch.weight);
        for(std::size_t p = 0; p < chunkCount; ++p)
            bunch.weight[p] *= weightUnitSI;

        bool const chargeDefaulted = !species.contains("charge");
        if(chargeDefaulted)
            std::cerr << "picBunch openPMD reader: species '" << speciesName
                      << "' has no 'charge' record; applying the default q = -e ("
                      << -kElementaryCharge << " C) per macro particle\n";
        if(!chargeDefaulted)
        {
            Record& chargeRecord = species["charge"];
            checkUnitDimension(chargeRecord, "charge", {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0});
            double const chargeUnitSI = scalarUnitSI(chargeRecord, "charge");
            loadScalarValuesInto(series, chargeRecord, offset, chunkCount, "charge", bunch.charge);
            for(std::size_t p = 0; p < chunkCount; ++p)
                bunch.charge[p] *= chargeUnitSI;
        }
        else
        {
            std::fill(bunch.charge.begin(), bunch.charge.end(), -kElementaryCharge);
        }

        if(species.contains("mass"))
        {
            Record& massRecord = species["mass"];
            checkUnitDimension(massRecord, "mass", {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0});
            double const massUnitSI = scalarUnitSI(massRecord, "mass");
            loadScalarValuesInto(series, massRecord, offset, chunkCount, "mass", bunch.restMass);
            for(std::size_t p = 0; p < chunkCount; ++p)
                bunch.restMass[p] *= massUnitSI;
        }
        else
        {
            // The buffer may be reused from an earlier chunk; a species without
            // a mass record must not inherit those values.
            std::fill(bunch.restMass.begin(), bunch.restMass.end(), kElectronMass);
        }
        for(std::size_t p = 0; p < chunkCount; ++p)
            if(!(bunch.restMass[p] > 0.0))
                bunch.restMass[p] = kElectronMass;

        // The momentum denominator is per particle and, for a macro-weighted
        // record, needs the weighting power.  Compute it once and reuse it for
        // the three components; the non-macro default keeps the historical
        // mass * c * 1.0 ordering exactly.
        std::vector<double> momentumDenominator;
        if(momentumMacro)
        {
            momentumDenominator.resize(chunkCount);
            for(std::size_t p = 0; p < chunkCount; ++p)
                momentumDenominator[p]
                    = bunch.restMass[p] * kSpeedOfLight * std::pow(bunch.weight[p], momentumPower);
        }
        auto const denominatorOf = [&](std::size_t p) {
            return momentumMacro ? momentumDenominator[p] : bunch.restMass[p] * kSpeedOfLight;
        };
        std::array<std::vector<double>*, 3> momenta{&bunch.ux, &bunch.uy, &bunch.uz};
        for(std::size_t a = 0; a < 3; ++a)
        {
            std::string const componentName = std::string(1, static_cast<char>('x' + a));
            std::string const path = "momentum/" + componentName;
            requireRecord(momentum, componentName, "momentum");
            loadValuesInto(series, momentum[componentName], offset, chunkCount, path, scratch);
            std::vector<double>& destination = *momenta[a];
            for(std::size_t p = 0; p < chunkCount; ++p)
            {
                double const denominator = denominatorOf(p);
                destination[p]
                    = (denominator != 0.0) ? scratch[p] * momentumUnits[a] / denominator : 0.0;
            }
        }
        }
    } // namespace

    BunchData readBunch(std::string const& fileName, std::size_t speciesIndex)
    {
        BunchSelection selection;
        selection.speciesIndex = speciesIndex;
        BunchData result;
        readBunchImpl(fileName, selection, 0u, 0u, result);
        return result;
    }

    BunchData readBunch(std::string const& fileName, BunchSelection const& selection)
    {
        BunchData result;
        readBunchImpl(fileName, selection, 0u, 0u, result);
        return result;
    }

    std::size_t readParticleCount(std::string const& fileName, BunchSelection const& selection)
    {
        Series series(fileName, openPMD::Access::READ_ONLY);
        auto [iteration, iterationIndex] = selectIteration(series, selection.iteration);
        (void)iterationIndex;
        auto& particles = iteration.particles;
        if(particles.empty())
            fail("the selected iteration contains no particle species");
        ParticleSpecies* speciesPtr = nullptr;
        if(selection.speciesName)
        {
            auto it = particles.find(*selection.speciesName);
            if(it == particles.end())
                fail("particle species '" + *selection.speciesName + "' is absent");
            speciesPtr = &it->second;
        }
        else
        {
            std::size_t const speciesIndex = selection.speciesIndex.value_or(0);
            auto speciesIt = particles.begin();
            for(std::size_t i = 0; i < speciesIndex && speciesIt != particles.end(); ++i)
                ++speciesIt;
            if(speciesIt == particles.end())
                fail("particle species index " + std::to_string(speciesIndex) + " is out of range");
            speciesPtr = &speciesIt->second;
        }
        ParticleSpecies& species = *speciesPtr;
        if(!species.contains("position") || !species["position"].contains("x"))
            fail("species has no position/x record");
        auto const extent = species["position"]["x"].getExtent();
        if(extent.empty() || extent[0] == 0u)
            fail("position/x has no particle extent");
        return static_cast<std::size_t>(extent[0]);
    }

    BunchData readBunchRange(
        std::string const& fileName,
        BunchSelection const& selection,
        std::size_t offset,
        std::size_t count)
    {
        BunchData result;
        readBunchImpl(fileName, selection, offset, count, result);
        return result;
    }

    void readBunchRangeInto(
        std::string const& fileName,
        BunchSelection const& selection,
        std::size_t offset,
        std::size_t count,
        BunchData& destination)
    {
        readBunchImpl(fileName, selection, offset, count, destination);
    }

    MeshGeometry readMeshGeometry(
        std::string const& fileName,
        std::string const& meshName,
        std::size_t meshIndex,
        std::optional<std::uint64_t> iterationIndex)
    {
        MeshGeometry geometry;
        Series series(fileName, openPMD::Access::READ_ONLY);
        auto [iteration, iterationIndexResolved] = selectIteration(series, iterationIndex);
        geometry.iteration = iterationIndexResolved;
        try
        {
            geometry.time = iteration.time<double>();
        }
        catch(std::exception const&)
        {
            geometry.time = 0.0;
        }
        try
        {
            geometry.timeUnitSI = iteration.timeUnitSI();
        }
        catch(std::exception const&)
        {
            geometry.timeUnitSI = 1.0;
        }

        auto& meshes = iteration.meshes;
        if(meshes.empty())
            throw MeshUnavailable("the first iteration contains no mesh records");
        Mesh* selected = nullptr;
        if(meshes.contains(meshName))
            selected = &meshes.at(meshName);
        else
        {
            auto it = meshes.begin();
            for(std::size_t i = 0; i < meshIndex && it != meshes.end(); ++i)
                ++it;
            if(it == meshes.end())
                throw MeshUnavailable(
                    "mesh record '" + meshName + "' is absent and mesh index is out of range");
            selected = &it->second;
        }

        std::vector<std::string> labels;
        try
        {
            labels = selected->axisLabels();
        }
        catch(std::exception const&)
        {
            labels.clear();
        }
        if(labels.size() != 3u)
            labels = {"x", "y", "z"};
        std::array<std::size_t, 3> labelIndex{0, 1, 2};
        for(std::size_t a = 0; a < 3; ++a)
        {
            std::string const wanted(1, static_cast<char>('x' + a));
            auto const it = std::find(labels.begin(), labels.end(), wanted);
            if(it == labels.end())
                fail("mesh '" + meshName + "': axisLabels is not a permutation of x, y, z");
            labelIndex[a] = static_cast<std::size_t>(std::distance(labels.begin(), it));
        }
        for(std::size_t a = 0; a < 3; ++a)
            geometry.axisLabels[a] = labels[labelIndex[a]];

        std::vector<double> spacing;
        std::vector<double> globalOffset;
        try
        {
            spacing = selected->gridSpacing<double>();
        }
        catch(std::exception const&)
        {
            spacing.clear();
        }
        try
        {
            globalOffset = selected->gridGlobalOffset();
        }
        catch(std::exception const&)
        {
            globalOffset.clear();
        }

        // gridUnitSI is mandatory for a physical mesh.  Silently assuming 1 is what
        // made a PIConGPU mesh (gridUnitSI = 1e-6) rescale the whole grid.
        if(!selected->containsAttribute("gridUnitSI"))
            fail("mesh '" + meshName + "': required attribute 'gridUnitSI' is missing");
        std::vector<double> gridUnitSI;
        try
        {
            gridUnitSI = selected->gridUnitSIPerDimension();
        }
        catch(std::exception const&)
        {
            fail("mesh '" + meshName + "': cannot read the gridUnitSI attribute");
        }
        if(gridUnitSI.size() == 1u)
            gridUnitSI = {gridUnitSI[0], gridUnitSI[0], gridUnitSI[0]};
        if(gridUnitSI.size() != 3u)
            fail("mesh '" + meshName + "': gridUnitSI must be a scalar or a 3-vector");
        for(double unit : gridUnitSI)
            if(!std::isfinite(unit) || !(unit > 0.0))
                fail("mesh '" + meshName + "': every gridUnitSI entry must be finite and > 0");

        auto const rawSpacing = meshGeometryVector(labelIndex, spacing, {1.0, 1.0, 1.0});
        geometry.gridUnitSI = meshGeometryVector(labelIndex, gridUnitSI, {1.0, 1.0, 1.0});
        // openPMD defines gridSpacing and gridGlobalOffset in units of gridUnitSI.
        // MeshGeometry is documented in physical SI units, so convert here.
        auto const rawOffset = meshGeometryVector(labelIndex, globalOffset, {0.0, 0.0, 0.0});
        for(std::size_t a = 0; a < 3; ++a)
        {
            geometry.gridSpacing[a] = rawSpacing[a] * geometry.gridUnitSI[a];
            geometry.gridGlobalOffset[a] = rawOffset[a] * geometry.gridUnitSI[a];
        }
        try
        {
            geometry.geometry = selected->geometryString();
        }
        catch(std::exception const&)
        {
            geometry.geometry = "cartesian";
        }
        try
        {
            geometry.dataOrder = (selected->dataOrder() == openPMD::Mesh::DataOrder::C) ? "C" : "F";
        }
        catch(std::exception const&)
        {
            geometry.dataOrder = "C";
        }

        // Shape from the first component, in stored axis-label order.
        auto it = selected->begin();
        if(it == selected->end())
            fail("mesh '" + meshName + "' has no record components");
        auto const extent = it->second.getExtent();
        if(extent.size() != 3u)
            fail("mesh '" + meshName + "': expected a 3-D dataset");
        for(std::size_t a = 0; a < 3; ++a)
            geometry.shape[a] = static_cast<std::size_t>(extent[labelIndex[a]]);
        return geometry;
    }

} // namespace picBunch
