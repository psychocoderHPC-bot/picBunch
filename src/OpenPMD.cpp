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

        double checkedUnitSI(RecordComponent const& component, std::string const& path)
        {
            double unit = 1.0;
            try
            {
                unit = component.unitSI();
            }
            catch(std::exception const&)
            {
                unit = 1.0;
            }
            if(!std::isfinite(unit) || !(unit > 0.0))
                fail(path + ": unitSI must be finite and > 0");
            return unit;
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

        template<typename T>
        std::vector<double> copyValues(std::shared_ptr<T> const& data, std::size_t count)
        {
            std::vector<double> result(count);
            for(std::size_t i = 0; i < count; ++i)
                result[i] = static_cast<double>(data.get()[i]);
            return result;
        }

        std::vector<double> loadValues(
            Series& series,
            RecordComponent& component,
            std::size_t count,
            std::string const& path)
        {
            if(component.constant())
            {
                if(!component.containsAttribute("value"))
                    fail(path + ": constant component without a 'value' attribute");
                return std::vector<double>(count, component.getAttribute("value").get<double>());
            }
            auto const extent = component.getExtent();
            std::size_t total = 1;
            for(auto const e : extent)
                total *= static_cast<std::size_t>(e);
            if(total < count)
                fail(path + ": dataset holds fewer values than the position extent");
            // openPMD defers non-constant reads; flush() executes the queued task.
            auto const datatype = component.getDatatype();
            openPMD::Offset const offset{0};
            openPMD::Extent const requested{count};
#define PICBUNCH_LOAD(type)                                                                                      \
    {                                                                                                            \
        auto raw = component.loadChunk<type>(offset, requested);                                                 \
        series.flush();                                                                                          \
        return copyValues(raw, count);                                                                           \
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
                units[a] = checkedUnitSI(record[componentName], path);
            }
            return units;
        }

        //! Read all three components of a vector record (one value per particle).
        std::array<std::vector<double>, 3> readVectorSeries(
            Series& series,
            Record& record,
            std::size_t count,
            std::string const& prefix)
        {
            std::array<std::vector<double>, 3> result;
            for(std::size_t a = 0; a < 3; ++a)
            {
                std::string const componentName = std::string(1, static_cast<char>('x' + a));
                std::string const path = prefix + "/" + componentName;
                requireRecord(record, componentName, prefix);
                result[a] = loadValues(series, record[componentName], count, path);
            }
            return result;
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
        BunchData readBunchImpl(std::string const& fileName, BunchSelection const& selection)
        {
        BunchData result;
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
        std::size_t const count = static_cast<std::size_t>(referenceExtent[0]);

        auto const positionValues = readVectorSeries(series, position, count, "position");
        auto const positionUnits = readComponentUnits(position, "position");

        std::array<std::vector<double>, 3> offsetValues;
        std::array<double, 3> offsetUnits{0.0, 0.0, 0.0};
        if(species.contains("positionOffset"))
        {
            Record& offset = species["positionOffset"];
            offsetValues = readVectorSeries(series, offset, count, "positionOffset");
            offsetUnits = readComponentUnits(offset, "positionOffset");
        }
        else
        {
            for(auto& component : offsetValues)
                component.assign(count, 0.0);
        }

        if(!species.contains("momentum"))
            fail("species '" + speciesName + "' has no 'momentum' record");
        Record& momentum = species["momentum"];
        checkUnitDimension(momentum, "momentum", {1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0});
        auto const momentumValues = readVectorSeries(series, momentum, count, "momentum");
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
        RecordComponent& weightingComponent = weightingRecord;
        double const weightUnitSI = checkedUnitSI(weightingComponent, "weighting");
        auto const rawWeight = loadValues(series, weightingComponent, count, "weighting");

        std::vector<double> chargePerParticle;
        bool const chargeDefaulted = !species.contains("charge");
        if(chargeDefaulted)
            std::cerr << "picBunch openPMD reader: species '" << speciesName
                      << "' has no 'charge' record; applying the default q = -e ("
                      << -kElementaryCharge << " C) per macro particle\n";
        if(!chargeDefaulted)
        {
            Record& chargeRecord = species["charge"];
            checkUnitDimension(chargeRecord, "charge", {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0});
            RecordComponent& chargeComponent = chargeRecord;
            double const chargeUnitSI = checkedUnitSI(chargeComponent, "charge");
            auto const rawCharge = loadValues(series, chargeComponent, count, "charge");
            chargePerParticle.resize(count);
            for(std::size_t p = 0; p < count; ++p)
                chargePerParticle[p] = rawCharge[p] * chargeUnitSI;
        }

        std::vector<double> restMass;
        if(species.contains("mass"))
        {
            Record& massRecord = species["mass"];
            checkUnitDimension(massRecord, "mass", {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0});
            RecordComponent& massComponent = massRecord;
            double const massUnitSI = checkedUnitSI(massComponent, "mass");
            auto const rawMass = loadValues(series, massComponent, count, "mass");
            restMass.resize(count);
            for(std::size_t p = 0; p < count; ++p)
                restMass[p] = rawMass[p] * massUnitSI;
        }

        Bunch& bunch = result.bunch;
        bunch.resize(count);
        for(std::size_t p = 0; p < count; ++p)
        {
            double const weight = rawWeight[p] * weightUnitSI;
            double const mass = (restMass.empty() || !(restMass[p] > 0.0)) ? kElectronMass : restMass[p];
            double const factor = momentumMacro ? std::pow(weight, momentumPower) : 1.0;
            double const denominator = mass * kSpeedOfLight * factor;

            bunch.x[p] = positionValues[0][p] * positionUnits[0] + offsetValues[0][p] * offsetUnits[0];
            bunch.y[p] = positionValues[1][p] * positionUnits[1] + offsetValues[1][p] * offsetUnits[1];
            bunch.z[p] = positionValues[2][p] * positionUnits[2] + offsetValues[2][p] * offsetUnits[2];
            bunch.ux[p] = (denominator != 0.0) ? momentumValues[0][p] * momentumUnits[0] / denominator : 0.0;
            bunch.uy[p] = (denominator != 0.0) ? momentumValues[1][p] * momentumUnits[1] / denominator : 0.0;
            bunch.uz[p] = (denominator != 0.0) ? momentumValues[2][p] * momentumUnits[2] / denominator : 0.0;
            bunch.weight[p] = weight;
            bunch.charge[p] = chargePerParticle.empty() ? -kElementaryCharge : chargePerParticle[p];
        }
        return result;
        }
    } // namespace

    BunchData readBunch(std::string const& fileName, std::size_t speciesIndex)
    {
        BunchSelection selection;
        selection.speciesIndex = speciesIndex;
        return readBunchImpl(fileName, selection);
    }

    BunchData readBunch(std::string const& fileName, BunchSelection const& selection)
    {
        return readBunchImpl(fileName, selection);
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
