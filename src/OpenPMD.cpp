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

        std::vector<double> loadValues(RecordComponent& component, std::size_t count, std::string const& path)
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
            auto const datatype = component.getDatatype();
            switch(datatype)
            {
            case Datatype::DOUBLE:
                return copyValues(component.loadChunk<double>(), count);
            case Datatype::FLOAT:
                return copyValues(component.loadChunk<float>(), count);
            case Datatype::INT:
                return copyValues(component.loadChunk<int>(), count);
            case Datatype::UINT:
                return copyValues(component.loadChunk<unsigned int>(), count);
            case Datatype::LONG:
                return copyValues(component.loadChunk<long>(), count);
            case Datatype::ULONG:
                return copyValues(component.loadChunk<unsigned long>(), count);
            case Datatype::LONGLONG:
                return copyValues(component.loadChunk<long long>(), count);
            case Datatype::ULONGLONG:
                return copyValues(component.loadChunk<unsigned long long>(), count);
            default:
                fail(path + ": unsupported dataset datatype " + openPMD::datatypeToString(datatype));
            }
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
                result[a] = loadValues(record[componentName], count, path);
            }
            return result;
        }

        std::pair<Iteration&, std::uint64_t> firstIteration(Series& series)
        {
            if(series.iterations.empty())
                fail("the series contains no iterations");
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

    BunchData readBunch(std::string const& fileName, std::size_t speciesIndex)
    {
        BunchData result;
        Series series(fileName, openPMD::Access::READ_LINEAR);
        auto [iteration, iterationIndex] = firstIteration(series);
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
            fail("the first iteration contains no particle species");
        auto speciesIt = particles.begin();
        for(std::size_t i = 0; i < speciesIndex && speciesIt != particles.end(); ++i)
            ++speciesIt;
        if(speciesIt == particles.end())
            fail("particle species index " + std::to_string(speciesIndex) + " is out of range");
        ParticleSpecies& species = speciesIt->second;
        std::string const speciesName = speciesIt->first;

        if(!species.contains("position"))
            fail("species '" + speciesName + "' has no 'position' record");
        Record& position = species["position"];
        checkUnitDimension(position, "position", {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

        requireRecord(position, "x", "position");
        auto const referenceExtent = position["x"].getExtent();
        if(referenceExtent.empty() || referenceExtent[0] == 0u)
            fail("position/x has no particle extent");
        std::size_t const count = static_cast<std::size_t>(referenceExtent[0]);

        auto const positionValues = readVectorSeries(position, count, "position");
        auto const positionUnits = readComponentUnits(position, "position");

        std::array<std::vector<double>, 3> offsetValues;
        std::array<double, 3> offsetUnits{0.0, 0.0, 0.0};
        if(species.contains("positionOffset"))
        {
            Record& offset = species["positionOffset"];
            offsetValues = readVectorSeries(offset, count, "positionOffset");
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
        auto const momentumValues = readVectorSeries(momentum, count, "momentum");
        auto const momentumUnits = readComponentUnits(momentum, "momentum");
        bool const momentumMacro = isMacroWeighted(momentum, true);
        double const momentumPower = weightingPower(momentum, 0.0);

        if(!species.contains("weighting"))
            fail("species '" + speciesName + "' has no 'weighting' record");
        Record& weightingRecord = species["weighting"];
        checkUnitDimension(weightingRecord, "weighting", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        RecordComponent& weightingComponent = weightingRecord;
        double const weightUnitSI = checkedUnitSI(weightingComponent, "weighting");
        auto const rawWeight = loadValues(weightingComponent, count, "weighting");

        std::vector<double> chargePerParticle;
        if(species.contains("charge"))
        {
            Record& chargeRecord = species["charge"];
            checkUnitDimension(chargeRecord, "charge", {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0});
            RecordComponent& chargeComponent = chargeRecord;
            double const chargeUnitSI = checkedUnitSI(chargeComponent, "charge");
            auto const rawCharge = loadValues(chargeComponent, count, "charge");
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
            auto const rawMass = loadValues(massComponent, count, "mass");
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

    MeshGeometry readMeshGeometry(std::string const& fileName, std::string const& meshName, std::size_t meshIndex)
    {
        MeshGeometry geometry;
        Series series(fileName, openPMD::Access::READ_LINEAR);
        auto [iteration, iterationIndex] = firstIteration(series);
        geometry.iteration = iterationIndex;
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
            fail("the first iteration contains no mesh records");
        Mesh* selected = nullptr;
        if(meshes.contains(meshName))
            selected = &meshes.at(meshName);
        else
        {
            auto it = meshes.begin();
            for(std::size_t i = 0; i < meshIndex && it != meshes.end(); ++i)
                ++it;
            if(it == meshes.end())
                fail("mesh record '" + meshName + "' is absent and mesh index is out of range");
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
        std::vector<double> gridUnitSI;
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
        try
        {
            gridUnitSI = selected->gridUnitSIPerDimension();
        }
        catch(std::exception const&)
        {
            gridUnitSI.clear();
        }
        geometry.gridSpacing = meshGeometryVector(labelIndex, spacing, {1.0, 1.0, 1.0});
        geometry.gridGlobalOffset = meshGeometryVector(labelIndex, globalOffset, {0.0, 0.0, 0.0});
        geometry.gridUnitSI = meshGeometryVector(labelIndex, gridUnitSI, {1.0, 1.0, 1.0});
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
