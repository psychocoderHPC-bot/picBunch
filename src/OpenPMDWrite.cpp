// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// openPMD writer: the six Yee-staggered field components as scalar mesh
// records.  Values are SI (V/m, T), grid metadata is SI, and every component
// carries its Yee staggering through the openPMD ``position`` attribute.

#include "picBunch/OpenPMDWriter.hpp"

#include <openPMD/openPMD.hpp>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

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

    } // namespace

    void writeField(Field const& field, std::string const& fileName, FieldWriteOptions const& options)
    {
        if(field.data[0].size() != field.grid.size())
            throw std::runtime_error("writeField: component data does not match grid.size()");

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
                component.storeChunkRaw(field.data[spec.firstComponent + a].data(), offset, extent);
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

} // namespace picBunch
