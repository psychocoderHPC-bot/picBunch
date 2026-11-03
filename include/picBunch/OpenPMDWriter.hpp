// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/Types.hpp"

#include <cstdint>
#include <string>

namespace picBunch
{
    /** Metadata of one written field iteration.
     *
     * The electrical field is written in V/m and the magnetic field in T, both
     * with ``unitSI = 1.0``.  ``time``/``timeUnitSI`` and ``timeOffset`` are
     * written on the iteration, ``iteration`` selects the openPMD iteration
     * index.
     */
    struct FieldWriteOptions
    {
        std::uint64_t iteration = 0;
        double time = 0.0;
        double timeUnitSI = 1.0;
        double timeOffset = 0.0;
    };

    /** Write ``E`` and ``B`` as openPMD / HDF5 mesh records.
     *
     * Layout and metadata:
     *  - six records ``E_x,E_y,E_z,B_x,B_y,B_z``, dataset order ``z,y,x``
     *    (x fastest), ``dataOrder = C``, ``geometry = cartesian``;
     *  - ``gridSpacing`` in metres and ``gridGlobalOffset`` from ``field.grid``,
     *    both in the stored ``z,y,x`` axis order;
     *  - ``gridUnitSI = 1.0`` because the values are already SI (V/m and T);
     *  - ``unitDimension`` = (1,1,-3,-1) for E and (0,1,-2,-1) for B;
     *  - per-component ``position`` set to the Yee staggering
     *    ``Ex(.5,0,0) Ey(0,.5,0) Ez(0,0,.5) Bx(0,.5,.5) By(.5,0,.5) Bz(.5,.5,0)``
     *    (expressed in cell-size units of the stored grid).
     *
     * PIConGPU restart currently ignores the mesh ``unitSI`` (verified defect),
     * so a PIConGPU consumer must convert with ``sim.unit.eField()`` /
     * ``sim.unit.bField()``; the values are deliberately written in SI.
     *
     * @throws std::runtime_error if openPMD cannot create the series or if the
     *         field data is not sized ``grid.size()``.
     */
    void writeField(
        Field const& field,
        std::string const& fileName,
        FieldWriteOptions const& options = FieldWriteOptions{});

} // namespace picBunch
