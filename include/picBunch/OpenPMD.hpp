// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/Types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace picBunch
{
    /** Build-time/runtime openPMD-api information.
     *
     * The reader and writer stages require the HDF5 backend; CMake rejects a
     * configuration whose openPMD-api was built without it.
     */
    struct OpenPMDInfo
    {
        std::string version;
        bool hdf5 = false;
    };

    OpenPMDInfo openPMDInfo();

    /** Read one particle species of the first iteration of an openPMD series.
     *
     * Positions use the openPMD convention ``x = position*unitSI + offset*unitSI_offset``
     * with no inference from unit equality.  Momenta become ``u = p/(m c)``; weights
     * carry ``macroWeighted``/``weightingPower``; the charge is read in SI Coulomb.
     * If the file has no ``charge`` record the frozen contract ``q_p = -w_p e`` is
     * applied.  If it has no ``mass`` record the electron mass is used.
     *
     * Missing required records (``position``, ``momentum``, ``weighting``) raise
     * ``std::runtime_error`` with the record path in the message.
     *
     * @param fileName openPMD series path (the ``%T`` pattern or a concrete file)
     * @param speciesIndex zero-based index of the particle species in the iteration
     */
    BunchData readBunch(std::string const& fileName, std::size_t speciesIndex = 0);

    /** Read the geometry metadata of a mesh record of the first iteration.
     *
     * Reads grid spacing, ``gridUnitSI``, ``gridGlobalOffset``, axis labels,
     * geometry, data order, and the iteration time/``timeUnitSI``.  The returned
     * vectors are in ``x, y, z`` order and the shape is the dataset extent mapped
     * to ``x, y, z`` from the stored axis-label order.
     *
     * @param fileName openPMD series path
     * @param meshName mesh record name, ``"E"`` by default (PIConGPU convention)
     * @param meshIndex zero-based index used when the requested name is absent
     * @throws std::runtime_error if the iteration has no mesh record
     */
    MeshGeometry readMeshGeometry(
        std::string const& fileName,
        std::string const& meshName = "E",
        std::size_t meshIndex = 0);

} // namespace picBunch
