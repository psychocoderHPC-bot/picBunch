// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/Types.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
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

    /** The input has no usable mesh record to read geometry from.
     *
     * Thrown only for "there is no mesh here" (empty mesh container or an
     * out-of-range mesh index).  A mesh that exists but carries invalid
     * metadata, such as a missing gridUnitSI, throws std::runtime_error so the
     * caller cannot mistake it for a benign absence.
     */
    class MeshUnavailable : public std::runtime_error
    {
    public:
        using std::runtime_error::runtime_error;
    };

    /** Read one particle species of the first iteration of an openPMD series.
     *
     * Positions use the openPMD convention ``x = position*unitSI + offset*unitSI_offset``
     * with no inference from unit equality.  Momenta become ``u = p/(m c)``; weights
     * carry ``macroWeighted``/``weightingPower`` (the openPMD standard default of
     * an absent ``macroWeighted`` is ``false``); the charge is read in SI Coulomb.
     * If the file has no ``charge`` record the frozen contract ``q_p = -e`` (per
     * macro particle) is applied and a warning naming the species is written to
     * ``stderr``.  A missing ``weightingPower`` with ``macroWeighted = true``
     * warns similarly and applies the standard default ``0``.  If it has no
     * ``mass`` record the electron mass is used.
     *
     * Missing required records (``position``, ``momentum``, ``weighting``) raise
     * ``std::runtime_error`` with the record path in the message.
     *
     * @param fileName openPMD series path (the ``%T`` pattern or a concrete file)
     * @param speciesIndex zero-based index of the particle species in the iteration
     */
    BunchData readBunch(std::string const& fileName, std::size_t speciesIndex = 0);

    /** Selector for :func:`readBunch`.
     *
     * Unset fields keep the defaults of the index-based overload (first
     * species, first iteration).  ``speciesName`` takes precedence over
     * ``speciesIndex`` when both are set.
     */
    struct BunchSelection
    {
        std::optional<std::size_t> speciesIndex;
        std::optional<std::string> speciesName;
        std::optional<std::uint64_t> iteration;
    };

    /** Read one particle species of a selected iteration by name or index. */
    BunchData readBunch(std::string const& fileName, BunchSelection const& selection);

    /** Total number of particles in the selected species. */
    std::size_t readParticleCount(std::string const& fileName, BunchSelection const& selection);

    /** Read a contiguous particle range of the selected species.
     *
     * ``offset + count`` must lie inside the species extent.  This is the
     * chunked-reader entry point; the units and record conventions are the
     * same as :func:`readBunch`.
     */
    BunchData readBunchRange(
        std::string const& fileName,
        BunchSelection const& selection,
        std::size_t offset,
        std::size_t count);

    /** Read the geometry metadata of a mesh record of the first iteration.
     *
     * Reads grid spacing, ``gridUnitSI``, ``gridGlobalOffset``, axis labels,
     * geometry, data order, and the iteration time/``timeUnitSI``.  The returned
     * vectors are in ``x, y, z`` order and the shape is the dataset extent mapped
     * to ``x, y, z`` from the stored axis-label order.  ``gridSpacing`` and
     * ``gridGlobalOffset`` are converted to physical SI by multiplying with
     * ``gridUnitSI``; the raw ``gridUnitSI`` values are returned alongside.
     * A missing or non-positive ``gridUnitSI`` is an error.
     *
     * @param fileName openPMD series path
     * @param meshName mesh record name, ``"E"`` by default (PIConGPU convention)
     * @param meshIndex zero-based index used when the requested name is absent
     * @throws MeshUnavailable if the iteration has no mesh record
     * @throws std::runtime_error if a mesh exists but its metadata is invalid
     */
    MeshGeometry readMeshGeometry(
        std::string const& fileName,
        std::string const& meshName = "E",
        std::size_t meshIndex = 0,
        std::optional<std::uint64_t> iterationIndex = std::nullopt);

} // namespace picBunch
