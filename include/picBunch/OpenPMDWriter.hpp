// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/FarTail.hpp"
#include "picBunch/Types.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string>

namespace picBunch
{
    /*! Queue and host-memory contract shared by every writer entry point.
     *
     * All inputs are host ``std::vector`` values.  A caller that produced them
     * on an alpaka device must call ``alpaka::onHost::wait(queue)`` before it
     * invokes a writer function.  The writer owns no queue and cannot wait for
     * a producer it does not own.  The compute stage in ``Compute.cpp`` follows
     * this order: it waits on its queue, then copies the device results into the
     * host vectors handed to this layer.  openPMD's deferred I/O is flushed
     * inside the writer before a loaded pointer is read or the series closes.
     */

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

    /** Write the bunch-box field into a global domain without allocating it.
     *
     * ``field`` holds the computed field on a small grid.  ``globalGridSize``
     * and ``globalGridOrigin`` describe the openPMD output domain; the field
     * grid must be aligned to it (``makeGrid`` does this when
     * ``ComputeOptions::globalGridSize`` is set) and contained in it.  Every
     * global node outside the field grid is written as exact ``0.0``.
     *
     * Writing zero is only justified when the field grid extends to the
     * tolerance-derived tail margin from ``deriveFieldTailMargin``, or when the
     * global domain ends before that margin (then no node between the source
     * and the domain edge is zeroed).  The CLI enforces this; a direct caller
     * must do the same instead of using this function to truncate silently.
     *
     * The global dataset is never materialised.  Each rank owns one contiguous
     * z-slab, subdivided into write tiles, so the extra memory is one tile per
     * rank.  With an MPI-enabled openPMD build all ranks write the same series
     * collectively through ``storeChunkRaw``; without it each rank writes its
     * own ``<fileName>.rankNNNN`` file and a warning is printed.  A single rank
     * still uses the same tiled path, which is what keeps the single-rank memory
     * bounded too.
     *
     * Metadata (extent, spacing, origin, unitSI, Yee ``position``) is written
     * for the global domain and is identical for every rank count.
     */
    void writeFieldToDomain(
        Field const& field,
        std::string const& fileName,
        FieldWriteOptions const& options,
        Idx3 const& globalGridSize,
        Vec3 const& globalGridOrigin,
        FarTail const* tail = nullptr);


    /** Configuration of a PIConGPU checkpoint-restart writer.

     * Everything physical is explicit; the writer never infers a unit.  The
     * target layout is the one PIConGPU's ``RestartFieldLoader`` consumes: an
     * HDF5 openPMD 1.1 file-based iteration, float32 fields in PIConGPU internal
     * units, and the input particle species copied with its attributes intact.
     */
    /** One species reconstructed from a bunch at its placed position.
     *
     * The writer creates the particle group instead of copying it from the
     * input.  Positions are SI metres, momenta are dimensionless
     * ``p/(m c)``, weights are the statistical weights and ``restMass`` is the
     * SI rest mass per macro particle.  The conversion to PIConGPU internal
     * units happens in the writer with the supplied units.
     *
     * ``momentum`` follows the PIConGPU openPMD convention: it is the
     * macro-particle momentum ``u * restMass * c * weight`` with
     * ``macroWeighted = true`` and ``weightingPower = 1``.  PIConGPU's
     * ``speciesAttributes.unitless`` defines the momentum attribute that way
     * and its restart loader assigns the stored record value directly, while
     * its consumers recover the per-particle ``u`` by dividing by ``weight``.
     */
    struct GeneratedParticles
    {
        std::string species;
        Bunch bunch; //!< positions already shifted to the placement
        Vec3 domainOrigin{0.0, 0.0, 0.0}; //!< SI metres
        Vec3 cellSize{1.0, 1.0, 1.0}; //!< SI metres
        IVec3 patchOffset{0, 0, 0}; //!< cells from domain origin
        UVec3 patchExtent{1, 1, 1};
        //! Patch regions.  Empty keeps the single ``patchOffset``/``patchExtent``
        //! layout.  With more than one, particles are reordered so each patch
        //! owns a contiguous range and ``numParticlesOffset`` is cumulative.
        std::vector<ParticlePatch> patches;
        //! PIConGPU momentum unit ``unit_mass * unit_speed`` in SI.
        double unitMassSpeed = 1.0;
        //! PIConGPU charge unit in Coulomb (``sim.unit.charge()``).  Required
        //! for the ``charge`` record; a non-positive value is an error.
        double unitCharge = 0.0;
        //! PIConGPU mass unit in kilograms (``sim.unit.mass()``).  Required for
        //! the ``mass`` record; a non-positive value is an error.
        double unitMass = 0.0;
        //! Rest mass used when a particle has no positive ``restMass`` entry.
        double fallbackMass = 0.0;
    };

    struct PicOnGPUWriteOptions
    {
        //! Checkpoint step; also the iteration index and the file name suffix.
        std::uint64_t step = 0;
        //! Directory that receives ``<prefix>_<step:06d>.h5`` and ``checkpoints.txt``.
        std::string directory = "checkpoints";
        //! File prefix; the iterationFormat is ``<prefix>_%06T``.
        std::string prefix = "checkpoint";
        //! openPMD series or concrete HDF5 file the particles are copied from.
        std::string inputFile;
        //! Iteration index used to locate the source particle group.
        std::uint64_t inputIteration = 0;
        /*! Particle species group name to require in the input iteration.
         * Empty disables the check.  Every species present is copied; a
         * multi-species checkpoint needs all of them for a PIConGPU restart.
         */
        std::string species;
        double time = 0.0;
        double timeUnitSI = 1.0;
        //! Stored ``dt`` attribute in units of ``timeUnitSI`` (not converted).
        double dt = 0.0;
        //! PIConGPU length unit in metres (``sim.unit.length()``).
        double unitLength = 1.0;
        //! PIConGPU electric-field unit in V/m (``sim.unit.eField()``).
        double unitEField = 1.0;
        //! PIConGPU magnetic-field unit in T (``sim.unit.bField()``).
        double unitBField = 1.0;
        //! PIConGPU time unit in seconds (``sim.unit.time()``).  Zero leaves the
        //! attribute unwritten (legacy CLI path, which has no deck inputs).
        double unitTime = 0.0;
        //! PIConGPU speed unit in m/s (``sim.unit.speed()``).
        double unitSpeed = 0.0;
        //! PIConGPU energy unit in J (``sim.unit.energy()``).
        double unitEnergy = 0.0;
        //! PIConGPU mass unit in kg (``sim.unit.mass()``).
        double unitMass = 0.0;
        //! PIConGPU charge unit in C (``sim.unit.charge()``).
        double unitCharge = 0.0;
        //! Lower bound for ``fields/picongpu_idProvider/nextId``.  The writer
        //! stores at least the total number of particles copied across all
        //! species, so it is never written smaller.
        std::uint64_t nextId = 0;
        //! When set, this species is reconstructed from particles instead of
        //! copied raw.  Every other input species is still copied.
        std::optional<GeneratedParticles> generatedParticles;
        /*! Write every field component and every particle property record in
         *  float64 instead of the default float32.  ``positionOffset`` stays
         *  int32 and ``id`` stays uint64.  A stock PIConGPU cannot load double
         *  particle records; the writer prints a warning and stores it in the
         *  ``picBunchPrecisionWarning`` root attribute.
         */
        bool float64 = false;
    };

    /** Write one PIConGPU checkpoint-restart iteration.

     * Emitted tree (all verified against a real PIConGPU checkpoint):
     *  - ``<directory>/<prefix>_<step:06d>.h5`` and ``<directory>/checkpoints.txt``;
     *  - series attributes ``openPMD``, ``iterationEncoding=fileBased``,
     *    ``iterationFormat=<prefix>_%06T``, ``meshesPath=fields/`` and
     *    ``picongpuIOVersionMajor=5`` / ``Minor=0``;
     *  - ``data/<step>/{iteration,sim_slides,dt,time,timeUnitSI}``;
     *  - ``fields/E/{x,y,z}`` and ``fields/B/{x,y,z}`` as float32 by default
     *    (float64 when requested) with extent ``(Nz,Ny,Nx)``, stored as
     *    ``field/unit`` in PIConGPU internal units;
     *  - ``gridSpacing*gridUnitSI`` recover metres, with both written in the
     *    ``z,y,x`` storage order;
     *  - ``fields/picongpu_idProvider/{startId,nextId}`` uint64; one global
     *    range for the whole checkpoint with ``startId = 0`` and
     *    ``nextId >= total particles`` over all copied species;
     *  - ``particles/<species>`` for every species in the input iteration,
     *    copied from its input group and normalized to the configured precision,
     *    including ``particlePatches`` when present.

     * Record attributes survive the copy; the numeric datasets are rewritten
     * to the configured precision so a checkpoint never mixes dtypes.  Missing
     * patch metadata is reported through a ``picBunchRestartWarning`` root
     * attribute and a message on ``stderr``; PIConGPU may then reject the
     * restart.  ``float64`` additionally records ``picBunchPrecisionWarning``
     * and prints the same warning on ``stderr``.

     * @throws std::runtime_error or std::invalid_argument on missing units,
     *         a missing input particle group, or an HDF5 failure.
     */
    void writePicOnGPUCheckpoint(Field const& field, PicOnGPUWriteOptions const& options);

} // namespace picBunch