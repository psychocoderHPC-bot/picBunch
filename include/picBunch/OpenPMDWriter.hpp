// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/FarTail.hpp"
#include "picBunch/Types.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
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
    /*! Checkpoint output backend.
     *
     * The checkpoint is an openPMD Series.  ``Hdf5`` writes a single
     * ``<prefix>_%06T.h5`` file (the openPMD HDF5 backend, the historical
     * default and the most widely tested restart path).  ``Adios2`` writes a
     * ``<prefix>_%06T.bp5`` directory (the openPMD ADIOS2/BP5 backend).  Both
     * produce an identical openPMD tree (one dataset per field component,
     * identical attributes) that PIConGPU's openPMD restart reads by file
     * extension: set ``openPMD.ext = bp5`` on the PIConGPU side to restart from
     * an ADIOS2 checkpoint.
     *
     * ``Auto`` derives the backend from the output path: a ``.bp5`` directory
     * selects ADIOS2, a ``.h5`` file selects HDF5, and any other destination
     * falls back to HDF5 (the openPMD default extension).
     */
    enum class CheckpointBackend
    {
        Auto,
        Hdf5,
        Adios2
    };

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
        //! Output particle group name written into the checkpoint.
        std::string species;
        //! Input group name this entry is reconstructed from.  Empty means
        //! ``species`` (the historical single-species behaviour).  Ignored when
        //! ``empty`` is set.
        std::string inputSpecies;
        /*! Write a checkpoint-only group with zero particles.
         *
         * The entry is never read from the input file and ``bunch`` stays empty.
         * It carries the six standard records as empty datasets and one
         * domain-covering ``particlePatches`` row with ``numParticles = 0``.
         */
        bool empty = false;
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
        /*! Input particle species groups that must exist in the input
         * iteration.  Empty disables the check.  Every other species present is
         * copied; a multi-species checkpoint needs all of them for a PIConGPU
         * restart.  A generated species consumes its ``inputSpecies`` group
         * instead of copying it.
         */
        std::vector<std::string> requiredInputSpecies;
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
        /*! Species reconstructed from particles instead of copied raw.
         *
         * Entries with ``inputSpecies`` set consume that input group (it is not
         * copied).  Entries with ``empty`` set are checkpoint-only.  Every
         * other input species is still copied.
         */
        std::vector<GeneratedParticles> generatedSpecies;
        /*! Write every field component and every particle property record in
         *  float64 instead of the default float32.  ``positionOffset`` stays
         *  int32 and ``id`` stays uint64.  A stock PIConGPU cannot load double
         *  particle records; the writer prints a warning and stores it in the
         *  ``picBunchPrecisionWarning`` root attribute.
         */
        bool float64 = false;
        /*! Field-absorber kind of the PIConGPU deck that restarts this
          * checkpoint: ``"none"`` (default), ``"exponential"`` or ``"pml"``.
          *
          * Only ``"pml"`` changes the emitted tree: it adds the eight
          * convolutional-PML slab records a ``--fieldAbsorber pml`` deck loads
          * on restart.  See ``writePicOnGPUCheckpointOpenPMD`` for the layout.
          * ``"exponential"`` is accepted because an exponential deck never
          * stores auxiliary field state; a PIConGPU restart reads it back from
          * the compiled parameters, so no checkpoint record is needed.
          */
        std::string absorber = "none";
        /*! Checkpoint output backend (see ``CheckpointBackend``).
         *
         * ``Auto`` (the default) derives the backend from the output path and
         * keeps the historical ``.h5`` output for a destination that is neither
         * a ``.bp5`` directory nor a ``.h5`` file.  ``Hdf5`` forces the ``.h5``
         * file; ``Adios2`` forces the ``.bp5`` directory.  The value is resolved
         * once and both the file name and the openPMD backend used for the
         * write follow it.
         */
        CheckpointBackend backend = CheckpointBackend::Auto;
    };

    //! Append ``step`` to ``<directory>/checkpoints.txt`` (idempotent).
    void appendCheckpointStep(std::filesystem::path const& directory, std::uint64_t step);

    /** openPMD-Series checkpoint writer (HDF5 and ADIOS2/BP5 backends).
     *
     * Writes the PIConGPU-restart tree through the backend-agnostic openPMD
     * C++ API, so one code path produces an identical openPMD layout for both
     * ``.h5`` (HDF5 backend) and ``.bp5`` (ADIOS2 backend).  The backend is
     * resolved from ``options.backend`` and the output path (see
     * ``CheckpointBackend``).  Every field component is one full-shape
     * ``(Nz, Ny, Nx)`` record component.
     *
     * @throws std::invalid_argument or std::runtime_error on missing units,
     *         a field that does not match the grid, or an openPMD failure.
     */
    void writePicOnGPUCheckpointOpenPMD(
        Field const& field,
        PicOnGPUWriteOptions const& options);

} // namespace picBunch
