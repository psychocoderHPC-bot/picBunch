# PIConGPU restart checkpoints

This note records what `--restart-format picongpu` writes, what PIConGPU accepts,
and which remaining differences are known to be non-blocking. Measurements are
from the acceptance run on `rosi_v100` (Tesla V100, CUDA 12.6, sm_70) with
PIConGPU PR #24 at `2baa2566`.

## Verified end state

A picBunch-written periodic checkpoint restarts PIConGPU with exit code 0. A
validation-only PIConGPU build (`maxwellSolver::None`, deck parameter only) loads
the checkpoint and writes the loaded state at step 1. All six components arrive
bitwise unchanged:

```
E/x E/y E/z B/x B/y B/z: bitwise_equal=True, max|diff|=0,
worst_rel_SI=0, max_ulp=0
```

The restart skipped the deck initializer (`initialization time: 1sec 833msec`
instead of about 116 s), and `restart/restart.log` ends with
`full simulation time: 1sec 916msec` and `run_rc=0`.

For an open-boundary LWFA checkpoint, the writer now copies the PML slabs from
the input checkpoint. The rerun on `rosi_v100` used the real checkpoint
`w7b-20261102/lwfa/gt-run/checkpoints/out_000002.h5` (638976 electrons, eight
`Convolutional PML` groups with six datasets each) and the same reduced deck
(64x128x64, `--periodic 0 0 1`). picBunch copied all eight groups and the restart
exited 0 with

```
initialization time:  3sec 986msec = 3.986 sec
100 % =        2 | time elapsed:                    0msec | avg time per step:   0msec
full simulation time:  4sec  31msec = 4.031 sec
```

The copied PML datasets are float32 and bitwise equal to the source. The main E
and B records are recomputed from the particles and stay float32.

## PML handling

The input checkpoint is scanned at `data/<inputIteration>/fields`. Every link
whose name starts with `Convolutional PML ` is copied with `H5Ocopy`, so the
component values, `position` and `unitSI` attributes survive unchanged. The
writer converts a non-float32 component to float32, rejects a component whose
extent differs from the output grid, and replaces the record geometry with the
main-field geometry. It never synthesizes PML state.

When the input checkpoint has no PML records, the writer keeps its previous
behaviour and writes only E, B and `picongpu_idProvider`. An open-boundary
restart then fails because PIConGPU loads the PML slabs unconditionally, with
`Key 'Convolutional PML E_xneg' does not exist`. Use an input checkpoint that
contains PML records for such a deck, or restart with periodic boundaries.

For a deck whose PML has never run, `[picongpu] absorber = "pml"` in the
configuration is the other option: the writer adds the eight
`Convolutional PML E/B_{xneg,xpos,yneg,ypos,zneg,zpos}` slab records with the
zero initial state, so the restart deck's PML loads from rest. The layout (six
cross components `xy,xz,yx,yz,zx,zy`, full `(Nz,Ny,Nx)` extent, main-field
geometry, parent E/B units and Yee positions) is fixed by the PIConGPU
checkpoint contract and is documented in
[Boundary conditions](configuration.md#boundary-conditions).

## Particle species and the id range

A PIConGPU restart loads every compiled species that is present in
`FileCheckpointParticles`. The writer therefore copies **every** species group
from the input iteration, not only the species used for the field calculation.
`--picongpu-species` only requires that the named species exists; it never drops
the other groups. A requested name that is absent fails with
`input has no particle species '<name>'`, and the checkpoint file is not left
behind.

`fields/picongpu_idProvider` holds one global id range for the whole
checkpoint. The writer stores `startId = 0` and
`nextId = max(options.nextId, total particles copied over all species)`, with
`maxNumProc = 1`. This matches PIConGPU's single-rank checkpoint state and is
consistent with the particles in the file. The id range does not depend on the
number of species or on the size of the bunch species alone.

The checkpoint has a single 1x1x1 id-provider mesh, which is the single-rank
contract of this tool. PIConGPU reads the mesh with the modulo rank mapping, so
restarting the same checkpoint on more than one rank would give every rank the
same `startId`; newly created particles could then overlap. Use a single rank
for a restart from a picBunch checkpoint.

## Non-blocking structural differences

A raw HDF5 diff against a PIConGPU-written checkpoint leaves the following
differences. PIConGPU restart does not read them, so they do not block a restart.

- Iteration metadata: `cell_width`, `cell_height`, `cell_depth`, `eps0`,
  `mue0`, `particleBoundary`, `particleBoundaryParameters` are absent. The
  `[picongpu]` TOML path now writes the full PIConGPU unit set
  (`unit_time`, `unit_speed`, `unit_length`, `unit_mass`, `unit_charge`,
  `unit_energy`, `unit_efield`, `unit_bfield`); the legacy CLI path writes
  only `unit_length`, `unit_efield` and `unit_bfield`.
- Iteration defaults: `dt` is stored as the deck `delta_t` and `timeUnitSI` is
  `1.0` (numerically equal to `unit_time`); PIConGPU writes `timeUnitSI =
  unit_time`. Restart ignores both.
- `RNGProvider3XorMin` is absent. PIConGPU reinitializes the RNG and logs a
  binning warning.
- Fields group metadata: `chargeCorrection`, `fieldBoundary` and
  `fieldBoundaryParameters` are absent; `BackgroundFieldIncluded` is uint8 0
  instead of a boolean true.
- `gridSpacing` is written as float64 (same value to float32 precision).
- String attributes are variable-length instead of fixed-length; values match.
- Species group extras such as `cellSizeSI`, `grid`, `numParticles`,
  `currentDeposition`, `particleInterpolation`, `particlePush` and
  `particleSmoothing` are absent or differ. PIConGPU derives them.
- `charge`, `mass` and `id` particle records are removed on purpose because
  PIConGPU derives them from the species flags.
- Particle `unitSI` metadata differs between the input and output; restart does
  not read it.

The reader accepts both openPMD 1.x plain scalar datasets and openPMD 2.0
scalar-record groups. The group form uses either a `value` child or a child
named after the record, with attributes on the group, the child, or both. This
is what lets picBunch read a PIConGPU checkpoint directly.
