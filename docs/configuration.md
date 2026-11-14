# Placement-interface configuration

The documented interface is:

```bash
picBunch --input bunch.h5 --destination checkpoints --config params.toml
```

`--input` is an openPMD file with the bunch species, `--destination` is the
PIConGPU checkpoint directory to create, and `--config` is a TOML file. No unit
or conversion value is accepted on the command line. Every required key below
must be present; a missing key fails with a message that names it. A malformed
TOML file also fails. Unknown keys are ignored with a warning on stderr.

The field pipeline is unchanged: `[accuracy]` selects the free-space
convolution target and the optional exact near shell.

## TOML schema

### `[picongpu]`

The table carries the independent PIConGPU deck inputs. Copy the values from
the deck that will restart the checkpoint (`simulation.param`,
`speciesConstants.param`, `physicalConstants.param`). picBunch derives every SI
unit factor exactly as `include/picongpu/unitless/simulation.unitless`:

```text
n_mpp       = base_density * dx*dy*dz / typical_particles_per_cell
unit_time   = delta_t
unit_speed  = speed_of_light
unit_length = unit_time * unit_speed
unit_mass   = base_mass * n_mpp
unit_charge = -base_charge * n_mpp
unit_energy = unit_mass * unit_length^2 / unit_time^2
unit_efield = 1 / (unit_time^2 / unit_mass / unit_length * unit_charge)
unit_bfield = unit_mass / (unit_time * unit_charge)
```

| key | required | meaning |
| --- | --- | --- |
| `delta_t` | yes | `SI::DELTA_T_SI`, the time step in seconds. |
| `cell_size` | yes | `SI::CELL_WIDTH/HEIGHT/DEPTH_SI`, three positive metres `[dx, dy, dz]`. May instead be given once as `domain.cell_size` (deprecated); if both are present they must be equal. |
| `base_mass` | no | `SI::BASE_MASS_SI` in kg. Default `9.1093837139e-31` (2022 CODATA electron mass). |
| `base_charge` | no | `SI::BASE_CHARGE_SI` in C, **signed**. Default `-1.602176634e-19`. |
| `base_density` | yes | `SI::BASE_DENSITY_SI` in m^-3. It sets the macro-particle weight and therefore the mass and charge units. |
| `typical_particles_per_cell` | no | `TYPICAL_PARTICLES_PER_CELL`. Default `2`. |
| `speed_of_light` | no | `SI::SPEED_OF_LIGHT_SI` in m/s. Default `2.99792458e8`. Override only if the deck does. |
| `step` | yes | Checkpoint step in `[0, 999999]`; also the file suffix. |
| `precision` | no | `"float32"` (default) or `"float64"`, applied uniformly to every field component and every particle property record. `positionOffset` stays int32 and `id` stays uint64. |
| `absorber` | no | Field-absorber kind of the deck that restarts the checkpoint, mirroring the deck's `--fieldAbsorber`: `"none"` (default), `"exponential"` or `"pml"`. An unknown value is rejected. See [Boundary conditions](#boundary-conditions). |
| `species` | yes | Species to read from `--input` and write to the checkpoint. A string (the historical form, one entry) or an ordered array of strings. Each entry is read, placed with the same `[placement]` block, field-computed and summed. |
| `output_species` | no | Optional array of output group names, positionally aligned with `species`. Its length must equal `species`; omitted or empty means "the output name equals the input name" for every entry. |
| `empty_species` | no | Optional array of deck species written as checkpoint-only groups with zero particles. They are **never** read from `--input`. A name may not also appear in `species` or `output_species`. |

The values must match the restart deck, not the deck that produced the input
bunch. PIConGPU restart ignores the particle `unitSI` and interprets momentum in
`unit_mass * unit_speed` and mass/charge in `unit_mass`/`unit_charge`, so a
mismatch scales every record.

#### Multiple species

`species` is an ordered list. Entry `i` is read with `BunchSelection{species[i]}`,
placed, and field-deposited on its own; the six field components are then summed.
The mesh field is linear in charge, so the sum is the field of the union of all
species. `base_charge`/`base_mass` are the physics source for the whole list;
there is no per-entry override.

`output_species` renames positionally: `species = ["b"]` with
`output_species = ["e"]` reads input group `b` and writes checkpoint group `e`.
The raw input group is still copied into the checkpoint (the "every input
species is copied" contract). With a BP5 input the raw group cannot be copied;
a renamed entry then acts as a move.

`empty_species` lists deck species that must exist in the checkpoint with zero
particles but have no input data. They get the standard records as empty
datasets, one domain-covering patch row with `numParticles = 0`, and no ids. A
name that is both generated and empty is a configuration error.

A generated species whose input group is absent from `--input` is an error; the
error names the input group.

The old manual `unit_length`, `unit_e`, `unit_b`, `unit_charge`, `unit_mass` and
`unit_speed` keys are no longer part of the schema. They are still accepted as
explicit overrides for backward compatibility: an override wins for that one
factor, always prints a deprecation warning, and is then validated. The default
path is pure derivation. `dt` is accepted as a deprecated alias for `delta_t`.

Particle momenta are written in PIConGPU internal units, i.e. the SI momentum
divided by `unit_mass * unit_speed`. The reader returns `u = p/(m c)` for the
species rest mass, so the writer converts with
`u * m_species * c / (unit_mass * unit_speed)`.

### `[domain]`

| key | required | meaning |
| --- | --- | --- |
| `cells` | yes | Three positive integers, the PIConGPU domain cell counts `[nx, ny, nz]`. |
| `cell_size` | yes | Three positive metres, the physical cell size `[dx, dy, dz]`. |
| `origin` | no | Three metres, the physical position of domain cell `(0,0,0)`. Default `[0,0,0]`. |

The checkpoint fields cover exactly `cells` nodes on this domain.

### `[placement]`

`position` is a vector ordered `[x, y, z]`, following PIConGPU notation: the
first component is `x`, then `y`, then `z`. It is the offset from the PIConGPU
domain origin to the bunch centroid. `offset_unit` selects the interpretation:
`"metres"` means metres from the domain origin, and `"fraction"` means a
fraction of the domain extent (one full domain length is `1.0`).

| key | required | meaning |
| --- | --- | --- |
| `center` | no | `"centroid"` (default) or `"none"`. `"centroid"` shifts the charge-weighted centroid to `position`. `"none"` keeps the absolute input coordinates and applies no shift, which is what a non-centred bunch needs. |
| `position` | yes (with `center = "centroid"`) | Three components `[x, y, z]`: the offset from the domain origin to the bunch centroid, in the unit selected by `offset_unit`. Optional when `center = "none"`; a non-zero value then warns that it is ignored. |
| `offset_unit` | yes (with `center = "centroid"`) | `"metres"` or `"fraction"` of the domain extent. Optional when `center = "none"`. |
| `patch` | no | `"bbox"` (default) or `"domain"`. |
| `patch_count` | no | Positive integer (default `1`): split the generated species along `z` into this many rank-overlap regions. |

The old `left`, `front` and `top` scalar keys are no longer accepted; the error
names `placement.position`. Use the vector.

Fractional values must be in `[0, 1]` in centroid mode. The placement uses the
charge-weighted centroid `sum(q w r) / sum(q w)`. The bounding-box centre is
printed in the log for comparison but does not drive the placement.

`center = "none"` reproduces the absolute input coordinates exactly (the writer
receives positions bit-identical to the reader output) and still checks domain
containment: a bunch that leaves the domain aborts with the same axis/distance
message. This is the mode for a bunch whose input centroid is not the domain
centre, e.g. a beam at `(45.4, 109.0, 45.4) um`.

A shifted bunch that would leave the domain is rejected, never clipped. The
error names the offending axis (`x`, `y` or `z`) and the distance outside the
domain in metres and cells.

`patch = "bbox"` writes one patch covering only the shifted bunch bounding box.
It is validated for single-rank restarts. `patch = "domain"` writes one patch
covering the whole domain and is the safe choice for a multi-rank restart; its
`positionOffset` still indexes the global cells, so particle data is unchanged.

`patch_count = N` splits the generated species along `z` into `N` rank-overlap
regions. Neighbouring regions overlap by one cell, particle ownership stays
disjoint, and the writer reorders the records so each patch has a contiguous
range and cumulative `numParticlesOffset`. Copied species keep the patches of
the input checkpoint. `patch_count = 1` is the previous layout.

### `[accuracy]`

| key | required | meaning |
| --- | --- | --- |
| `tolerance` | yes | Relative free-space convolution target; positive, floor `1e-14`. It sizes the FFT padding and velocity boxes, and the near-field remainder order: each decade below `1e-3` raises the Gauss order, capped at 32 nodes per axis, and pairs whose requirement exceeds the cap (including every gamma >= 100 oblique source) are routed to the exact evaluator. The default `1e-3` keeps the order-5 remainder. It does not bound the far-tail error or the checkpoint precision. |
| `near_shell` | no | `"none"` (default), `"fast"` (accurate near-source evaluator), or `"exact"` (Lorentz-contracted adaptive reference; about `1e-4` axial and `1e-3` oblique at gamma 10, about `2%` at the closest axial node). |
| `near_shell_radius` | no | Shell radius in PIC cells; `0` (default) derives it from `tolerance`. |
| `near_shell_exact_cells` | no | Exact s-space routing of the fast near field. `-2` (default) is automatic: the observer-containing TSC support cell is exact, the rest keeps the order-5 remainder. `-1` forces legacy quadrature, `0` selects the observer cell only, `1` adds the six face-adjacent cells, `2` selects every in-support cell. A tolerance `<= 1e-4` makes the automatic policy full-support exact, and a tile with gamma `>= 100` is always full exact. The observer-cell default is neutral to 9% cheaper than base on the CPU and costs 0-1% on the GPU on-node workload; the GPU spread 12-cell cloud pays up to 30% because the per-source exact branch diverges whole warps. In-support L2 accuracy improves roughly twofold (static `1.80e-3 -> 5.74e-4`, axial gamma 10 E `9.35e-3 -> 3.14e-3`, B `3.55e-2 -> 2.35e-3`). The full-support mode is the accuracy reference for in-support points; it costs `0.93-1.19x` base on the CPU, `1.0-1.04x` on the GPU on-node workload, and `0.90-2.17x` on the GPU spread cloud. |
| `near_shell_tile` | no | Tile edge in nodes. `0` (default) selects the automatic rule: a sampled near-set estimate picks `2` when its (source, node) pair count is below `1/(1.25*1.05)` of the tile-4 count, otherwise `4`. The `1.25` is the measured tile-4/tile-2 pair throughput on the V100 dense campaign and `1.05` is a safety margin. An explicit positive value always wins; the compute stage logs the chosen edge. On the fast/fast-gpu path an explicit tile is bounded: its half-width must not exceed `near_shell_radius` (so the edge must not exceed twice the radius), and the edge must not exceed the documented maximum of `4` nodes. |

The fast/fast-gpu path is guarded against unphysical near-shell outputs. `tile = 0`
is the safe default and keeps the automatic rule. An explicit tile whose half-width
exceeds the shell radius (tile edge above twice `near_shell_radius`) is rejected, as
is an explicit tile above `4` nodes: the automatic rule and every validated campaign
cover only tiles `2` and `4`, and tile `8` at the default radius `3` is the first
measured spike case (35x the stable peak). A larger tile is therefore rejected at
configuration time instead of producing a spike. In addition, after the correction
the fast path compares the corrected peak E against the uncorrected mesh field
and aborts if the correction raises that peak by more than `3x`; the message names
the radius, tile and fast order. The underlying boundary treatment of the near shell
is under investigation, so these guards turn a silent spike into a diagnosable
failure rather than claiming a numerical fix.

The accuracy target is not an end-to-end guarantee. The README error-budget table
separates mesh/deposit, velocity binning, near quadrature, far tail and output
precision. For a moving bunch with angular spread no near-shell mode is uniformly
accurate in-support: `none` is the safest default, `fast` is validated for stationary
and near-axial motion, and `exact` is an adaptive reference that is accurate except for
a measured few-percent residual at the closest axial node.

### `[checkpoint]`

| key | required | meaning |
| --- | --- | --- |
| `backend` | no | `"auto"` (default), `"adios2"` or `"hdf5"`. `auto` derives the backend from `--destination`: a `.bp5` directory selects the ADIOS2/BP5 backend and a `.h5` directory selects the HDF5 backend. Both backends are written through the same openPMD-Series writer, so both produce the same tree: one full-shape `(Nz, Ny, Nx)` record component per field component, identical geometry and unit attributes, and the same particle records. PIConGPU selects the container with `--checkpoint.openPMD.ext bp5` (or `h5`). |

## Boundary conditions

`[picongpu] absorber` selects the boundary-condition contract of the emitted
checkpoint. It mirrors the deck's `--fieldAbsorber` for the boundary kinds and
must be set to `pml` whenever the restarting deck uses the convolutional PML:

| `absorber` | checkpoint content | restarting deck |
| --- | --- | --- |
| `none` (default) | E, B, idProvider, particles only. | Periodic boundaries (`--periodic 1 1 1`), or any deck that loads no auxiliary field state. |
| `exponential` | Identical to `none`. The exponential absorber stores no field state; PIConGPU reads its parameters back from the compiled `fieldAbsorber.param` on restart. | `--fieldAbsorber exponential` decks. |
| `pml` | The previous tree plus the twelve zero-filled `Convolutional PML` slab records: six components `E` and `B`, each with a `neg` and a `pos` slab record (`E_xneg`…`E_zpos`, and the same six for `B`); the full per-record layout is in [openPMD backend selection](openpmd-backends.md#field-absorber-key). | `--fieldAbsorber pml` decks. |

The PML records are all zero. picBunch computes a free-space field and has no
PML state, and the PML auxiliary fields of a deck that has not run are exactly
zero (PIConGPU resets the slab buffers at startup, see `Field::reset` in
`include/picongpu/fields/absorber/pml/Field.hpp`). A checkpoint written right
after initialization therefore stores zeros in these slabs, and restoring them
is the faithful initial condition. The slabs describe the same grid as E and B;
only the deck's compiled PML parameters (`fieldAbsorber.param`) and boundary
choice determine how the slabs evolve after restart.

A restart of an open-boundary deck without these records fails with
`Key 'Convolutional PML E_yneg' does not exist (read-only)`. Workarounds for a
deck that does not need the PML: restart with periodic boundaries, or switch
the deck to the exponential absorber.

The boundary-condition matrix of the full placement interface (periodic,
exponential, PML) is documented in [PIConGPU restart checkpoints](picongpu-restart.md).

## Large domains under MPI

Run the configured path under `mpirun -n R` and keep `--fft-mode auto` (the
default). With more than one rank, auto selects the distributed z-slab FFT:
each rank owns one slab of the padded spectrum and one z-slab of the six output
components, and `makeGrid` scales its per-rank padded-node cap by `R * 8`.
`--fft-mode replicated` keeps the full field on every rank and does not scale
the cap. A domain whose padded box is above `2^28` nodes therefore needs at
least two ranks; it is rejected on one rank with the `limit 268435456 nodes`
message.

The rank count lowers the per-process memory, not the aggregate. The
distributed path holds three complex arrays over the padded box
(`3 * M^3 * 16 B` total across ranks), so the device, not the host, sets the
first large-domain limit. The checkpoint writer is collective: every rank
opens the openPMD Series on `MPI_COMM_WORLD`, creates the same full-shape
datasets, and stores only its own z-slab at its global offset. No rank gathers
or allocates the full field, and the checkpoint tree is independent of the rank
count. Particle records and `checkpoints.txt` stay on rank 0.

## Chunked field computation is not supported

picBunch does **not** ship a chunked or slab-tiled field-computation path, and
there is no `accuracy.field_slab_planes` configuration key. A checkpoint is
written only from a fully materialised field; the peak device memory is
therefore the full padded FFT box plus the six output components (about 370
bytes per grid cell on the reference small case, 400x160x440 cells at 2e-7 m;
measured 10.5 GiB on the GV100). Host RSS on the same case was 1.6 GiB.

The slab approach was implemented and measured, then removed. Two wrapper bugs
were fixed first: a wrong `idist` that aliased every z-row, and a batched 2-D
plan that never transformed z. The slab field still disagreed with the untiled
reference: the relative E error was 0.45-0.67, and the spectral-curl B path
overflowed to about `1e307` because the half-spectrum Hermitian symmetry is
broken on the DC and Nyquist planes. That is a correctness gap, not a tuning
problem, so the feature is not shippable.

The correct future approach is a decomposition that preserves the global z
spectrum: either a distributed FFT along z with the correct global halo, or an
overlap-save scheme whose per-slab transform is embedded in the full-plane
spectrum rather than an independent small transform. Any such path must prove
all six components against the untiled field to a stated tolerance before it is
accepted, and it must keep one full-shape record component per field component
in the checkpoint.

## Particle patch and records

The writer creates one particle patch for the species. With
`patch = "bbox"`, its `offset` is the lowest cell index of the shifted bunch
bounding box relative to the domain origin and its `extent` covers the bounding
box. With `patch = "domain"`, `offset` is zero and `extent` is the full domain.
`numParticles` is the bunch size and `numParticlesOffset` is zero.

Records are in PIConGPU internal units:

- `position`: fractional in-cell coordinate in `[0, 1)`, `unitSI` is the cell
  size, stored at the configured precision. `positionOffset`: integer cell
  index, stored as int32.
- `momentum`: `p / (unit_mass * unit_speed)`, stored at the configured
  precision.
- `weighting`: statistical weight with `macroWeighted = 1` and
  `weightingPower = 1`, stored at the configured precision.
- `charge`: `q / unit_charge` with `unitSI = unit_charge`, read from the input
  bunch and converted with the derived `unit_charge`.
- `mass`: `m / unit_mass` with `unitSI = unit_mass`, read from the input bunch
  and converted with the derived `unit_mass`. When the input has no positive
  rest mass the derived `unit_mass` is used.
- `id`: uint64 global index.

A non-positive derived `unit_charge`/`unit_mass` is rejected at configuration
parse time and again when the corresponding record is written, so the writer
never invents a conversion.

The checkpoint also contains `fields/picongpu_idProvider`, every other species
present in the input iteration, and `checkpoints.txt`.

## Input position conventions

The reader evaluates positions verbatim as
`x = position * unitSI_position + positionOffset * unitSI_offset`. Two
`positionOffset` conventions are therefore accepted and give the same absolute
positions:

- integer cell indices with `positionOffset.unitSI` equal to the cell size (the
  convention PIConGPU and this writer use);
- offsets already in metres with `positionOffset.unitSI = 1`.

No conversion or inference is applied. A non-standard `unitSI` that equals the
cell size is numerically correct because the stored offset is an integer cell
index in that case.

The input backend is selected from the path: a `.h5` file uses HDF5, and a
`.bp`/`.bp5` directory (for example `particles_000000.bp5`) uses ADIOS2/BP5.
A BP5 directory needs an openPMD-api built with ADIOS2; without it the reader
fails with a message naming the directory and the missing backend.

## Precision

`precision` selects the storage type of all six field components and every
particle property record: `position`, `momentum`, `weighting`, `charge` and
`mass`. It applies uniformly; the writer never mixes dtypes inside one
checkpoint. `positionOffset` stays int32 and `id` stays uint64.

`float32` is the default. It matches a stock single-precision PIConGPU, whose
`float_X` is float32, so a default checkpoint restarts without conversion.

The fractional in-cell `position` is stored in `[0, 1)`, not in metres. A
float32 fraction has a resolution of about `6e-8` (half an ulp near 1). At a
cell size of `1e-7` m that is about `6e-15` m of absolute position error, and a
float32 PIConGPU pusher cannot represent anything finer. `float64` reduces the
fraction error to about `1e-16` (about `1e-23` m at the same cell size), so the
extra precision only pays off when the consuming PIConGPU also stores
`float_X` in double precision.

`float64` is therefore only usable with a PIConGPU build that converts
checkpoint dtypes. Stock PIConGPU cannot: it aborts a restart from a double
particle record with

```
Type conversion during chunk loading not yet implemented! Data: DOUBLE; Load as: FLOAT
```

When `precision = "float64"` is selected, the writer prints this restriction on
stderr and stores the same text in the checkpoint root attribute
`picBunchPrecisionWarning`. The checkpoint also roughly doubles in size.

## Dependency pin

TOML parsing uses [toml++](https://github.com/marzer/tomlplusplus) v3.4.0 at
commit `30172438cee64926dc41fdd9c11fb3ba5b2ba9de`, fetched by CMake
`FetchContent`. The pin is recorded in `CMakeLists.txt`.

## Known deviation: native ROCm cleanup call

The code is otherwise pure alpaka, but one native ROCm vendor API call remains
in `src/Compute.cpp`: a HIP-only `rocfft_cleanup()` teardown guard, compiled only
for the ROCm/HIP backend. This `rocfft_cleanup()` is the only native vendor API
call in the tree; all other device work goes through alpaka. It works around a
ROCm 7.2 static-destruction-order bug in librocfft's process-global HIP-stream
pool, which otherwise segfaults at exit and regresses the `picBunch.Io` test.
The teardown is process-global and must run once per process, after all FFT plans
have been destroyed and while the alpaka queue and HIP device are still alive.
The guard is declared before the plans in its scope and destroyed after them, so
it enforces that ordering. The native call itself is wrapped in a thread-safe
`std::once_flag`/`std::call_once`, so repeated `computeField` calls cannot
destroy the global pool more than once while every call still waits for its
queue before leaving the scope. This is a documented constraint deviation, not a
hidden one: pending an alpakaVendor `finalize()` API, the guard stays. Once
`alpaka::fft::onHost::finalize()` (idempotent, once-guarded) exists in
alpakaVendor, it replaces the guard. The open engineering items, including this
teardown, are tracked in [known-limitations.md](known-limitations.md).

The crash history, the proposed API and the validation steps are in
[rocm-fft-teardown.md](rocm-fft-teardown.md).

## Example

A complete example is committed as `examples/picbunch.toml`.

## Legacy CLI

The old flags remain available as an advanced path. They are not the documented
interface. In particular, the unit flags (`--picongpu-unit-length`,
`--picongpu-unit-e`, `--picongpu-unit-b`) and `--restart-format picongpu` are
advanced only; the documented `--input/--destination/--config` usage takes every
unit and conversion from the TOML file and rejects no unit value because it
accepts none on the command line.
