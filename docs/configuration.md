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

| key | required | meaning |
| --- | --- | --- |
| `unit_length` | yes | PIConGPU length unit in metres (`sim.unit.length()`). |
| `unit_e` | yes | PIConGPU electric-field unit in V/m (`sim.unit.eField()`). |
| `unit_b` | yes | PIConGPU magnetic-field unit in T (`sim.unit.bField()`). |
| `dt` | yes | Time step in seconds, stored in the checkpoint. |
| `step` | yes | Checkpoint step in `[0, 999999]`; also the file suffix. |
| `precision` | no | `"float32"` (default) or `"float64"`, applied uniformly to every field component and every particle property record. `positionOffset` stays int32 and `id` stays uint64. |
| `species` | yes | Species to read from `--input` and write to the checkpoint. |
| `unit_charge` | no | Charge unit in C. Default `1.602176634e-19`. |
| `unit_mass` | yes | PIConGPU mass unit in kg, i.e. `sim.unit.mass()`. It must match the deck; do not assume the electron mass. PIConGPU restart ignores the particle `unitSI` and interprets momentum in `sim.unit.mass()*unit_speed`, so a mismatch scales every momentum. |
| `unit_speed` | no | Speed unit in m/s. Default `299792458.0`. |

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

The operator names the axes by intent: `left` is `+x`, `front` is `+y`, and
`top` is `+z`. Each value is the offset from the PIConGPU domain origin to the
bunch centre.

| key | required | meaning |
| --- | --- | --- |
| `offset_unit` | yes | `"fraction"` of the domain extent or `"metres"`. |
| `left` | yes | Offset along `+x`. |
| `front` | yes | Offset along `+y`. |
| `top` | yes | Offset along `+z`. |
| `patch` | no | `"bbox"` (default) or `"domain"`. |
| `patch_count` | no | Positive integer (default `1`): split the generated species along `z` into this many rank-overlap regions. |

Fractional values must be in `[0, 1]`. The placement uses the charge-weighted
centroid `sum(q w r) / sum(q w)`. The bounding-box centre is printed in the log
for comparison but does not drive the placement.

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
| `tolerance` | yes | Relative free-space convolution target; positive, floor `1e-14`. It sizes the FFT padding and velocity boxes, not the near-source, far-tail or checkpoint-precision error. |
| `near_shell` | no | `"none"` (default), `"fast"` (accurate near-source evaluator), or `"exact"` (Lorentz-contracted adaptive reference; about `1e-4` axial and `1e-3` oblique at gamma 10, about `2%` at the closest axial node). |
| `near_shell_radius` | no | Shell radius in PIC cells; `0` (default) derives it from `tolerance`. |
| `near_shell_tile` | no | Tile edge in nodes; default `4`. |

The accuracy target is not an end-to-end guarantee. The README error-budget table
separates mesh/deposit, velocity binning, near quadrature, far tail and output
precision. For a moving bunch with angular spread no near-shell mode is uniformly
accurate in-support: `none` is the safest default, `fast` is validated for stationary
and near-axial motion, and `exact` is an adaptive reference that is accurate except for
a measured few-percent residual at the closest axial node.

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
  bunch and converted with `unit_charge`.
- `mass`: `m / unit_mass` with `unitSI = unit_mass`, read from the input bunch
  and converted with `unit_mass`. When the input has no positive rest mass the
  configured `unit_mass` is used.
- `id`: uint64 global index.

A missing or non-positive `unit_charge`/`unit_mass` factor is an error when the
corresponding record is written, so the writer never invents a conversion.

The checkpoint also contains `fields/picongpu_idProvider`, every other species
present in the input iteration, and `checkpoints.txt`.

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

## Example

A complete example is committed as `examples/picbunch.toml`.

## Legacy CLI

The old flags remain available as an advanced path. They are not the documented
interface. In particular, the unit flags (`--picongpu-unit-length`,
`--picongpu-unit-e`, `--picongpu-unit-b`) and `--restart-format picongpu` are
advanced only; the documented `--input/--destination/--config` usage takes every
unit and conversion from the TOML file and rejects no unit value because it
accepts none on the command line.
