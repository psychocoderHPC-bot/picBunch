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
| `precision` | no | `"float32"` (default) or `"float64"` for particle records and fields. |
| `species` | yes | Species to read from `--input` and write to the checkpoint. |
| `unit_charge` | no | Charge unit in C. Default `1.602176634e-19`. |
| `unit_mass` | no | Mass unit in kg. Default `9.1093837015e-31`. |
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

Fractional values must be in `[0, 1]`. The placement uses the charge-weighted
centroid `sum(q w r) / sum(q w)`. The bounding-box centre is printed in the log
for comparison but does not drive the placement. If the shifted bunch would
leave the domain, the run fails instead of writing a patch PIConGPU cannot
restart.

### `[accuracy]`

| key | required | meaning |
| --- | --- | --- |
| `tolerance` | yes | Relative free-space convolution target; positive, floor `1e-14`. |
| `near_shell` | no | `"none"` (default) or `"exact"`. |
| `near_shell_radius` | no | Shell radius in PIC cells; `0` (default) derives it from `tolerance`. |
| `near_shell_tile` | no | Tile edge in nodes; default `4`. |

## Particle patch and records

The writer creates one particle patch for the species. Its `offset` is the
lowest cell index of the shifted bunch bounding box relative to the domain
origin; its `extent` covers the bounding box. `numParticles` is the bunch size
and `numParticlesOffset` is zero.

Records are in PIConGPU internal units:

- `position`: fractional in-cell coordinate in `[0, 1)`, `unitSI` is the cell
  size. `positionOffset`: integer cell index, stored as int32.
- `momentum`: `p / (unit_mass * unit_speed)`.
- `weighting`: statistical weight with `macroWeighted = 1` and
  `weightingPower = 1`.

The checkpoint also contains `fields/picongpu_idProvider`, every other species
present in the input iteration, and `checkpoints.txt`.

## Precision

`precision` selects the storage type of the particle records and the six field
components. `float32` is the default and matches a standard single-precision
PIConGPU build. Use `float64` only for a double-precision PIConGPU build; it
doubles the checkpoint size and is not beneficial for a single-precision deck.

## Dependency pin

TOML parsing uses [toml++](https://github.com/marzer/tomlplusplus) v3.4.0 at
commit `30172438cee64926dc41fdd9c11fb3ba5b2ba9de`, fetched by CMake
`FetchContent`. The pin is recorded in `CMakeLists.txt`.

## Example

A complete example is committed as `examples/picbunch.toml`.

## Legacy CLI

The old flags remain available as an advanced path. They are not the documented
interface, and the unit flags (`--picongpu-unit-*`) are not part of the
documented usage.
