# Restarting PIConGPU from a picBunch checkpoint

This note is the end-to-end checklist for the restart workflow: write a
checkpoint with picBunch, then start PIConGPU from it with an open-boundary or
periodic deck. The checkpoint format, its structural differences from a
PIConGPU-written file and the PML handling are recorded in
[picongpu-restart.md](picongpu-restart.md); the TOML schema, the placement rule
and the precision switch are in [configuration.md](configuration.md).

## Restart checklist

A restart works only when the picBunch run and the PIConGPU deck agree on the
physics inputs, the grid, the species and the boundary setup. Check each row
before running; a mismatch either aborts the restart or silently rescales
every particle record.

### Step 1. Copy the physics inputs into the TOML

The `[picongpu]` table carries the independent deck inputs. Copy them 1:1 from
the deck that will restart the checkpoint (`simulation.param`,
`speciesConstants.param`, `physicalConstants.param`). picBunch derives every SI
unit factor from them, exactly as
`include/picongpu/unitless/simulation.unitless`; do not transcribe the `unit_*`
factors themselves.

| TOML key | deck source | what a mismatch does |
| --- | --- | --- |
| `delta_t` | `SI::DELTA_T_SI` in `simulation.param` | time step and every time-derived unit factor wrong |
| `cell_size` | `SI::CELL_WIDTH/HEIGHT/DEPTH_SI` in `simulation.param` | field spacing and mass/charge units wrong |
| `base_mass` | `SI::BASE_MASS_SI` in `speciesConstants.param` | mass unit wrong, momentum rescaled |
| `base_charge` | `SI::BASE_CHARGE_SI` (signed) in `speciesConstants.param` | charge unit wrong, momentum and fields rescaled |
| `base_density` | `SI::BASE_DENSITY_SI` in `simulation.param` | macro-particle weight, mass and charge units wrong |
| `typical_particles_per_cell` | `TYPICAL_PARTICLES_PER_CELL` in `simulation.param` | same as `base_density` (sets the weight) |

The full set of keys, including `step`, `precision` and the species lists, is
in [configuration.md](configuration.md#toml-schema).

### Step 2. Match the species

Three lists matter, and they must be consistent with the deck:

| list | TOML key | deck counterpart |
| --- | --- | --- |
| read from `--input`, placed, field-computed and written | `species` (string or array) | the bunch species the deck pushes |
| checkpoint group name for each `species` entry | `output_species` (positional) | the deck species name, e.g. `e` |
| deck species written with zero particles | `empty_species` (array) | every deck species the deck will load |

The writer copies every species group from the input iteration and appends the
`empty_species` as zero-particle groups. PIConGPU loads every compiled species
that the deck declares, and a declared species that is absent from the
checkpoint aborts the restart before any field is read, with

```
Key 'e' does not exist (read-only).
```

`e` is the example deck name. The measured abort for a deck that declares the
ion species `i` is word for word `Key 'i' does not exist (read-only)`, recorded
in [the 4-rank restart
measurement](measurements/picongpu_restart_4rank.md). List every deck species
in `output_species` or `empty_species`, or drop it from the deck.

### Step 3. Match the grid and the placement

| input | TOML key | deck counterpart |
| --- | --- | --- |
| domain cell counts | `domain.cells` | `-g` on the PIConGPU command line |
| cell size | `domain.cell_size` | `SI::CELL_*_SI` |
| physical position of cell `(0,0,0)` | `domain.origin` | the deck origin (PIConGPU restart is index-based, see below) |
| bunch centroid shift | `placement.center`, `placement.position`, `placement.offset_unit` | where the deck places the bunch |

Use `placement.center = "none"` when the input coordinates are already
absolute and the deck keeps them. A `placement.center = "centroid"` run shifts
the charge-weighted centroid to `position`. For a multi-rank restart use
`placement.patch = "domain"` and `placement.patch_count` equal to the number of
ranks along the split axis; the 4-rank measurement used `patch = "domain"`,
`patch_count = 4` and 4 ranks in z.

### Step 4. Run picBunch

```bash
picBunch --input bunch.h5 --destination checkpoints --config params.toml
```

The destination extension selects the checkpoint backend: a `.h5` destination
uses HDF5, a `.bp5` destination uses ADIOS2/BP5, and both produce the same
tree. The `[checkpoint] backend` key can force one (`auto`, `adios2`, `hdf5`);
`auto` is the default.

### Step 5. Prepare the PIConGPU deck

| deck input | requirement |
| --- | --- |
| grid | `-g nx ny nz` equal to `domain.cells` of the TOML |
| cell size and time step | same SI values as `cell_size` and `delta_t` |
| species | every species in the deck's `speciesDefinition.param` must exist in the checkpoint: a real group for `species`/`output_species`, an empty group for `empty_species` |
| initialization pipeline | empty for the species that come from the checkpoint, so no profile injects particles on top of the loaded state |
| boundary setup | `--periodic` and the field absorber must match the matrix below, and must match the boundary the checkpoint was written for |

An empty initialization pipeline means the `InitPipeline` in
`speciesInitialization.param` lists no `CreateDensity` entry for the species
that the checkpoint provides. A deck with a non-empty profile for those species
runs the profile first and the restart loads on top of it, so the particle
state no longer matches the checkpoint.

### Step 6. Start the restart

```bash
picongpu -g 24 24 96 --periodic 1 1 1 \
  --checkpoint.restart --checkpoint.restart.backend openPMD \
  --checkpoint.openPMD.ext h5 \
  --checkpoint.restart.directory checkpoints --checkpoint.restart.step 0 \
  --checkpoint.period 100 --checkpoint.directory out -s 1000
```

The measured 4-rank form used `mpirun -n 4` with `-d 1 1 4` and the same
checkpoint options. A single-rank run drops `-d`.

## Origin convention

The front/top/left corner of the PIConGPU domain, cell `(0,0,0)`, is the origin
of the cell index space. `domain.origin` in the TOML is the physical position
of that corner. Particles keep their absolute coordinates: the checkpoint
stores `positionOffset` as integer cell indices and `position` as the fractional
in-cell offset, and PIConGPU restart places them by cell index, not by
reapplying the deck origin.

A restart is therefore index-based. The deck `gridGlobalOffset` is not
reproduced by picBunch: the writer emits the fields on the `domain.cells` grid
with the cell indices of that domain, and the restart reads them back at the
same indices. If the deck domain does not start at the same physical corner as
`domain.origin`, the loaded state is shifted by the difference. Keep the two
origins equal, or account for the shift explicitly.

## Boundary-condition matrix

The restart checkpoint carries the six field components, the particles and the
id provider. It does not carry any boundary-specific auxiliary state other than
the `Convolutional PML ...` records of the input checkpoint. The boundary the
deck runs with must therefore match what the checkpoint holds.

| boundary | deck setup | restart from a picBunch checkpoint |
| --- | --- | --- |
| periodic | `--periodic 1 1 1`, no absorber on those axes | works. This is the verified path: the acceptance runs in [picongpu-restart.md](picongpu-restart.md) used `--periodic 0 0 1`, and the 4-rank measurement used `--periodic 1 1 1` |
| absorbing (exponential) | `--periodic 0 0 0` (or `0` on the absorbing axes) plus `--fieldAbsorber exponential` and the matching `--<s>_boundary` values | works. The checkpoint fields are the free-space field on the domain, and the exponential absorber acts on the field after restart. The absorber itself keeps no checkpoint state |
| convolutional PML | `--fieldAbsorber pml` (the default for a non-periodic deck) | does not work today. A picBunch checkpoint does not contain the PML auxiliary fields, so a PML deck cannot restart from it |

The PML case fails because PIConGPU loads the PML slabs unconditionally and the
checkpoint has no PML records: the measured abort is
`Key 'Convolutional PML E_xneg' does not exist`. The writer never synthesizes
PML state. It copies the `Convolutional PML ...` groups of the input
checkpoint verbatim, so a deck that restarts from a checkpoint whose input
carried PML records (a previous PIConGPU checkpoint, for example) restarts
without this error. The open-boundary rerun in [picongpu-restart.md](picongpu-restart.md)
is that case: the input had eight `Convolutional PML` groups and the restart
exited 0 with the PML datasets bitwise equal to the source.

Two workarounds keep a PML-like open boundary with a picBunch checkpoint:

1. Run the deck with `--fieldAbsorber exponential` and the matching
   `--<s>_boundary` values. The exponential absorber is a field-space filter
   and needs no checkpoint state.
2. Run the deck with periodic boundaries (`--periodic 1 1 1`). This changes the
   physics at the domain edges and is only acceptable when the bunch does not
   reach them before the region of interest is recorded.

The follow-up work is a PML-state writer in picBunch that derives the slab
state for the configured absorber, so a PML deck can restart directly. Until
that lands, the two workarounds above are the supported paths.

## PNG slice output

The `PngPlugin` renders one PNG per axis pair per configured instance. The
instance name is `<species>_png`, so a deck with species `e` and `i` has the
instances `e_png` and `i_png`. Inside one instance, `--<name>_png.axis` takes
a two-character axis code (`yx`, `yz`, `zx`, ...) and renders the slice of the
plane spanned by the two named axes at the `--<name>_png.slicePoint` fraction
of the third axis. The first character is the vertical axis of the rendered
image.

The plugin registers the flags as multi-token vectors and pairs the tokens
positionally: token `i` of `axis` pairs with token `i` of `slicePoint`,
`period` and `folder`. A second plane needs a second token in each flag of the
same instance:

```bash
--e_png.period 10 10 \
--e_png.axis yz yx \
--e_png.slicePoint 0.5 0.5 \
--e_png.folder pngYz pngYx
```

Repeating `--<name>_png.axis` under the same instance does not create a second
instance, because the instance is registered once per species. When the other
flags carry a single token, the plugin fills the missing tokens with the last
given value, so both slices render at the same period and slice point. Repeating
`--<name>_png.axis` without repeating `slicePoint` and `period` therefore
silently overwrites the first pairing for the second index; repeat every flag
with one token per plane.

The PNG options are compile-time: the build needs the PNG dependency enabled
(`PIC_ENABLE_PNG`), otherwise `--<name>_png` is listed as disabled in
`--help`. The `png.param` file carries the color scales, channel normalization
and the `preChannel1/2/3` functors that select which field components map to
the three color channels. It does not select the axis, the slice point, the
period or the folder; those are command-line only.

A ready-to-copy `png.param`. The file sets the color scales, the channel
normalization and the three channel functors; it does not select the axis,
slice point, period or folder. The values below are the stock PIConGPU
defaults from `include/picongpu/param/png.param`.

```cpp
namespace picongpu
{
    constexpr float_64 scale_image = 1.0;
    constexpr bool scale_to_cellsize = true;
    constexpr bool white_box_per_GPU = false;

    namespace visPreview
    {
#define EM_FIELD_SCALE_CHANNEL1 -1
#define EM_FIELD_SCALE_CHANNEL2 -1
#define EM_FIELD_SCALE_CHANNEL3 -1

        constexpr float_64 customNormalizationSI[3] = {
            5.0e12 / sim.si.getSpeedOfLight(), 5.0e12, 15.0};

        constexpr float_X preParticleDens_opacity = 0.25;
        constexpr float_X preChannel1_opacity = 1.0;
        constexpr float_X preChannel2_opacity = 1.0;
        constexpr float_X preChannel3_opacity = 1.0;

        namespace preParticleDensCol = colorScales::red;
        namespace preChannel1Col = colorScales::blue;
        namespace preChannel2Col = colorScales::green;
        namespace preChannel3Col = colorScales::none;

        DINLINE float_X preChannel1(float3_X const& field_B,
                                    float3_X const& field_E,
                                    float3_X const& field_Current)
        {
            return field_E.x() * field_E.x();
        }

        DINLINE float_X preChannel2(float3_X const& field_B,
                                    float3_X const& field_E,
                                    float3_X const& field_Current)
        {
            return field_E.y();
        }

        DINLINE float_X preChannel3(float3_X const& field_B,
                                    float3_X const& field_E,
                                    float3_X const& field_Current)
        {
            return -1.0_X * field_E.y();
        }
    } // namespace visPreview
} // namespace picongpu
```

The matching command line renders the xy plane at `x = 0.5` (axis code `yz`)
and the zy plane at `z = 0.5` (axis code `yx`):

```bash
picongpu -g 64 64 64 --periodic 1 1 1 \
  --checkpoint.restart --checkpoint.restart.backend openPMD \
  --checkpoint.openPMD.ext h5 \
  --checkpoint.restart.directory checkpoints --checkpoint.restart.step 0 \
  --e_png.period 10 10 \
  --e_png.axis yz yx \
  --e_png.slicePoint 0.5 0.5 \
  --e_png.folder pngYz pngYx \
  --checkpoint.period 100 --checkpoint.directory out -s 1000
```

Both planes share `--e_png.period 10` because it is the only period token and
the plugin repeats the last token for the second index. Give a second
`--e_png.period` token to render the two planes on different periods.

The files are written to `<folder>/<name>_png_<axis>_<slicePoint>.png`, for
example `pngYz/e_png_yz_0.5.png` and `pngYx/e_png_yx_0.5.png`. The plugin
skips a slice that the simulation setup forbids (a moving window forbids the
`xz` plane, a 2D simulation forbids any slice that uses `z`) and prints a
warning for the skipped axis.

## Known limitations

The restart workflow is subject to the general picBunch limitations. The three
that bear directly on a restart run:

| limitation | effect on a restart |
| --- | --- |
| large domains need the distributed multi-rank path | a domain whose padded box is above `2^28` nodes on one rank is rejected on a single rank with the `limit 268435456 nodes` message. Run under `mpirun -n R` and keep `--fft-mode auto` (the default); auto selects the distributed z-slab FFT and scales the per-rank cap by `R * 8`. See [configuration.md, Large domains under MPI](configuration.md#large-domains-under-mpi) |
| single-process tiling is not supported | there is no chunked or slab-tiled field-computation path. The checkpoint is written only from a fully materialised field, so the peak device memory is the full padded FFT box plus the six output components. A domain that does not fit on one device must use the multi-rank path above |
| per-rank padded-node cap | `makeGrid` rejects a padded box above the cap for the given rank count. The cap is `2^28` nodes per rank and scales by `R * 8` with the distributed FFT. A domain that fits with padding 1 can be rejected with padding 2 or 3 |

The full list of open engineering items, with symptoms, evidence and plans, is
in [known-limitations.md](known-limitations.md). The accuracy envelope of the
field itself is in the [README error budget](../README.md#error-budget) and
[algorithm.md](algorithm.md).
