# openPMD backend selection (HDF5 / ADIOS2 BP5)

The field writer (`writeField`, `writeFieldToDomain`) and the particle reader
open the series with `openPMD::Series(name, ...)` and no backend options. The
backend therefore follows the filename extension:

- `.bp` -> ADIOS2, default engine BP5;
- `.h5` -> HDF5.

No code change or CLI flag is needed. A `.bp` output path selects ADIOS2 and a
`.bp` input path is read through the ADIOS2 backend. This is openPMD-api's
documented selection rule (extension rule when the JSON `backend` parameter is
absent).

## Environment used for the verification

On `jureca-h100`, source the pinned environment from the external
`picbunch-repro-20261108` run mirror (`r3/env.sh`). It puts
`$HOME/opt/adios2-2.11.0/{lib,lib64}` and
`$HOME/opt/openPMD-0.17.1-adios2/{lib,lib64}` on `LD_LIBRARY_PATH` and sets
`HDF5_ROOT=/usr/lib/x86_64-linux-gnu/hdf5/openmpi`.

picBunch must be configured against the openPMD install that was built with
`openPMD_USE_ADIOS2=ON` and `openPMD_USE_HDF5=ON`:

```text
-DopenPMD_DIR=$HOME/opt/openPMD-0.17.1-adios2/lib64/cmake/openPMD
-DCMAKE_PREFIX_PATH="$HOME/opt/openPMD-0.17.1-adios2;$HOME/opt/adios2-2.11.0"
-DHDF5_ROOT=/usr/lib/x86_64-linux-gnu/hdf5/openmpi
```

## Verified equivalence (main @ 478f6e1, no source change)

The same in-memory field was written with `picBunchWriteSample` to `sample.h5`
and `sample.bp` (six components, extent `6x5x4`, 720 values):

- openPMD Python reports `backend=HDF5` and `backend=ADIOS2`.
- `bpls -lav sample.bp` lists six variables and 51 attributes; the file is a
  BP5 directory (`md.0`, `data.0`, `mmd.0`).
- Every component: dtype `float64`, shape `(6,5,4)`, `max_abs_diff = 0.0`,
  `0` non-bitwise values.
- `axisLabels = [z,y,x]`, `dataOrder = C`, `geometry = cartesian`,
  `gridSpacing`, `gridGlobalOffset`, `gridUnitSI`, `unitDimension`, every
  component `unitSI`, and every Yee `position` are identical.

The same particle bunch was written to `.h5` and `.bp`; picBunch was then run on
both with `--deterministic-deposit`. The HDF5 output payloads are bit-identical
(4,374,000 values, `max_abs_diff = 0.0`, `0` non-bitwise), so the reader
returns the same data from either backend. The HDF5 *container bytes* differ
(file layout/metadata), so the file sha256 is not comparable; the datasets are.

No payload field is backend-limited. Only the container differs.

## PIConGPU restart checkpoints use the openPMD writer

The application writes restart checkpoints with
`writePicOnGPUCheckpointOpenPMD` (`publishPicOnGPUOpenPMD`), not with the HDF5
C API. It builds an openPMD-Series and selects the backend like the field
writer: a `.h5` destination uses the HDF5 backend, a `.bp5` destination uses the
ADIOS2/BP5 backend, and the `[checkpoint] backend` key (`auto`, `adios2`,
`hdf5`) overrides the path-derived choice. Both containers carry the same tree:
`openPMDextension = 1`, the `/data/<step>` iteration, one full-shape
`(Nz, Ny, Nx)` record component per field component, and the same particle
records, including zero-particle `empty` species.

The legacy raw-HDF5 writer `writePicOnGPUCheckpoint` (`publishPicOnGPU`) and
its direct `H5*` implementation have been removed. The library sources under
`src/` and `include/` contain no direct HDF5 C API calls; HDF5 remains
available as an openPMD backend. The h5py-only
`tools/verify/check_picongpu_restart.py` inspects the openPMD HDF5 checkpoint
tree, while `check_openpmd.py` verifies the openPMD field writer.

## Field-absorber key

`[picongpu] absorber` in the picBunch configuration describes the boundary
setup of the deck that restarts the checkpoint and mirrors its
`--fieldAbsorber` value. It accepts `none` (the default), `exponential` and
`pml`; an unknown value is rejected. `none` and `exponential` write the same
tree because the exponential absorber keeps no field state that a restart
loads. `pml` additionally writes the twelve zero-filled `Convolutional PML`
slab records an open-boundary PIConGPU deck reads on restart: six field
components `E` and `B`, each with a `neg` and a `pos` slab record
(`E_xneg`, `E_xpos`, `E_yneg`, `E_ypos`, `E_zneg`, `E_zpos`, and the same six
for `B`). Each record carries the six cross components `xy`, `xz`, `yx`, `yz`,
`zx`, `zy` as full `(Nz, Ny, Nx)` datasets, with the main-field geometry, the
parent `E`/`B` `unitSI` and unit dimension, and the parent Yee positions
duplicated across the split components.
Without those slabs the restart aborts with
`Key 'Convolutional PML E_xneg' does not exist`. Set `absorber = "pml"` when
the restart deck uses the convolutional PML: a checkpoint written with
`absorber = "none"` has no PML slabs and cannot restart a PML deck.

CTest results depend on the build configuration and the optional tests it
registers. The original ADIOS2-enabled verification build (main @ 478f6e1,
ADIOS2 2.11.0 / openPMD 0.17.1 as pinned above) registered 23 tests and
passed 23/23. The final H100 `sm_90` build on the r2 stack reports 26/26,
including the padding and BP5 regression tests. Compare counts only between
builds with the same options: a different configuration can register a
different number of tests.

## `--openPMD.range` axis order and the single-range limit

The `--openPMD.range` option is read by the PIConGPU openPMD writer plugin when
a deck restarts from a checkpoint and dumps a field plane. The range tokens are
in **physical (x, y, z) order**: the first token is the x range, the second the
y range, the third the z range. This is easy to get backwards because the
stored openPMD array order is `[z, y, x]` (the plugin writes the full-shape
`(Nz, Ny, Nx)` record component with `axisLabels = [z, y, x]`), so a range that
looks like it fixes z actually fixes x, and vice versa.

This was settled empirically on `jureca-h100` (sm_90 build, PIConGPU source @
`f75efb0dc`) with an intentionally unequal grid of `64 x 128 x 32` cells (x, y,
z) so the output shape alone identifies the fixed axis. With
`--fieldAbsorber exponential` and `--autoAdjustGrid 0` the grid stays at exactly
`64 x 128 x 32`, and the full-domain dump has stored shape `[32, 128, 64]` with
`axisLabels = [z, y, x]`. The measured single-cell slices (a `10:11` window, the
second cell, so the plane is non-symmetric and the fixed axis is visible):

| `--openPMD.range` | stored shape `(Nz, Ny, Nx)` | `gridGlobalOffset` (cells, `[z, y, x]`) | fixed axis |
|-------------------|-----------------------------|------------------------------------------|------------|
| `10:11,:,:`       | `[32, 128, 1]`              | `[0, 0, 10]`                             | x          |
| `:,10:11,:`       | `[32, 1, 64]`               | `[0, 10, 0]`                             | y          |
| `:,:,10:11`       | `[1, 128, 64]`              | `[10, 0, 0]`                             | z          |

The `gridGlobalOffset` of the written component confirms the mapping: for
`10:11,:,:` only the last (x) offset entry is `10 * cellSize`, i.e. the first
range token addresses the axis that openPMD labels `x`; for `:,:,10:11` only the
first (z) entry is set.

Two caveats that cost a wrong plane in a prior production run:

- **Only one range per run.** The plugin binds a single global
  `--openPMD.range` (one `Option<std::string>` in `openPMDWriter.x.cpp`); there
  is no per-instance range. Repeating the flag is not honoured: the **first**
  value is kept and later duplicates are silently ignored (verified on
  `jureca-h100`: `10:11,:,:` then `:,:,10:11` produced the x-fixed plane only).
  A paired `--openPMD.file` has the same first-value-wins behaviour, so two
  spatial planes cannot be written in one invocation. To get a y-x plane at
  fixed z and a y-z plane at fixed x, either run two separate jobs restarted
  from the same checkpoint step (each with its own `--openPMD.range` and a
  distinct `--openPMD.file`), or write the full domain once and slice the fields
  afterwards with `h5py` / the openPMD-api.

- **Do not pad a range token with a leading or trailing comma to "keep the
  position" of the fixed axis.** The range string is split on `,` with
  `std::sregex_token_iterator`, which does not emit empty tokens, and the parser
  then skips any empty slice it does encounter. An empty token is therefore
  dropped and the remaining tokens shift down to the lower dimension indices:
  `:,10:11,:` is parsed as if the x token were `10:11` and would fix x, not y.
  Write the full three-token range with the two non-fixed axes as explicit `:`
  (as in the table above) so every token lands on the intended axis.

A ready-to-copy production pair for a `640 x 1536 x 640` (x, y, z) grid
(restart step 10, cell `1.772e-7` m), run as two separate jobs from the same
checkpoint:

```bash
# y-x plane at fixed z = 320 (third token)
--openPMD.period 10 --openPMD.source fields_all --openPMD.particle 0 \
--openPMD.ext h5 --openPMD.file yx_z320 \
--openPMD.range :,:,320:321

# y-z plane at fixed x = 320 (first token)
--openPMD.period 10 --openPMD.source fields_all --openPMD.particle 0 \
--openPMD.ext h5 --openPMD.file yz_x320 \
--openPMD.range 320:321,:,:
```

Expected stored shapes (order `[z, y, x]`): the y-x plane is `[1, 1536, 640]`
(one z cell, full y, full x) and the y-z plane is `[640, 1536, 1]` (full z,
full y, one x cell).
