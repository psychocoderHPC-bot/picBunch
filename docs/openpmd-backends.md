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
loads. `pml` additionally writes the eight zero-filled `Convolutional PML E_*`
and `B_*` slab records an open-boundary PIConGPU deck reads on restart.
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
