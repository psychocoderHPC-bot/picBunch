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

## PIConGPU restart checkpoints stay HDF5

`writePicOnGPUCheckpoint` (`publishPicOnGPU`) is not openPMD IO: it writes the
PIConGPU restart layout with the HDF5 C API (`H5Fcreate`, `H5Dcreate`,
`H5Dwrite`, `H5Ocopy`), hardcodes the `.h5` suffix, and CMake requires HDF5 for
this path. The restart verifiers (`tools/verify/check_picongpu_restart.py`,
`check_openpmd.py`) are h5py-only and expect `checkpoint_%06d.h5`.

This path intentionally remains HDF5. Moving it to BP5 would be a checkpoint
format rewrite (several hundred lines across the writer, CMake and the restart
tests), not a backend switch, and the reproduction does not need it: every
openPMD-level dataset (bunch input and E/B output) can use ADIOS2/BP5.

CTest results depend on the build configuration and the optional tests it
registers. The original ADIOS2-enabled verification build (main @ 478f6e1,
ADIOS2 2.11.0 / openPMD 0.17.1 as pinned above) registered 23 tests and
passed 23/23. The final H100 `sm_90` build on the r2 stack reports 26/26,
including the padding and BP5 regression tests. Compare counts only between
builds with the same options: a different configuration can register a
different number of tests.
