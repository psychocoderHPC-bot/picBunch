# picBunch

Standalone bunch-field initializer. It reads an openPMD particle series, computes the
free-space electric and magnetic field of the bunch on a uniform grid, and writes the six
Yee-staggered components back as an openPMD series. A PIConGPU-restart-compatible
checkpoint is an optional output mode.

The pipeline is openPMD particles in, free-space hybrid field out, openPMD field written
back. The reader, TSC deposit, moving point kernel, padded FFT convolution, Yee read-out,
openPMD writer, and PIConGPU restart writer are implemented. Written field values are SI:
E in V/m and B in T, with `unitSI = 1.0` and the matching `unitDimension`. v0.1 supports a
single shared velocity; an input with a momentum spread exits non-zero. `--tolerance` is
the only accuracy knob. The physics validation gates, the near-source limitation and their
provenance are recorded in [docs/algorithm.md](docs/algorithm.md).

## Documentation

- **Paper:** [docs/paper/paper.pdf](docs/paper/paper.pdf) covers the method,
  measured cost and accuracy, and PIConGPU checkpoint cross-validation at
  v0.1.3.

## Layout

```
include/picBunch/   public headers (types, TSC deposit, point-field kernel, FFT self test)
src/                host implementation, per-backend device sources under src/cuda, src/hip
tests/              smoke test (CTest), GoogleTest unit tests and reference CSVs
tools/verify/       independent HDF5 consumer checks (h5py, no openPMD binding)
docs/               algorithm and validation notes
```

## Dependencies

| Dependency | Pin | Notes |
| --- | --- | --- |
| alpaka | `7b92361468d3280c260200198081ae7e98f0bc56` (reports 3.0.0) | fetched by CMake `FetchContent` |
| alpakaVendor | `af24cc7b2bd125056e9889e270d3af7bdf972c6f` | fetched by CMake `FetchContent`; FFT via cuFFT/rocFFT/FFTW |
| openPMD-api | system install | HDF5 backend required; configure fails without it |
| HDF5 C API | system install | checkpoint writer; the parallel build matching openPMD is preferred |
| MPI | system install | only when HDF5 is parallel, for the HDF5 headers |
| FFTW 3 | system (`fftw3`, `fftw3f`) | host FFT backend |
| GoogleTest | system first, `FetchContent` fallback | only when `PICBUNCH_BUILD_GTESTS=ON` |

CMake 3.25+ and a C++20 compiler are required. The CUDA build needs CUDA 12.x, the HIP
build needs ROCm with rocFFT. The field pipeline is double precision. A float32 build is
not supported and its accuracy is not claimed.

## Build

### CUDA, Tesla V100 (sm_70)

The `cuda-v100` preset sets `PICBUNCH_ENABLE_CUDA=ON`, `PICBUNCH_ARCH=70`, and
`CMAKE_BUILD_TYPE=Release`. The validated install is openPMD 0.17.1 with the HDF5
backend at `/opt/openpmd-hdf5-0.17.1/lib/cmake/openPMD`:

```bash
export PATH=/usr/local/cuda/bin:$PATH
env -u LD_PRELOAD cmake --preset cuda-v100 \
  -DopenPMD_DIR=/opt/openpmd-hdf5-0.17.1/lib/cmake/openPMD
env -u LD_PRELOAD cmake --build --preset cuda-v100 -j 8
env -u LD_PRELOAD ctest --preset cuda-v100
```

`nvcc` must be on `PATH`. An explicit `-DCMAKE_CUDA_ARCHITECTURES=...` takes precedence
over `PICBUNCH_ARCH`. Device sources are picked up only from `src/cuda/*.cu`.

### HIP, Radeon RX 7900 (gfx1100)

CMake 3.28 rejects the `hipcc` wrapper as `CMAKE_HIP_COMPILER`, so point both compiler
variables at the ROCm compiler driver. `CMAKE_HIP_ARCHITECTURES` and
`CMAKE_PREFIX_PATH` select the device architecture and the ROCm install, and
`openPMD_DIR` names the exact openPMD package. This is the complete invocation on the
validated `hal` host (ROCm under `/opt/rocm`, openPMD 0.17.0):

```bash
cmake -S . -B build/hip -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DPICBUNCH_ENABLE_HIP=ON \
      -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
      -DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++ \
      -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
      -DCMAKE_PREFIX_PATH=/opt/rocm \
      -DopenPMD_DIR=/opt/openpmd/0.17.0/lib/cmake/openPMD
cmake --build build/hip -j 8
ctest --test-dir build/hip --output-on-failure
```

Two build traps are handled by `CMakeLists.txt`; no extra flags are needed:

- **OpenMP with HIP.** alpaka's HIP backend rejects OpenMP when the C++ compiler is not
  Clang. The default C++ compiler on many hosts is GNU, so the project forces
  `alpaka_DEP_OMP=OFF` for the HIP build unless the user defines it. Passing
  `-Dalpaka_DEP_OMP=ON` still reaches alpaka's error, by design.
- **C language for an ADIOS2-enabled openPMD.** openPMD's ADIOS2 MPI target links
  `MPI::MPI_C`, and CMake's `FindMPI` only creates per-language MPI targets for languages
  enabled by `project()`. The project enables C before `find_package(openPMD)` so that
  target exists.

Pin a free device before running, for example `HIP_VISIBLE_DEVICES=0`.

Other presets: `hip-gfx1100`, `cpu-debug`. Every preset builds into `build/<preset>/`,
which is gitignored. The presets set only the backend and architecture; pass compiler and
dependency paths on the command line.

## Run

```bash
picBunch --input <openPMD file/series> [--species <name|index>] \
         [--output <openPMD file>] [--input-iteration N] \
         [--output-iteration N] [--tolerance 1e-3] \
         [--cell-size <dx>[,<dy>,<dz>]] [--grid <nx>,<ny>,<nz>] \
         [--origin <x>,<y>,<z>] [--backend auto|cuda|hip|cpu]
```

Example with mesh metadata in the input:

```bash
picBunch --input bunch_%T.h5 --species electrons --output field_%T.h5 \
         --input-iteration 0 --output-iteration 0 --tolerance 1e-3
```

For a particle-only input without mesh metadata, give the geometry explicitly in metres.
`--grid` takes even node counts; `--origin` is the physical position of node `(0,0,0)`:

```bash
picBunch --input bunch.h5 --species electrons --output field.h5 \
         --cell-size 1e-7,1e-7,1e-7 --grid 64,64,32 --origin 0,0,0 \
         --tolerance 1e-3
```

An explicit `--cell-size` always wins over the mesh-derived value, and the run fails with a
clear message when neither source exists. Nothing is assumed or rescaled. The physical cell
size is taken from the input mesh (`E`) as `gridSpacing * gridUnitSI` when the input has a
mesh with a positive `gridUnitSI`; a missing or non-positive `gridUnitSI` is an error. The
reader applies that conversion once and returns physical SI spacing and offset; the CLI does
not apply `gridUnitSI` again. When the input has a mesh, its extent (dataset shape) is the
default output grid, its `gridSpacing` the node spacing and its `gridGlobalOffset` the origin.
`--grid`, `--cell-size` and `--origin` override those defaults individually. Without a mesh,
the grid is the particle bounding box plus the margin and `--cell-size` is required. A particle
outside the resulting grid is an error that names the count and the bounds; it is never dropped
silently.

`--tolerance` is the only accuracy knob. It is a relative target for the delivered
free-space convolution: padding 2 is used for targets at or above the measured padding-2
bound (`1e-12`), padding 3 for targets between that bound and the measured double-precision
floor (`1e-14`), and smaller targets are rejected with a non-zero exit. It does not change
the interpolation order, the deposit scheme or the node refinement.

### PIConGPU restart checkpoint

`--restart-format picongpu` writes a checkpoint that PIConGPU's `RestartFieldLoader` can
consume instead of the standard SI openPMD series:

```bash
picBunch --input bunch_%T.h5 --species electrons \
         --restart-format picongpu --output-dir checkpoints \
         --picongpu-step 0 \
         --picongpu-unit-length 5.773502691896257e-08 \
         --picongpu-unit-e 8850761452125.0 \
         --picongpu-unit-b 29522.962355927582
```

All three unit flags are mandatory; a missing one is an error, never a default. The writer
emits `checkpoints/<prefix>_<step:06d>.h5` plus `checkpoints.txt`. Fields are float32 with
extent `(Nz,Ny,Nx)`, stored as `field/unit` in PIConGPU internal units. The input particle
species is copied as a raw HDF5 object, so its attributes and `particlePatches` survive.
`--picongpu-species` overrides the copied species and `--picongpu-dt` sets the stored `dt`.
When the input checkpoint contains `Convolutional PML ...` records, the writer copies those
records from the input instead of recomputing them. See
[docs/picongpu-restart.md](docs/picongpu-restart.md).

## Test status

The final state is the v0.1.3 commit `da3cd4239926f5da31b5794a36f8184e0b3164da`:
12/12 CTest entries pass on both validated backends.

- **rosi_v100, Tesla V100-SXM2-32GB, CUDA 12.6, sm_70:** 12/12 CTest entries passed at
  `da3cd4239926f5da31b5794a36f8184e0b3164da` on 2026-11-04. GPU 0 was idle before and after
  the run, and all four GPUs reported 0% and 0 MiB.
- **hal, Radeon RX 7900 XTX, gfx1100, HIP:** 12/12 CTest entries passed at
  `da3cd4239926f5da31b5794a36f8184e0b3164da` on 2026-11-04. This is a shared host, so the
  run gives no timing claim. gfx1100 is a consumer RDNA3 part without a fast native FP64
  path; its FP64 throughput is unrepresentative of data-centre AMD GPUs.

Historical counts belong to earlier commits: 10/10 on hal at `339ca7ad` and 12/12 on
rosi at `bed4344`, both before v0.1.3. The v0.1.3 numbers above supersede those counts.

The 12 rosi entries are `smoke`, `Deposit`, `Kernel`, `Fft`, `Io`, `Physics`, `Padding`,
`Restart`, `OpenPMDVerify`, `PicOnGPUVerify`, `ScalarRecordSample`, `IoScalarRecord`.
`OpenPMDVerify` and `PicOnGPUVerify` read the written files with h5py alone. The last two
build and read an openPMD 2.0 scalar-record file.

## Accuracy

The headline metric is per-component relative error against the TSC-convolved
continuum reference, never pooled. It was measured with the `picBunch` executable itself
on `rosi_v100` (V100, CUDA), `kRefinement = 1`, one source on a node, against the
independent `~/pr24/pr24w53-ref/indfield.py` (order 12, refine 3). A separate
absolute-normalised metric `|ref-mesh|/(1+max(|ref|,|mesh|))` was used by an earlier
campaign; it is labelled where it appears and is not the basis of the crossover claim.
The committed raw tables are in [docs/measurements/](docs/measurements/), with the rosi
campaign workdir recorded per file as the reproduction hint.

**Coulomb limit.** One stationary particle at a node, far nodes at `d = 8,16,24,32` along
all six axis directions. The mesh field against the direct TSC stencil sum of the analytic
Coulomb kernel agrees to **4.88e-14** (maximum absolute `2.28e-25`), above the `1e-12`
threshold. This is the zero-padded FFT convolution checked against the discrete sum.

**Single-particle far field against the TSC-convolved continuum.** The metric is
per-component relative error, never pooled, measured with the `picBunch` executable
itself on `rosi_v100` (V100, CUDA), `kRefinement = 1`, one source on a node, against the
independent `indfield.py` (order 12, refine 3). Aspect 1 and aspect 4 are the beam aspect
ratios of that campaign. The `3e-4` relative crossover:

| case | 3e-4 relative crossover |
| --- | --- |
| gamma 1, aspect 1 | **d = 12 cells** |
| gamma 1, aspect 4 | **d = 32 cells** |
| gamma 10, aspect 1 | **not reached within d = 32**; 6.3e-3 at d = 32 |

Inside the TSC support (`|s| < 1.5` cells) this metric is untested; no claim is made
there.

At `d = 48` cells, aspect 1, `h = 1`, the same relative metric gives:

| gamma | worst relative error at d=48 | reference |
| --- | --- | --- |
| 1 | **1.38e-7** (1.4e-7) | TSC-convolved continuum |
| 10 | **2.65e-4** (2.6e-4) | TSC-convolved continuum |

Gamma 100 does not converge in the perpendicular direction: the mesh-to-continuum error
is `3.75e-1` at 48 cells.

**Error depends on `r/h` only. Refining `h` at fixed cell-distance does not help.** The
kernel is homogeneous of degree -2, the TSC shape is one node spacing wide, and the Yee
offsets are in cell units. Measured at `h = 1, 1/2, 1/4, 1/8`, the relative errors are
unchanged to machine precision (`max rel change 0.0`) at gamma 1, 10 and 100 and aspect 1
and 4. The only geometric knob is the physical distance measured in cells.

## Cross-validation against PIConGPU's exact reference

The reference is the same exact TSC-convolved quadrature as PIConGPU
`fieldInitializer=referenceOnly`, evaluated by the standalone driver in
`~/pr24/b1-20261102`. picBunch was run at `339ca7ad3903249ebf3859f5e106d63db4dc3df5`
(the field path is unchanged through v0.1.3, `da3cd4239926f5da31b5794a36f8184e0b3164da`),
tolerance `1e-3`, on `rosi_v100`.
The metric here is the absolute-normalised `|ref-mesh|/(1+max(|ref|,|mesh|))`; it is
not the per-component relative error of the Accuracy section and is not a crossover
claim.

**318 staggered points**, 64x64x32 grid, cell 1e-7 m, N=1e4, identical Yee offsets
(snap deviation 0):

| case | worst relative error |
| --- | --- |
| gamma 1, stationary | E/x 3.66e-2, E/y 4.19e-2, E/z 4.86e-2; B identically 0 |
| gamma 10, drift +x | E/x 6.08e-2, E/y 5.16e-1, E/z 6.20e-1, B/y 1.23e-1, B/z 1.51e-1 |

**Whole field**, N=1000, 32x32x16 grid, cell 2e-7 m:

| case | worst relative error |
| --- | --- |
| gamma 1 | E/x 1.48e-1, E/y 1.49e-1, E/z 1.22e-1 |
| gamma 10 | E/x 1.36e-1, E/y 1.98, E/z 1.93, B/y 4.42e-1, B/z 4.63e-1 |

The bulk agrees much better: median per-node relative error is 0.4% to 3.3%. The
reference's own floor is 5.0e-7 to 2.7e-6 (float32 dump limit), so picBunch is the limiting
side.

Caveat: the 318-point groups were **distance from the bunch centroid** (inside `d <= 1.6`,
face `1.6 < d <= 8`, outside `d > 8`), not a per-source support test. The sample set is
clustered at those radii. A separate N-sweep on the same 318 points shows the disagreement
is **per-source and flat in N**, not a superposition defect: outside the TSC support the
worst error stays at about `1e-3` to `3e-2` from N=1 to N=1000. The O(1) values at N >= 10
appear only when the global worst sample is inside a particle's TSC support, within
0.42 to 0.65 cells of a source; that is the known near-source regime. The disagreement is
localised to the TSC support and does not grow with particle count.

## PIConGPU checkpoint transport

A picBunch checkpoint restarts PIConGPU with all six field components bitwise identical:
`E/x E/y E/z B/x B/y B/z` have `max|diff| = 0`, `worst_rel_SI = 0`, `max_ulp = 0`. The
verified run is a 16-particle periodic case on `rosi_v100` with PIConGPU PR #24 at
`2baa2566`, a validation-only `maxwellSolver::None` build that dumps the loaded state at
step 1. The restart skipped the deck initializer and exited 0.

For the open-boundary LWFA case, picBunch copies all eight `Convolutional PML` groups
(48 datasets) from the input checkpoint. The values and `position`/`unitSI` attributes
survive, and the copied float32 datasets are bitwise equal to the source. The reduced
64x128x64 LWFA geometry (`--periodic 0 0 1`) then restarts PIConGPU: exit 0,
`initialization time: 3.986 sec`, `full simulation time: 4.031 sec`.

Limit: **PML is copied from the same input file.** A particle-only input has no PML records,
so the writer emits none and cannot restart an open-boundary deck. Use an input checkpoint
that already contains PML state, or restart with periodic boundaries.

## Performance on V100

Measured with the synthetic benchmark on `rosi_v100`, node `gv002`, GPU 0, CUDA 12.6,
Release, double, padding 2. GPU 0 was idle before the sweep (0%, 0 MiB, 135 MHz) and all
runs used `CUDA_VISIBLE_DEVICES=0`. Each configuration ran one untimed warm-up plus one
device warm-up and **3 timed repeats**; the table gives medians. Benchmark code:
`experimental/bench` at `3d05752`, tested picBunch source `dc69c16`. The committed raw
scaling matrix, Coulomb-limit and d-sweep CSVs are in
[docs/measurements/](docs/measurements/); the rosi workdir is
`~/pr24/sb1-20261102/`.

| N | box (cells) | total (ms) | deposit | kernel | FFT | read-out | device (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e6 | 20 | 61.92 | 53.61 | 0.53 | 4.31 | 3.45 | 38.9 |
| 2e6 | 20 | 64.76 | 56.51 | 0.43 | 3.80 | 3.69 | 69.4 |
| 2e6 | 50 | 85.67 | 52.82 | 2.85 | 13.89 | 15.20 | 127.4 |
| 2e6 | 100 | 232.97 | 51.89 | 22.66 | 75.97 | 82.40 | 468.7 |

The pipeline is deposit-dominated at large N (about 52 to 57 ms at N=2e6 on every box) and
FFT-dominated at large box (kernel 0.43 to 22.66 ms and FFT 3.80 to 75.97 ms from box 20 to
100). Peak exact device use is **69.4 MiB** at N=2e6/20-cell and **468.7 MiB** at
N=2e6/100-cell. No configuration approached the 32 GB device limit. The four independent
N=2e6/20-cell repeats span 60.35 to 64.81 ms, about 7% scatter.

Beyond the compute pipeline, the fresh-clone reproduction run at the documented smoke
geometry (64x64x32 grid, cell 1e-7 m, 100 particles, gamma 1) timed the whole executable
end to end on rosi V100: deposit 1.6 ms, kernel 1.4 ms, FFT 6.8 ms, read-out 7.6 ms,
writer 46.1 ms, total 63.5 ms. The openPMD writer dominates at small field sizes; the
table above covers the compute stages of the large-N sweep, not the write path.

## Changelog

- **v0.1.3** makes the input mesh define the whole output grid (extent, spacing, origin) when
  present, rejects particles outside it, and keeps the bunch-AABB path for mesh-free input.
- **v0.1.2** honours the input mesh `gridGlobalOffset` as the default output-grid origin;
  `--origin` remains an override.
- **v0.1.1** fixes v0.1 mishandling non-unit `gridUnitSI`. The reader already returned
  physical SI spacing and the CLI multiplied by `gridUnitSI` again, so a 1e-6 m input
  cell became 1e-12 m. The reader now owns the single conversion, and an infeasible
  resulting grid exits with a clear error instead of a `bad_alloc`-style message.

## Known gaps in v0.1

- **Near shell not implemented.** The mesh path has no analytic near-source correction.
  Its accuracy below the crossovers above is a stated limitation, not a tuning target.
- **Anisotropic cells bound near-field accuracy.** When the largest and smallest spacing
  differ by a factor of 2 or more the near-field error is set by the shortest axis, because
  accuracy depends on distance measured in units of the local spacing. The executable warns
  on stderr in that case and computes the request unchanged.
- **Single shared velocity.** A momentum spread exits non-zero; per-particle velocity is
  not supported.
- **Double precision required.** The field pipeline and its accuracy claims are double.
- **Padding required.** With `fftPadding = 1` the FFT box equals the output box and
  source-observer lags beyond half the box wrap. The measured artifact is **9.275e-3**
  (single stationary source, 20^3 box, worst component Ey). The production policy never
  selects 1.
- **Free-space truncation outside the box.** Fields are computed only on the requested
  grid; there is no extrapolation beyond it.
- **No multi-GPU.** One device, one allocation, one node.

## Reproduce

Build and test commands are in [Build](#build). The exact test run for the 12/12 result is:

```bash
git clone https://github.com/psychocoderHPC-bot/picBunch
cd picBunch
export PATH=/usr/local/cuda/bin:$PATH
env -u LD_PRELOAD cmake --preset cuda-v100 \
  -DopenPMD_DIR=/opt/openpmd-hdf5-0.17.1/lib/cmake/openPMD
env -u LD_PRELOAD cmake --build --preset cuda-v100 -j 8
env -u LD_PRELOAD ctest --preset cuda-v100
```

Accuracy and cross-validation use the independent judge
`~/pr24/pr24w53-ref/indfield.py` on `rosi_v100`. The mesh sweep is
`~/pr24/n2-20261102/n2b.py` (`s1` distance, `s2` spacing, `s3` smoothing, `s4`
scale invariance). The cross-validation scripts are in `~/pr24/w7-20261102/`
(`compare_points.py`, `compare_field.py`, `tiny3way.py`) against the reference in
`~/pr24/b1-20261102/`. Checkpoint acceptance artifacts are in
`~/pr24/w7b-20261102/` and `~/pr24/w7c-20261102/`. The performance sweep is
`~/pr24/sb1-20261102/` (`matrix.csv`, `coulomb.csv`, `dsweep.csv`).

The raw measurement data behind the accuracy, refinement and performance claims is
committed under [docs/measurements/](docs/measurements/), with one provenance line and
a SHA256 per file in [docs/measurements/README.md](docs/measurements/README.md). The
rosi working directories named in that file are the reproduction hints; the transient
`/tmp` mirror is no longer the source of record. To verify the committed bytes from the
clone root:

```bash
sha256sum -c docs/measurements/MANIFEST.sha256
```

## License

MPL-2.0, see [LICENSE](LICENSE).
