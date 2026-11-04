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
| toml++ | `30172438cee64926dc41fdd9c11fb3ba5b2ba9de` (v3.4.0) | fetched by CMake `FetchContent`; configuration parser |
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

The documented interface takes one TOML configuration and creates a PIConGPU
checkpoint:

```bash
picBunch --input bunch.h5 --destination checkpoints --config params.toml
```

The schema, the placement rule, the particle-patch layout and the precision
switch are in [docs/configuration.md](docs/configuration.md); a complete example
is [examples/picbunch.toml](examples/picbunch.toml).

The older flags remain available as an advanced path:

```bash
picBunch --input <openPMD file/series> [--species <name|index>] \
         [--output <openPMD file>] [--input-iteration N] \
         [--output-iteration N] [--tolerance 1e-3] \
         [--cell-size <dx>[,<dy>,<dz>]] [--grid <nx>,<ny>,<nz>] \
         [--origin <x>,<y>,<z>] [--backend auto|cuda|hip|cpu] \
         [--near-shell none|exact] [--near-shell-radius <cells>] \
         [--near-shell-tile <nodes>]
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

The default `--near-shell none` keeps the plain padded-FFT mesh, whose
near-source field is only mesh-accurate. `--near-shell exact` replaces the mesh
contribution of every source within the shell by the exact TSC-convolved field
of that source (the vendored FP64 reference quadrature), assembling
`F_mesh(all) - F_mesh(near) + F_exact(near)` with a per-tile local zero-padded
FFT for the subtraction. The mode is opt-in because the exact evaluator costs
quadrature time per (source, node, component). `--near-shell-radius` is in PIC
cells and defaults to `3` at tolerance `1e-3`, plus one cell per decade below
that, capped at `6`. The near set is chosen per `--near-shell-tile` nodes, so
the effective radius exceeds the nominal one by at most half a tile. The three
flags are parsed by the shipped binary and the run summary prints
`near shell: mode=... R=... PIC cells tile=... nodes`. The measured worst-case
envelope is `1e-3` or better for gamma ~ 1, aspect 1 at R >= 3 PIC cells, and for
gamma ~ 10, aspect 1, oblique velocity at R >= 4; gamma 1 at aspect 4 needs about
32 cells and is outside the shell's reach. The exact evaluator costs about
`3.1e-4 s` per (source, node, component) on one 16-thread CPU socket, so it is
opt-in and not lookup-class. `docs/algorithm.md` has the per-case table.

`--tolerance` is the only accuracy knob. It is a relative target for the delivered
free-space convolution: padding 2 is used for targets at or above the measured padding-2
bound (`1e-12`), padding 3 for targets between that bound and the measured double-precision
floor (`1e-14`), and smaller targets are rejected with a non-zero exit. It does not change
the interpolation order, the deposit scheme or the node refinement.

## Global output domains and MPI

By default the computation grid and the written grid are the same. When the bunch is much
smaller than the target field domain, `--global-grid` and `--global-origin` decouple them:

```bash
mpiexec -n 4 picBunch --input bunch.h5 --cell-size 1e-6 \
  --global-grid 2048,2048,2048 --global-origin -1.024e-3,-1.024e-3,-1.024e-3 \
  --output field.h5
```

The convolution still runs on the padded bunch box. The compute box is snapped to global
nodes so every computed value lands on a global node. The global array is never allocated.
Each rank owns one contiguous z-slab, divided into write tiles of about 128 MiB, so the
extra memory per rank is one tile. Rank 0 reads the bunch and broadcasts it; every rank then
computes its share of the field (see below), and all ranks write disjoint `storeChunk`
regions. Extent, spacing, origin, `unitSI` and the Yee `position` attribute are identical
for every rank count.

### Rank-parallel compute

The bunch is replicated, not redistributed: the near shell and the mesh both need every
source on every rank. Two stages are rank-parallel.

* **Mesh by component.** Every rank deposits the same sources and transforms the same
  `rho` spectrum. The six staggered convolutions are divided across ranks; a rank evaluates
  only the components `c` with `c % size == rank`, then one `MPI_Allreduce` sums the six
  arrays. This is component parallelism, not a domain-decomposed FFT. A per-rank FFT on a
  subdomain would lose long-range contributions. Deposit and the `rho` transform are
  replicated and form the Amdahl serial fraction.
* **Near shell by output tile.** The tile loop is partitioned by linear tile index. Each
  rank builds the near-source list and the local subtraction FFT for its own tiles, writes
  the correction only there, and one all-reduce sums the corrections.

Without MPI both paths run serially, and the results are unchanged. `--deterministic-deposit`
is still required for bitwise repeatability of the deposit itself.

#### Measured rank scaling (v0.6.0)

Measured on `rosi_v100` (one V100 per rank, 48 cores, `OMP_NUM_THREADS=48/N` per rank,
`--deterministic-deposit`) after the node had drained. The load average at the start of
each rank block was 1.9, 8.4 and 11.8 for 1, 2 and 4 ranks; the rise is the preceding
measurement, and no foreign process was running. Near-shell times are the near-shell loop;
mesh times are the in-binary stage total; write times are the `writer` stage of a 256^3
global domain (0.77 GB).

| case | configuration | ranks | wall | speedup | efficiency | Amdahl `s` |
| --- | --- | --- | --- | --- | --- | --- |
| near shell | N=128, 32^3, R=1.5, tile 8 | 1 | 36.61 s | 1.00 | 100% | - |
| near shell | same | 2 | 29.60 s | 1.24 | 62% | 0.62 |
| near shell | same | 4 | 27.29 s | 1.34 | 34% | 0.66 |
| mesh | N=2e6, 112^3 | 1 | 522.2 ms | 1.00 | 100% | - |
| mesh | same | 2 | 436.3 ms | 1.20 | 60% | 0.67 |
| mesh | same | 4 | 433.6 ms | 1.20 | 30% | 0.77 |
| write | N=2e6, 256^3, 0.77 GB | 1 | 7.41 s | 1.00 | 100% | - |
| write | same | 2 | 4.34 s | 1.71 | 85% | 0.17 |
| write | same | 4 | 15.49 s | 0.48 | 12% | n/a (I/O-bound) |

The mesh stage breakdown shows the limit directly: deposit is 317, 307 and 332 ms at
1, 2 and 4 ranks, while the source FFT is 88, 58 and 46 ms. Only the six component
convolutions are parallel, so the replicated deposit and `rho` transform cap the speedup.
The near shell improves from 1 to 2 ranks and then plateaus, because the static
round-robin tile split leaves uneven work and every rank still pays the all-reduce of the
full correction array. The collective writer is I/O-bound: two ranks overlap metadata and
flush traffic, but four writers contend on the shared filesystem and run 2.1x slower than
one rank. These limits are algorithmic and I/O limits, not accuracy effects.

### Multi-rank particle patches

`placement.patch_count = N` splits the generated particle patch along z into `N`
rank-overlap regions. Neighbouring regions overlap by one cell; particle ownership stays
disjoint. The writer assigns each particle to the first patch that contains its cell,
reorders the records so each patch owns a contiguous range, and writes cumulative
`numParticlesOffset`. `patch_count = 1` keeps the previous single-patch layout. Use
`placement.patch = "domain"` for a restart.

**Tail margin and far-tail evaluation.** The field outside the bunch does not vanish; it
decays as `1/r^2`. The code first solves the multipole bound for the smallest centre distance
`d`. With `Q` the net charge, `p` the dipole moment, `D` the traceless quadrupole tensor
built from the actual particle positions, `||D||_2` its largest absolute eigenvalue,
`R` the support radius, `gamma_max = max sqrt(1+|u_i|^2)` and `k = 1/(4 pi eps0)`:

```
E_tail(d) = gamma_max * k * ( |Q|/d^2 + 2|p|/d^3 + 3.5*||D||_2/d^4 )
            <= tolerance * field_peak
```

The last term replaces the former worst-case remainder `24 Qabs R^2/(d-R)^4`, which is not
a physical moment. The `gamma_max` factor covers the boosted field of a moving bunch.
The margin in cells is `ceil((d - half_extent_axis)/cell_size_axis)` per axis. The peak used
for the solve is `min(max|E|, c*max|B|)` (a zero B peak is ignored); since
`|B_tail| <= |E_tail|/c`, this keeps both E and B below `tolerance` relative to their own
peaks.

For a charged, extended bunch the monopole alone can still require a large exact box: the
documented 20-cell, 4096-electron bunch (R = 16.85 cells, 1e-6 m cells, peak 45722.5 V/m)
needs 350 cells from the new quadrupole bound and 360 cells from the old one, so the exact
box would exceed the `2^28` padded-node cap at tolerance `1e-3`. Therefore the writer does
not zero-fill the global domain. Beyond a measured switch radius it evaluates the actual
monopole + dipole + quadrupole field (`include/picBunch/FarTail.hpp`). The switch is `6R`;
the measured worst per-component deviation from an independent direct charge sum is
`0.0853 V/m` at `6R` (target `45.7 V/m`, ratio `1.9e-3`) and `0.0166 V/m` at `8R`.
At `2R` it is `48 V/m` and fails, which is why `6R` is the floor. The exact compute box
extends to the switch (plus two cells for the global-node snap) when the quadrupole margin
itself is infeasible. Zero fill now exists only on the explicit diagnostic
`--allow-field-truncation` path. If the domain ends before the exact box, the box is clipped
to the whole domain and no interior hole is written. The margin and switch are reported as
`field tail: ... bound=... V/m` and `far tail: ... switch=... cells`.

The margin rule is checked by `Margin.MarginBoxMatchesLargerReference`: for a dense 4096
particle bunch the field on a box extended by exactly the derived margin differs from a
3x larger reference box by at most `146.5 V/m` per component at `tolerance*field_peak =
2576.8 V/m`. `Margin.FarTailMatchesDirectSum` checks the expansion against the direct sum.

PIConGPU restart on a domain larger than the written one must still treat the written file
as a finite-domain field. The bound above applies inside the written domain; it says nothing
about the field outside the file extent.

**Physics constraint:** the FFT is not domain-decomposed. A per-rank FFT on a subdomain would
lose long-range contributions. The convolution covers the bunch box globally; ranks divide
the six components (and the near-shell output tiles), and the write is decomposed by z-slab.
This is exact only while the bunch box contains the sources that matter, which is the
existing free-space truncation limitation.

MPI is optional: configure with `-DPICBUNCH_ENABLE_MPI=ON`. openPMD must be built with MPI
for one collectively written series. If it is not, each rank writes its own
`<output>.rankNNNN` file and the program prints a warning. That fallback was not exercised
on rosi because its openPMD 0.17.1 has MPI enabled.

```bash
export PATH=/usr/local/cuda/bin:$PATH
env -u LD_PRELOAD cmake --preset cuda-v100 -DPICBUNCH_ENABLE_MPI=ON \
  -DopenPMD_DIR=/opt/openpmd-hdf5-0.17.1/lib/cmake/openPMD
env -u LD_PRELOAD cmake --build --preset cuda-v100 -j 8
env -u LD_PRELOAD OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
  mpiexec -n 4 ./build/cuda-v100/picBunch ...
```

Measured on rosi_v100 (V100, CUDA 12.6, openPMD 0.17.1 with MPI), 20-cell bunch, cell
`1e-6` m, deterministic deposit:

Measured on jureca-h100 (H100 PCIe, CUDA 13.4, openPMD 0.17.0 with MPI) at tolerance
`1e-3`, with the 20-cell 4096-electron bunch above:

| configuration | wall | writer | compute box |
| --- | --- | --- | --- |
| 512^3 global, 1 rank | 54.3 s | 39.8 s | 212^3 |
| 512^3 global, 2 ranks | 43.2 s | 39.8 s | 212^3 |

The 512^3 output is 6.44 GB (six double components). The compute box is the measured
far-tail switch (94 cells) and the outer shell carries the quadrupole field, not zeros.
The 1-rank and 2-rank files have `max_abs_diff = 0.0` over all six datasets and equal mesh
metadata. 1-, 2- and 4-rank runs on a 64^3 domain with `--deterministic-deposit` are bitwise
identical. Without that flag the CUDA deposit uses `atomicAdd`, so two runs differ by at most
2.5e-16 relative even at one rank; use the flag when a bitwise comparison is required.

### Measured single-rank limits

The v0.1 pipeline allocated the output grid and the padded FFT work arrays. Measured on
rosi_v100, GPU 0 (idle for this sweep):

| output N | padded box | peak device | peak host RSS | total time |
| --- | --- | --- | --- | --- |
| 64^3 | 128^3 | 392 MiB | 158 MB | 0.23 s |
| 128^3 | 256^3 | 1.30 GiB | 299 MB | 0.58 s |
| 192^3 | 384^3 | 3.36 GiB | 689 MB | 1.62 s |
| 256^3 | 512^3 | 7.40 GiB | 1.22 GB | 2.96 s |
| 320^3 | 640^3 | 14.1 GiB | 2.38 GB | 6.16 s |

The code rejects any request whose padded node count exceeds `2^28` nodes
(`src/Compute.cpp:244` and `src/Compute.cpp:454`), so 384^3 fails before allocation. A
2048^3 output would need a 4096^3 padded box: 2.75 TB for the five FFT work arrays plus
412 GB for six double field components, about 3.2 TB total. The measured trend of about
460 bytes per output node extrapolates to about 3.7 TiB. With the chunked writer the global
array is not allocated, so a 2048^3 write is bounded by disk (412 GB) and by time (about
1 h at the measured 115 MB/s aggregate write rate), not by memory. The `2^28` limit applies
only to the compute box.

### PIConGPU restart checkpoint

The documented interface writes the checkpoint that PIConGPU's `RestartFieldLoader` can
consume:

```bash
picBunch --input bunch.h5 --destination checkpoints --config params.toml
```

`[picongpu]` supplies `unit_length`, `unit_e`, `unit_b`, `dt`, `step` and `precision`; no
unit or conversion value is accepted on the command line. `[domain]` and `[placement]`
supply the geometry. See [docs/configuration.md](docs/configuration.md).

The legacy CLI path `--restart-format picongpu` remains as an advanced option. It is not
the documented interface and still requires the three advanced unit flags, which are never
needed by the TOML interface:

```bash
picBunch --input bunch_%T.h5 --species electrons \
         --restart-format picongpu --output-dir checkpoints \
         --picongpu-step 0 \
         --picongpu-unit-length 5.773502691896257e-08 \
         --picongpu-unit-e 8850761452125.0 \
         --picongpu-unit-b 29522.962355927582
```

In the advanced path all three unit flags are mandatory; a missing one is an error, never a
default. The writer
emits `checkpoints/<prefix>_<step:06d>.h5` plus `checkpoints.txt`. Fields are float32 with
extent `(Nz,Ny,Nx)`, stored as `field/unit` in PIConGPU internal units. Every particle
species group in the input iteration is copied as a raw HDF5 object, so its attributes and
`particlePatches` survive; a restart needs all of them. `--picongpu-species` only requires
that the named species is present and never drops the others. `--picongpu-dt` sets the
stored `dt`. The single `picongpu_idProvider` range covers the particles of every copied
species (`nextId` is at least their total count). When the input checkpoint contains
`Convolutional PML ...` records, the writer copies those records from the input instead of
recomputing them. See [docs/picongpu-restart.md](docs/picongpu-restart.md).

## Test status

The v0.6.0 release tree passes 18/18 CTest entries on rosi (CUDA 12.6, sm_70) and on
jureca-h100 (CUDA 13.4, sm_90). On hal (HIP, gfx1100) 17/18 entries pass; `picBunch.Io`
reports all 14 executed assertions green and one skip, then segfaults inside
`librocfft.so` during process teardown under ROCm 7.2 (a teardown crash, not a field or I/O
assertion). The v0.5.0 commit passed 18/18 on all three hosts. The entries are
`picBunch.Config` (TOML parsing, defaults and required keys), `picBunch.Placement`
(charge-weighted centring, `bbox` and `domain` patches, out-of-domain rejection, dtype and
`charge`/`mass` records) plus `picBunch.Margin` (margin and far-tail validation) and
`picBunch.NearShellCli`, which runs the shipped binary with `--near-shell exact` and asserts
a finite field that changes inside the shell. The `Domain.GridStaysOnBunchBox` test checks
that a 1024^3 global request keeps the compute box small;
`Domain.ChunkedWriteZerosAndOffset` documents the zero-fill writer path used by the
diagnostic flag.

The v0.1.3 final state was the commit `da3cd4239926f5da31b5794a36f8184e0b3164da`:
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

Extended to non-unit cells (W3n, W3o). One stationary electron at `1e-7` m isotropic cells
agrees to **7.8e-5** relative error at 12 cells against the absolute Coulomb field in V/m.
At `1,1,4` m anisotropic cells the deviation from the point Coulomb is the convolution of the
point with the TSC charge shape, whose physical size is 1.5 cells per axis (1.5 m in x and y,
6 m in z for this cell), not a node-volume or implementation defect. The exact continuous
TSC-convolved field (converged quadrature over the anisotropic shape) and the shipped
free-space mesh, both against the analytic point Coulomb at the E_x Yee point, short axis:

| d (cells) | TSC-convolved continuum | shipped mesh |
| --- | --- | --- |
| 4 | 0.7861 | 0.8684 |
| 8 | 0.9253 | 0.9393 |
| 12 | 0.9638 | **0.9681** |
| 16 | 0.9790 | 0.9807 |
| 24 | 0.9904 | 0.9909 |

The 3.2% gap at 12 cells is the anisotropic TSC convolution itself: the continuum is 3.6%
below the point Coulomb, and the mesh reproduces that continuum to 0.44% (0.05% at 24 cells),
the discrete deposit discretisation, which decays as about d^-2. For isotropic cells the TSC
shape is symmetric, its quadrupole vanishes, and the continuum equals the point Coulomb to
machine precision, which is why the isotropic anchor is 7.8e-5. No node-volume factor is
present or needed: a `1e-7` m case that lacked `cellx*celly*cellz` would be `1e-21` of this.
The `Physics.AbsoluteMagnitude*` tests pin the SI magnitude on every backend.

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

- **v0.6.0** parallelises the mesh component convolutions and the near-shell output tiles
  across MPI ranks, with one all-reduce per field, and adds `placement.patch_count` to
  write one checkpoint patch per rank-overlap region. 1-, 2- and 4-rank mesh and near-shell
  fields are bitwise identical at a fixed problem. The replicated deposit/source FFT cap the
  mesh speedup at about 1.2x, the near shell plateaus at about 1.34x from static-tile
  imbalance, and the collective writer is I/O-bound and slower at 4 ranks.
- **v0.5.0** adds the operator-facing TOML interface
  (`--input/--destination/--config`), charge-weighted centring with `left/front/top`
  placement, `patch = "bbox" | "domain"`, uniform float32 checkpoint records by default
  (float64 opt-in with the stock-PIConGPU conversion warning), and `charge`/`mass` records.
  The documented interface takes every unit and conversion from the configuration, so no
  unit flag appears on its command line. The old flags stay available only as an advanced
  path.
- **v0.4.1** makes `--near-shell`, `--near-shell-radius` and `--near-shell-tile` reachable
  from the shipped CLI (v0.4.0 advertised them but rejected them with exit 2), adds the
  end-to-end `NearShellCli` test, replaces the worst-case tail remainder with a quadrupole
  moment bound, and fills a global output domain beyond the exact box with the measured
  monopole/dipole/quadrupole far tail instead of zeros. The documented 512^3 / `1e-3`
  recipe runs again and 1-vs-2-rank output is bitwise identical. Adds the empirical margin
  and far-tail tests.
- **v0.4.0** adds an opt-in exact near shell (`--near-shell exact`, with
  `--near-shell-radius`/`--near-shell-tile`) that assembles
  `F_mesh(all) - F_mesh(near) + F_exact(near)` from the vendored FP64 reference
  quadrature, and replaces the unconditional zero fill of `--global-grid` output
  with a tolerance-derived field-tail margin. The margin bounds the monopole,
  dipole and a conservative higher-multipole remainder; the compute box expands
  to it (or to the whole global domain when that is smaller), and
  `--allow-field-truncation` is the explicit opt-out. The PIConGPU checkpoint
  writer now copies every input species and writes one checkpoint-wide
  `picongpu_idProvider` id range. The referee-facing audit test battery adds
  padding 2/3, div-E and high-density checks plus two disabled scaffolds.
- **v0.3.1** settles the anisotropic Coulomb anchor and the R2C parity. At `1,1,4` m anisotropic
  cells the field at 12 cells is **0.9681x** the analytic point Coulomb: the exact TSC-convolved
  continuum is **0.9638x**, so the gap is the anisotropic TSC convolution (3.6% below the point
  Coulomb), not a node-volume or implementation defect, and the mesh reproduces the continuum to
  0.44%. The Accuracy section now quantifies the deviation from the point Coulomb against distance
  as an aspect limit instead of bounding it only at 40 cells. The R2C path accepts an odd padded
  extent (parity-correct kernel centring and relaxed even-only guards); `Physics.OddExtentR2cParity`
  fails without it.
- **v0.3.0** adds `--global-grid`/`--global-origin` and optional MPI. The convolution stays
  on the padded bunch box; the global field is written as exact-zero-filled chunks so the
  global array is never allocated. Adds `--deterministic-deposit` for bitwise comparisons
  and `picBunch.Domain` tests. CUDA builds pass 13/13 CTest entries with and without MPI;
  HIP passes 13/13. A 512^3 global domain with a 20-cell bunch writes in 61 s on 2 ranks
  with 316 MB host memory per rank.
- **v0.1.5** adds an absolute-magnitude regression test. W3n claimed the read-out scale
  `kRefinement^3/M^3` was missing the physical node volume `cellx*celly*cellz`. One stationary
  electron at `1e-7` m cells measured **1.00002x** the analytic Coulomb field in V/m, so the claim
  is refuted and the scale is unchanged. The new `Physics.AbsoluteMagnitude*` tests pin the SI
  magnitude: they fail by the full volume factor if `cellx*celly*cellz` is ever applied, and cover
  `1e-7` m isotropic and `1,1,4` m anisotropic cells.
- **v0.1.4** warns on stderr when the largest and smallest cell spacing differ by a factor of 2 or
  more, naming both spacings and the shortest-axis near-field limit; the computation is unchanged.
- **v0.1.3** makes the input mesh define the whole output grid (extent, spacing, origin) when
  present, rejects particles outside it, and keeps the bunch-AABB path for mesh-free input.
- **v0.1.2** honours the input mesh `gridGlobalOffset` as the default output-grid origin;
  `--origin` remains an override.
- **v0.1.1** fixes v0.1 mishandling non-unit `gridUnitSI`. The reader already returned
  physical SI spacing and the CLI multiplied by `gridUnitSI` again, so a 1e-6 m input
  cell became 1e-12 m. The reader now owns the single conversion, and an infeasible
  resulting grid exits with a clear error instead of a `bad_alloc`-style message.

## Known gaps in v0.1

- **Near shell (implemented in v0.4.1).** The v0.1 mesh path had no analytic near-source
  correction; v0.4.1 exposes `--near-shell exact`. Its measured envelope is stated above and
  in `docs/algorithm.md`; it is opt-in because it costs quadrature time per pair.
- **Anisotropic cells set the Coulomb-limit offset.** The delivered field is the TSC-convolved
  field, which for non-cubic cells differs from the point Coulomb by the anisotropic TSC
  convolution: at aspect 4 (z/x) the continuum is 21% below the point Coulomb at 4 cells and
  3.6% at 12 cells along the short axis, decaying as about d^-2 (see the Accuracy table). The
  mesh reproduces that continuum to the discrete deposit tolerance (0.44% at 12 cells, 0.05%
  at 24 cells). This is a quantified aspect limit, not a "cubic cells only" restriction. The
  executable also warns on stderr when the largest and smallest spacing differ by a factor of
  2 or more, and computes the request unchanged.
- **Single shared velocity.** A momentum spread exits non-zero; per-particle velocity is
  not supported.
- **Double precision required.** The field pipeline and its accuracy claims are double.
- **Padding required.** With `fftPadding = 1` the FFT box equals the output box and
  source-observer lags beyond half the box wrap. The measured artifact is **9.275e-3**
  (single stationary source, 20^3 box, worst component Ey). The production policy never
  selects 1.
- **Zero fill only as a diagnostic.** Global output is filled beyond the exact box with the
  measured monopole/dipole/quadrupole far tail, not zeros. `--allow-field-truncation` is the
  only path that writes zeros, and it warns that those values are not bounded.
- **Rank-parallel compute is single-node.** MPI divides the six mesh components, the
  near-shell output tiles and the global write. The bunch is replicated on every rank;
  there is no domain-decomposed FFT and no multi-node redistribution. Tested with 1, 2 and
  4 ranks on one node only. Without `--deterministic-deposit`, the GPU deposit is not
  bitwise reproducible.

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
