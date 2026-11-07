# picBunch

Standalone bunch-field initializer. It reads an openPMD particle series, computes the
free-space electric and magnetic field of the bunch on a uniform grid, and writes the six
Yee-staggered components back as an openPMD series. A PIConGPU-restart-compatible
checkpoint is an optional output mode.

The pipeline is openPMD particles in, free-space hybrid field out, openPMD field written
back. The reader, TSC deposit, moving point kernel, padded FFT convolution, Yee read-out,
openPMD writer, and PIConGPU restart writer are implemented. Written field values are SI:
E in V/m and B in T, with `unitSI = 1.0` and the matching `unitDimension`. A bunch with a
momentum spread is split into velocity boxes and summed over the classes; a single shared
velocity keeps the historical one-box path. `--tolerance` sizes the padded FFT; it is not
a global accuracy target (see [Error budget](#error-budget)). The physics validation
gates, the near-source limitation and their provenance are recorded in
[docs/algorithm.md](docs/algorithm.md).

## Documentation

- **Paper:** [docs/paper/paper.pdf](docs/paper/paper.pdf) covers the method,
  measured cost and accuracy, and PIConGPU checkpoint cross-validation at
  v0.1.3.

## Layout

```
include/picBunch/   public headers (types, TSC deposit, point-field kernel, FFT self test)
src/                host sources with inline ALPAKA_FN_ACC device functors;
                    the separate CUDA near-shell kernel is src/cuda/*.cu
tests/              smoke test (CTest), GoogleTest unit tests and reference CSVs
tools/verify/       independent HDF5 consumer checks (h5py, no openPMD binding)
docs/               algorithm and validation notes
```

## Dependencies

| Dependency | Pin | Notes |
| --- | --- | --- |
| alpaka3 | `e91f6f4d8d1da321a6423a8ad5b4954562c57a9a` | fetched by CMake `FetchContent` from `alpaka-group/alpaka3` |
| alpakaVendor | `33a4a4f401de2b157de195bb68e8ec61d7460cee` | fetched by CMake `FetchContent` from `alpaka-group/alpakaVendor`; FFT via cuFFT/rocFFT/FFTW, including strided FFTW plans |
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
over `PICBUNCH_ARCH`. Device functors are inline in the host sources; the only separate
device translation unit is picked up from `src/cuda/*.cu`.

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
         [--near-shell none|exact|fast] [--near-shell-radius <cells>] \
         [--near-shell-tile <nodes>] [--near-shell-fast-order <n>]
         [--near-shell-oblique-order <auto|N>]
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

**Near-shell regime limits.** The default `--near-shell none` is the safest choice: it
keeps the plain mesh field and makes no regime-specific accuracy claim. `fast` is the
accurate backend for stationary and near-axial motion. At the default order it gives
`1.69e-4` for moving axial gamma 10. On the host, each oblique in-support pair is gated
by the estimated octant residual; when the residual exceeds
`max(1e-6, 0.25*tolerance)` the pair falls back to the validated Lorentz-contracted
reference. The host gate reaches `<=1e-3` for moving oblique in-support: gated pairs
return the reference quadrature and the non-gated residual stays below
`0.25*tolerance` (`docs/measurements/field_error_energy_spread.csv`). The CUDA
in-kernel path has no contracted fallback: gamma-10 oblique in-support measures
`5.34e-2` at `(0,0.5,0)` and `1.50e-1`/`1.54e-1` at two other points, and beta 0.5
oblique measures `9.3e-4` to `1.2e-3`. The in-kernel oblique case is not a `1e-3` path
until the contracted reference runs in the kernel. `exact`
integrates moving in-support points in Lorentz-contracted coordinates, where the
beaming anisotropy moves out of the kernel into a smooth density; against the order-12
`indfield` arbiter it reaches about `1e-4` at cells (1,0,0) and (0,0.5,0) and about
`3%` at the closest in-support node for gamma 10, and agrees to about `1e-3` for the
oblique points. It is therefore a usable reference, but the closest node is not a
blanket `1e-3` path; validate production in-support points against `indfield`.

The default `--near-shell none` keeps the plain padded-FFT mesh, whose
near-source field is only mesh-accurate. `--near-shell exact` replaces the mesh
contribution of every source within the shell by the exact TSC-convolved field
of that source (the vendored FP64 reference quadrature), assembling
`F_mesh(all) - F_mesh(near) + F_exact(near)` with a per-tile local zero-padded
FFT for the subtraction. The mode is opt-in because the exact evaluator costs
quadrature time per (source, node, component). It is kept for comparison; it
is not trustworthy for moving in-support points (see below). `--near-shell-radius` is in PIC
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

`--near-shell fast` replaces that adaptive evaluator with the closed-form port: the
singular part is the exact eight-term `asinh` box (24 primitives) and the remainder is
plain Gauss-Legendre on the subtracted integrand `(S - W*)K` over the TSC knot panels,
split at `s*`; `--near-shell-fast-order` sets the remainder order (default 5, 8,000
nodes in-support). Moving axial velocities use the exact gamma-stretched box. The fast
backend is the accurate one for near-source points. Against the independent `indfield`
reference (order 12, refine 3) it reaches worst-case `7.99e-4` stationary interior
(order 5), `1.96e-5` face, `1.57e-7` outside, and `1.69e-4` axial gamma 10 interior
(order 8). Moving oblique interior goes through the host residual gate and reaches
`<=1e-3`; the ungated octant alone is `8.15e-2` at the default order and is not a
`1e-3` path. The arbiter-validated in-support number is `1.6%`: at the hardest in-support node
of a gamma 10 bunch (`Ey` at cells `(32, 31.5, 31)`, `0.53` cells from source #94),
`fast` reads `2.043e9 V/m` against the order-12/refine-3 arbiter `2.077e9 V/m`.
`--near-shell exact` integrates the in-support points in Lorentz-contracted
coordinates and agrees with the same arbiter to about `1e-4` off-axis and about `2%`
at the closest axial node; it is the reference mode, not a blanket `1e-3` path.
Validate production in-support points against the order-12 `indfield` arbiter. The
assembled field is unchanged outside the near tiles (bitwise) and the seam jump
excess is `2.0e-10` stationary.

`--near-shell fast` runs the same evaluator on the CUDA accelerator when the build
has a CUDA backend and a device is present, and otherwise on the host. The CUDA kernel
evaluates one (component, node, source chunk) per thread with a single accumulator,
using the eight-term `asinh` box, the subtracted-Gauss-Legendre remainder, the
singularity-aware Duffy tetrahedral box for oblique motion (device order capped at 4),
the direct 27-point TSC mesh-near stencil and a uniform source cell list. The
quadrature nodes and weights, the Yee offsets and the E/B units travel as kernel
parameters, so no device-side global table is read. `fast-gpu` (TOML
`accuracy.near_shell = "fast-gpu"`) remains an explicit alias. On rosi_v100 (64^3
output, R = 3, tile 4, 2000 particles, 8.77e6 evaluations) the device path runs in
`2.5 s` against `18.3 s` on the host, a `7.3x` speedup, and the two agree to
`5.7e-15` worst per-component relative. HIP and CPU builds use the host fallback with
one stderr warning.

On `hal` (A30, sm_80) the device mesh-near term evaluates the 27-point TSC stencil
`sum_s w(s) K((node-j)h + yee*h)` per (source, node) pair with the binned
representative velocity, so the per-tile local FFT and its plan setup are gone from
this path. The direct stencil matches the local-FFT subtraction to `2.528e-15` maximum
point relative (2e3 case, `PICBUNCH_NEAR_SHELL_LOCAL_FFT=1`). A uniform cell list with
one cell of `radius` PIC cells replaces the per-tile source scan. On the 200k
production case the near-shell wall falls from `282.0 s` to `231.0 s`, a `1.22x`
speedup (`docs/measurements/field_error_timing.csv`). The kernel is compute-bound near
the FP64 roofline, so this is far below a 10x target. The device in-kernel oblique
accuracy limitation is in the regime limits above; a device `1e-3` path needs the
contracted reference in the kernel and is future work.

On an idle H100 (jureca-h100, `jrc0880`) at 2e6 particles in a 20-cell bunch the
measured mesh compute stages (deposit + kernel + FFT + read-out) are `0.052 s`; the
earlier conservative figure was `0.136 s`. The fast evaluator measures `1.8466e7`
pairs/s stationary at order 5 and `3.9977e6` pairs/s axial gamma 10 at order 8. Pair
counts are `1.7e8` support-only, `1.7e9` for the 3-cell shell and `3.4e9` for the
4-cell shell. The projected wall times below add the measured mesh time to the
near-evaluator time:

| configuration | projected wall (cost floor) | worst per component (Ex, Ey, Ez, Bx, By, Bz) |
| --- | --- | --- |
| mesh only | 0.052 s | no near correction (in-support O(1)) |
| support-only shell, o5 | 9.26 s | 7.99e-4, 3.55e-4, 6.05e-4, 0, 0, 0 |
| 3-cell shell, o5 | 92.1 s | 7.99e-4, 3.55e-4, 6.05e-4, 0, 0, 0 (<=1.6e-7 out) |
| 4-cell shell, o5 | 184.2 s | 7.99e-4, 3.55e-4, 6.05e-4, 0, 0, 0 (<=1.6e-7 out) |
| moving axial support, o8 | 42.6 s | 1.69e-4, 1.68e-4, 1.36e-5, 1.67e-4, 1.69e-4, 0 |

So `1e-3` is reached in a projected `9.26 s` support-only and `184.2 s` with the full
4-cell shell, below a projected `300 s`. Moving axial gamma 10 reaches `1.69e-4` in a
projected `42.6 s` support-only; its 3- and 4-cell shells exceed a projected `300 s`.
These near wall times are the measured standalone H100 kernel rate times the pair
count: the shipped `--near-shell fast` path evaluates the same algebra on the host, so
they are an algorithm cost floor, not the CPU wall time of the CLI. Moving oblique
in-support goes through the host residual gate and reaches `<=1e-3`; the ungated octant
is `8.15e-2` (per component Ex 2.43e-2, Ey 7.42e-2, Ez 6.32e-2, Bx 8.15e-2,
By 2.96e-2, Bz 3.89e-2).
`--near-shell exact` evaluates in-support points in Lorentz-contracted coordinates:
the Li'enard kernel becomes a Coulomb kernel and the beaming structure moves into a
smooth TSC density, so the pyramid singularity cancellation is exact for any velocity.
Tested against the order-12 `indfield` arbiter at gamma 10, the axial points
(1,0,0) and (0,0.5,0) agree to about `1e-4`, the oblique points to about `1e-3`, and
the closest axial node retains about `2%`. The driver uses the same evaluator, so its
moving output is a reference again, but the residual at the closest node is reported
rather than hidden.

### Velocity binning

The mesh kernel is exact for one shared `u` and linear in charge, so a mixed bunch is the
sum over velocity classes. `deriveVelocityBinning` aligns a beam frame to the
charge-weighted mean momentum, partitions the observed `u` extent into a uniform
`N1 x N2 x N3` box grid, and evaluates each box with its centre velocity. The geometry
depends only on the `u` extent and the box counts, never on the membership, so the near
shell can use the same full-bunch binning on every per-tile local mesh and the mesh
subtraction cancels bin for bin. Nodes inside the shell therefore keep the exact
per-particle-velocity field and only the outside-shell field carries the binning error.

The error model is
`modelError = C * Sum_b (q_b/Q) * (gamma_b * Delta_theta_b + (2/gamma_b) * Delta_gamma_b/gamma_b)`,
where `q_b/Q` is the charge fraction of box `b`, `gamma_b` the largest gamma in the box,
`Delta_theta_b` the largest angle to the box representative (the absolute-charge-weighted mean momentum of the box), and `Delta_gamma_b` the gamma range. The angular term
`gamma_b * Delta_theta_b` and `C = 1` are unchanged. The single node-bounded energy term
`(2/gamma_b) * Delta_gamma_b/gamma_b` replaces the former `Delta_gamma_b/gamma_b +
delta_gamma_b` pair, which charged one node effect twice.

The energy term is calibrated on the archived 5% energy-spread points only (gamma 10,
16 and 64 boxes). At those points it gives `3.9e-3` and `9.2e-4` against measured
`1.12e-3` and `2.72e-4`, a `3.4x` conservative over-prediction (down from `26.7x` and
`25.9x`). `C = 1` remains an assumption for the angular term, not a reproduced fit. The
model makes no accuracy claim beyond gamma 10, 5% energy spread and the measured 16-64
box range, and it is not a tolerance guarantee. `--bin-model-constant` overrides it for
diagnostics.

`--velocity-bins N` requests a fixed count and `--velocity-bins 0` (the default) takes the
smallest power of two whose estimate is at or below `--tolerance`, capped by
`--max-velocity-bins` (default 256). When the cap is too small the run exits non-zero and
prints the required count and the error at the cap as separate numbers, for example
`the velocity-bin model requires at least 8388608 boxes for tolerance 0.001000
(estimated error at 8388608 boxes: ...; at the 256-box cap: 1.867e-01)`.
The single-velocity case is detected within `1e-12` relative and returns one box, so its
field is bitwise the unbinned field.

The archived regime points are in `docs/measurements/velocity_bin_calibration.csv`:
a 5% energy spread measures `1.12e-3` at 16 boxes and `2.72e-4` at 64 (reaches `1e-3`
at 64); a 5 degree angular spread measures `0.421` at 16 and `0.295` at 64 and does
not reach `1e-3`, because it scales only as `1/sqrt(N_transverse)`. For the 1 mrad
x-divergence bunch used for the paper (`N = 200`, `64^3` cells, cell `1e-7` m,
`--near-shell fast`), the auto policy chooses 32 boxes (`32x1x1`, estimate `8.35e-4`);
the resulting `B_y` peak is `8.57e2 T` against `B_z = 2.34e4 T` (ratio `3.7e-2`), while
the exact per-particle reference is `B_y = 8.24 T`, ratio `3.5e-4`. The angular case is
accepted and runs, but binning does not reproduce the small `B_y` residual; the
reference needs per-particle velocities.

### Velocity moments

`--velocity-moments off|auto|2|3` (default `off`) targets bunches with a velocity or
energy spread. The six-component point Liénard kernel is expanded in a multivariate
Taylor series about the absolute-charge-weighted mean velocity `u0`. The moment densities
(`20` multi-indices for order 3, `10` for order 2) are deposited once, and each field
component comes from one inverse transform of the moment/kernel spectral sum. The
per-velocity-box deposit/FFT chain is replaced; the binned path remains the fallback.

Measured on 200 000 particles with a 5% energy spread, gamma 10, `64^3` cells and
`--near-shell none` (40 stratified nodes, reference: TSC order-5 far / order-8 refine-1
near, A30 on `hal`; `docs/measurements/field_error_timing.csv` and
`docs/measurements/field_error_energy_spread.csv`): order 3 runs in `1.85 s` against
`35.46 s` for 256 boxes, a `19x` reduction in CLI wall. It lowers the mesh-limited
band error by `10.3x` at `[6,12)` cells and `6.1x` beyond 12 cells against B=256; it is
`1.21x` lower at `[1.5,3)` and `1.31x` lower at `[3,6)`. Order 2 is slightly worse than
order 3 at `[6,12)` and better beyond 12 cells; order 3 is the better default. The full
per-band table is in [Error budget](#error-budget).

Every order is validated, not only `auto`. The code probes the order-3 remainder
`||F3|| / ||F0..Fn||` in k-space, allreduced across ranks: for `auto`/order 3 this is the
existing order-2-vs-3 probe, and for order 2 it estimates the next term of the same
series. A probe below `0.05` keeps the selected order; otherwise the run prints an
actionable one-line warning and falls back to the binned path. `auto` selects order 3
and follows the same rule. The report then carries `used=false`, `fellBack=true` and the
measured `probe`, so a caller can tell a rejected moment run from an infeasible one.
On the 200k gamma-10 case the probe was `1.762e-4` and auto selected order 3
(`2.10 s` CLI). An explicit order can be forced through for diagnostics with
`--velocity-moments-allow-unconverged`; the CLI warns, and the report sets
`allowUnconverged=true`, so the delivered field is clearly marked as unconverged. A
non-finite moment field (any of the six components, or the probe) is never delivered:
the run warns and falls back even when the override is set.

When `--velocity-moments auto|2|3` is set and the auto binning policy fails, the run
warns and runs the moment path instead of aborting at the bin cap. This also applies to
the near-shell mesh setup: the near shell needs a fixed box geometry for the subtraction,
so it uses a bounded fallback binning at `--max-velocity-bins` and keeps the pinned
`velocityMomentsCentre`, rather than aborting before the moment path can run.

Two limitations remain. The moment path always uses the replicated full FFT, independent
of `options.fftMode`; the one-rank and two-rank fields agree to `2.05e-16` relative L2
(`docs/measurements/field_error_energy_spread.csv`), and more ranks do not reduce the
replicated transform cost. The path does not change the in-support `[0,1.5)` error,
which stays at about `0.57` relative L2 because the shared TSC/mesh representation
dominates there; use the near shell for that region.

With `--near-shell`, the expansion centre is pinned to the full-bunch `u0` for every
per-tile local mesh. The subtraction is then the same truncated kernel operator applied
to the near subset, so it cancels exactly the contribution the global mesh added.
Recomputing the centre on a tile would cancel a different expansion and leave a residual
inside the shell, which is why the near-shell code passes the fixed centre down. The
near shell follows the resolved decision: the fixed order when the moment path engaged,
the binned path when auto, size or the scratch guard made the base mesh fall back.

### Error budget

`--tolerance` sizes only the padded-FFT free-space convolution. It is not an end-to-end
accuracy target: the delivered error is the sum of independent terms, and a run that
meets `--tolerance` does not automatically meet the same number end to end. For a bunch
with an energy spread the measured per-band errors are:

| path | CLI wall | [0,1.5) | [1.5,3) | [3,6) | [6,12) | >12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| binned B=8 | 1.75 s | 5.878e-1 | 1.198e-2 | 1.008e-2 | 2.494e-3 | 9.677e-3 |
| binned B=64 | 11.20 s | 5.837e-1 | 1.172e-2 | 9.521e-3 | 1.460e-3 | 4.780e-3 |
| binned B=256 | 35.46 s | 5.779e-1 | 1.110e-2 | 8.378e-3 | 8.104e-4 | 2.505e-3 |
| moment order 2 | 1.82 s | 5.684e-1 | 9.281e-3 | 6.469e-3 | 9.834e-5 | 2.191e-4 |
| moment order 3 | 1.85 s | 5.681e-1 | 9.164e-3 | 6.380e-3 | 7.851e-5 | 4.131e-4 |
| moment auto | 2.10 s | 5.681e-1 | 9.164e-3 | 6.380e-3 | 7.851e-5 | 4.131e-4 |

Relative L2 per Chebyshev distance-to-nearest-source band (cells). Measurement: 200 000 particles, gamma 10,
5% energy spread, `64^3` cells of `1e-6` m, TSC, `--near-shell none`, A30 on `hal`, 40
stratified nodes; reference is the TSC-convolved field (order-5 far, order-8 refine-1
near). Raw rows: `docs/measurements/field_error_energy_spread.csv`; walls:
`docs/measurements/field_error_timing.csv`.

The in-support `[0,1.5)` band is dominated by the shared TSC deposit and mesh
representation error. Every mesh-only path here measures between `0.568` and `0.588`
relative L2, including both moment orders and auto; the moment expansion does not bring
it below this floor. Use the near shell for that region.

Supported regime for this evidence: gamma 10 with 5% energy spread. gamma 100 is outside
the validated regime: even a 5% energy spread makes the Taylor remainder too large, so
the order probe normally rejects orders 2 and 3 and the run falls back to the binned
path. Before the validity guard existed, explicit orders returned nonphysical fields at
gamma 100 (measured peak `2.36e7` for order 3 against `5.11e5` for the binned path on the
C3 gate) while `auto` correctly fell back. Outside the TSC
support the binned path improves with box count, and both moment orders beat B=256 in
every band outside `[0,1.5)`. Order 3 is best up to 12 cells; order 2 is slightly better
beyond 12 cells (`2.19e-4` against `4.13e-4`). Against B=256 order 3 is `1.21x` lower at
`[1.5,3)`, `1.31x` at `[3,6)`, `10.3x` at `[6,12)`, `6.1x` beyond 12 cells, and is `19x`
faster in CLI wall (`1.85 s` against `35.46 s`).

The angular regime is not supported at `1e-3`. A 5 degree per-axis angular spread
measures `0.421` at 16 boxes and `0.295` at 64 against the TSC-convolved continuum
reference (`docs/measurements/velocity_bin_calibration.csv`) and does not reach the
target. On the gate's 1-degree-divergence bunch the auto binning policy requires
`8388608` boxes for `--tolerance 1e-3` (model estimate at the 256-box cap:
`1.867e-1`) and exits non-zero by design.

The per-term controls are:

| term | controlled by | scope and recorded behaviour |
| --- | --- | --- |
| mesh / TSC deposit | `--tolerance` (padding), fixed `kRefinement = 1`, TSC order | `--tolerance` selects padding 2 at or above the measured padding-2 bound (`1e-12`) and padding 3 between that bound and the double-precision floor (`1e-14`); tighter targets exit non-zero. It does not change the interpolation order, deposit scheme or refinement. Near-source continuum error is not bounded by this term. |
| velocity binning | `--velocity-bins`, `--bin-model-constant`, `--max-velocity-bins` | `C = 1` is assumed for the angular term. The energy term is `(2/gamma_b) * Delta_gamma_b/gamma_b`, calibrated on the archived 5% energy-spread points (gamma 10, 16-64 boxes); it is `3.4x` conservative there and claims nothing beyond that range. A 5 degree angular spread measures `0.421` at 16 and `0.295` at 64 and does not reach `1e-3`. This is not a tolerance guarantee. See `docs/measurements/velocity_bin_calibration.csv`. |
| near quadrature | `--near-shell none/fast/exact`, `--near-shell-radius`, `--near-shell-fast-order`, `--near-shell-oblique-order` (TOML `accuracy.near_shell_oblique_order`) | On the host, `fast` gates each oblique in-support pair on the octant residual and falls back to the Lorentz-contracted reference when it exceeds `max(1e-6, 0.25*tolerance)`; gated oblique in-support points reach `<=1e-3`. The CUDA in-kernel path has no contracted fallback: gamma-10 oblique in-support measures `5.34e-2` at `(0,0.5,0)` and `1.50e-1`/`1.54e-1` at two other points, and beta 0.5 oblique measures `9.3e-4` to `1.2e-3`. `exact` uses Lorentz-contracted in-support quadrature: about `1e-4` axial, `1e-3` oblique, about `2%` at the closest axial node. See `docs/measurements/field_error_energy_spread.csv`. |
| far tail | multipole switch radius and exact-box margin | the asymptotic moving multipole fills the global domain outside the exact box: the exact Liénard monopole of the total charge at the mean velocity plus the rest-frame dipole/quadrupole evaluated at the Lorentz-stretched point and boosted back. It is exact for a single point source and in the principal dipole alignments; for an extended bunch it is the leading moving multipole beyond the measured switch radius, not a full Liénard sum. |
| output precision | checkpoint `precision` | the field pipeline is float64; the default checkpoint stores float32 (`6e-8` fractional resolution) unless `precision = "float64"` is requested. |

The mesh term is the only one `--tolerance` controls directly. Pick the near-shell mode
from the near-shell regime limits above, check the binning term or `--velocity-moments`
for a momentum spread, and account for the checkpoint dtype when a downstream consumer
is single precision.

### Precision and accuracy safeguards

The compute path is double precision end to end: deposit, TSC convolution,
FFT, kernel, near-shell quadrature and Yee read-out all run in FP64.  The
default PIConGPU checkpoint is FP32 for loader compatibility; the separate
quantization budget is in the error-budget table above.  The following
accuracy fixes are in the current tree and are covered by tests:

| area | change | measured effect | test |
| --- | --- | --- | --- |
| far tail | exact Liénard monopole plus rest-frame dipole/quadrupole, evaluated at the Lorentz-stretched point and boosted back | one moving gamma-10 source reproduces the point kernel to `<1e-12`; the old isotropic `gamma_max` scaling was `~1000x` high on axis | `Margin.MovingFarTailMatchesPointKernel`, `Margin.MovingDipolarCloudBound` |
| near shell, moving in-support | Lorentz-contracted reference quadrature; the original static path is preserved bit-for-bit | axial off-axis `~1e-4`, closest axial node `~2%`, oblique `~1e-3`; the old axial path was `~9%` to `~10x` | `NearShellAccuracy.*` |
| velocity binning | box representative is the absolute-charge-weighted mean momentum | 5 degree, gamma-10 bunch: global L2 error `13.2% -> 11.2%` at 16 boxes, `4.31% -> 4.13%` at 64 boxes against 128 boxes | `VelocityBins.MeanRepresentativeIsExactForOneParticlePerBin` |
| output boundary | source halo wraps the full TSC support into the padded FFT box | a particle at node 0 deposits `0.669921875 -> 1.0` of its charge | `Deposit.PaddedSourceHaloConservesChargeAtBoundaries`, `Physics.BoundarySourceMatchesLargerComputeBox` |
| checkpoint | PIConGPU macro-momentum weighting and patch-contiguous ordering | weight 100 round-trips `u` exactly; A,B,A,B patches stay inside their extents | `Restart.GeneratedMomentumUsesPIConGPUMacroWeighting`, `Restart.GeneratedPatchesAreContiguousForInterleavedInput` |

The measured tables behind these statements are
`docs/measurements/velocity_bin_representative.csv`,
`docs/measurements/near_shell_device_speedup.csv`,
`docs/measurements/field_error_energy_spread.csv`,
`docs/measurements/field_error_timing.csv` and the test assertions above.  The model
behind the binning term is in `docs/algorithm.md`.

## Chunked particles and grouped far-tail multipoles

**Chunked reading.** `--max-particle-chunk-gb <GB>` (float, e.g. `1.5`) streams
the selected species in chunks of at most that many gigabytes.  The chunk is
read on rank 0 and broadcast, then the field of each chunk is computed on the
same fixed grid and added to the output field; the FFT of each chunk is repeated,
so the option trades memory for FFT time.  It requires explicit
`--grid`, `--origin` and `--cell-size`, and a single velocity class
(`--velocity-bins 1`).  `--global-grid` is not combined with chunking yet.
Measured: 200 particles in 1-particle chunks reproduce the unchunked field to
`4.8e-16` (`docs/measurements/chunking_and_grouping.csv`).

**Grouped multipoles.** `--multipole-group-cells <N>` (default `16`, `0` = one
global multipole) splits the bunch into spatial supercells of `N` PIC cells and
computes one far-tail multipole per occupied supercell.  The tail outside the
exact box sums the group multipoles instead of one global expansion.  For a
two-cluster gamma-10 bunch the worst far-tail error against the direct Liénard
sum at `6R`/`8R`/`12R` falls from `1.72e-3` (global) to `1.81e-4` (grouped,
4-cell groups).  The evaluation cost grows with the number of occupied groups
per tail node, so choose the group size from the bunch extent: large groups are
cheaper, small groups are more accurate near the transition.

## Particle shapes

`--shape cic|tsc|pqs` selects the charge-assignment shape for the deposit
(default `tsc`).  The support radii are 1, 1.5 and 2 PIC cells, matching
PIConGPU's `CIC`, `TSC` and `PQS` species shapes.  The shape is not stored in
the openPMD particle records, so the value must mirror the simulation's
`shape.param`; use the same shape in the PIC run.

The shape changes the deposited field: the FFT kernel remains the sampled point
kernel and the source shape sets the smoothing.  The shape term is a deliberate
bias against the point field, not a mesh error: at 2, 4 and 8 cells the
stationary point-charge smoothing error is `-8.31e-3`, `-5.62e-4`, `-3.55e-5`
for CIC, `-1.20e-2`, `-8.41e-4`, `-5.33e-5` for TSC and `-1.55e-2`,
`-1.12e-3`, `-7.10e-5` for PQS (`docs/measurements/shape_smoothing.csv`).
TSC is the closest of the three to the point field at these distances; PQS has
the widest support and the largest bias here.  Choose the shape for PIConGPU
consistency, not for a smaller point-field error.

The near shell currently supports `tsc` only and exits with a clear error for
CIC/PQS; the mesh path supports all three.

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

* **Mesh by component (replicated FFT, the default).** Every rank deposits the same sources
  and transforms the same `rho` spectrum on the whole padded box. The six staggered
  convolutions are divided across ranks; a rank evaluates only the components `c` with
  `c % size == rank`, then one `MPI_Allreduce` sums the six arrays. This is component
  parallelism. It is a global transform on every rank, not a per-rank transform of a
  truncated subdomain.
* **Distributed FFT (validated, automatic under MPI).** `--fft-mode distributed`,
  or `auto` whenever MPI runs with more than one rank, decomposes the padded grid
  into contiguous z-slabs; single-rank `auto` keeps the replicated path below the
  configured padded-node threshold. Each rank
  performs the local x and y 1-D transforms, transposes with `MPI_Alltoallv` so that z
  becomes local, transforms z, multiplies the kernel spectrum, and inverts in reverse
  order. This is the exact global transform, with one copy of each work array per rank.
  After the inverse transform each rank keeps only its output z-slab of all six
  components, runs the staggered read-out on that slab, and exchanges the output planes
  once per component so every rank owns exactly the contiguous z-slab the chunked writer
  writes. No rank allocates or gathers the full output field. An explicit
  `--fft-mode distributed` on one rank runs the same slab algorithm and provides the
  single-rank reference for the written-file comparison. The replicated path remains the
  default for boxes that fit.
* **Near shell by output tile.** The tile loop is partitioned by linear tile index. Each
  rank builds the near-source list and the local subtraction FFT for its own tiles, writes
  the correction only there, and one all-reduce sums the corrections.

**MPI limitations.** The bunch is replicated, so particle memory does not fall with rank
count. Component parallelism saturates at six ranks (one component per rank); beyond that
the near-shell tile split is the only additional decomposition. There is no
domain-decomposed deposit or distributed particle read. The distributed FFT keeps one
work array per rank on the padded grid, so it needs the box to fit in aggregate memory.
Rank-count comparisons are not bitwise unless `--deterministic-deposit` is set and the
same FFT path is used.

The two FFT paths must not be confused with a third, invalid one. A rank that transforms
only its own subdomain without communication computes a cyclic convolution on the truncated
local box and loses long-range contributions. That is wrong. A distributed FFT with global
transposes computes the same transform as one process and is exact; it is the standard way
to make large boxes fit in aggregate memory. The code currently does the replicated global
transform by default and the transposed z-slab transform when selected.

Without MPI both paths run serially, and the results are unchanged. `--deterministic-deposit`
is still required for bitwise repeatability of the deposit itself.

### Replicated FFT buffer allocation (v0.9.1)

The replicated mesh path allocated its five transform buffers (`rhoReal`, `rhoSpec`,
`kernelReal`, `kernelSpec`, `outReal`) with 3-D `allocUnified` extents. On CUDA and HIP
that allocation pads every innermost row to 128 bytes when `Mx` is not a multiple of 16
doubles. The per-bin `memset` is pitched: it clears the row data but not the padding
bytes. Those bytes lie inside the flat prefix that the contiguous FFT plan and the flat
deposit, kernel and read-out loops read as data. The velocity-bin loop reuses the buffers,
so the second and later non-empty bins read the previous bin's charge from the gaps, and
the field grows with the bin count. The scope is exactly that case: more than one
non-empty velocity bin on CUDA or HIP with a padded `Mx` that is not a multiple of 16.
A single-velocity run clears a fresh buffer once and is unaffected; the distributed path
already used flat 1-D buffers.

v0.9.1 allocates all five buffers as flat 1-D `std::size_t` counts. Every consumer in
this path already uses the dense x-fastest layout and the vendor plans are built with
`inembed == extents`, so the 3-D allocation added padding that nothing used.

The `picBunchFftSizeCheck` tool reproduces the production buffer lifetime (one
allocation, a pitched `memset` and a forward/inverse pass per bin) and compares the
summed multi-bin field with the direct free-space sum of the same deposits. With the
flat buffers the relative error is a round-off constant at every bin count, at both
failing local-box sizes:

| local box M | bins | HIP gfx1100 free-space rel | CUDA sm_70 free-space rel |
| --- | ---: | ---: | ---: |
| 24 | 32 | 5.37e-16 | 4.30e-16 |
| 24 | 64 | 6.38e-16 | 5.32e-16 |
| 24 | 128 | 6.38e-16 | 5.32e-16 |
| 40 | 32 | 3.72e-16 | 6.37e-16 |
| 40 | 64 | 3.51e-16 | 7.01e-16 |
| 40 | 128 | 3.51e-16 | 7.01e-16 |

The same tool with the old 3-D buffers and 128 bins gives 91.6 (M = 24) and 48.9
(M = 40) relative error, so the failure is the allocation and not the FFT.

End to end, the `64^3`, 200-source, 5% energy-spread, 0.1 degree case with
`--near-shell fast` gives `E_x` peaks 1.6118e6, 1.6118e6 and 1.6120e6 V/m at 32, 64 and
128 bins. Before the fix the same case gave 2.30e7 to 2.35e7 V/m for 32 bins and more, a
14x growth.

Single-velocity regression: the 20-cell mesh-only case (one stationary electron,
padding 1, `20^3`) and the accuracy anchor (one electron, `u_z = 10`, `10^3`, padding 2)
produce bitwise-identical field arrays before and after the change (SHA-256 `ad84e38b…`
and `eb379368…`; the anchor stays at 1.65e-7 of the point scale at node (4,4,9)). No
published single-velocity number changes. The power-of-two rounding of near-shell local
boxes used on an earlier diagnostic branch is not needed and is not merged: the flat
allocation alone makes every tested local size exact.

#### Distributed FFT measurements

The generic transform and transpose match a replicated c2c plan to `1.6e-16`,
and the roundtrip is at FFT roundoff.  The end-to-end production field was
revalidated on hal (HIP+MPI, ROCm 7.2) after fixing the device-deposit stride:
one rank with a single source matches the replicated field to `3.2e-16`, and a
200-particle `64^3` run at 4 ranks matches to `5.4e-16` in the written file.
The same 4-rank case on jureca-h100 (CUDA 13.4, sm_90, four H100s) gives
`2.48e-16` global relative and `2.9e-11` worst absolute difference on a
`8.65e4` V/m peak.  The host FFTW backend, with the strided-plan support in
the current alpakaVendor, matches the replicated field to `1.94e-16` at two
MPI ranks.  The regression test `Fft.DistributedProductionMatchesReplicatedWithDeviceDeposit`
covers both the explicit and the auto-selected path with the device deposit.
The previous mismatch was a device TSC deposit that ignored `rhoStride`, so the
distributed complex buffer received the source at half its index; the host
deterministic path was unaffected.

Timing, `256^3` output (`512^3` padded), 128 particles, grid-only run (no writer),
`rosi_v100` with idle GPUs (medians of 3 runs at 2 ranks and 2 runs at 4 ranks):

| ranks | distributed FFT (ms) | transpose (ms) | read-out (ms) | transpose fraction |
| --- | --- | --- | --- | --- |
| 2 | 3809 | 10788 | 1589 | 73.9% of FFT+transpose, 63.6% of the reported total |
| 4 | 1905 | 6782 | 864 | 78.1% of FFT+transpose, 67.9% of the reported total |

On `jureca-h100` with idle GPUs the 2-rank medians are `1237 ms` FFT, `7089 ms`
transpose and `9455 ms` total (85.1% and 75.0% transpose fractions). The read-out is
larger than before because it now scales each local slab and performs the output-plane
exchange; that exchange is included in the read-out time.

Memory, `640^3` output (`1280^3` padded), 128 particles, `jureca-h100`:

| ranks | path | peak host RSS/rank | peak device/rank |
| --- | --- | --- | --- |
| 1 | replicated | rejected by the 2^28 padded-node guard | - |
| 2 | distributed | 41.14 GB | 81.1 GB (unified, at the device limit) |
| 4 | distributed | 20.64 GB | 65.3 GB |

Host RSS falls `41.1 -> 20.6 GB` (2.0x) from 2 to 4 ranks instead of approaching the old
`12.6 GB` replicated-field floor. The six-component output is now slab-local: `6.3 GB`
per rank at 2 ranks and `3.1 GB` at 4 ranks. The three distributed complex work arrays
model to `50.3 GB`/rank at 2 ranks and `25.2 GB`/rank at 4 ranks; the measured device peak
is set by those arrays plus the FFT plan workspace. The only remaining replicated source
constant is the bunch array, read once and broadcast because the deposit and the transform
need every source on every rank. At 128 particles it is about `8 KiB`, so it is not the
dominant term; a many-particle bunch would eventually make it one. The writer also keeps
six write tiles of about 128 MiB (`~0.77 GB`/rank), independent of rank count.

The transposes and the output-plane exchange cost real time even when the box fits in
memory. They move the padded array twice per convolution and dominate the distributed run
at these sizes; memory is the reason to use the distributed path, not speed.

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
not zero-fill the global domain. Beyond a measured switch radius it evaluates an asymptotic
moving multipole (`include/picBunch/FarTail.hpp`): the exact Liénard monopole of the total
charge moving at the charge-weighted mean velocity, plus the rest-frame dipole and
quadrupole evaluated at the Lorentz-stretched observation point and boosted back to the
lab frame. For `beta = 0` this reduces exactly to the static multipole; for one
particle it reproduces `pointFieldNormalized` in every direction. For an extended bunch it
is the leading moving multipole, not a full Liénard multipole sum. The switch is `6R`;
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
2576.8 V/m`. `Margin.FarTailMatchesDirectSum` checks the stationary expansion against the
direct sum; `Margin.MovingFarTailMatchesPointKernel` checks the gamma-10 axial, transverse
and oblique directions against `pointFieldNormalized` to `1e-12`, and
`Margin.NeutralBunchUsesTotalChargeForMeanVelocity` pins the neutral-bunch weighting, and
`Margin.MovingDipolarCloudBound` pins the higher-multipole moving transform.

PIConGPU restart on a domain larger than the written one must still treat the written file
as a finite-domain field. The bound above applies inside the written domain; it says nothing
about the field outside the file extent.

**Physics constraint:** every FFT path must transform the whole padded bunch box. A per-rank
FFT on a truncated subdomain, with no communication, would lose long-range contributions and
is not used. The default path replicates the whole box; the optional distributed path
decomposes it by z-slabs and restores the global transform with all-to-all transposes. Both
compute the same global convolution. The convolution is exact only while the padded box
contains the sources that matter, which is the existing free-space truncation limitation.

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

`[picongpu]` supplies the independent PIConGPU deck inputs `delta_t`, `cell_size`,
`base_mass`, `base_charge`, `base_density` and `typical_particles_per_cell` (plus
optional `speed_of_light`), together with `step` and `precision`. picBunch derives
every SI factor (`unit_time`, `unit_speed`, `unit_length`, `unit_mass`, `unit_charge`,
`unit_energy`, `unit_efield`, `unit_bfield`) exactly as
`include/picongpu/unitless/simulation.unitless`, so the operator repeats only the deck
setup and never transcribes a unit. The inputs must match the restart deck, not the deck
that produced the input bunch: PIConGPU restart ignores the particle `unitSI` and
interprets momentum in `sim.unit.mass()*unit_speed`, so a mismatch scales every
momentum. `[domain]` and `[placement]` supply the geometry. See
[docs/configuration.md](docs/configuration.md).

Placement was validated end to end on `rosi_v100`: a 12-marker bunch placed at xz middle
and y first quarter of a 32x64x32 domain has centroid fractions
`(0.500000, 0.250000, 0.500000)`, restarts in PIConGPU for one step, and shows no axis
permutation or mirror; the per-particle displacement follows `dt*c*u/gamma` to
`4.1e-7` cells with `unit_mass = sim.unit.mass()`. With an electron-mass `unit_mass` the
restarted momentum is 1e4 too large. Evidence:
`docs/measurements/picongpu_placement_validation.md`.

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

The v0.9.0 release tree passes 19/19 CTest entries on rosi_v100 (CUDA 12.6,
sm_70) and hal (HIP, gfx1100, ROCm 7.2); jureca-h100 (CUDA 13.4, sm_90) passes
19/19 on each merged source branch. The v0.9.1 tree passes 19/19 on hal and on
fwk394 (CUDA 12.5, sm_70). The rosi_v100 route was unavailable during the v0.9.1
release, so the v0.9.1 CUDA checklist ran on the sm_70 node fwk394.
The ROCm 7.2 `librocfft` teardown segfault is gone: the HIP build now calls
`rocfft_cleanup()` after the plans are destroyed while the queue and runtime are still
alive. The previously failing `picBunch.Io` entry passed five consecutive repeats on hal.
The entries are
`picBunch.Config` (TOML parsing, defaults and required keys), `picBunch.Placement`
(charge-weighted centring, `bbox` and `domain` patches, out-of-domain rejection, dtype and
`charge`/`mass` records) plus `picBunch.Margin` (margin and far-tail validation) and
`picBunch.NearShellCli`, which runs the shipped binary with `--near-shell exact` and asserts
a finite field that changes inside the shell, and `picBunch.VelocityBins` (binning model,
single-velocity equivalence, box assignment and the auto-policy error). The `Domain.GridStaysOnBunchBox` test checks
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

- **v0.9.1** fixes the replicated FFT buffer allocation. Five transform buffers were
  3-D `allocUnified` extents, which pad each innermost row to 128 bytes when the padded
  `Mx` is not a multiple of 16; the pitched per-bin `memset` left those bytes uncleared,
  and the contiguous plan read them as data. Reused across the velocity-bin loop, the
  gaps carried the previous bin's charge into the transform and the field grew with the
  bin count. The buffers are now flat 1-D counts. The defect affected only multi-bin runs
  on CUDA/HIP; single-velocity fields are bitwise unchanged, and the power-of-two local-box
  rounding from the diagnostic branch is not merged. Adds `picBunchFftSizeCheck`. The
  v0.9.1 tree passes 19/19 CTest on hal (HIP) and fwk394 (CUDA 12.5, sm_70); the rosi_v100
  route was unavailable at release time.
- **v0.9.0** adds the optional z-slab distributed FFT (`--fft-mode distributed|auto`)
  and the device near-field backend (`--near-shell fast-gpu`). The distributed path
  keeps the six-component output slab-local, exchanges the output planes once per
  component, and writes each rank's slab directly; it agrees with the replicated path to
  `5.4e-16` relative and the written 1-, 2- and 4-rank files are bitwise equal. Host
  memory per rank falls `41.1 -> 20.6 GB` at `640^3` output, but the transposes dominate
  the run time and the bunch stays replicated. The `fast-gpu` backend runs the fast
  near-shell algebra on CUDA, agrees with the host fast path to `5.4e-16..8.0e-16`
  relative on H100 (stationary, axial gamma 10, oblique), and falls back to `fast` on
  CPU/HIP builds or when no device is present. The v0.9.0 tree passes 19/19 CTest on
  rosi_v100 and hal; the merged source branches each pass 19/19 on jureca-h100.
- **v0.8.0** adds tolerance-driven velocity binning for mixed bunches. A momentum spread
  no longer exits with the single-shared-velocity error: the field is summed over
  velocity boxes, the near shell stays bin-free because every pair keeps its own velocity,
  and the near-shell fast backend and its full mode forwarding, the HIP teardown guard and
  the MPI component/tile state are preserved. `--velocity-bins`, `--bin-model-constant`
  and `--max-velocity-bins` expose the policy; an infeasible auto request exits with the
  number of boxes needed. The shipped value `C = 1` is an assumption, not a
  refitted calibration. Measured accuracy: a 5% energy spread is `1.12e-3` at 16 boxes
  and `2.72e-4` at 64, so it meets `1e-3` at 64 boxes; a 5 degree angular spread does
  not reach `1e-3`. The `picBunch.VelocityBins` test brings the suite to 19/19 on all three hosts.
- **v0.7.0** adds `--near-shell fast` (TOML `accuracy.near_shell = "fast"`,
  `--near-shell-fast-order`): the exact eight-term `asinh` box plus a gamma-stretched
  axial box and plain Gauss-Legendre on the subtracted `(S - W*)K` remainder, order 5 by
  default. It reaches `7.99e-4` stationary and `1.69e-4` for moving axial gamma 10 at
  order 8, so `1e-3` is met end to end; moving oblique in-support remains `8.15e-2`
  (documented limit). `--near-shell exact` stays available for comparison. Also fixes a
  dispatch bug that produced the plain mesh field for `--near-shell fast`, and guards the
  ROCm 7.2 `librocfft` teardown segfault with `rocfft_cleanup()` after plan destruction.
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
  correction; v0.4.1 exposes `--near-shell exact`; `--near-shell fast` adds the closed-form
   evaluator. Its measured envelope is stated above and in `docs/algorithm.md`. The fast
   path is the accurate one: it validates against the order-12 `indfield` arbiter to
   `1.6%` at the closest moving in-support node, while the vendored exact backend and the
   driver reference overestimate that node by about `10x` (they share a quadrature).
- **Anisotropic cells set the Coulomb-limit offset.** The delivered field is the TSC-convolved
  field, which for non-cubic cells differs from the point Coulomb by the anisotropic TSC
  convolution: at aspect 4 (z/x) the continuum is 21% below the point Coulomb at 4 cells and
  3.6% at 12 cells along the short axis, decaying as about d^-2 (see the Accuracy table). The
  mesh reproduces that continuum to the discrete deposit tolerance (0.44% at 12 cells, 0.05%
  at 24 cells). This is a quantified aspect limit, not a "cubic cells only" restriction. The
  executable also warns on stderr when the largest and smallest spacing differ by a factor of
  2 or more, and computes the request unchanged.
- **Velocity binning.** A momentum spread is supported through velocity boxes with the
  measured limits above. Binning replaces each particle by its box-centre velocity, so it
  does not reproduce the small `B_y` residual of an angular spread; that case needs
  per-particle velocities. An auto request above the box cap exits with the needed count.
- **Double precision required.** The field pipeline and its accuracy claims are double.
- **Padding required.** With `fftPadding = 1` the FFT box equals the output box and
  source-observer lags beyond half the box wrap. The measured artifact is **9.275e-3**
  (single stationary source, 20^3 box, worst component Ey). The production policy never
  selects 1.
- **Zero fill only as a diagnostic.** Global output is filled beyond the exact box with the
  measured monopole/dipole/quadrupole far tail, not zeros. `--allow-field-truncation` is the
  only path that writes zeros, and it warns that those values are not bounded.
- **Rank-parallel compute is single-node.** MPI divides the six mesh components, the
  near-shell output tiles and the global write. The bunch is replicated on every rank: it
  is read once and broadcast because the deposit and the transform need all sources. The
  distributed FFT removes both the replicated padded box and the replicated output field;
  each rank keeps and writes only its own z-slab. At 128 particles the replicated bunch is
  not a memory limit, but a large bunch would become the remaining constant. The near shell
  still uses the replicated mesh path because its correction buffers are full-grid host
  arrays. There is no multi-node redistribution. Tested with 1, 2 and 4 ranks on one node
  only. Without `--deterministic-deposit`, the GPU deposit is not bitwise reproducible.

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
