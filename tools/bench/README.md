# picBunchBench: field-pipeline benchmark

`picBunchBench` times the public `picBunch::computeField()` pipeline on
synthetic bunches built in memory. It needs no openPMD particle input, links the
same library the CLI uses, and reports the grid/padded FFT box that
`makeGrid()` selected. The tool is opt-in and does not change the compute core,
the CLI, or any public API.

It is the harness behind `docs/measurements/matrix.csv` (the paper's
Table `tab:matrix`), `docs/measurements/coulomb.csv`, and
`docs/measurements/dsweep.csv`.

## Build

`PICBUNCH_BUILD_BENCH` defaults to `OFF`. Configuring still requires an
HDF5-enabled openPMD-api install because the rest of the project needs it; the
benchmark target itself does not link openPMD.

CPU (host backend), any machine:

```bash
cmake -S . -B build-bench -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPICBUNCH_BUILD_BENCH=ON \
  -DPICBUNCH_BUILD_TESTS=OFF \
  -DopenPMD_DIR=<openPMD install prefix or its lib/cmake/openPMD>
cmake --build build-bench --target picBunchBench -j 16
```

CUDA on the H100 node (`jrc0880`, `sm_90`, CUDA 13.4), verified on
2026-11-08:

```bash
export PATH=/usr/local/cuda/bin:/usr/lib/x86_64-linux-gnu/openmpi/bin:$PATH
# nvcc writes its intermediate files here; the container's /tmp is a 16 MB tmpfs
export TMPDIR=/path/on/scratch/nvcc-tmp
cmake -S . -B build-bench -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPICBUNCH_ENABLE_CUDA=ON \
  -DPICBUNCH_ARCH=90 \
  -DPICBUNCH_BUILD_BENCH=ON \
  -DPICBUNCH_BUILD_TESTS=OFF \
  -DopenPMD_DIR=/path/to/openPMD/install/lib/cmake/openPMD  # your install prefix
cmake --build build-bench --target picBunchBench -j 8
```

The install ships `mpicc.openmpi` without an unversioned `mpicc`; if CMake
cannot find MPI, add a shim directory with `mpicc`/`mpicxx` symlinks to the
`.openmpi` names, or pass `-DMPI_C_COMPILER=.../mpicc.openmpi` and
`-DMPI_CXX_COMPILER=.../mpicxx.openmpi`. Use a modest `-j` when `TMPDIR` is
small.

## Environment and hygiene

Pin one GPU and do not inherit preload shims:

```bash
CUDA_VISIBLE_DEVICES=0 env -u LD_PRELOAD ./build-bench/picBunchBench ...
```

`computeField` opens device 0 of the visible set. Before timing, record the GPU,
its clocks, and whether the node is busy:

```bash
nvidia-smi --query-gpu=index,name,clocks.current.sm,clocks.current.memory,utilization.gpu,memory.used --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Only an exclusive allocation gives stable numbers; a shared node adds scatter
to the medians. Do not kill processes you do not own.

## Case groups

All timings use padding 2 and double precision. The documented default
protocol is **2 untimed warm-ups + 9 timed repeats** (`--warmup W`,
`--repeat R` override either count; the counts are reported in the CSV).
The
bunch fills `[0, box_cells]^3` with a deterministic low-discrepancy sequence,
carries one shared +z momentum (`|u| = sqrt(gamma^2 - 1)`), and sets
`weight = 1`, `charge = +e` per macro particle. `--spacing` is the node spacing
(`ComputeOptions::cellSize`).

### Single configuration

```bash
CUDA_VISIBLE_DEVICES=0 env -u LD_PRELOAD ./build-bench/picBunchBench \
  --particles 2000000 --box-cells 20 --gamma 10 --spacing 1 \
  --repeat 3 --warmup 1 --csv single.csv --raw-csv single.raw.csv
```

`--raw-csv PATH` writes one row per warm-up and per timed repeat; that file is
the input to `summarize.py`.

### Paper matrix (`--sweep`)

```bash
CUDA_VISIBLE_DEVICES=0 env -u LD_PRELOAD ./build-bench/picBunchBench \
  --sweep --repeat 3 --csv matrix.csv
```

One device-level warm-up runs first, then the paper matrix. Each row uses the
default 2 warm-up + 9 timed repeats unless `--warmup`/`--repeat` say otherwise.
The order is:

- A particle scaling: box 20, gamma 10, N = 1e3, 1e4, 1e5, 1e6, 2e6
- B box scaling: N = 2e6, gamma 10, box = 20, 50, 100
- C gamma scaling: box 20, N = 2e6, gamma = 1, 10, 100
- D spacing: box 20, N = 2e6, gamma 10, spacing = 1, 1/2, 1/4

Runtime order of magnitude on an H100: the full sweep is a few minutes. The
N = 2e6 rows dominate (tens of ms of compute each); the box-100 and
spacing-1/4 rows cost the most because the FFT box grows to M = 220 and 180.

### Coulomb limit (`--coulomb`, u = 0)

```bash
CUDA_VISIBLE_DEVICES=0 env -u LD_PRELOAD ./build-bench/picBunchBench \
  --coulomb --csv coulomb.csv
```

One stationary particle sits at a node in a 64-cell box. The mesh field at far
nodes along all six axis directions, d = 8, 16, 24, 32, is compared with the
direct TSC stencil sum of the analytic Coulomb kernel. The tool prints
`COULOMB_MAX_REL` on stderr and exits 3 if it is not below `1e-12`. Runs in
under a second.

For one output node `x` and component `c`, the reference is

```
E_exact_c(x) = q * f_c * sum_{a,b,l=-1}^{1} W(a) W(b) W(l) * K_c(x + delta_c*h - (p + (a,b,l)*h))
```

with `q = kElementaryCharge`, `W = tscShape`, `delta_c = kYeeOffset[c]`,
`K_c = pointFieldNormalized`, and `f_c = kCoulombConstant` for
`Ex,Ey,Ez` or `kCoulombConstant / kSpeedOfLight` for `Bx,By,Bz`. The
residual is `r_c(x) = field_c(x) - E_exact_c(x)`. Per node the tool reports:

- `rel_error = sqrt(sum_{c in E} r_c^2 / sum_{c in E} E_exact_c^2)`, the
  E-component residual normalized by the E-component reference norm.
- `l2_abs_error = sqrt(sum_{c in E} r_c^2)`, the absolute L2 norm of the three
  E-component residuals (this is exactly `rel_error * scale`).
- `scale = sqrt(sum_{c in E} E_exact_c^2)`, the E-component reference norm.
- `max_abs_error = max_{c in Ex,Ey,Ez,Bx,By,Bz} |r_c|`, the true elementwise
  maximum absolute residual over all six components at that node. At `u = 0`
  the reference B components are zero, so this is normally an E component; the
  stderr line names the component that owns the global maximum.

### Distance sweep (`--d-sweep`)

```bash
CUDA_VISIBLE_DEVICES=0 env -u LD_PRELOAD ./build-bench/picBunchBench \
  --d-sweep --csv dsweep.csv
```

For gamma 1 and 10, a single particle sits in a 128-cell box and the mesh field
at axis distances d = 2..48 is compared with the analytic moving point field.
The worst relative error over the six axis directions and six components is
reported per d, so the `3e-4` crossover can be read off. Runs in a few seconds.

## Output schema

Timing CSV (`--sweep` and single configuration), one header and one row per
configuration:

```
particles,box_cells,gamma,spacing,padding,Mx,My,Mz,nx,ny,nz,
deposit_min_ms,deposit_median_ms,kernel_min_ms,kernel_median_ms,
fft_min_ms,fft_median_ms,readout_min_ms,readout_median_ms,
total_min_ms,total_median_ms,total_max_ms,
device_alloc_MiB,peak_device_MiB,peak_host_MiB
```

- `M*` is the padded FFT array per axis, `n*` the output grid per axis.
- Stage columns are min and median over the timed repeats, in milliseconds.
  `total_*` is the sum of deposit, kernel, FFT, and read-out per sample; min,
  median, and max over the repeats.
- `device_alloc_MiB` is the exact allocation computed from the padded box and
  particle count (three real `M^3` arrays, two packed complex half-spectra, four
  per-particle arrays). `peak_device_MiB` is sampled with `cudaMemGetInfo` in a
  helper thread; `peak_host_MiB` is the reset `VmHWM` of the process.

`--coulomb` writes `d,rel_error,l2_abs_error,scale,max_abs_error`; `--d-sweep`
writes `gamma,d,worst_rel_E,worst_rel_B,worst_rel_all`. The first four coulomb
columns keep the historic order (`rel_error` at column 2, the L2 residual at
column 3, `scale` at column 4); `max_abs_error` is appended at column 5 so the
old archive can still be read positionally.

`plan_min_ms`/`plan_median_ms` (appended, so older columns keep their
positions) time the otherwise un-attributed region from `computeField` entry to
the first deposit: velocity binning, grid setup, allocations, and the r2c/c2r
plan/workspace build. The existing stage columns keep their meaning; the plan
cost is reported separately and is not part of `total_*`.

`--raw-csv` schema, one row per warm-up or repeat (input to `summarize.py`):

```
row_id,repeat,phase,clock_mhz,plan_ms,deposit_ms,kernel_ms,fft_ms,readout_ms,total_ms
```

`clock_mhz` is left empty by the bench; fill it from the external 2 s
`nvidia-smi` sampler if per-repeat clock attribution is wanted.

## Timing protocol and `summarize.py`

`tools/bench/summarize.py` applies the pre-registered protocol from
`cost-and-noise.md` (P2) to a `--raw-csv` file:

```bash
python3 tools/bench/summarize.py run.raw.csv --b0-iqr 0.015 --out run.summary.csv
```

- Primary statistic is the median with the IQR (P25-P75), plus min/max.
- **Bimodal check first.** If a stage's P75/P25 > 2.5 the row is marked
  `bimodal:<stage>`; that stage is split at the largest adjacent gap and both
  cluster medians with counts are reported. The total is reported as two
  mode-totals (sum of stage cluster medians), never a single number, and no
  sample is dropped.
- **Otherwise** a repeat is discarded when its total is outside
  `[0.7, 1.4] x` the row median, or its median SM clock is more than 10% below
  the row median clock. Fewer than 6 survivors flags the row for a fresh
  9-repeat set.
- The acceptance gate is `IQR/median <= 2 x` the measured B0 IQR floor
  (`--b0-iqr`, a fraction such as `0.015`), replacing the old 2.586% gate.

`python3 tools/bench/summarize.py --self-test` runs synthetic clean, bimodal,
outlier, insufficient-survivor and clock-guard rows.

**Clock pinning is not used by default.** `nvidia-smi -lgc` is technically
possible from a root container, but the paper campaign did not pin, a leaked
lock persists on a shared HPC node after the allocation, and the measured
~3.3% warm floor is handled by the IQR gate and the rejection rule above. Keep
the pre-row idle/foreign-PID gate instead, and record a 2 s `nvidia-smi`
sampler trace (SM clock, power, temperature, throttle reasons) for post-hoc
attribution. A clock-locked B0 A/B is an optional diagnostic only, and must be
released with `nvidia-smi -rgc`.

## Known output quirks (preserved for reproducibility)

`--d-sweep` still reproduces the archived file byte for byte. The coulomb
archive is kept as-is; new runs use the corrected, additive schema above:

- `--d-sweep` writes its header once per gamma value, so the CSV contains the
  header line twice (before the gamma = 1 block and before the gamma = 10
  block).
- The archived `docs/measurements/coulomb.csv` labels its third column
  `max_abs_error`, but its values are the E-component L2 residual now reported
  as `l2_abs_error`. New runs append the true `max_abs_error`; the paper's
  quoted "2.28e-25" is the `l2_abs_error` value for the `d = 16`, `-x`
  direction, not an elementwise maximum.
