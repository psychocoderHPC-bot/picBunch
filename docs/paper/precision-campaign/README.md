# Precision measurement campaign

This directory re-measures every precision figure of the paper on a fresh
picBunch build and reduces the raw fields on the measurement node. The
scaling/performance figures are deliberately out of scope: `F_a`/`F_b`
(`docs/measurements/matrix.csv`) and the H100 MPI rank scaling
(`docs/paper/data/line/h100_mpi_scaling.csv`) are not run here.

## Scope

| case | figures | reference |
| --- | --- | --- |
| `line-w4b` | `F_w4b_rel_error` | sample-mode exact driver, order 12 |
| `line-gamma` | `F_gamma_err_dist` | sample-mode exact driver, order 12 |
| `hires-g*` | `gp_g*` | sample-mode exact driver, order 12 |
| `bdata2-*` | `val_forward_*`, `val_fluct_*` | sample-mode exact driver, order 12 |
| `var1b-1mrad` | `val_1mrad_xy_*` | sample-mode exact driver, order 12 |
| `large1-*` | `large_s*` | sample-mode exact driver, order 4 |
| `multivel` | `F_multivel_dist` | self-reference at `B=256` |

The reference is the header-only FP64 `exact_field_driver` vendored from the
reproduction campaign (`reference/`), run in `--samples` mode so only the plane
or ring points a figure consumes are evaluated. The order and maximum
refinement level per case match the committed measurement records.

## Files

* `cases.json` - the case registry: runs, mesh geometry, reference settings,
  reduction settings and the estimated wall per case.
* `gen_inputs.py` - deterministic openPMD 2.0 bunch inputs (seed-fixed).
* `run_campaign.py` - GPU scheduler, reference builder, provenance writer.
* `reduce_slices.py` - HDF5 field + reference to the renderer slice schema.
* `reduce_line.py` - `w4b`, `gamma` and `multivel` reductions.
* `reference/` - the CPU-only exact driver and its header.

## Build on the measurement node

The validated command on `jureca-h100` (4x H100 PCIe, CUDA 13.4.92, gcc 13.3,
openPMD 0.17.1 with HDF5 and ADIOS2 2.12.1):

```bash
export PATH=/usr/local/cuda/bin:$PATH
export TMPDIR=/home/ubuntu/picbunch-work-20261112/tmp   # /tmp is a 16 MB tmpfs
cmake -S . -B build/paper -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DPICBUNCH_ENABLE_CUDA=ON \
  -DPICBUNCH_BUILD_GTESTS=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DopenPMD_DIR=/home/ubuntu/opt/openPMD-0.17.1-adios2-2.12.1-r2/lib/cmake/openPMD \
  -DCMAKE_PREFIX_PATH=/home/ubuntu/build-adios2-2.12.1
cmake --build build/paper -j 64
```

`-DCMAKE_CUDA_ARCHITECTURES=native` does not work in this container: CMake
cannot see the device during its compiler probe and fails with "no NVIDIA GPU
was detected". H100 PCIe is `sm_90`, so the explicit value is exact.

Measured with warm FetchContent caches: configure 22 s, build 40 s at
`-j 64`. A cold configure adds the alpaka3/alpakaVendor/toml++ clones (about
1-2 minutes). First configure must have `TMPDIR` on the 35 TB scratch mount;
with the default 16 MB `/tmp` `nvcc` fails with "No space left on device". No
CUDA 13.4 or gcc 13 source changes were needed.

The reference driver has no CUDA/openPMD dependency:

```bash
docs/paper/precision-campaign/reference/build_reference.sh \
  /home/ubuntu/picbunch-work-20261112/campaign/bin/exact_field_driver
```

## Launch

```bash
# balance and cost table only
./run_campaign.py --root /home/ubuntu/picbunch-work-20261112/campaign \
  --bin build/paper/picBunch --estimate

# short pilot (4 representative cases, reduced reference order/tolerance)
./run_campaign.py --root ... --bin ... --pilot \
  --only line-w4b,hires-g10,var1b-1mrad,multivel

# full precision campaign (both commits are written into every record)
./run_campaign.py --root ... --bin ... \
  --harness-commit <harness-sha> \
  --build-commit 00fb8f7fb22a53748636c70e32bbdb8dd85c23c4
```

Write each run into a **new** attempt directory. The campaign roots are
append-only: do not reuse a root whose `prov/index.jsonl` already holds a run.

The driver writes a private tree under `--root`:

```
inputs/      generated bunches and the .sources.bin sidecars
run/<case>/  picBunch openPMD HDF5 fields
ref/<case>/  exact-driver sample points, raw output and npz
reduced/     renderer-schema npz/csv (the deliverable)
campaign/    ingest-source exports for the field_export map entries
w4b/, reduced/line/, reduced/multivel/   ingest-source CSV/npz
prov/records/<case>.json   one provenance record per case
prov/index.jsonl, prov/campaign_summary.json, logs/
```

### GPU scheduling

Execution is a two-pool pipeline, not a per-slot sequential loop. Four GPU
workers each own one `CUDA_VISIBLE_DEVICES` slot and pull mesh jobs from a
shared work queue, so every mesh that is ready starts as soon as a GPU is
free. A separate pool (`--ref-workers`, default 8, `--ref-threads`, default
16) evaluates references for different cases concurrently on the 144 cores.
The reference has no dependency on the mesh, so both stages of the same case
start together; a case is reduced once all its mesh and reference jobs finish.

This replaces the earlier LPT loop, which bound each case to one slot and ran
that slot's mesh and reference sequentially: while a CPU-bound reference ran,
its GPU idled and the next case's mesh waited. The old run showed at most two
simultaneous picBunch processes despite four slots. `--estimate` still prints
the LPT cost table; it is a planning aid only and no longer drives execution.

`probe_gpu_parallelism.py` is the short four-case check. It samples
`nvidia-smi` at 5 Hz while `line-gamma`, `line-w4b`, `multivel` and
`hires-g10` run in a fresh attempt directory and writes
`gpu_parallelism.json` (simultaneous-GPU histogram, per-GPU peak/mean
utilisation, per-case slot and wall):

```bash
./probe_gpu_parallelism.py --root <new-attempt-dir> --bin build/paper/picBunch \
  --driver <campaign>/bin/exact_field_driver --harness-commit <sha>
```

## Fixed case semantics

* **bdata2 near shell and source.** `bdata2-forward` and
  `bdata2-fluctuation` run with `--near-shell fast --near-shell-radius 3
  --near-shell-tile 4 --near-shell-fast-order 5`, the flags the committed
  slices recorded. Without them the plain mesh reaches errors of order 1-10.
  Both sources also use the committed diagonal-line geometry
  (`stratified_line`: per-axis sigma 6 cells at centroid (32,32,32)), not a
  three-dimensional cloud. The committed `forward.bunch.h5` and
  `forward_sources.npz` are that line; a three-dimensional cloud of the same
  width leaves the slice errors 1.3-3.6x above the committed ones.
* **multivel source and panel.** `gen_multivel` writes two different
  artefacts. The field run (`mv-e5`, `mv-a5`) reproduces the archived paper
  bunch: N=200, macro weight 5e-13 C, +y drift, and a stratified normal
  `t ~ N(32, 6)` cells along the cube diagonal (`pos = (t, t, t)`), which
  is the per-axis sigma=6, centroid (32,32,32) line present in the committed
  `mvpaper_{e5,a5}.h5`. The 512-source figure panel (`mvpanel-*`) is a
  three-dimensional Gaussian of sigma 3e-6 m at 2.4e-5 m with +z parallel
  momentum and unit weight, matching the committed
  `multivel_{energy,angular}.npz`. The panel is a figure-only sample and is
  not the field source; a three-dimensional sigma=6 cloud at the field grid
  gives about 5.5 kT instead of the committed 24.7 kT.

## Provenance

One `prov/records/<case>.json` per case records **both** the harness commit
(`harness_commit`, also kept as `git_commit`) and the commit the binary was
built from (`build_commit`, default `00fb8f7fb22a53748636c70e32bbdb8dd85c23c4`),
plus the binary path and SHA-256. It also records a SHA-256 of the case
configuration, every mesh and reference command (full argv and environment),
the GPU slot(s) each mesh used, host, UTC start/end, exit status and the
SHA-256 of every produced file. `prov/index.jsonl` and
`prov/campaign_summary.json` summarise the run and repeat both commits and the
binary hash.

## Schema and additive keys

`reduced/` is written in the exact key layout the paper renderers read
(`hires`, `bdata2`, `large1`, `var1b`, `line`). The layout was checked against
`docs/paper/scripts/ingest_measurements.py::required_npz_keys` and
`required_csv_columns`; the pilot `hi_g10.npz` passed the real
`_transform_field_export` with no missing keys.

The `hires` and `bdata2` cases also write the source export the existing
ingest map expects (`campaign/fields/*.npz` with `<plane>_<comp>_pic` and
`_ref`). The `line` cases write `w4b/w4b_paper_metric.csv`,
`reduced/line/gamma_err_dist.csv` and `reduced/multivel/*`. The `large1` and
`var1b` map entries use a `field_h5` source with pinned hashes; the direct
`reduced/` files are authoritative and the manager can repoint those two
entries after a review.

Additive metadata in the slice files (present but not part of the required
key set): `reference_node_density` is not added; `xy_E2_peak`,
`xy_ref_E2_peak`, `xy_B2_peak`, `xy_ref_B2_peak` are already consumed by the
renderers. No charge-density array is added: the renderers take the bunch
support from `sources` (already in the `bdata2`/`var1b` schema) and plot field
projections plus their error rows. A future charge-density panel would need a
new `<plane>_rho` key; adding it here would be unused and is therefore left
out.

## Pilot evidence

Pilot run 2026-11-12 on `jureca-h100`, four cases in four GPU slots, reduced
reference settings (`order 8`, `max_level 4`, `tol 1e-2`):

* `hires-g10` (128^3): 3.8 s wall, mesh compute 1.4-2.9 s.
* `var1b-1mrad` (32x32x16, 64 bins): 15.5 s wall.
* `line-w4b` (8 meshes, 8 references): 21.4 s wall; mesh compute 0.5-2.8 s.
* `line-gamma` (6 meshes, 6 references): 82.8 s wall.
* `bdata2-forward` (64^3, two planes): 100.6 s wall at pilot reference order.
* `large1-g1` (128x128x64, two planes): 143.3 s wall at pilot reference order.
* `multivel` (10 runs, 1-256 bins, 64^3): 15.4 s wall; mesh compute 0.6-2.9 s.

Every pilot output loaded with numpy, had the expected key set and the
expected shapes (`128x128` for `hires`, `32x32` for `var1b`, `128x128`/`64x128`
for `large1`, `64x64` for `bdata2`, `64^3` fields plus `(512, 8)` panels for
`multivel`, 56 rows for `w4b`, 18 for `gamma`). The gamma=1 `w4b` rows
reproduce the committed CSV through `d=12`, and the `large1-g1` centroid
matches the committed case registry exactly. The larger-`d` `w4b` rows differ
because the widened direction lattice samples different worst-case nodes than
the legacy ER1 lattice; the harness reports the fresh measurement, not a
replay of the committed digits.

Production reference settings are coarser in the pilot; the per-case
`est` values in `cases.json` are used for the production plan and are
conservative. See `PILOT.md` for the measured table.

## Out of scope

No scaling/performance case is generated or run. The two excluded sources are
listed in `cases.json::campaign.excluded_out_of_scope`. No figure PDF is
regenerated and no renderer, `figstyle.py`, or `.tex` file is edited.
