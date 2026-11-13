# Measurement data

Raw, text-only measurement tables behind the accuracy and performance claims in
the README and the paper. No HDF5 files, checkpoints, NumPy archives or build
output are stored here. Each entry gives the measured quantity, the host, the
code revision it belongs to, the measurement date and the file SHA256. The same
hashes are listed in `MANIFEST.sha256`.

The campaign was run as `pr24-20261102` on `rosi_v100` (node `gv002`), except for
the MV1 velocity-binning points, which come from `jureca-h100`; each entry states
its own host. The original working copies live under
`/home/ubuntu/pr24/<task>-20261102/`; that path is the reproduction hint, not a
requirement for reading the data.

## Performance and Coulomb limit (SB1)

- `matrix.csv` — scaling matrix (N x box), per-stage medians, plus appended `plan_min_ms,plan_median_ms`. **Re-measured on `jureca-h100` (node `jrc0880.jureca`, 4x H100 PCIe 80 GB, `sm_90`, CUDA 13.4.92), 2 warm-up + 9 timed repeats, 2026-11-08.** The committed table behind `tab:matrix` is now this H100 re-measurement (2+9 protocol, clock 1755 MHz); the original V100 values are superseded. Protocol and per-row classification in `matrix_h100_20261108.md`. Commit: picBunch `e1ddfb5` (+ `47f2436`); bench source `tools/bench/bench.cpp` sha256 `b9cff799f466bbb6900605a9af7859a0d675448fdba94cf8cb6004078134a3ec`, `tools/bench/summarize.py` sha256 `562c4cd18475392f79fef98f67b57a305856bc6f1c2371e121bf397cea4c5214`. This file carries the per-row gate-consistent totals (kept repeats, one outlier dropped where the pre-registered rule fired) and the stage medians quoted in `tab:matrix`. The paper's reduced copy, `docs/paper/data/scaling/matrix.csv`, is a byte-identical row copy of this file plus the appended classification columns `row_class`, `total_iqr_ms`, `mode_total_low_ms`, `mode_total_high_ms`, `mode_total_iqr_ms`, `mode_low_n`, `mode_high_n`, `n_used`, `bimodal_stage`; it is what `F_a`/`F_b` render from, so every plotted point lands on a table value. SHA256 `e1bc403866301410c2ce6d481dcc292c402cfa795b52290df947afa2d673b72c`.
- `matrix_h100_20261108.md` — the H100 re-measurement record for `matrix.csv`: host, stack, binary/source sha256, 2+9 protocol, B0 gate (12.4956 %), the 14 per-row medians/IQRs, the two bimodal rows (100-cell and h=0.25) with their two modes and counts, and protocol adherence. Date 2026-11-08.
- `coulomb.csv` — Coulomb-limit check of the zero-padded FFT convolution against the direct discrete TSC sum. Re-measured on the H100 stack with `matrix.csv` (byte-identical to the earlier `r9` snapshot). Same host and date as `matrix.csv`. SHA256 `b483e021042d8bcb84cd6f8081a3ff52d78b9300985728c48129c907cfb49584`.
- `dsweep.csv` — d-sweep of the convolution error. Re-measured on the H100 stack with `matrix.csv` (byte-identical to the earlier `r9` snapshot). Same host and date as `matrix.csv`. SHA256 `e3687a3df32333b249837c3163703882073f0a6e66791c1a514aeceb10042b70`.

## V100 MPI rank scaling (retained, not in the paper)

- `mpi_writer_scaling.csv` — collective openPMD/HDF5 writer scaling on one
  `rosi_v100` node, run 2026-11-06 (256^3 nodes, six double components,
  0.80 GB, rosi local filesystem). The reduced plot table is
  `docs/paper/data/line/mpi_scaling.csv`. The V100 MPI figure
  `F_mpi_rank_scaling` is intentionally not part of the paper: the manuscript
  includes only the H100 figure `F_mpi_rank_scaling_h100`. The data and the
  renderer (`docs/paper/scripts/render_scaling.py::mpi`) are kept so the
  earlier measurement stays reproducible; no shipped PDF or provenance record
  claims the V100 figure. SHA256 `89421ae17aae70d7e3cc6e729e84468704cc54929d593d94398ff303a684df3b`
  (source) and `88b4a9d176a87df1b1d18f43b14ca9ca3a6bf132b46fd18d9e1537b131ca4499`
  (reduced table).

## Refinement and h-scaling (F4)

- `f4_per_spacing_before.csv` — per-h field errors before the scale fix. Host `rosi_v100`, node `gv002`; PIConGPU `experimental/mesh-h3` at `fef43bbd231595fed2795d26affd5d55b9ebd379`. Date 2026-11-03. SHA256 `c15cafcde3101547f81e83c970fe9134bcacc1a4bc0a52a62790a4f339338fbf`.
- `f4_per_spacing_after.csv` — per-h field errors after the scale fix. Same host and PIConGPU revision as `f4_per_spacing_before.csv`, date 2026-11-03. SHA256 `ca78a419ca616babeee38d183e7989bb4426735b7aca61bfb04cd9ec94da6303`.
- `f4_refine_gate_before.csv` — refine-gate raw rows before the fix. Same host, revision and date. SHA256 `9e99a7fd4e3d1b6f6172a458156b1f278f1549b492b8d2123fdfe827cd3a5ab1`.
- `f4_refine_gate.csv` — refine-gate raw rows after the fix. Same host, revision and date. SHA256 `069f53b7e809bf0c8f8ffd76fd5b95549c7a5d1b0954dbc299934cd8d7e884a9`.
- `f4_refine_gate_before_summary.csv` — summary gate verdict before the fix; worst relative change 2.476e-1, gate 2e-3, FAIL. Same host, revision and date. SHA256 `1cb2ac9eed712bc5a6cd100505b281ff1215d8ce8754db431fa277b23230e5d9`.
- `f4_refine_gate_summary.csv` — summary gate verdict after the fix; worst relative change 9.132e-4, gate 2e-3, PASS. Same host, revision and date. SHA256 `96cbbb1c2260c4f2e37431e7e8d342cf71674b1a1582e7ead6a3cb0f0ee2f8ed`.

## Accuracy (W4)

- `w4b_rel_error.csv` — per-component relative error versus cell distance for the paper metric: worst over `Ex,Ey,Ez` of `|ref-mesh|/max(|ref|,|mesh|)` with a per-component floor `|ref|>=1e-3` of the component peak, against the order-12 independent continuum reference (refine 3). Re-measured on `jureca-h100` (node `jrc0880`, 4x H100 PCIe 80 GB), 2026-11-09; picBunch `e1ddfb55eff92dc12cc2067b52253aa218914ff5`, binary SHA256 `d6c964ae775dbc2f9c23cc66b81e201388284392fe54c09a3b271de48f4f8275`. Gammas 1/10/50/100, aspects 1/4, `d=2..32`; the aspect-4 curves coincide with aspect 1 to machine precision at padding 1. The legacy rosi_v100 `bfaa1e26` table is superseded; its gamma-10 aspect-1 series is reproduced within 1-14%. SHA256 `04e170d7f8a690dcfc281c2926d70e48c7aebd9fd09ef1db28a8d8f144197dea`.
- `accuracy_results.json` — full W4 accuracy results, per case and per distance, both references and all components. Host `rosi_v100`, node `gv002`; picBunch `bfaa1e26d6c2aff52f7b74b729d068be253e4d30`. Date 2026-11-03. SHA256 `b5b68f07c4083d2744b36d3e5376350013e589481f4020c97b418485b464dc6c`.

## Reference identification and N-scaling (W7t)

- `w7t_reference_and_scaling.md` — Q1 answer (what the exact-quadrature reference computes, with source line references) and Q2 N-scaling verdict. Host `rosi_v100`; picBunch `339ca7ad3903249ebf3859f5e106d63db4dc3df5`. Date 2026-11-03. SHA256 `35a3a77f59cc3618d2d16548dd9b81616171ebd39a618fc868b0f433c7331590`.
- `w7t_q2_table.log` — raw N-scaling table printed by the Q2 script. Same host, revision and date. SHA256 `064c18771003e9b42ae7cf30698131afa98bd208ef86ab89526595eda71d4216`.

## Velocity binning (MV1)

- `velocity_bin_calibration.csv` — recorded 16- and 64-box worst-case binning errors for the 5% energy-spread and 5 degree angular-spread cases, with host, campaign commit, date and source. Host `jureca-h100`, CUDA 13.4; campaign `pr24-20261102`, commit `02891fe6`; date 2026-11-02. These are the endpoints quoted in the root README; the original 1/2/4/8-box rows, per-box model sums and `--fit` output were not archived, so `C = 1` is an assumption, not a refitted calibration. SHA256 `f024347ec66287c74585f80f79e382c327d9b3000ba75cbb33921abb3c92425b`.
- `velocity_bin_calibration_fit.txt` — output of `tools/velocity-bins/calibrate_velocity_bins.py` from the CSV. It records the archived points, the target verdict and the statement that `C = 1` is assumed. Generated at repository commit `6fab4a3` on 2026-11-06. SHA256 `01b21daa1a89b807bed6529f1bf58667ed21eae4a89afaca5b09c55e09225c1e`.

## Field-error and energy-spread gate (integrated N2/N3/N5 branch)

- `field_error_energy_spread.csv`: banded relative L2 for binned `B=8/64/256` and `--velocity-moments 2/3/auto` at 200 000 particles, gamma 10, 5% energy spread, `64^3` cells of `1e-6` m, TSC, `--near-shell none`, 40 stratified nodes; reference is the TSC-convolved field (order-5 far, order-8 refine-1 near). Also records the single-velocity branch-vs-main equality, the one-rank vs two-rank moment equality, the auto-bin cap errors, and the near-shell host/device accuracy rows. Host `hal`, A30, CUDA 13.4, sm_80; branch `field-error-energy-spread-20261107` at `ca4163cff964f735fc7202a8cc743fdb4ff34259`. Date 2026-11-07. SHA256 `a6db6f1520f6f6fafd57d78f353b94e55a808018aec2ded373fbd0f556cd9c34`.
- `field_error_timing.csv`: CLI, compute, near-shell and peak-RSS medians for the same paths, plus the 2e3-particle near-shell equivalence run and the 200 000-particle production near-shell run (231.0 s against the pre-change 282.0 s). Same host, revision and date as `field_error_energy_spread.csv`. SHA256 `2b489150b02bba9b2eb7756af8e4ef2e1d08e48a4d58c23ea53161840e5050a9`.

## Fast-near exact-field reference (D2)

- `fastnear_reference_52pt.csv` — exact CPU reference for the fast-near 52-point
  set used by the fast-near validation table: per configuration 20 interior
  points (`|s| <= 1.35` cells), 16 face points (one coordinate at +/-1.5 cells)
  and 16 outside points (one coordinate at +/-1.6 to 3.6 cells). Configurations
  are `g1` (`u = 0`), `g10-axial` (`u = (0,0,sqrt(99))`) and `g10-oblique`
  (`u = sqrt(99) (1,2,1)/sqrt(6)`); each point stores all six normalized field
  components. Values come from the host-independent
  `tools/velocity-bins/indfield.py` at order 12, refine 2, unit cells, source at
  the origin. Host `fwk394`, integration tip
  `a70c7e98620bcd0aaea71f7c11c96d0c822317d6`, two runs byte-identical.
  SHA256 `15bf50984349ba94d561bbed7de8da4376f9f1e3fbb66d38da40cc92ea488c81`.
- `fastnear_reference_52pt.provenance.json` — generator and reference-module
  hashes, parameters, seed, host, determinism and self-consistency records.
- `fastnear_fast_52pt.csv` — measured picBunch fast-near evaluator values at
  the same 52 points, one row per point with all six normalized field
  components, evaluated against `fastnear_reference_52pt.csv`. CPU-only and
  header-only (no alpaka, CUDA or FFTW); two runs byte-identical. Host `hal`,
  final commit `69bfd77d03493a10e0da4c8fdad25c6480c0582a`, g++ 13.3.0. The final
  default (base order/ray plus exact observer-cell routing, gamma>=100
  auto-exact) gives `3.71e-4` stationary, `1.40e-4` moving axial and
  `1.94e-3` moving oblique interior; face and outside rows reproduce the
  published table to <1%. The former `5.34e-2`/`1.50e-1`/`1.54e-1` device
  in-kernel oblique values and the `8.15e-2` ungated-octant value are retired
  for the observer cell. SHA256
  `a30479eb50ec002194b1424edd8bb1b37605c2f8f68fcf0042d5e42b050df461`.
- `fastnear_fast_52pt.provenance.json` — tool command, base commit,
  compiler, reference-input SHA256, determinism and self-test records.

Regenerate from the repository root:

```bash
python3 tools/verify/gen_fastnear_reference.py
python3 tools/verify/gen_fastnear_reference.py --check docs/measurements/fastnear_reference_52pt.csv
```

The generator needs only NumPy. It does not run picBunch and does not need a
GPU. The point set is the one defined by the original validation driver
(`default_rng(12345)`, fixed case and region order); the CSV is written with
stable formatting so repeated runs are byte-identical.

The fast-near values are evaluated by the CPU-only driver:

```bash
g++ -std=c++20 -O2 -Iinclude tools/verify/pb_fast_near_52pt.cpp -o pb_fast_near_52pt
pb_fast_near_52pt --self-test
pb_fast_near_52pt --ref docs/measurements/fastnear_reference_52pt.csv --out docs/measurements/fastnear_fast_52pt.csv
```

It needs only a C++20 compiler; it links no alpaka, CUDA or FFTW. The
`--self-test` checks determinism and fast-vs-exact wiring; the `--ref` run
reproduces the committed `fastnear_fast_52pt.csv` byte-for-byte.

## Padding and R2C/C2R (N1b)

- `n1b_padding_r2c.md` — wraparound padding sweep, per-component padding detail, R2C/C2R equivalence and the bin rule. Host `rosi_v100`, CPU-only, OMP 16 threads; Python model of the picBunch `kRefinement = 1` mesh convolution, no repository commit. Date 2026-11-03. SHA256 `8146c07c6b5b989b33d0c40a299e0b7cfbe3a52958fc8be213d4d6b3dd43d8c4`.
- `n1b_padding_distance.csv` — raw padding-versus-distance rows behind `n1b_padding_r2c.md`. Same host and date; Python model, no repository commit. SHA256 `574c79371755fdf355a8ea6a59c5b3cb3e80c12aabb12bfa8afe052cf19f72a7`.

## 4-rank restart on a decomposable grid (V6b)

- `picongpu_restart_4rank.md` — 4-rank PIConGPU restart of a picBunch-written openPMD checkpoint: 24 x 24 x 96 global grid, `-d 1 1 4`, per-rank 24 x 24 x 24, no auto-adjust; 256-electron compact bunch (per-slab counts [0, 133, 123, 0]); exit 0, all six E/B components bitwise, `max_abs_diff = 0.0`. Host `jureca-h100`, node `jrc0880`, 4x H100 PCIe, CUDA 13.4.92; picBunch `d6c964ae`; PIConGPU 0.9.0-dev deck `b87c053`, ADIOS2 2.12.1, openPMD 0.17.1. Date 2026-11-08. SHA256 `9186b513eceb3dfea613aed2c6439185d4fe05ac3b80bd76c0b25ac5b51dd968`.

## Cost table and writer-inclusive e2e, BP5 re-measurement (2026-11-08)

- `cost_e2e_bp5_20261108.md` — clock/throttle probe (no clock cap: load at the full 1755 MHz, throttle reason 0x0), the cost rows R=1.5/3/4 at 2e6 particles on the centred 20-cell drawer (printed pair counts confirm the geometry), the v0.9.1 A/B decomposition of the R=1.5 gap (1.42x final-vs-v0.9.1 device path, 1.26x v0.9.1-here-vs-published host/stack), and the writer-inclusive e2e rows (BP5 here vs HDF5 in the published rows), including the g20 readout bimodality. Host `jureca-h100`, node `jrc0880`, 4x H100 PCIe, exclusive `dc-h100` allocation, ADIOS2 2.12.1 + openPMD 0.17.1 (BP5), HDF5 1.10.10, gcc 13.3.0, CUDA 13.4.92; final binary `e1ddfb5` sha256 `d6c964ae...`, v0.9.1 `c0bf82c` rebuilt on the same stack sha256 `6948df58...`. Date 2026-11-08. SHA256 see `MANIFEST.sha256`.
- `cost_shell_bp5_20261108.csv` — reduced cost rows (E1 R=1.5 1+3 repeats, E2 v0.9.1 A/B, E3r3, E3r4) plus the E0 probe row: command, geometry, versions, clocks, throttle, input and binary sha256, per-repeat detail in the `notes` column. Same host, stack and date as `cost_e2e_bp5_20261108.md`. SHA256 see `MANIFEST.sha256`.
- `e2e_bp5_20261108.csv` — reduced writer-inclusive e2e rows (g20/g100, N=2e6): totals, IQR/min/max, per-repeat readout, writer share, clocks, backend and the published HDF5 comparison. Same host, stack and date. SHA256 see `MANIFEST.sha256`.

## Hybrid near-field cost (2026-11-10)

- `cost_hybrid_final_20261110.csv` — near-shell wall for the pre-fix base
  `68b7554` and the final default `69bfd77` at `64^3` output, `R=3` cells,
  tile 4, 2000 sources at the grid centre (6,144,000 pairs). CPU rows are the
  interleaved A/B median of 5 timed pairs; GPU rows are median of 7 timed runs
  after 2 warm-ups on an idle V100, with and without a 12-cell source cloud.
  The default is `0.91`--`0.96x` the base CPU wall, `1.006`--`1.009x` on the
  on-node GPU workload, and `1.16`--`1.30x` with the source cloud. SHA256 see
  `MANIFEST.sha256`.
- `cost_hybrid_final_20261110.provenance.json` — commits, binary SHA-256,
  geometry, commands, protocol, clocks and the exact-pair share.

## Distributed FFT re-validation, HIP/gfx1100 (2026-11-09)

- `distributed_fft_validation.csv` — HIP rows re-measured on the current branch `repro/figures-20261108` at `688c2288137d4e2efacd14116b203a0525c444f5` with the same inputs and parameters as the original `defc079` run. Host `hal`, AMD Radeon RX 7900 XTX `gfx1100` (96 CU), ROCm 7.2.0 / HIP 7.2.26015, rocFFT 0.1.70200, openPMD 0.17.0, HDF5 1.10.10, Open MPI 3.1. Binary sha256 `7c9fac29771ac3d7bb4ce96acc892d146fed235b2ed16f87434d4609c9d57785`. One rank: `3.2e-16` (unchanged); four ranks: `5.35e-16` (original `5.375e-16`, roundoff-level). The one-rank replicated output is bit-identical to the original run. Date 2026-11-09. SHA256 see `MANIFEST.sha256`.
- `distributed_fft_revalidation_20261109.json` — full provenance: original and current commit, commands, inputs, grid/origin, toolchain, binary hash, raw-log location and the measured values. SHA256 see `MANIFEST.sha256`.

## Chunked read/compute overlap (2026-11-11)

- `chunk_overlap_20261111.csv` — chunked-pipeline medians for `N=1e6` and
  `N=1e7` at the 20- and 100-cell boxes, plus whole-input rows
  (`--max-particle-chunk-gb 1.0`). Columns: wall median and IQR, main-pass
  reader time, compute (deposit+kernel+FFT+readout), per-call setup (`plan`),
  the time the main thread waited in the prefetch `get()` (`reader_wait`), the
  representative-pass read, the hidden-reader fraction (`overlap`), and peak
  RSS. Host `rosi_v100`, node `gv001`, 4x Tesla V100-SXM2 32GB, driver
  570.158.01, CUDA 12.6.85, gcc 13.3.0, openPMD 0.17.1 / HDF5 1.10.10, date
  2026-11-11. Serial per host, one GPU, `OMP_NUM_THREADS=8`, one warm-up plus 5
  timed repeats (whole-input rows: 3), medians. Inputs are the
  `tools/e2e/gen_inputs.py` main-family files on the node's GPFS mount
  (`input_N1000000.h5` 72 MB, `input_N10000000.h5` 687 MiB). Baseline binary
  (main `870c2e6`) sha256 `0ca13fefd75dd9a58f6a8b2ce6de40d93994b6d93212ea824d72d189f25a0ede`;
  streaming binary sha256 `9ba8761f1a636cd38922f9db1db088c31aa074b72654ed6be820efd08eed8203`.
  The flag bounds the two resident particle buffers, so the effective chunk is
  half the flag; at the same effective chunk the output is bit-identical to the
  baseline under `--deterministic-deposit` (CPU and CUDA), and at the same CLI
  flag the largest field change on the fixed `N=1e6` 20-cell case is
  `3.22e-8` V/m on a `1.87e7` V/m peak (`1.7e-15` relative) because the chunk
  boundaries move. SHA256 `a6d6d653e5dd3c4f244e9cd204b2e934b001c6cb0290ee978b0f4a4d8ee64f21`.


## Distributed large-domain path, GV100 (2026-11-13)

- `distributed_large_domain_20261113.md` — the configured TOML path with the
  rank context wired before `makeGrid`. Records the per-rank cap scaling, the
  medium `440x800x480` runs that pin the 32 GiB GV100 at 2 and 4 ranks, the
  reduced medium `440x320x480` that completes at 4 and 8 ranks, and the small
  case's 4-rank-vs-1-rank agreement (worst `2.61e-10`, all six components).
  Host `fwk394`, Quadro GV100 32 GiB, 125 GiB host, CUDA `sm_70`, Open MPI
  4.1.6, openPMD 0.17.1, date 2026-11-13. The 2-rank reduced-medium run
  completed with non-physical values and is recorded as a risk, not a result.
  SHA256 see `MANIFEST.sha256`.
- `distributed_large_domain_20261113.json` — machine-readable form of the same
  tables: cap limits, per-rank and total memory, wall and stage times, GPU
  peaks, checkpoint sizes, per-component checksums and agreement, and the
  checkpoint structure. SHA256 see `MANIFEST.sha256`.

## Automatic near-shell tile (2026-11-12)

- `near_shell_tile_auto.csv` — tile 2 / tile 4 A/B for the automatic near-shell
  rule: dense 12-cell cloud at `N = 1e6` (`273.4` s against `621.7` s,
  `1.301e9` against `3.339e9` pairs) and `N = 1e7` tile 2 (`2,757.1` s,
  `1.302e10` pairs), compact 2000 sources at `N = 2e3` and `1e5`, sparse 2000
  sources over 48 cells, and the 4^3 fallback where the pair counts are within
  the throughput margin. The `chosen` column marks the automatic decision.
  Host `fwk394`, Quadro GV100 32 GB (sm_70), driver 555.52.04, CUDA 12.5.82,
  Intel Xeon W-2175; fast shell, order 5, `--exact-cells 0`, `R = 3`, FFT
  padding 2, deterministic LCG sources, `OMP_NUM_THREADS=8`; binary sha256
  `a86a77ed5000522f96dd6bd0b359eb951055d451126674a06ff9533486e33343`. Date
  2026-11-12. `rosi_v100` provisioning failed (Slurm configuration permission
  error), so the A/B ran on the same sm_70 Volta generation; the archived
  `rosi_v100` dense rows above are unchanged and provide the 1.25 throughput
  constant. SHA256 see `MANIFEST.sha256`.
- `near_shell_tile_auto.provenance.json` — host, toolchain, binary hash,
  settings, the sampled-estimate rule and the rosi-unavailable reason. SHA256
  see `MANIFEST.sha256`.
