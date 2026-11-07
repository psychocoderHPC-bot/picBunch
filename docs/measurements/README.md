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

- `matrix.csv` — scaling matrix (N x box), per-stage medians. Host `rosi_v100`, node `gv002`, 4x Tesla V100-SXM2-32GB, CUDA 12.6, sm_70. Commit: picBunch `dc69c16a21f1dc330746365e3409560cb2399ed9`; benchmark branch `experimental/bench` at `3d05752a493e12f3a0a03c6754f92f202a70a482`. Date 2026-11-03. SHA256 `b7a22e3d00998409604d7e3ecacafdd2118c637b2f76cb0108a505bfec8d2dfa`.
- `coulomb.csv` — Coulomb-limit check of the zero-padded FFT convolution against the direct discrete TSC sum. Same host, revision and date as `matrix.csv`. SHA256 `b483e021042d8bcb84cd6f8081a3ff52d78b9300985728c48129c907cfb49584`.
- `dsweep.csv` — d-sweep of the convolution error. Same host, revision and date as `matrix.csv`. SHA256 `e3687a3df32333b249837c3163703882073f0a6e66791c1a514aeceb10042b70`.

## Refinement and h-scaling (F4)

- `f4_per_spacing_before.csv` — per-h field errors before the scale fix. Host `rosi_v100`, node `gv002`; PIConGPU `experimental/mesh-h3` at `fef43bbd231595fed2795d26affd5d55b9ebd379`. Date 2026-11-03. SHA256 `c15cafcde3101547f81e83c970fe9134bcacc1a4bc0a52a62790a4f339338fbf`.
- `f4_per_spacing_after.csv` — per-h field errors after the scale fix. Same host and PIConGPU revision as `f4_per_spacing_before.csv`, date 2026-11-03. SHA256 `ca78a419ca616babeee38d183e7989bb4426735b7aca61bfb04cd9ec94da6303`.
- `f4_refine_gate_before.csv` — refine-gate raw rows before the fix. Same host, revision and date. SHA256 `9e99a7fd4e3d1b6f6172a458156b1f278f1549b492b8d2123fdfe827cd3a5ab1`.
- `f4_refine_gate.csv` — refine-gate raw rows after the fix. Same host, revision and date. SHA256 `069f53b7e809bf0c8f8ffd76fd5b95549c7a5d1b0954dbc299934cd8d7e884a9`.
- `f4_refine_gate_before_summary.csv` — summary gate verdict before the fix; worst relative change 2.476e-1, gate 2e-3, FAIL. Same host, revision and date. SHA256 `1cb2ac9eed712bc5a6cd100505b281ff1215d8ce8754db431fa277b23230e5d9`.
- `f4_refine_gate_summary.csv` — summary gate verdict after the fix; worst relative change 9.132e-4, gate 2e-3, PASS. Same host, revision and date. SHA256 `96cbbb1c2260c4f2e37431e7e8d342cf71674b1a1582e7ead6a3cb0f0ee2f8ed`.

## Accuracy (W4)

- `w4b_rel_error.csv` — per-component relative error versus cell distance, extracted from the W4 relative-error table. Host `rosi_v100`, node `gv002`; picBunch `bfaa1e26d6c2aff52f7b74b729d068be253e4d30`. Date 2026-11-03. Recovered from the run mirror `/tmp/alpaka-orchestration/pr24-20261102/paper/w4b_rel_error.csv`; the rosi accuracy directory holds no CSV (only the JSON and a Markdown summary). SHA256 `530bcba455da842a7241d1defa92aaf6753279a02ce48c8c21ef1ea4eb3d77d4`.
- `accuracy_results.json` — full W4 accuracy results, per case and per distance, both references and all components. Host `rosi_v100`, node `gv002`; picBunch `bfaa1e26d6c2aff52f7b74b729d068be253e4d30`. Date 2026-11-03. SHA256 `b5b68f07c4083d2744b36d3e5376350013e589481f4020c97b418485b464dc6c`.

## Reference identification and N-scaling (W7t)

- `w7t_reference_and_scaling.md` — Q1 answer (what the exact-quadrature reference computes, with source line references) and Q2 N-scaling verdict. Host `rosi_v100`; picBunch `339ca7ad3903249ebf3859f5e106d63db4dc3df5`. Date 2026-11-03. SHA256 `35a3a77f59cc3618d2d16548dd9b81616171ebd39a618fc868b0f433c7331590`.
- `w7t_q2_table.log` — raw N-scaling table printed by the Q2 script. Same host, revision and date. SHA256 `064c18771003e9b42ae7cf30698131afa98bd208ef86ab89526595eda71d4216`.

## Velocity binning (MV1)

- `velocity_bin_calibration.csv` — recorded 16- and 64-box worst-case binning errors for the 5% energy-spread and 5 degree angular-spread cases, with host, campaign commit, date and source. Host `jureca-h100`, CUDA 13.4; campaign `pr24-20261102`, commit `02891fe6`; date 2026-11-02. These are the endpoints quoted in the root README; the original 1/2/4/8-box rows, per-box model sums and `--fit` output were not archived, so `C = 1` is an assumption, not a refitted calibration. SHA256 `a4efd91976d2fb8d7d6f350ae6790fa8c4f3cf17a121bb550e9e407208557774`.
- `velocity_bin_calibration_fit.txt` — output of `tools/velocity-bins/calibrate_velocity_bins.py` from the CSV. It records the archived points, the target verdict and the statement that `C = 1` is assumed. Generated at repository commit `6fab4a3` on 2026-11-06. SHA256 `a370973c3b892263a57557a32d304c1d6b0c574c0f1ab8cb35909cd0e7ce004d`.

## Field-error and energy-spread gate (integrated N2/N3/N5 branch)

- `field_error_energy_spread.csv`: banded relative L2 for binned `B=8/64/256` and `--velocity-moments 2/3/auto` at 200 000 particles, gamma 10, 5% energy spread, `64^3` cells of `1e-6` m, TSC, `--near-shell none`, 40 stratified nodes; reference is the TSC-convolved field (order-5 far, order-8 refine-1 near). Also records the single-velocity branch-vs-main equality, the one-rank vs two-rank moment equality, the auto-bin cap errors, and the near-shell host/device accuracy rows. Host `hal`, A30, CUDA 13.4, sm_80; branch `field-error-energy-spread-20261107` at `ca4163cff964f735fc7202a8cc743fdb4ff34259`. Date 2026-11-07. SHA256 `a6db6f1520f6f6fafd57d78f353b94e55a808018aec2ded373fbd0f556cd9c34`.
- `field_error_timing.csv`: CLI, compute, near-shell and peak-RSS medians for the same paths, plus the 2e3-particle near-shell equivalence run and the 200 000-particle production near-shell run (231.0 s against the pre-change 282.0 s). Same host, revision and date as `field_error_energy_spread.csv`. SHA256 `2b489150b02bba9b2eb7756af8e4ef2e1d08e48a4d58c23ea53161840e5050a9`.

## Padding and R2C/C2R (N1b)

- `n1b_padding_r2c.md` — wraparound padding sweep, per-component padding detail, R2C/C2R equivalence and the bin rule. Host `rosi_v100`, CPU-only, OMP 16 threads; Python model of the picBunch `kRefinement = 1` mesh convolution, no repository commit. Date 2026-11-03. SHA256 `8146c07c6b5b989b33d0c40a299e0b7cfbe3a52958fc8be213d4d6b3dd43d8c4`.
- `n1b_padding_distance.csv` — raw padding-versus-distance rows behind `n1b_padding_r2c.md`. Same host and date; Python model, no repository commit. SHA256 `574c79371755fdf355a8ea6a59c5b3cb3e80c12aabb12bfa8afe052cf19f72a7`.
