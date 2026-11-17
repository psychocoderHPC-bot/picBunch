# Regeneration: velocity-bin 60-run validation campaign

## Environment

- Host: jureca-h100, 1 node, 4x NVIDIA H100 sm_90
- CUDA: launched with `env -u LD_PRELOAD` (the gateway shell exports
  `LD_PRELOAD=libfakeroot-sysv.so` which breaks CUDA jobs)
- GPU pinning: `CUDA_VISIBLE_DEVICES=<N>` per lane
- Branch: `issue-58-velocity-bin-report` at `6b451144a8ccb7d8db6af7c9ac974a2b8be2654a`
  (based on `tint2-integration`)
- Build: Release, single binary `picBunchVelocityBinsMeasure`

## Step 1: Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target picBunchVelocityBinsMeasure -j$(nproc)
```

The binary is `build/picBunchVelocityBinsMeasure`.

## Step 2: Generate the six spec bunches

```bash
python3 tools/velocity-bins/gen_synthetic_bunches.py \
    --out cases \
    --cell 2.5e-7,2.5e-7,2.5e-7 \
    --grid 32,32,32 \
    --origin 0,0,0 \
    --center 4e-6,4e-6,4e-6
```

This writes `cases/<case>.bunch` and `cases/<case>.meta.json` for each of
the six cases (uniform, two_cluster, core_halo, exp_tail, skewed,
single_velocity). Total particles: 9000 (spec Section 3 per-case counts;
see `spec_notes.md` Section 1 for the contradiction with the stated total
of 5000).

Verify with `--dry-run --verify` before the full generation if needed.

## Step 3: Run the gate

```bash
python3 tools/velocity-bins/gate_stationary_ab.py \
    --binary build/picBunchVelocityBinsMeasure \
    --workdir gate-work \
    --json gate-result.json
```

The gate runs a stationary/moving A/B comparison at gamma 1e9 and at the
case-3.1 velocity, plus a 32-particle bunch at gamma 1101. Expected: all
three checks PASS with the ratio agreeing with gamma to better than 1e-3
relative.

## Step 4: Run the 60-run matrix (4 lanes)

The matrix is 6 cases x {uniform, adaptive, auto} x K in {2, 4, 8} = 54
explicit runs plus 6 auto-count runs = 60 total. Four lanes run in
parallel, one per GPU.

Lane 0 (GPU 0):

```bash
env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0 \
    python3 tools/velocity-bins/run_matrix.py \
        --binary build/picBunchVelocityBinsMeasure \
        --cases-dir cases \
        --out out \
        --lane 0 --lanes 4 \
        --bins 2 4 8 \
        --rules uniform adaptive auto \
        --geometry matrix \
        --cell 2.5e-7,2.5e-7,2.5e-7 \
        --grid 32,32,32 \
        --origin 0,0,0 \
        --center 4e-6,4e-6,4e-6 \
        --deterministic-deposit
```

Lanes 1, 2, 3 are identical except `--lane 1`, `--lane 2`, `--lane 3` and
`CUDA_VISIBLE_DEVICES=1`, `=2`, `=3` respectively.

Each lane runs 15 runs. The matrix runner writes one JSON per run under
`out/` and a per-run work directory under `out/.work/`.

Wall time per run: 1.0 to 4.4 s (measure 0.9 to 4.2 s + compare 0.1 to
0.5 s). Total wall time for 60 runs on 4 GPUs: about 35 s per lane, about
2.5 min end-to-end including the first TSC reference computation.

The TSC reference is cached per case under `out/.work/reference-cache/`.
The cache key is `sha256(particles, grid, origin, cell, order, refine, eps,
mask)`. One reference per case serves all 10 runs of that case. The first
lane to hit a case computes the reference (about 193 s per particle on a
686-node mask, TSC order 12, refine 1); the other three lanes hit the
cache.

To resume an interrupted matrix, add `--resume`.

## Step 5: Summarize

```bash
python3 tools/velocity-bins/summarize_matrix.py \
    --manifest out/manifest.json \
    --out report \
    --markdown report/velocity_matrix_report.md \
    --criteria report/criteria.json
```

This produces the auto-generated report (`velocity_matrix_report.md`), the
criterion verdicts (`criteria.json`), and the campaign manifest
(`manifest.json`).

## Wall times (observed)

| Phase | Duration |
|---|---|
| 6 TSC reference computations (first hit per case) | 2 h 45 m total (193 s/particle, 686-node mask, order 12, refine 1) |
| 54 explicit runs (measure + compare, cached reference) | about 35 s compute per lane |
| 6 auto-count runs | included in the per-lane 15 runs |
| Total matrix wall time (4 GPUs) | about 2.5 min for the 60 runs after references are cached |

The TSC reference computation dominates the first run of each case.
Subsequent runs (including a full re-run with `--resume`) hit the cache and
complete in seconds.

## Constraints and notes

- The reference cache is keyed on the particle file and geometry, not on
  the rule or K. Deleting `out/.work/reference-cache/` forces a full
  recomputation (about 2 h 45 m for all six cases).
- The `--deterministic-deposit` flag is required for bit-reproducible
  results. Without it, the atomic deposit differs in about 17,602 of
  36,864 entries at the 4.2e-13 level between identical runs.
- The measurement driver does not support MPI or device indexing. Four GPUs
  are four independent processes, each pinned to one device.
- The operator mismatch described in `summary.md` Section 5 affects the
  comparison step (`compare_velocity_bins.py`), not the measurement step.
  Regenerating the matrix with a fixed comparison script does not require
  re-running the measurement binary.
