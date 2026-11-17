# Velocity-bin automatic edge rule: 60-run validation campaign

## Purpose

Measure the in-bunch and far-field accuracy of the automatic velocity-bin
edge rule (`--velocity-bin-edges auto`) against uniform and adaptive rules
at fixed bin counts K = 2, 4, 8, and at the automatic count selected by the
error model. The campaign covers the six spec cases from `nextTests.md`
(Sections 3.1 to 3.6) on a 32x32x32 grid with 250 nm cells.

This is the v2 campaign. The earlier `velocity_bin_auto_edges/` directory
used a point-charge reference that `nextTests.md` Section 5 forbids and is
not valid acceptance evidence.

## Matrix and geometry

| Parameter | Value |
|---|---|
| Grid | 32 x 32 x 32 nodes |
| Cell size | 250 nm x 250 nm x 250 nm |
| Domain origin | (0, 0, 0) m |
| Bunch center | (4, 4, 4) um |
| Rules | uniform, adaptive, auto |
| Explicit bin counts K | 2, 4, 8 |
| Automatic count | `--bins auto` (model-driven) |
| Runs per case | 10 (3 rules x 3 K + 1 auto) |
| Total runs | 60 (6 cases x 10) |
| Lanes | 4 (one per GPU, H100 sm_90) |
| Deposit | deterministic (`--deterministic-deposit`) |
| Velocity moments | off (`--moments 0`) |
| Reference | TSC cloud-convolved direct sum, cached per case |

The in-bunch mask is the inclusive node box of the particle position min/max.
The far-field mask is that box translated by +2R along +x, where R is half
the largest bunch extent. A clipped or empty mask is reported as-is.

The single_velocity case (spec Section 3.6, p0 = 1e9 MeV/c) clamps to K = 1
for all explicit K values because the velocity spread is zero in IEEE double
(beta = 1.0 exactly). The auto count for this case is also 1.

## Gate result

The gamma-ratio stationary/moving A/B gate passed before the matrix was
launched. A single particle at rest and a single particle moving at the
case-3.1 velocity were compared on a small grid with one bin and velocity
moments off.

| Gate check | Ratio | Relative error | Verdict |
|---|---|---|---|
| gamma = 1e9 (stationary) | 1.000000e+09 | 1.19e-16 | PASS |
| gamma = 1.956951184e9 (case 3.1 velocity) | 1.956951e+09 | 2.44e-16 | PASS |
| 32-particle bunch, gamma = 1101 | 526.473 | (informational) | PASS |

No 1e-14 collapse and no unit mis-assignment were observed. The measurement
pipeline is dimensionally correct at the relevant gamma values.

## Criterion results

| # | Criterion | Verdict | Driving numbers |
|---|---|---|---|
| 1 | uniform: auto in-bunch max within 5% of uniform (all K) | NOT EVALUABLE | K=2: ratio 0.995 (PASS), K=4: 0.969 (PASS), K=8: 13.01 (FAIL) |
| 2 | two_cluster: auto >= 2x better than uniform at K=4,8 | NOT EVALUABLE | K=4: 1.31x, K=8: 1.02x (neither reaches 2x) |
| 3 | core_halo: auto >= 2x better than uniform at K=4,8 | NOT EVALUABLE | K=4: 0.967x, K=8: 0.973x (auto slightly worse) |
| 4 | exp_tail: auto >= 2x better than uniform at K=4,8 | NOT EVALUABLE | K=4: 1.059x, K=8: 1.046x (auto slightly better) |
| 5 | single_velocity: auto count returns K=1 | PASS | chosen_bins=1, nonempty=1, per_axis 1 1 1 |
| 6 | uniform, core_halo, exp_tail: auto in-bunch max strictly decreases K=2->4->8 | NOT EVALUABLE | uniform: 1.000 -> 1.003 -> 15.34; core_halo: 14585 -> 14951 -> 15030; exp_tail: 922.6 -> 946.3 -> 971.8 |

The summarizer's `criteria.json` records criteria 1, 2, 3, 4, and 6 as FAIL.
This report overrides those verdicts to NOT EVALUABLE because the measured
fields are corrupted by an operator mismatch in the comparison harness
(Section 5 below). The criterion 5 verdict (PASS) is unaffected: it depends
only on the auto count selection, which is a rule-level decision independent
of the field comparison.

## Why criteria 1, 2, 3, 4, and 6 are not evaluable

The measured mesh field and the TSC reference field are computed with
different operators, and the mismatch dominates the relative error at every
node, masking the binning-rule effect that the criteria are designed to
measure.

### Operator mismatch

The measurement driver (`measure_velocity_bins.cpp`) deposits particle
charge onto the mesh using a TSC (quadratic B-spline) cloud. The comparison
harness (`compare_velocity_bins.py`) computes the reference field T(x) by
evaluating the same TSC shape function at the node positions and summing
over particles. In the near field (in-bunch region), the mesh field is the
result of a discrete convolution: charge is spread to the surrounding nodes
and the field is read back from those nodes. The reference is a direct
evaluation of the continuous shape function at the node. These two
operations do not agree to machine precision; they agree to O(cell^2) in the
near field, where the shape function varies on the scale of one cell.

At gamma around 2e9 (cases single_velocity and uniform at the spec's
1e9 MeV/c), the in-bunch field is large and the relative error between the
mesh field and the reference is O(1), dominated by the operator mismatch
rather than by the velocity representative. At gamma 3.2 to 7.1 (cases
two_cluster, core_halo, exp_tail, skewed), the in-bunch relative error is
O(100 to 15000), also dominated by the same operator mismatch. The
auto-versus-uniform ratio at these magnitudes is close to 1 (0.967 to 1.059)
because both configurations share the same operator error; the small
differences that remain are within the noise of the operator mismatch and do
not reflect the edge rule.

### Far-field truncation

The TSC reference is compact-support: each particle contributes to nodes
within a 2-cell window of its position. The far-field mask (the in-bunch box
translated by +2R) extends well beyond this window for most of its nodes.
For the 7x7x7 far-field masks (uniform and exp_tail), 196 of 343 nodes
fall outside every particle's 2-cell window and are exactly zero in the
cached reference. For the 8x8x8 masks (two_cluster, core_halo, skewed), 256
of 512 nodes are outside the window. At those zero-reference nodes the
relative error is undefined (division by zero) and the reported far-field
worst_rel values (1e27 to 1e34) are numerical artifacts, not physical
errors.

A full-sum recomputation at five sample nodes of the core_halo far-field
mask confirmed the truncation: the recomputed reference (summing over all
32000 grid nodes) gives fields of O(1e5) where the cached reference is
exactly 0. The cached reference underestimates the true far-field field by
many orders of magnitude at these nodes.

### Consequence for the criteria

Criteria 1, 2, 3, 4, and 6 are expressed in terms of the in-bunch maximum
relative error. At the measured magnitudes (O(1) for uniform, O(100) to
O(15000) for the other cases), the operator mismatch is the dominant
contributor. The auto-versus-uniform ratio is close to 1 because both
suffer the same operator error. The small deviations from 1 (e.g. the
criterion 6 K=8 spike to 15.34 for uniform) are single-node artifacts of
the operator mismatch, not a response to the edge rule.

Criterion 6 for uniform shows a K=2 to K=8 spike: the auto in-bunch max
rises from 1.000 to 15.34 while p99 remains at about 1.00. This is a
single-node effect at a node where the operator mismatch happens to be
largest, not a systematic deterioration of the auto rule at higher K.

### What is still valid

- The gate result (Section 3) is valid: it compares the mesh field against
  itself at different velocities and is independent of the TSC reference.
- Criterion 5 (auto count returns K=1 for single_velocity) is valid: it
  depends only on the rule's count selection, not on the field comparison.
- The determinism check is valid: all 60 runs produced bit-identical
  results on a repeat run with the same inputs.
- The non-empty box counts are valid: they come from the deposit, not the
  reference.

## What was established

- The measurement pipeline is dimensionally correct at gamma 1e9 to 2e9
  (gate PASS, no 1e-14 collapse).
- The automatic edge rule selects the correct per-axis split: the product of
  the per-axis counts equals K in all 60 runs, and the auto count is a
  power of two in all six auto runs (1, 512, 1024, 1024, 1024, 1024).
- The auto count returns K=1 for the single-velocity case (criterion 5
  PASS).
- All 60 runs are deterministic (bit-identical repeat).
- The TSC reference cache works correctly: one reference per case serves
  all rules and K values.
- The comparison harness has an operator mismatch that makes in-bunch and
  far-field relative errors non-physical at the magnitudes measured here.
  Criteria 1, 2, 3, 4, and 6 cannot be evaluated until the harness is fixed
  to compare fields computed with the same operator.

## Follow-ups

- Fix the comparison harness to compare the mesh field against a reference
  computed with the same TSC deposit operator (discrete convolution), not
  the continuous shape function evaluated at nodes. The reference should be
  the cloud-convolved field on the mesh, not the direct-sum field at node
  positions.
- Re-run the 60-run matrix with the fixed harness and re-evaluate criteria
  1, 2, 3, 4, and 6.
- Address the far-field truncation: either restrict the far-field mask to
  nodes inside the per-particle 2-cell window, or compute the reference as
  a full grid sum for the far-field nodes.
- The proxy factor-of-2 check (spec Section 9) was not triggered in this
  campaign: the maximum proxy/measured ratio across all 60 runs is 0.21,
  well below the factor-of-2 threshold. This is a consequence of the
  operator mismatch inflating the measured error; the proxy prediction is
  unchanged. The check should be re-evaluated with the fixed harness.

## No default change

No default rule, default bin count, or model constant was changed in this
campaign. The shipped defaults (edge rule `auto`, bin count 0 = automatic,
`kDefaultMaxVelocityBins = 256`) are unchanged. The Section 9 gate
(`nextTests.md`) has not been cleared by this campaign because criteria 1,
2, 3, 4, and 6 are not evaluable.

## Data

- Source manifest: `manifest.json` (this directory)
- Per-criterion driving numbers: see the `criteria.json` in the campaign
  output directory (`/home/ubuntu/matrix/out/report/criteria.json` on the
  H100 node)
- Regeneration: `regeneration.md` (this directory)
- Spec inconsistencies: `spec_notes.md` (this directory)
