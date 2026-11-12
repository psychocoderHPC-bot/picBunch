# Pilot and cost estimate

Run: 2026-11-12 on `jureca-h100` (`jrc0880`), exclusive 4x NVIDIA H100 PCIe
80 GB, 144 cores, CUDA 13.4.92, gcc 13.3.0, openPMD 0.17.1 (HDF5 + ADIOS2
2.12.1). Repo `psychocoderHPC-bot/picBunch` at
`00fb8f7fb22a53748636c70e32bbdb8dd85c23c4` (build commit; the harness commit
is a separate field in every production record).

* picBunch binary SHA-256 `717212ed4ad0be862f6b65ab2d2b7d9ace914fda2aeec6a4c8a4f2167e7808e6`
* exact reference driver SHA-256 `a0d08be28b075fdb5e094e5a99800af2022826698e770e58d36fce9b16fa5f0e`

## Build

Configure 22 s, build 40 s at `-j 64` (warm FetchContent caches). No CUDA 13.4
or gcc 13 source changes were needed. `-DCMAKE_CUDA_ARCHITECTURES=native`
fails the CMake compiler probe in this container ("no NVIDIA GPU was
detected"); `90` is the exact H100 PCIe architecture. The default 16 MB
`/tmp` tmpfs cannot hold `nvcc` temporaries, so `TMPDIR` moves to the 35 TB
scratch mount.

## Pilot cases

Reduced reference settings (`order 8`, `max_level 4`, `tol 1e-2`) so the pilot
is short. Four cases ran in four GPU slots. `wall_s` is the full case
(input generation is outside the timer, mesh + reference + reduction inside):

| case | runs | mesh compute (range) | pilot `wall_s` |
| --- | --- | --- | --- |
| `hires-g10` | 1 x 128^3 | 1.4-2.9 s | 3.8 |
| `var1b-1mrad` | 1 x 32x32x16, 64 bins | 2.7 s | 15.5 |
| `line-w4b` | 8 meshes + 8 references | 0.5-2.8 s | 21.4 |
| `line-gamma` | 6 meshes + 6 references | 1.6-2.8 s | 82.8 |
| `bdata2-forward` | 1 x 64^3, 2 planes | 0.7 s | 100.6 |
| `large1-g1` | 1 x 128x128x64, 2 planes | 0.7 s | 143.3 |
| `multivel` | 10 runs, 1-256 bins, 64^3 | 0.6-2.9 s | 15.4 |

The largest pilot blocks are the two dense references: `bdata2-forward`
(49 152 sample points x 200 sources) took 100.6 s and `large1-g1`
(196 608 x 100) took 143.3 s at the pilot order 8. The production `large1`
reference uses order 4 and should be cheaper; the production `bdata2`
reference uses order 12 and is the one case block whose reference may roughly
double.

## Production reference calibration

Two cases were re-run with the production reference settings (`order 12`,
`max_level 5`, `tol 1e-3`):

| case | sample points | sources | reference wall | total `wall_s` |
| --- | --- | --- | --- | --- |
| `hires-g10` | 98 304 | 1 | ~4 s | 5.7 |
| `var1b-1mrad` | 6 144 | 200 | ~30 s | 35.3 |

This gives a conservative reference rate of about `2.4e-5 s` per
(sample point x source), which is the `--ref-rate` default. Extrapolated to
the remaining slice cases:

| case | points x sources | reference estimate |
| --- | --- | --- |
| `bdata2-forward` | 49 152 x 200 | ~4 min |
| `bdata2-fluctuation` | 49 152 x 200 | ~4 min |
| `large1-g1`, `large1-g10` | 196 608 x 100, order 4 | ~3-7 min each |
| `hires-g1p0` ... `hires-g100` | 98 304 x 1 | ~4 s each |
| `line-gamma` | 18 x 1 | seconds; the 384^3 meshes dominate |
| `line-w4b` | 1008 x 1 | seconds |

## Production estimate

`run_campaign.py --estimate` with the default `--ref-rate 2e-5` prints a
per-slot LPT assignment. The conservative upper bound, before the measured
mesh times are applied, is:

```
per-slot estimated wall (s): 1166 1185 1151 1217
estimated campaign wall: 1217 s (0.34 h)
```

The pilot measured mesh compute well below the `mesh_s` values in
`cases.json` (for example 128^3 is 1.4-2.9 s versus the 60 s estimate), so the
measured-scaled estimate is roughly 0.1-0.2 h. The full precision set fits the
4 h allocation with a wide margin. **No case is dropped.** The only excluded
work is the out-of-scope scaling/performance block.

## Validation

* `hi_g10.npz`: 51 keys, 45 required by
  `ingest_measurements.required_npz_keys`, none missing; `xy_*` arrays
  `128x128`; the real `_transform_field_export` consumed it with no error.
* `slices_var1b.npz`: 35 keys, `xy_*` arrays `32x32`, `sources (200,3)`.
* `multivel_fields_*_b16/b64.npz`: six `64^3` components plus `array_axes`,
  `cell_m`, `grid`, `case`, `bins`, `charge_C`, `beam_axis`, `metric`;
  `multivel_energy.npz`/`multivel_angular.npz` carry `particles (512,8)`.
* `w4b_rel_error.csv`: 56 rows with the required six columns; the gamma=1 rows
  reproduce the committed CSV to the printed precision
  (`d=2` 0.11650316569 vs 0.1165031662; `d=32` 1.429e-6 vs 1.429e-6).
* `forward_slice.npz`: 63 keys, all 63 required; the real
  `_transform_field_export` consumed the `campaign/fields/bdata2-forward.npz`
  export (63 arrays, `sources (200, 7)`).
* `slices_g1.npz`: 60 keys, all 58 required; `xy` arrays `128x128`, `zy`
  `64x128`; the charge-weighted centroid is
  `(1.27649588e-05, 1.27838583e-05, 6.30329885e-06)` m, identical to the
  committed case registry.
* `w4b_rel_error.csv` and `gamma_err_dist.csv` have the required columns and
  56/18 rows. The direction lattice was widened from axis-only to axis +
  face-diagonal + cube-diagonal, so `d=48` is covered (the axis-only lattice
  fell off the 96-node grid); `gamma=10,h=0.25,d=48` now reproduces the
  committed `6.77651668e-4` to six digits. The `w4b` gamma=1 rows reproduce
  the committed CSV through d=12 (`d=2` 0.11650316569 vs 0.1165031662); the
  larger-d rows differ because the widened direction set samples different
  worst-case nodes than the legacy ER1 lattice.
* One provenance record was written for each pilot case, with both the
  harness and build commits, the binary SHA-256, configuration hash, full
  commands, GPU slot, UTC times, exit code and output SHA-256.

## Not run

The full campaign was not started, per the task bounds. `bdata2-*`,
`large1-*` and `line-gamma` have no pilot mesh/reference output yet; their
costs above come from the measured reference rate and the conservative case
registry.
