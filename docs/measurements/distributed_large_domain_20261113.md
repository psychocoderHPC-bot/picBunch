# Distributed large-domain path (fwk394, 2026-11-13)

The configured TOML path now wires the MPI rank context before `makeGrid`, so
the per-rank padded-node cap scales with the rank count exactly as it does on
the legacy CLI path. This note records the large-domain behaviour that the fix
makes reachable, on one Quadro GV100 (32 GiB, `sm_70`) with a 125 GiB host.

Measured on top of `2befa2e` with the B9c wiring fix, CUDA build `sm_70`,
Open MPI 4.1.6, openPMD 0.17.1, `CUDA_VISIBLE_DEVICES=1`. Raw run directories
and logs live under `/tmp/b9c/runs`. The machine-readable copy is
`distributed_large_domain_20261113.json`.

## Cap scaling

`kMaxPaddedNodes = 2^28` is a per-rank budget. A distributed context multiplies
it by `size * 8` (`Compute.cpp:1643-1645`). The medium domain
(`440x800x480` at `2e-7 m`) has a padded box of `880x1600x960 = 1.352e9` nodes.

| ranks | accepted limit (nodes) | medium padded box | result |
| --- | --- | --- | --- |
| 1 | 268,435,456 | 1,352,000,000 | rejected |
| 2 | 4,294,967,296 | 1,352,000,000 | accepted |
| 4 | 8,589,934,592 | 1,352,000,000 | accepted |

The single-rank rejection message is exactly:

```
picBunch: error: makeGrid: padded FFT grid for output grid 440x800x480 with padding 2 is infeasible (limit 268435456 nodes); check the input gridSpacing/gridUnitSI or pass --cell-size
```

`--fft-mode auto` selects the distributed z-slab transform for every multi-rank
run. With `--fft-mode replicated` the cap is not scaled (the replicated path
needs the full box on every rank), so the medium domain is rejected with the
same `268435456` limit before any allocation.

## Medium domain: 440x800x480 @ 2e-7 m (origin 0,0,0)

| ranks | status | wall | field compute | per-rank peak host RSS | total host peak | GPU peak | checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | rejected by cap | 3 s | - | 13 MB | 13 MB | 1 MiB | - |
| 2 | GPU wall, aborted | 131 s | did not finish | 27.2 / 24.6 GB | 51.8 GB | 32,482 MiB | - |
| 4 | GPU wall, aborted | 56 s | did not finish | ~12.4 GB each | ~49 GB | 32,482 MiB | - |
| 4, replicated | rejected by cap | 3 s | - | - | - | 1 MiB | - |

The distributed working set is three complex arrays over the padded box,
`3 * M^3 * 16 B = 64.9 GB`. Each rank holds `M^3/size` of each array, so the
rank count lowers the per-process share but not the aggregate. On a single
32 GiB device the medium domain is oversubscribed about 2x, pins the GPU at
32,482 MiB, and pages the rest through unified memory. The earlier 2-rank
legacy-CLI run was observed for more than 6 hours without producing a
checkpoint. The first wall is the GV100, not the 125 GiB host RAM.

## Reduced medium: 440x320x480 @ 2e-7 m (origin 0, 90e-6, 0)

This domain still exceeds the single-rank cap (`880x640x960 = 540,672,000`
padded nodes) and fits the device. It contains the same bunch.

| ranks | status | wall | field compute | writer | per-rank peak host RSS | total host peak | GPU peak | checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | rejected by cap | 3 s | - | - | 13 MB | 13 MB | 1 MiB | - |
| 2 | completed, field wrong | 110 s | 87.8 s | 19.2 s | 11.0 / 9.96 GB | 20.96 GB | 32,482 MiB | 2.42 GB |
| 4 | completed, verified | 85 s | 67.9 s | 14.6 s | 5.57 / 5.57 / 5.06 / 5.05 GB | 21.25 GB | 32,482 MiB | 2.42 GB |
| 8 | completed, verified | 83 s | 63.8 s | 15.6 s | 4.80 / 2.86 / 2.86 / 2.86 / 2.61 / 2.61 / 2.61 / 2.61 GB | 23.82 GB | 32,482 MiB | 2.42 GB |

The 4-rank field is checked by rank-count invariance against the 8-rank run:
the largest relative component difference is `5.23e-10`. The 8-rank run uses
the same physical domain, so this is a decomposition check, not a new physics
reference. The 2-rank runs completed but wrote non-physical values (see Risks);
they are not part of the validated claim.

## Small case: 220x400x240 @ 4e-7 m (origin 0,0,0), 4 ranks vs 1 rank

This case fits one rank (`440x800x480 = 168,960,000` padded nodes) and is the
only configuration with a true 1-rank reference.

| ranks | FFT layout | wall | field compute | per-rank peak host RSS | total host peak | GPU peak | checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | replicated | 16 s | 1.89 s | 1.60 GB | 1.60 GB | 8,008 MiB | 708,456,890 B |
| 2 | distributed | 37 s | 24.8 s | 3.54 / 3.21 GB | 6.75 GB | 8,614 MiB | 708,456,890 B |
| 4 | distributed | 37 s | 26.2 s | 1.85 / 1.84 / 1.68 / 1.68 GB | 7.05 GB | 9,234 MiB | 708,456,890 B |

All six components agree between 4 ranks (distributed) and 1 rank
(replicated), relative to the component peak:

| component | shape (z,y,x) | sum abs(f) | sum f^2 | max abs(f) | FNV-1a (4 ranks) | rel. vs 1 rank |
| --- | --- | --- | --- | --- | --- | --- |
| E_x | 240,400,220 | 1.30695870e3 | 5.17422187e0 | 3.156916e-2 | 598193977 | 4.50e-13 |
| E_y | 240,400,220 | 6.36556143e-2 | 2.03252982e-9 | 5.316027e-8 | 3208217658 | 2.61e-10 |
| E_z | 240,400,220 | 1.32733287e3 | 5.18300550e0 | 3.639979e-2 | 2097375995 | 1.95e-13 |
| B_x | 240,400,220 | 4.50961016e1 | 2.14543239e-3 | 1.005931e-4 | 1628605003 | 3.53e-11 |
| B_y | 240,400,220 | 1.38279857e-2 | 5.09946411e-10 | 2.371504e-7 | 103245218 | 1.83e-12 |
| B_z | 240,400,220 | 4.16029735e1 | 1.84239905e-3 | 9.542757e-5 | 2228691555 | 2.33e-12 |

The worst relative difference is `2.61e-10`, below the `1e-6` requirement. The
FNV-1a digests differ between rank counts because the last float32 bits move
with the reduction order; the sums and maxima above are the same to the shown
digits.

## Checkpoint structure

Multi-rank runs write `checkpoint_000000.h5` through the openPMD-Series
writer. The distributed field contributes one z-slab per rank; rank 0 gathers
the slabs and writes the full-shape records. The file is one dataset per field
component, not one per slab:

- `data/0/fields/E/{x,y,z}` and `data/0/fields/B/{x,y,z}`, one record component
  each, dataset shape `(Nz, Ny, Nx)` and dtype `float32`;
- `axisLabels = [z, y, x]`, `gridSpacing` and `gridGlobalOffset` in metres,
  `unitSI` from `unitSI/--picongpu-unit-*`;
- species `b` with 49,932 particles and the checkpoint-only empty species `e`
  with 0 particles;
- `checkpoints.txt` contains `0`.

Both the small 4-rank file and the reduced-medium 4- and 8-rank files open with
openPMD 0.17.0 and report the full shapes and the two species. The small 1-rank
and 4-rank files are byte-size identical (708,456,890 B); the reduced-medium
2-, 4- and 8-rank files are identical (2,419,770,554 B).

## Risks and follow-up

- **2 ranks at 13 GiB per rank produce a wrong field.** The reduced-medium
  2-rank runs completed but wrote finite non-physical values (`E_x` max
  `4.43e10` against `5.01e-2` at 4 and 8 ranks). Both repeats produced the
  same values; 4 and 8 ranks agree to `5.2e-10`, and the 2-rank small case
  agrees with its 1-rank reference to `2.6e-10`. The failure appears only at
  the larger per-rank slab, where the device is pinned at 32 GiB, and looks
  like an unchecked vendor-workspace failure rather than a decomposition
  error. Surface the cuFFT/plan allocation status and re-check 2 ranks before
  using that configuration.
- **The medium domain does not fit one GV100 at any rank count.** The aggregate
  working set is set by `M^3`, not by the rank count. Use a domain with
  `3 * M^3 * 16 B` below the device budget (with headroom for plan workspace),
  or a device with more memory.
- The checkpoint gather holds the full six-component field on rank 0 (8.1 GB
  for the medium, 2.6 GB for the reduced medium) before the float32 write.
  That is a rank-0 host cost, not a distributed-compute cost.
