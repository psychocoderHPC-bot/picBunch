# Near-shell scaling on V100

This note records the dense-cloud near-shell cost on `rosi_v100` and the tile
choice that follows from it. It covers the fast evaluator with a 12-cell source
cloud at `N = 1e6` and `N = 1e7`, output boxes of 64, 100 and 128 cells, tile
sizes 2 and 4, and radii `R = 3` and `R = 4`. The companion accuracy and cost
data stay in [measurements](measurements/).

## Measurement conditions

- Host: `rosi_v100`, Tesla V100-SXM2-32GB, 32 GB per card, CUDA 12.6, driver
  570.158.01. Four cards ran one measurement lane each.
- Source code: `870c2e6d5fe610d8f298dfa974287199c7011abb`.
- Binary: `picBunchNearShellMeasure`, sha256
  `c18876b56aca32c18b7b42156e1bf640518c2672440d1410b9b3bf0bd62de7e4`.
- Settings: fast shell, order 5, `--exact-cells 0`, FFT padding 2,
  `OMP_NUM_THREADS=8`, sources drawn from a uniform 12-cell cloud at the box
  centre, deterministic deposit.
- The reported seconds are the near-shell loop only. The mesh stage was
  `0.59` to `4.77` s and is not included. Each 1e6/64 row is the median of three
  runs; the other rows are single runs after the same setup. Four cards ran one
  measurement lane each in parallel, with eight OpenMP threads per lane.
- `pairs` counts (source, node) evaluations after the near set is built. Tile 4
  rebuilds a larger near set per tile than tile 2, so the pair count grows with
  the tile size at fixed `N`.

## Dense cloud at N = 1e6

| N | box | tile | R | near (s) | pairs | Mpairs/s | host RSS (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e6 | 64 | 2 | 3 | 257.93 | 1,301,361,504 | 5.045 | 712.3 |
| 1e6 | 64 | 4 | 3 | 585.97 | 3,338,959,488 | 5.698 | 499.7 |
| 1e6 | 64 | 2 | 4 | 498.71 | 2,663,316,480 | 5.340 | 1077.0 |
| 1e6 | 64 | 4 | 4 | 902.06 | 5,780,709,504 | 6.408 | 570.8 |
| 1e6 | 100 | 2 | 3 | 258.15 | 1,301,361,504 | 5.041 | 815.5 |
| 1e6 | 100 | 4 | 3 | 583.72 | 3,339,158,016 | 5.721 | 595.0 |
| 1e6 | 128 | 2 | 3 | 258.37 | 1,301,361,504 | 5.037 | 969.5 |
| 1e6 | 128 | 4 | 3 | 586.18 | 3,338,959,488 | 5.696 | 752.4 |

At `R = 3` the box size changes the time by at most 0.2% at tile 2 and 0.4% at
tile 4.
The 12-cell cloud and the `R = 3` support set dominate, and the padded mesh is a
small part of the run. At `R = 4` the pair count grows by `2.05x` at tile 2 and
`1.73x` at tile 4 relative to `R = 3`; the near-shell time grows by `1.93x` and
`1.54x`.

## Dense cloud at N = 1e7

| N | box | tile | R | near (s) | near (min) | pairs | Mpairs/s | host RSS (MiB) | device peak (MiB) | wall (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1e7 | 64 | 2 | 3 | 2607.47 | 43.5 | 13,015,206,528 | 4.992 | 4738.4 | 1596 | 2612.93 |
| 1e7 | 64 | 4 | 3 | 5281.13 | 88.0 | 33,394,531,584 | 6.323 | 2424.8 | 1724 | 5286.45 |
| 1e7 | 100 | 4 | 3 | 5202.24 | 86.7 | 33,394,762,752 | 6.419 | 2556.6 | 1724 | 5208.23 |
| 1e7 | 128 | 4 | 3 | 5280.05 | 88.0 | 33,394,531,584 | 6.325 | 2680.9 | 1724 | 5285.83 |

Ten times the sources produces exactly ten times the pairs, and the near-shell
time grows by the same factor at each tile size. The measured throughput is
stable at `5.0` Mpairs/s for tile 2 and `6.3` Mpairs/s for tile 4.

## tile 4 / tile 2 ratios

| N | box | R | time ratio | pair ratio |
| --- | --- | --- | --- | --- |
| 1e6 | 64 | 3 | 2.272 | 2.566 |
| 1e6 | 64 | 4 | 1.809 | 2.171 |
| 1e6 | 100 | 3 | 2.261 | 2.566 |
| 1e6 | 128 | 3 | 2.269 | 2.566 |
| 1e7 | 64 | 3 | 2.025 | 2.566 |

Tile 4 evaluates more pairs per source, but it evaluates them faster per pair.
At 1e6 it loses about 2.27x in wall time at `R = 3`; at 1e7 the larger tile
recovers part of that and loses 2.03x. The `R = 4` case has the smallest time
ratio because the support set growth is partly shared.

## Recommendation

For a dense 12-cell cloud at `N >= 1e6`, use tile 2 with `R = 3`. It is the
fastest measured configuration and the only one that keeps the 1e7 case under
an hour: `43.5` minutes against `88.0` minutes at tile 4. Switching the default
from tile 4 to tile 2 in this regime halves the near-shell time and leaves the
pair count at about 39% of the tile 4 count.

For a compact or on-node source distribution, use tile 4. The compact case was
measured at tile 4 only in this campaign: 2000 sources at the centre of a 64^3
grid give `6.144e6` pairs in `0.710` s (`8.65` Mpairs/s, about `1.7` times the
dense tile 2 throughput), see
[measurements/near_shell_device_speedup.csv](measurements/near_shell_device_speedup.csv).
Tile 4 also has the lower host RSS in every dense row of this campaign. The
compact case was not re-run at tile 2, so re-measure before changing the
default for that regime.

Do not treat tile 4 as the universal default. It remains the right choice when
the source set per node is small or the cloud is compact, and it is the current
default because of that regime, not because of the dense-cloud cost.

## Projections

The pair count and the near-shell time are linear in `N` in the measured range.
Linear extrapolation from the 1e7 rows gives:

| N | tile 2 / R3 | tile 4 / R3 |
| --- | --- | --- |
| 1e6 | 4.30 min (measured) | 9.77 min (measured) |
| 1e7 | 43.5 min (measured) | 88.0 min (measured) |
| 2e7 | 87 min | 176 min |
| 1e8 | 7.2 h | 14.7 h |

These are extrapolations, not measurements. They assume the pair count stays
linear, the host and device memory fit, and the evaluator cost per pair does
not change. None of those assumptions was tested above `N = 1e7`. A 1e8 run
also needs about 47 GB of host RSS at tile 2 on the measured slope, so host
memory is the first limit to check, ahead of the 1.6 GB device peak.

## Memory

Host RSS rises with `N` and with the box size, and it is lower at tile 4 in
every dense row. At 1e7 the tile 2 run used `4.7` GB of host RSS against
`2.4` GB at tile 4. Device peak was `1.6` GB at tile 2 and `1.7` GB at tile 4,
so the device side is almost flat across tiles and the host side is not. The
cause of the higher tile 2 host RSS was not profiled. Any change to the pair
storage or the per-tile source list must re-measure it.

## Gaps

- **H100 data were not available at measurement time.** The H100 file under
  [measurements](measurements/) is the earlier 200-particle rank scaling, not
  this dense-cloud matrix. Do not compare the two.
- **gfx1100 was contended and about 40x slower.** A 2000-source compact probe
  on the shared RX 7900 XTX took `28.4` s against `0.704` s on the idle V100,
  and a repeat on the same shared card took `81.4` s. The dense 1e6 runs were
  killed before they produced a pair count. These numbers are a contention upper
  bound, not a hardware comparison. A HIP scaling row needs an exclusive card
  and a profiler run.
- There is no tile 2 compact measurement and no `R = 4` row at `N = 1e7`. Both
  are cheap to add to the same driver and should be measured before the tile
  default changes in code.
- The driver reports the near-shell loop. End-to-end cost also includes the
  reader, deposit, mesh FFT and writer, which are small here but are tracked
  separately in the cost tables.
