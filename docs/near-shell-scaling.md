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

A direct tile 2 / tile 4 A/B on the same compact case (2000 sources at the
centre of a 64^3 grid) shows tile 2 is faster, not tile 4: `0.373` s against
`0.718` s on a Quadro GV100, with `2.495e6` against `6.144e6` pairs. The earlier
compact row measured tile 4 only, so it did not establish tile 4 as the
optimum; tile 4 wins per evaluated pair (`8.55` against `6.7` Mpairs/s) but
evaluates 2.46 times as many pairs. At `N = 1e5` the same split is `16.6` s at
tile 2 against `31.5` s at tile 4 (`1.248e8` against `3.072e8` pairs). A sparse
2000-source cloud spread over 48 cells behaves the same way: `2.60e6` pairs at
tile 2 against `6.69e6` at tile 4.

Tile 4 keeps the lower host RSS in every dense row of this campaign. That
memory advantage, not a runtime advantage, is the reason to force it for a
memory-bound deployment.

Do not treat tile 4 as the universal default. The measured optimum follows the
near-set pair count, not the physical compact/sparse label.

## Automatic tile choice

The CLI and TOML default is `--near-shell-tile auto` (`near_shell_tile = 0`).
The automatic rule mirrors the per-source membership test that builds the near
set. It walks a deterministic, evenly strided sample of at most 4096 sources
and, for tile edges 2 and 4, counts the clipped tile node volume of every tile
whose node box is within the shell radius. Scaling by `N / sample` gives an
estimate of the (source, node) pair count; the cost is a few milliseconds even
at `N = 1e7`.

Tile 2 is chosen only when

```
pairs(4) > 1.25 * 1.05 * pairs(2)
```

The `1.25` is the measured tile-4/tile-2 pair throughput on the V100 campaign
(`6.3` against `5.0` Mpairs/s) and `1.05` is a safety margin, so the rule only
switches when tile 2 wins by at least about 5% against the best fixed edge. An
explicit `--near-shell-tile N` always wins and skips the estimate. The compute
stage prints the selected edge and both estimates:

```
near shell: automatic tile=2 (tile 2 pair saving exceeds the tile 4 throughput
  gain; estimated pairs: tile 2 = 1.3e+09, tile 4 = 3.34e+09)
```

The pair ratio is the quantity that matters. A 12-cell cloud and a compact or
sparse cloud on a grid larger than one tile-4 block all shrink their near set
by roughly 2.4-2.6x at tile 2, so the rule selects tile 2 in all three cases.
Tile 4 remains the result only when the two pair counts are within the
throughput margin, which happens when the whole source cloud already fits one
tile-4 block (for example a 4^3 grid: `3.34e5` against `3.84e5` pairs, where
the two edges agree within run-to-run noise of about 3%) or the grid itself is
at most one tile wide.

Verification host: Quadro GV100 (sm_70, CUDA 12.5) on `fwk394`, because
`rosi_v100` was not reachable during this run. The rule is geometric and the
throughput constant comes from the archived rosi V100 campaign, so the choice
transfers; the absolute seconds do not. On this host the dense 12-cell cloud
selects tile 2: `273.4` s at `N = 1e6` against `621.7` s at tile 4, and
`2,757.1` s (`46.0` min) at `N = 1e7` with `13,015,206,528` pairs. The tile 4
`N = 1e7` row is not re-run on this host; the archived rosi V100 value remains
`88.0` min.

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
- The compact tile 2 / tile 4 A/B and the sparse rows above are single-run or
  median-of-three measurements on a Quadro GV100, not a second V100 campaign.
  There is still no `R = 4` row at `N = 1e7`.
- The automatic rule has no host-RSS estimate. If a deployment is memory-bound,
  force `--near-shell-tile 4`; tile 2 still uses about twice the host RSS at
  `N = 1e7`.
- The driver reports the near-shell loop. End-to-end cost also includes the
  reader, deposit, mesh FFT and writer, which are small here but are tracked
  separately in the cost tables.
