# H100 in-support near-shell field: kernel, throughput, accuracy, 2e6 projection

Run `pr24-20261102`, task H1. Branch `experimental/h100-near` (from `main` = `v0.5.0` = `7b4f1766`).

## System
- JURECA-DC node jrc0880, 4x NVIDIA H100 PCIe (sm_90), CUDA 13.4, 144 cores.
- Exclusive Slurm allocation (partition `dc-h100`); `OMP_NUM_THREADS=48` for the host reference phase.
- Kernel: `tools/near-shell/nearshell_kernel.cu`, self-contained. Math ported from
  `picongpu@2baa256` `include/picongpu/particles/neuralBunch/ReferenceKernel.hpp`.

## Method
Per-target-point in-support field = singular part (`box_field`, divergence-theorem face integral)
plus remainder (Gauss quadrature over the TSC support, octant-split to remove the s=t singularity).
Three evaluators:
- (a) singular only
- (b) stationary singular + remainder
- (c) moving (gamma=10) singular + remainder

Two kernels: thread-per-pair (`pair_eval_kernel`, the primary throughput kernel) and warp-per-target
(`warp_target_kernel`, shared-memory source staging + Kahan accumulation, used for the multi-GPU
scaling point).

Reference: 348-point stationary corpus including near-face points, plus a 69-point moving corpus.
Converged reference = singular@16 + remainder order-doubled to 1e-10. Cross-checked against 5
independently computed probes (source at origin, unit cells).

## Accuracy (worst-case relative error)

| configuration | worst rel err | reaches 1e-3 |
|---|---|---|
| converged reference vs 5 independent probes | 4.47e-5 | n/a |
| (a) singular only | 1.00e+00 | no (S near 0 at the face, by construction) |
| (b) stationary, remainder n=16 | 1.72e-03 | no |
| (b) stationary, remainder n=24 | 4.68e-04 | yes |
| (b) stationary, remainder n=48 | 1.41e-13 | yes (machine precision) |
| (c) moving gamma=10, nang=24/nr=14/ng=14 | 3.01e-02 | no |

- (b) reaches 1e-3 at n=24; n=24 is the 1e-3 configuration used for the throughput measurement.
- (c) does not reach 1e-3. The moving error is non-converging in quadrature order (nang=24 is the
  best point at 3e-2; nang=16 gives 1.17e-1, nang=32 gives 1.33e-1). Reaching 1e-3 for the moving
  case requires the singular-peak fix (PIConGPU commit `b87c0535e`), out of scope here.

## Throughput (1x H100, pairs/s; one pair = one (target, in-support source) field evaluation)

| evaluator | thread-per-pair | warp-per-target |
|---|---|---|
| (a) singular only | 4.0e8 (1.4e9 on a cold GPU) | n/a |
| (b) stationary, n=24 (1e-3) | 2.08e5 | 6.0e5 |
| (c) moving gamma=10, nang=24 | 4.98e3 | n/a |

- The near field is dominated by the O(n^3) remainder quadrature, not the O(1) singular face
  integral. At 1e-3 (n=24) it is roughly 2000x slower per pair than the singular part alone. This
  is expected: the in-support field is evaluated exactly, with no far-field or multipole
  acceleration inside the near shell.
- The warp-per-target kernel is about 3x faster than thread-per-pair for case (b) (shared-memory
  staging of the S=128 sources per target, plus Kahan summation), so it is the realistic
  production rate.

## Multi-GPU scaling (warp-per-target, case b, Nt=8192, S=128)

| GPUs | total pairs/s | speedup vs 1 GPU |
|---|---|---|
| 1 | 6.0e5 | 1.00x |
| 2 | 1.02e6 | 1.71x |
| 3 | 1.34e6 | 2.24x |
| 4 | 1.34e6 | 2.23x |

- Sub-linear (2.23x on 4 GPUs) because the per-GPU workload (2048 targets) does not saturate the
  H100. The workload is embarrassingly parallel, so larger per-GPU batches scale closer to linear.
- The H100s are PCIe (no peer-to-peer), so each GPU holds its own geometry copy.

## Projection to 2e6 particles (near-shell pair counts, 300 s budget)

Near-shell pair counts for 2e6 particles: 1.7e8 (support-only, about 85 neighbours), 1.7e9
(3-cell shell, about 850), 3.4e9 (4-cell shell, about 1700). Time = pairs / rate.

| pair count | warp 1 GPU | warp 4 GPU | thread-pair 1 GPU | thread-pair 4 GPU |
|---|---|---|---|---|
| 1.7e8 (support-only) | 283 s (meets) | 127 s (meets) | 817 s | 367 s |
| 1.7e9 (3-cell shell) | 2833 s | 1273 s | 8169 s | 3671 s |
| 3.4e9 (4-cell shell) | 5666 s | 2546 s | 16338 s | 7343 s |

4-GPU times use the measured 2.23x speedup.

### Verdict
- 300 s at 1e-3 is met only for the support-only pair count (1.7e8) and only with the
  warp-per-target kernel (283 s on 1 GPU, 127 s on 4 GPUs). The thread-per-pair kernel misses
  (817 s on 1 GPU, 367 s on 4 GPUs).
- The 3-cell (1.7e9) and 4-cell (3.4e9) shells miss 300 s even on 4 GPUs, by about 4x and 8.5x.
- The moving case (c) is about 40x slower than the stationary case and misses by 114x or more;
  it also does not reach 1e-3 without the peak fix.

### Root cause
The in-support near field is exact and dominated by the O(n^3) Gauss remainder quadrature
(2.4e5 pairs/s at 1e-3), not the cheap singular face integral (4e8 pairs/s). Cost scales with
shell width times n^3, so only the tightest shell fits a 300 s budget on current hardware.

## How to build and run
- Build: `tools/near-shell/build.sh` (nvcc -O3 -std=c++17 -arch=sm_90 -Xcompiler=-fopenmp). The
  node /tmp is a 16 M tmpfs, so the build needs a scratch TMPDIR (build.sh sets one when /tmp is
  small).
- Run: `env -u LD_PRELOAD OMP_NUM_THREADS=48 ./nearshell_kernel --ngpu 4`. The `-u LD_PRELOAD`
  drops the host-side `libfakeroot-sysv.so` preload that otherwise breaks OpenMP semaphores.

## Caveats
- Throughput (a) varies about 3x with GPU temperature (4e8 warm, 1.4e9 cold). (b) and (c) are
  stable to about 10%. All rates are compute-bound and were measured on the exclusive allocation.
- The projection uses the measured single-GPU rate and the measured (sub-linear) 4-GPU speedup, so
  it is a lower bound. A real 2e6 run has far more work per GPU and scales at least as well,
  likely better, than the 2.23x point.
- "1e-3" for (b) is the worst-case relative error over the 348-point corpus including near-face
  points; the corresponding absolute error at n=24 is 1.2e-4.
