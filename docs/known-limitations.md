# Known limitations

This document tracks the open engineering items of the host-loop and kernel port.
It is not the physics error budget. The measured accuracy envelopes and their
provenance stay in the [README error budget](../README.md#error-budget) and in
[algorithm.md](algorithm.md). Each entry below records the symptom, the evidence
or measurement that establishes it, why the item is not fixed in this change, and
a concrete plan.

## 1. Far-field accuracy plateaus at gamma >= 10

**Symptom.** For gamma 10 and above the far-field relative error of the
node-sampled point Liénard kernel stops improving. At the production padding the
error plateaus near `1.09e-2`. The measured error at kernel oversampling `m = 1`
is identical for gamma 10, 30 and 100, so refining gamma no longer buys accuracy.

**Evidence.** The plateau is reproduced with the production padding and the
`m = 1` kernel. The analytic Fourier symbol of the point kernel is correct, but
its discrete evaluation on the production padded box only reaches about `1e-2`.
Raising the kernel oversampling to `m = 16` lowers the measured error to about
`1e-7`, at a cost of several times the kernel build. The mechanism is geometric:
the moving point kernel compresses the source into a sheet of thickness `d/gamma`.
When that thickness falls below the node spacing, the node-sampled kernel cannot
resolve the sheet, and the aliasing error floors independently of gamma.

**Why it is not fixed here.** A fix changes the operator, not a local constant.
The current FFT convolution evaluates the point kernel by direct node sampling.
Resolving the sheet by brute force (`m = 16`) is too expensive, and a proper split
needs a new validation path and a new cost model against the order-12 `indfield`
arbiter. That is outside a host-loop port.

**Plan.**
1. Add a convergence harness that reports far-field relative error against gamma
   at fixed padding and fixed `m`, so the plateau is a tracked quantity.
2. Prototype an Ewald-like split: a smooth long-range part evaluated by FFT and a
   short-range part evaluated locally near the source, with the splitting length
   chosen from gamma and the cell size. The alternative is a local correction that
   uses the exact s-space evaluator inside the unresolved `d/gamma` sheet.
3. Quantify the `<= 10%` far-field budget as a separate line in the error budget,
   with the mesh/deposit, velocity binning, near quadrature, far-kernel sampling
   and output-precision terms listed independently.
4. Gate the change on gamma 10 to 100 far-field accuracy at production padding
   and on a measured cost that stays within an agreed multiple of the kernel build.

## 2. Distributed MPI read-out stages through unified memory

**Symptom.** The distributed read-out stages per-slab output planes in unified
(managed) allocations. The first touch of a migrated page stalls the read-out, and
the staging path does not share the allocation discipline that fixed the
replicated path. This is the same page-migration class of problem as the
replicated buffer fix.

**Evidence.** The replicated path allocated its transform buffers as 3-D
`allocUnified` extents. The 128-byte row padding was never cleared, and reused
buffers carried the previous velocity bin's charge into the transform; the fix was
to allocate flat 1-D counts (see the v0.9.1 notes in the README). The distributed
read-out now performs the output-plane exchange and scales each local slab, and its
staging still crosses the managed boundary. The distributed timing tables show the
read-out cost including that exchange.

**Why it is not fixed here.** A staging change must keep the written slab
bit-identical to the replicated path and to a different rank count. Moving the
staging without that invariant risks a silent rounding difference. A parallel task
implements the device-staging fix and uses this entry as its reference.

**Plan.** Stage the output planes in device memory and copy each component once
into the writer slab, removing managed staging from the read-out hot path. Verify
bit-identity against the replicated field for 1, 2 and 4 ranks, and keep the
existing distributed regression test as the guard. Record the new read-out time in
the distributed timing table.

## 3. Register pressure of the exact near-field kernel

**Symptom.** The exact near-field device kernel uses 255 registers and admits one
block per SM. That is 25% occupancy on `sm_70` and 12.5% on `sm_80`/`sm_90`.
Depending on architecture and order it spills 36 to 232 bytes per thread.

**Evidence.** The register, spill and occupancy figures come from the `ptxas`
resource report for the exact near-field kernel. The kernel is bound by FP64 libm
throughput (`sqrt`, `asinh`, division), not by occupancy: the measured throughput
follows the transcendental issue rate, so adding resident warps does not recover
the lost time.

**Why it is not fixed here.** The register count is set by the inlined exact
s-space path, the Duffy tetrahedral box and the eight-term `asinh` box, all with
long live ranges. The device copy deliberately evaluates the 27 support cells
inline so the shared 7.8 KB moment array is not spilled to local memory. Reducing
pressure without changing accuracy needs per-architecture measurement.

**Plan.**
1. Shorten live ranges and recompute cheap intermediates instead of keeping them
   live across the quadrature.
2. Evaluate whether the shared moment table belongs in constant or global memory
   for the device path.
3. Split the exact path and the remainder into separate kernels so each carries a
   smaller live set.
4. Measure registers, spills and occupancy on `sm_70`, `sm_80` and `sm_90`, and
   accept a change only if the near-shell accuracy and wall time do not regress.

## 4. Host and device evaluator duplication

**Symptom.** The exact s-space evaluator exists twice. The host implementation is
`detail::polyBoxFieldS` in `include/picBunch/detail/FastNear.hpp`. The device
implementation is `dpolyBoxFieldS` in `src/FastNearDevice.cpp`, whose comment
states that it mirrors the host code. The two copies use different memory
strategies: the host shares 108 unique face moments across neighbouring cells,
while the device evaluates the two faces per axis inline for each of the 27 cells.

**Evidence.** The duplication is visible in the two source files and in the
device comment. The current outputs agree, with device and host fields matching to
about `4.1e-14` of the peak, but there is no single source of truth and no test
that would catch a divergence in a future edit.

**Why it is not fixed here.** A shared `ALPAKA_FN_HOST_ACC` functor must compile
for host, CUDA and HIP, and a naive merge would force the host moment layout onto
the device path and reintroduce the local-memory spill. A parallel task performs
the merge together with the consistency test.

**Plan.** Extract one `ALPAKA_FN_HOST_ACC` functor, or one generated table plus
code, with the memory layout selected per backend behind a single interface. The
host keeps the shared-moment path and the device keeps the inline path. Add a
consistency test that evaluates the 52-point set and random source-observer pairs
on host and device and asserts agreement at round-off. Reference this entry from
the merged implementation.

## 5. ROCm FFT teardown

**Symptom.** On ROCm 7.2 the process can segfault during static destruction in
librocfft's process-global HIP-stream pool. A native `rocfft_cleanup()` guard in
`src/Compute.cpp` avoids the crash and is ordered after the FFT plans and before
the alpaka queue and HIP device are destroyed.

**Evidence.** The guard is the only native vendor API call left in the tree. It is
wrapped in `std::once_flag`/`std::call_once` and fixed the `picBunch.Io` teardown
regression on the validated `hal` host. The full teardown analysis is in the
companion [ROCm FFT teardown](rocm-fft-teardown.md).

**Why it is not fixed here.** The portable replacement is an idempotent
`alpaka::fft::onHost::finalize()` in alpakaVendor. No ROCm machine is available
here, so the guard cannot be removed and re-measured in this change.

**Plan.** Track the alpakaVendor `finalize()` API. When it exists, replace the
native guard and rerun the teardown and `picBunch.Io` tests on a ROCm host. Update
the teardown document with the measured exit behavior and drop the native call.

## 6. CPU kernel-build sub-stage regression

**Symptom.** The fused compute pass raises the CPU kernel-build sub-stage by 21%,
from `3.43 ms` to `4.15 ms`. The end-to-end time does not regress because the FFT
and read-out stages become cheaper by more than that.

**Evidence.** The two stage times were measured with the per-stage timer on the
CPU host path, before and after the fused pass restructure. The aggregate stage
sum is lower after the change, so the fusion wins overall despite the local
kernel-build cost.

**Why it is not fixed here.** Reverting the kernel-build part alone would also
undo the FFT and read-out gain, which is larger. Splitting the fusion again needs
a profile of where the fused pass spends its time.

**Plan.** Revisit the fusion if the kernel-build share grows or becomes the
bottleneck. Profile the fused pass, then either restore a dedicated build pass or
specialize the fused loop by stage. Track the sub-stage time in the CPU cost table
so the regression is visible next to the total.

## 7. No in-repo CI

**Symptom.** There is no workflow in the repository. The test suite is run
manually, so a change can land without the regression tests having run, and the
registered CTest count depends on the build options.

**Evidence.** The repository has no CI configuration. The README build section
documents manual configure, build and CTest commands for each backend, and the
docs note that CTest counts differ between configurations.

**Why it is not fixed here.** The validated configurations need GPU and ROCm hosts
that standard hosted runners do not provide, and the reference data are large.
Adding CI needs a decision on hosted versus self-hosted capacity.

**Plan.** Add a GitHub Actions workflow for a CPU/FFTW configure, build and CTest
run, plus a documentation link check. Add optional self-hosted CUDA and ROCm jobs
for the validated presets. Require the CPU job on every change. If the project
does not want hosted CI, record the manual release checklist instead.

## 8. Scalar min/max extrema in the ported loops

**Symptom.** A few host-side scalar `std::min`/`std::max` accumulators in the
ported loops still run serially instead of through an alpaka reduction.

**Evidence.** `velocityMomentsRadius` in `src/VelocityMoments.cpp` now uses
`alpaka::onHost::transformReduce` with a SIMD-capable max functor from
`include/picBunch/detail/Reduction.hpp`. The remaining host extrema are
`maxGamma` and `sourceRadius` in `src/Compute.cpp` (`deriveFieldTailMargin`),
`supportRadius` and `gammaMax` in `src/FarTail.cpp` (`deriveFarTail`),
`Bunch::bounds`, the per-box gamma range and maximum member angle in
`src/VelocityBinning.cpp`, and the field peaks in `src/main.cpp`. The
`firstBadMomentum`, `firstBadCharge`, `scaleU`, and `uMin`/`uMax` entries in an
earlier version of this section were stale: those run inside alpaka kernels,
and their serial `std::min`/`std::max` branches only execute when atomics are
disabled on the CPU lane.

**Why it is not fixed here.** The reduction is exact for min/max, but it is
slower for these host-array scans. Release CPU, median of 9 samples, 1e6
particles: `Bunch::bounds` 2.4 -> 8.5 ms, `deriveFarTail` 13.6 -> 33.5 ms, and
the velocity-binning statistics pass 64 -> 90 ms. The converted
`velocityMomentsRadius` is 6.8 -> 4.8 ms at 4e6 particles and about even at
1e6 (1.5-1.9 ms against 1.6-1.7 ms serial).

**Plan.** Keep the serial extrema unless profiling shows that one dominates.
The `Compute.cpp` moment scans and the `main.cpp` field peaks stay in the pass
that owns those files.
