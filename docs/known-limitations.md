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

**Why it is not fixed here.** The guard is a temporary workaround. The portable
replacement is an idempotent `alpaka::fft::onHost::finalize()` in alpakaVendor.
The API, the ordering contract, the implementation sketch, the picBunch migration
and the test plan are in
[the finalize proposal](proposals/alpaka-fft-finalize.md) and summarized in the
[ROCm FFT teardown](rocm-fft-teardown.md). No ROCm machine is available here, so
the guard cannot be removed and re-measured in this change.

**Plan.** Track the alpakaVendor `finalize()` API against the proposal. When it
exists, replace the native guard and rerun the teardown and `picBunch.Io` tests
on a ROCm host. Update the teardown document with the measured exit behavior and
drop the native call.

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

## 7. CI coverage depends on self-hosted capacity

**Symptom.** The repository now has a workflow, but only the CPU job runs on a
hosted runner. The CUDA and HIP jobs need self-hosted machines with the `gpu`,
`cuda` or `rocm` labels. Without those runners the device suites are skipped or
left unrequired, so a device regression can still land.

**Evidence.** [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) has the
`cpu`, `gpu-cuda` and `gpu-hip` jobs. The CPU job builds openPMD 0.17.1 from
source with HDF5 and MPI, caches the install prefix and runs the full CTest
suite. The HIP job runs the `FastNearConsistency`, `NearShellAccuracy` and
`NearShellAlpaka` groups as a required gate before the full suite. The README
build section still documents the manual commands, and the registered CTest
count depends on the build options.

**Why it is not fixed here.** Hosted runners do not provide a GPU, ROCm or the
large reference data. The GPU jobs are optional until the project provides
runners for the labels above.

**Plan.** Attach the self-hosted CUDA and ROCm runners, then mark both device
jobs as required checks. Keep the CPU job required on every change. Run
`actionlint` on workflow edits. Keep the openPMD cache key tied to the version,
the HDF5/MPI options and the runner image.

## 8. Scalar min/max extrema in the ported loops

**Symptom.** Eight scalar `std::min`/`std::max` accumulator sites in the ported
host loops still reduce one element at a time on the serial side instead of
going through an alpaka reduction.

**Evidence.** The sites are the moment scans in `src/Compute.cpp`
(`deriveFieldTailMargin`: `maxGamma`, `sourceRadius`), the far-tail pass in
`src/FarTail.cpp` (`deriveFarTail`: `supportRadius`, `gammaMax`), and the
velocity-binning scans in `src/VelocityBinning.cpp` (`firstBadMomentum`,
`firstBadCharge`, `scaleU`, `uMin`/`uMax`). Each is a running extremum over a
loop that already has a documented deterministic summation order.

**Why it is not fixed here.** The conversion to `alpaka::onHost::reduce` is
numerically exact for min/max, but it changes the loop structure around the
ordered floating-point accumulations these scans share. Doing it in the same
change as the clamp and span-type fixes would widen the bit-identity surface
without a dedicated validation run, so it is deferred.

**Plan.** Convert the eight sites to `alpaka::onHost::reduce` in a separate
change, then rerun the bit-stability pin, the read-out bitwise test and the
near-field golden/consistency suite on CPU and CUDA, and record the exact
extrema list in the commit message.

## 9. Dense near-shell scaling has no H100 or HIP row

**Symptom.** The dense-cloud near-shell cost is measured on V100 only. The
H100 and gfx1100 rows needed to choose a tile size per host are missing, so the
tile recommendation is proven for `sm_70` and only estimated for the other
backends.

**Evidence.** [Near-shell scaling](near-shell-scaling.md) records the V100
measurements at `N = 1e6` and `N = 1e7`, box 64/100/128, tile 2/4 and `R = 3/4`.
No H100 machine was available for that campaign. The H100 file in
[measurements](measurements/) is the earlier 200-particle rank scaling, not
this matrix. The gfx1100 probe ran on a shared card, was about 40x slower than
the idle V100, and the dense runs were killed before they produced a pair
count.

**Why it is not fixed here.** The V100 node was the only exclusive device
available. A HIP row needs an exclusive gfx1100 card and a profiler to separate
contention from kernel cost.

**Plan.** Re-run the same driver on an idle H100 and on an exclusive gfx1100
card, with the box, tile and radius grid in the scaling note. Add the tile 2
compact row and the `R = 4` row at `N = 1e7`. Update the recommendation and the
projections only after those rows exist.
