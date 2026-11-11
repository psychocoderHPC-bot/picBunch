# rocFFT teardown on ROCm 7.2

This note records a process-exit crash in the ROCm build, the temporary
workaround in `src/Compute.cpp`, and the API that should replace it.

## Symptom

On ROCm 7.2 with gfx1100 (`hal`), `picBunch.Io` segfaulted at process exit.
The crash happened in librocfft while its global HIP stream pool was torn down
during static destruction. The first fix landed in commit `833b627`.

## Root cause

`rocfft_setup()` creates a process-global pool of HIP streams. alpakaVendor
registers `atexit(rocfft_cleanup)` the first time a plan is used. C++ gives no
ordering guarantee between the static destructors and `atexit` handlers of
libamdhip64 and librocfft. The HIP runtime can finalize before rocfft destroys
its pool. `hipStreamDestroy` then runs on streams that no longer exist, and the
process segfaults after `main` returns.

Plan RAII does not help. The stream pool is process-global, so its lifetime is
not tied to any plan object. Destroying every plan correctly still leaves the
teardown order between the two libraries unspecified.

## ROCm 7.2 HIP inliner defect

PR #14 added a batched tile kernel as a second caller of `dshapeField()` and
`dmeshComp()` in the same HIP module. On ROCm 7.2 gfx1100, HIP clang stopped
inlining both helpers at `-O3` once the second caller existed. It emitted
out-of-line device functions and passed a reference into the kernel's by-value
`DeviceParams` kernarg block through the call. The outlined function does not
materialise that reference in private memory, so the quadrature tables read
garbage and the near field is wrong. The plain C++ `inline` keyword did not
keep the calls inlined.

The failure does not crash and the process exits with status 0. Only the field
values are wrong, so a test that checks exit status or a loose field tolerance
can pass on a broken build.

Commit `4b1250d` marks both helpers `ALPAKA_FN_INLINE`, which carries the
backend force-inline attribute. Each kernel then reads its own parameter block
and the device result matches the host evaluator. The annotation pins codegen
on ROCm 7.2; it does not change the numerics.

### Detection

Run the device-vs-host near-field tests on the HIP build:

```bash
ctest --test-dir build/hip -R 'NearShellAccuracy\.Device' --output-on-failure
```

These tests compare the device evaluator against the host `polyBoxFieldS` path
at round-off. They were CUDA-only before `4b1250d`; the fix enables them for
`PICBUNCH_BACKEND_HIP`. A wrong near field without a crash is the signature of
the outlined call. The field difference against the pre-fix build was
`8.6e-5` to `9.9e-5`.

A full near-shell run on gfx1100 should report `NearShellAccuracy` 30/30,
`NearShellAlpaka` 5/5, `FastNearConsistency` 3/3, `FastNearPoly` 6/6 and
`Margin` 11/11.

## Current mitigation

`src/Compute.cpp` registers a process-exit handler after the FFT plans have been
created, instead of calling cleanup inline at the end of `computeField`:

```cpp
static std::once_flag rocfftAtexitOnce;
std::call_once(
    rocfftAtexitOnce,
    [] { std::atexit([] { static_cast<void>(rocfft_cleanup()); }); });
```

Calling cleanup inline was wrong: the librocfft stream pool is process-global,
so the first `computeField` destroyed it and every later `computeField` in the
same process failed in `rocfft_plan_create` with error 1. The process also
crashed on exit.

The handler is registered after both plans are built. C++ runs `atexit`
handlers in reverse order of registration. alpakaVendor registers its own
`rocfft_cleanup` hook when the first plan is created, and the HIP runtime
registers its static destructors before that. Registering ours later therefore
makes it run first: before the HIP static destructors, after the plans (locals,
destroyed when `computeField` returns) are gone, and while the HIP runtime is
still alive.

`std::call_once` registers exactly one handler per process. When it runs,
alpakaVendor's own atexit hook still calls `rocfft_cleanup()` afterwards; that
second call is assumed to be a no-op by librocfft. The idempotency of
`rocfft_cleanup()` is the contract this workaround relies on. It is not
documented by ROCm, so re-check it when the ROCm version changes.

The block is compiled only with `ALPAKAV_DEP_ROCFFT`. It is absent from CPU and
CUDA builds and has no effect there.

This `rocfft_cleanup()` call is the only native vendor call in the tree. The
alpakaVendor pin (`33a4a4f401de2b157de195bb68e8ec61d7460cee`) has no public
teardown entry point, and the process-global ordering cannot be expressed
through plan RAII. The call is required until that entry point exists.

## Why the handler cannot be removed

Removing the handler reproduces the ROCm 7.2 segfault. T5b attempted the removal
and was reverted after the audit in T5c. The crash is in process-exit teardown,
so a normal test run that checks only the field output does not catch it unless
the process exit status is checked.

## Clean replacement

Add a process-wide teardown entry point to alpakaVendor. The full proposal,
including the API contract, implementation sketch, migration note and test
plan, is in
[proposals/alpaka-fft-finalize.md](proposals/alpaka-fft-finalize.md).

```cpp
namespace alpaka::fft::onHost
{
    // Process-wide FFT backend shutdown. Idempotent, no exceptions.
    void finalize() noexcept;
}
```

Semantics:

- HIP/rocFFT: destroy librocfft's process-global stream pool with
  `rocfft_cleanup()`. The call is guarded by a process-wide `std::once_flag`,
  so only the first `finalize()` per process reaches the vendor API.
- cuFFT, FFTW and oneMKL: no-op.
- The caller destroys every plan and waits on its queue before the call. The
  function takes no queue and does not synchronize.
- The queue and device must outlive the call.
- Only the first call does work. Later calls return immediately.
- The function does not throw and must not race with plan creation or
  destruction. Call it once, after the last FFT operation.

With that API in the pin, the body of the registered `atexit` handler becomes
`alpaka::fft::onHost::finalize()` and `#include <rocfft/rocfft.h>` disappears.
The handler registration remains, because the call still has to run after plan
destruction and before the HIP static destructors. Calling `finalize()` inline
at the end of each `computeField` is still wrong: it would destroy the
process-global pool after the first call while later calls create plans. The
registered handler then contains no native vendor code.

## Current state after the inline workaround

The teardown guard is present in `src/Compute.cpp` and the inliner fix is in
`src/FastNearDevice.cpp`. Both are on main at `870c2e6`.

The guard was validated on ROCm 7.2, gfx1100, with five `picBunch.Io` runs and
multi-`computeField` processes as part of the 34/34 HIP run in `741a46b`. The
inline fix was validated on the same target: `NearShellAccuracy` 30/30 with
the four device-vs-host tests enabled for HIP, `NearShellAlpaka` 5/5,
`FastNearConsistency` 3/3, `FastNearPoly` 6/6, `Margin` 11/11 and the CLI.
CPU and CUDA sm_70 also pass 30/30. sm_90 was not reachable from the build
host for `4b1250d`.

The HIP path is therefore validated for ROCm 7.2 only. Neither workaround is a
fix for the underlying vendor behavior.

## Retest policy on ROCm updates

Run this policy on every ROCm version, HIP clang or gfx-target change, and
before moving the ROCm pin:

1. Build the HIP preset and run the device-vs-host near-field tests:
   `ctest --test-dir build/hip -R 'NearShellAccuracy\.Device' --output-on-failure`.
2. Run the full near-shell suite and check the counts recorded above.
3. Apply the five-run Io policy. Each run is a separate process, and the
   policy fails on the first non-zero exit:

   ```bash
   for i in 1 2 3 4 5; do
     ctest --test-dir build/hip -R '^picBunch\.Io$' --output-on-failure || break
   done
   ```

4. Run one process with repeated `computeField` calls and confirm that every
   call produces the same field and that no call fails in `rocfft_plan_create`.
5. Record the ROCm version, gfx target, commit and test counts in this note.

The five-run Io policy passes when every run exits with status 0 and no run
segfaults during static destruction. One exit-time crash keeps the teardown
workaround open. Do not shorten the policy to a single run: the crash happens
after `main` returns, and a repeated process is the cheapest way to expose the
static-destruction order. Keep the HIP device-vs-host tests enabled; do not
return them to a CUDA-only guard.

Do not remove `ALPAKA_FN_INLINE` from `dshapeField()` or `dmeshComp()` because
a build happens to pass. Remove it only after a ROCm release demonstrably
handles the outlined call, and only when the device-vs-host tests pass without
the annotation and the field difference is at round-off.

Re-check `rocfft_cleanup()` idempotency on each ROCm update. The current guard
depends on a second call being safe. That contract is not documented by ROCm.

## Risks

- Double cleanup relies on librocfft's own idempotency. The once flag in
  picBunch prevents a second call from this path, but the contract that a second
  `rocfft_cleanup()` is safe lives inside librocfft and is not documented here.
- A plan created after the handler has run recreates the stream pool, and the
  once flag has already fired. That pool is not cleaned again by picBunch; only
  the original alpakaVendor `atexit` path remains and with it the original
  ordering risk. The handler therefore has to be the last FFT teardown in the
  process, which holds because it is registered at process exit.
- The second `rocfft_cleanup()` from alpakaVendor's atexit hook relies on
  librocfft's idempotency. If a future ROCm release makes the call destructive
  on a second use, the process would crash at exit even with this workaround.
- A plan created after the handler runs recreates the stream pool and is not
  cleaned by this once-guarded handler.
