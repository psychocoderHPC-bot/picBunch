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

## Current mitigation

`RocfftTeardownGuard` in `src/Compute.cpp` performs the cleanup from application
code, while the alpaka queue and the HIP device are still alive:

```cpp
static std::once_flag rocfftCleanupOnce;
struct RocfftTeardownGuard
{
    decltype(queue)& queue;
    ~RocfftTeardownGuard()
    {
        alpaka::onHost::wait(queue);
        std::call_once(rocfftCleanupOnce, [] { static_cast<void>(rocfft_cleanup()); });
    }
} rocfftTeardownGuard{queue};
```

The guard is declared before the FFT plans in the same scope. C++ destroys it
after those plans, so the pool is cleaned after every plan is gone and before
the queue and device are released. `std::call_once` makes the native cleanup
run once per process; every call still waits for its own queue.

The block is compiled only with `ALPAKAV_DEP_ROCFFT`. It is absent from CPU and
CUDA builds and has no effect there.

This `rocfft_cleanup()` call is the only native vendor call in the tree. The
alpakaVendor pin (`33a4a4f401de2b157de195bb68e8ec61d7460cee`) has no public
teardown entry point, and the process-global ordering cannot be expressed
through plan RAII. The call is required until that entry point exists.

## Why the guard cannot be removed

Removing the guard reproduces the ROCm 7.2 segfault. T5b attempted the removal
and was reverted after the audit in T5c. The crash is in process-exit teardown,
so a normal test run that checks only the field output does not catch it unless
the process exit status is checked.

## Clean replacement

Add a process-wide teardown entry point to alpakaVendor:

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

With that API in the pin, the vendor body of the guard becomes:

```cpp
alpaka::onHost::wait(queue);
alpaka::fft::onHost::finalize();
```

and `#include <rocfft/rocfft.h>` disappears. The ordering guard remains,
because the call still has to run after plan destruction and before device
release. The guard then contains no native vendor code.

## Validation status

CPU and CUDA builds are validated. The ROCm path was not validated in the
environment where this note was written, because no ROCm machine was available
there. The mitigation exists for ROCm 7.2 only.

Run this on ROCm 7.2, gfx1100, with a HIP build:

```bash
for i in 1 2 3 4 5; do
  ctest --test-dir build/hip -R '^picBunch\.Io$' --output-on-failure || break
done
```

The check passes when all five runs exit with status 0 and no run segfaults
during static destruction. One exit-time crash keeps the workaround open.

## Risks

- Double cleanup relies on librocfft's own idempotency. The once flag in
  picBunch prevents a second call from this path, but the contract that a second
  `rocfft_cleanup()` is safe lives inside librocfft and is not documented here.
- A plan created after `finalize()` recreates the stream pool, and the once flag
  has already fired. That pool is not cleaned by `finalize()` again; only the
  original `atexit` path remains and with it the original ordering risk. Treat
  `finalize()` as the last FFT operation in the process.
- The call is correct only while the queue and device are alive. A call from a
  static destructor, or after the HIP runtime finalized, has the same defect as
  the original bug.
