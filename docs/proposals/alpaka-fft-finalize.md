# Proposal: `alpaka::fft::onHost::finalize()`

Status: draft for alpakaVendor. Target pin: after
`33a4a4f401de2b157de195bb68e8ec61d7460cee`.

## Problem

rocFFT keeps a process-global pool of HIP streams. On ROCm 7.2 the library
destroys that pool from a static destructor and calls `hipStreamDestroy` after
the HIP runtime has finalised, which segfaults at process exit. Plan RAII cannot
fix this because the pool is not owned by any plan.

picBunch currently calls `rocfft_cleanup()` from a native `atexit` handler in
`src/Compute.cpp`. The handler is registered after the FFT plans are built, so
it runs before the HIP static destructors. It is the only native vendor call in
the tree and it relies on a second `rocfft_cleanup()` being harmless. The
ordering problem and the temporary guard are described in
[../rocm-fft-teardown.md](../rocm-fft-teardown.md).

This proposal moves that teardown behind one public alpakaVendor entry point.

## Proposed API

Header: `<alpaka/fft.hpp>` (umbrella) with the declaration in the onHost FFT
detail headers.

```cpp
namespace alpaka::fft::onHost
{
    //! Process-wide FFT backend shutdown.  Idempotent, no exceptions.
    void finalize() noexcept;
}
```

The function takes no queue and no device. It does not synchronize.

## Semantics

| Backend | Behavior on the first call | Later calls |
| --- | --- | --- |
| HIP / rocFFT | call `rocfft_cleanup()` once | return immediately |
| CUDA / cuFFT | no-op | no-op |
| CPU / FFTW | no-op | no-op |
| oneMKL | no-op | no-op |

- `finalize()` is idempotent. A process-wide `std::once_flag` guards the vendor
  call, so only the first call reaches librocfft.
- `finalize()` is thread-safe with respect to itself. It is not synchronized
  against plan creation, plan destruction or FFT execution.
- `finalize()` never throws. A vendor error is dropped.
- `finalize()` does not replace plan destruction. Plans still follow their RAII
  lifetime and must be destroyed before the call.
- Creating a plan after `finalize()` is unsupported. The call is the last FFT
  operation in the process.

## Ordering contract

The caller must satisfy all of these:

1. Destroy every FFT plan before the call. No plan may outlive the pool.
2. Keep the queue and the device alive across the call. Do not destroy the
   queue first.
3. Call before the HIP runtime static destructors. At process exit this means
   registering the handler after the plans and after the HIP runtime has
   registered its destructors, so the handler runs first.
4. Do not race the call with plan creation, plan destruction or FFT work.
5. Do not create a plan after the call.

`finalize()` does not wait for the queue. The caller waits before it destroys
the queue or calls `finalize()`.

The call can be made directly while the runtime is alive, or from a late
`atexit` handler. The `atexit` form is required when the queue or device is
destroyed during static destruction.

## Reference implementation sketch

```cpp
// alpakaVendor FFT onHost finalize translation unit
#include <cstdlib>
#include <mutex>

#if defined(ALPAKAV_DEP_ROCFFT) && ALPAKAV_DEP_ROCFFT
#    include <rocfft/rocfft.h>
#endif

namespace alpaka::fft::onHost
{
    void finalize() noexcept
    {
#if defined(ALPAKAV_DEP_ROCFFT) && ALPAKAV_DEP_ROCFFT
        static std::once_flag once;
        std::call_once(once, [] { static_cast<void>(rocfft_cleanup()); });
#endif
    }
} // namespace alpaka::fft::onHost
```

Route the existing internal hook through `finalize()`:

```cpp
#if defined(ALPAKAV_DEP_ROCFFT) && ALPAKAV_DEP_ROCFFT
    static std::once_flag atexitOnce;
    std::call_once(
        atexitOnce,
        []
        {
            std::atexit([] { alpaka::fft::onHost::finalize(); });
        });
#endif
```

This makes the second cleanup a no-op inside alpakaVendor. It removes the
dependency on librocfft's undocumented idempotency for the vendor's own hook.
It does not change the ordering contract: the caller still registers its late
handler after plan construction.

## Migration note for picBunch

Keep the `atexit` registration and the `std::once_flag` in `src/Compute.cpp`.
Replace only the native call in the handler:

```cpp
// before
static std::once_flag rocfftAtexitOnce;
std::call_once(
    rocfftAtexitOnce,
    [] { std::atexit([] { static_cast<void>(rocfft_cleanup()); }); });
```

```cpp
// after
static std::once_flag fftFinalizeAtexitOnce;
std::call_once(
    fftFinalizeAtexitOnce,
    [] { std::atexit([] { alpaka::fft::onHost::finalize(); }); });
```

Then remove `#include <rocfft/rocfft.h>` and every `ALPAKAV_DEP_ROCFFT`
conditional that exists only for the teardown guard. Leave the branch that
keeps HIP plans owned by `computeField` unchanged unless plan destruction at
process exit is also guaranteed.

Do not call `finalize()` inline at the end of each `computeField`. The pool is
process-global. An inline call destroys it after the first call while later
calls still create plans, which reproduces the `rocfft_plan_create` error 1
failure.

## Test plan

Run on ROCm 7.2, gfx1100, HIP build, and on CPU and CUDA for the no-op paths.

1. **Five sequential Io runs.**
   ```bash
   for i in 1 2 3 4 5; do
     ctest --test-dir build/hip -R '^picBunch\.Io$' --output-on-failure || break
   done
   ```
   Every run must exit with status 0. No run may segfault during static
   destruction.

2. **Repeated `computeField` in one process.** Call `computeField` at least
   three times, including two calls with the same shape. Every call must return
   the same field to round-off, and no call may fail in `rocfft_plan_create`.

3. **Idempotent `finalize()`.** Call `finalize()` twice after all plans are
   destroyed. The process must exit with status 0 and the second call must not
   reach librocfft.

4. **No-op backends.** Call `finalize()` in a CPU/FFTW test and in a CUDA test.
   Both must link and run without a rocFFT teardown symbol and without changing
   the FFT results.

5. **Ordering test.** Register `finalize()` from an `atexit` handler after plan
   construction, then let the process exit with the queue still alive at handler
   time. The process must exit with status 0.

6. **Device-vs-host near-field regression.** Run
   `ctest --test-dir build/hip -R 'NearShellAccuracy\.Device'` after the
   migration. Teardown changes must not hide the ROCm 7.2 inliner failure.

Acceptance: all six pass on ROCm 7.2 gfx1100, the full CPU and CUDA suites
pass, and picBunch contains no `rocfft` include and no `rocfft_cleanup()` call.
