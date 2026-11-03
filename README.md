# picBunch

Standalone bunch-field initializer: reads an openPMD particle series, computes the
free-space electric and magnetic field of the bunch on a uniform grid, and writes the
six Yee-staggered components back as an openPMD series.

The project is at skeleton stage. The build, the dependency pins and the smoke test
exist; the physics validation gates are still open (see [docs/algorithm.md](docs/algorithm.md)).

## Layout

```
include/picBunch/   public headers (types, TSC deposit, point-field kernel, FFT self test)
src/                host implementation, per-backend device sources under src/cuda, src/hip
tests/              smoke test (CTest) and, optionally, GoogleTest unit tests
docs/               algorithm and validation notes
```

## Dependencies

| Dependency | Pin | Notes |
| --- | --- | --- |
| alpaka | `7b92361468d3280c260200198081ae7e98f0bc56` (reports 3.0.0) | fetched by CMake `FetchContent` |
| alpakaVendor | `af24cc7b2bd125056e9889e270d3af7bdf972c6f` | fetched by CMake `FetchContent`; FFT via cuFFT/rocFFT/FFTW |
| openPMD-api | system install | HDF5 backend required; configure fails without it |
| FFTW 3 | system (`fftw3`, `fftw3f`) | host FFT backend |
| GoogleTest | system first, `FetchContent` fallback | only when `PICBUNCH_BUILD_GTESTS=ON` |

CMake 3.25+ and a C++20 compiler are required. The CUDA build needs CUDA 12.x, the HIP
build needs ROCm with rocFFT.

## Build

`openPMD_DIR` points at the openPMD-api package directory (the install prefix or its
`lib/cmake/openPMD` subdirectory):

```bash
cmake --preset cuda-v100 -DopenPMD_DIR=/path/to/openpmd/lib/cmake/openPMD
cmake --build --preset cuda-v100
ctest --preset cuda-v100
```

`PICBUNCH_ARCH` is the default for `CMAKE_CUDA_ARCHITECTURES` (CUDA) or
`CMAKE_HIP_ARCHITECTURES` (HIP). An explicit `-DCMAKE_CUDA_ARCHITECTURES=...` takes
precedence. Device sources are picked up only from `src/cuda/*.cu` (CUDA build) or
`src/hip/*.hip` (HIP build); the same sources are never compiled by both languages.

Other presets: `hip-gfx1100`, `cpu-debug`. Every preset builds into
`build/<preset>/`, which is gitignored.

## Test

`picBunchSmoke` prints the alpaka device name, the FFT backend and plan build status,
and the openPMD version with its HDF5 status. It is registered with CTest as
`picBunch.smoke`.

## License

MPL-2.0, see [LICENSE](LICENSE).
