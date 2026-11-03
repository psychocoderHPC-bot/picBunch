# picBunch

Standalone bunch-field initializer: reads an openPMD particle series, computes the
free-space electric and magnetic field of the bunch on a uniform grid, and writes the
six Yee-staggered components back as an openPMD series.

The pipeline (openPMD reader, TSC deposit, moving point kernel, padded FFT
convolution, Yee read-out, openPMD writer) is implemented. The physics
validation gates and their provenance are recorded in
[docs/algorithm.md](docs/algorithm.md).

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

## Run

```bash
picBunch --input <openPMD file/series> [--species <name|index>] \
         [--output <openPMD file>] [--iteration N] [--tolerance 1e-3] \
         [--backend auto|cuda|hip|cpu]
```

`--tolerance` is the only accuracy knob. Bin counts, interpolation orders and
FFT sizes are implementation details and are not exposed. The physical cell
size is taken from the input mesh (`E`) when the input has one; otherwise it
falls back to 1 m and prints a notice. The tool prints the grid used and a
per-stage timing table. Unsupported input (for example a momentum spread in
v0.1) exits non-zero with a message.

Example:

```bash
picBunch --input bunch_%T.bp5 --species electrons --output field_%T.h5 \
         --iteration 0 --tolerance 1e-3
```

## Test

`picBunchSmoke` prints the alpaka device name, the FFT backend and plan build status,
and the openPMD version with its HDF5 status. It is registered with CTest as
`picBunch.smoke`.

The GoogleTest suite is built when `PICBUNCH_BUILD_GTESTS=ON` (the default).
CTest registers six groups:

| test | what it proves |
| --- | --- |
| `picBunch.Deposit` | TSC partition of unity, support (+-1.5 cells), single-particle weight |
| `picBunch.Kernel` | kernel symmetry/antisymmetry, stationary Coulomb + TSC limit |
| `picBunch.Fft` | round-trip, convolution vs direct, R2C/C2R vs C2C packed-plane handling |
| `picBunch.Io` | openPMD write/read values and all mesh metadata |
| `picBunch.Physics` | one particle vs an independent high-order reference |
| `picBunch.Padding` | no-padding wraparound artefact and padded convergence |

## License

MPL-2.0, see [LICENSE](LICENSE).
