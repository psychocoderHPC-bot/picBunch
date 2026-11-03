# Benchmarking picBunch

`picBunchBench` measures the field pipeline inside picBunch on synthetic
bunches. It links the public `picBunch::computeField()` entry point, times each
stage through `StageTimes`, and reports the grid `makeGrid()` selected. It needs
no openPMD particle input and does not change the compute core.

## Build

```bash
cmake -S . -B build-bench -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPICBUNCH_ENABLE_CUDA=ON \
  -DPICBUNCH_ARCH=70 \
  -DPICBUNCH_BUILD_BENCH=ON \
  -DPICBUNCH_BUILD_TESTS=OFF \
  -DopenPMD_DIR=/opt/openpmd-hdf5-0.17.1
cmake --build build-bench --target picBunchBench -j 16
```

`PICBUNCH_BUILD_BENCH` defaults to off; the benchmark is an opt-in add-on.

## Single configuration

```bash
CUDA_VISIBLE_DEVICES=0 env -u LD_PRELOAD \
  ./build-bench/picBunchBench \
    --particles 2000000 --box-cells 20 --gamma 10 --spacing 1 \
    --repeat 3 --csv single.csv
```

The bunch fills `[0, box]^3` with a deterministic low-discrepancy sequence and
carries one shared momentum along +z (`|u| = sqrt(gamma^2 - 1)`), which is the
v0.1 single-velocity constraint. Every macro particle has `weight = 1` and
`charge = +e`. `--spacing` is the node spacing (`ComputeOptions::cellSize`), so
a fixed box with `--spacing 0.5` doubles the node count per axis.

The CSV columns are: configuration (`particles,box_cells,gamma,spacing,padding`),
grid (`Mx,My,Mz,nx,ny,nz`), per-stage min and median milliseconds
(`deposit,kernel,fft,readout`), `total_min_ms,total_median_ms,total_max_ms`, the exact allocation
`device_alloc_MiB` computed from the FFT box and particle count, and the
sampled `peak_device_MiB,peak_host_MiB`. One row is printed per configuration; repeats
default to 3. `peak_device_MiB` is sampled with `cudaMemGetInfo` in a helper
thread, `peak_host_MiB` is the reset `VmHWM` of the process.

## Paper matrix

```bash
CUDA_VISIBLE_DEVICES=0 env -u LD_PRELOAD \
  ./build-bench/picBunchBench --sweep --repeat 3 --csv matrix.csv
```

Runs, with padding 2:

- A particle scaling: box 20, gamma 10, N = 1e3, 1e4, 1e5, 1e6, 2e6
- B box scaling: N = 2e6, gamma 10, box = 20, 50, 100
- C gamma scaling: box 20, N = 2e6, gamma = 1, 10, 100
- D spacing: box 20, N = 2e6, gamma 10, spacing = 1, 1/2, 1/4

## Coulomb limit (u = 0)

```bash
CUDA_VISIBLE_DEVICES=0 env -u LD_PRELOAD \
  ./build-bench/picBunchBench --coulomb --csv coulomb.csv
```

Compares the mesh field of one stationary particle at far nodes against the
direct TSC stencil sum of the analytic Coulomb kernel. With padding 2 the
zero-padded FFT convolution equals that sum to roundoff; the tool prints
`COULOMB_MAX_REL` and exits 3 if it is not below 1e-12.

## Distance sweep

```bash
CUDA_VISIBLE_DEVICES=0 env -u LD_PRELOAD \
  ./build-bench/picBunchBench --d-sweep --csv dsweep.csv
```

For gamma 1 and 10 (aspect 1), a single particle sits in a 128-cell box and the
mesh field at axis distances d = 2..48 is compared with the analytic moving
point field. The worst relative error over the six axis directions and six
components is reported per d, so the 3e-4 crossover can be read off instead of
quoting one number. B is zero at gamma 1 by construction.

## Host hygiene

Record the GPU, its clocks, and whether the node is idle before a run:

```bash
nvidia-smi --query-gpu=index,name,clocks.current.sm,clocks.current.memory,utilization.gpu,memory.used \
  --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Pin one GPU with `CUDA_VISIBLE_DEVICES`; `computeField` always opens device 0 of
the visible set. Do not kill processes you do not own.
