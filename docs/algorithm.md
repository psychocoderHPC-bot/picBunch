# Algorithm

This document names the pipeline stages and records what each stage must prove.
Numbers are recorded only where they have been measured in validation; geometry
and build conditions are stated with them.

## Stages

1. **Reader** - open an openPMD particle series and load one species: positions,
   dimensionless momenta, weights, charge. Position convention is the openPMD
   standard `x = position * unitSI + positionOffset * unitSI_offset`. Momentum is
   converted with `u = p * unitSI / (m c)`. Scalar records are accepted in both
   openPMD 1.x plain-dataset form and openPMD 2.0 group form (child `value` or a
   child named after the record, attributes on the group or the child). Mesh
   geometry follows the same rule:
   the physical spacing is `gridSpacing * gridUnitSI`, and the physical offset is
   `gridGlobalOffset * gridUnitSI`. A mesh without a positive `gridUnitSI` is
   rejected instead of being silently rescaled. TODO: numbers from validation
   (species size, units round-trip error).

2. **Deposit** - TSC assignment of `weight * charge` onto the uniform output grid.
   The deposit must conserve total charge and must match the device deposit node
   for node. TODO: numbers from validation (charge conservation error, deposit
   versus reference).

3. **Kernel** - evaluate the free-space field of the bunch as a convolution of the
   deposited source with the shape-weighted, uniformly moving point-charge kernel.
   There is no separate near-field quadrature: the smoothing comes from the
   deposit-convolved-with-kernel composition, which is O(s^2) in the assignment
   shape width. The `D = 0` guard in the analytic point-field kernel never fires
   for the six Yee components, because each component's half-cell staggering
   keeps |R| > 0 at every grid node. Near the source the remaining error is the
   coarseness of the v0.1 TSC assignment at `kRefinement = 1`; see the known
   limitation below. TODO: numbers from validation (relative field error per
   band, moving-bunch gamma cases).

4. **FFT** - transform the zero-padded source and kernel, multiply in Fourier
   space, and transform back. The FFT is reached through the alpakaVendor portable
   API (cuFFT, rocFFT, or FFTW depending on the build). TODO: numbers from
   validation (plan build time, transform throughput, padding factor).

5. **Readout** - sample the six field components at their Yee-staggered positions,
   interpolate from the padded FFT grid to the output grid, apply units so E is in
   V/m and B is in T. TODO: numbers from validation (staggering check, divergence
   and curl residuals).

6. **Writer** - validate all six components before any store, write `E` and `B`
   as openPMD mesh records with components
   `x,y,z`, stored C-order with axis labels `z,y,x` (x fastest), `geometry =
   cartesian`, `gridSpacing` in metres, `gridUnitSI = 1.0`, `gridGlobalOffset`
   from the grid origin, `timeOffset = 0`, and a configurable iteration number.
   Each component carries its Yee staggering through the `position` attribute:
   `Ex(.5,0,0) Ey(0,.5,0) Ez(0,0,.5) Bx(0,.5,.5) By(.5,0,.5) Bz(.5,.5,0)`.
   Values are written in SI (V/m for E, T for B) with `unitSI = 1.0`, and the
   `unitDimension` exponents are E = (1,1,-3,-1) and B = (0,1,-2,-1). The
   series is written under a temporary name in the destination directory and
   renamed only after a successful flush, so a failed write leaves no partial
   output file.

   **PIConGPU restart:** `--restart-format picongpu` writes float32 internal-unit
   fields and copies the input checkpoint's `Convolutional PML ...` records
   verbatim, converting only a non-float32 slab. With no PML records in the
   input it writes none and an open-boundary restart cannot proceed. The
   verified restart and field-arrival numbers are in
   [picongpu-restart.md](picongpu-restart.md).

   **PIConGPU restart defect:** PIConGPU restart currently ignores the mesh
   `unitSI` attribute (verified defect), so a PIConGPU consumer of these files
   must convert with `sim.unit.eField()` / `sim.unit.bField()`. The writer does
   not silently switch to internal PIConGPU units; it writes SI and says so
   here.

## Tolerance contract

`--tolerance` is the only user-facing accuracy knob. It is a relative target for
the delivered *discrete free-space TSC convolution*, i.e. the discrete sum the
pipeline is designed to evaluate. It is not a target for the continuum Coulomb
field near the source; the difference there comes from the fixed TSC deposit at
`kRefinement = 1` and no tolerance value can reduce it.

For a single-velocity bunch the target controls exactly one implementation
choice: the FFT zero-padding factor. It does not change the interpolation order
(fixed at TSC), the deposit scheme, the node spacing, or the refinement. The
mapping is a measured error-vs-padding model:

| requested tolerance | padding | basis |
| --- | --- | --- |
| `>= 1e-12` | 2 | measured padding-2 self-convergence is at most `8.37e-16` (toy) / `4.15e-16` (500-particle cloud), below the conservative bound `1e-12` |
| `1e-14 .. 1e-12` | 3 | tighter than the padding-2 bound, so the next padding is used |
| `< 1e-14` | rejected (non-zero exit) | below the measured double-precision floor; refusing is better than silently under-delivering |

Padding 1 is never selected by the policy. It is a diagnostic override used by
the padding test only.

For a multi-velocity bunch the same target additionally sizes the velocity
boxes (see "Velocity binning"): the box count is the smallest power of two
whose calibrated error model stays below the target, and an infeasible
requirement fails with the needed count instead of running.

## Velocity binning

The mesh kernel is exact only for a single shared `u`.  The field is linear in
charge, so the field of a mixed bunch is the sum over velocity classes of the
field of each class.  The particles are split into uniform boxes in `u`-space,
in a frame aligned to the charge-weighted mean momentum direction, and each
box is evaluated with its box-centre `u` (one box: the charge-weighted mean
`u`).  Each box pays one deposit, one density transform, six kernel builds,
six kernel transforms, six products, six inverse transforms and six read-outs,
accumulated into the same output; empty boxes cost nothing.

The box geometry depends only on the bunch `u` extent and the box counts,
never on which particles fall where.  That keeps the near shell consistent:
its per-tile mesh subtraction reuses the full-bunch boxes on a sub-bunch, and
its exact term evaluates every (source, node) pair with the source's own
velocity.  The mesh cancels bin for bin and the near shell is therefore
bin-free: nodes inside the shell carry the exact per-particle-velocity field,
and only the outside-shell field carries the binning error.

Error model.  Replacing a particle `u` by its box representative perturbs the
field by a transverse tilt of the boosted pattern, of relative order
`gamma * Delta_theta`, and an energy shift, of relative order
`Delta_gamma / gamma`.  With `q_b` the charge in box `b`, `Q` the total
absolute charge, `gamma_b` the largest `gamma` in the box, `Delta_theta_b`
the largest member angle against the representative, and `Delta_gamma_b` the
`gamma` range of the box, the estimated relative error is

    modelError = C * Sum_b (q_b / Q) * (gamma_b * Delta_theta_b
                                         + Delta_gamma_b / gamma_b)

`C` is calibrated against the TSC-convolved continuum reference by running a
mixed bunch with 1, 2, 4, 8 and 16 boxes and fitting the measured worst-case
relative error against the model sum; the shipped constant is the MV1
calibration (run `pr24-20261102`, jureca-h100, CUDA 13.4).  `--velocity-bins`
runs a fixed count (calibration/diagnostics), `--bin-model-constant` overrides
`C`, and `--max-velocity-bins` (default 256) caps the auto policy; an
infeasible requirement fails naming the needed count.

## Padding

The free-space convolution needs zero padding. With `fftPadding = 1` the FFT
box equals the output box, so source-observer lags beyond half the box wrap and
add a spurious periodic image; the unit test reproduces this artefact. With
`fftPadding >= 2` every lag fits inside the box and the FFT result equals the
free-space discrete TSC sum (no images). `fftPadding = 1` is therefore a
diagnostic/test option only, never a production choice.

The unit test now evaluates a direct, non-cyclic sum with its own TSC shape and
its own Coulomb field at probe nodes where the periodic image is closest. The
padded result matches that reference to better than `1e-10` relative; the
unpadded result disagrees by more than `1e-3`. This is what the test proves, not
merely that two padded runs agree with each other.

The wraparound artefact is geometry- and box-dependent; quote it only together
with the geometry. Measured values (rosi_v100, CUDA/V100, `-j 8`):

- toy geometry (27 charges on a 3x3x3 lattice spanning about 20 cells,
  `marginCells = 6`, `cellSize = 1 m`): `max|pad1-pad2| / max|field| =
  3.994542e-02`, interior (two nodes skipped per axis) `2.130691e-02`.
- a single stationary source on a `20^3` box: `9.275e-3` (worst component Ey);
  the exact geometry and per-component split are in the N1b artifact.
- for the toy geometry, `padding=2` versus `padding=3` agrees to `8.37e-16`, i.e.
  padding 2 is converged to roundoff.
- for a 500-particle, `cellSize = 1e-6 m` cloud on an `n = (18,18,18)` box,
  `padding=2` versus `padding=3` agrees to `4.15e-16`; the code policy uses the
  conservative bound `1e-12` so that this number does not need to be re-measured
  for every geometry.

## Known limitation of v0.1

The single-particle physics test matches an independent high-order free-space
reference at `1.6e-7` in the far field but only at `1.4e-2` for the near-source
point P2, asserted at `2e-2`. The cause is the coarse `kRefinement = 1` TSC
assignment quadrature near the source, not an FFT error. v0.1 exposes no
refinement knob, so this near-source accuracy is a stated limitation, not a
tuning target.

## Sanitizers

`-DPICBUNCH_SANITIZE=thread` or `-DPICBUNCH_SANITIZE=address` instruments only
picBunch's own targets; the fetched alpaka/alpakaVendor targets stay
uninstrumented. Use the CPU backend. Exact ThreadSanitizer reproduction:

```bash
cmake -S . -B build/tsan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
      -DPICBUNCH_SANITIZE=thread -DopenPMD_DIR=/opt/openpmd-hdf5-0.17.1/lib/cmake/openPMD
cmake --build build/tsan -j 8
OMP_NUM_THREADS=1 TSAN_OPTIONS=halt_on_error=1 \
      ./build/tsan/tests/picBunchUnitTests --gtest_filter=Physics.*
```

`OMP_NUM_THREADS=1` avoids the unrelated alpaka host-queue/OmpBlocks races that
also exist in the baseline; they are out of scope here and are not claimed fixed.
Without it, the deposit path must report no `heap-use-after-free`. The same
build recipe with `-DPICBUNCH_SANITIZE=address` gives AddressSanitizer.

## HIP validation on hal

The HIP backend was built and validated on `hal`, a **shared** host (AMD Radeon
RX 7900 XTX, `gfx1100`, ROCm 7.2.0, rocFFT). Because other jobs may run on the
same host, no timing or throughput numbers are claimed from this run; only
correctness and warning counts are recorded.

Build invocation (CMake 3.28, no manual workarounds; see README for why
`amdclang++` instead of the `hipcc` wrapper and why C is enabled):

```bash
cmake -S . -B build/hip -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DPICBUNCH_ENABLE_HIP=ON \
      -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
      -DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++ \
      -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
      -DCMAKE_PREFIX_PATH=/opt/rocm \
      -DopenPMD_DIR=/opt/openpmd/0.17.0/lib/cmake/openPMD
cmake --build build/hip -j 8
ctest --test-dir build/hip --output-on-failure
```

Results on that host:

- The original HIP validation passed 8/8 CTest entries registered there (the
  h5py-only consumer checks were not registered). Re-running this configuration
  with the h5py checks available passes 10/10.
- The build reports zero compiler warnings.
- Moving bunch at `gamma ~= 10`, worst relative error per field component:
  `Ex = 1.0e-3`, `Ey = 1.6e-4`, `Ez = 8.1e-5` at the near test point; the far
  test point is at most `1.6e-7`.
- R2C versus C2C transform agreement: `7.1e-15`.

These numbers are properties of the code on this device; they are not a
performance claim about `hal`.

## Open items

- The plain mesh remains the default; values below its crossovers are a stated
  limitation. `--near-shell exact` is an opt-in correction that replaces the
  mesh contribution of sources within a shell by the exact TSC-convolved field
  (vendored FP64 reference quadrature), using a per-tile local zero-padded FFT
  for the `F_mesh(near)` subtraction. It is not lookup-class and its measured
  accuracy/cost envelope is reported with the change that added it. `--near-shell
  fast` is the accurate backend for near-source points (README Accuracy): it
  validates against the order-12 `indfield` arbiter to `1.6%` at the closest
  moving in-support node. `exact` is kept for comparison only; it shares a
  quadrature with the driver reference and overestimates moving oblique
  in-support points by about `10x` at the closest node.
- The FFT padding policy uses a conservative measured bound rather than a
  runtime error estimate, so it never pays for a second transform. Production
  cost/error measurements are recorded above.
- A refinement policy that improves the near-source accuracy is deferred; v0.1
  is fixed at `kRefinement = 1`.
- File I/O is implemented: the writer is covered by the I/O round-trip test and
  the reader by the CLI, the physics test, and the reader-default test. The v0.1
  reader emits a `std::cerr` warning when it applies the `-e` charge default or
  the `weightingPower = 0` default for `macroWeighted = true`.
