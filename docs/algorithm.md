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
    rejected instead of being silently rescaled. Measured units round-trip
    error: the openPMD writer/reader round-trip
    (`Io.FieldRoundTripPreservesValuesAndMetadata`, 4x5x6 grid) is
    bit-identical for every value and for all metadata, and the momentum
    convention round-trip test asserts exact agreement for a weight-100
    species (see README "Precision and accuracy safeguards").

2. **Deposit** - TSC assignment of `weight * charge` onto the uniform output grid.
    The deposit must conserve total charge and must match the device deposit node
    for node. Measured: the TSC deposit of a single particle of weight 1 and
    charge 1 conserves total charge to 1e-14
    (`Deposit.SingleParticleWeightOne`), and the padded host deposit
    conserves charge to 1e-12 per refinement value 1..3
    (`Deposit.PaddedDepositConservesChargeForRefinement`) and to 1e-14 at
    every boundary node of the 27-node corner/edge/face set
    (`Deposit.PaddedSourceHaloConservesChargeAtBoundaries`), where the pre-fix
    boundary deposit was 0.669921875 instead of 1.0
    (`docs/measurements/precision_improvements.csv`).

3. **Kernel** - evaluate the free-space field of the bunch as a convolution of the
   deposited source with the shape-weighted, uniformly moving point-charge kernel.
   There is no separate near-field quadrature: the smoothing comes from the
   deposit-convolved-with-kernel composition, which is O(s^2) in the assignment
   shape width. The `D = 0` guard in the analytic point-field kernel never fires
   for the six Yee components, because each component's half-cell staggering
   keeps |R| > 0 at every grid node. Near the source the remaining error is the
    coarseness of the v0.1 TSC assignment at `kRefinement = 1`; see the known
    limitation below. Measured: per-cell-distance relative error bands are in
    `docs/measurements/w4b_rel_error.csv` (stationary case), and the moving
    gamma-10 cases are recorded in the single-particle physics test
    (`Physics.*`): `1.6e-7` far field, `1.4e-2` near-source point P2 (asserted
    at `2e-2`), as listed in the "Known limitation of v0.1" section below.

4. **FFT** - transform the zero-padded source and kernel, multiply in Fourier
    space, and transform back. The FFT is reached through the alpakaVendor portable
    API (cuFFT, rocFFT, or FFTW depending on the build). Measured: the plan
    build and a delta/round-trip self-test pass
    (`smoke.cpp`, `Fft.RoundTripKnownArray` asserts round-trip error < 1e-12);
    the padding factor is fixed by the tolerance policy table above.
    Plan build time and transform throughput are not measured; see the cost
    table in the paper for the measured per-row FFT stage times
    (`docs/measurements/matrix.csv`).

5. **Readout** - sample the six field components at their Yee-staggered positions,
    interpolate from the padded FFT grid to the output grid, apply units so E is in
    V/m and B is in T. Measured: the Yee staggering is verified
    bit-exactly (`Io.FieldRoundTripPreservesValuesAndMetadata`); the discrete
    Gauss-law residual on the Yee lattice is reported per case in
    `Audit.DiscreteGaussLawOnYeeLattice` (no roundoff gate; the residual is a
    discretisation residual, not an FFT or padding artefact, as the
    padding-2 vs padding-3 equality proves).

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

`--tolerance` sizes only the FFT zero padding. It is a relative target for the
delivered *discrete free-space TSC convolution*, i.e. the discrete sum the
pipeline is designed to evaluate. It is not an end-to-end accuracy target, and
it is not a target for the continuum Coulomb field near the source; the
difference there comes from the fixed TSC deposit at cell resolution
(`kRefinement = 1`) and no tolerance value can reduce it.

For a single-velocity bunch the target controls exactly one implementation
choice: the FFT zero-padding factor. It does not change the interpolation order
(fixed at TSC), the deposit scheme, or the node spacing, and it
is not an end-to-end accuracy guarantee. The README error-budget table lists the
mesh, binning, near-quadrature, far-tail and output-precision terms separately. The
mapping is a measured error-vs-padding model:

| requested tolerance | padding | basis |
| --- | --- | --- |
| `>= 1e-12` | 2 | measured padding-2 self-convergence is at most `8.37e-16` (toy) / `4.15e-16` (500-particle cloud), below the conservative bound `1e-12` |
| `1e-14 .. 1e-12` | 3 | tighter than the padding-2 bound, so the next padding is used |
| `< 1e-14` | rejected (non-zero exit) | below the measured double-precision floor; refusing is better than silently under-delivering |

Padding 1 is never selected by the policy. It is a diagnostic override used by
the padding test only.

For a multi-velocity bunch the same target additionally sizes the velocity
boxes (see "Velocity binning"), or `--velocity-moments` replaces the boxes with
a moment expansion: the box count is the smallest power of two whose calibrated
error model stays below the target, and an infeasible requirement fails with
the needed count and the error at the cap instead of running. The model is an
estimate, not a tolerance guarantee.

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
`gamma * Delta_theta`, and an energy shift.  With `|q_b|` the absolute charge in
box `b`, `Q` the total absolute charge, `gamma_b` the largest `gamma` in the
box, `Delta_theta_b` the largest member angle against the representative (taken
as `pi` when the representative is zero and any member is moving, so a
counterstreaming box cannot hide its error), and `Delta_gamma_b` the `gamma`
range of the box, the estimated relative error is

    modelError = C * Sum_b (|q_b| / Q) * (gamma_b * Delta_theta_b
                                         + (2 / gamma_b) * Delta_gamma_b / gamma_b)

Absolute charges keep opposite signs from cancelling a local error. The angular
term and `C = 1` are unchanged from the original calibration. The single
node-bounded energy term `(2/gamma_b) * Delta_gamma_b/gamma_b` replaces the
former `Delta_gamma_b/gamma_b + delta_gamma_b` pair, which charged the same
node effect twice and over-predicted the archived 5% energy-spread points by
`26.7x` and `25.9x`.

The energy term is calibrated on the archived energy-spread points only: gamma
10, 5% energy spread, 16 and 64 boxes. It gives `3.9e-3` at 16 boxes and `9.2e-4`
at 64 against measured `1.12e-3` and `2.72e-4`, a `3.4x` conservative
over-prediction. It selected 64 boxes at `--tolerance 1e-3` instead of failing
above the 256-box cap. The angular term is not refitted: `C = 1` reproduces the
archived 5 degree point, and the 5 degree angular-spread case measures `0.421`
at 16 and `0.295` at 64 and does not reach `1e-3`. No guarantee is made beyond
gamma 10, 5% energy spread and the measured 16-64 box range. The stored points
are in `docs/measurements/velocity_bin_calibration.csv`;
`tools/velocity-bins/calibrate_velocity_bins.py` reproduces the record from the
CSV.

`--velocity-bins` runs a fixed count (calibration/diagnostics),
`--bin-model-constant` overrides `C`, and `--max-velocity-bins` (default 256)
caps the auto policy. When the cap is too small the run exits non-zero and
prints the required count and the error at the cap as separate numbers, for
example
`the velocity-bin model requires at least 8388608 boxes for tolerance 0.001000
(estimated error at 8388608 boxes: ...; at the 256-box cap: 1.867e-01)`.

One exception keeps cold, nearly monoenergetic beams usable. Under the default
cap the auto policy uses one box when the bunch is essentially single-velocity:
the charge-weighted relative spread of `|u|` is at most `2%` and every particle
is within `50 mrad` of the charge-weighted mean direction (the same
representative the one-box path uses). The `gamma * Delta_theta` worst case can
ask for millions of boxes for such a beam even though one velocity class is the
physical model. The run prints one line with the measured spread and divergence
and reports the one-box `modelError`. That estimate may exceed `--tolerance`;
when it does the run proceeds with a warning rather than claiming the target is
met, because the binning model is conservative for a cold relativistic beam (the
operator's gamma-1101, 4.2 mrad bunch has a one-box estimate of `1.33e+01` and
still runs). An explicit `--velocity-bins N` always wins, and a non-default
`--max-velocity-bins` disables the fallback so an explicit infeasible cap still
fails with the message above. A broad distribution, including the 5% energy
spread and 5 degree angular calibration cases, does not meet the criterion and
keeps the historical split-or-fail behaviour.

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
  fast` keeps the base order and ray and routes only the observer-containing TSC
  support cell to the exact s-space evaluator; the remaining in-support cells keep
  the order-5 (or gamma-derived) remainder quadrature. `--near-shell-exact-cells`
  exposes the routing: `-2` automatic (default), `-1` legacy only, `0` observer
  cell, `1` observer plus the six face-adjacent cells, `2` every in-support cell.
  A tolerance of `1e-4` or below resolves `-2` to full support, and a tile with
  gamma `>= 100` is always full exact because its contracted tube is not
  resolvable on the legacy quadrature. On the 52-point set the default gives
  `3.71e-4` stationary, `1.40e-4` moving axial gamma 10 and `1.94e-3` moving
  oblique interior. `exact` is kept for comparison only; it shares a quadrature
  with the driver reference and overestimates moving oblique in-support points by
  about `10x` at the closest node. The fast path now rejects an explicit tile
  whose half-width exceeds the shell radius (edge above `2 * near_shell_radius`)
  and an explicit tile above the measured maximum of `4` nodes; `tile = 0` keeps
  the automatic rule. After the correction it also aborts when the corrected peak
  E exceeds the uncorrected peak by more than `3x`, naming the radius, tile
  and fast order. The boundary treatment that produces the measured tile-8 and
  radius-6 spikes is under investigation; these are guards, not the numerical fix.
- The FFT padding policy uses a conservative measured bound rather than a
  runtime error estimate, so it never pays for a second transform. Production
  cost/error measurements are recorded above.
- A refinement policy that improves the near-source accuracy is deferred; v0.1
  is fixed at `kRefinement = 1`.
- File I/O is implemented: the writer is covered by the I/O round-trip test and
  the reader by the CLI, the physics test, and the reader-default test. The v0.1
  reader emits a `std::cerr` warning when it applies the `-e` charge default or
  the `weightingPower = 0` default for `macroWeighted = true`.
