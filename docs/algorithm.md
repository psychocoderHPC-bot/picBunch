# Algorithm

This document names the pipeline stages and records what each stage must prove.
Numbers are deliberately absent until they are measured in validation.

## Stages

1. **Reader** - open an openPMD particle series and load one species: positions,
   dimensionless momenta, weights, charge. Position convention is the openPMD
   standard `x = position * unitSI + positionOffset * unitSI_offset`. Momentum is
   converted with `u = p * unitSI / (m c)`. TODO: numbers from validation
   (species size, units round-trip error).

2. **Deposit** - TSC assignment of `weight * charge` onto the uniform output grid.
   The deposit must conserve total charge and must match the device deposit node
   for node. TODO: numbers from validation (charge conservation error, deposit
   versus reference).

3. **Kernel** - evaluate the free-space field of the bunch as a convolution of the
   deposited source with the shape-weighted, uniformly moving point-charge kernel,
   including the near-field quadrature where the point kernel is unresolved.
   TODO: numbers from validation (relative field error per band, moving-bunch
   gamma cases).

4. **FFT** - transform the zero-padded source and kernel, multiply in Fourier
   space, and transform back. The FFT is reached through the alpakaVendor portable
   API (cuFFT, rocFFT, or FFTW depending on the build). TODO: numbers from
   validation (plan build time, transform throughput, padding factor).

5. **Readout** - sample the six field components at their Yee-staggered positions,
   interpolate from the padded FFT grid to the output grid, apply units so E is in
   V/m and B is in T. TODO: numbers from validation (staggering check, divergence
   and curl residuals).

6. **Writer** - write `E` and `B` as openPMD mesh records with components
   `x,y,z`, stored C-order with axis labels `z,y,x` (x fastest), `geometry =
   cartesian`, `gridSpacing` in metres, `gridUnitSI = 1.0`, `gridGlobalOffset`
   from the grid origin, `timeOffset = 0`, and a configurable iteration number.
   Each component carries its Yee staggering through the `position` attribute:
   `Ex(.5,0,0) Ey(0,.5,0) Ez(0,0,.5) Bx(0,.5,.5) By(.5,0,.5) Bz(.5,.5,0)`.
   Values are written in SI (V/m for E, T for B) with `unitSI = 1.0`, and the
   `unitDimension` exponents are E = (1,1,-3,-1) and B = (0,1,-2,-1).

   **PIConGPU restart defect:** PIConGPU restart currently ignores the mesh
   `unitSI` attribute (verified defect), so a PIConGPU consumer of these files
   must convert with `sim.unit.eField()` / `sim.unit.bField()`. The writer does
   not silently switch to internal PIConGPU units; it writes SI and says so
   here.

## Padding

The free-space convolution needs zero padding. With `fftPadding = 1` the FFT
box equals the output box, so source-observer lags beyond half the box wrap and
add a spurious periodic image; the unit test reproduces this artefact. With
`fftPadding >= 2` every lag fits inside the box and the FFT result equals the
free-space discrete TSC sum (no images). `fftPadding = 1` is therefore a
diagnostic/test option only, never a production choice.

## Open items

- No accuracy gate has been run for this standalone tool yet.
- The near-field band policy and the FFT padding factor still need a measured
  trade-off between cost and error.
- The writer is implemented and covered by the I/O round-trip test; the reader
  is covered by the CLI and the physics test.
