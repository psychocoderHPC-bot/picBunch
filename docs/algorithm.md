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

6. **Writer** - write `E_x, E_y, E_z, B_x, B_y, B_z` as openPMD scalar mesh records
   with axis labels, grid spacing, global offset, grid unit SI, time offset, time
   unit SI, and the `position` attribute describing each component's staggering.
   TODO: numbers from validation (reader/writer round-trip error, file size).

## Open items

- No accuracy gate has been run for this standalone tool yet.
- The near-field band policy and the FFT padding factor still need a measured
  trade-off between cost and error.
- The reader and writer stages currently only expose the openPMD build information;
  the file I/O itself is not implemented.
