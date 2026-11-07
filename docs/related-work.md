# Related implementation check: lennertsprenger notebooks

Reviewed branches:

- `lennertsprenger/PIConGPU-BunchInit`, branch `lienard-wiechert`
  (`8ee0900dd61279cb9b1165e9832be50bc7880ed9`): `bunchInit.ipynb` and
  `analyseCheckpoint.ipynb`.
- `lennertsprenger/PIConGPU-BunchInit`, branch `master`
  (`961c3bbc99efaf8105a7d1ad1f7e09a6c7836a31`): the same notebooks without the
  Liénard-Wiechert field branch.
- `lennertsprenger/picongpu`, branch `poisson_solver`
  (`c6795e29e836c8f69d6bd51f01ba25131edffb36`): the commit changes
  `PoissonSolverImpl.tpp` and `Stencil.kernel` to run a PIC Poisson solve.

What the notebooks do
- `FieldParallel` assigns one field node per CUDA thread and loops over all
  particles; `particleParallel` does the opposite.  Both evaluate the
  Liénard-Wiechert field of a point charge at every field node, including a
  retarded-time root solve (`param1`), and atomic-add the six components.
- The field is computed on a chunk of the output grid (`chunk_size`), so GPU
  memory is bounded by a field chunk, not by the particles.
- The shape functions (CIC, TSC, PQS) appear in the charge-density test, not in
  the point-field loop.
- The Poisson branch is an in-simulation Poisson solve for the PIC grid; it is
  not a free-space open-boundary field initializer.

Why none of them is a drop-in improvement
- Cost: the direct field loop is O(N_field * N_particle).  picBunch uses an
  O(N + M log M) FFT convolution plus a direct near-shell correction only
  inside the TSC support.  For 10^6-10^8 particles and large output grids the
  direct loop is orders of magnitude slower.
- Retardation: picBunch is for a uniformly moving bunch.  For uniform motion
  the present-time velocity field equals the Liénard-Wiechert velocity field;
  the retarded-time root solve adds cost and conditioning without changing the
  physics.  Radiation/acceleration is outside the scope.
- Shape: the notebooks do not convolve the source shape into the field.  A
  direct point sum avoids TSC smoothing but is exactly the expensive path the
  near shell replaces.
- Field chunking: the notebook chunks the output grid.  picBunch already
  decomposes the output into z-slabs (distributed FFT) and tiles the near shell;
  no new mechanism is needed.
- Poisson solve: the PIC Poisson solver assumes the simulation boundary
  conditions; picBunch targets the free-space field, where the zero-padded FFT
  convolution is the appropriate open-boundary operator.

Shape-function check
A direct 3D quadrature of the stationary point-charge field smoothed by CIC,
TSC and PQS at 2, 4 and 8 cells gives the following relative smoothing error
(negative means the smoothed field is below the point field):

| shape | d=2 | d=4 | d=8 |
| --- | ---: | ---: | ---: |
| CIC | -8.31e-3 | -5.62e-4 | -3.55e-5 |
| TSC | -1.20e-2 | -8.41e-4 | -5.33e-5 |
| PQS | -1.55e-2 | -1.12e-3 | -7.10e-5 |

PQS is not more accurate than TSC for the field-smoothing metric; it has a wider
support and a larger smoothing bias at these distances.  CIC is also worse than
TSC here.  TSC remains the shipped shape.

Conclusion
The notebooks are useful as an independent, direct Liénard-Wiechert reference
and as a demonstration of field chunking, but they do not offer a performance
or accuracy improvement over the FFT + near-shell design.  No code was ported.
