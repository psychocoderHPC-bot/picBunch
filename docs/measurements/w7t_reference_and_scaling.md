# W7t - Q1 reference-quantity identification, Q2 particle-count scaling

Run `pr24-20261102`, task W7t. Host `rosi_v100` (exclusive, 48 cores; CPU runs,
`env -u LD_PRELOAD -u FAKEROOTKEY`, `-j 8`; GPU 0 left idle). Read-only on repos,
no GitHub writes. Remote workdir `/home/ubuntu/pr24/w7s-20261102/`; mirror of
this file and the scripts/logs in the external `pr24-20261102` run mirror.

The previous attempt (W7s) left only a RESULTS.md stub and partial scripts; its
`q2sub.py` sub-packing was wrong (treated the reference `.bin` as interleaved
particle records). The `.bin` is component-major (SoA). Q2 below uses a corrected
packer and the already-generated `refg1_N.npz` references, re-verified here.

## Q1 - What the exact-quadrature reference computes

**Answer: the TSC-convolved field, not the point field and not a mixture. It is the
same quantity picBunch computes.**

The 318-sample files (`b1-20261102/reffield/gamma{1,10}_samples_64cube.npz`,
`values_si`) are produced by `exact_field_driver --samples`, which calls the same
header-only reference as PIConGPU `fieldInitializer=referenceOnly`:

- Driver: `tools/neural-bunch/reference/cpp/exact_field_driver.cpp:123`
  `nbr::shape_field(target, src.x, src.u, dx, dy, dz, settings, E, B);` and
  `:127` `acc += -src.w * dx * dx * value;` (sample loop `:114-131`).
- Reference kernel definition: `include/picongpu/particles/neuralBunch/ReferenceKernel.hpp:625`
  `inline void shape_field(...)`; point kernel `:142`, TSC shape `:131`.
- TSC integration (smooth/out-of-support path):
  `ReferenceKernel.hpp:281` `rule.weights.push_back(half * wn[i] * tsc_shape(node));`,
  and the 27-sub-box in-support integral weights
  `ReferenceKernel.hpp:428` / `:522` / `:596`
  `W = tsc_shape(s.x) * tsc_shape(s.y) * tsc_shape(s.z);`.
- Production device port of the same definition:
  `include/picongpu/particles/neuralBunch/ExactShapeField.hpp:589`
  `exactShapeFieldPoint(...)`; TSC factorization `:157 exactTscShape`, smooth
  quadrature `:222 exactSmoothIntegral` with weights at `:239`, in-support
  `:472 exactBoxIntegral`; the point kernel it convolves is
  `include/picongpu/particles/neuralBunch/PointField.hpp:80 pointFieldSlot`
  (`:88` calls `pointFieldDx`, defined in `Features.hpp:242`).

Definition of the reference quantity for one source:
`F_c(x_obs) = -w * dx^2 * Integral_{s in [-1.5,1.5]^3} TSC(s) * pointField_c(x_obs - x_src - s) ds`,
summed over sources. That is the field of a TSC-shaped (charge-convolved) moving
source, evaluated by error-controlled quadrature with no FFT/mesh. picBunch deposits
the same TSC shape and solves the same free-space convolution, so the two compute
the same reference quantity. The measured disagreement is therefore an
implementation/discretization difference (exact continuum quadrature vs picBunch's
spectral solver and near-source handling), not a point-vs-TSC quantity mismatch.

## Q2 - Disagreement vs particle count (gamma=1, 64x64x32, cell 1e-7 m)

Deterministic subsets = first N particles in id order (ids are sorted; verified).
Same 318 staggered sample points, origin (0,0,0), picBunch CPU build at
`experimental/mvp` 339ca7ad, `--tolerance 1e-3`. Reference = exact `shape_field`
values from `refg1_N.npz`; N=10 and N=100 references re-generated with the driver
and matched bit-for-bit. Error = `max|picBunch-ref| / max|ref|` per component over
all 318 points. `worst_pt` gives the component/location of the global worst point,
`dmin` its distance to the nearest particle in cells, `n_in` the number of sample
points inside the union of the particles' TSC support boxes. B is identically zero
in both for this stationary bunch.

| N | Ex (all pts) | Ey | Ez | Bx | By | Bz | worst_pt | dmin | n_in | Ex (outside) | Ey (outside) | Ez (outside) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.504e-03 | 4.459e-04 | 5.904e-04 | 0 | 0 | 0 | out/Ex | 5.69 | 0 | 2.504e-03 | 4.459e-04 | 5.904e-04 |
| 2 | 1.390e-03 | 2.620e-03 | 1.861e-03 | 0 | 0 | 0 | out/Ey | 3.86 | 0 | 1.390e-03 | 2.620e-03 | 1.861e-03 |
| 10 | 1.092e+00 | 1.020e+00 | 1.148e+00 | 0 | 0 | 0 | in/Ez | 0.42 | 21 | 9.851e-03 | 6.162e-04 | 1.441e-02 |
| 100 | 5.328e-01 | 6.659e-01 | 8.975e-01 | 0 | 0 | 0 | in/Ez | 0.42 | 79 | 3.112e-02 | 1.605e-02 | 1.245e-02 |
| 1000 | 2.041e-01 | 1.600e-01 | 2.513e-01 | 0 | 0 | 0 | in/Ez | 0.65 | 250 | 3.375e-04 | 1.513e-03 | 5.375e-04 |

**Verdict: the disagreement does not grow with N; it is not a superposition
(weighting/deposit/accumulation) error.** Outside the particle TSC support the
worst per-component error stays at the per-source floor ~1e-3 to 3e-2 across
N=1..1000. The O(1) values at N>=10 occur only because the global worst sample
point is then inside a particle's TSC support, within 0.42-0.65 cells of a
source; that is the known near-source in-support quadrature/grid regime, and its
relative error falls again as N grows (reference peak grows). The step from N=2
to N=10 is the first sample entering a source's TSC support, not accumulation
growth. A flat (non-growing) per-source floor is what a kernel/reference-type
difference produces; a charge-weighting or accumulation defect would grow with N.

Artifacts on rosi: `w7t_q2.py`, `w7t_q2_table.log`, `refg1_*.npz`, `pb_1_*.h5`,
`rb_1_*.h5`, `data_1_*.bin`.
