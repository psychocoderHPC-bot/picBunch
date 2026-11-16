# Specification inconsistencies found during implementation

This note records the places in `nextTests.md` (the acceptance contract for the
velocity-binning validation) that are internally inconsistent or physically
impossible as written, and what this implementation does about each one. It
contains no measured accuracy or performance results; the numbers below are
either arithmetic or taken from the generator's analytic moment records. The
measured results and pass/fail verdicts live in a separate report.

## 1. Total particle count contradicts the per-case counts

Spec reference: Section 3 states "total particles across all bunches stay under
5000"; Sections 3.1 to 3.6 list per-case counts of 1000, 1000, 2000, 2000, 2000
and 1000, which sum to 9000.

Conflict: the total of the per-case counts exceeds the stated total by 80%.
Both statements cannot hold.

Consequence for validation: none of the criteria in Section 7 depends on the
total, but a reader who checks the generated bunches against the total sees a
violation.

Implementation: the per-case counts are reproduced exactly, because the counts
and seeds are what make the cases reproducible and acceptance-relevant. The
generator's dry-run output prints "total particles: 9000", making the deviation
visible in every run. The total is stated as 9000 in the measurement report.

## 2. Velocity distributions put a large fraction of draws above c

Spec reference: Section 3.3 specifies the halo as a Gaussian in v with mean
0.98 c and spread 0.03 c; Section 3.5 specifies a log-normal in v with median
0.97 c and log-standard-deviation 0.05.

Conflict: an untruncated Gaussian or log-normal in v is a density over the
whole real line. With these parameters, a sizable share of the draws lands
above c, where no physical electron velocity exists and the reduced momentum
u = v / sqrt(1 - v^2) would be imaginary. Approximate, from the untruncated
laws: about 25% of the halo draws and about 27% of the log-normal draws
(exact values for the fixed seeds are seed-dependent and are not recorded in
the metadata). Section 3.4's exponential tail is the only case that is
already truncated (at v_max = 0.999 c).

Consequence for validation: the requested means are not the means of the
implementable distributions. A generator that kept the full draws could not be
run at all; a generator that silently changed the parameters would no longer
match the spec.

Implementation: the generator truncates the Gaussian and log-normal draws to
0 < v < c by rejection sampling and records the realised moments per group in
the per-case metadata (`<case>.meta.json`, key `measured`), alongside the
analytic moments of the truncated law (key `analytic_after_truncation`) as a
cross-check. The realised group means from the fixed seeds are 0.9682 c for
the core_halo halo and 0.9490 c for the skewed log-normal, against the
requested 0.98 c and 0.97 c (log-normal median). The analytic
post-truncation values, 0.9672 c and 0.9488 c, agree with the realised means
to within the sampling error of 500 and 2000 draws respectively. This is a
deviation from the written parameters, caused by the physical bound v < c,
and it is reported here rather than resolved silently.

## 3. Gamma regime of the specified bunches vs. the model's calibration

Spec reference: Sections 3.1 and 3.6 specify p0 = 1e9 MeV/c (Section 3.1 adds
dp = 1e8 MeV/c); Section 3 defines v = pc / sqrt(m0^2 c^2 + p^2).

Conflict, arithmetic: for an electron (m0 c^2 = 0.51099895 MeV), p0 = 1e9 MeV/c
gives u = p/(m0 c) = 1.957e9 and gamma = sqrt(1 + u^2) = 1.957e9. In IEEE
double precision, beta = u / gamma then evaluates to exactly 1.0 (1 - beta = 0;
the exact gap is about 1.3e-19, below machine epsilon). The velocity carries no
information in the code's own units, and the generator's metadata records
|v|/c = 1.000000 with zero spread for both cases, with the resolvable spread
carried in the |u| statistics instead. Section 3.1's claim that the velocity
spread is "small relative to c, so the distribution of v is effectively
uniform" is therefore a statement about |u|, not about v.

Conflict, calibration: the shipped content model was calibrated at two ends of
the gamma axis. The angular term (C_ang * Delta_theta^2) was calibrated on the
W37 campaign at gamma 500 (50 to 150 mrad divergence, 1 to 16 boxes), and the
energy term ((2/gamma_b) * Delta_gamma_b / gamma_b) on the archived 5%
energy-spread points at gamma 10, 16 and 64 boxes
(`docs/measurements/velocity_bin_calibration.csv`). The validation cases sit in
between: Sections 3.2 to 3.5 are at gamma 3.2 to 7.1 (0.95 c to 0.99 c), and
Sections 3.1 and 3.6 are at gamma 1.957e9.

Consequence for validation: the proxy is evaluated outside its calibrated
gamma range for all six cases. The Section 9 rule (record both the proxy
prediction and the measured error when they disagree by more than a factor of
2) is the only built-in defence against a miscalibrated model, and it is to be
expected to fire more often than in the calibration regime.

Implementation: the cases are generated exactly as specified; the deviation
from the calibration range is documented here and flagged in the measurement
report. The generator's metadata records the |u| statistics for the two
high-gamma cases, since |v|/c is unresolvable there.

## 4. "In-bunch" region: the inherited statistic is a no-restriction

Spec reference: Section 6 asks for the in-bunch maximum relative error inside
the "bunch bounding box" and a far-field error in "a box of the same size
centered on the bunch but offset by 2 R".

Conflict: the pre-existing comparison code
(`tools/velocity-bins/compare_velocity_bins.py`) restricted its statistics to
nodes within 16 cells of a particle. On the small domains this spec implies
(a few dozen cells per dimension, a 2R = 2 um bunch), every node of the domain
is within 16 cells of some particle, so the inherited "in-bunch" notion was
not a restriction at all; it silently meant "the whole grid".

Consequence for validation: a metric computed over the whole grid includes
field values far from the bunch, where the discretisation error and the binning
error have different magnitudes, so the number cannot be compared against the
spec's intended region.

Implementation: the new metrics (the `metrics_v2` block) define the in-bunch
mask as the inclusive node box of the particle position min/max, and the
far-field mask as that same box translated by +2R along +x, where R is half
the largest bunch extent. When the domain cannot hold the translated box with
the required 2-cell margins, the comparison fails loudly with the required
domain size instead of clipping silently (a `clip` policy exists for
diagnostics and records the clipping in the output).

## 5. Reference field: the pre-existing 1-D script used a point-charge sum

Spec reference: Section 5 requires the cloud-convolved direct sum as the
reference and forbids a point-charge sum, whose near-node field diverges while
the deposition field is finite.

Conflict: the pre-existing 1-D validation script
(`tools/velocity-bins/validate_adaptive_1d.py`) computes its reference as a
per-particle sum over the Lienard-Wiechert point-charge Green's function,
exactly the model Section 5 excludes. Its recorded numbers compare the binned
field against a singularity the real deposition does not have.

Consequence for validation: numbers recorded by that script are not admissible
as evidence under Section 5, even though they were useful during development
of the edge rule in 1D.

Implementation: the matrix uses the TSC cloud-convolved continuum reference
(`indfield.py` via `compare_velocity_bins.py`), which smears each
macroparticle with the same shape function the deposition uses. The 1-D
script's numbers are superseded by this reference for the acceptance
evaluation; the script remains in the tree for the 1-D structural check it
was built for.

## 6. Bunch file format in the run example

Spec reference: the Section 8 example invokes the measurement tool with
`--bunch .../uniform.json`.

Conflict: the measurement tool (`measure_velocity_bins.cpp`) reads a plain-text
column file with the layout `x y z ux uy uz weight charge` (one header line,
`#` lines skipped), not JSON. There is no JSON bunch reader.

Consequence for validation: the example command as written cannot be run.

Implementation: the generator writes a text bunch file per case
(`<case>.bunch`) plus a separate per-case metadata JSON
(`<case>.meta.json`) with the requested parameters, the analytic
post-truncation moments and the measured statistics. The metadata file is
documentation and verification data, not input to the tool.

## 7. Startup cost units

Spec reference: Section 6 asks for the startup cost of the edge selection in
microseconds.

Conflict: the measurement tool reports the derivation cost in milliseconds, as
`bin_derive_ms` in the per-run meta file. The driver that assembles the matrix
(`tools/velocity-bins/run_matrix.py`) converts it to microseconds by
multiplication by 1000 when building the manifest.

Consequence for validation: a value copied from the meta file into the report
without conversion would be off by a factor of 1000 (a 0.43 ms reading is
430 us, not 0.43 us).

Implementation: the report reads the converted value from the manifest and
states the conversion explicitly, `startup_us = bin_derive_ms * 1000`.

## 8. K per axis or in total

Spec reference: Section 4 runs each bunch at fixed bin counts K = 2, 4, 8 and
Section 6 reports "how many of the K boxes contain at least one particle".
Section 10 names the ambiguity explicitly: the rule may partition velocity
space independently per axis (a tensor-product grid) or as full boxes in 3D.

Conflict: Section 4 does not say whether K is the number of boxes per axis or
the total number of boxes. Section 6's "the K boxes" only identifies a set of
boxes if K is the total; under a per-axis reading the total would be a
function of K and "non-empty of the K boxes" would have no referent.

Consequence for validation: the box counts in the report are only
interpretable once the reading is fixed.

Implementation: the code treats K as the total box count and distributes it
across the active velocity axes with `chooseSplit`
(`src/VelocityBinning.cpp`), so the product of the per-axis counts equals K.
A `--bins 4` run records `per_axis 2 2 1` and `bins_used 4`; a beam-like bunch
records `per_axis 1 1 4`. The meta file carries both the per-axis split
(`per_axis`) and the total (`bins_used`), and the report reads the box counts
as totals, comparing the non-empty box count against `bins_used`.

## 9. Default `auto` vs. the gate in Section 9

Spec reference: Section 4 states that the default edge rule is `auto` and that
it applies also when the user passes an explicit bin count; Section 9 forbids
changing the default rule or the default bin count until the proxy is shown to
be adequate on all six bunches.

Conflict: the pre-adaptive code default was `uniform`, so "the default is
auto" is itself a default change, and Section 9 holds that change back until
the proxy is validated on the six spec bunches. The two sections constrain the
same quantity in opposite directions.

Consequence for validation: any run that omits `--velocity-bin-edges` uses the
new default, so "compare against older results" runs must pass `uniform`
explicitly, as Section 4 already notes.

Implementation: the shipped default is `auto`, as Section 4 requires. The
Section 9 gate is applied to the default bin count (still 0, i.e. automatic,
unchanged) and to any further rule change; the measurement report evaluates
the Section 9 factor-of-2 check per case and records the proxy prediction and
the measured error where they disagree.

## 10. "Materially worse" in criterion 1

Spec reference: Section 7, criterion 1 says the automatic rule is "never
materially worse than uniform edges", with "materially" meaning the in-bunch
maximum relative error is "not larger by more than 5 percent".

Conflict: it is not stated whether 5 percent is relative (the automatic error
may be at most 1.05 times the uniform error) or absolute (the difference may be
at most 0.05 in absolute relative-error units). For a maximum relative error
that can be of order 1, the two readings differ by an order of magnitude.

Consequence for validation: the verdict on criterion 1 depends on the reading.

Implementation: the summarizer
(`tools/velocity-bins/summarize_matrix.py`) evaluates the relative reading
(automatic error <= 1.05 x uniform error at equal K) and states that reading in
the report, so the other reading cannot be applied silently.

## 11. What this means for the acceptance criteria

Most of the criteria in Section 7 are expressed as in-bunch maximum relative
error. Two methodological cautions apply to how such numbers should be read,
independently of the measured values.

At gamma around 2e9 (cases 3.1 and 3.6), beta is indistinguishable from 1 in
the code's units (Section 3 of this note). The in-bunch field of a
near-light-speed beam is set by the per-particle gamma and the longitudinal
structure, which the transverse velocity binning does not resolve, and the
mesh discretisation of the near-field contribution is large on a 32^3 grid.
An in-bunch maximum relative error for these two cases may therefore be
dominated by discretisation rather than by the binning rule, in which case it
measures the mesh and the metric, not the edge placement.

The far-field metric is less affected: far from the bunch the field is smooth
on the grid, the discretisation contribution is small, and the error
attributable to the velocity representative is comparatively cleaner. The
lower-gamma cases 3.2 to 3.5 (gamma 3.2 to 7.1) sit between the two
calibration ends of the model, but their in-bunch metric remains informative
for comparing uniform, adaptive and auto at equal K, since all three
configurations share the same mesh and the same reference. The criteria should
therefore be read with the region and the gamma in mind: a far-field or
lower-gamma number that separates the edge rules is direct evidence about the
rule; an in-bunch maximum at gamma 2e9 is evidence about the rule only after
the discretisation floor is accounted for. This is a caution about
interpretation, not a conclusion about any of the six cases.
