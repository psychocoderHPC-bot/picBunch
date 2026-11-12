# Paper figure pipeline

Renders the paper figures from reduced data committed under
`docs/paper/data/`. Every path is resolved from this directory, so rendering
runs from a fresh clone without environment variables. The measurement ingest
is the exception: it requires a campaign root through `--measurement-root` or
`PICBUNCH_MEASUREMENT_ROOT`.

## Shared field style

All field figures (`gp_*`, `val_*`, `large_*`) use one geometry and one font
ladder from `figstyle.py`. The canvas is the full text width
(`6.2677 in`, the measured `\textwidth`), with four data panels per row
(three signed axis slices plus the squared magnitude), ONE shared colour bar
per row, and fixed inch margins. `field_layout` solves the figure height from
the row count so every data panel is the same physical square: `1.0945 in`
(`27.80 mm`) wide and high, aspect ratio `1.0000`, independent of the row and
column count. The density-row maps use the same constant. Only the figure
height changes with the row count.

The font ladder is fixed: axis labels `9 pt`, panel titles and figure notes
`8 pt`, tick labels and legends `7.5 pt`, in-panel annotations `7 pt`. Every
value is at or below the `10 pt` body text. Every raster panel, in every
family, uses the one `imshow` policy: `interpolation='nearest'`,
`resample=False`, and an extent built from the pixel centres plus/minus half a
cell. A data pixel is a visible block with exact shared edges; nothing is
smoothed and no per-cell anti-aliased edge can leave a white seam at 300 dpi.
The full rule set is in `STYLE.md`, and `check_figstyle_overrides.py` fails if
a renderer sets style locally.

## Axis convention

The propagation axis is a property of each data set, not a single global
choice. `axis.py` records it in `PROPAGATION_AXIS` and records the physical
axis of every stored array in `ARRAY_AXES`. A renderer calls
`axis.to_display`, which builds a permutation from the named axes. A missing
entry raises `BlockedAxisError`; the code does not guess an orientation.

The evidence is the magnetostatic structure `B = (v/c^2) x E`: the B component
along the propagation axis is identically zero and the longitudinal E
component is suppressed. Measured in the reduced data:

| data set | signature | propagation axis |
|---|---|---|
| BDATA2 forward | `By == 0`, Ey suppressed | +y |
| HIRES g1p0-g100 | `By == 0`, Ey suppressed | +y |
| VAR1b 1 mrad | +y beam | +y |
| LARGE1 g10 | `Bx == 0`, By and Bz nonzero, Ex/Ey=0.034 | **+x** |
| LARGE1 g1 | all B components zero, mean gamma 1 | **stationary** |

The +y families plot the beam axis y horizontally and x or z vertically.
LARGE1 is the axial-x data set and is plotted as such: the x-y cut puts the
beam axis x horizontally and y vertically; the z-y cut puts y horizontally and
z vertically because the beam axis x is out of that plane. `large_*` figures
must not be relabelled to +y and must keep the `Bx = 0` panel.

`ARRAY_AXES` also records that the LARGE1 z-y slice is stored as `[z, y]`,
unlike the BDATA2 and HIRES z-y slices, which are stored as `[y, z]`. The
LARGE1 x-y slice is stored as `[y, x]`. Only the LARGE1 and HIRES reduced npz
files carry the layout metadata (`array_axes_xy`, plus `array_axes_zy` for
LARGE1); the BDATA2 and VAR1b files have no such keys and rely on the explicit
`ARRAY_AXES` table plus `check_stored_shape` before every render. The layouts
come from the composers (`analyze_bdata2.py`, `hi_compose.py`,
`make_large_domain.py`).

`run_all.py --out-dir DIR` writes `DIR/axis-assertions.txt`. For each 2D
figure it records the propagation axis, the stored and display axes, the
horizontal coordinate range, the location of the maximum and the
field-weighted centroid, and PASS/FAIL. For LARGE1 g10 the renderer also
asserts the `Bx == 0`, By/Bz nonzero signature before drawing. The g1 case
asserts all B components are zero and checks labels only, since a stationary
bunch has no beam axis. A mapping self-test fails if a stored band is not
placed on the recorded horizontal axis.

The paper captions must state the axial-$x$ geometry and keep "$B_x = 0$"
for LARGE1. A future +y large-domain run is an
optional cross-orientation validation, not a relabel of these figures.

## Error-panel mask

The error rows of the slice figures (`val_*`, `large_*`) mask a pixel when the
local field magnitude of the plotted part is below a shared floor:
`sqrt(sum_c ref_c**2) >= 1e-3 * max_c(peak_c)` for E or B, where `peak_c` is
the largest `|ref_c|` on that plane. The three components share the floor, so
a component that is intrinsically near zero keeps its structured far field
instead of being cut into salt-and-pepper speckle. The error itself stays
normalised per component by that component's own peak. The reduced npz files
store this mask in their `*_valid_*` arrays; `ingest_measurements.py` writes
the same rule and `render_slices.py`/`render_large.py` recompute it from the
reference arrays. The HIRES single-electron `gp_*` figures keep their
per-component mask convention in `render_pergamma.py`.

Masked pixels are light grey (`#dddddd`), never white. The finite error
colormap (`magma`) is truncated at 0.90 of its range so the worst finite error
is orange and cannot be read as the mask. `render_split_field_rows` asserts
that no finite error value maps to the masked colour and that the colormap top
is neither white nor grey. Each figure carries a grey "masked" legend patch.

## Colour norms

All signed field triads in `val_*` and `large_*` use one shared symmetric-log
bar per row (`SymLogNorm`, `linthresh = 1e-3 * shared maximum`), the same
convention as `gp_*`. A shared linear bar maps a real but suppressed component
(for example the gamma=10 longitudinal `E_x` at about 0.5% of the dominant
component) onto the near-white zero centre and hides its structure. Each
component panel therefore also carries a compact peak label: `peak` on one
line and, on the next, the plane peak of that component (`max |arr_c|` over
the plane) divided by `cvmax`, the largest plane peak over the three
components of the row, which sets the shared colour scale (`vmin=-cvmax`,
`vmax=+cvmax`). The value is shown as a percentage to at most one decimal;
peaks below 1% are given to one significant figure (e.g. `.04%`), the panel
that sets the shared maximum reads `max`, and only an exactly zero peak reads
`0%`. The label is drawn at 7 pt and `render_split_field_rows` asserts that
the rendered text stays below 30% of its square panel width. A small percentage means the component is
faint on that scale. The triad keeps one shared bar; the annotation makes the
per-panel scale explicit without changing the layout.

The `E^2`/`B^2` panel in a field row uses `LogNorm` from `1e-6` of that
panel's peak to its peak, with decade ticks plus the peak and a `(log)` bar
label. Zero magnitude is the physical floor and is drawn with the colormap's
lowest colour, not the masked grey. Error rows use `LogNorm` from the fixed
`1e-6` floor to a decade ceiling, with a marked `1e-3` gate tick and the bar
label "relative error"; the error is dimensionless. Colour-bar tick labels sit
left of the bar and the rotated quantity label right of it, each in its own
reserved slot. The TSC-support contour is dark grey and solid so it stays
visible on both the grey mask and the finite colours.

## Layout guard

`figstyle.save` draws every figure and asserts that no text box (including
tick labels, axis labels and titles, colour-bar labels, in-panel annotations,
and figure text) intersects a non-owning axes rectangle or another text box.
Each record in `<outdir>/layout_assertions.json` tags every text as
`tick_label`, `axis_label`, `annotation`, or `figure_text` and stores the
minimum signed clearance in points (positive gap, negative overlap) between
axis text and foreign axes or foreign axis text. Any collision raises
`AssertionError` before the PDF is saved, so a new collision fails the render
instead of shipping.

## Scripts

| file | role | inputs | outputs |
|---|---|---|---|
| `figstyle.py` | canonical style module: `apply_paper_style`, palette, colormaps, raster policy, colour bars, field/error renderers, deterministic save, layout guard | none | PDF, PNG, `audit/*.json`, `layout_assertions.json` |
| `STYLE.md` | machine-checkable style contract and rationale | none | none |
| `check_figstyle_overrides.py` | fail if a renderer sets fonts, canvas, colormap, raster, mask or colour-bar style locally | `render_*.py`, `run_all.py` | report; exit code |
| `axis.py` | beam-axis convention, array-axis table, coordinate helpers, assertions, `self_test` | none | assertion dicts |
| `cases.py` | repo-relative paths and the case registry | `docs/paper/data/cases.json` | path helpers |
| `render_slices.py` | `val_*` BDATA2 forward/fluctuation and VAR1b 1 mrad slices | `data/bdata2/*.npz`, `data/var1b/*.npz` | 10 PDF/PNG |
| `render_large.py` | `large_*` LARGE1 128x128x64 slices | `data/large1/slices_g{1,10}.npz` | 6 PDF/PNG |
| `render_pergamma.py` | `gp_*` HIRES 128^3 single-electron slices | `data/hires/hi_*.npz` | 7 PDF/PNG |
 | `render_scaling.py` | `F_a`, `F_b`, `F_w4b_rel_error`, `F_gamma_err_dist`, `F_multivel_dist`, and `F_mpi_rank_scaling_h100` when its reduced table exists; the V100 `F_mpi_rank_scaling` stays callable but is not in the default set | `data/scaling/matrix.csv`, `data/line/*.csv`, `data/line/multivel_*.npz`, `data/line/h100_mpi_scaling.csv` | 5 PDF/PNG by default, plus the H100 figure once data lands |
| `run_all.py` | render every included figure and write the assertion log; `--strict` fails when a figure's reduced data is pending | all of the above | 28 PDF/PNG by default, 29 once the H100 table lands, 30 with `--include-unreferenced`, `axis-assertions.txt`, `render-summary.json`, `layout_assertions.json` |
| `check_provenance.py` | validate provenance records, verify SHA-256 digests, require every figure to be covered or declared not-reproduced; `--strict` requires fresh-h100 or a reference-only fresh-reference record for data figures | `provenance/records/*.json`, `provenance/provenance.schema.json`, `provenance/not_reproduced.json`, `provenance/non_data_figures.json`, `figs/*.pdf` | report; exit code |
| `make_provenance.py` | emit a schema-valid record for a case from hashed paths and host metadata | `data/cases.json` | `provenance/records/<case>.json` |
| `compose_mpi_scaling.py` | reduce per-rank H100 campaign CSVs into the single H100 MPI-scaling table | `mpi_g256_r{1,2,4}.csv` | `data/line/h100_mpi_scaling.csv` |
| `ingest_measurements.py` | ingest fresh campaign outputs into the reduced data and emit `fresh-h100` records; `--plan`/`--check` write nothing | `measurement_map.json`, fresh output root | reduced `data/**`, `provenance/records/*.json` |
| `measurement_map.json` | declarative source file/field -> reduced target mapping and per-case provenance metadata | none | none |
| `test_ingest_measurements.py` | synthetic proof of complete ingestion, strict gate, missing-source failure, and temp-only writes | temporary synthetic outputs | PASS/FAIL lines, nothing in the tree |
| `selftest_mpi_h100.py` | synthetic end-to-end smoke test for the composer and the H100 renderer | temporary synthetic CSVs only | PASS/FAIL lines; temp PDF, nothing in the tree |

Run:

```bash
python3 docs/paper/scripts/run_all.py --out-dir "$HOME/picbunch-staging"
```

Output goes to a staging directory. The committed `docs/paper/figs/` is
written by the final replot from fresh measurements, not by this pipeline.
Rendering is deterministic: Agg backend, fixed PDF and PNG metadata, no RNG.
Two consecutive runs produce byte-identical files.

`run_all.py` renders only figures the manuscript includes. The defaults drop
one previously generated orphan: the V100 `F_mpi_rank_scaling`. The V100
MPI-scaling figure is intentionally not part of the paper: the manuscript
includes only the H100 figure `F_mpi_rank_scaling_h100`, so there is no
shipped V100 PDF and no provenance record claiming one. Its renderer stays
callable in `render_scaling.py::mpi` and its reduced measurement data stays in
`data/line/mpi_scaling.csv` (source `docs/measurements/mpi_writer_scaling.csv`)
so the earlier run remains reproducible. Pass `--include-unreferenced` to
render the V100 orphan in a staging directory. No measurement data is deleted.
The field-evaluation pipeline figure is a TikZ drawing in
`docs/paper/sec_method.tex` (`\label{fig:pipeline}`), compiled inline, and is
therefore not rendered by this pipeline.

## H100 MPI scaling

The H100 figure reads one reduced table, `data/line/h100_mpi_scaling.csv`, with
one row per rank. The campaign writes one raw CSV per rank, for example
`mpi_g256_r1.csv`, `mpi_g256_r2.csv` and `mpi_g256_r4.csv`. Each raw row is one
repeat and must carry `repeat,total_s,fft_s,writer_s`; the optional metadata
columns are `host,gpu_model,grid_cells,particles` (command-line values fill
gaps). Lines starting with `#` are ignored.

The reduced table has these columns:

| column | meaning |
|---|---|
| `case_id` | `h100-mpi-scaling` |
| `host`, `gpu_model` | measured host and GPU model |
| `grid_cells`, `particles` | grid edge and particle count |
| `ranks`, `repeats` | rank count and number of repeats |
| `total_median_s`, `fft_median_s`, `writer_median_s` | median over repeats |
| `total_spread_s`, `fft_spread_s`, `writer_spread_s` | full range `max - min` |

Rows are sorted by `ranks`; numbers are written with six significant digits, so
the file is diff-stable without dropping measured digits. Produce the table and
then the figure with:

```bash
python3 docs/paper/scripts/compose_mpi_scaling.py \
    --raw-dir <campaign-log-dir> \
    --host jureca-h100 --gpu-model "NVIDIA H100 PCIe 80 GB" \
    --particles 2000000 --grid 256
python3 docs/paper/scripts/run_all.py --out-dir "$HOME/picbunch-staging"
```

Until the raw CSVs land, `cases.json` records the case as
`measurement_kind: not-reproduced` and `run_all.py` prints the missing figure
and case. The default run continues with the other figures; `--strict` (or
`--strict-provenance`) exits non-zero instead of treating the absent table as a
silent skip. The figure is never rendered from a synthetic or placeholder
table.

## Provenance gate

Every figure must be backed by a record under `docs/paper/provenance/records/`
(or explicitly declared not-reproduced), and every referenced data file is
checked against its recorded SHA-256. The final replot cannot succeed without
the gate:

```bash
python3 docs/paper/scripts/run_all.py --out-dir "$HOME/picbunch-staging" \
    --check-provenance --strict-provenance
```

`run_all.py --check-provenance` runs the same checker and returns non-zero on a
gap; `--strict-provenance` additionally demands `measurement_kind=fresh-h100`
for every data figure, or a `fresh-reference` record with `reference_only:
true` for a figure that plots a freshly recomputed analytic reference, and fails
if a figure's reduced table is still pending.
The gate can also be run on its own:

```bash
python3 docs/paper/scripts/check_provenance.py
python3 docs/paper/scripts/check_provenance.py --strict
```

New or replaced records are generated with `make_provenance.py`, never by
hand-editing JSON. See `docs/paper/provenance/SCHEMA.md` for the fields, the
null policy and the self-test.

## Ingesting fresh measurements

`ingest_measurements.py` is the mechanical block-6 step between a fresh campaign
and the replot. It reads the campaign outputs from `--measurement-root DIR` or,
when the flag is absent, from `PICBUNCH_MEASUREMENT_ROOT`. There is no built-in
absolute default; the script exits with an actionable error when neither is
set. It applies the declarative mapping in `measurement_map.json`, validates the
exact keys and column headers the renderers read, writes the reduced files
under `docs/paper/data`, and emits one `fresh-h100` record per ready case
through `make_provenance.py`.

Set the root once for the block-6 sequence:

```bash
export PICBUNCH_MEASUREMENT_ROOT=/path/to/campaign-output
```

`measurement_map.json` states, for every renderer input, the campaign case id,
the source file and fields, and the transform:

| family | targets | source / transform |
|---|---|---|
| `val_forward` | `bdata2/forward_slice.npz`, `bdata2/forward_sources.npz` | `campaign/fields/bdata2-forward.npz`; slice extraction, axis check, error/valid derivation |
| `val_fluct` | `bdata2/fluctuation_slice.npz` | `campaign/fields/bdata2-fluctuation.npz`; same transform |
| `val_1mrad` | `var1b/slices_var1b.npz` | `valid3/out/1mrad_fig.h5`; `field_h5` xy slice, error/valid, sources from the order-12 reference |
| `gp_g*` | the seven `hires/hi_*.npz` | `campaign/fields/hires-*.npz`; xy slice, `E2`/`B2`, unit conversion to um (no fresh run yet) |
| `large_s*` | `large1/slices_g1.npz`, `slices_g10.npz` | `valid3/out/large_g{1,10}.h5` + `large_g{1,10}_128.bin`; `field_h5` xy/zy slice, SI scaling, stored-axis check |
| `F_a`, `F_b` | `scaling/matrix.csv` | `docs/measurements/matrix.csv`; column copy, drops the classification columns |
| `F_w4b` | `line/w4b_rel_error.csv` | `w4b/w4b_paper_metric.csv`; column copy with the corrected paper metric |
| `F_gamma` | `line/gamma_err_dist.csv` | `reduced/line/gamma_err_dist.csv`; column copy |
| `F_multivel` | `line/multivel_{energy,angular}.npz`, `line/multivel_binning.csv`, `line/multivel_peaks.csv`, `line/multivel_fields_*.npz` | `reduced/multivel/*`; particle/column copy, peak scalars and field arrays |
 | `F_mpi_rank_scaling_h100` | `line/h100_mpi_scaling.csv` | `r9/raw/newstack/mpi_medians.json`; median/spread reduction |


The `field_h5` transform reads an openPMD HDF5 field output directly
(`data/0/meshes/{E,B}/{x,y,z}`, shape `(nz,ny,nx)`) plus either a raw
six-component driver `.bin` (scaled to SI at the case cell size) or an npz of
2-D reference arrays. It slices by physical axis and validates the stored
layout with `axis.expected_stored_shape`, so the reduced file matches the
renderer contract without an intermediate export. `reference.zero_below`
zeros only reference values far below any physical field when the analytic
component is identically zero (the stationary LARGE1 gamma=1 magnetic field);
the map records the threshold.

A missing source leaves that case's legacy files untouched and reports the case
as pending. A source whose recorded `sha256` differs from the mapping aborts
the run before any write. A case is written atomically after every one of its
targets has been built and validated, so a half-ingested case cannot land.

The block-6 sequence is:

```bash
# 1. show readiness without writing
python3 docs/paper/scripts/ingest_measurements.py --plan

# 2. validate every source, checksum and renderer key; non-zero if pending
python3 docs/paper/scripts/ingest_measurements.py --check

# 3. write the ready reduced data and the fresh-h100 records
python3 docs/paper/scripts/ingest_measurements.py

# 4. render, then run the strict provenance gate
python3 docs/paper/scripts/run_all.py --out-dir "$HOME/picbunch-staging" \
    --check-provenance --strict-provenance

# 5. FIGURE_AUDIT: inspect "$HOME/picbunch-staging/axis-assertions.txt",
#    "$HOME/picbunch-staging/render-summary.json" and the staged PDFs/PNGs.
#    The audit is a read-only check; it writes no figure.

# 6. commit the reduced data and records together
git add docs/paper/data docs/paper/provenance/records
git commit
```

`ingest_measurements.py` refuses to guess a missing key. Its key and column
contracts are copied from the renderer reads, so a renamed array fails loudly
before `run_all.py` sees it. `test_ingest_measurements.py` proves the three
required properties with synthetic inputs in temporary directories only; it
asserts that `docs/paper/data` and `docs/paper/provenance/records` are
byte-identical before and after.

## Data

`docs/paper/data/` holds the reduced inputs, about 15 MB compressed. The files
keep only the arrays the renderers read. Dropped material includes the HIRES
z-y planes and peak/worst scalars (the `gp_*` figures use the x-y plane), the
VAR1b z-y plane, BDATA2 `E2`/`B2` arrays and peak scalars, LARGE1 object tables
and peak scalars, and every unused CSV column. No array value was changed.

`scaling/matrix.csv` is the reduced copy of
`docs/measurements/matrix.csv` (the 2026-11-08 H100 re-measurement behind
`tab:matrix`). The original columns are byte-identical to the source; the
reduced copy appends the per-row gate classification the `F_a`/`F_b` renderer
plots from: `row_class` (`quotable`/`bimodal`), `total_iqr_ms` (IQR of the
kept repeats), `mode_total_low_ms`/`mode_total_high_ms` (the two mode totals
of the bimodal rows, sum of the per-stage cluster medians),
`mode_total_iqr_ms` (IQR of the per-mode stage cluster medians, i.e. the
half-spread the mode points are plotted with), `mode_low_n`/`mode_high_n`
(mode counts), `n_used` (kept repeats), and `bimodal_stage` (the stage(s)
that split). The ingest maps the case to the in-repo source with
`root: repo` and pins both the repo file and the canonical campaign
artifact (`canonical_sha256`), dropping the classification columns on the
way so the reduced file round-trips to the source rows.

`cases.json` is the machine-readable registry. Each case records
`provenance: legacy-reduced`, its `propagation_axis`, its configuration, host,
picBunch version, seed, source data files with SHA-256, and the figures it
feeds. LARGE1 is recorded as `x` for g10 and `stationary` for g1 by design.
These are reduced copies of earlier runs, not fresh measurements.

The legacy `rebuild_line.py` carried the MPI and gamma-error values as Python
literals. They now live in `data/line/mpi_scaling.csv` and
`data/line/gamma_err_dist.csv`; the multivel panel (c) values live in
`data/line/multivel_binning.csv`. The `F_gamma_err_dist`, `F_w4b_rel_error`
and `F_multivel_dist` families are now fresh-h100 data from the 2026-11-09
jureca-h100 campaign: gamma uses the current-model h=1/4 rows, W4b uses the
corrected paper metric for all four gammas, and multivel uses the fresh
-100 pC field arrays and binning plus the peak-scalar table.

`F_mpi_rank_scaling_h100.pdf`, which the current paper includes, is now a
`fresh-h100` measurement: the reduced table `data/line/h100_mpi_scaling.csv`
is committed from the BP5 rank-scaling campaign (1 warm-up + 3 repeats,
medians), reduced from `r9/raw/newstack/mpi_medians.json` on the final ADIOS2
2.12.1 stack. `cases.json` and
`provenance/records/h100-mpi-scaling.json` both record the case as
`fresh-h100`; the strict provenance gate accepts it.
