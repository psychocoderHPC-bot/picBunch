# Two-rank V100 replot preview

Preview rebuild of the picBunch paper with the two-rank V100 reduced data.
Base: commit `e640452` (PR #65 content, parent `b29f55e`).
Branch: `paper/2rank-v100-replot`. Not pushed.

This is a preview. Provenance completeness is **not** claimed: the committed
`docs/paper/provenance/records/*.json` still hash the H100 reduced files, so
`check_provenance.py` reports 12 SHA-256 mismatches. The records were not
regenerated because no fresh H100 campaign and no `fresh-h100` metadata exist
for this run.

## Data sources per figure

Replaced with the new two-rank V100 reduced files (copied from
`/tmp/paper-replot-data/reduced/`, `line/` and `multivel/` copies are
byte-identical for the shared files):

| figure(s) | reduced data | new source |
|---|---|---|
| `gp_g1p0`, `gp_g1p1`, `gp_g1p5`, `gp_g2p0`, `gp_g10`, `gp_g50`, `gp_g100` | `data/hires/hi_*.npz` | 2-rank V100 |
| `large_sxy_g1`, `large_sxy_g10`, `large_szy_g1`, `large_szy_g10` | `data/large1/slices_g1.npz`, `slices_g10.npz` | 2-rank V100 |
| `F_gamma_err_dist` | `data/line/gamma_err_dist.csv` | 2-rank V100 |
| `F_w4b_rel_error` | `data/line/w4b_rel_error.csv` | 2-rank V100 |
| `F_multivel_dist` | `data/line/multivel_energy.npz`, `multivel_angular.npz`, `multivel_binning.csv` | 2-rank V100 |

The two `multivel_*.npz` files were byte-identical to the committed copies, so
only `multivel_binning.csv` actually changed for that family.

## Cases still on H100 (committed files left unchanged)

No new two-rank reduced data exists for these, so the committed H100 files were
kept untouched and their figures were not re-rendered:

- `data/bdata2/*.npz` -> `val_forward_xy`, `val_forward_zy`, `val_fluct_xy`, `val_fluct_zy`.
- `data/var1b/slices_var1b.npz` -> `val_1mrad_xy_B`, `val_1mrad_xy_E`.
- `data/scaling/matrix.csv` -> `F_a_scaling_vs_N`, `F_b_scaling_vs_box`.
- `data/line/h100_mpi_scaling.csv` -> `F_mpi_rank_scaling_h100`.
- `data/line/multivel_fields_*.npz` (4 x ~10-12 MB) and
  `data/line/multivel_peaks.csv` (legacy): kept, not copied. Not a paper
  metric; not consumed by the regenerated figures.

## Build

From `/tmp/paper-replot/docs/paper`:

```bash
latexmk -pdf -interaction=nonstopmode main.tex
cp main.pdf paper.pdf
latexmk -c
```

Result: exit code 0, **49 pages** (same as the previous `paper.pdf`, file
2768135 B vs 2775153 B). No undefined references. One benign warning
("Text page 32 contains only floats"). Auxiliary outputs were removed; only
`paper.pdf` is committed.

Figures were rendered into a staging directory and only the 14 affected PDFs
were copied into `docs/paper/figs/`, so the unchanged scaling, bdata2, var1b
and H100-MPI figures were not regenerated.

## Changed files

26 files changed: 12 reduced-data files, 13 figure PDFs, `paper.pdf`.
`F_gamma_err_dist.pdf` is byte-identical and therefore not in the diff.

### Visibly changed figures

Pixel diff of old vs new render (300 dpi PNG, >0.05% differing pixels):

- `gp_g1p0` (2.84%), `gp_g2p0` (1.34%), `gp_g10` (0.81%),
  `gp_g50` (0.64%), `gp_g1p1` (0.57%), `gp_g100` (0.42%),
  `gp_g1p5` (0.06%): all seven hires figures change; g50 and g100 change over
  the widest region.
- `F_multivel_dist` (0.97%, lower panels): visible.

### Byte-changed but visually identical

- `large_sxy_g1`, `large_sxy_g10`, `large_szy_g1`, `large_szy_g10`: PDF bytes
  differ by a few bytes, pixels identical (bbox = none).
- `F_w4b_rel_error`: bytes differ, pixels identical; the CSV change is float
  noise, max relative change 6.0e-8.
- `F_gamma_err_dist`: new CSV, but the plotted d=2,8 points are unchanged; PDF
  byte-identical.

## Checks run

- `check_figstyle_overrides.py`: PASS (6 renderer sources).
- `render_large.py` axis assertions: 4/4 PASS.
- `latexmk` build: exit 0, 49 pages.

## Checks that did not run / limitations

- `run_all.py` was not used as-is: it would also re-render `F_a`, `F_b` and
  `F_mpi_rank_scaling_h100` (unchanged H100 data) and its optional
  `--check-provenance` fails here. Targeted renderers plus the LaTeX build were
  used instead.
- `check_provenance.py` exits 1 with 12 SHA-256 mismatches. Expected: records
  still describe the H100 inputs. Provenance was intentionally not updated.
- No fresh-H100 / two-rank provenance metadata, so no `fresh-h100` record.
- matplotlib was installed in an isolated venv `/tmp/paper-venv`
  (numpy 1.26.4, matplotlib 3.11.2) because the system/user numpy 2.5.5 is
  ABI-incompatible with the apt matplotlib build. The existing TeX install was
  used unchanged.
