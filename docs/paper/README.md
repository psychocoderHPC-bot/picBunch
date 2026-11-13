# picBunch paper

Sources and rendered PDF of the picBunch method, accuracy, and cost paper,
reporting the review-fix `v0.9.2` state (after tag `v0.9.1`,
commit `c0bf82c2251448abfaad66aa03b18ab3a9ab9322`). Individual measurements name
the artifact they used.

## Layout

`main.tex` plus the main-text sections (`sec_*.tex`), the Methods section
(`sec_methods.tex`), and the Appendices (`app_*.tex`); `refs.bib` is the
local bibliography; `figs/` holds the rendered figure PDFs.
`paper.pdf` is the rendered result.

Main text: title, abstract, Introduction, Physical model and field
definition, Numerical algorithm, Physical consistency, Validation and
accuracy, Cross-validation and checkpoint transport, Performance and
scaling, Discussion, Conclusion. Methods records the configurations,
commands, and data/code availability. Appendices carry the derivations,
the seven-velocity sweep, the mid-bunch and large-domain slices, the
multi-velocity binning measurements, the padding details, and the far-tail
switch validation.

## Build

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode main.tex
```

The build needs a LaTeX distribution with `natbib`, `siunitx`, `pgfplots`,
`longtable`, and `hyperref`. `refs.bib` is local, so no external
bibliography database is required. After the sources change, copy
`main.pdf` over `paper.pdf`.

## Figures

Every figure PDF in `figs/` is rendered from the recorded measurement
data at its print width (A4 text block, 6.2677 in) so that the 9 pt fonts
print at 9 pt. The shared style module fixes the canvas width to the
LaTeX inclusion width, uses Okabe-Ito hues for the gamma encoding, one
colour bar per row with the quantity and SI unit (E in V/m, B in T, a
dimensionless relative error), and one font ladder from 10 pt body text down
to 7 pt annotations and 7.5 pt ticks. The smallest span placed in
`paper.pdf` is 6.3 pt (a 7 pt mathtext script at the 0.9 script scale).
The audit
(`audit_figures.py` in the rework directory, output in `audit/`) records
for every included figure the printed minimum font size, the colour bars
per row, the colour-bar labels, and any overlapping text boxes; all
figures pass. `figstyle.save` enforces the same text/axes and text/text
clearance at render time and writes `layout_assertions.json`; a collision
aborts the render. The `gp_g*` per-gamma figures are rendered from the
`HIRES/hi_*.npz` slices; the `val_*` and `large_*` slice figures and the
line/scatter figures are rendered from their recorded npz/CSV inputs
via the `FIGS_REPLOT` rebuild scripts.
