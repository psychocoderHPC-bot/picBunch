Generated file. Regenerate from the renderers' audit data and from PyMuPDF
measurements of `docs/paper/paper.pdf` and the native PDFs in
`docs/paper/figs/`; do not hand-edit the table.

## Contract

- **One raster size: 27.80 mm.** `RASTER_PANEL_MM = BUNCH_MAP_MM = 27.80` in
  `docs/paper/scripts/figstyle.py`. Every field/error panel and the density-row
  maps are the same square; the measured panel edge in the PDFs is 78.72 pt
  (27.77 mm), against the declared 27.80 mm.
- **Four panels plus one shared colour bar per field/error row.** A row holds
  the three signed components and the squared magnitude (or the three error
  components and the squared error) and one shared colour bar at the row's
  right edge. The density row is three coordinate-plane maps plus one
  marginal-profile axis, also on one shared bar and at the same 27.80 mm map
  size.
- Field figures are included at `\linewidth` (6.2677 in, 159.2 mm) with an
  exact canvas, so `scale = 1.000`. Line figures use `bbox_inches="tight"` and
  are included at a fraction of `\linewidth`; their native page width is
  smaller than the placed width, so `scale > 1` and the placed text is larger
  than the native text.

## Method

`placed_in = fraction x 6.2677 in`; `scale = placed_in / pdf_in`;
`min_pt = smallest native span x scale`. `pdf_in` is the page width of the
native figure PDF, and `min_pt` was cross-checked against the placed span in
`docs/paper/paper.pdf`. Figure numbers follow the `\includegraphics` order in
`docs/paper/*.tex`, with `\ContinuedFloat` grouping. No renderer audit JSON or
`layout_assertions.json` is committed in this tree, so the panel and colour-bar
counts are derived from the PDF geometry (square raster rectangles per row and
the thin shared-bar images), and the overlap count is recorded as
`not recorded` rather than inferred. `3+prof` is a density row of three
coordinate-plane maps plus one marginal-profile axis.

PASS = measured `min_pt >= 6.0` and, for a field/error row, four panels and one
shared colour bar.

Count: 23 unique figure PDFs populate the 17 PDF figure numbers (Fig. 2-18);
Fig. 1 is the TikZ pipeline diagram. Smallest placed span: 6.30 pt (field
family). Largest canvas width: 6.268 in (159.2 mm). Overlapping text pairs:
not recorded.

| fig | file | pdf_in | placed_in | scale | min_pt | panels/row | cbars/row | overlaps | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 2 | F_w4b_rel_error.pdf | 5.516 | 5.954 | 1.079 | 7.29 | n/a | n/a | not recorded | PASS |
| 3 | F_gamma_err_dist.pdf | 3.586 | 3.886 | 1.084 | 7.32 | n/a | n/a | not recorded | PASS |
| 4 | F_a_scaling_vs_N.pdf | 3.610 | 3.886 | 1.076 | 7.27 | n/a | n/a | not recorded | PASS |
| 4 | F_b_scaling_vs_box.pdf | 3.636 | 3.886 | 1.069 | 7.22 | n/a | n/a | not recorded | PASS |
| 5 | F_mpi_rank_scaling_h100.pdf | 3.279 | 3.635 | 1.109 | 8.32 | n/a | n/a | not recorded | PASS |
| 6 | gp_g1p0.pdf | 6.268 | 6.268 | 1.000 | 6.30 | 4,4,4,4 | 1,1,1,1 | not recorded | PASS |
| 7 | gp_g1p1.pdf | 6.268 | 6.268 | 1.000 | 6.30 | 4,4,4,4 | 1,1,1,1 | not recorded | PASS |
| 8 | gp_g1p5.pdf | 6.268 | 6.268 | 1.000 | 6.30 | 4,4,4,4 | 1,1,1,1 | not recorded | PASS |
| 9 | gp_g2p0.pdf | 6.268 | 6.268 | 1.000 | 6.30 | 4,4,4,4 | 1,1,1,1 | not recorded | PASS |
| 10 | gp_g10.pdf | 6.268 | 6.268 | 1.000 | 6.30 | 4,4,4,4 | 1,1,1,1 | not recorded | PASS |
| 11 | gp_g50.pdf | 6.268 | 6.268 | 1.000 | 6.30 | 4,4,4,4 | 1,1,1,1 | not recorded | PASS |
| 12 | gp_g100.pdf | 6.268 | 6.268 | 1.000 | 6.30 | 4,4,4,4 | 1,1,1,1 | not recorded | PASS |
| 13 | val_forward_xy.pdf | 6.268 | 6.268 | 1.000 | 6.30 | 3+prof,4,4,4,4 | 1,1,1,1,1 | not recorded | PASS |
| 13 | val_forward_zy.pdf | 6.268 | 6.268 | 1.000 | 6.30 | 3+prof,4,4,4,4 | 1,1,1,1,1 | not recorded | PASS |
| 14 | val_fluct_xy.pdf | 6.268 | 6.268 | 1.000 | 6.75 | 3+prof,4,4 | 1,1,1 | not recorded | PASS |
| 14 | val_fluct_zy.pdf | 6.268 | 6.268 | 1.000 | 6.75 | 3+prof,4,4 | 1,1,1 | not recorded | PASS |
| 15 | val_1mrad_xy_E.pdf | 6.268 | 6.268 | 1.000 | 6.75 | 3+prof,4,4 | 1,1,1 | not recorded | PASS |
| 15 | val_1mrad_xy_B.pdf | 6.268 | 6.268 | 1.000 | 6.75 | 3+prof,4,4 | 1,1,1 | not recorded | PASS |
| 16 | large_sxy_g10.pdf | 6.268 | 6.268 | 1.000 | 6.30 | 3+prof,4,4,4,4 | 1,1,1,1,1 | not recorded | PASS |
| 16 | large_szy_g10.pdf | 6.268 | 6.268 | 1.000 | 6.30 | 3+prof,4,4,4,4 | 1,1,1,1,1 | not recorded | PASS |
| 17 | large_sxy_g1.pdf | 6.268 | 6.268 | 1.000 | 6.30 | 3+prof,4,4,4,4 | 1,1,1,1,1 | not recorded | PASS |
| 17 | large_szy_g1.pdf | 6.268 | 6.268 | 1.000 | 6.30 | 3+prof,4,4,4,4 | 1,1,1,1,1 | not recorded | PASS |
| 18 | F_multivel_dist.pdf | 6.268 | 6.268 | 1.000 | 6.75 | 3+prof (hist) | 1,1 | not recorded | PASS |
