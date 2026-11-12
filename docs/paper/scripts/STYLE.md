# Paper figure style contract

`figstyle.py` is the single source of truth for every style choice in
`docs/paper/figs`. Renderers pass data, titles and geometry only. Style values
do not appear in a renderer, and `check_figstyle_overrides.py` fails the build
if one reappears.

This file is the audit checklist: each rule states the value, the reason, and
the canonical function that implements it.

## 1. Single entry point

`apply_paper_style(kind="line")` is the only rcParam writer.

| kind | used by | difference |
|---|---|---|
| `line` | `render_scaling.py` (`F_*` line/bar/hist figures) | raster rcParams are set but inert |
| `field` | `render_pergamma.py`, `render_slices.py`, `render_large.py` (`gp_*`, `val_*`, `large_*`) | identical ladder; raster policy applies |

Both kinds share every font, line width, tick, grid, colour and save setting.
The kind selects no visual value; it documents intent and keeps the raster
rcParams explicit for field output. `apply_style` and `apply_field_style` no
longer exist, so there is one ladder rather than two.

`save()` records `style_tokens()` in every `<outdir>/audit/<figure>.json`, so a
reviewer can diff two figures' resolved style without opening the PDF.

## 2. Canvas and aspect per panel count

| figure class | width | height | aspect |
|---|---|---|---|
| field family (`gp_*`, `val_*`, `large_*`) | `6.2677 in` (`width_in(1.0)`) | solved by `field_layout(nrows)` | each data panel square, `1.0000` |
| two-panel line figures | `width_in(0.58)` | `5.35 in` | plot-driven |
| single-panel line figures | `width_in(0.62 or 0.95)` | `3.35-4.15 in` | plot-driven |
| `F_multivel_dist` | `width_in(0.98)` | `4.95 in` | plot-driven |

`field_layout(nrows)` fixes four panels per row, two colour bars per row and
the inch margins; it returns `panel_width_in == panel_height_in` independent of
the row count. Only the canvas height grows with rows. Renderers must call
`figure()` or `subplots()`; neither takes a raw `figsize`.

The canvas width equals the LaTeX include width, so a 9 pt label in the PDF is
9 pt in print. No hidden shrink factor is allowed.

## 3. Raster policy (one policy for all raster panels)

All field rasters use `imshow`:

```python
ax.imshow(arr, origin="lower",
          extent=(xlo, xhi, ylo, yhi), aspect="auto",
          cmap=..., norm=..., interpolation="nearest", resample=False)
```

`draw_raster_panel(ax, x, y, arr, quantity, norm, cmap)` is the only raster
call. `x` and `y` are the pixel **centres**; the helper checks uniform spacing
and builds the extent as `centre +/- half a cell`, so the outer pixel edge
coincides with the physical cell edge and each pixel centre lands on the data
coordinate.

Why this policy:

- **Seam-free at 300 dpi.** `pcolormesh(shading="flat")` anti-aliases cell
  edges. At 300 dpi the anti-aliased edge can be lighter than either cell,
  producing the hairline white seams seen in the field family. Even with
  `edgecolors="none"` and `linewidth=0`, renderer/backend combinations can
  leave a partial seam. `imshow` draws one image and rasterises pixel blocks
  without per-cell edge geometry, so there is no per-cell seam to expose.
- **No interpolation artefacts.** `interpolation="nearest"` and
  `resample=False` copy the nearest source pixel; nothing is smoothed,
  sharpened or blended, and a one-cell feature stays one cell.
- **Correct pixel-centre alignment.** With the extent at cell edges, a data
  value at coordinate `x_i` occupies `[x_i - dx/2, x_i + dx/2]`. A contour or
  annotation placed at `x_i` sits at the pixel centre, not half a cell off.
- **One texture.** `gp_*`, `val_*` and `large_*` now share the same raster
  path, so the fig16+ family and the single-particle family cannot differ in
  texture.

Masked and invalid pixels use `nan`/masked arrays and the colormap's `set_bad`
colour; the renderer must not draw a face colour over the panel.

## 4. Colormaps per quantity class

`cmap_for(quantity)` returns a private copy with the bad colour set.

| quantity | colormap | reason |
|---|---|---|
| signed field component (`signed_field`) | `RdBu_r` | diverging, white at zero, sign is readable |
| squared magnitude (`squared_field`) | `viridis` | perceptually uniform, ordered, print-safe |
| relative error (`error`) | `magma` truncated at `0.90` | the pale tail would read as white/no data; truncation keeps the worst finite error distinguishable |
| charge density (`density`) | `cividis` | perceptually uniform and distinct from `viridis` so density and E^2 panels cannot be confused |
| 2D particle density (`hist`) | `viridis` | matches the squared-magnitude sequential class |

The masked / invalid / blank colour is exactly one grey: `MASKED_COLOR =
"#dddddd"`. `#cccccc`, `0.90`, `white` mask panels and any second grey are
forbidden. `BLANK_COLOR` is an alias, not a second value.

Squared-magnitude panels use `zero_floor=True`: zero is the physical log
floor, not missing data, so it takes the colormap's own lowest colour. True
masks in those panels still use the one grey.

## 5. Scale and floor rules

| quantity | norm | range |
|---|---|---|
| signed component triad | `SymLogNorm(base=10)` | `[-vmax, +vmax]`, `linthresh = 1e-3 * vmax` |
| squared magnitude, field rows | `LogNorm` | `peak * 1e-6 .. peak` (six decades) |
| relative error | `LogNorm` | `ERROR_FLOOR = 1e-6 .. max(decade ceiling, 1e-3)` |

`error_limits()` and `error_ticks()` are shared by every error panel. The
lower floor is `1e-6`, the acceptance gate is `1e-3`, and the gate is always a
tick and always drawn as a dash-dot contour (`GATE_STYLE`). A finite error
below the floor is clamped to the floor colour; it is not silently promoted to
a larger value. `sq_limits()` and `sq_ticks()` are the shared squared-magnitude
helpers.

A row shares one normalisation across its component triad and one normalisation
for its squared panel. The two bars per row are therefore always distinct.

## 6. Colour bars

`draw_colorbar(fig, mappable, cax=... | ax=..., ticks=..., ticklabels=...,
label=...)` is the only colour-bar call.

| property | value |
|---|---|
| position | field rows: one bar at the row right edge for the triad, one for E^2/B^2 |
| tick side | field `cax`: left into the reserved gap; attached `ax` bar: right |
| label side | right, rotated |
| tick length / pad | `2.0 pt` / `1.5 pt` |
| tick label size | `7.5 pt` |
| quantity label size / pad | `9 pt` / `3 pt` |
| outline | `0.6 pt` |
| attached bar pad / fraction / shrink | `0.02` / `0.05` / `0.96`, anchored top |

The side rule is geometric, not stylistic: a bar between panels must put its
ticks in the reserved gap, while an attached bar at the right edge must keep
them outside the donor axes. Both use the same fonts, outline and label
placement. A `LogNorm` bar without explicit tick labels is labelled in plain
scientific notation (`1e-10`) at the tick size, matching the field-family
bars; no mathtext exponent is drawn at a sub-size script.

## 7. Typography and axis tokens

| token | value |
|---|---|
| family | `DejaVu Sans`, mathtext `dejavusans` |
| body text | `10 pt` |
| axis label | `9 pt` |
| panel title / figure note / row label | `8 pt` |
| tick label / legend | `7.5 pt` |
| in-panel annotation | `7 pt` |
| line width | `0.8 pt` |
| tick direction | `out` |
| major tick length / width | `2.5 pt` / `0.6 pt` |
| tick label pad | `1.5 pt` |
| grid | off |

No figure text may exceed the 10 pt body text. Axis labels come from
`axis.axis_labels()`; a renderer must not hard-code a unit or a component
name. Tick locators are chosen per panel geometry, but tick font, length,
width and padding come from the ladder.

### Mathtext sub/superscripts

Matplotlib's default script scale is `0.7x`, which put the smallest scripted
tokens below the 6 pt legibility floor (a 7 pt blank-panel note in a 0.98x
figure printed at 4.8 pt, a 7.5 pt log-tick exponent in the 0.951x
`F_multivel` canvas at 5.0 pt). `figstyle.MATH_SCRIPT_SCALE = 0.9` raises the
private `matplotlib._mathtext.SHRINK_FACTOR` so the worst placed script is
`7.0 * 0.98 * 0.9 = 6.17 pt`. The non-script ladder sizes are unchanged, so
the canvas and height budget do not move. The factor is recorded in every
audit's `style_tokens()["math_script_scale"]`.

## 8. Panel letters

`panel_title(ax, "a", title)` writes `(a) title`, left aligned at
`PANEL_TITLE_PAD = 2.5 pt`. Every multi-panel line figure uses it. Field
figures identify panels by their row label and component title instead of a
letter; they must not add a second letter convention.

## 9. Save contract

`save(fig, outdir, name, manifest, tight=False)`:

| property | value |
|---|---|
| dpi | `300` for PNG; PDF is vector |
| bbox | `tight` for line figures; exact canvas for field figures |
| padding | `0.03 in` |
| transparent | `False` |
| PDF metadata date | fixed `2026-11-02T00:00:00+00:00` |
| PNG metadata date | fixed string |
| fonts | PDF/PS Type 42, SVG paths as text |
| audit | `<outdir>/audit/<name>.json` with geometry, font ladder, norms, floor, mask colour and `style_tokens()` |
| layout guard | `assert_no_layout_collisions()` fails on any text/axes overlap |

A field figure keeps its exact canvas so LaTeX includes it at 1:1 and every
family prints the same physical panel.

## 10. What the checker forbids in a renderer

`check_figstyle_overrides.py` (run it before every commit that touches a
renderer):

| rule | forbidden token |
|---|---|
| font size | `fontsize=` |
| tick label size | `labelsize=` |
| resolution | `dpi=` |
| colormap | `cmap=` |
| canvas | `figsize=` |
| raster filter | `interpolation=` |
| raster shading | `shading=` |
| mask grey | `#cccccc`, `#dddddd` |
| raw colour literal | any `#rrggbb` / `#rgb` value; use a palette token |
| local constants | `MASKED_*=`, `CMAP_*=`, `ERROR_FLOOR=`, `SQ_FLOOR_RATIO=` |
| raw rcParams | `rcParams[...] =`, `rcParams.update(` |
| raw panel background | `set_facecolor(` |
| direct raster | `.imshow(`, `.pcolormesh(` |
| legacy ladder | `apply_style(`, `apply_field_style(` |

The checker also asserts that `figstyle.py` exposes `apply_paper_style()` and
does not expose the two legacy ladders. It prints `file:line` for every hit and
exits `1` on any violation.

## 11. Allowed per-figure choices

Data content and geometry remain renderer decisions: axis limits, tick
locations, legend placement, marker and line semantics, annotation text and
panel layout. Those are not style overrides. The test is whether the choice
belongs to the shared visual language (font, colour, raster, bar, save) or to
the figure's data (limits, labels, annotations). The former lives in
`figstyle.py`; the latter stays in the renderer.

## 12. Open items for the panel and axis workers

- The charge-density projection panel must call `cmap_for("density")` and
  `draw_raster_panel`; it must not define a density colormap or mask grey.
- The fig16+ axis correction changes axis orientation only; it must keep
  `axis.axis_labels()` and the shared layout.
- The seam follow-up is now structurally solved by the `imshow` policy; any
  remaining seam work must stay inside `draw_raster_panel`.
- Final regeneration must use `run_all.py` so the audit JSON is refreshed for
  every figure.
