# picBunch paper

Sources and rendered PDF for the picBunch method, cost, and accuracy paper.
The manuscript reports the shipped v0.9.1 state (tag `v0.9.1`, commit
`c0bf82c2251448abfaad66aa03b18ab3a9ab9322`). Individual measurements name the
commit they used.

## Build

From this directory:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The build needs a LaTeX distribution with `natbib`, `siunitx`, `pgfplots`,
and `hyperref`. `refs.bib` is local, so no external bibliography database is
required. The figure PDFs in `figs/` are the rendered inputs; the scripting
that produced them lives outside this documentation tree.

`paper.pdf` is the rendered result. Rebuild it with the commands above and copy
`main.pdf` over `paper.pdf` when the sources change.

## Multi-velocity distribution figure

`figs/F_multivel_dist.pdf` is the sampled velocity distribution behind
`tab:multivel` and is included as `fig:multivel` in `sec_accuracy.tex`. It is
rendered through the shared `figstyle` module at the paper inclusion width.
