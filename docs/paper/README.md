# picBunch paper

Sources and rendered PDF for the picBunch method, cost, and accuracy paper.
The manuscript reports the shipped v0.9.0 state (tag `v0.9.0`, commit
`970fc52f8fb5b150f2d3f777b4c9d6e99da0fc6b`). Individual measurements name the
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
