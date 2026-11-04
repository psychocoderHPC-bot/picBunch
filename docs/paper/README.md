# picBunch paper

Sources and rendered PDF for the picBunch method, cost, and accuracy paper.
The manuscript reports the shipped v0.7.0 state (tag `v0.7.0`, commit
`a152c518107e3de382312efb712ed3ce4de69027`). Individual measurements name the
commit they used; the measured field path is unchanged through v0.7.0.

## Build

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode main.tex
```

The build needs a LaTeX distribution with `latexmk`, `natbib`, `siunitx`,
`pgfplots`, and `hyperref`. `refs.bib` is local, so no external bibliography
database is required. The figure PDFs in `figs/` are the rendered inputs; the
scripting that produced them lives outside this documentation tree.

`paper.pdf` is the rendered result. Rebuild it with the command above and copy
`main.pdf` over `paper.pdf` when the sources change.
