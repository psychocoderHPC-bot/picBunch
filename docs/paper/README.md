# picBunch paper

Sources and rendered PDF for the picBunch method, cost, and accuracy paper.
The manuscript reports the shipped v0.5.0 state (tag `v0.5.0`, commit
`7b4f17661e5be355e4d19c1ab296b83905d1b8c4`). Individual measurements name the
commit they used; the measured field path is unchanged through v0.5.0.

The cost section contains one explicitly marked unfinished measurement
(`[TODO --- UNFINISHED MEASUREMENT]` in `sec_accuracy.tex`) for the final
300 s verdict. It is a placeholder, not a result.

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
