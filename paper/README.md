# Paper

LaTeX source for *Strategic Analysis of Card-Based Liar's Poker: Combinatorial Foundations and Nash Equilibrium Approximation* (Connor M. Lockhart, University of Maryland).

## Build

```bash
cd paper/
latexmk -pdf Liars-poker.tex
```

`latexmk` resolves the bibliography (biber/biblatex against `references.bib`) and runs `pdflatex` enough times to settle cross-references. Output is `Liars-poker.pdf`.

## Files

- `Liars-poker.tex` — main document.
- `references.bib` — bibliography in biber/biblatex format.
- `figures/` — pre-rendered figures and data caches. The paper does **not** use `\input{}` for any LaTeX table snippets; every figure and table is rendered to PDF by a Python script and included via `\includegraphics`.
- `figures/*.json` — data caches that the figure-generation scripts in `src/training/probs/` read on rerun (so Monte Carlo isn't recomputed unless caches are stale).
- `Liars-poker.{aux,log,out,bbl,bcf,fls,fdb_latexmk,run.xml,synctex.gz,blg,pdf}` — `latexmk` build artifacts. Regenerated on build; safe to delete.
- `Liar's-poker.*` — orphan build artifacts from a prior filename (apostrophe variant). Kept for git history only; will be cleaned in P6.

## Regenerating figures

Figure scripts moved to `src/training/probs/` in P1. To regenerate the figures (and their JSON caches):

```bash
python -m training.probs.generate_prob_tables
python -m training.probs.compute_conditional_probs
```

After regeneration, rebuild the PDF with `latexmk -pdf Liars-poker.tex`.
