# Internal Role Stability in Voynichese

This repository contains the publication package for the paper.

The paper studies internal role stability in Voynichese. It does not claim decipherment, lexical translation, formal grammar, or semantics. The main contribution is a reproducible role layer based on morphology aware units, multiple transcription support, confound controls, independent induction support, and tokenization robustness checks.

## Current result

The promoted reliable role layer covers 1,903 of 7,107 token occurrences, or 26.78 percent of the analyzed core. It appears in 812 of 1,709 lines, or 47.51 percent of the analyzed lines.

The descriptive good but not reliable enough tier adds 272 token occurrences. It is included for transparency and future work, but it is not treated as promoted evidence. With this descriptive tier included, coverage reaches 2,175 token occurrences and 858 lines.

## Reliability tiers

- Global reliable
- Conditioned reliable
- Environment sensitive reliable
- Tokenization robust reliable
- Good but not reliable enough
- Low reliability or unassigned

## Tokenization robustness

The final analysis does not assume that visible space tokenization is the only possible unit boundary. It checks whether candidate role units remain supported under surface token, morphology, broad shape, Stolfi inspired signature, edge, and boundary merge variants. A new unit is promoted only when it has both independent induction support and tokenization support.

## Key files

- `paper/main.tex`
- `paper/references_manual.tex`
- `scripts/run_role_stability_pipeline.py`
- `scripts/make_tables_figures.py`
- `data/processed/tokenization_robust_reliable_catalog.csv`
- `data/processed/reliability_coverage_summary.csv`
- `data/processed/occurrence_reliability_assignments.csv`

## Reproduce

```bash
python scripts/run_role_stability_pipeline.py
python scripts/make_tables_figures.py
cd paper
latexmk -pdf main.tex
```
