This repository contains the code, derived data, and paper source for “Internal Role Stability in Voynichese: Morphological Foundations and Multi-Transcription Controls.” The project studies recurring internal role patterns in Voynichese using morphology-aware grouping, multi-transcription support, tokenization robustness checks, and manuscript metadata controls. It does not propose a decipherment or translation.

# Internal Role Stability in Voynichese

Paper title: **Internal Role Stability in Voynichese: Morphological Foundations and Multi-Transcription Controls**

Repository: <https://github.com/ahaankallat/voynich-role-stability>

## Summary

This project studies whether written units in Voynichese keep stable internal roles after morphology-aware grouping, multiple transcription support, controls for manuscript metadata, independent induction checks, and tokenization robustness tests.

The final promoted reliable layer covers 1,903 of 7,107 token occurrences, or 26.8 percent of the analyzed core, and appears in 812 of 1,709 lines, or 47.5 percent of analyzed lines. A broader descriptive layer covers 2,175 token occurrences and 858 lines. The descriptive "good but not reliable enough" tier is retained for transparency and future work, but it is not treated as promoted evidence.

## Claims

The project claims that some Voynichese units can be described as internally stable role units under the stated operational tests. The evidence is distributional and reproducible from the derived project tables included here.

The project does not claim decipherment, plaintext recovery, lexical translation, phonetic values, word meanings, parts of speech, formal grammar, or semantics.

## Repository Structure

- `paper/`: LaTeX paper source, references, generated tables, and generated figures.
- `scripts/`: Reproducible analysis and table/figure generation scripts.
- `src/voynich_audit/`: Minimal Python package namespace for the project.
- `data/processed/`: Derived tables used by the analysis and paper.
- `data/README.md`: Data source and redistribution notes.
- `docs/`: Method and data documentation.
- `MANIFEST_SHA256.csv`: Checksums for the release contents.

## Reproduction

Install dependencies with Python 3.10 or newer:

```bash
python3 -m pip install -r requirements.txt
```

Run the main analysis and regenerate tables and figures:

```bash
python3 scripts/run_role_stability_pipeline.py
python3 scripts/make_tables_figures.py
```

The Makefile also provides focused targets:

```bash
make pass30
make pass31
make tables
```

## Compile the Paper

From the repository root:

```bash
make paper
```

The paper compiles from `paper/main.tex` using `paper/references_manual.tex`, `paper/tables/`, and `paper/figures/`.

## Validate Checksums

Validate the release manifest:

```bash
python3 scripts/validate_checksums.py
```

The manifest was regenerated for the final public repository contents after intentionally excluding compiled PDFs, LaTeX build products, temporary files, Python caches, and operating system metadata.

## Citation

Use the metadata in `CITATION.cff` when citing this repository. Suggested citation title:

> Internal Role Stability in Voynichese: Morphological Foundations and Multi-Transcription Controls

## Data Source and Licensing Note

The repository includes derived project tables needed to reproduce the analyses in the paper. It does not package raw third-party transcription witnesses as newly redistributed source material. Review the licensing and attribution requirements for IVTFF, EVA, Takahashi, Currier, Zandbergen, Landini, and related transcription resources before redistributing raw transcription data.

The manuscript is Beinecke MS 408 at Yale University Library.
