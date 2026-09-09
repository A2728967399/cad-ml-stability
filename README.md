# CAD machine-learning selection stability

Analysis-code repository for the manuscript **Stability of machine-learning algorithm, feature and hyperparameter selection across random training--test splits: an empirical methodological study of 1304 patients with coronary artery disease**.

**Version 1.0.0 code snapshot; MIT licensed.** The public repository is [A2728967399/cad-ml-stability](https://github.com/A2728967399/cad-ml-stability); anonymous access was verified on 2026-09-10. Versioned release and archival identifiers are verified externally at the release/deposition records, rather than embedded self-referentially in this snapshot. See `RELEASE_STATUS.md`.

This directory is deliberately separate from the clinical workspace. Only explicitly reviewed source files and documentation are eligible for the release archive. Patient data, clinical predictions, membership tables, fitted objects, clinical checkpoints, signed forms, author spreadsheets, manuscript PDFs, and private audit logs are not included.

## Contents

- `source/`: analysis sources in English-named directories, with limited path adaptations documented against the original study sources.
- `SOURCE_MANIFEST.json`: original-source and adapted-file provenance; no clinical run manifest is included.
- `PATH_PORTABILITY.md`: the directory mapping, path-only changes and implications for using retained clinical run artifacts.
- `SOURCES.md`: which parts of the analysis the files cover, and which controlled inputs are required for clinical reproduction.
- `tools/synthetic_smoke.py` and `tests/`: newly written tests using artificial inputs only.
- `tools/split_precision.py`: an aggregate-only, split-unit jackknife precision calculation for paired ranking-stability summaries, with separate synthetic tests.
- `PRIVACY_REVIEW.md`: the code-only packaging review.
- `LICENSE`: the authorized MIT software licence; it does not grant access to patient data.

All file and directory names in this versioned snapshot are English/ASCII. Original Chinese data-column names remain inside the code because they define its input interface; these are labels, not patient records. Start with `SOURCES.md`, rather than executing every script in the archive.

## Testing without patient data

Use a compatible Python environment with the dependencies in `source/strict_development/analysis/requirements_nested.txt`. These are the recorded dependencies of the frozen clinical analysis, not a claim that installation has been tested on every platform. Additional plotting and model-interpretation dependencies are explained in `SOURCES.md`.

The new helper's `--help` describes the explicitly synthetic test options. Its generated inputs and outputs are temporary and are not included in this repository. The original synthetic unit tests can be run from the repository root:

```sh
python -m unittest discover -s source/strict_development/tests -p 'test_*.py'
python -m unittest discover -s tests -p 'test_*.py'
python tools/synthetic_smoke.py --help
```

Do not run all scripts labelled `--self-test`: some original analysis scripts still load controlled clinical data in that mode. Several historical scripts also have import-time effects. See the warnings in `SOURCES.md`.

## Reproducing the clinical analysis

Reproduction requires institutionally authorized access to the input data and retained analysis artifacts described in `SOURCES.md`. These inputs are deliberately not distributed. This is an analysis-source archive, not a data-free executable reproduction of the manuscript, and it does not promise exact regeneration of all English figure layouts.

Synthetic tests check selected software behaviour only. They do not reproduce or independently validate the patient-based findings. The original 29 study sources and their source manifest were not changed to add the precision tool, and the patient-based models were not refitted.

The separate `split_precision.py` tool estimates split-unit leave-one-out jackknife Monte Carlo standard errors for mean pairwise rank correlations and their paired contrast. It requires three hash-verified aggregate files from the retained analysis (`per_split_metrics.csv`, `rank_correlations.csv` and `summary.json`), which are not included. It neither loads patient rows nor treats overlapping split pairs as independent observations. Its precision is conditional on the retained cohort and computational design, not population uncertainty or a hypothesis test. The aggregate calculation was performed separately from the synthetic software tests; its generated JSON and TeX outputs are not part of this code archive.

## Data and reuse

No dataset is included, including synthetic dataset files. Artificial test arrays are created at runtime. Patient-level data access remains subject to institutional approval and is not granted by access to this code. The rights holder authorized distribution of this code and documentation under the MIT licence in `LICENSE`; third-party dependencies retain their own licences.
