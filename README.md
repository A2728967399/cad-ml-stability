# CAD machine-learning selection stability

Analysis-code preparation repository for the manuscript **Stability of machine-learning algorithm, feature and hyperparameter selection across random training--test splits: an empirical methodological study of 1304 patients with coronary artery disease**.

**Status: private code preparation, not a public software release.** The destination is [A2728967399/cad-ml-stability](https://github.com/A2728967399/cad-ml-stability). A software licence and archival identifier have not yet been assigned. A private repository is not anonymously accessible and must not be described as publicly available in the manuscript.

This directory is deliberately separate from the clinical workspace. Only explicitly reviewed source files and documentation are eligible for the release archive. Patient data, clinical predictions, membership tables, fitted objects, clinical checkpoints, signed forms, author spreadsheets, manuscript PDFs, and private audit logs are not included.

## Contents

- `source/`: analysis sources in English-named directories, with limited path adaptations documented against the original study sources.
- `SOURCE_MANIFEST.json`: original-source and adapted-file provenance; no clinical run manifest is included.
- `PATH_PORTABILITY.md`: the directory mapping, path-only changes and implications for using retained clinical run artifacts.
- `SOURCES.md`: which parts of the analysis the files cover, and which controlled inputs are required for clinical reproduction.
- `tools/synthetic_smoke.py` and `tests/`: newly written tests using artificial inputs only.
- `PRIVACY_REVIEW.md`: the code-only packaging review.

All repository file and directory names are English/ASCII. Original Chinese data-column names remain inside the code because they define its input interface; these are labels, not patient records. Start with `SOURCES.md`, rather than executing every script in the archive.

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

Synthetic tests check selected software behaviour only. They do not reproduce or independently validate the patient-based findings. Preparing this archive did not rerun the clinical models, change manuscript results, or add permutation or Monte Carlo precision analyses.

## Data and reuse

No dataset is included, including synthetic dataset files. Artificial test arrays are created at runtime. Patient-level data access remains subject to institutional approval and is not granted by access to this code. Until the rights holder approves a software licence, this preparation does not grant an open-source reuse licence.
