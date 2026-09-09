# English directory-path adaptation

The source archive uses English directory names while retaining traceability to
the original research code. This is a directory-path adaptation, not a revision
of the analysis. The original research files and the pre-adaptation code archive
have not been edited.

## Directory mapping

Paths below are relative to the original project collection or to this
repository's `source/` directory, respectively.

| Original project directory | Repository source directory |
| --- | --- |
| `Q4_方法学重建_20260908` | `strict_development` |
| `Springer_方法学原稿更新版_20260908` | `supplementary_analysis` |
| `Springer_英文翻译版_20260908` | `reporting` |
| `临床篇_八模型/Q3_GS0重分析` | `legacy_clinical/reanalysis` |
| `联邦学习_冠脉病变` | `clinical_pipeline` |

The two-level legacy directory remains two levels deep. Existing `analysis/`,
`tests/`, `scripts/`, and `eval/` subdirectories are retained. All actual source
file and directory names are ASCII. Chinese data-column names and explanatory
text inside the original code remain unchanged; translating them would exceed a
directory-only adaptation. Original directory names appear in this document and
the manifest solely as provenance text.

## Permitted source edits

Only the directory-name strings in the mapping above were replaced in applicable
source files and in the original test's historical-source lookup. The legacy
two-level path uses two corresponding directory tokens. No functions, control
flow, assertions, numerical constants, hyperparameter grids, feature definitions,
outcome definitions, scores, or statistical calculations were changed.

`SOURCE_MANIFEST.json` has exactly 29 source entries. Each records:

- `path`, `sha256`, and `bytes` for the repository file;
- `original_path` and `original_sha256` for the original source file;
- `changes` documenting directory relocation and any replaced directory strings.

These are source-code provenance records. They do not contain patient records,
dataset checksums, or clinical run-manifest contents. New navigation READMEs,
documentation, and test tools are outside this 29-entry source manifest.

## Checks completed for the adaptation

The pre-adaptation backup was the 47-file code-only archive with SHA-256
`ad61552648aebc1808ad218dcc5dbc46115bfc1e8043c46418de3fb673860369`.
Every original archived source was checked against both that backup and the
unchanged original research file.

For all 29 repository source files, the verifier constructed the expected bytes
by applying only the approved directory-token substitutions to the original
bytes. The actual files matched those expected bytes exactly. Results:

- 13 files contain directory-path string edits.
- 16 files remain byte-identical to their originals.
- All 27 Python files parsed successfully. Their syntax trees were identical
  after mapping only the approved directory strings in the original string
  constants; no other syntax-tree change was present.
- File hashes and sizes match the new source manifest.
- Every actual file and directory name under `source/` is ASCII.

The 13 files with source-text edits are:

```text
strict_development/analysis/audit_historical.py
strict_development/analysis/cohort_audit.py
strict_development/analysis/summarize_strict.py
strict_development/tests/test_nested_pipeline.py
supplementary_analysis/analysis/build_supplement.py
supplementary_analysis/analysis/complete_original_metrics.py
supplementary_analysis/analysis/complete_original_models.py
supplementary_analysis/analysis/sync_evidence.py
reporting/analysis/software_environment_v1.py
reporting/analysis/study_numbers_v1.py
reporting/analysis/sync_editorial_numbers.py
legacy_clinical/reanalysis/scripts/01_q3_analysis.py
legacy_clinical/reanalysis/scripts/02_q3_figures_tables.py
```

`strict_development/analysis/nested_pipeline.py` required no source edit. Its
SHA-256 remains
`0d1d41393b06f57f8a79499b0476c67b2021d28698f2106d0317e6f683693f8f`.
The `EXPECTED_PIPELINE_SHA` guard in `complete_original_models.py` therefore
remains unchanged and still matches. No checksum guard was disabled or bypassed.
Synthetic-test execution and its actual runtime are documented separately in
`TEST_REPORT.md`; the static checks above do not claim a clinical rerun.

## What the adaptation does not provide

No clinical dataset, individual prediction, fitted model, clinical checkpoint,
original run manifest, or patient-derived result has been added. Clinical input
paths and parameter-table names in code describe external controlled inputs, not
files distributed here. Existing source code can contain checksums or identifiers
of its historical dependencies; those constants were not rewritten to fabricate
new evidence or make missing inputs pass verification.

Frozen run configurations and summaries may refer to their original absolute
paths, input and code hashes, execution environment, and verification records.
Renaming repository directories does not make those records resumable or
portable automatically. Original clinical run manifests and checkpoints were
not changed. A future authorized rerun requires genuine inputs, an appropriate
environment, and a separate, explicitly recorded configuration; it must not be
presented as the original run merely because this source is available.

The archive documents analysis and reporting implementation, not a standalone
build of the English submission files. Exact English figure styling and PDF page
layout are outside its coverage. See `SOURCES.md` for the analysis map, additional
dependencies, and warnings about scripts with import-time actions or clinical
data-dependent self-test modes.
