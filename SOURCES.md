# Analysis source coverage and dependencies

This repository contains a path-adapted archive of 29 explicitly selected code,
test, and dependency-specification files. All actual directory and file names
under `source/` use ASCII characters. Original relative directory depth is
retained, and the corresponding directory-path strings in code are adapted to
the English names. Thirteen files have these path-only edits; the other 16 remain
byte-identical to their original sources. Statistical logic, data-column names,
features, model settings, outcomes, and scores are unchanged.

`SOURCE_MANIFEST.json` records each file's repository path, SHA-256 digest, size,
original project-relative path and source digest, and permitted path changes.
It contains code-file checksums only, not checksums or contents of the original
clinical dataset or run manifests. Original path names are retained as provenance
text in that manifest, not as actual repository directory names. See
`PATH_PORTABILITY.md` for the mapping and verification boundaries.

The archive documents the implementation used across the analysis and subsequent
reporting stages. It is not a self-contained reproduction package for the
clinical estimates: the required patient data, fitted models, prediction files,
clinical checkpoints, and frozen run records are not included. Synthetic tests
can assess specific implementation properties, but cannot validate the reported
clinical estimates or replace the original data and run records.

## Source groups

For brevity, the coverage map below uses these directory names:

- **Q4**: `source/strict_development/analysis/`
- **Recovery**: `source/supplementary_analysis/analysis/`
- **Legacy Q3**: `source/legacy_clinical/reanalysis/scripts/`
- **English reporting**: `source/reporting/analysis/`
- **Metric library**: `source/clinical_pipeline/eval/fl_metrics.py`

## Analysis coverage

| Component | Included implementation | Role and boundary |
| --- | --- | --- |
| Strict model development | Q4 `nested_pipeline.py` | Outcome-only splitting, fold-local preprocessing and feature selection, tuning, all-feature control, the conditional fixed-parameter comparator, and the primary out-of-fold procedure. The original code is unchanged. |
| Repeated-split summaries and comparisons | Q4 `summarize_strict.py` | Independently checks recorded predictions and preprocessing boundaries; summarizes AUC, Brier score, calibration, rankings, selected features, hyperparameters, algorithm selection, and paired all-feature comparisons. |
| Primary conditional inference | Q4 `summarize_strict.py` | `primary_conditional_intervals` supplies the primary DeLong covariance, paired differences, and Holm adjustment. `independent_metrics` recomputes AUC, Brier score, and calibration. These are not copied directly from the earlier Q3 results. |
| Checkpoint integrity | Q4 `validate_checkpoints.py` | Checks membership, prediction, metric, parameter, and checkpoint consistency using independently calculated AUC and Brier scores. Requires controlled clinical checkpoint files. |
| Feature-set similarity and performance | Q4 `structure_performance.py` | Descriptive Jaccard versus within-algorithm AUC differences across dependent split pairs. Requires verified summary files and the original analysis specification, neither supplied here. |
| Out-of-fold score scales | Q4 `oof_scale_audit.py` | Compares pooled and within-fold AUC summaries using recorded primary out-of-fold predictions; does not refit the models. |
| Simulation | Q4 `simulation.py`, `verify_simulation.py` | Implements the factorial simulation and independent verification of its recorded summaries and data-generating mechanism. Simulation outputs and run records are not included. The verifier does not independently refit all original models. |
| Training events | Q4 `audit_training_events.py` | Summarizes captured warnings, fit counts, missing warning coverage, and recorded operational events. A missing warning record is not interpreted as zero warnings. |
| Cohort and baseline summaries | Q4 `cohort_audit.py`; Recovery `complete_original_metrics.py` | Supports baseline reporting and the original-style baseline tables. Requires controlled input records and stored analysis membership. |
| HL, thresholds, decision curves, and subgroups | Recovery `complete_original_metrics.py` | Computes the Hosmer-Lemeshow summaries, threshold-based classifications, paired bootstrap decision-curve summaries, and clinical-subtype summaries from the strict primary predictions. Extracts the original HL and high-sensitivity-threshold functions from the metric library and Legacy Q3 source. |
| Sensitivity analyses and fitted-model interpretation | Recovery `complete_original_models.py` | Restores the frozen primary specification and checks it against stored predictions; implements the locked-specification sensitivity analyses, conditional Logistic and LASSO bootstrap summaries, and random-forest SHAP interpretation. Requires the original development records and frozen checkpoint. |
| LASSO diagnostic path | Recovery `complete_lasso_diagnostic_path.py` | Uses `complete_original_models.py` and the frozen Q4 pipeline to reconstruct the diagnostic path. This is separate from the repeated-development feature-selection analysis. |
| Numerical reporting | Recovery `sync_evidence.py`, `build_supplement.py`; English reporting `sync_editorial_numbers.py`, `study_numbers_v1.py`, `software_environment_v1.py` | Maps verified outputs to table rows, numerical macros, and software-version statements. The corresponding manuscript, generated files, audit records, and clinical outputs are deliberately not included. |
| Historical implementation and provenance | Q4 `audit_historical.py`; Legacy Q3 `01_q3_analysis.py`, `02_q3_figures_tables.py`, `05_ranking_stability.py`, `07_stability_retuned.py`, `08_cutoff_stability.py`, `clinical_units.py`; metric library | Preserves the earlier implementation, original threshold function, constants, grids, and reporting dependencies. Historical Q3 analyses are not the current strict outcome-only analysis. |

The Legacy Q3 code is retained for more than historical interest. The recovery
code extracts `threshold_at_min_sensitivity` from `01_q3_analysis.py`; the nested
tests inspect that file's constants and model grids through its syntax tree.
The recovery run also records the original analysis and figure scripts among its
provenance inputs. The metric library supplies the original Hosmer-Lemeshow
function. Omitting these sources would leave gaps in the implementation record.

The archived historical source can contain comments or audit labels that predate
later author confirmations. Those comments describe that source version and
should not be read as an updated statement of the study's clinical eligibility
or ethics status.

## External dependencies and portability

The directory mirror preserves source relationships, not every historical runtime
dependency. In particular:

- Clinical scripts reference the original sibling dataset path, historical
  parameter table, and Q4 result directories. Some accept explicit input paths;
  others retain their original project-relative defaults.
- Summaries and frozen run configurations can themselves contain absolute paths
  to verification records or historical parameter sources. Those external records
  are not included. Relocating code does not resolve their paths automatically.
- Checkpoint and recovery guards compare code, configuration, environment, and
  input checksums. In particular, `complete_original_models.py` pins the original
  Q4 pipeline source checksum. The pipeline required no source edit, so this
  guard remains unchanged. Do not manufacture placeholder manifests, rewrite
  original run records, or bypass these guards. The directory adaptation does not
  establish that an old run can be resumed in the new layout.
- Several audit scripts are specific to the recorded design: for example, the
  primary out-of-fold audit checks the original cohort counts, and the training
  event audit checks the planned repeated and primary runs. These are not general
  purpose defaults for a new dataset.
- `structure_performance.py` expects the external
  `strict_development/audit/structure_performance_spec.md`. That analysis
  specification is not part of this code-only archive.
- Figure-related historical code also depends on plotting libraries and fonts.
  The later English figure translation, layout, and PDF-production workflow is
  outside this archive. Exact reconstruction of the published English figure
  appearance or manuscript page layout is not guaranteed.

`Q4/requirements_nested.txt` records the observed frozen clinical execution
packages, including Python 3.11.16 in its comment. `Q4/requirements_figures.txt`
records additional observed figure-generation packages. These are execution
records, not versions inferred from the current manuscript-building computer.

The frozen full model-recovery run additionally recorded **SHAP 0.51.0** in its
`config.shap_version` field. SHAP is required for that script's interpretation
stage but is not listed in the original nested-analysis requirements file.
Only this software-version field was consulted for the present documentation;
the run manifest is not included in the repository.

## Tests and execution cautions

The two archived test modules are
`source/strict_development/tests/test_nested_pipeline.py` and
`source/strict_development/tests/test_training_events.py`. They use synthetic
arrays or constructed metadata for their checks; the nested test also reads the
Legacy Q3 source through its syntax tree. Their execution still requires
compatible installed packages. Test results should identify the actual test
environment and should not be presented as a rerun of the clinical analysis.

Do not import or run every archived script as a smoke test:

- Legacy Q3 `05_ranking_stability.py`, `07_stability_retuned.py`, and
  `08_cutoff_stability.py` execute their clinical workflow at module level.
- Legacy Q3 `01_q3_analysis.py` and `02_q3_figures_tables.py` create output
  directories during import, although their main workflows are guarded.
- `summarize_strict.py --self-test` includes checks against an external clinical
  benchmark and the real input data after its synthetic checks.
- `complete_original_metrics.py --self-test` still reads the clinical inputs and
  performs a reduced bootstrap analysis. It is not a data-free test.
- Running a producer or audit can write new outputs. The copied source is an
  archive, not an instruction to execute those workflows automatically.

## Materials deliberately excluded

No dataset, patient-level record, fitted model, individual prediction, clinical
run manifest, checkpoint, original audit output, or clinical result file is
included. The archive also excludes patient-derived bootstrap arrays, source-row
membership and mappings, SHAP beeswarm records, and model-recovery outputs.
Names such as `deidentified` or `nonidentifying` in the original code do not make
the corresponding patient-derived files suitable for automatic publication.

Only the selected code, tests, and dependency files are mirrored. Navigation
READMEs added within `source/` are new repository documentation, not part of the
29-file source manifest. New synthetic fixtures, documentation, or verification
tools are likewise separate from the path-adapted original source archive.
