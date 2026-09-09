# Historical source lineage

These files preserve earlier development code and dependencies used by the later analysis. The two-level legacy_clinical/reanalysis directory layout retains the original relative depth; applicable references to clinical_pipeline were adapted, without changing analysis logic. These are not the current primary-analysis entry points. Several have import-time side effects, including data loading and training. Do not import or run them indiscriminately; see the repository-root SOURCES.md and PATH_PORTABILITY.md.
