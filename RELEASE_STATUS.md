# Version 1.0.0 snapshot status

- Local source inventory: 29 original study sources have been mapped to English-only file paths. Path adaptations are separately documented and checked against the original sources; the adapted files are not described as all byte-identical to the original clinical run code.
- Safe software tests: 40 unit tests passed in the stages documented in TEST_REPORT.md: the previously rerun 30 tests and 10 additional split-precision tests. The artificial interface check and reduced artificial eight-model smoke run also passed. These software tests used no clinical data.
- GitHub destination: `https://github.com/A2728967399/cad-ml-stability`; public visibility and anonymous access verified on 2026-09-10.
- Software licence: MIT, with rights-holder authorization; see LICENSE. This does not authorize access to patient data.
- Snapshot version: 1.0.0. Versioned release and archival status must be checked against the external GitHub release and deposition records. No DOI, future publication claim or self-referential commit identifier is embedded in this snapshot.
- Patient-based reanalysis in this preparation: not performed.
- Additional precision calculation: a separate split-unit jackknife tool was added without modifying the 29 study sources. It was applied to retained aggregate files, not patient rows; generated precision outputs are excluded from the code archive. No new permutation test or patient-based model fitting was performed.

The code-publication workflow must not overwrite the retained clinical analysis runs. The manuscript and submission bundles must cite only identifiers that have actually been assigned and independently verified. This snapshot documents its contents; the existence or status of a later archival record is established externally.
