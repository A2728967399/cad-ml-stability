# Published software version

## Archived version 1.0.0

- [GitHub release v1.0.0](https://github.com/A2728967399/cad-ml-stability/releases/tag/v1.0.0)
- [Zenodo archive: DOI 10.5281/zenodo.22682152](https://doi.org/10.5281/zenodo.22682152)
- [Tagged source snapshot](https://github.com/A2728967399/cad-ml-stability/tree/v1.0.0), commit `c22d3c9252dfe23307e035580fd586c569d7dded`
- Software licence: [MIT](LICENSE), authorized by the rights holder. The licence does not grant access to patient data.

Public access, the version tag and the archived code ZIP were verified on 2026-09-10. The GitHub release asset and the Zenodo deposit contain the same ZIP:

```text
File:   analysis_code_v1_0_0_20260910.zip
Size:   217166 bytes
SHA256: 28ddac45a89492c9c80477f37f3df5d253b9cb77b626d520623d6e3b16922878
```

This checksum identifies the attached code ZIP, not GitHub's automatically generated source-code archives.

## Current branch and archived snapshot

Use the tagged or archived version above to inspect the exact code package cited in the manuscript. The `main` branch includes post-release navigation updates to `README.md`, this file and the corresponding integrity manifest; the analysis code is unchanged. These updates do not replace the v1.0.0 tag, release asset or Zenodo archive.

`CONTENTS_SHA256.json` records the hashes of the other 50 files in the copy that contains it. The archived ZIP retains its own original manifest. A current-branch download therefore need not be byte-identical to the archived ZIP.

## Contents and verification scope

The release contains 51 code, documentation and integrity-manifest files. Its 29 original study sources are mapped to English-only file paths, with path adaptations documented in [PATH_PORTABILITY.md](PATH_PORTABILITY.md) and `SOURCE_MANIFEST.json`. Adapted files are not represented as all byte-identical to the original clinical run code.

[TEST_REPORT.md](TEST_REPORT.md) records 40 passing unit tests in two stages (30 study/interface tests and 10 split-precision tests), along with an artificial interface check and a reduced artificial eight-model smoke run. These tests used no clinical data and do not independently validate the patient-based findings. No new software tests or patient-based model fitting were performed for this documentation update.

The separate split-unit jackknife tool uses retained aggregate inputs and does not modify the 29 study sources. Neither those aggregate inputs nor generated precision results are distributed. Patient datasets, patient-level predictions, split membership tables, fitted models, clinical checkpoints and signed documents are also excluded. Clinical reproduction requires authorized access to the controlled inputs described in [SOURCES.md](SOURCES.md).
