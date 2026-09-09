# Code-only packaging privacy review

Review date: 2026-09-10; updated for the English-directory migration.

**Outcome: no blocking disclosure was identified in the reviewed snapshot.**
This is a bounded static packaging review, not a guarantee of anonymity, a
security certification, or approval to publish patient-derived outputs. The
software licence and public archival release remain separate decisions.

## Scope and identity

The source review covers 29 selected source/dependency files: 27 Python files
and two requirements files. The original set totalled 488390 bytes; the
path-adapted set totals 488110 bytes. The repository documentation, eight
navigation README files and two new synthetic utilities were reviewed separately
from that source set. `.git` and external clinical artifacts are excluded.
The release scope is 47 allowlisted code, documentation and control files,
including this report. `CONTENTS_SHA256.json` is regenerated only after those
files are finalized, giving 48 archive entries. A retained pre-migration copy of
that generated manifest is not evidence about the adapted release. The final
all-content manifest and archive are separate packaging checks.

All 29 entries in `SOURCE_MANIFEST.json` contain the original project-relative
path and SHA-256, the current ASCII path, current SHA-256 and size, and a change
record. The original source digests and all adapted file digests/sizes were
independently checked. The manifest SHA-256 is
`78acee35ecbf61a12c15bacb262c3053cd9970b568f469522984fd771b734ac0`.
The migration changed directory strings in 13 files; 16 files remain
byte-identical to their original sources. The adapted set must not be described
as entirely byte-identical to the clinical execution sources.

The preparation repository and the 29 original code/dependency files used for
comparison were inspected. No external clinical input,
prediction table, patient identifier list, signed form, fitted model or private
thesis document was opened. No clinical script, model fitting, synthetic helper
or test suite was executed as part of this review. Python syntax trees were
parsed without importing or executing the reviewed modules.

## Checks performed

- Inventoried filenames and extensions; checked for data, archive, image,
  fitted-object, result and credential files, including hidden files outside
  `.git`. All actual repository-relative file and directory names were ASCII.
- Independently reconstructed each adapted source by applying only the six
  documented directory-token replacements to its original UTF-8 bytes. Every
  result matched the delivered bytes exactly. All 27 Python syntax trees were
  also identical after those path replacements. No parameter, feature name,
  outcome definition, numerical analysis or control-flow change was identified.
- Checked the new helper separately: reverting its single directory-string
  change reproduced its previous SHA-256. Its synthetic test module remained
  byte-identical. No test-fixture or artificial data-generation rule changed.
- Scanned the text of all reviewed files for credential assignments, common
  access-token formats, private-key headers, email/URL and absolute-path
  patterns, possible personal identifiers and encoded payload indicators.
- Parsed all 29 Python files, including the two new utilities, and inspected
  flagged literal containers, long strings, source constants, synthetic test
  fixtures, and clinical input/output references. This was targeted manual
  inspection supported by whole-file static scans, not a line-by-line formal
  program verification.
- Read repository documentation and the eight navigation files; checked the
  distinction between archived clinical code, artificial test inputs, and
  excluded clinical artifacts. The final test report and expanded ignore rules
  were also inspected; this review did not independently rerun the reported tests.
- Checked for `LICENSE`, `LICENCE` and `COPYING` files. None was present; no
  licence was added by this review.

## Findings and boundaries

1. No actual patient records, individual predictions, membership tables,
   signed documents, fitted objects, embedded patient-data arrays or usable
   credentials were identified. The reviewed files were Python, requirements,
   Markdown, source/content inventory metadata and `.gitignore`; no clinical data
   or output file was present.
2. Names of clinical fields and excluded input/output files occur in the
   implementation. They are schema and program references, not their contents.
   Current source paths use English directory labels. Original path text in
   provenance records is descriptive metadata, not an additional file or
   directory. No personal home directory or credential-bearing URL was
   identified. Two absolute path
   references point only to standard Windows font locations.
3. Frozen reporting code contains some aggregate study counts, reporting
   constants and whole-file integrity digests for controlled inputs or retained
   results. These are not individual records or per-patient identifiers.
   Nevertheless, it would be inaccurate to describe the entire repository as
   containing no data-file checksums or no aggregate study numbers. The separate
   `SOURCE_MANIFEST.json` contains only the selected code-file checksums.
4. Test literals and the new helper's fixtures are explicitly constructed or
   generated artificial inputs. They are not sampled patient rows and are not
   evidence that the clinical findings have been reproduced.
5. Several archived scripts can read controlled clinical files, create
   patient-derived outputs, or have import-time side effects. Some `--self-test`
   branches also access clinical inputs. A source-code privacy finding does
   not authorize running every script or releasing what those scripts produce.
   The documentation states these execution boundaries.

## Release safeguards

Continue to upload only an explicitly reviewed filename allowlist. Do not use a
recursive upload of the clinical workspace, include `.git`, or assume that
`.gitignore` is a complete disclosure barrier. Producer scripts can create
patient-derived CSV, NPZ/NPY, JSON, figures and serialized objects; not every
possible output suffix is covered by the current ignore rules. Files labelled
`deidentified`, `nonidentifying` or `CONTROLLED` are not automatically public.

The directory-only adaptations are documented in `PATH_PORTABILITY.md`; the
primary `nested_pipeline.py` and its pinned source digest remained unchanged.
No source was changed to remove provenance or bypass integrity checks. Any new
file, modified utility, output inclusion, repository-history transfer or public
release archive requires a fresh check of the actual material being released.
Repository access does not confer a licence to use the code or access patient
data. Confirm the rights holder's software licence and verify the real public
version and archival identifier before describing this code as publicly
available in the manuscript.
