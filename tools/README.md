# Preparation and aggregate-precision utilities

synthetic_smoke.py generates artificial inputs in a temporary directory. With no arguments it prints help; --check-only checks the input interface, and --run explicitly executes the reduced smoke pipeline. No clinical data are read. Temporary inputs and outputs are not distributed.

split_precision.py computes split-unit jackknife precision for paired ranking-stability summaries using three hash-verified retained aggregate inputs, not patient rows. Those input files and the generated JSON/TeX results are excluded from this repository. Use --help for its interface; its synthetic unit tests require no retained analysis artifacts. The calculation is conditional computational precision, not a population confidence interval or a new model-fitting workflow.
