# Packaging utility tests

test_synthetic_smoke.py contains 12 tests of the synthetic preparation utility. test_split_precision.py contains 10 tests of the aggregate-precision implementation using constructed arrays only; it does not read the retained aggregate inputs. These are software tests, not validation of clinical results. They are separate from the retained original analysis tests under source/. See the repository-root README.md for commands and TEST_REPORT.md for observed results and scope.
