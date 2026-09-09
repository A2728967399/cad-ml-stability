"""Conditional split-level jackknife precision; aggregate input only.

No patient data are read, no model is fitted, and no hypothesis test is produced.
The CLI verifies the frozen aggregate files before generating JSON/TeX summaries.
Numerical kernels are separately reusable and tested using synthetic matrices.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata

SCHEMES = ("fixed_historical_conditional", "retuned")
MODELS = ("Logistic回归", "随机森林", "K近邻", "梯度提升", "SVM",
          "XGBoost", "LightGBM", "多层感知机")
FROZEN = {
    "per_split_metrics.csv": "88833af6ded973f1a45b8cc9930896238910d2737e663554cd98694c62004f46",
    "rank_correlations.csv": "1ee2b20aef27399a9863e783a8bdd0357c6c6296764d6d62db1dfd4bcc9069fc",
    "summary.json": "9a8016c640e00140453bccc96de1c9b14b537ca45b6ac2da81464a3172f6f03b",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def correlation_kernel(auc):
    """Pearson correlations of per-split descending average-tied AUC ranks."""
    auc = np.asarray(auc, dtype=float)
    require(auc.ndim == 2 and min(auc.shape) >= 2, "Need a split-by-model matrix")
    require(np.isfinite(auc).all(), "Non-finite AUC")
    require(((auc >= 0) & (auc <= 1)).all(), "AUC outside [0,1]")
    ranks = rankdata(-auc, axis=1, method="average")
    centered = ranks - ranks.mean(axis=1, keepdims=True)
    length = np.sqrt((centered ** 2).sum(axis=1, keepdims=True))
    require((length > 0).all(), "Undefined ranking correlation: all algorithms tied")
    vectors = centered / length
    kernel = np.clip(vectors @ vectors.T, -1.0, 1.0)
    return kernel, ranks


def jackknife_pair_mean(kernel):
    """Delete one randomization unit, not one of the dependent pair entries."""
    kernel = np.asarray(kernel, dtype=float)
    require(kernel.ndim == 2 and kernel.shape[0] == kernel.shape[1], "Square kernel required")
    n = len(kernel)
    require(n >= 4 and np.isfinite(kernel).all(), "Need >=4 finite split units")
    require(np.allclose(kernel, kernel.T, rtol=0, atol=2e-14), "Kernel is not symmetric")
    upper = kernel[np.triu_indices(n, 1)]
    total = float(upper.sum())
    row_sum = kernel.sum(axis=1) - np.diag(kernel)
    leave_one = (total - row_sum) / math.comb(n - 1, 2)
    estimate = total / math.comb(n, 2)
    center = float(leave_one.mean())
    mcse = math.sqrt((n - 1) / n * float(np.sum((leave_one - center) ** 2)))
    pair_sd = float(np.std(upper, ddof=1))
    # A first-order jackknife does not consistently estimate a degenerate
    # U-statistic's variance. Do not silently interpret near-zero influence as
    # exact precision if the pair kernel itself still varies.
    warning = pair_sd > 1e-10 and float(np.ptp(leave_one)) < 1e-10
    return {
        "estimate": estimate,
        "mcse": mcse,
        "n_splits": n,
        "n_dependent_pairs": math.comb(n, 2),
        "near_degenerate_first_order_warning": bool(warning),
        "leave_one_range": [float(leave_one.min()), float(leave_one.max())],
    }, leave_one


def paired_precision(fixed, retuned):
    require(np.shape(fixed) == np.shape(retuned), "Paired matrices differ in shape")
    fixed_kernel, fixed_ranks = correlation_kernel(fixed)
    retuned_kernel, retuned_ranks = correlation_kernel(retuned)
    outputs = {}
    for name, kernel in (("fixed", fixed_kernel), ("retuned", retuned_kernel),
                         ("fixed_minus_retuned", fixed_kernel - retuned_kernel)):
        outputs[name], _ = jackknife_pair_mean(kernel)
    outputs["tied_rank_split_counts"] = {
        name: int(sum(len(set(row)) < len(row) for row in ranks))
        for name, ranks in (("fixed", fixed_ranks), ("retuned", retuned_ranks))
    }
    return outputs, (fixed_kernel, retuned_kernel)


def load_frozen(folder):
    payloads = {}
    for name, expected in FROZEN.items():
        data = (folder / name).read_bytes()
        require(hashlib.sha256(data).hexdigest() == expected, "Frozen input hash mismatch: " + name)
        payloads[name] = data.decode("utf-8-sig")
    source = json.loads(payloads["summary.json"])
    require(source["complete"] and source["publication_result_set_ready"]
            and source["n_completed"] == source["expected"] == 200,
            "Frozen result set is not complete/publication-ready")
    rows = list(csv.DictReader(payloads["per_split_metrics.csv"].splitlines()))
    require(len(rows) == 3200, "Expected exactly 200 x 8 x 2 metric rows")
    indexed = {}
    for row in rows:
        key = (row["scheme"], int(row["split"]), row["model"])
        require(key not in indexed, "Duplicate metric key")
        require(key[0] in SCHEMES and key[1] in range(200) and key[2] in MODELS,
                "Unexpected strategy/split/model")
        indexed[key] = float(row["auc"])
    arrays = [np.array([[indexed[(scheme, i, model)] for model in MODELS]
                        for i in range(200)]) for scheme in SCHEMES]
    output, kernels = paired_precision(*arrays)
    pairs = list(csv.DictReader(payloads["rank_correlations.csv"].splitlines()))
    require(len(pairs) == 39800, "Expected exactly 19,900 pairs per strategy")
    seen = set()
    max_error = 0.0
    for row in pairs:
        scheme, i, j = row["scheme"], int(row["split_i"]), int(row["split_j"])
        key = (scheme, i, j)
        require(scheme in SCHEMES and 0 <= i < j < 200 and key not in seen,
                "Invalid/duplicate pair key")
        seen.add(key)
        rho = float(row["rho"])
        require(math.isfinite(rho) and -1 <= rho <= 1, "Invalid source rho")
        error = abs(rho - kernels[SCHEMES.index(scheme)][i, j])
        max_error = max(max_error, error)
        require(error < 2e-14, "Recomputed pair differs from frozen result")
    for name, scheme in zip(("fixed", "retuned"), SCHEMES):
        require(abs(output[name]["estimate"] - source["rank_correlations"][scheme]["mean"]) < 2e-14,
                "Recomputed mean differs from frozen summary")
    output.update({
        "schema": "conditional-split-precision-v1",
        "source_sha256": FROZEN,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "validation": {"all_pairs_reconciled": True, "max_absolute_pair_error": max_error,
                       "paired_splits": 200, "algorithms": 8, "seeds": "1000-1199"},
        "scope": "Post hoc numerical precision conditional on the observed cohort and recorded randomization/development mechanism; first-order split-unit jackknife, not population confidence intervals, hypothesis tests, or a pure tuning effect.",
        "patient_data_read": False,
        "models_refitted": False,
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__},
    })
    return output


def tex_macros(report):
    lines = ["% Generated by tools/split_precision.py from hash-verified aggregate results.",
             "% MCSE is conditional randomization precision, not population uncertainty."]
    for key in ("fixed", "retuned", "fixed_minus_retuned"):
        label = key.replace("_", "-")
        for field in ("estimate", "mcse"):
            value = report[key][field]
            lines.append(r"\expandafter\def\csname result@precision-" + label + "-" + field
                         + r"\endcsname{" + f"{value:.4f}" + "}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", required=True, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--tex-output", required=True, type=Path)
    args = parser.parse_args()
    report = load_frozen(args.aggregate_dir.resolve())
    require(not any(report[key]["near_degenerate_first_order_warning"]
                    for key in ("fixed", "retuned", "fixed_minus_retuned")),
            "Near-degenerate first-order diagnostic; do not auto-publish MCSE")
    inputs = {(args.aggregate_dir / name).resolve() for name in FROZEN}
    outputs = (args.json_output.resolve(), args.tex_output.resolve())
    require(len(set(outputs)) == 2 and not set(outputs) & inputs, "Output overlaps input/other output")
    for path in outputs:
        require(not path.exists(), "Output exists; retain the previous analysis and choose a new output")
        path.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    args.tex_output.write_text(tex_macros(report), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("fixed", "retuned", "fixed_minus_retuned")}, indent=2))


if __name__ == "__main__":
    main()
