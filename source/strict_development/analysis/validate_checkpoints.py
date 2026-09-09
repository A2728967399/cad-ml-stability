#!/usr/bin/env python3
"""Read-only, independently calculated prediction/metadata checkpoint audit.

Does not import the development script. AUC is recomputed from rank sums,
not from sklearn.metrics; Brier is recomputed directly. Pass --output only
when a separate audit JSON artifact is wanted. Partial runs are reported as
partial, and --require-complete fails if fewer than the planned splits exist.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_auc(y, score):
    y = np.asarray(y)
    positives = y == 1
    n1, n0 = int(positives.sum()), int((y == 0).sum())
    return float((rankdata(score, method="average")[positives].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def audit_run(path, require_complete=False):
    path = Path(path)
    run = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    count = groups = predictions = 0
    errors = []
    max_auc_difference = max_brier_difference = 0.
    convergence_count = 0
    parameters_checked = 0
    for checkpoint in sorted((path / "checkpoints").iterdir()):
        if not checkpoint.is_dir() or checkpoint.name.startswith("."):
            continue
        manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
        try:
            assert manifest["run_hash"] == run["run_hash"], "Wrong run identity"
            assert manifest["state"] == "complete", "Incomplete checkpoint"
            for name, sha in manifest["output_sha256"].items():
                assert digest(checkpoint / name) == sha, f"Hash mismatch: {name}"
            membership = pd.read_csv(checkpoint / "membership.csv")
            assert membership.source_row.is_unique, "Duplicated source row in membership"
            assert set(membership.partition) == {"train", "test"}, "Invalid partition"
            test = membership[membership.partition == "test"].set_index("source_row")
            metrics = {(row["scheme"], row["model"]): row for row in json.loads((checkpoint / "metrics.json").read_text(encoding="utf-8"))}
            pred = pd.read_csv(checkpoint / "predictions.csv")
            assert len(metrics) == pred.groupby(["scheme", "model"]).ngroups, "Metric/prediction group count mismatch"
            for (scheme, model), group in pred.groupby(["scheme", "model"], sort=False):
                assert group.source_row.is_unique, "Duplicated patient prediction"
                assert set(group.source_row) == set(test.index), "Prediction/test membership mismatch"
                assert np.array_equal(group.y.to_numpy(), test.loc[group.source_row].y.to_numpy()), "Prediction labels mismatch"
                assert np.isfinite(group[["p", "score"]].to_numpy()).all(), "Non-finite prediction"
                assert group.p.between(0, 1).all(), "Probability outside [0,1]"
                auc = independent_auc(group.y.to_numpy(), group.score.to_numpy())
                brier = float(np.mean((group.y.to_numpy() - group.p.to_numpy()) ** 2))
                metric = metrics[(scheme, model)]
                da, db = abs(auc - metric["auc"]), abs(brier - metric["brier"])
                assert da < 1e-12, f"AUC mismatch {da}"
                assert db < 1e-12, f"Brier mismatch {db}"
                max_auc_difference, max_brier_difference = max(max_auc_difference, da), max(max_brier_difference, db)
                convergence_count += metric["fit_warnings"]["convergence"]
                if scheme == "retuned":
                    assert metric["params"] in run["config"]["grids"][model], "Retuned parameters outside declared grid"
                    parameters_checked += 1
                groups += 1
                predictions += len(group)
            audit = json.loads((checkpoint / "audit.json").read_text(encoding="utf-8"))["development"]
            representations = [audit["final_representation"]] + [fold["representation"] for fold in audit["model_cv_folds"]]
            for representation in representations:
                convergence_count += representation["warnings"]["convergence"]
                for fold in representation.get("lasso_cv_folds", []):
                    assert fold["preprocessing"]["fit_rows_sha256"] == fold["train_members_sha256"], "LASSO preprocessing membership mismatch"
                    assert fold["valid_members_sha256"] != fold["train_members_sha256"], "LASSO train/validation identity collision"
            for fold in audit["model_cv_folds"]:
                assert fold["representation"]["preprocessing"]["fit_rows_sha256"] == fold["train_members_sha256"], "Model-CV representation membership mismatch"
            convergence_count += sum(candidate["warnings"]["convergence"] for candidates in audit["candidates"].values() for candidate in candidates)
            count += 1
        except (AssertionError, KeyError, ValueError) as error:
            errors.append({"checkpoint": checkpoint.name, "error": str(error)})
    planned = run["config"]["n_splits_planned"]
    if require_complete and count != planned:
        errors.append({"run": "incomplete", "completed": count, "planned": planned})
    return {"run_hash": run["run_hash"], "mode": run["config"]["mode"], "publication_eligible": run["config"]["publication_eligible"],
            "completed_checkpoints_verified": count, "planned": planned, "state": "complete" if count == planned else "partial",
            "prediction_groups_verified": groups, "predictions_verified": predictions, "retuned_parameter_sets_checked": parameters_checked,
            "auc_max_absolute_difference": max_auc_difference, "brier_max_absolute_difference": max_brier_difference,
            "convergence_warnings_recorded": convergence_count, "errors": errors,
            "audit_code_sha256": digest(__file__), "run_manifest_sha256": digest(path / "run_manifest.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit_run(args.run, args.require_complete)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    raise SystemExit(1 if report["errors"] else 0)


if __name__ == "__main__":
    main()
