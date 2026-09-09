#!/usr/bin/env python3
"""Strictly nested, outcome-stratified development of the eight historical models.

No old analysis module is imported. Full mode preserves the original grids and
100 LASSO C values. Smoke mode is a deliberately reduced software check and is
never a clinical analysis. See README_nested.md for the layer hierarchy.
"""
from __future__ import annotations

import os
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_var] = "1"

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
import traceback
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import optimize, special, stats
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

SEED = 42
FEATURES = [
    "性别", "年龄", "高血压史", "糖尿病史", "吸烟史", "饮酒史", "BMI", "心率",
    "舒张压", "收缩压", "肌酸激酶", "乳酸脱氢酶", "肌酸激酶同工酶", "血清肌钙蛋白T",
    "NLR", "dNLR", "MLR", "PLR", "SII", "SIRI", "AISI", "TyG",
    "丙氨酸氨基转移酶", "天门冬氨酸氨基转移酶", "谷氨酰转肽酶", "总胆红素", "白蛋白",
    "肌酐", "尿素", "尿酸", "总胆固醇", "甘油三酯", "低密度脂蛋白", "高密度脂蛋白",
    "TC-HDLDL", "血钾", "血氯", "血钠", "血钙", "葡萄糖", "PR间期", "QRS时限",
    "QTc间期", "T Axis", "左室射血分数", "心指数", "左室舒张末内径", "心室收缩容量", "室壁运动评分",
]
CATEGORICAL = {"性别", "高血压史", "糖尿病史", "吸烟史", "饮酒史", "室壁运动评分"}
RANGES = {
    "BMI": (12, 70), "收缩压": (60, 260), "舒张压": (30, 160), "心率": (30, 220),
    "左室射血分数": (10, 85), "心指数": (1.0, 8.0), "左室舒张末内径": (25, 90),
    "心室收缩容量": (5, 400), "PR间期": (80, 400), "QRS时限": (40, 250),
    "QTc间期": (300, 650), "T Axis": (-180, 180), "NLR": (0, 200), "dNLR": (0, 200),
    "MLR": (0, 50), "PLR": (0, 2000), "SII": (0, 50000), "SIRI": (0, 500),
    "AISI": (0, 50000), "TyG": (5, 15),
}
MODELS = {
    "Logistic回归": (LogisticRegression(max_iter=5000, random_state=SEED), {"C": [0.01, 0.1, 1, 10]}),
    "随机森林": (RandomForestClassifier(random_state=SEED, n_jobs=1),
                 {"n_estimators": [200, 400], "max_depth": [3, 5], "min_samples_leaf": [20, 40]}),
    "K近邻": (KNeighborsClassifier(n_jobs=1), {"n_neighbors": [15, 25, 35], "weights": ["uniform", "distance"]}),
    "梯度提升": (GradientBoostingClassifier(random_state=SEED),
                 {"n_estimators": [100, 200], "max_depth": [2, 3], "learning_rate": [0.05, 0.1], "min_samples_leaf": [20, 40]}),
    "SVM": (SVC(probability=True, random_state=SEED), {"C": [0.1, 1, 10], "gamma": ["scale", 0.05]}),
    "XGBoost": (XGBClassifier(random_state=SEED, eval_metric="logloss", verbosity=0, n_jobs=1, tree_method="hist"),
                {"n_estimators": [200, 400], "max_depth": [2, 3], "learning_rate": [0.05, 0.1], "reg_alpha": [0, 1], "reg_lambda": [1, 5]}),
    "LightGBM": (LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=1),
                 {"n_estimators": [200, 400], "max_depth": [2, 3], "learning_rate": [0.05, 0.1], "reg_alpha": [0, 1], "reg_lambda": [1, 5]}),
    "多层感知机": (MLPClassifier(random_state=SEED, max_iter=3000, early_stopping=True),
                   {"hidden_layer_sizes": [(20,), (50,), (32, 16)], "alpha": [0.5, 1.0, 2.0]}),
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def serializable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(type(value).__name__)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=serializable, allow_nan=False)


def object_hash(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def describe_params(value):
    """Represent estimator defaults (including NaN sentinels) in strict JSON."""
    if isinstance(value, dict):
        return {k: describe_params(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [describe_params(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return {"nonfinite_default": str(value)}
    return value


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def members_hash(rows):
    return object_hash(sorted(int(r) for r in rows))


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, default=serializable, allow_nan=False))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_csv(path, frame):
    path = Path(path)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def runtime_versions():
    return {"python": sys.version, "platform": platform.platform(),
            "packages": {k: importlib.metadata.version(k) for k in
                         ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost", "joblib"]}}


def load_cohort(raw, exclude_gs_lt2=False):
    """Read only predictor/score/formula columns, never patient ID or subtype."""
    allowed = set(FEATURES + ["Gensini评分", "白细胞计数", "中性粒细胞计数"])
    frame = pd.read_csv(raw, encoding="utf-8-sig", usecols=lambda c: c in allowed)
    missing = allowed - set(frame.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    frame = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    gs = frame["Gensini评分"]
    if gs.isna().any() or (gs < 0).any():
        raise ValueError("Missing/negative GS: outcome undefined")
    den = frame["白细胞计数"] - frame["中性粒细胞计数"]
    frame["dNLR"] = (frame["中性粒细胞计数"] / den).where((den > 0) & (frame["中性粒细胞计数"] >= 0))
    frame["TC-HDLDL"] = frame["总胆固醇"] - frame["高密度脂蛋白"] - frame["低密度脂蛋白"]
    corrections = []
    for feature, (lo, hi) in RANGES.items():
        bad = frame[feature].notna() & ~frame[feature].between(lo, hi)
        corrections.append({"feature": feature, "rule": [lo, hi], "n_source_to_missing": int(bad.sum())})
        frame.loc[bad, feature] = np.nan
    keep = gs >= 2 if exclude_gs_lt2 else gs != 0
    rows = np.flatnonzero(keep.to_numpy())  # zero-based original data-record position; never an ID
    X = frame.loc[keep, FEATURES].copy()
    X.index = rows
    y = (gs.loc[keep].to_numpy() > 37).astype(int)
    audit = {"n_source": len(frame), "excluded_gs_eq0": int((gs == 0).sum()),
             "n_gs_positive_lt2_source": int(((gs > 0) & (gs < 2)).sum()),
             "n_analyzed": len(X), "events": int(y.sum()), "non_events": int((y == 0).sum()),
             "cohort_rule": "GS>=2 sensitivity" if exclude_gs_lt2 else "GS!=0 working cohort",
             "outcome": "GS>37; historically data-derived threshold, now locked",
             "source_row_convention": "zero-based data record index, excluding CSV header",
             "cohort_members_sha256": members_hash(rows), "range_corrections": corrections,
             "predictors": FEATURES, "clinical_subtype_read_or_used": False}
    return X, y, audit


class FoldPreprocessor:
    """All data-dependent preprocessing is fit solely on supplied training rows."""
    def __init__(self, features=None, missing_limit=.15):
        self.features = list(FEATURES if features is None else features)
        self.missing_limit = missing_limit

    def fit(self, X):
        source = X[self.features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        missing = source.isna().mean()
        self.kept = missing[missing <= self.missing_limit].index.tolist()
        if not self.kept:
            raise ValueError("No eligible predictors in this training fold")
        self.impute, self.logs = {}, []
        for feature in self.kept:
            values = source[feature].dropna()
            self.impute[feature] = float(values.mode().iloc[0] if feature in CATEGORICAL else values.median())
            if feature not in CATEGORICAL and values.min() >= 0 and values.nunique() > 1 and abs(float(stats.skew(values))) > 2:
                self.logs.append(feature)
        matrix = self._unscaled(source)
        self.scaler = StandardScaler().fit(matrix)
        self.meta = {"fit_rows_sha256": members_hash(X.index), "n_fit": len(X),
                     "features_kept": self.kept, "features_dropped": missing[missing > self.missing_limit].index.tolist(),
                     "missing_rate_train": missing.to_dict(), "impute_values": self.impute,
                     "log1p_features": self.logs, "scaler_mean": self.scaler.mean_.tolist(),
                     "scaler_scale": self.scaler.scale_.tolist()}
        return self

    def _unscaled(self, X):
        frame = X[self.kept].copy().fillna(self.impute)
        for feature in self.logs:
            if (frame[feature] <= -1).any():
                raise ValueError(f"Out-of-domain held-out log1p value for {feature}; no silent clipping")
            frame[feature] = np.log1p(frame[feature])
        matrix = frame.to_numpy(dtype=float)
        if not np.isfinite(matrix).all():
            raise ValueError("Non-finite values after preprocessing")
        return matrix

    def transform(self, X):
        return self.scaler.transform(self._unscaled(X))


def warning_record(caught):
    categories = Counter(type(item.message).__name__ for item in caught)
    convergence = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
    examples = sorted(set(str(item.message) for item in caught))[:5]
    return {"total": len(caught), "convergence": convergence, "categories": dict(categories), "examples": examples}


def combine_warnings(records):
    counts, examples = Counter(), set()
    total = convergence = 0
    for record in records:
        total += record["total"]
        convergence += record["convergence"]
        counts.update(record["categories"])
        examples.update(record["examples"])
    return {"total": total, "convergence": convergence, "categories": dict(counts), "examples": sorted(examples)[:12]}


def lasso_fold_path(X, y, tr, va, Cs, features):
    prep = FoldPreprocessor(features).fit(X.iloc[tr])
    Ztr, Zva = prep.transform(X.iloc[tr]), prep.transform(X.iloc[va])
    scores, recs = [], []
    for c in Cs:
        estimator = LogisticRegression(C=float(c), l1_ratio=1., solver="liblinear", max_iter=5000, random_state=SEED)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(Ztr, y[tr])
            scores.append(float(roc_auc_score(y[va], estimator.decision_function(Zva))))
        recs.append(warning_record(caught))
    return scores, {"train_members_sha256": members_hash(X.index[tr]),
                    "valid_members_sha256": members_hash(X.index[va]), "preprocessing": prep.meta}, combine_warnings(recs)


def fit_representation(X, y, config, workers=1, features=None):
    """LASSO CV has its own training-fold preprocessing, then final full-fit prep."""
    features = list(FEATURES if features is None else features)
    prep = FoldPreprocessor(features).fit(X)
    if config["all_features"]:
        return prep, list(prep.kept), {"method": "all eligible features", "selected": list(prep.kept),
                                       "preprocessing": prep.meta, "warnings": combine_warnings([])}
    Cs = np.asarray(config["lasso_cs"])
    cv = StratifiedKFold(config["lasso_folds"], shuffle=True, random_state=SEED)
    folds = list(cv.split(X, y))
    outputs = Parallel(n_jobs=workers)(delayed(lasso_fold_path)(X, y, tr, va, Cs, features) for tr, va in folds)
    scores = np.asarray([out[0] for out in outputs])
    means, sds = scores.mean(axis=0), scores.std(axis=0, ddof=1)
    best = int(np.argmax(means))
    cutoff = means[best] - sds[best] / np.sqrt(len(folds))
    first = int(np.flatnonzero(means >= cutoff)[0])
    final_warns, estimator, selected_idx = [], None, None
    Z = prep.transform(X)
    # Preserve the historical empty-model fallback, explicitly recorded below.
    for chosen in range(first, len(Cs)):
        estimator = LogisticRegression(C=float(Cs[chosen]), l1_ratio=1., solver="liblinear", max_iter=5000, random_state=SEED)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(Z, y)
        final_warns.append(warning_record(caught))
        selected_idx = np.flatnonzero(estimator.coef_[0] != 0)
        if len(selected_idx):
            break
    if not len(selected_idx):
        raise RuntimeError("No LASSO features at any eligible fallback C")
    selected = [prep.kept[j] for j in selected_idx]
    audit = {"method": "nested LASSO 1SE", "selected": selected, "C": float(Cs[chosen]),
             "C_initial_1se": float(Cs[first]), "empty_model_fallback_steps": chosen - first,
             "one_se_cutoff": float(cutoff), "best_cv_auc": float(means[best]),
             "chosen_cv_auc": float(means[chosen]), "coef": dict(zip(selected, estimator.coef_[0][selected_idx].tolist())),
             "path": {"C": Cs.tolist(), "fold_auc": scores.tolist(), "mean_auc": means.tolist(), "sd_auc": sds.tolist()},
             "preprocessing": prep.meta, "lasso_cv_folds": [out[1] for out in outputs],
             "warnings": combine_warnings([out[2] for out in outputs] + final_warns)}
    return prep, selected, audit


def representation_matrix(prep, selected, X):
    indices = [prep.kept.index(feature) for feature in selected]
    return prep.transform(X)[:, indices]


def prepare_inner_folds(X, y, config, workers=1, features=None):
    """One immutable representation per exact training membership, reused by grids."""
    cache, audit = [], []
    cv = StratifiedKFold(config["model_folds"], shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(cv.split(X, y)):
        prep, selected, rep = fit_representation(X.iloc[tr], y[tr], config, workers, features)
        cache.append((representation_matrix(prep, selected, X.iloc[tr]), y[tr],
                      representation_matrix(prep, selected, X.iloc[va]), y[va]))
        audit.append({"fold": fold, "train_members_sha256": members_hash(X.index[tr]),
                      "valid_members_sha256": members_hash(X.index[va]), "representation": rep})
    return cache, audit


def predict_pair(name, model, matrix):
    prob = np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)
    score = np.asarray(model.decision_function(matrix), dtype=float) if name == "SVM" else prob
    return prob, score


def candidate_cv(name, params, cache):
    scores, recs = [], []
    for Xtr, ytr, Xva, yva in cache:
        estimator = clone(MODELS[name][0]).set_params(**params)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(Xtr, ytr)
            _, score = predict_pair(name, estimator, Xva)
        scores.append(float(roc_auc_score(yva, score)))
        recs.append(warning_record(caught))
    return {"params": params, "mean_auc": float(np.mean(scores)), "fold_auc": scores,
            "sd_auc": float(np.std(scores, ddof=1)), "warnings": combine_warnings(recs)}


def tune_models(cache, config, workers=1):
    jobs = [(name, params) for name in config["models"] for params in config["grids"][name]]
    values = Parallel(n_jobs=workers)(delayed(candidate_cv)(name, params, cache) for name, params in jobs)
    candidates = {name: [] for name in config["models"]}
    for (name, _), result in zip(jobs, values):
        candidates[name].append(result)
    # Same deterministic first-maximum tie rule as GridSearchCV/ParameterGrid.
    best = {name: max(rows, key=lambda row: row["mean_auc"]) for name, rows in candidates.items()}
    return best, candidates


def calibration_metrics(y, p):
    """Unpenalized binomial MLE: joint slope/intercept plus intercept-only CITL."""
    z = special.logit(np.clip(p, 1e-6, 1 - 1e-6))
    design = np.column_stack([np.ones(len(z)), z])
    def objective(beta):
        eta = design @ beta
        return float(np.sum(np.logaddexp(0, eta) - y * eta))
    def gradient(beta):
        return design.T @ (special.expit(design @ beta) - y)
    result = optimize.minimize(objective, np.array([0., 1.]), jac=gradient, method="BFGS", options={"gtol": 1e-6})
    ok = bool(result.success or np.linalg.norm(gradient(result.x)) < 1e-5)
    try:
        citl = float(optimize.brentq(lambda a: float(np.sum(special.expit(a + z) - y)), -100, 100))
    except ValueError:
        citl = None
    return {"cal_intercept_joint": float(result.x[0]) if ok else None,
            "cal_slope": float(result.x[1]) if ok else None, "cal_intercept_only": citl,
            "calibration_converged": ok, "calibration_message": str(result.message)}


def fitted_predictions(name, params, Ztr, ytr, Zte, yte, rows):
    estimator = clone(MODELS[name][0]).set_params(**params)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(Ztr, ytr)
        prob, score = predict_pair(name, estimator, Zte)
    metrics = {"auc": float(roc_auc_score(yte, score)), "brier": float(brier_score_loss(yte, prob)),
               **calibration_metrics(yte, prob), "fit_warnings": warning_record(caught)}
    predictions = pd.DataFrame({"source_row": rows, "y": yte, "p": prob, "score": score})
    return metrics, predictions


def develop_and_evaluate(Xtr, ytr, Xte, yte, config, historical, workers=1):
    started = time.monotonic()
    cache, inner_audit = prepare_inner_folds(Xtr, ytr, config, workers)
    best, candidates = tune_models(cache, config, workers)
    prep, selected, representation = fit_representation(Xtr, ytr, config, workers)
    Ztr, Zte = representation_matrix(prep, selected, Xtr), representation_matrix(prep, selected, Xte)
    rows, predictions = [], []
    for name in config["models"]:
        for scheme, params in [("retuned", best[name]["params"])] + ([("fixed_historical_conditional", historical[name])] if historical else []):
            metric, prediction = fitted_predictions(name, params, Ztr, ytr, Zte, yte, Xte.index.to_numpy())
            rows.append({"scheme": scheme, "model": name, "params": params, "n_selected": len(selected),
                         "selected_features": selected, "inner_cv_auc": best[name]["mean_auc"] if scheme == "retuned" else None,
                         **metric})
            prediction["scheme"], prediction["model"] = scheme, name
            predictions.append(prediction)
    audit = {"final_representation": representation, "model_cv_folds": inner_audit, "candidates": candidates,
             "cache_scope": "exact inner-training membership; reused across all algorithms/hyperparameter candidates only",
             "seconds": time.monotonic() - started}
    return rows, pd.concat(predictions, ignore_index=True), audit


def strict_primary_oof(Xtr, ytr, config, workers=1):
    """Each OOF fold reruns the entire hierarchy; no test-set labels enter here."""
    predictions, audits = [], []
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED + 101)
    for fold, (tr, va) in enumerate(cv.split(Xtr, ytr)):
        print(f"[{utc()}] primary OOF fold {fold + 1}/5", flush=True)
        _, pred, audit = develop_and_evaluate(Xtr.iloc[tr], ytr[tr], Xtr.iloc[va], ytr[va], config, {}, workers)
        pred["oof_fold"] = fold
        predictions.append(pred)
        audits.append({"oof_fold": fold, "train_members_sha256": members_hash(Xtr.index[tr]),
                       "valid_members_sha256": members_hash(Xtr.index[va]), "development": audit})
    pred = pd.concat(predictions, ignore_index=True)
    thresholds = {}
    for name, group in pred.groupby("model", sort=False):
        fpr, tpr, cuts = roc_curve(group.y, group.p)
        eligible = np.flatnonzero(np.isfinite(cuts))
        best = eligible[np.argmax((tpr - fpr)[eligible])]
        thresholds[name] = {"youden_probability_threshold": float(cuts[best]), "oof_auc": float(roc_auc_score(group.y, group.score)),
                            "n_oof": len(group), "threshold_source": "strict development-only OOF; applied unchanged to holdout"}
    return pred, audits, thresholds


def split_members(X, y, split, primary=False):
    seed = SEED if primary else 1000 + split
    tr, te = train_test_split(np.arange(len(X)), train_size=.75, random_state=seed, stratify=y)
    return tr, te, {"seed": seed, "n_train": len(tr), "n_test": len(te),
                    "events_train": int(y[tr].sum()), "events_test": int(y[te].sum()),
                    "train_members_sha256": members_hash(X.index[tr]), "test_members_sha256": members_hash(X.index[te]),
                    "stratification": "outcome only"}


def verify_checkpoint(folder, run_hash):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["run_hash"] != run_hash or manifest["state"] != "complete":
        raise RuntimeError(f"Checkpoint identity/status mismatch: {folder}")
    for name, digest in manifest["output_sha256"].items():
        if file_hash(folder / name) != digest:
            raise RuntimeError(f"Checkpoint output hash mismatch: {folder / name}")
    return manifest


def run_split(split, args_dict, config, run_hash, historical):
    args = argparse.Namespace(**args_dict)
    key = "primary_42" if args.primary else f"split_{split:03d}"
    output = Path(args.output)
    final = output / "checkpoints" / key
    if final.exists():
        manifest = verify_checkpoint(final, run_hash)
        return {"split": split, "status": "resumed_verified", "seconds": manifest["seconds"]}
    state_path = output / "status" / f"{key}.json"
    atomic_json(state_path, {"state": "running", "split": split, "pid": os.getpid(), "started_utc": utc(), "run_hash": run_hash})
    started = time.monotonic()
    try:
        X, y, _ = load_cohort(args.raw, args.exclude_gs_lt2)
        tr, te, membership = split_members(X, y, split, args.primary)
        print(f"[{utc()}] {key} started: n_train={len(tr)} n_test={len(te)} mode={args.mode}", flush=True)
        metrics, predictions, audit = develop_and_evaluate(X.iloc[tr], y[tr], X.iloc[te], y[te], config, historical, args.workers)
        partial = output / "checkpoints" / f".{key}.partial.{os.getpid()}"
        partial.mkdir(exist_ok=False)
        predictions["split"] = split
        for row in metrics:
            row["split"] = split
        pd.DataFrame({"source_row": X.index, "partition": np.where(np.isin(np.arange(len(X)), tr), "train", "test"), "y": y}).to_csv(partial / "membership.csv", index=False)
        predictions.to_csv(partial / "predictions.csv", index=False, encoding="utf-8-sig")
        atomic_json(partial / "metrics.json", metrics)
        atomic_json(partial / "audit.json", {"membership": membership, "development": audit})
        if args.oof:
            oof, oof_audit, thresholds = strict_primary_oof(X.iloc[tr], y[tr], config, args.workers)
            oof.to_csv(partial / "oof_predictions.csv", index=False, encoding="utf-8-sig")
            atomic_json(partial / "oof_audit.json", oof_audit)
            for name in thresholds:
                data = predictions[(predictions.model == name) & (predictions.scheme == "retuned")]
                classified = data.p.to_numpy() >= thresholds[name]["youden_probability_threshold"]
                true = data.y.to_numpy()
                thresholds[name]["test_sensitivity"] = float(classified[true == 1].mean())
                thresholds[name]["test_specificity"] = float((~classified[true == 0]).mean())
            atomic_json(partial / "thresholds.json", thresholds)
        elapsed = time.monotonic() - started
        manifest = {"state": "complete", "run_hash": run_hash, "split": split, "key": key, "mode": args.mode,
                    "completed_utc": utc(), "seconds": elapsed, "membership": membership,
                    "output_sha256": {p.name: file_hash(p) for p in sorted(partial.iterdir()) if p.is_file()}}
        atomic_json(partial / "manifest.json", manifest)
        os.replace(partial, final)
        atomic_json(state_path, manifest)
        print(f"[{utc()}] {key} COMPLETE {elapsed:.1f}s", flush=True)
        return {"split": split, "status": "complete", "seconds": elapsed}
    except Exception as error:
        atomic_json(state_path, {"state": "failed", "split": split, "run_hash": run_hash, "failed_utc": utc(),
                                 "error": str(error), "traceback": traceback.format_exc()})
        raise


def read_historical(path, models):
    if not path:
        return {}
    source = pd.read_csv(path, encoding="utf-8-sig")
    values = {row["模型"]: json.loads(row["最佳参数"]) for _, row in source.iterrows()}
    if not set(models) <= set(values):
        raise ValueError("Historical parameter table lacks requested models")
    for name, params in values.items():
        if "hidden_layer_sizes" in params:
            params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
    return {name: values[name] for name in models}


def build_config(args):
    names = list(MODELS)
    grids = {name: list(ParameterGrid(grid)) for name, (_, grid) in MODELS.items()}
    if args.mode == "smoke":
        grids = {name: [grid[0]] for name, grid in grids.items()}
    config = {"analysis_id": "Q4_strict_nested_v1", "mode": args.mode,
              "publication_eligible": args.mode == "full", "raw_sha256": file_hash(args.raw),
              "analysis_code_sha256": file_hash(__file__), "runtime": runtime_versions(),
              "historical_params_sha256": file_hash(args.historical_params) if args.historical_params else None,
              "historical_comparator_interpretation": "descriptive, conditional on historical parameters; historical development membership may overlap new test sets",
              "historical_parameter_source": str(Path(args.historical_params).resolve()) if args.historical_params else None,
              "n_splits_planned": 1 if args.primary else args.splits, "start_split": args.start,
              "primary_seed42": args.primary, "strict_primary_oof_requested": args.oof,
              "train_ratio": .75, "split_seed_rule": "42" if args.primary else "1000 + split",
              "outcome_cutoff_locked": 37, "outcome_cutoff_provenance": "historical full-working-cohort median; exploratory estimand, not newly estimated in each split",
              "stratification": "outcome only; clinical subtype neither read nor used",
              "exclude_gs_lt2": args.exclude_gs_lt2, "all_features": args.all_features,
              "missing_fraction_limit": .15, "models": names, "grids": grids,
              "grid_candidates_total": sum(map(len, grids.values())),
              "estimator_base_params": {name: describe_params(estimator.get_params()) for name, (estimator, _) in MODELS.items()},
              "lasso_cs": np.logspace(-3, 3, 100).tolist() if args.mode == "full" else [.001, .1, 1., 1000.],
              "model_folds": 5 if args.mode == "full" else 2, "lasso_folds": 5 if args.mode == "full" else 2,
              "cv_seed": SEED, "features": FEATURES, "categorical": sorted(CATEGORICAL), "range_rules": RANGES,
              "log_rule": "nonnegative observed training values and abs(skew)>2; log1p; heldout values<=-1 fail closed",
              "lasso_tie_rule": "first maximum AUC; smallest C within one SE of best; first nonempty fallback at >= chosen C",
              "lasso_api": "sklearn1.9 l1_ratio=1, liblinear; tested coefficient-identical to historical penalty=l1",
              "thresholds": "no test-derived thresholds; optional strictly nested development-only OOF"}
    return config


def initialize_run(output, config, cohort):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    run_hash = object_hash(config)
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior["run_hash"] != run_hash or prior["config"] != json.loads(canonical(config)):
            raise RuntimeError("Resume refused: code, input, environment or configuration changed; choose a new output directory")
    else:
        if any(output.iterdir()):
            raise RuntimeError("Nonempty output without trusted run manifest; refusing to adopt")
        atomic_json(manifest_path, {"created_utc": utc(), "run_hash": run_hash, "config": config, "cohort": cohort})
    (output / "checkpoints").mkdir(exist_ok=True)
    (output / "status").mkdir(exist_ok=True)
    return run_hash


def aggregate(output, run_hash, expected):
    output = Path(output)
    metrics, features, hyperparameters, statuses = [], [], [], []
    for folder in sorted((output / "checkpoints").iterdir()):
        if folder.name.startswith(".") or not folder.is_dir():
            continue
        manifest = verify_checkpoint(folder, run_hash)
        statuses.append({"split": manifest["split"], "seconds": manifest["seconds"], "key": manifest["key"]})
        for row in json.loads((folder / "metrics.json").read_text(encoding="utf-8")):
            hyperparameters.append({"split": row["split"], "scheme": row["scheme"], "model": row["model"], "parameters": canonical(row["params"])})
            metrics.append({k: v for k, v in row.items() if k not in ["params", "selected_features", "fit_warnings"]})
        rep = json.loads((folder / "audit.json").read_text(encoding="utf-8"))["development"]["final_representation"]
        features.append({"split": manifest["split"], "n_selected": len(rep["selected"]), "features": "|".join(rep["selected"]), "lasso_C": rep.get("C")})
    if metrics:
        frame = pd.DataFrame(metrics)
        frame["rank"] = frame.groupby(["split", "scheme"])["auc"].rank(ascending=False, method="min").astype(int)
        frame["gap_to_same_test_max"] = frame.groupby(["split", "scheme"])["auc"].transform("max") - frame.auc
        atomic_csv(output / "metrics.csv", frame)
        atomic_csv(output / "features.csv", pd.DataFrame(features))
        atomic_csv(output / "hyperparameters.csv", pd.DataFrame(hyperparameters))
    atomic_json(output / "run_status.json", {"updated_utc": utc(), "run_hash": run_hash,
                                            "n_completed": len(statuses), "n_planned": expected,
                                            "state": "complete" if len(statuses) == expected else "partial",
                                            "completed_splits": statuses})
    return len(statuses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--historical-params")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["full", "smoke"], default="full")
    parser.add_argument("--splits", type=int, default=200, help="Planned repetitions; part of resume identity")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, help="Only execute this many pending splits without changing planned design")
    parser.add_argument("--workers", type=int, default=8, help="Candidate/path parallel jobs; 1 when using split-workers")
    parser.add_argument("--split-workers", type=int, default=1)
    parser.add_argument("--all-features", action="store_true")
    parser.add_argument("--exclude-gs-lt2", action="store_true")
    parser.add_argument("--primary", action="store_true", help="One outcome-stratified seed42 primary holdout instead of repetitions")
    parser.add_argument("--oof", action="store_true", help="Additional strictly nested five-fold development OOF; requires --primary")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 1 <= args.split_workers <= 8 or args.workers * args.split_workers > 8:
        parser.error("workers × split-workers must be between 1 and 8")
    if args.split_workers > 1 and args.workers != 1:
        parser.error("Use --workers 1 for split-level parallelism to avoid nested process pools")
    if args.oof and not args.primary:
        parser.error("--oof requires --primary")
    if args.splits < 1 or args.start < 0 or (args.limit is not None and args.limit < 1):
        parser.error("Invalid split count/start/limit")
    config = build_config(args)
    _, _, cohort = load_cohort(args.raw, args.exclude_gs_lt2)
    historical = read_historical(args.historical_params, config["models"])
    run_hash = initialize_run(args.output, config, cohort)
    expected = 1 if args.primary else args.splits
    indices = [0] if args.primary else list(range(args.start, args.start + args.splits))
    pending = []
    for split in indices:
        folder = Path(args.output) / "checkpoints" / ("primary_42" if args.primary else f"split_{split:03d}")
        if folder.exists():
            verify_checkpoint(folder, run_hash)
        else:
            pending.append(split)
    if args.limit:
        pending = pending[:args.limit]
    print(f"run_hash={run_hash} mode={args.mode} grids={config['grid_candidates_total']} pending_this_invocation={len(pending)} planned={expected}", flush=True)
    if args.mode == "smoke":
        print("SMOKE TEST ONLY: reduced folds/grid, NOT publication evidence", flush=True)
    # Exclusive live-run lock. A stale lock is explicit evidence to inspect, never silently removed.
    lock = Path(args.output) / "run.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Existing run lock: {lock}; inspect process before removing a stale lock")
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(canonical({"pid": os.getpid(), "utc": utc(), "run_hash": run_hash, "argv": sys.argv}))
    try:
        if args.split_workers == 1:
            for split in pending:
                run_split(split, vars(args), config, run_hash, historical)
                aggregate(args.output, run_hash, expected)
        else:
            with ProcessPoolExecutor(max_workers=args.split_workers) as pool:
                futures = [pool.submit(run_split, split, vars(args), config, run_hash, historical) for split in pending]
                for future in as_completed(futures):
                    future.result()
                    aggregate(args.output, run_hash, expected)
        completed = aggregate(args.output, run_hash, expected)
        print(f"verified_completed={completed}/{expected}; results={args.output}", flush=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
