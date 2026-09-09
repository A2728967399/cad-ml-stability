# -*- coding: utf-8 -*-
"""Q3优化版：泄露受控的8模型比较、校准、DCA与敏感性分析。

设计要点
--------
1. 主分析排除9例Gensini评分为0的记录；经作者病历核实，这些患者为既往
   明确冠心病并接受PCI后复查、本次造影未见再狭窄，不符合主要分析要求的
   当前造影解剖入选标准；保留全部记录作为病例纳入敏感性分析。
2. 明确的生理不可能预测值置为缺失；所有插补、变换、标准化和LASSO
   仅由训练集估计。
3. 主分析采用训练集LASSO（1-SE规则）获得的特征；学位论文16项固定
   特征作为敏感性分析，不与主分析混用。
4. Youden阈值由开发集5折外层交叉拟合预测确定；每个外折均重新执行
   预处理、LASSO与超参数优化，再将阈值原样应用于留出测试集。
5. AUC是主要比较指标；DeLong检验以Logistic回归为参照并进行Holm校正。
6. DCA用测试集bootstrap给出95%置信区间；不使用综合排名定义“最优”。
7. 内部--留出一致性对照采用5个外层验证折AUC的均值减测试AUC；该差值
   不解释为最终模型的“泛化差”，也不比较性质不同的训练表观AUC。
8. 不写出patient_ID，结果目录只保存去标识化行号和汇总结果。
"""

from __future__ import annotations

import json
import hashlib
import os
import pickle
import sys
import time
import warnings
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


SEED = 42
TRAIN_RATIO = 0.75
CV = StratifiedKFold(5, shuffle=True, random_state=SEED)
OUTER_FOLDS = 5
OUTER_CV = StratifiedKFold(
    OUTER_FOLDS, shuffle=True, random_state=SEED + 101)
BOOTSTRAPS = 1000
POSTHOC_AUC_MARGIN = 0.05
TRIAGE_SENSITIVITY_TARGET = 0.85

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TAB = ROOT / "tables"
RES.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

FL_ROOT = ROOT.parents[1] / "clinical_pipeline"
RAW = FL_ROOT / "data" / "raw" / "source_clean.csv"
sys.path.insert(0, str(FL_ROOT / "eval"))
import fl_metrics as M  # noqa: E402


FEATURES_49 = [
    "性别", "年龄", "高血压史", "糖尿病史", "吸烟史", "饮酒史", "BMI", "心率",
    "舒张压", "收缩压", "肌酸激酶", "乳酸脱氢酶", "肌酸激酶同工酶", "血清肌钙蛋白T",
    "NLR", "dNLR", "MLR", "PLR", "SII", "SIRI", "AISI", "TyG",
    "丙氨酸氨基转移酶", "天门冬氨酸氨基转移酶", "谷氨酰转肽酶", "总胆红素", "白蛋白",
    "肌酐", "尿素", "尿酸", "总胆固醇", "甘油三酯", "低密度脂蛋白", "高密度脂蛋白",
    "TC-HDLDL", "血钾", "血氯", "血钠", "血钙", "葡萄糖", "PR间期", "QRS时限",
    "QTc间期", "T Axis", "左室射血分数", "心指数", "左室舒张末内径", "心室收缩容量",
    "室壁运动评分",
]

THESIS16 = [
    "心率", "乳酸脱氢酶", "肌酸激酶同工酶", "血清肌钙蛋白T", "SII", "AISI",
    "白蛋白", "低密度脂蛋白", "高密度脂蛋白", "血氯", "血钠", "葡萄糖",
    "QTc间期", "心室收缩容量", "糖尿病史", "室壁运动评分",
]

CATEGORICAL = ["性别", "高血压史", "糖尿病史", "吸烟史", "饮酒史", "室壁运动评分"]
SUBTYPE = {1: "STEMI", 2: "NSTEMI", 3: "不稳定型心绞痛", 4: "稳定型心绞痛"}

# 仅用于识别明确不可能值，不用于删除统计学离群值。
RANGE = {
    "BMI": (12, 70), "收缩压": (60, 260), "舒张压": (30, 160), "心率": (30, 220),
    "左室射血分数": (10, 85), "心指数": (1.0, 8.0),
    "左室舒张末内径": (25, 90), "心室收缩容量": (5, 400),
    "PR间期": (80, 400), "QRS时限": (40, 250), "QTc间期": (300, 650),
    "T Axis": (-180, 180), "NLR": (0, 200), "dNLR": (0, 200),
    "MLR": (0, 50), "PLR": (0, 2000), "SII": (0, 50000),
    "SIRI": (0, 500), "AISI": (0, 50000), "TyG": (5, 15),
}


MODELS = OrderedDict({
    "Logistic回归": (
        LogisticRegression(max_iter=5000, random_state=SEED),
        {"C": [0.01, 0.1, 1, 10]},
    ),
    "随机森林": (
        RandomForestClassifier(random_state=SEED, n_jobs=1),
        {"n_estimators": [200, 400], "max_depth": [3, 5],
         "min_samples_leaf": [20, 40]},
    ),
    "K近邻": (
        KNeighborsClassifier(n_jobs=1),
        {"n_neighbors": [15, 25, 35], "weights": ["uniform", "distance"]},
    ),
    "梯度提升": (
        GradientBoostingClassifier(random_state=SEED),
        {"n_estimators": [100, 200], "max_depth": [2, 3],
         "learning_rate": [0.05, 0.1], "min_samples_leaf": [20, 40]},
    ),
    "SVM": (
        SVC(probability=True, random_state=SEED),
        {"C": [0.1, 1, 10], "gamma": ["scale", 0.05]},
    ),
    "XGBoost": (
        XGBClassifier(random_state=SEED, eval_metric="logloss", verbosity=0,
                      n_jobs=1, tree_method="hist"),
        {"n_estimators": [200, 400], "max_depth": [2, 3],
         "learning_rate": [0.05, 0.1], "reg_alpha": [0, 1],
         "reg_lambda": [1, 5]},
    ),
    "LightGBM": (
        LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=1),
        {"n_estimators": [200, 400], "max_depth": [2, 3],
         "learning_rate": [0.05, 0.1], "reg_alpha": [0, 1],
         "reg_lambda": [1, 5]},
    ),
    "多层感知机": (
        MLPClassifier(random_state=SEED, max_iter=3000, early_stopping=True),
        {"hidden_layer_sizes": [(20,), (50,), (32, 16)],
         "alpha": [0.5, 1.0, 2.0]},
    ),
})


def discrimination_score(name: str, model, X: pd.DataFrame) -> np.ndarray:
    """返回用于ROC/AUC的连续评分；概率指标仍使用predict_proba。"""
    if name == "SVM":
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def holm_adjust(p_values: list[float]) -> np.ndarray:
    """Holm step-down adjusted P values, preserving input order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def threshold_at_min_sensitivity(y: np.ndarray, p: np.ndarray,
                                 target: float) -> float:
    """在训练集OOF ROC上选择满足目标敏感度时特异度最高的阈值。"""
    fpr, tpr, thresholds = roc_curve(y, p)
    candidates = np.flatnonzero((tpr >= target) & np.isfinite(thresholds))
    if len(candidates) == 0:
        raise RuntimeError(f"无法达到目标敏感度{target:.2f}")
    best_fpr = np.min(fpr[candidates])
    tied = candidates[np.isclose(fpr[candidates], best_fpr)]
    # 并列时取最高阈值，避免无谓增加假阳性。
    return float(np.max(thresholds[tied]))


def paired_auc_difference(y: np.ndarray, p_candidate: np.ndarray,
                          p_reference: np.ndarray, alpha: float = 0.05):
    """配对DeLong AUC差值、标准误、双侧区间与P值。"""
    preds = np.vstack([np.asarray(p_candidate, float),
                       np.asarray(p_reference, float)])
    aucs, cov = M._structural_components(preds, np.asarray(y).astype(int))
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if not np.isfinite(var) or var <= 0:
        if abs(diff) < 1e-12:
            return float(aucs[0]), float(aucs[1]), diff, 0.0, diff, diff, 1.0
        return float(aucs[0]), float(aucs[1]), diff, np.nan, np.nan, np.nan, np.nan
    se = float(np.sqrt(var))
    zcrit = float(stats.norm.ppf(1 - alpha / 2))
    lo, hi = diff - zcrit * se, diff + zcrit * se
    z = diff / se
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return float(aucs[0]), float(aucs[1]), diff, float(z), float(lo), float(hi), p


def prepare_source() -> tuple[pd.DataFrame, list[dict]]:
    df = pd.read_csv(RAW, encoding="utf-8-sig")
    df = df.copy()
    df["Gensini评分"] = pd.to_numeric(df["Gensini评分"], errors="coerce")
    if df["Gensini评分"].isna().any():
        raise ValueError("Gensini评分存在缺失，无法定义主要结局")

    # 修正大论文中非标准dNLR定义：标准dNLR=N/(WBC-N)。
    wbc = pd.to_numeric(df["白细胞计数"], errors="coerce")
    neu = pd.to_numeric(df["中性粒细胞计数"], errors="coerce")
    den = wbc - neu
    df["dNLR"] = np.where((neu >= 0) & (den > 0), neu / den, np.nan)

    # TC-HDLDL实测与TC-HDL-C-LDL-C逐例相等，规范命名为残余胆固醇计算值。
    df["TC-HDLDL"] = (
        pd.to_numeric(df["总胆固醇"], errors="coerce")
        - pd.to_numeric(df["高密度脂蛋白"], errors="coerce")
        - pd.to_numeric(df["低密度脂蛋白"], errors="coerce")
    )

    audit = []
    for f, (lo, hi) in RANGE.items():
        if f not in df:
            continue
        x = pd.to_numeric(df[f], errors="coerce")
        bad = x.notna() & ((x < lo) | (x > hi))
        if bad.any():
            audit.append({"变量": f, "置为缺失数": int(bad.sum()),
                          "规则": f"<{lo}或>{hi}"})
            df.loc[bad, f] = np.nan
    return df, audit


def stratification_labels(df: pd.DataFrame, y: np.ndarray) -> pd.Series:
    subtype = pd.to_numeric(df["冠心病类型"], errors="coerce").fillna(0).astype(int)
    return subtype.astype(str) + "_" + pd.Series(y, index=df.index).astype(str)


def split_indices(df: pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    strata = stratification_labels(df, y)
    idx = np.arange(len(df))
    return train_test_split(idx, train_size=TRAIN_RATIO, random_state=SEED,
                            stratify=strata)


def preprocess(df: pd.DataFrame, idx_tr: np.ndarray, idx_te: np.ndarray):
    X = df[FEATURES_49].apply(pd.to_numeric, errors="coerce").copy()
    miss_train = X.iloc[idx_tr].isna().mean()
    kept = miss_train[miss_train <= 0.15].index.tolist()
    dropped = miss_train[miss_train > 0.15].index.tolist()
    X = X[kept]

    impute, log_features = {}, []
    for f in kept:
        tr = X.iloc[idx_tr][f].dropna()
        if tr.empty:
            raise ValueError(f"训练集变量{f}全缺失")
        if f in CATEGORICAL:
            val = float(tr.mode().iloc[0])
        else:
            val = float(tr.median())
        impute[f] = val
        X[f] = X[f].fillna(val)

        # 仅对非负且训练集明显右偏的连续变量做log1p。
        if f not in CATEGORICAL and float(tr.min()) >= 0 and abs(float(stats.skew(tr))) > 2:
            X[f] = np.log1p(X[f])
            log_features.append(f)

    scaler = StandardScaler().fit(X.iloc[idx_tr])
    Z = pd.DataFrame(scaler.transform(X), columns=kept, index=df.index)
    meta = {
        "features_kept": kept,
        "features_dropped_missing_gt15pct": dropped,
        "missing_rate_train": miss_train.round(6).to_dict(),
        "impute_value": impute,
        "log1p_features": log_features,
        "scaler_mean": dict(zip(kept, scaler.mean_.tolist())),
        "scaler_scale": dict(zip(kept, scaler.scale_.tolist())),
    }
    return Z, meta


def select_lasso_1se(Xtr: pd.DataFrame, ytr: np.ndarray, cv=CV):
    Cs = np.logspace(-3, 3, 100)
    means, sds = [], []
    for c in Cs:
        model = LogisticRegression(C=c, penalty="l1", solver="liblinear",
                                   max_iter=5000, random_state=SEED)
        scores = cross_val_score(model, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=8)
        means.append(float(scores.mean()))
        sds.append(float(scores.std(ddof=1)))
    means, sds = np.asarray(means), np.asarray(sds)
    best = int(np.argmax(means))
    cutoff = means[best] - sds[best] / np.sqrt(cv.get_n_splits())
    eligible = np.where(means >= cutoff)[0]
    chosen = int(eligible[0])  # 最小C=最强正则化

    # 若1-SE点产生空模型，向右移动到第一个有非零系数的点。
    fitted = None
    for j in range(chosen, len(Cs)):
        m = LogisticRegression(C=float(Cs[j]), penalty="l1", solver="liblinear",
                               max_iter=5000, random_state=SEED).fit(Xtr, ytr)
        if np.count_nonzero(m.coef_[0]) > 0:
            chosen, fitted = j, m
            break
    if fitted is None:
        raise RuntimeError("LASSO未选择任何特征")

    coef = pd.Series(fitted.coef_[0], index=Xtr.columns)
    selected = coef[coef != 0].index.tolist()
    path = pd.DataFrame({
        "C": Cs, "lambda": 1 / Cs, "CV_AUC均值": means,
        "CV_AUC标准差": sds, "是否1SE选择": np.arange(len(Cs)) == chosen,
    })
    return selected, coef.loc[selected].sort_values(key=np.abs, ascending=False), path, {
        "C": float(Cs[chosen]), "lambda": float(1 / Cs[chosen]),
        "cv_auc": float(means[chosen]), "best_cv_auc": float(means[best]),
        "one_se_cutoff": float(cutoff),
    }


def selection_stability(Xtr: pd.DataFrame, ytr: np.ndarray, c_value: float,
                        n_boot: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 9)
    hit = np.zeros(Xtr.shape[1], dtype=int)
    coef_abs = np.zeros(Xtr.shape[1], dtype=float)
    for _ in range(n_boot):
        ii = rng.integers(0, len(ytr), len(ytr))
        # 极小概率下单类样本重抽；本队列结局均衡，通常不会发生。
        if np.unique(ytr[ii]).size < 2:
            continue
        m = LogisticRegression(C=c_value, penalty="l1", solver="liblinear",
                               max_iter=5000, random_state=SEED).fit(Xtr.iloc[ii], ytr[ii])
        co = m.coef_[0]
        hit += co != 0
        coef_abs += np.abs(co)
    return (pd.DataFrame({"特征": Xtr.columns, "选择频率": hit / n_boot,
                          "平均绝对系数": coef_abs / n_boot})
            .sort_values(["选择频率", "平均绝对系数"], ascending=False))


def outer_crossfit_training(df: pd.DataFrame, y: np.ndarray,
                            idx_tr: np.ndarray):
    """在开发集内生成诚实的外层折外预测。

    每个外层训练折从原始变量重新估计预处理、LASSO特征集与模型超参数；
    外层验证折在上述步骤中完全不可见。该预测仅用于内部性能描述与阈值
    选择，固定留出测试集始终不参与开发。
    """
    ytr = y[idx_tr]
    strata = stratification_labels(df.iloc[idx_tr].reset_index(drop=True), ytr)
    oof_prob = {name: np.full(len(idx_tr), np.nan) for name in MODELS}
    oof_score = {name: np.full(len(idx_tr), np.nan) for name in MODELS}
    audit_rows = []

    for fold, (fit_pos, val_pos) in enumerate(
            OUTER_CV.split(np.zeros(len(idx_tr)), strata)):
        fit_abs, val_abs = idx_tr[fit_pos], idx_tr[val_pos]
        Z_fold, prep_fold = preprocess(df, fit_abs, val_abs)
        Xfit, Xval = Z_fold.iloc[fit_abs], Z_fold.iloc[val_abs]
        yfit = y[fit_abs]
        selected, _coef, _path, lasso_meta = select_lasso_1se(Xfit, yfit)
        Xfit, Xval = Xfit[selected], Xval[selected]

        for name, (estimator, grid) in MODELS.items():
            t0 = time.time()
            gs = GridSearchCV(clone(estimator), grid, scoring="roc_auc", cv=CV,
                              n_jobs=8, pre_dispatch="2*n_jobs")
            gs.fit(Xfit, yfit)
            best = gs.best_estimator_
            prob = best.predict_proba(Xval)[:, 1]
            score = discrimination_score(name, best, Xval)
            oof_prob[name][val_pos] = prob
            oof_score[name][val_pos] = score
            audit_rows.append({
                "outer_fold": fold,
                "n_outer_train": len(fit_pos),
                "n_outer_valid": len(val_pos),
                "events_outer_train": int(y[fit_abs].sum()),
                "events_outer_valid": int(y[val_abs].sum()),
                "preprocess_kept": len(prep_fold["features_kept"]),
                "preprocess_dropped": "|".join(
                    prep_fold["features_dropped_missing_gt15pct"]),
                "log1p_features": "|".join(prep_fold["log1p_features"]),
                "lasso_C": lasso_meta["C"],
                "lasso_lambda": lasso_meta["lambda"],
                "n_selected": len(selected),
                "selected_features": "|".join(selected),
                "模型": name,
                "最佳参数": json.dumps(gs.best_params_, ensure_ascii=False),
                "内层CV_AUC": float(gs.best_score_),
                "outer_valid_AUC": float(
                    roc_auc_score(y[val_abs], score)),
                "训练耗时秒": round(time.time() - t0, 1),
            })
        print(f"outer cross-fit fold {fold + 1}/{OUTER_FOLDS}: "
              f"train={len(fit_pos)} valid={len(val_pos)} "
              f"selected={len(selected)}", flush=True)

    pred_rows = []
    for name in MODELS:
        if not (np.isfinite(oof_prob[name]).all()
                and np.isfinite(oof_score[name]).all()):
            raise RuntimeError(f"{name}外层交叉拟合预测不完整")
        fold_of_row = np.empty(len(idx_tr), dtype=int)
        for fold, (_fit_pos, val_pos) in enumerate(
                OUTER_CV.split(np.zeros(len(idx_tr)), strata)):
            fold_of_row[val_pos] = fold
        pred_rows.extend({
            "模型": name, "split": "train_outer_oof", "row": int(i),
            "y": int(yy), "p": float(pp), "score": float(ss),
            "outer_fold": int(ff),
        } for i, (yy, pp, ss, ff) in enumerate(zip(
            ytr, oof_prob[name], oof_score[name], fold_of_row)))
    return pd.DataFrame(pred_rows), pd.DataFrame(audit_rows)


def tune_and_evaluate(Xtr: pd.DataFrame, ytr: np.ndarray,
                      Xte: pd.DataFrame, yte: np.ndarray,
                      train_oof: pd.DataFrame):
    fitted, perf_rows, cv_rows, pred_rows = {}, [], [], []
    for name, (estimator, grid) in MODELS.items():
        t0 = time.time()
        gs = GridSearchCV(estimator, grid, scoring="roc_auc", cv=CV, n_jobs=8,
                          pre_dispatch="2*n_jobs", return_train_score=True)
        gs.fit(Xtr, ytr)
        best = gs.best_estimator_
        fitted[name] = best

        # 分类、校准和DCA使用概率；ROC/AUC使用连续区分评分。训练OOF来自
        # 完整管线的外层交叉拟合，而非固定全开发集特征和参数后的重用预测。
        oof = train_oof[train_oof["模型"] == name].sort_values("row")
        if len(oof) != len(ytr) or not np.array_equal(oof.y.to_numpy(), ytr):
            raise RuntimeError(f"{name}外层交叉拟合标签或行数不一致")
        oof_prob = oof.p.to_numpy(float)
        threshold = float(M.youden_threshold(ytr, oof_prob))
        triage_threshold = threshold_at_min_sensitivity(
            ytr, oof_prob, TRIAGE_SENSITIVITY_TARGET)
        p_te = best.predict_proba(Xte)[:, 1]
        score_tr = discrimination_score(name, best, Xtr)
        score_te = discrimination_score(name, best, Xte)
        report = M.full_report(yte, p_te, thr=threshold)
        report["auc"], report["auc_lo"], report["auc_hi"] = M.auc_ci(yte, score_te)
        train_auc = float(roc_auc_score(ytr, score_tr))
        fold_auc = np.asarray([
            roc_auc_score(g.y.to_numpy(), g.score.to_numpy())
            for _fold, g in oof.groupby("outer_fold", sort=True)
        ], dtype=float)
        outer_auc_mean = float(fold_auc.mean())
        outer_auc_sd = float(fold_auc.std(ddof=1))
        triage_train = M.binary_metrics(ytr, oof_prob, triage_threshold)
        triage_test = M.binary_metrics(yte, p_te, triage_threshold)
        report.update({
            "模型": name,
            "训练OuterCV_AUC均值": outer_auc_mean,
            "训练OuterCV_AUC标准差": outer_auc_sd,
            "训练表观AUC": train_auc,
            "OuterCV_测试AUC差": outer_auc_mean - report["auc"],
            "Youden阈值_训练外层OOF": threshold,
            "高敏感度阈值_训练外层OOF": triage_threshold,
            "高敏感度阈值_训练外层OOF敏感度": triage_train["sensitivity"],
            "高敏感度阈值_训练外层OOF特异度": triage_train["specificity"],
            "高敏感度阈值_测试敏感度": triage_test["sensitivity"],
            "高敏感度阈值_测试特异度": triage_test["specificity"],
            "高敏感度阈值_测试TP": triage_test["tp"],
            "高敏感度阈值_测试FP": triage_test["fp"],
            "高敏感度阈值_测试TN": triage_test["tn"],
            "高敏感度阈值_测试FN": triage_test["fn"],
        })
        perf_rows.append(report)
        cv_rows.append({
            "模型": name, "最佳参数": json.dumps(gs.best_params_, ensure_ascii=False),
            "内层CV_AUC": float(gs.best_score_),
            "内层CV_AUC标准差": float(gs.cv_results_["std_test_score"][gs.best_index_]),
            "训练耗时秒": round(time.time() - t0, 1),
        })
        for row in oof.to_dict("records"):
            pred_rows.append({**row, "threshold": threshold,
                              "triage_threshold": triage_threshold})
        pred_rows.extend({
            "模型": name, "split": "test", "row": int(i), "y": int(yy),
            "p": float(pp), "score": float(ss), "outer_fold": -1,
            "threshold": threshold, "triage_threshold": triage_threshold,
        } for i, (yy, pp, ss) in enumerate(zip(yte, p_te, score_te)))
        print(f"{name:<12} CV={gs.best_score_:.4f} test={report['auc']:.4f} "
              f"time={time.time()-t0:.0f}s", flush=True)
    perf = pd.DataFrame(perf_rows).set_index("模型")
    return fitted, perf, pd.DataFrame(cv_rows), pd.DataFrame(pred_rows)


def delong_table(perf: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    ref = predictions[(predictions["模型"] == "Logistic回归") &
                      (predictions["split"] == "test")].sort_values("row")
    rows, raw_p = [], []
    for name in MODELS:
        if name == "Logistic回归":
            continue
        g = predictions[(predictions["模型"] == name) &
                        (predictions["split"] == "test")].sort_values("row")
        auc1, auc0, diff, z, lo, hi, p = paired_auc_difference(
            ref.y.to_numpy(), g.score.to_numpy(), ref.score.to_numpy())
        rows.append({
            "模型": name, "AUC": float(auc1),
            "AUC差值_vs_Logistic": diff,
            "差值95CI下限": lo, "差值95CI上限": hi,
            "差值95CI完全位于事后正负0.05界内": bool(
                lo > -POSTHOC_AUC_MARGIN and hi < POSTHOC_AUC_MARGIN),
            "排除AUC增益超过0.05": bool(hi < POSTHOC_AUC_MARGIN),
            "Z": float(z), "P_未经校正": float(p),
        })
        raw_p.append(float(p))
    adj = holm_adjust(raw_p)
    for row, p in zip(rows, adj):
        row["P_Holm"] = float(p)
        row["Holm校正后显著"] = bool(p < 0.05)
    return pd.DataFrame(rows)


def dca_bootstrap(y: np.ndarray, pred: dict[str, np.ndarray],
                  thresholds: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 21)
    n = len(y)
    rows = []
    for name, p in pred.items():
        for pt in thresholds:
            pos = p >= pt
            nb = ((pos & (y == 1)).sum() / n
                  - (pos & (y == 0)).sum() / n * pt / (1 - pt))
            boot = np.empty(BOOTSTRAPS)
            all_boot = np.empty(BOOTSTRAPS)
            for b in range(BOOTSTRAPS):
                ii = rng.integers(0, n, n)
                yy, pp = y[ii], p[ii]
                po = pp >= pt
                boot[b] = ((po & (yy == 1)).sum() / n
                           - (po & (yy == 0)).sum() / n * pt / (1 - pt))
                prev = yy.mean()
                all_boot[b] = prev - (1 - prev) * pt / (1 - pt)
            prev = y.mean()
            treat_all = prev - (1 - prev) * pt / (1 - pt)
            rows.append({
                "模型": name, "阈值概率": float(pt), "净收益": float(nb),
                "净收益95CI下限": float(np.quantile(boot, 0.025)),
                "净收益95CI上限": float(np.quantile(boot, 0.975)),
                "全部干预": float(treat_all),
                "全部干预95CI下限": float(np.quantile(all_boot, 0.025)),
                "全部干预95CI上限": float(np.quantile(all_boot, 0.975)),
                "均不干预": 0.0,
            })
    return pd.DataFrame(rows)


def dca_paired_differences(y: np.ndarray, pred: dict[str, np.ndarray],
                           thresholds: np.ndarray) -> pd.DataFrame:
    """用同一组bootstrap索引估计模型相对treat-all及Logistic的净收益差。"""
    rng = np.random.default_rng(SEED + 22)
    n = len(y)
    rows = []
    for pt in thresholds:
        boot_idx = rng.integers(0, n, size=(BOOTSTRAPS, n))

        def nb(yy, pp):
            pos = pp >= pt
            return ((pos & (yy == 1)).sum() / len(yy)
                    - (pos & (yy == 0)).sum() / len(yy) * pt / (1 - pt))

        point = {name: nb(y, p) for name, p in pred.items()}
        prev = y.mean()
        all_point = prev - (1 - prev) * pt / (1 - pt)
        boot_model = {name: np.empty(BOOTSTRAPS) for name in pred}
        boot_all = np.empty(BOOTSTRAPS)
        for b, ii in enumerate(boot_idx):
            yy = y[ii]
            prev_b = yy.mean()
            boot_all[b] = prev_b - (1 - prev_b) * pt / (1 - pt)
            for name, p in pred.items():
                boot_model[name][b] = nb(yy, p[ii])
        ref = boot_model["Logistic回归"]
        for name in pred:
            da = boot_model[name] - boot_all
            dl = boot_model[name] - ref
            rows.append({
                "模型": name, "阈值概率": float(pt),
                "净收益差_vs全部干预": float(point[name] - all_point),
                "差值95CI下限_vs全部干预": float(np.quantile(da, 0.025)),
                "差值95CI上限_vs全部干预": float(np.quantile(da, 0.975)),
                "净收益差_vs_Logistic": float(point[name] - point["Logistic回归"]),
                "差值95CI下限_vs_Logistic": float(np.quantile(dl, 0.025)),
                "差值95CI上限_vs_Logistic": float(np.quantile(dl, 0.975)),
            })
    return pd.DataFrame(rows)


def fit_with_locked_params(best_models: dict, Xtr: pd.DataFrame, ytr: np.ndarray,
                           Xte: pd.DataFrame, yte: np.ndarray,
                           scenario: str, extra: dict | None = None) -> list[dict]:
    rows = []
    extra = extra or {}
    for name, best in best_models.items():
        m = clone(best).fit(Xtr, ytr)
        score = discrimination_score(name, m, Xte)
        rows.append({"情景": scenario, "模型": name, "n_test": len(yte),
                     "events_test": int(yte.sum()), "AUC": roc_auc_score(yte, score),
                     **extra})
    return rows


def baseline_table(df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """面向预测模型论文的描述表：含缺失数和SMD，不做基线P值筛选。"""
    rows = []
    for f in FEATURES_49:
        x = pd.to_numeric(df[f], errors="coerce")
        miss = int(x.isna().sum())
        if f in CATEGORICAL:
            # 二分类变量按1的比例；室壁运动评分作为有序变量另按连续值描述。
            if f != "室壁运动评分":
                p0, p1 = x[y == 0].mean(), x[y == 1].mean()
                pbar = (p0 + p1) / 2
                den = np.sqrt(max(pbar * (1 - pbar), 1e-12))
                smd = (p1 - p0) / den
                rows.append({
                    "变量": f, "总体": f"{int((x == 1).sum())} ({x.mean()*100:.1f}%)",
                    "低GS组": f"{int((x[y == 0] == 1).sum())} ({p0*100:.1f}%)",
                    "高GS组": f"{int((x[y == 1] == 1).sum())} ({p1*100:.1f}%)",
                    "缺失数": miss, "SMD": smd,
                })
                continue
        a, b = x[y == 0].dropna(), x[y == 1].dropna()
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        smd = (b.mean() - a.mean()) / pooled if pooled > 0 else np.nan
        fmt = lambda z: f"{z.median():.2f} ({z.quantile(.25):.2f}, {z.quantile(.75):.2f})"
        rows.append({"变量": f, "总体": fmt(x.dropna()), "低GS组": fmt(a),
                     "高GS组": fmt(b), "缺失数": miss, "SMD": smd})

    sub = pd.to_numeric(df["冠心病类型"], errors="coerce")
    for code, label in list(SUBTYPE.items()) + [(0, "分型未知")]:
        mask = sub.isna() if code == 0 else sub.eq(code)
        p0, p1 = mask[y == 0].mean(), mask[y == 1].mean()
        pbar = (p0 + p1) / 2
        den = np.sqrt(max(pbar * (1 - pbar), 1e-12))
        rows.append({
            "变量": f"冠心病分型：{label}",
            "总体": f"{int(mask.sum())} ({mask.mean()*100:.1f}%)",
            "低GS组": f"{int(mask[y == 0].sum())} ({p0*100:.1f}%)",
            "高GS组": f"{int(mask[y == 1].sum())} ({p1*100:.1f}%)",
            "缺失数": 0, "SMD": (p1 - p0) / den,
        })
    return pd.DataFrame(rows)


def logistic_model_table(model: LogisticRegression, Xtr: pd.DataFrame,
                         ytr: np.ndarray, prep: dict,
                         n_boot: int = BOOTSTRAPS) -> pd.DataFrame:
    """输出可复现的最终Logistic公式及固定特征集条件下的bootstrap区间。"""
    names = list(Xtr.columns)
    point = np.r_[float(model.intercept_[0]), model.coef_[0].astype(float)]
    draws = np.empty((n_boot, len(point)), dtype=float)
    rng = np.random.default_rng(SEED + 31)
    completed = 0
    while completed < n_boot:
        ii = rng.integers(0, len(ytr), len(ytr))
        if np.unique(ytr[ii]).size < 2:
            continue
        fitted = clone(model).fit(Xtr.iloc[ii], ytr[ii])
        draws[completed] = np.r_[float(fitted.intercept_[0]), fitted.coef_[0]]
        completed += 1
    lo = np.quantile(draws, 0.025, axis=0)
    hi = np.quantile(draws, 0.975, axis=0)

    rows = [{
        "特征": "截距", "变换": "不适用", "缺失填补值": np.nan,
        "标准化均值": np.nan, "标准化尺度": np.nan,
        "系数": point[0], "系数95CI下限": lo[0], "系数95CI上限": hi[0],
        "标准化OR": np.nan, "OR95CI下限": np.nan, "OR95CI上限": np.nan,
    }]
    log_features = set(prep["log1p_features"])
    for j, name in enumerate(names, start=1):
        rows.append({
            "特征": name,
            "变换": "log1p后标准化" if name in log_features else "原值标准化",
            "缺失填补值": prep["impute_value"][name],
            "标准化均值": prep["scaler_mean"][name],
            "标准化尺度": prep["scaler_scale"][name],
            "系数": point[j], "系数95CI下限": lo[j], "系数95CI上限": hi[j],
            "标准化OR": np.exp(point[j]), "OR95CI下限": np.exp(lo[j]),
            "OR95CI上限": np.exp(hi[j]),
        })
    return pd.DataFrame(rows)


def main():
    source_df, audit = prepare_source()
    excluded = source_df["Gensini评分"].to_numpy() == 0
    analysis_source_rows = np.flatnonzero(~excluded)
    audit.append({"变量": "病例记录", "置为缺失数": int(excluded.sum()),
                  "规则": "主分析排除Gensini评分=0；完整队列作为敏感性分析"})
    df = source_df.loc[~excluded].reset_index(drop=True)
    cutoff = float(df["Gensini评分"].median())
    y = (df["Gensini评分"].to_numpy() > cutoff).astype(int)
    idx_tr, idx_te = split_indices(df, y)
    Z, prep = preprocess(df, idx_tr, idx_te)

    selected, lasso_coef, lasso_path, lasso_meta = select_lasso_1se(
        Z.iloc[idx_tr], y[idx_tr])
    stability = selection_stability(Z.iloc[idx_tr], y[idx_tr], lasso_meta["C"])
    print(f"队列={len(df)} train={len(idx_tr)} test={len(idx_te)} "
          f"events={y.sum()} selected={len(selected)}")
    print("LASSO特征：", selected)

    outer_pred, outer_audit = outer_crossfit_training(df, y, idx_tr)
    fitted, perf, cvtab, predictions = tune_and_evaluate(
        Z.iloc[idx_tr][selected], y[idx_tr], Z.iloc[idx_te][selected], y[idx_te],
        outer_pred)

    logistic_formula = logistic_model_table(
        fitted["Logistic回归"], Z.iloc[idx_tr][selected], y[idx_tr], prep)

    # DeLong + Holm
    delong = delong_table(perf, predictions)

    # DCA及其bootstrap区间
    test_pred = {
        name: predictions[(predictions["模型"] == name) &
                          (predictions["split"] == "test")].sort_values("row").p.to_numpy()
        for name in MODELS
    }
    test_score = {
        name: predictions[(predictions["模型"] == name) &
                          (predictions["split"] == "test")].sort_values("row").score.to_numpy()
        for name in MODELS
    }
    dca_thresholds = np.round(np.arange(0.10, 0.81, 0.05), 2)
    dca = dca_bootstrap(y[idx_te], test_pred, dca_thresholds)
    dca_diff = dca_paired_differences(
        y[idx_te], test_pred, dca_thresholds)

    # 敏感性1：同一划分、同一特征和已锁定超参数，仅改变GS定义。
    sens_rows = []
    cutoffs = {
        "P33": float(df["Gensini评分"].quantile(1 / 3)),
        "P50_主分析": cutoff,
        "均值": float(df["Gensini评分"].mean()),
    }
    for label, cut in cutoffs.items():
        yy = (df["Gensini评分"].to_numpy() > cut).astype(int)
        sens_rows.extend(fit_with_locked_params(
            fitted, Z.iloc[idx_tr][selected], yy[idx_tr],
            Z.iloc[idx_te][selected], yy[idx_te], label,
            {"Gensini界值": cut, "特征集": "主分析训练集LASSO"}))

    # 敏感性2：重新纳入9例GS=0记录，重新分层划分并使用主要分析锁定的指标与参数。
    y_full = (source_df["Gensini评分"].to_numpy() > cutoff).astype(int)
    tr2, te2 = split_indices(source_df, y_full)
    Z_full, _ = preprocess(source_df, tr2, te2)
    sens_rows.extend(fit_with_locked_params(
        fitted, Z_full.iloc[tr2][selected], y_full[tr2],
        Z_full.iloc[te2][selected], y_full[te2],
        "重新纳入Gensini=0", {"Gensini界值": cutoff, "特征集": "主分析训练集LASSO"}))

    # 敏感性3：沿用学位论文固定16项特征，以量化特征集选择对结论的影响。
    t16 = [f for f in THESIS16 if f in Z.columns]
    sens_rows.extend(fit_with_locked_params(
        fitted, Z.iloc[idx_tr][t16], y[idx_tr], Z.iloc[idx_te][t16], y[idx_te],
        "学位论文固定16项", {"Gensini界值": cutoff, "特征集": "学位论文16项"}))

    # 亚组：主模型在非心肌梗死（UA/SAP）测试患者中的区分度，不重新训练。
    subtype = pd.to_numeric(df["冠心病类型"], errors="coerce").to_numpy()
    non_mi = np.isin(subtype[idx_te], [3, 4])
    mi = np.isin(subtype[idx_te], [1, 2])
    subgroup_rows = []
    for group_name, mask in (("非心肌梗死（UA/SAP）", non_mi), ("心肌梗死（STEMI/NSTEMI）", mi)):
        for name, score in test_score.items():
            yy = y[idx_te][mask]
            auc = roc_auc_score(yy, score[mask]) if np.unique(yy).size == 2 else np.nan
            subgroup_rows.append({"亚组": group_name, "模型": name,
                                  "n": int(mask.sum()), "高GS例数": int(yy.sum()),
                                  "AUC": auc})

    # 输出（均为去标识化或汇总结果）
    perf.round(6).to_csv(RES / "表_主分析模型性能.csv", encoding="utf-8-sig")
    cvtab.to_csv(RES / "表_交叉验证与超参数.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(RES / "predictions_deidentified.csv", index=False)
    outer_audit.to_csv(
        RES / "表_5折outer_crossfit审计.csv", index=False, encoding="utf-8-sig")
    delong.round(6).to_csv(RES / "表_DeLong_Holm.csv", index=False, encoding="utf-8-sig")
    dca.round(6).to_csv(RES / "表_DCA_bootstrap.csv", index=False, encoding="utf-8-sig")
    dca_diff.round(6).to_csv(
        RES / "表_DCA配对净收益差.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sens_rows).round(6).to_csv(
        RES / "表_敏感性分析.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(subgroup_rows).round(6).to_csv(
        RES / "表_临床亚组性能.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit).to_csv(RES / "数据清理审计.csv", index=False, encoding="utf-8-sig")
    lasso_path.to_csv(RES / "LASSO_1SE路径.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"特征": lasso_coef.index, "标准化系数": lasso_coef.values}).to_csv(
        RES / "表_LASSO特征.csv", index=False, encoding="utf-8-sig")
    logistic_formula.round(8).to_csv(
        RES / "表_Logistic最终模型.csv", index=False, encoding="utf-8-sig")
    stability.round(6).to_csv(RES / "表_LASSO稳定性.csv", index=False, encoding="utf-8-sig")
    baseline_table(df, y).round({"SMD": 4}).to_csv(
        TAB / "表1_基线特征_SMD.csv", index=False, encoding="utf-8-sig")

    model_bundle = {
        "models": fitted, "selected_features": selected, "all_features": list(Z.columns),
        "X": Z.to_numpy(), "feature_names": list(Z.columns), "y": y,
        "idx_train": idx_tr, "idx_test": idx_te,
        "analysis_source_rows": analysis_source_rows,
    }
    with open(RES / "fitted_models.pkl", "wb") as f:
        pickle.dump(model_bundle, f)

    cohort = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_data_sha256": sha256_file(RAW),
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "n_source": int(len(source_df)),
        "excluded_gensini_eq0": int(excluded.sum()),
        "n_total": int(len(df)), "n_train": int(len(idx_tr)), "n_test": int(len(idx_te)),
        "cutoff": cutoff, "events_total": int(y.sum()),
        "low_gs": int((y == 0).sum()), "high_gs": int((y == 1).sum()),
        "subtype_missing": int(df["冠心病类型"].isna().sum()),
        "gensini_eq0": int(excluded.sum()),
        "p33": float(df["Gensini评分"].quantile(1 / 3)),
        "mean_gensini": float(df["Gensini评分"].mean()),
        "posthoc_auc_margin": POSTHOC_AUC_MARGIN,
        "triage_sensitivity_target": TRIAGE_SENSITIVITY_TARGET,
        "selected_features": selected, "lasso": lasso_meta, "preprocessing": prep,
        "training_oof_method":
            "5-fold outer cross-fitting within development set",
        "outer_folds": OUTER_FOLDS, "outer_seed": SEED + 101,
        "inner_folds": CV.get_n_splits(),
    }
    with open(RES / "analysis_meta.json", "w", encoding="utf-8") as f:
        json.dump(cohort, f, ensure_ascii=False, indent=2)

    print("\n测试集性能：")
    cols = ["auc", "auc_lo", "auc_hi", "sensitivity", "specificity", "f1",
            "brier", "cal_slope", "cal_intercept", "hl_p",
            "OuterCV_测试AUC差"]
    print(perf[cols].sort_values("auc", ascending=False).round(4).to_string())
    print("\nDeLong（Holm校正）：")
    print(delong.round(4).to_string(index=False))
    print(f"\n结果目录：{RES}")


if __name__ == "__main__":
    main()
