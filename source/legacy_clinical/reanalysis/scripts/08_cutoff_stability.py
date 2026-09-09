# -*- coding: utf-8 -*-
"""不同结局界值下的模型排序稳定性。

每个界值先在固定主要划分（seed=42）中确定并锁定模型规格，再使用
相同的200个随机种子（1000--1199）分别按各自结局分层重复开发。
本脚本仅作描述性结局定义敏感性分析，不把三个相关任务解释为
控制单一因素的剂量--反应实验。
"""
import os
import sys
import time
import importlib.util
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, train_test_split

Q3 = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("q3", Q3 / "scripts" / "01_q3_analysis.py")
q3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q3)
import fl_metrics as M

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
src, _ = q3.prepare_source()
df = src[src["Gensini评分"].to_numpy() != 0].reset_index(drop=True)
gs = df["Gensini评分"].to_numpy()

CUTS = {
    # 与01主分析及敏感性分析保持完全一致：这里的“P33”指1/3分位数，
    # 而不是会因线性插值得到22.99的第33百分位数。
    "P33": float(pd.Series(gs).quantile(1 / 3)),
    "均值": float(gs.mean()),
}
TASK_INFO = {
    "P33": {"Gensini界值": CUTS["P33"], "阳性率%": 100 * np.mean(gs > CUTS["P33"])},
    "P50": {"Gensini界值": float(np.median(gs)), "阳性率%": 100 * np.mean(gs > np.median(gs))},
    "均值": {"Gensini界值": CUTS["均值"], "阳性率%": 100 * np.mean(gs > CUTS["均值"])},
}
allrows = []
t0 = time.time()
for tag, cut in CUTS.items():
    y = (gs > cut).astype(int)
    strata = q3.stratification_labels(df, y)
    print(
        f"\n=== {tag}：界值 Gensini>{cut:.2f}，阳性 {y.sum()}/{len(y)} "
        f"({y.mean() * 100:.1f}%) ===",
        flush=True,
    )

    idx_tr0, idx_te0 = q3.split_indices(df, y)
    Z0, _ = q3.preprocess(df, idx_tr0, idx_te0)
    sel0, _c, _p, _i = q3.select_lasso_1se(Z0.iloc[idx_tr0], y[idx_tr0])
    best = {}
    for name, (estimator, grid) in q3.MODELS.items():
        search = GridSearchCV(
            clone(estimator), grid, scoring="roc_auc", cv=q3.CV,
            n_jobs=8, pre_dispatch="2*n_jobs",
        )
        search.fit(Z0.iloc[idx_tr0][sel0], y[idx_tr0])
        best[name] = search.best_params_
    print(f"  主要划分LASSO入选{len(sel0)}项；模型规格已锁定", flush=True)

    for split in range(N):
        idx = np.arange(len(df))
        i_tr, i_te = train_test_split(
            idx,
            train_size=q3.TRAIN_RATIO,
            random_state=1000 + split,
            stratify=strata,
        )
        Z, _ = q3.preprocess(df, i_tr, i_te)
        selected, _c, _p, _i = q3.select_lasso_1se(Z.iloc[i_tr], y[i_tr])
        Xtr, Xte = Z.iloc[i_tr][selected], Z.iloc[i_te][selected]
        for name in q3.MODELS:
            model = clone(q3.MODELS[name][0]).set_params(**best[name]).fit(Xtr, y[i_tr])
            score = q3.discrimination_score(name, model, Xte)
            allrows.append({
                "界值": tag,
                "split": split,
                "模型": name,
                "auc": M.auc_ci(y[i_te], score)[0],
                "n_selected": len(selected),
            })
        if (split + 1) % 25 == 0:
            print(f"    {split + 1}/{N}  累计{(time.time() - t0) / 60:.1f} min", flush=True)

A = pd.DataFrame(allrows)
A["rank"] = A.groupby(["界值", "split"])["auc"].rank(
    ascending=False, method="min"
).astype(int)
A.to_csv(Q3 / "results" / "表_界值稳定性_逐次.csv", index=False, encoding="utf-8-sig")

P50 = pd.read_csv(Q3 / "results" / "表_排序稳定性_逐次.csv", encoding="utf-8-sig")
P50 = P50[["split", "模型", "auc"]].copy()
P50["界值"] = "P50"
P50["rank"] = P50.groupby("split")["auc"].rank(
    ascending=False, method="min"
).astype(int)
F50 = pd.read_csv(Q3 / "results" / "表_特征稳定性_逐次.csv", encoding="utf-8-sig")
P50 = P50.merge(F50[["split", "n_selected"]], on="split", how="left")
A = pd.concat([A, P50], ignore_index=True)

out = []
for tag in ["P33", "P50", "均值"]:
    group = A[A["界值"] == tag]
    pivot = group.pivot_table(index="split", columns="模型", values="auc")
    ids = sorted(pivot.index)
    rhos = np.array([
        spearmanr(pivot.loc[ids[i]], pivot.loc[ids[j]])[0]
        for i in range(len(ids)) for j in range(i + 1, len(ids))
    ])
    mean_auc = group.groupby("模型")["auc"].mean()
    leader = str(mean_auc.idxmax())
    spans = group.groupby("模型")["rank"].agg(lambda x: (x.min(), x.max()))
    out.append({
        "界值": tag,
        "Gensini界值": TASK_INFO[tag]["Gensini界值"],
        "阳性率%": TASK_INFO[tag]["阳性率%"],
        "AUC均值": group["auc"].mean(),
        "AUC极差(模型间)": mean_auc.max() - mean_auc.min(),
        "排序ρ均值": rhos.mean(),
        "ρ<0比例%": (rhos < 0).mean() * 100,
        "平均AUC最高模型": leader,
        "该模型居首比例%": 100 * np.mean(
            (group.loc[group["模型"] == leader, "rank"] == 1).to_numpy()
        ),
        "名次覆盖1-8的模型数": int(sum(1 for value in spans if value == (1, 8))),
        "LASSO入选中位": group.groupby("split")["n_selected"].first().median(),
    })

summary = pd.DataFrame(out)
print(summary.round(4).to_string(index=False))
summary.to_csv(Q3 / "results" / "表_界值稳定性_汇总.csv", index=False, encoding="utf-8-sig")
print(f"\n用时{(time.time() - t0) / 60:.1f} min")
