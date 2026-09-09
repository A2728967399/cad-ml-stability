# -*- coding: utf-8 -*-
"""Q3优化版投稿图表：Python/matplotlib单一后端。"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from clinical_units import display_name


ROOT = Path(__file__).resolve().parents[1]
RES, TAB, FIG = ROOT / "results", ROOT / "tables", ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
_SOURCE_CANDIDATES = [
    ROOT.parent / "clinical_pipeline" / "data" / "raw" / "source_clean.csv",
    ROOT.parents[1] / "clinical_pipeline" / "data" / "raw" / "source_clean.csv",
]
SOURCE = next((path for path in _SOURCE_CANDIDATES if path.exists()),
              _SOURCE_CANDIDATES[0])

SINGLE_W = 3.31   # 84 mm, Springer单栏
DOUBLE_W = 6.85   # 174 mm, Springer双栏
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.2,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
    "savefig.bbox": "tight",
})

COLORS = {
    "Logistic回归": "#1f4e79",
    "随机森林": "#d97706",
    "K近邻": "#9ca3af",
    "梯度提升": "#7c8db5",
    "SVM": "#4f8a8b",
    "XGBoost": "#a16b8a",
    "LightGBM": "#6f9e6e",
    "多层感知机": "#b07d62",
}
MARKERS = {"Logistic回归": "o", "随机森林": "s", "SVM": "^"}
KEY_MODELS = ("Logistic回归", "随机森林", "SVM")


def save_pub(fig, stem: str):
    base = FIG / stem
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(base.with_suffix(".png"), dpi=300)
    plt.close(fig)


def load_all():
    perf = pd.read_csv(RES / "表_主分析模型性能.csv", encoding="utf-8-sig").set_index("模型")
    pred = pd.read_csv(RES / "predictions_deidentified.csv")
    dca = pd.read_csv(RES / "表_DCA_bootstrap.csv", encoding="utf-8-sig")
    sens = pd.read_csv(RES / "表_敏感性分析.csv", encoding="utf-8-sig")
    subgroup = pd.read_csv(RES / "表_临床亚组性能.csv", encoding="utf-8-sig")
    stability = pd.read_csv(RES / "表_LASSO稳定性.csv", encoding="utf-8-sig")
    dl = pd.read_csv(RES / "表_DeLong_Holm.csv", encoding="utf-8-sig")
    meta = json.load(open(RES / "analysis_meta.json", encoding="utf-8"))
    bundle = pickle.load(open(RES / "fitted_models.pkl", "rb"))
    return perf, pred, dca, sens, subgroup, stability, dl, meta, bundle


def panel_label(ax, letter):
    ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontweight="bold",
            fontsize=9, va="top")


def figure1_flow_distribution(meta):
    raw = pd.read_csv(SOURCE, encoding="utf-8-sig")
    gs_series = pd.to_numeric(raw["Gensini评分"], errors="coerce")
    if gs_series.isna().any():
        raise ValueError("图1源数据存在缺失或非数值Gensini评分")
    gs = gs_series[gs_series != 0].to_numpy()
    if len(gs) != meta["n_total"]:
        raise ValueError("图1的Gensini评分样本量与主分析队列不一致")

    fig, axes = plt.subplots(
        1, 2, figsize=(DOUBLE_W, 3.35),
        gridspec_kw={"width_ratios": [0.88, 1.12]},
    )
    ax = axes[0]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def box(y, text, face, h=0.145, bold=False):
        x, w = 0.05, 0.90
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015",
            linewidth=0.8, edgecolor="#425466", facecolor=face))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=7.2, linespacing=1.35, fontweight="bold" if bold else "normal")
        return y, h

    boxes = [
        box(0.81, f"2023年1月—2024年12月\n经冠状动脉造影确诊的冠心病患者\nn = {meta['n_source']:,}", "#eaf1f8", bold=True),
        box(0.60, f"排除既往PCI后复查且本次GS=0的病例\nn = {meta['excluded_gensini_eq0']}", "#fff1f0"),
        box(0.39, f"主分析队列  n = {meta['n_total']:,}\n低GS组 {meta['low_gs']}例；高GS组 {meta['high_gs']}例\n分型缺失{meta['subtype_missing']}例仍保留", "#edf7ed", bold=True),
        box(0.18, f"分层随机划分（75%/25%）\n开发集 n = {meta['n_train']}；留出测试集 n = {meta['n_test']}", "#f4f0fa", bold=True),
    ]
    for (y1, _), (y2, h2) in zip(boxes[:-1], boxes[1:]):
        ax.add_patch(FancyArrowPatch((0.5, y1), (0.5, y2 + h2),
                                     arrowstyle="-|>", mutation_scale=10,
                                     linewidth=0.8, color="#425466"))
    ax.text(0.5, 0.075,
            "开发集：5折外层交叉拟合；每折重做预处理、LASSO和模型参数优化\n合并外层预测定阈值；留出测试集仅评价一次",
            ha="center", va="center", fontsize=6.7, color="#374151")
    panel_label(ax, "a")

    ax = axes[1]
    bins = np.arange(0, np.ceil(gs.max() / 5) * 5 + 5, 5)
    ax.hist(gs, bins=bins, color="#9fbad0", edgecolor="white", linewidth=0.35)
    cutoffs = [
        (meta["p33"], f"1/3分位数：{meta['p33']:.0f}分", "#6b7280", ":"),
        (meta["cutoff"], f"中位数：{meta['cutoff']:.0f}分", COLORS["Logistic回归"], "-"),
        (meta["mean_gensini"], f"均值：{meta['mean_gensini']:.2f}分", COLORS["随机森林"], "--"),
    ]
    for value, label, color, style in cutoffs:
        ax.axvline(value, color=color, ls=style, lw=1.25, label=label)
    ax.set(xlabel="Gensini评分（分）", ylabel="患者数", xlim=(-2, 190))
    ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    ax.legend(loc="upper right", fontsize=6.0)
    ax.text(0.98, 0.73, f"n = {len(gs):,}", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.5, color="#374151")
    panel_label(ax, "b")
    fig.subplots_adjust(wspace=0.32)
    save_pub(fig, "图1_研究流程")


def figure2_discrimination(perf, pred):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.99),
                             gridspec_kw={"width_ratios": [1.12, 0.88]})
    ax = axes[0]
    for name in perf.sort_values("auc", ascending=False).index:
        g = pred[(pred["模型"] == name) & (pred.split == "test")].sort_values("row")
        fpr, tpr, _ = roc_curve(g.y, g.score)
        is_key = name in KEY_MODELS
        ax.plot(fpr, tpr, color=COLORS[name], lw=1.6 if is_key else 0.9,
                alpha=1 if is_key else 0.72,
                label=f"{name}  {perf.loc[name, 'auc']:.3f}")
    ax.plot([0, 1], [0, 1], color="#9ca3af", ls="--", lw=0.8)
    ax.set(xlabel="1-特异度", ylabel="敏感度", xlim=(0, 1), ylim=(0, 1))
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", ncol=1, handlelength=1.8, borderaxespad=0.5)
    panel_label(ax, "a")

    ax = axes[1]
    order = perf.sort_values("auc", ascending=True).index
    yy = np.arange(len(order))
    for i, name in enumerate(order):
        auc, lo, hi = perf.loc[name, ["auc", "auc_lo", "auc_hi"]]
        ax.errorbar(auc, i, xerr=[[auc - lo], [hi - auc]], fmt="o",
                    color=COLORS[name], ecolor=COLORS[name], ms=4, capsize=2,
                    lw=1.1)
        ax.text(hi + 0.004, i, f"{auc:.3f}", va="center", fontsize=6.2)
    ax.axvline(perf.loc["Logistic回归", "auc"], color=COLORS["Logistic回归"],
               ls="--", lw=0.8, alpha=0.75)
    ax.set_yticks(yy, order)
    ax.set_xlabel("测试集AUC（95% CI）")
    ax.set_xlim(0.54, 0.77)
    ax.grid(axis="x", color="#e5e7eb", lw=0.6)
    panel_label(ax, "b")
    fig.subplots_adjust(wspace=0.48)
    save_pub(fig, "图2_测试集区分度")


def figure3_calibration_dca(pred, dca):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.83))
    ax = axes[0]
    for name in KEY_MODELS:
        g = pred[(pred["模型"] == name) & (pred.split == "test")].sort_values("row")
        obs, mean = calibration_curve(g.y, g.p, n_bins=10, strategy="quantile")
        ax.plot(mean, obs, marker=MARKERS[name], ms=3.6, lw=1.4,
                color=COLORS[name], label=name)
    ax.plot([0, 1], [0, 1], color="#6b7280", ls="--", lw=0.8, label="理想校准")
    ax.set(xlabel="预测概率", ylabel="观察比例", xlim=(0, 1), ylim=(0, 1))
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left")
    panel_label(ax, "a")

    ax = axes[1]
    for name in KEY_MODELS:
        g = dca[(dca["模型"] == name) & dca["阈值概率"].between(0.2, 0.6)]
        ax.plot(g["阈值概率"], g["净收益"], color=COLORS[name], lw=1.5,
                label=name)
        ax.fill_between(g["阈值概率"], g["净收益95CI下限"], g["净收益95CI上限"],
                        color=COLORS[name], alpha=0.12, linewidth=0)
    g0 = dca[(dca["模型"] == "Logistic回归") & dca["阈值概率"].between(0.2, 0.6)]
    ax.plot(g0["阈值概率"], g0["全部干预"], color="#6b7280", ls="--", lw=1,
            label="全部干预")
    ax.axhline(0, color="#111827", ls=":", lw=0.9, label="均不干预")
    ax.set(xlabel="阈值概率", ylabel="净收益", xlim=(0.2, 0.6))
    ax.legend(loc="upper right")
    panel_label(ax, "b")
    fig.subplots_adjust(wspace=0.34)
    save_pub(fig, "图3_校准与决策曲线")


def subgroup_bootstrap(pred, bundle):
    raw = pd.read_csv(SOURCE, encoding="utf-8-sig")
    source_rows = np.asarray(bundle["analysis_source_rows"])
    subtype = pd.to_numeric(
        raw.iloc[source_rows]["冠心病类型"], errors="coerce").to_numpy()
    idx_te = np.asarray(bundle["idx_test"])
    y = np.asarray(bundle["y"])[idx_te]
    groups = {
        "非心肌梗死\n（UA/SAP）": np.isin(subtype[idx_te], [3, 4]),
        "心肌梗死\n（STEMI/NSTEMI）": np.isin(subtype[idx_te], [1, 2]),
    }
    rng = np.random.default_rng(2025)  # 对实测测试集做bootstrap重抽样，不生成模拟数据
    rows = []
    for group, mask in groups.items():
        for name in KEY_MODELS:
            p = pred[(pred["模型"] == name) & (pred.split == "test")].sort_values("row").score.to_numpy()[mask]
            yy = y[mask]
            auc = roc_auc_score(yy, p)
            vals = []
            for _ in range(2000):
                ii = rng.integers(0, len(yy), len(yy))
                if np.unique(yy[ii]).size == 2:
                    vals.append(roc_auc_score(yy[ii], p[ii]))
            rows.append({"亚组": group, "模型": name, "n": len(yy), "AUC": auc,
                         "下限": np.quantile(vals, .025), "上限": np.quantile(vals, .975)})
    return pd.DataFrame(rows)


def figure4_robustness(sens, stability, pred, bundle, meta):
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_W, 2.99),
                             gridspec_kw={"width_ratios": [1.05, 0.88, 1.15]})
    ax = axes[0]
    order_x = ["P33", "P50_主分析", "均值"]
    labels = [f"1/3分位数\n{meta['p33']:.0f}分", f"中位数\n{meta['cutoff']:.0f}分",
              f"均值\n{meta['mean_gensini']:.2f}分"]
    for name in COLORS:
        g = sens[(sens["模型"] == name) & sens["情景"].isin(order_x)].copy()
        g["ord"] = pd.Categorical(g["情景"], categories=order_x, ordered=True)
        g = g.sort_values("ord")
        key = name in KEY_MODELS
        ax.plot(range(3), g.AUC, marker="o" if key else None,
                color=COLORS[name], lw=1.5 if key else 0.8,
                alpha=1 if key else 0.60, ms=3, label=name)
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("测试集AUC")
    ax.set_ylim(0.60, 0.76)
    ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    ax.legend(loc="lower left", ncol=1, fontsize=5.6)
    panel_label(ax, "a")

    ax = axes[1]
    sub = subgroup_bootstrap(pred, bundle)
    sub.to_csv(RES / "表_亚组AUC_bootstrap_CI.csv", index=False, encoding="utf-8-sig")
    ypos = {"非心肌梗死\n（UA/SAP）": 1, "心肌梗死\n（STEMI/NSTEMI）": 0}
    for name, offset in (("Logistic回归", -0.12), ("随机森林", 0.0), ("SVM", 0.12)):
        g = sub[sub.模型 == name]
        for row in g.itertuples():
            y0 = ypos[row.亚组] + offset
            ax.errorbar(row.AUC, y0,
                        xerr=[[row.AUC - row.下限], [row.上限 - row.AUC]],
                        fmt=MARKERS[name], ms=3.8, capsize=2, lw=1,
                        color=COLORS[name], label=name if row.亚组.startswith("非") else None)
    ax.axvline(0.5, color="#9ca3af", ls="--", lw=0.8)
    counts = sub.groupby("亚组")["n"].first().to_dict()
    mi_key = "心肌梗死\n（STEMI/NSTEMI）"
    non_mi_key = "非心肌梗死\n（UA/SAP）"
    ax.set_yticks([0, 1], [
        f"心肌梗死\n(n={counts[mi_key]})",
        f"非心肌梗死\n(n={counts[non_mi_key]})",
    ])
    ax.set_xlim(0.40, 0.82)
    ax.set_xlabel("亚组AUC（bootstrap 95% CI）")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              borderaxespad=0)
    panel_label(ax, "b")

    ax = axes[2]
    top = stability.head(12).sort_values("选择频率")
    colors = ["#5b8db8" if v >= 0.70 else "#a8bfd2" for v in top["选择频率"]]
    ax.barh(top["特征"].map(display_name), top["选择频率"],
            color=colors, height=0.68)
    ax.axvline(0.50, color="#6b7280", ls="--", lw=0.8)
    ax.set(xlabel="Bootstrap选择频率", xlim=(0, 1.02))
    ax.grid(axis="x", color="#e5e7eb", lw=0.6)
    panel_label(ax, "c")
    fig.subplots_adjust(wspace=0.55)
    save_pub(fig, "图4_稳健性与特征稳定性")


def supplementary_all_calibration(pred):
    names = list(COLORS)
    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE_W, 3.46), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), names):
        g = pred[(pred["模型"] == name) & (pred.split == "test")].sort_values("row")
        obs, mean = calibration_curve(g.y, g.p, n_bins=10, strategy="quantile")
        ax.plot(mean, obs, marker="o", ms=2.5, lw=1.1, color=COLORS[name])
        ax.plot([0, 1], [0, 1], color="#9ca3af", ls="--", lw=0.7)
        ax.set_title(name, pad=2)
        ax.set(xlim=(0, 1), ylim=(0, 1))
    for ax in axes[:, 0]: ax.set_ylabel("观察比例")
    for ax in axes[-1, :]: ax.set_xlabel("预测概率")
    fig.subplots_adjust(wspace=0.28, hspace=0.34)
    save_pub(fig, "图S1_全部模型校准")


def supplementary_all_dca(dca):
    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.68))
    for name in COLORS:
        g = dca[(dca["模型"] == name) & dca["阈值概率"].between(0.2, 0.7)]
        ax.plot(g["阈值概率"], g["净收益"], color=COLORS[name], lw=1,
                alpha=0.86, label=name)
    g0 = dca[(dca["模型"] == "Logistic回归") & dca["阈值概率"].between(0.2, 0.7)]
    ax.plot(g0["阈值概率"], g0["全部干预"], color="#4b5563", ls="--", lw=1,
            label="全部干预")
    ax.axhline(0, color="#111827", ls=":", lw=0.9, label="均不干预")
    ax.set(xlabel="阈值概率", ylabel="净收益", xlim=(0.2, 0.7))
    ax.legend(ncol=2, loc="upper right", fontsize=5.4)
    save_pub(fig, "图S2_全部模型DCA")


def supplementary_lasso_diagnostics(bundle, meta):
    path = pd.read_csv(RES / "LASSO_1SE路径.csv", encoding="utf-8-sig")
    if (path["lambda"] <= 0).any():
        raise ValueError("LASSO路径中的lambda必须严格大于0才能取对数")
    X = np.asarray(bundle["X"])
    names = list(bundle["feature_names"])
    idx_tr = np.asarray(bundle["idx_train"])
    y = np.asarray(bundle["y"])[idx_tr]
    Xtr = X[idx_tr]

    coef_path = []
    for c in path["C"].to_numpy(float):
        with warnings.catch_warnings():
            # sklearn 1.8开始迁移penalty接口；保持与主分析拟合规格完全一致。
            warnings.filterwarnings("ignore", message="'penalty' was deprecated.*")
            warnings.filterwarnings("ignore", message="Inconsistent values: penalty=l1.*")
            model = LogisticRegression(
                C=float(c), penalty="l1", solver="liblinear",
                max_iter=5000, random_state=2025,
            ).fit(Xtr, y)
        coef_path.append(model.coef_[0])
    coef_path = np.asarray(coef_path)
    coef_source = pd.DataFrame(coef_path, columns=names)
    coef_source.insert(0, "log10_lambda", np.log10(path["lambda"].to_numpy(float)))
    coef_source.insert(0, "lambda", path["lambda"].to_numpy(float))
    coef_source.to_csv(RES / "LASSO_系数路径.csv", index=False, encoding="utf-8-sig")

    selected = pd.read_csv(RES / "表_LASSO特征.csv", encoding="utf-8-sig")
    focus = selected.reindex(selected["标准化系数"].abs().sort_values(ascending=False).index)
    focus = focus.head(5)["特征"].tolist()
    focus_colors = ["#1f4e79", "#d97706", "#4f8a8b", "#a16b8a", "#6f9e6e"]
    x = np.log10(path["lambda"].to_numpy(float))
    chosen = int(np.flatnonzero(path["是否1SE选择"].astype(str).str.lower().eq("true"))[0])
    best = int(path["CV_AUC均值"].idxmax())

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.92),
                             gridspec_kw={"width_ratios": [1.08, 0.92]})
    ax = axes[0]
    for j, name in enumerate(names):
        if name not in focus:
            ax.plot(x, coef_path[:, j], color="#cbd5e1", lw=0.45, alpha=0.75)
    for name, color in zip(focus, focus_colors):
        j = names.index(name)
        ax.plot(x, coef_path[:, j], color=color, lw=1.25,
                label=display_name(name))
    ax.axvline(x[chosen], color="#111827", ls="--", lw=0.9,
               label="1-SE选择点")
    ax.axhline(0, color="#9ca3af", lw=0.6)
    ax.set(xlabel=r"$\log_{10}(\lambda)$", ylabel="标准化系数")
    ax.legend(loc="upper right", fontsize=5.6, ncol=2)
    panel_label(ax, "a")

    ax = axes[1]
    mean = path["CV_AUC均值"].to_numpy(float)
    se = path["CV_AUC标准差"].to_numpy(float) / np.sqrt(5)
    ax.plot(x, mean, color=COLORS["Logistic回归"], lw=1.35,
            label="5折CV平均AUC")
    ax.fill_between(x, mean - se, mean + se, color=COLORS["Logistic回归"],
                    alpha=0.16, linewidth=0, label="±1 SE")
    ax.axvline(x[best], color="#6b7280", ls=":", lw=1.0,
               label="最高平均AUC")
    ax.axvline(x[chosen], color=COLORS["随机森林"], ls="--", lw=1.1,
               label="1-SE选择点")
    ax.axhline(float(meta["lasso"]["one_se_cutoff"]), color="#9ca3af",
               ls="-.", lw=0.8, label="1-SE阈值")
    ax.scatter(x[chosen], mean[chosen], s=18, color=COLORS["随机森林"], zorder=3)
    ax.set(xlabel=r"$\log_{10}(\lambda)$", ylabel="开发集交叉验证AUC",
           ylim=(0.48, 0.71))
    ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    ax.legend(loc="lower left", fontsize=5.6)
    panel_label(ax, "b")
    fig.subplots_adjust(wspace=0.36)
    save_pub(fig, "图S4_LASSO筛选诊断")


def model_explanation_figure(bundle, perf):
    import shap

    logistic = pd.read_csv(RES / "表_Logistic最终模型.csv", encoding="utf-8-sig")
    coef = logistic[logistic["特征"] != "截距"].copy()
    coef["绝对系数"] = coef["系数"].abs()
    coef = coef.sort_values("绝对系数", ascending=True)

    selected = bundle["selected_features"]
    all_names = bundle["feature_names"]
    jj = [all_names.index(f) for f in selected]
    X = np.asarray(bundle["X"])
    Xte = X[np.asarray(bundle["idx_test"])][:, jj]
    # 以随机森林代表非线性集成模型作探索性解释；不据此认定其为最优模型或作因果推断。
    model = bundle["models"]["随机森林"]
    sv = shap.TreeExplainer(model).shap_values(Xte)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    imp = pd.DataFrame({"特征": selected, "平均绝对SHAP值": np.abs(sv).mean(axis=0)})
    imp = imp.sort_values("平均绝对SHAP值", ascending=False)
    imp.to_csv(RES / "表_SHAP_随机森林.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(
        1, 2, figsize=(DOUBLE_W, 4.02),
        gridspec_kw={"width_ratios": [1.08, 0.92]},
    )
    ax = axes[0]
    ypos = np.arange(len(coef))
    colors = np.where(coef["系数"] >= 0, COLORS["Logistic回归"], "#b07d62")
    xerr = np.vstack([
        coef["系数"].to_numpy() - coef["系数95CI下限"].to_numpy(),
        coef["系数95CI上限"].to_numpy() - coef["系数"].to_numpy(),
    ])
    for i, (_, row) in enumerate(coef.iterrows()):
        ax.errorbar(row["系数"], i, xerr=xerr[:, i:i+1], fmt="o",
                    color=colors[i], ecolor=colors[i], ms=3.5,
                    capsize=1.8, lw=0.9)
    ax.axvline(0, color="#6b7280", ls="--", lw=0.8)
    ax.set_yticks(ypos, coef["特征"].map(display_name))
    ax.set_xlabel("标准化回归系数 β（bootstrap 95% CI）")
    ax.grid(axis="x", color="#e5e7eb", lw=0.6)
    panel_label(ax, "a")

    top = imp.head(12).sort_values("平均绝对SHAP值")
    ax = axes[1]
    ax.barh(top["特征"].map(display_name), top["平均绝对SHAP值"],
            color="#5b8db8", height=0.68)
    ax.set_xlabel("平均 |SHAP值|")
    ax.grid(axis="x", color="#e5e7eb", lw=0.6)
    panel_label(ax, "b")
    fig.subplots_adjust(wspace=0.62)
    save_pub(fig, "图5_模型解释")

    shap.summary_plot(sv, Xte, feature_names=[display_name(x) for x in selected],
                      max_display=12, show=False,
                      plot_size=(DOUBLE_W, 3.40), color_bar_label="指标取值")
    fig = plt.gcf()
    for ax in fig.axes:
        ax.tick_params(labelsize=6.5)
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
        for t in ax.texts:
            t.set_fontsize(6.5)
    fig.axes[0].set_xlabel("SHAP值（对模型输出的影响）", fontsize=7)
    fig.savefig(FIG / "图S3_随机森林SHAP蜂群.pdf")
    fig.savefig(FIG / "图S3_随机森林SHAP蜂群.svg")
    fig.savefig(FIG / "图S3_随机森林SHAP蜂群.tiff", dpi=600,
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(FIG / "图S3_随机森林SHAP蜂群.png", dpi=300)
    plt.close(fig)


def publication_tables(perf, dl, sens, meta):
    baseline = pd.read_csv(TAB / "表1_基线特征_SMD.csv", encoding="utf-8-sig")
    keep = ["年龄", "性别", "BMI", "高血压史", "糖尿病史", "吸烟史"]
    keep += meta["selected_features"]
    keep += [f"冠心病分型：{x}" for x in ("STEMI", "NSTEMI", "不稳定型心绞痛",
                                          "稳定型心绞痛", "分型未知")]
    seen, ordered = set(), []
    for x in keep:
        if x not in seen:
            ordered.append(x); seen.add(x)
    compact = baseline[baseline["变量"].isin(ordered)].copy()
    compact["_ord"] = compact["变量"].map({x: i for i, x in enumerate(ordered)})
    compact = compact.sort_values("_ord").drop(columns="_ord")
    compact.to_csv(TAB / "表1_主要基线特征.csv", index=False, encoding="utf-8-sig")

    pmap = dl.set_index("模型")["P_Holm"].to_dict()
    dlmap = dl.set_index("模型")
    rows = []
    for name, r in perf.sort_values("auc", ascending=False).iterrows():
        if name == "Logistic回归":
            delta_ci = "参照"
        else:
            d = dlmap.loc[name]
            delta_ci = (f"{d['AUC差值_vs_Logistic']:+.3f}"
                        f"（{d['差值95CI下限']:+.3f}–{d['差值95CI上限']:+.3f}）")
        rows.append({
            "模型": name,
            "AUC（95% CI）": f"{r.auc:.3f}（{r.auc_lo:.3f}–{r.auc_hi:.3f}）",
            "ΔAUC vs Logistic（95% CI）": delta_ci,
            "敏感度": round(r.sensitivity, 3), "特异度": round(r.specificity, 3),
            "精确率": round(r.precision, 3), "F1": round(r.f1, 3),
            "Brier分数": round(r.brier, 4), "校准斜率": round(r.cal_slope, 3),
            "校准截距": round(r.cal_intercept, 3),
            "训练外层OOF阈值": round(r["Youden阈值_训练外层OOF"], 3),
            "P_Holm（vs Logistic）": "参照" if name == "Logistic回归" else f"{pmap[name]:.3f}",
        })
    pd.DataFrame(rows).to_csv(TAB / "表2_模型性能与校准.csv", index=False,
                              encoding="utf-8-sig")

    piv = sens.pivot_table(index="模型", columns="情景", values="AUC")
    piv = piv[["P33", "P50_主分析", "均值", "重新纳入Gensini=0", "学位论文固定16项"]]
    piv.round(3).to_csv(TAB / "表3_敏感性分析.csv", encoding="utf-8-sig")


def main():
    perf, pred, dca, sens, subgroup, stability, dl, meta, bundle = load_all()
    publication_tables(perf, dl, sens, meta)
    figure1_flow_distribution(meta)
    figure2_discrimination(perf, pred)
    figure3_calibration_dca(pred, dca)
    figure4_robustness(sens, stability, pred, bundle, meta)
    model_explanation_figure(bundle, perf)
    supplementary_all_calibration(pred)
    supplementary_all_dca(dca)
    supplementary_lasso_diagnostics(bundle, meta)
    print("投稿图表已生成")


if __name__ == "__main__":
    main()
