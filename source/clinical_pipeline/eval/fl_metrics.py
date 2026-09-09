# -*- coding: utf-8 -*-
"""
临床预测模型评价指标内核 —— 覆盖学位论文报告的全部指标

  区分度  : AUC(+DeLong 95%CI)、准确率、精确率、敏感度、特异度、F1
  统计检验: DeLong 两模型 AUC 比较
  校准度  : Brier、Hosmer-Lemeshow、校准斜率/截距、校准曲线点
  临床价值: 决策曲线净收益 (DCA)
  阈值    : Youden 指数

注意：PFLlib 自带的 AUC 不能用（见项目 README）。本模块一律
      对 logits 做 softmax 取阳性类概率，再算标准二分类指标。
"""
import numpy as np
from scipy import stats
from sklearn import metrics as skm


# --------------------------------------------------------------- 基础
def to_prob(logits):
    """PFLlib 模型输出的是未归一化 logits，转成阳性类概率"""
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim == 1:
        return 1.0 / (1.0 + np.exp(-z))
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True))[:, 1]


def youden_threshold(y, p):
    fpr, tpr, thr = skm.roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def binary_metrics(y, p, thr):
    y = np.asarray(y).astype(int)
    yhat = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = skm.confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    d = lambda a, b: float(a) / b if b else float('nan')
    return {'accuracy': d(tp + tn, tp + tn + fp + fn),
            'precision': d(tp, tp + fp),
            'sensitivity': d(tp, tp + fn),
            'specificity': d(tn, tn + fp),
            'f1': d(2 * tp, 2 * tp + fp + fn),
            'threshold': float(thr),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)}


# --------------------------------------------------------------- DeLong
def _midrank(x):
    """带并列处理的中位秩（Sun & Xu 2014 快速 DeLong 的组件）"""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N, dtype=float)
    out[J] = T
    return out


def _structural_components(preds, y):
    """preds: (k, n) 每行一个模型的分数；返回 AUC 向量与协方差矩阵"""
    y = np.asarray(y).astype(int)
    pos = preds[:, y == 1]
    neg = preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    if m == 0 or n == 0:
        k = preds.shape[0]
        return np.full(k, np.nan), np.full((k, k), np.nan)

    k = preds.shape[0]
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(np.concatenate([pos[r], neg[r]]))

    # Sun & Xu (2014) 快速 DeLong
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    # V01 / V10 结构分量
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s01 = np.cov(v01) if k > 1 else np.array([[np.var(v01[0], ddof=1)]])
    s10 = np.cov(v10) if k > 1 else np.array([[np.var(v10[0], ddof=1)]])
    cov = np.atleast_2d(s01) / m + np.atleast_2d(s10) / n
    return aucs, cov


def auc_ci(y, p, alpha=0.05):
    """AUC 及 DeLong 95% 置信区间"""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return float('nan'), float('nan'), float('nan')
    auc = float(skm.roc_auc_score(y, p))
    _, cov = _structural_components(p[None, :], y)
    se = float(np.sqrt(max(cov[0, 0], 0.0)))
    z = stats.norm.ppf(1 - alpha / 2)
    return auc, float(np.clip(auc - z * se, 0, 1)), float(np.clip(auc + z * se, 0, 1))


def delong_test(y, p1, p2):
    """比较同一批样本上两个模型的 AUC；返回 (auc1, auc2, Z, P)"""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float('nan'), float('nan'), float('nan'), float('nan')
    preds = np.vstack([np.asarray(p1, float), np.asarray(p2, float)])
    _, cov = _structural_components(preds, y)
    a1 = float(skm.roc_auc_score(y, preds[0]))
    a2 = float(skm.roc_auc_score(y, preds[1]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if not np.isfinite(var) or var <= 0:
        # 两组预测完全同秩（含自己跟自己比）-> 差异恒为 0
        if abs(a1 - a2) < 1e-12:
            return a1, a2, 0.0, 1.0
        return a1, a2, float('nan'), float('nan')
    z = (a1 - a2) / np.sqrt(var)
    return a1, a2, float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


# --------------------------------------------------------------- 校准
def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def hosmer_lemeshow(y, p, g=10):
    """返回 (chi2, df, pvalue)；df = g-2，与学位论文表3-12 一致"""
    y = np.asarray(y).astype(float)
    p = np.asarray(p, dtype=float)
    n = len(y)
    order = np.argsort(p)
    y, p = y[order], p[order]
    edges = np.linspace(0, n, g + 1).astype(int)
    chi2 = 0.0
    used = 0
    for i in range(g):
        a, b = edges[i], edges[i + 1]
        if b <= a:
            continue
        obs1 = y[a:b].sum(); exp1 = p[a:b].sum()
        nb = b - a
        obs0 = nb - obs1; exp0 = nb - exp1
        if exp1 > 0:
            chi2 += (obs1 - exp1) ** 2 / exp1
        if exp0 > 0:
            chi2 += (obs0 - exp0) ** 2 / exp0
        used += 1
    df = max(used - 2, 1)
    return float(chi2), int(df), float(1 - stats.chi2.cdf(chi2, df))


def calibration_slope_intercept(y, p, eps=1e-6):
    """把 logit(p) 作为唯一自变量做 logistic 回归。
    理想校准：斜率=1、截距=0；斜率<1 提示预测过于极端。"""
    from sklearn.linear_model import LogisticRegression
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float('nan'), float('nan')
    m = LogisticRegression(C=np.inf, solver='lbfgs', max_iter=1000).fit(lp, y)
    return float(m.coef_[0][0]), float(m.intercept_[0])


def calibration_curve_points(y, p, n_bins=10, strategy='quantile'):
    from sklearn.calibration import calibration_curve
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return np.array([]), np.array([])
    frac, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy=strategy)
    return mean_pred, frac


# --------------------------------------------------------------- DCA
def dca_net_benefit(y, p, thresholds=None):
    """决策曲线分析。返回 dict: pt -> {model, all, none}
    Net Benefit = TP/N - FP/N * pt/(1-pt)，与学位论文 2.5.6 一致"""
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 0.996, 0.01), 3)
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    N = len(y)
    prev = y.mean()
    out = {}
    for pt in thresholds:
        w = pt / (1 - pt) if pt < 1 else np.inf
        yhat = (p >= pt).astype(int)
        tp = int(((yhat == 1) & (y == 1)).sum())
        fp = int(((yhat == 1) & (y == 0)).sum())
        out[float(pt)] = {'model': tp / N - fp / N * w,
                          'all': prev - (1 - prev) * w,
                          'none': 0.0}
    return out


# --------------------------------------------------------------- 汇总
def full_report(y, p, thr=None, hl_groups=10, dca_thresholds=None):
    """一次算齐学位论文口径的所有指标"""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    if thr is None:
        thr = youden_threshold(y, p)
    a, lo, hi = auc_ci(y, p)
    r = binary_metrics(y, p, thr)
    chi2, df, hp = hosmer_lemeshow(y, p, hl_groups)
    slope, icpt = calibration_slope_intercept(y, p)
    r.update({'n': int(len(y)), 'n_pos': int(y.sum()), 'prevalence': float(y.mean()),
              'auc': a, 'auc_lo': lo, 'auc_hi': hi,
              'brier': brier(y, p),
              'hl_chi2': chi2, 'hl_df': df, 'hl_p': hp,
              'cal_slope': slope, 'cal_intercept': icpt})
    if dca_thresholds is not None:
        nb = dca_net_benefit(y, p, dca_thresholds)
        for pt, v in nb.items():
            r[f'nb_{pt:.2f}'] = v['model']
    return r
