# -*- coding: utf-8 -*-
"""排序稳定性分析：重复随机划分下，模型名次与LASSO特征集是否可复现？

设计
  · 队列、结局定义（Gensini>37）与主分析完全一致，不随划分变化
  · 每次划分内部完整重跑：填补 -> log1p -> 标准化 -> LASSO(1-SE) 特征选择
    （全部限定训练集，与主分析同一函数）
  · 为隔离“随机划分本身”造成的变异，超参数锁定为主分析选中的值；
    这些参数来自同一数据集的一次主划分，故本分析只作描述性固定方案
    敏感性评价，不作P值/置信区间推断，也不解释为无偏泛化验证
  · 记录每次划分中 8 个模型的测试集 AUC、名次，以及被选中的特征
"""
import os, sys, json, time, importlib.util, itertools, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import clone

Q3 = str(Path(__file__).resolve().parents[1])
spec = importlib.util.spec_from_file_location('q3', f'{Q3}/scripts/01_q3_analysis.py')
q3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q3)                      # 有 __main__ 保护，不会执行 main()
import fl_metrics as M

N_SPLITS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OUT = f'{Q3}/results'

# ---- 队列与结局：与主分析一致 ----
src, _ = q3.prepare_source()
df = src[src['Gensini评分'].to_numpy() != 0].reset_index(drop=True)
cutoff = float(df['Gensini评分'].median())
y = (df['Gensini评分'].to_numpy() > cutoff).astype(int)
subtype = pd.to_numeric(df['冠心病类型'], errors='coerce').fillna(0).astype(int)
strata = subtype.astype(str) + '_' + pd.Series(y).astype(str)
print(f'队列 {len(df)} 例，界值 Gensini>{cutoff:.0f}，阳性 {y.sum()} 例')

# ---- 锁定主分析选中的超参数 ----
cvt = pd.read_csv(f'{OUT}/表_交叉验证与超参数.csv', encoding='utf-8-sig')
BEST = {r['模型']: json.loads(r['最佳参数']) for _, r in cvt.iterrows()}
for k, v in BEST.items():
    if 'hidden_layer_sizes' in v:
        v['hidden_layer_sizes'] = tuple(v['hidden_layer_sizes'])
NAMES = list(q3.MODELS)
print(f'锁定超参数：{len(BEST)} 个模型\n')

auc_rows, feat_rows = [], []
t0 = time.time()
for s in range(N_SPLITS):
    idx = np.arange(len(df))
    idx_tr, idx_te = train_test_split(idx, train_size=q3.TRAIN_RATIO,
                                      random_state=1000 + s, stratify=strata)
    Z, _meta = q3.preprocess(df, idx_tr, idx_te)
    Xtr, Xte = Z.iloc[idx_tr], Z.iloc[idx_te]
    ytr, yte = y[idx_tr], y[idx_te]

    sel, _coef, _path, _info = q3.select_lasso_1se(Xtr, ytr)
    feat_rows.append({'split': s, 'n_selected': len(sel), 'features': '|'.join(sel)})

    Xtr_s, Xte_s = Xtr[sel], Xte[sel]
    for nm in NAMES:
        est = clone(q3.MODELS[nm][0]).set_params(**BEST[nm])
        est.fit(Xtr_s, ytr)
        p = est.predict_proba(Xte_s)[:, 1]
        score = q3.discrimination_score(nm, est, Xte_s)
        auc_rows.append({'split': s, '模型': nm,
                         'auc': M.auc_ci(yte, score)[0],
                         'cal_slope': M.calibration_slope_intercept(yte, p)[0],
                         'brier': M.brier(yte, p)})
    if (s + 1) % 10 == 0:
        el = time.time() - t0
        print(f'  {s+1}/{N_SPLITS}  已用 {el/60:.1f} min，预计还需 '
              f'{el/(s+1)*(N_SPLITS-s-1)/60:.1f} min', flush=True)

A = pd.DataFrame(auc_rows)
F = pd.DataFrame(feat_rows)
A['rank'] = A.groupby('split')['auc'].rank(ascending=False, method='min').astype(int)
A.to_csv(f'{OUT}/表_排序稳定性_逐次.csv', index=False, encoding='utf-8-sig')
F.to_csv(f'{OUT}/表_特征稳定性_逐次.csv', index=False, encoding='utf-8-sig')

print('\n' + '=' * 84)
print(f'模型名次分布（{N_SPLITS} 次随机划分，超参数锁定）')
print('=' * 84)
S = A.groupby('模型').agg(
    AUC均值=('auc', 'mean'), AUC标准差=('auc', 'std'),
    AUC范围=('auc', lambda x: f'{x.min():.3f}–{x.max():.3f}'),
    名次中位=('rank', 'median'),
    名次IQR=('rank', lambda x: f'{x.quantile(.25):.0f}–{x.quantile(.75):.0f}'),
    名次范围=('rank', lambda x: f'{x.min()}–{x.max()}'),
    第一名比例=('rank', lambda x: f'{(x == 1).mean()*100:.1f}%'),
    前三比例=('rank', lambda x: f'{(x <= 3).mean()*100:.1f}%'))
print(S.sort_values('名次中位').round(4).to_string())

print('\n' + '=' * 84)
print('任意两次划分之间，8 个模型名次的一致性')
print('=' * 84)
from scipy.stats import spearmanr
piv = A.pivot_table(index='模型', columns='split', values='auc')
rs = np.array([
    spearmanr(piv[i], piv[j]).statistic
    for i, j in itertools.combinations(piv.columns, 2)
])
print(f'  枚举全部 {len(rs):,} 个划分对')
print(f'  Spearman rho：均值 {rs.mean():.3f}   中位 {np.median(rs):.3f}   '
      f'四分位 {np.quantile(rs,.25):.3f}–{np.quantile(rs,.75):.3f}')
print(f'  rho < 0 的比例：{(rs < 0).mean()*100:.1f}%   '
      f'rho > 0.5 的比例：{(rs > 0.5).mean()*100:.1f}%')

print('\n' + '=' * 84)
print('LASSO 特征集的稳定性')
print('=' * 84)
print(f'  入选特征个数：中位 {F.n_selected.median():.0f}，'
      f'四分位 {F.n_selected.quantile(.25):.0f}–{F.n_selected.quantile(.75):.0f}，'
      f'范围 {F.n_selected.min()}–{F.n_selected.max()}')
cnt = {}
for s_ in F.features:
    for f_ in (s_.split('|') if s_ else []):
        cnt[f_] = cnt.get(f_, 0) + 1
C = pd.Series(cnt).sort_values(ascending=False) / len(F) * 100
print(f'  出现在全部 {N_SPLITS} 次中的比例：')
print('   ', '; '.join(f'{k} {v:.0f}%' for k, v in C.head(12).items()))
print(f'  选择频率 >=80% 的特征数：{(C >= 80).sum()}；'
      f'>=50%：{(C >= 50).sum()}；曾被选中过的特征总数：{len(C)}')
C.round(1).rename('选择频率%').to_csv(f'{OUT}/表_特征选择频率.csv', encoding='utf-8-sig')
print(f'\n用时 {(time.time()-t0)/60:.1f} min；结果已写入 {OUT}')
