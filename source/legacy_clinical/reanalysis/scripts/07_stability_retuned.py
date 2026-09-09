# -*- coding: utf-8 -*-
"""补充稳定性分析：每次划分内重新调参（完整建模策略）

与 05_ranking_stability.py 的唯一差别是超参数不再锁定，而是在每次划分的
训练集内用分层 5 折交叉验证重新网格搜索。使用相同的随机种子序列
（1000+s），因此两组结果在同一批划分上可直接配对比较，用以回答：
"锁定超参数是否低估了划分不稳定性？"
"""
import os, sys, json, time, importlib.util, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import clone

Q3 = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('q3', Q3 / 'scripts' / '01_q3_analysis.py')
q3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(q3)
import fl_metrics as M

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OUT = Q3 / 'results'

src, _ = q3.prepare_source()
df = src[src['Gensini评分'].to_numpy() != 0].reset_index(drop=True)
cutoff = float(df['Gensini评分'].median())
y = (df['Gensini评分'].to_numpy() > cutoff).astype(int)
strata = q3.stratification_labels(df, y)
print(f'队列 {len(df)}，界值 >{cutoff:.0f}，阳性 {y.sum()}；{N} 次划分，每次重新调参\n')

rows, hp_rows, feat_rows = [], [], []
t0 = time.time()
for s in range(N):
    idx = np.arange(len(df))
    idx_tr, idx_te = train_test_split(idx, train_size=q3.TRAIN_RATIO,
                                      random_state=1000 + s, stratify=strata)
    Z, _ = q3.preprocess(df, idx_tr, idx_te)
    Xtr, Xte = Z.iloc[idx_tr], Z.iloc[idx_te]
    ytr, yte = y[idx_tr], y[idx_te]
    sel, _c, _p, _i = q3.select_lasso_1se(Xtr, ytr)
    feat_rows.append({'split': s, 'n_selected': len(sel), 'features': '|'.join(sel)})
    Xtr_s, Xte_s = Xtr[sel], Xte[sel]

    for nm, (est, grid) in q3.MODELS.items():
        gs = GridSearchCV(clone(est), grid, scoring='roc_auc', cv=q3.CV,
                          n_jobs=8, pre_dispatch='2*n_jobs')
        gs.fit(Xtr_s, ytr)
        m = gs.best_estimator_
        p = q3.discrimination_score(nm, m, Xte_s)
        rows.append({'split': s, '模型': nm, 'auc': M.auc_ci(yte, p)[0]})
        hp_rows.append({'split': s, '模型': nm,
                        '最佳参数': json.dumps(gs.best_params_, ensure_ascii=False,
                                            default=str)})
    if (s + 1) % 5 == 0:
        el = time.time() - t0
        print(f'  {s+1}/{N}  已用 {el/60:.1f} min，预计还需 '
              f'{el/(s+1)*(N-s-1)/60:.1f} min', flush=True)

A = pd.DataFrame(rows)
A['rank'] = A.groupby('split')['auc'].rank(ascending=False, method='min').astype(int)
A.to_csv(OUT / '表_排序稳定性_重新调参_逐次.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(hp_rows).to_csv(OUT / '表_排序稳定性_重新调参_超参数.csv',
                             index=False, encoding='utf-8-sig')
pd.DataFrame(feat_rows).to_csv(OUT / '表_排序稳定性_重新调参_特征.csv',
                               index=False, encoding='utf-8-sig')

from scipy.stats import spearmanr
def summarize(D, tag):
    piv = D.pivot_table(index='split', columns='模型', values='auc')
    sp = D.groupby('模型').agg(
        AUC均值=('auc', 'mean'), 名次中位=('rank', 'median'),
        名次IQR=('rank', lambda x: f'{x.quantile(.25):.0f}–{x.quantile(.75):.0f}'),
        名次范围=('rank', lambda x: f'{x.min()}–{x.max()}'),
        居首比例=('rank', lambda x: f'{(x == 1).mean()*100:.1f}%'))
    rs = []
    ids = sorted(piv.index)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            rs.append(spearmanr(piv.loc[ids[i]], piv.loc[ids[j]])[0])
    rs = np.array(rs)
    d = piv['随机森林'] - piv['Logistic回归']
    print('\n' + '=' * 80)
    print(f'{tag}（{len(ids)} 次划分，{len(rs)} 个划分对）')
    print('=' * 80)
    print(sp.sort_values('名次中位').round(4).to_string())
    print(f'\n  划分对排序一致性 Spearman ρ：均值 {rs.mean():.3f}，中位 {np.median(rs):.3f}，'
          f'负相关比例 {(rs<0).mean()*100:.1f}%')
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f'  随机森林 vs Logistic 配对 ΔAUC：均值 {d.mean():+.4f}，'
          f'胜出 {(d>0).mean()*100:.1f}%，逐次 2.5–97.5 百分位 {lo:+.4f}–{hi:+.4f}')
    return rs, sp, d

rs_new, sp_new, d_new = summarize(A, '每次重新调参（完整建模策略）')

# 同一批划分下的锁定超参数结果，作对照
B = pd.read_csv(OUT / '表_排序稳定性_逐次.csv', encoding='utf-8-sig')
B = B[B.split < N].copy()
B['rank'] = B.groupby('split')['auc'].rank(ascending=False, method='min').astype(int)
rs_old, sp_old, d_old = summarize(B, '锁定主分析超参数（原分析，同一批划分）')

print('\n' + '=' * 80)
print('结论：锁定超参数是否低估了不稳定性？')
print('=' * 80)
print(f'  排序一致性 ρ 均值   锁定 {rs_old.mean():.3f}  ->  重新调参 {rs_new.mean():.3f}')
print(f'  ρ<0 的划分对比例    锁定 {(rs_old<0).mean()*100:.1f}%  ->  重新调参 {(rs_new<0).mean()*100:.1f}%')
print(f'  随机森林居首比例    锁定 {sp_old.loc["随机森林","居首比例"]}  ->  '
      f'重新调参 {sp_new.loc["随机森林","居首比例"]}')
print(f'  RF−LR 平均 ΔAUC     锁定 {d_old.mean():+.4f}  ->  重新调参 {d_new.mean():+.4f}')
cmp = pd.DataFrame({'锁定超参数': sp_old['名次范围'], '重新调参': sp_new['名次范围']})
print('\n各模型名次范围：')
print(cmp.to_string())
cmp.to_csv(OUT / '表_调参方案对稳定性的影响.csv', encoding='utf-8-sig')
print(f'\n用时 {(time.time()-t0)/60:.1f} min')
