#!/usr/bin/env python3
"""Build only the assigned supplement row files from completed read-only outputs.

No clinical/model fits, no plotting or compilation, no edits to authored sections,
sync_evidence.py, or the root-owned S5/S10/S11/S12/S13 generated rows.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
Q4 = ROOT.parent / 'strict_development'
G = ROOT / 'generated'
M = ROOT / 'results/original_metrics'
R = ROOT / 'results/model_recovery/full'
P = Q4 / 'results/strict_primary_oof/checkpoints/primary_42'
PS = Q4 / 'results/strict_primary_summary'
SOURCES = {}
WRITTEN = {}
OWNED = {f'table_s{i}_rows.tex' for i in [1, 2, 3, 4, 6, 7, 8, 9, 14]} | {'supplement_methods.tex', 'supplement_enhanced_results.tex', 'supplement_table_macros.tex'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def track(path):
    path = Path(path)
    SOURCES[str(path)] = sha(path)
    return path


def js(path):
    return json.loads(track(path).read_text(encoding='utf-8'))


def data(path):
    return pd.read_csv(track(path))


def txt(path):
    return track(path).read_text(encoding='utf-8')


def write(name, text):
    if name not in OWNED:
        raise RuntimeError('Non-owned output refused: ' + name)
    G.mkdir(exist_ok=True)
    dest = G / name
    temp = dest.with_name(dest.name + '.partial')
    temp.write_text('% Generated from completed current results; no model fit in this builder.\n' + text + '\n', encoding='utf-8')
    temp.replace(dest)
    WRITTEN[name] = sha(dest)


def esc(value):
    text = str(value)
    mapping = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}'}
    return ''.join(mapping.get(ch, ch) for ch in text)


def fmt(value, digits=3):
    return '未单列' if value is None or pd.isna(value) else f'{float(value):.{digits}f}'


def ci(lo, hi):
    return fmt(lo) + '--' + fmt(hi)


def row(values):
    return ' & '.join(map(str, values)) + r' \\'


def display(name):
    return '左心室收缩末期容积' if name == '心室收缩容量' else str(name)


def table(caption, label, headers, rows, note='', landscape=False, widths=None):
    columns = widths or ('l' + 'r' * (len(headers) - 1))
    beginning = '\\clearpage\n' + ('\\begin{landscape}\n' if landscape else '')
    out = [beginning, r'\begin{table}[p]\centering\footnotesize', r'\setlength{\tabcolsep}{4pt}',
           '\\caption{' + caption + '}\\label{' + label + '}', '\\begin{tabular}{' + columns + '}', r'\toprule', row(headers), r'\midrule']
    out += rows
    out += [r'\bottomrule\end{tabular}', r'\par\medskip\begin{minipage}{0.97\linewidth}\footnotesize ' + note + r'\end{minipage}', r'\end{table}']
    if landscape:
        out += [r'\end{landscape}']
    return '\n'.join(out)


def main():
    track(__file__)
    assert js(M / 'summary.json')['status'] == 'complete'
    assert js(M / 'summary.json')['n_bootstrap'] == 1000
    rec = js(R / 'summary.json')
    assert rec['state'] == 'complete' and rec['publication_eligible']
    assert rec['lr_bootstrap_completed'] == 1000 and rec['lasso_bootstrap_completed'] == 300
    assert all(sha(R / name) == digest for name, digest in rec['output_sha256'].items())
    for sub in ['strict_summary', 'strict_all_features_summary', 'strict_primary_summary']:
        assert js(Q4 / 'results' / sub / 'summary.json')['complete']
    assert js(Q4 / 'results/structure_performance/summary.json')['status'] == 'complete'
    sim_complete = js(Q4 / 'results/simulation/COMPLETE.json')
    assert sim_complete['records'] == 400
    models = [r['model'] for r in js(P / 'metrics.json')]
    assert len(models) == len(set(models)) == 8

    baseline = data(M / 'baseline_original_style.csv')
    assert len(baseline) == 54
    write('table_s1_rows.tex', '\n'.join(row([esc(display(r['变量'])), esc(r['总体']), esc(r['低GS组']), esc(r['高GS组']), int(r['缺失数']), fmt(r['SMD'])]) for _, r in baseline.iterrows()))
    holdout = data(PS / 'primary_holdout_conditional.csv').set_index('model')
    folds = data(PS / 'oof_score_scale_by_fold.csv')
    assert len(folds) == 40 and (folds.groupby('model').size() == 5).all()
    primary_metrics = {r['model']: r for r in js(P / 'metrics.json')}
    s2 = []
    for name in models:
        r = primary_metrics[name]
        f = folds.loc[folds.model == name, 'score_auc']
        params = esc(json.dumps(r['params'], ensure_ascii=False)).replace(', ', ',\\allowbreak ')
        s2.append(row([esc(name), params, fmt(r['inner_cv_auc']), fmt(f.mean()) + r'$\pm$' + fmt(f.std(ddof=1)),
                       '未单列', fmt(f.mean() - r['auc']), '未单列']))
    write('table_s2_rows.tex', '\n'.join(s2))
    classification = data(M / 'classification_youden.csv')
    classification = classification[classification['sample'] == 'test'].set_index('model')
    hl = data(M / 'hosmer_lemeshow.csv').set_index('model')
    s3 = []
    for name in models:
        r, h = classification.loc[name], holdout.loc[name]
        s3.append(row([esc(name), fmt(r.threshold), '/'.join(str(int(r[v])) for v in ['tp','fp','tn','fn']),
                       *[fmt(r[v]) for v in ['accuracy','sensitivity','specificity','precision','f1']],
                       fmt(h.auc) + ' (' + ci(h.auc_conditional_ci_low,h.auc_conditional_ci_high) + ')', fmt(h.brier),
                       fmt(h.cal_slope) + '/' + fmt(h.cal_intercept_joint), '$<0.001$' if hl.loc[name,'hl_p'] < .001 else fmt(hl.loc[name,'hl_p'])]))
    write('table_s3_rows.tex', '\n'.join(s3))
    representation = js(P / 'audit.json')['development']['final_representation']
    freq = data(R / 'lasso_bootstrap_frequencies.csv')
    assert len(freq) == 49 and (freq.bootstrap_completed == 300).all()
    write('table_s4_rows.tex', '\n'.join(row([esc(display(r.feature)) + ('$^{*}$' if r.feature in representation['selected'] else ''),
                        fmt(representation['coef'].get(r.feature, 0.)), fmt(100*r.selection_frequency,1), fmt(r.mean_absolute_coefficient)]) for _, r in freq.iterrows()))
    dca = data(M / 'dca_key_thresholds.csv')
    assert len(dca) == 32 and (dca.n_bootstrap == 1000).all()
    write('table_s6_rows.tex', '\n'.join(row([esc(r.model), fmt(r.threshold,2), fmt(r.delta_vs_all), ci(r.delta_vs_all_ci_low,r.delta_vs_all_ci_high),
                                            fmt(r.delta_vs_lr),ci(r.delta_vs_lr_ci_low,r.delta_vs_lr_ci_high)]) for _,r in dca.iterrows()))
    groups = data(M / 'subgroup_auc.csv')
    subgroup_labels = {'nonMI':'非心肌梗死', 'MI':'心肌梗死', 'unknown':'分型未知'}
    visible = groups[groups.subgroup.isin(subgroup_labels)]
    assert len(visible) == 24 and (visible.groupby('subgroup').size() == 8).all()
    assert all(set(part.model) == set(models) for _, part in visible.groupby('subgroup'))
    unknown = visible[visible.subgroup == 'unknown']
    assert (unknown[['n','events','non_events','n_bootstrap_requested','n_bootstrap_valid','n_bootstrap_single_class']].to_numpy()
            == np.array([10, 3, 7, 1000, 971, 29])).all()
    write('table_s7_rows.tex', '\n'.join(row([subgroup_labels[r.subgroup],esc(r.model),int(r.n),int(r.events),fmt(r.auc)]) for _,r in visible.iterrows()))
    coeff = data(R / 'logistic_coefficients.csv')
    assert len(coeff) == 14 and (coeff.bootstrap_completed == 1000).all()
    s8 = []
    for _,r in coeff.iterrows():
        intercept = r.feature == 'intercept'
        s8.append(row(['截距' if intercept else esc(display(r.feature)), '不适用' if intercept else ('log1p后标准化' if r['transform'].startswith('log1p') else '原值标准化'),
                       *['--' if intercept else fmt(r[v]) for v in ['imputation','training_mean_transformed','training_sd_transformed']],
                       fmt(r.coefficient),ci(r.ci_lower,r.ci_upper), '--' if intercept else fmt(r.OR_per_training_SD)+' ('+ci(r.OR_ci_lower,r.OR_ci_upper)+')']))
    write('table_s8_rows.tex', '\n'.join(s8))
    hi = data(M / 'lr_high_sensitivity.csv').set_index('sample')
    assert set(hi.index) == {'test','oof'}
    test, oof = hi.loc['test'], hi.loc['oof']
    write('table_s9_rows.tex', row(['Logistic回归',fmt(test.threshold),fmt(oof.sensitivity),fmt(oof.specificity),
                                  '/'.join(str(int(test[v])) for v in ['tp','fp','tn','fn']),fmt(test.sensitivity),fmt(test.specificity)]))
    cohort = js(Q4 / 'results/strict_primary_oof/run_manifest.json')['cohort']
    rules = [r for r in cohort['range_corrections'] if r['n_source_to_missing']]
    total_missing = sum(r['n_source_to_missing'] for r in rules)
    assert total_missing == 45
    s14 = [row([esc(display(r['feature'])), '$<$'+f'{r["rule"][0]:g}'+'或$>$'+f'{r["rule"][1]:g}', '置为缺失，不删除病例',str(r['n_source_to_missing'])+'个数据值']) for r in rules]
    s14 += [row(['病例记录','Gensini评分$=0$','据作者确认：既往PCI后复查，本次未见再狭窄；从工作队列排除并另行回纳敏感性','9例']),
            row(['资格待核','Gensini评分为1、1.5、1.5','现阶段保留；本次CAG入选资格须逐例核实，不记为确认合格','3例'])]
    write('table_s14_rows.tex','\n'.join(s14))

    # Retain the already audited Q4 complete numerical tables, not a rewritten
    # supplement. Only these six new appendix tables are appended after S1-S4.
    all_features = txt(Q4 / 'manuscript/generated/all_features.tex')
    simulation_main = txt(Q4 / 'manuscript/generated/simulation.tex').split('\\begin{table}[!htbp]',2)[1]
    simulation_main = '\\begin{table}[!htbp]' + simulation_main
    simulation_main = simulation_main[:simulation_main.index('\\end{table}') + len('\\end{table}')]
    sims = js(Q4 / 'results/simulation/summary.json')
    assert len(sims) == 8 and sum(r['replicates'] for r in sims) == 400
    raw_sim = data(Q4 / 'results/simulation/per_model.csv')
    sim_rows = []
    for r in sorted(sims,key=lambda v:(v['signal'],v['rho'],v['n'])):
        cells = ['线性' if r['signal']=='linear' else '非线性',r['n'],fmt(r['rho'],1),fmt(r['bayes_auc_mean'])+' ('+fmt(r['bayes_auc_mcse'])+')']
        for model in ['LR','RF','SVM']:
            v=raw_sim[(raw_sim.scenario==r['scenario'])&(raw_sim.model==model)].independent_auc
            assert len(v)==50
            cells += [fmt(v.mean())+' ('+fmt(v.std(ddof=1)/np.sqrt(50))+')']
        cells += [fmt(100*r[m+'_test_win_fraction'],1)+' ('+fmt(100*r[m+'_test_win_mcse'],1)+')' for m in ['LR','RF','SVM']]
        sim_rows.append(row(cells))
    sim_models = table('全部8种模拟情景的独立评价AUC与小测试集居首率（括号为MCSE）','tab:simulation_models',
                       ['信号','$n$','$\\rho_X$','Bayes','LR','RF','SVM','LR居首\\%','RF居首\\%','SVM居首\\%'],sim_rows,
                       '每情景50个独立数据集；各算法AUC为独立5000例评价均值，括号为MCSE。居首率与其MCSE以百分点表示；边界比例的代入MCSE为0不代表精确确定。仅模拟LR、RF、SVM三个家族，非八个临床算法。',True)
    structure = txt(Q4 / 'manuscript/generated/structure_performance.tex')
    split_token = '{\\footnotesize\\setlength{\\tabcolsep}{4pt}'
    structure = split_token + structure.split(split_token)[1]
    events = js(Q4 / 'results/training_events/summary.json')
    assert events['complete']
    event_rows=[]
    for name,label in [('main','共同LASSO及配对'),('all_features','全变量对照'),('primary','单次及严格OOF')]:
        e=events['runs'][name]
        event_rows.append(row([label,f'{e["completed_checkpoints"]}/{e["planned_checkpoints"]}',e['completed_development_fit_calls'],e['future_warning_events'],e['convergence_warning_events'],e['fit_calls_without_structured_warning_record']]))
    path_report=js(ROOT/'results/model_recovery/lasso_path/summary.json')
    event_rows += [row(['本轮模型恢复与补齐','完成',1340,rec['warnings']['categories'].get('FutureWarning',0),rec['warnings']['convergence'],0]),
                   row(['S4固定系数路径','100/100',path_report['n_model_fits'],path_report['warning_aggregate']['categories'].get('FutureWarning',0),path_report['warning_aggregate']['convergence'],0])]
    event_table=table('训练事件、警告类型与结构化记录覆盖','tab:training_events',
                       ['运行','完成/计划','模型拟合数','Future警告','收敛警告','未记录警告拟合'],event_rows,
                       '只计完成开发中的顶层estimator.fit，不重复计缓存或模型内部树。警告发出次数不等于失败模型数；OOF外折40次最终拟合未单独持久化fit warnings，记为未知而非零。LASSO路径仅有聚合警告，不能按C定位。全变量运行4个正在计算划分曾因4路至6路并行调整中断后重算，完整检查点未删，非算法失败。')
    enhanced = ['\\clearpage\n\\section*{增强结果：已完成的配对对照、模拟与训练审计}',
                '以下补充表S15--S20不替代原S1--S14和图S1--S4。全变量对照与结构--性能对应为事后描述；模拟不构成临床外部验证。',
                all_features, '\\clearpage', simulation_main, sim_models, '\\clearpage',structure,event_table]
    write('supplement_enhanced_results.tex','\n'.join(enhanced).replace('\\FloatBarrier{}','\\clearpage'))
    methods = r'''\clearpage
\section*{补充方法}
\subsection*{完整开发层级与实现边界}
临床任务均固定GS$>37$，仅按结局分层。每次75\%/25\%划分中，模型参数在开发数据内5折搜索；每个模型调参训练折重新进行LASSO自身5折、100个C的选择，每个LASSO训练折再独立估计缺失率、填补、变换和标准化。最终在完整开发集拟合表示并使用选定模型参数。仅相同训练成员的中间表示可以跨算法和参数复用。临床完整网格共113组合，逐项与正文一致；无LASSO对照也使用同一完整网格及折内预处理。没有过采样、欠采样或自动类别权重。单次978例的5折OOF评价在各外折重跑整套开发，不能用内层选参分数代替验证性能。

SVM为RBF核，区分度使用决策函数；概率沿用SVC内部Platt映射。其内部概率子程序在传入的已拟合表示上运行，没有在每个概率子折重做LASSO。MLP启用早停，最大迭代3000，内部停止集10\%，以准确率监测，连续无改善10轮、容差0.0001；内部停止子集也已参与当前训练范围的表示拟合。这两项内部程序不接触模型CV验证、OOF验证或最终留出测试成员，但不能另行称为独立无偏校准或验证。

条件固定200次对照读取早期Q3交叉验证与超参数表；其开发成员可能与新测试成员重叠，不能把两方案差值当作无偏的纯调参效应。本次参数组合复现率则匹配重新开发的seed42主分析参数，与历史固定参数匹配率不同。49项共同LASSO表示限制了算法比较结论；全变量对照在相同200次患者划分中独立调参，不能据其个别有利结果事后替换主分析。

\subsection*{概率、条件bootstrap与模型恢复}
H--L检验沿原函数：按预测概率排序并等人数切为10组，使用组内观测及预期事件计算统计量，自由度为组数减2；未作多重比较校正，仅辅助描述。KNN有8个组边界切开并列概率，故检验受排序与分组规则影响。DCA以测试患者配对重抽样1000次，全部模型和比较策略共用每次抽样索引；0.30、0.40、0.50、0.60为表列阈值，区间为逐点经验百分位区间，不是同时置信带。

校准联合截距和斜率采用未惩罚二项Logistic最大似然；另估计斜率固定为1的总体截距。对数几率输入仅为数值计算截至$[10^{-6},1-10^{-6}]$。若BFGS未报告成功，仅当梯度L2范数小于$10^{-5}$才按原规则接受；16条精度提示的估计均满足该规则并已独立复算，主与全变量各模型有效分母均200/200。图形继续使用原十分位分组曲线，不以后来LOWESS图替换原设计。

本轮在核对原始数据、冻结代码、核心库版本及成员哈希后恢复最终模型；全部8模型326名测试患者预测最大概率差为$1.11\times10^{-16}$。LR的1000次bootstrap固定训练预处理、13项支持及$C=0.01$；LASSO的300次bootstrap固定49项训练矩阵和$C=0.04328761281083059$，不重新选C。区间与频率均不传播完整选择过程不确定性。图S4额外计算100个既定C在固定完整训练矩阵上的系数路径，不重新CV或选择；CV均值与标准误读取原冻结的折内预处理结果。SHAP按树路径依赖定义、无外部背景，对全部326例类别1概率验证基值与贡献加和；最大误差$1.34\times10^{-15}$以内。逐患者派生数据仅供受控核验，不自动授权公开。

\subsection*{结构与性能的事后描述}
对共同LASSO的200次重调参划分，计算各变量集合的Jaccard相似度，并与同一算法在两次划分中的AUC绝对差作Spearman描述。每模型包含19900个相互依赖的划分对；不作独立样本显著性检验，也不把相关关系解释为变量变化造成性能变化。

\subsection*{模拟目标、生成机制与完整搜索范围}
模拟用于考察样本量、相关结构、信号形式与排序/按测试集择优的关系，仅比较LR、RF、SVM三个代表家族，不模拟八个临床算法或共同LASSO。设$p=15$，$X_0\sim N(0,1)$，$X_j=\rho X_{j-1}+\sqrt{1-\rho^2}\epsilon_j$，各$\epsilon_j\sim N(0,1)$相互独立；$\Sigma_{jk}=\rho^{|j-k|}$，$w=(1,-0.8,0.6,0.4,-0.4)^{\mathsf T}$。线性评分为
\[\eta_L=\frac{\sum_{j=0}^{4}w_jX_j}{\sqrt{w^{\mathsf T}\Sigma_{0:4,0:4}w}},\]
非线性评分为
\[\eta_N=\frac{1.1(X_0X_1-\rho)}{\sqrt{1+\rho^2}}+\frac{0.65(X_2^2-1)}{\sqrt2}+0.45X_4.\]
结局按$Y\sim\mathrm{Bernoulli}\{\mathrm{expit}(1.25\eta)\}$产生，不强制事件比例相同；Bayes排序使用真实$\eta$。两机制不假定同一难度。样本量200/800、相关系数0/0.6、线性/非线性的全部笛卡尔积形成8个情景，各独立重复50次，合计400次。每次按结局分层为75\%开发和25\%小测试数据，并另生成5000例独立评价集。随机种子为202609080加情景序号乘10000加重复序号。

模拟开发内采用3折分层CV，标准化位于每折Pipeline中。LR搜索$C\in\{0.1,1,10\}$；RF固定100树，最大深度3或不限，叶节点最小样本5或15；SVM搜索$C\in\{0.1,1,10\}$及$\gamma\in\{\mathrm{scale},0.05\}$，不拟合概率。各家族以开发CV AUC选参，再按最高CV AUC选择算法；精确并列依LR、RF、SVM顺序打破。独立评价集不选参或选算法。另记录小测试集赢家在其测试AUC与独立AUC之间的差，度量按测试集择优的乐观差。

平均指标的Monte Carlo标准误（MCSE）为独立重复标准差除以$\sqrt{50}$；居首率MCSE为$\sqrt{\hat p(1-\hat p)/50}$。边界比例的代入MCSE为0不代表精确确定。跨数据集排序相关均值为二阶U统计量，采用逐一删除数据集的jackknife MCSE，不能把每情景1225个相关对当作独立重复。全部8种情景、独立评价结果及MCSE均列于S17--S18，不择有利情景报告。
'''
    historical = data(Q4 / 'results/strict_summary/hyperparameter_summary.csv')
    assert len(historical) == 8 and set(historical.model) == set(models)
    historical_rows = []
    for name in models:
        value = historical.loc[historical.model == name, 'historical_parameters'].iloc[0]
        params = json.loads(value)
        description = esc(json.dumps(params, ensure_ascii=False)).replace(', ', ',\\allowbreak ')
        historical_rows.append(row([esc(name), description]))
    history_table = '\n'.join([r'\clearpage\begin{table}[p]\centering\small',
                               r'\textbf{历史固定参数来源表（无编号；非本次seed42所选参数）}\par\medskip',
                               r'\begin{tabular}{@{}p{3.0cm}p{12.0cm}@{}}\toprule',
                               row(['模型','200次条件固定对照采用的历史参数']),r'\midrule',*historical_rows,
                               r'\bottomrule\end{tabular}\par\medskip',
                               r'\begin{minipage}{0.94\linewidth}\footnotesize 本表读取已核查的历史参数字段，来源为既有Q3交叉验证与超参数记录，不使用本次seed42参数替代。历史开发成员可能与新重复划分的测试成员重叠；仅供条件对照复现，不代表无信息重用的模型开发验证。\end{minipage}',
                               r'\end{table}\clearpage'])
    methods = methods.replace(r'\subsection*{概率、条件bootstrap与模型恢复}',history_table+'\n'+r'\subsection*{概率、条件bootstrap与模型恢复}')
    # Preload row bodies before entering any alignment. Runtime \input inside
    # tabular can leave non-expandable input bookkeeping before \bottomrule's
    # \noalign. These expandable macros preserve every row token and avoid it.
    macro_lines = [r'\newcommand{\TabRows}[1]{\csname SupplementRows@#1\endcsname}']
    row_names = [f'table_s{i}_rows' for i in range(1, 15)] + ['table_retuned_rows']
    for name in row_names:
        body = txt(G / (name + '.tex'))
        macro_lines += [r'\expandafter\def\csname SupplementRows@' + name + r'\endcsname{%', body.rstrip(), '}']
    write('supplement_table_macros.tex', '\n'.join(macro_lines))
    comment = '% INPUT_SHA256 ' + json.dumps(SOURCES, ensure_ascii=False, sort_keys=True)
    write('supplement_methods.tex', comment + '\n' + methods)
    assert all(sha(path) == value for path,value in SOURCES.items())
    assert set(WRITTEN) == OWNED
    print(json.dumps({'status':'PASS','written_files':WRITTEN,'source_count':len(SOURCES),
                      'rows':{'S1':54,'S2':8,'S3':8,'S4':49,'S6':32,'S7':len(visible),'S8':14,'S9':1,'S14':len(s14)},
                      'added_tables':['S15 allfeatures','S16 allfeatures calibration','S17 eight simulation scenarios+MCSE','S18 simulation models+MCSE','S19 structure','S20 warnings'],
                      'training_or_compilation':False},ensure_ascii=False))


if __name__ == '__main__':
    main()
