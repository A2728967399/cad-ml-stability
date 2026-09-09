"""Generate manuscript tables and values from frozen, audited numerical outputs.

No model training, data selection, or manuscript prose rewriting is performed.
"""
from pathlib import Path
import csv
import json
import hashlib
import statistics
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
Q4 = ROOT.parent / 'strict_development'
OUT = ROOT / 'generated'
ORDER = ['随机森林', 'SVM', 'K近邻', '梯度提升', 'XGBoost', 'Logistic回归', '多层感知机', 'LightGBM']
SHORT = dict(zip(ORDER, ['rf', 'svm', 'knn', 'gbdt', 'xgb', 'lr', 'mlp', 'lgb']))
SOURCES = {}
VALUES = {}


def read_csv(path):
    path = Path(path)
    SOURCES[str(path.relative_to(ROOT.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def read_json(path):
    path = Path(path)
    SOURCES[str(path.relative_to(ROOT.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding='utf-8-sig'))


def f(value, digits=3, signed=False):
    value = float(value)
    if abs(value) < 0.5 * 10 ** -digits:
        value = 0.0
    return format(value, ('+' if signed else '') + f'.{digits}f')


def interval(lo, hi, digits=3, signed=False):
    return '$' + f(lo, digits, signed) + '$--$' + f(hi, digits, signed) + '$'


def value(key, v, digits=3, signed=False):
    text = f(v, digits, signed)
    VALUES[key] = {'raw': float(v), 'formatted': text}
    return text


def rows(name, content):
    (OUT / (name + '.tex')).write_text('\n'.join(content) + '\n', encoding='utf-8')


def tex_params(params):
    parts = []
    for key, val in sorted(params.items()):
        parts.append(key.replace('_', r'\_') + '=' + str(val).replace('_', r'\_'))
    return r'\allowbreak '.join(parts)


def primary_tables():
    data = read_csv(Q4 / 'results/strict_primary_summary/primary_holdout_conditional.csv')
    by_model = {r['model']: r for r in data}
    oof = {r['model']: r for r in read_csv(Q4 / 'results/strict_primary_summary/oof_score_scale_audit.csv')}
    main, delong = [], []
    for model in ORDER:
        r = by_model[model]
        code = SHORT[model]
        for key, digits in [('auc', 3), ('brier', 4), ('cal_slope', 3), ('cal_intercept_joint', 3), ('youden_probability_threshold', 3), ('test_sensitivity', 3), ('test_specificity', 3)]:
            value(code + '-' + key, r[key], digits)
        for key in ['auc_conditional_ci_low', 'auc_conditional_ci_high', 'delta_auc_vs_lr', 'delta_conditional_ci_low', 'delta_conditional_ci_high']:
            value(code + '-' + key, r[key], 3, key.startswith('delta'))
        for key in ['mean_fold_score_auc', 'sd_fold_score_auc']:
            value(code + '-' + key, oof[model][key])
        value(code + '-outer-minus-test', float(oof[model]['mean_fold_score_auc']) - float(r['auc']), 3, True)
        auc = f(r['auc']) + '（' + interval(r['auc_conditional_ci_low'], r['auc_conditional_ci_high']) + '）'
        delta = '参照' if model == 'Logistic回归' else '$' + f(r['delta_auc_vs_lr'], signed=True) + '$（' + interval(r['delta_conditional_ci_low'], r['delta_conditional_ci_high'], signed=True) + '）'
        main.append(' & '.join([model, auc, delta, f(r['brier'], 4), '$' + f(r['cal_slope']) + '/' + f(r['cal_intercept_joint']) + '$', '参照' if not r['p_holm'] else f(r['p_holm'])]) + r' \\')
        if model != 'Logistic回归':
            p = float(r['delong_p_unadjusted'])
            z = statistics.NormalDist().inv_cdf(1-p/2) * np.sign(float(r['delta_auc_vs_lr']))
            within = float(r['delta_conditional_ci_low']) >= -.05 and float(r['delta_conditional_ci_high']) <= .05
            delong.append(' & '.join([model, f(r['auc']), '$' + f(r['delta_auc_vs_lr'], signed=True) + '$', interval(r['delta_conditional_ci_low'], r['delta_conditional_ci_high']), '$' + f(z) + '$', f(p), f(r['p_holm']), '是' if within else '否']) + r' \\')
    rows('primary_main_rows', main)
    rows('table_s5_rows', delong)


def repeated_tables():
    data = read_csv(Q4 / 'results/strict_summary/per_split_metrics.csv')
    summary = read_json(Q4 / 'results/strict_summary/summary.json')
    hp = read_csv(Q4 / 'results/strict_summary/hyperparameters.csv')
    primary_hp = {r['model']: r['params'] for r in read_json(Q4 / 'results/strict_primary_oof/checkpoints/primary_42/metrics.json')}
    historical_hp = {r['model']: json.loads(r['historical_parameters']) for r in read_csv(Q4 / 'results/strict_summary/hyperparameter_summary.csv')}
    for scheme, short in [('fixed_historical_conditional', 'fixed'), ('retuned', 'retuned')]:
        lines = []
        for model in ORDER:
            part = [r for r in data if r['scheme'] == scheme and r['model'] == model]
            assert len(part) == 200
            a = np.array([float(r['auc']) for r in part])
            d = np.array([float(r['delta_lr']) for r in part])
            rank = np.array([float(r['rank']) for r in part])
            quant = np.quantile(d, [.025, .25, .75, .975])
            qr = np.quantile(rank, [.25, .5, .75])
            prefix = SHORT[model] + '-' + short
            for key, val, dp, sign in [('delta', d.mean(), 4, True), ('beats', (d > 0).mean()*100, 1, False), ('win', (rank == 1).mean()*100, 1, False), ('lo', quant[0], 4, True), ('hi', quant[3], 4, True), ('rankmed', qr[1], 0, False), ('rankq1', qr[0], 1, False), ('rankq3', qr[2], 1, False), ('rankmin', rank.min(), 0, False), ('rankmax', rank.max(), 0, False)]:
                value(prefix + '-' + key, val, dp, sign)
            lines.append(' & '.join([model, f(a.mean(), 4) + '（' + f(a.std(ddof=1), 4) + '）', '$' + f(d.mean(), 4, True) + '$', interval(quant[1], quant[2], 4), interval(quant[0], quant[3], 4), '--' if model == 'Logistic回归' else f((d>0).mean()*100, 1), f((np.abs(d)<=.05).mean()*100, 1), f(qr[1], 0) + '（' + f(qr[0], 1) + '--' + f(qr[2], 1) + '）', f(rank.min(), 0) + '--' + f(rank.max(), 0), f((rank==1).mean()*100, 1)]) + r' \\')
        rows('table_s10_rows' if short == 'fixed' else 'table_retuned_rows', lines)
        for key in ['mean', 'median', 'q25', 'q75']:
            value('rho-' + short + '-' + key, summary['rank_correlations'][scheme][key])
        value('rho-' + short + '-negative', summary['rank_correlations'][scheme]['negative_fraction']*100, 1)
    hp_lines, hp_evidence = [], []
    for model in ORDER:
        current = primary_hp[model]
        items = [json.loads(r['parameters']) for r in hp if r['model'] == model and r['scheme'] == 'retuned']
        counts = Counter(json.dumps(p, sort_keys=True) for p in items)
        modal, count = counts.most_common(1)[0]
        match = sum(p == current for p in items)/200
        hist_match = sum(p == historical_hp[model] for p in items)/200
        value(SHORT[model] + '-param-match', match*100, 1)
        value(SHORT[model] + '-param-modal', count/2, 1)
        value(SHORT[model] + '-param-count', len(counts), 0)
        hp_lines.append(' & '.join([model, str(len(counts)), f(count/2, 1), f(match*100, 1), tex_params(json.loads(modal)), tex_params(current)]) + r' \\')
        hp_evidence.append({'model': model, 'current_match': match, 'historical_match': hist_match, 'unique': len(counts), 'modal_fraction': count/200, 'primary_parameters': current})
    rows('table_s13_rows', hp_lines)
    (OUT/'hyperparameter_reference.json').write_text(json.dumps(hp_evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    feature = read_csv(Q4 / 'results/strict_summary/feature_frequency.csv')
    feature.sort(key=lambda r: (-float(r['frequency']), r['display_name']))
    rows('table_s11_rows', [r['display_name'] + ' & ' + f(float(r['frequency'])*100, 1) + ' & ' + ('高频（描述性）' if float(r['frequency']) >= .8 else r'低于80\%') + r' \\' for r in feature])
    comparison = []
    for label, key in [('排序相关均值', 'rho-{s}-mean'), ('排序相关中位数', 'rho-{s}-median'), (r'负相关比例（\%）', 'rho-{s}-negative'), (r'随机森林居首（\%）', 'rf-{s}-win'), (r'Logistic回归居首（\%）', 'lr-{s}-win'), (r'随机森林平均$\Delta$AUC', 'rf-{s}-delta'), (r'随机森林优于Logistic（\%）', 'rf-{s}-beats')]:
        left = VALUES[key.format(s='fixed')]
        right = VALUES[key.format(s='retuned')]
        direction = '上升' if right['raw'] > left['raw'] else ('下降' if right['raw'] < left['raw'] else '相同')
        comparison.append(label + ' & $' + left['formatted'] + '$ & $' + right['formatted'] + '$ & ' + direction + r' \\')
    rows('table_s12_rows', comparison)
    for key in ['min', 'max', 'median', 'q25', 'q75']:
        value('features-' + key, summary['n_selected'][key], 0)
    value('features-ever', summary['ever_selected'], 0)


def restored_tables():
    metrics = ROOT / 'results/original_metrics'
    recovered = ROOT / 'results/model_recovery/full'
    base = {r['变量']: r for r in read_csv(metrics/'baseline_original_style.csv')}
    mapping = [('年龄', '年龄（岁）'), ('性别', '男性'), ('BMI', r'BMI（kg/m$^2$）'), ('高血压史', '高血压史'), ('糖尿病史', '糖尿病史'), ('吸烟史', '吸烟史'), ('冠心病分型：STEMI', 'STEMI'), ('冠心病分型：NSTEMI', 'NSTEMI'), ('冠心病分型：不稳定型心绞痛', '不稳定型心绞痛'), ('冠心病分型：稳定型心绞痛', '稳定型心绞痛'), ('冠心病分型：分型未知', '临床分型缺失'), ('肌酸激酶', '肌酸激酶（U/L）'), ('血清肌钙蛋白T', r'\mbox{血清肌钙蛋白T}（ng/mL）'), ('SIRI', 'SIRI'), ('左室射血分数', r'左室射血分数（\%）'), ('室壁运动评分', '室壁运动评分')]
    def chinese(v):
        return v.replace(' (', '（').replace(')', '）').replace(', ', '，').replace('%', r'\%')
    rows('baseline_main_rows', [' & '.join([label, chinese(base[key]['低GS组']), chinese(base[key]['高GS组']), f(base[key]['SMD'])]) + r' \\' for key,label in mapping])
    sens = read_csv(recovered/'sensitivity_metrics.csv')
    scenarios = ['P33_locked', 'P50_locked_baseline', 'mean_locked', 'readmit_GS0_locked', 'thesis16_locked']
    lookup = {(r['scenario'], r['model']): r for r in sens}
    sens_order = ['随机森林', 'Logistic回归', '梯度提升', 'XGBoost', 'LightGBM', 'K近邻', '多层感知机', 'SVM']
    rows('sensitivity_main_rows', [' & '.join([m] + [f(lookup[s,m]['auc']) for s in scenarios]) + r' \\' for m in sens_order])
    for r in sens:
        value(SHORT[r['model']]+'-'+r['scenario'], r['auc'])
    for r in read_csv(metrics/'dca_key_thresholds.csv'):
        threshold = str(round(float(r['threshold'])*100))
        for field in ['delta_vs_all', 'delta_vs_all_ci_low', 'delta_vs_all_ci_high', 'delta_vs_lr', 'delta_vs_lr_ci_low', 'delta_vs_lr_ci_high']:
            value(SHORT[r['model']]+'-dca-'+threshold+'-'+field, r[field])
    for r in read_csv(metrics/'subgroup_auc.csv'):
        for field in ['auc', 'ci_low', 'ci_high']:
            value(SHORT[r['model']]+'-'+r['subgroup']+'-'+field, r[field])
    for r in read_csv(metrics/'hosmer_lemeshow.csv'):
        value(SHORT[r['model']]+'-hl-p', r['hl_p'], 4)


def main():
    OUT.mkdir(exist_ok=True)
    primary_tables()
    repeated_tables()
    restored_tables()
    table_macros = []
    for command, filename in [('ResultBaselineRows', 'baseline_main_rows'), ('ResultPrimaryRows', 'primary_main_rows'), ('ResultSensitivityRows', 'sensitivity_main_rows')]:
        content = (OUT/(filename+'.tex')).read_text(encoding='utf-8').strip()
        table_macros.append('\\newcommand{\\' + command + '}{%\n' + content + '%\n}')
    rows('main_table_macros', table_macros)
    definitions = [r'\providecommand{\V}[1]{\csname result@#1\endcsname}']
    definitions += [r'\expandafter\def\csname result@' + k + r'\endcsname{' + v['formatted'] + '}' for k,v in sorted(VALUES.items())]
    rows('numbers', definitions)
    manifest = {'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'sources': SOURCES, 'values': VALUES}
    (ROOT/'audit/number_sources.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v['formatted'] for k,v in VALUES.items() if k.startswith(('rf-', 'lr-', 'rho-', 'features-'))}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
