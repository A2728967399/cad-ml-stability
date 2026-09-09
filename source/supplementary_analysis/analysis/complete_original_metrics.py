#!/usr/bin/env python3
"""Complete original-paper descriptive metrics from frozen predictions; no fitting.

Only the current project results/original_metrics directory is writable. Q3/Q4
modules are parsed as source, never imported. No identity columns are loaded.
Default 1000 ordinary patient bootstrap replicates; --self-test uses 20 instead.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import ast
import hashlib
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
Q4 = PROJECT / 'strict_development'
Q3_CODE = PROJECT / 'legacy_clinical/reanalysis/scripts/01_q3_analysis.py'
METRIC_CODE = PROJECT / 'clinical_pipeline/eval/fl_metrics.py'
NESTED_CODE = Q4 / 'analysis/nested_pipeline.py'
RAW = PROJECT / 'clinical_pipeline/data/raw/source_clean.csv'
PRIMARY = Q4 / 'results/strict_primary_oof/checkpoints/primary_42'
MODELS = ['Logistic回归', '随机森林', 'K近邻', '梯度提升', 'SVM', 'XGBoost', 'LightGBM', '多层感知机']
GROUPS = ['nonMI', 'MI', 'unknown']
DCA_SEED = 64
SUBGROUP_SEED = 2025
CHECKS = []


def check(condition, name, detail=None):
    CHECKS.append({'name': name, 'passed': bool(condition), 'detail': detail})
    if not condition:
        raise AssertionError(f'{name}: {detail}')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    s = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def members_hash(rows):
    return canonical_hash(sorted(int(x) for x in rows))


def json_clean(value):
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_clean(v) for v in value]
    if isinstance(value, np.generic):
        return json_clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(out, name, value):
    (out / name).write_text(json.dumps(json_clean(value), ensure_ascii=False,
                                     indent=2, allow_nan=False), encoding='utf-8')


def save_csv(out, name, data):
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    frame.to_csv(out / name, index=False, encoding='utf-8-sig', float_format='%.17g')


def read_csv(path, **kwargs):
    return pd.read_csv(path, encoding='utf-8-sig', float_precision='round_trip', **kwargs)


def literal_constants(path, names):
    result = {}
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    result[target.id] = ast.literal_eval(node.value)
    check(set(result) == set(names), f'constants:{path.name}')
    return result


def isolated_function(path, name):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    check(len(nodes) == 1, f'function_found:{name}')
    # Only the explicitly requested function is compiled. No file-level code runs.
    env = {'np': np, 'stats': stats, 'roc_curve': roc_curve}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return env[name], {'file': str(path), 'function': name,
                       'line_start': nodes[0].lineno, 'line_end': nodes[0].end_lineno,
                       'function_source_sha256': hashlib.sha256(
                           ast.get_source_segment(path.read_text(encoding='utf-8'), nodes[0]).encode('utf-8')).hexdigest()}


def auc_u(y, score):
    y = np.asarray(y, int)
    n1 = int(y.sum()); n0 = len(y) - n1
    if not n1 or not n0:
        return np.nan
    ranks = stats.rankdata(np.asarray(score, float), method='average')
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def classification(y, p, threshold):
    y = np.asarray(y, int); pred = np.asarray(p) >= threshold
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0))); fn = int(np.sum(~pred & (y == 1)))
    div = lambda a, b: float(a / b) if b else np.nan
    return {'n': len(y), 'events': int(y.sum()), 'threshold': float(threshold),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'sensitivity': div(tp, tp + fn), 'specificity': div(tn, tn + fp),
            'precision': div(tp, tp + fp), 'npv': div(tn, tn + fn),
            'accuracy': div(tp + tn, len(y)), 'f1': div(2 * tp, 2 * tp + fp + fn)}


def load_inputs():
    old = literal_constants(Q3_CODE, {'FEATURES_49', 'CATEGORICAL', 'RANGE', 'SUBTYPE'})
    frozen = literal_constants(NESTED_CODE, {'FEATURES', 'CATEGORICAL', 'RANGES'})
    check(old['FEATURES_49'] == frozen['FEATURES'], 'feature_definitions_match')
    check(set(old['CATEGORICAL']) == frozen['CATEGORICAL'], 'categorical_definitions_match')
    check(old['RANGE'] == frozen['RANGES'], 'range_definitions_match')
    check(old['SUBTYPE'] == {1: 'STEMI', 2: 'NSTEMI', 3: '不稳定型心绞痛', 4: '稳定型心绞痛'}, 'subtype_dictionary')
    run = json.loads((PRIMARY.parents[1] / 'run_manifest.json').read_text(encoding='utf-8'))
    manifest = json.loads((PRIMARY / 'manifest.json').read_text(encoding='utf-8'))
    check(manifest['state'] == 'complete', 'primary_complete')
    check(manifest['run_hash'] == run['run_hash'], 'run_hash_alignment')
    check(file_hash(RAW) == run['config']['raw_sha256'], 'raw_hash_matches_frozen_run')
    check(file_hash(NESTED_CODE) == run['config']['analysis_code_sha256'], 'pipeline_hash_matches_frozen_run')
    check(manifest['membership']['stratification'] == 'outcome only', 'unchanged_outcome_stratification')
    for name, digest in manifest['output_sha256'].items():
        check(file_hash(PRIMARY / name) == digest, f'frozen_manifest_hash:{name}')
    columns = list(dict.fromkeys(old['FEATURES_49'] + ['Gensini评分', '白细胞计数', '中性粒细胞计数', '冠心病类型']))
    check('patient_ID' not in columns, 'identity_columns_not_requested')
    raw = read_csv(RAW, usecols=columns)
    check(set(raw.columns) == set(columns), 'only_authorized_columns_loaded')
    numeric = raw.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    gs = numeric['Gensini评分']
    check(gs.notna().all() and gs.ge(0).all(), 'outcome_finite_nonnegative')
    keep = gs.ne(0)
    check(len(raw) == 1313 and int(keep.sum()) == 1304 and int((~keep).sum()) == 9, 'cohort_counts')
    cleaned = numeric.loc[keep].copy()  # retain zero-based raw data-record positions
    denominator = cleaned['白细胞计数'] - cleaned['中性粒细胞计数']
    cleaned['dNLR'] = (cleaned['中性粒细胞计数'] / denominator).where(
        (denominator > 0) & cleaned['中性粒细胞计数'].ge(0))
    cleaned['TC-HDLDL'] = cleaned['总胆固醇'] - cleaned['高密度脂蛋白'] - cleaned['低密度脂蛋白']
    corrections = []
    for name, (lo, hi) in old['RANGE'].items():
        bad = cleaned[name].notna() & ~cleaned[name].between(lo, hi)
        corrections.append({'source_column': name, 'n_to_missing': int(bad.sum()), 'lo': lo, 'hi': hi})
        cleaned.loc[bad, name] = np.nan
    raw_subtype = raw.loc[keep, '冠心病类型']
    subtype = pd.to_numeric(raw_subtype, errors='coerce')
    check(not (raw_subtype.notna() & subtype.isna()).any(), 'no_unparsed_nonmissing_subtype')
    check(set(subtype.dropna().unique()).issubset({1, 2, 3, 4}), 'subtype_codes_exclusive_known_values',
          sorted(float(v) for v in subtype.dropna().unique()))
    cleaned['冠心病类型'] = subtype
    members = read_csv(PRIMARY / 'membership.csv')
    check(members.source_row.is_unique and set(members.source_row) == set(cleaned.index), 'membership_exact_raw_row_union')
    check(set(members.partition) == {'train', 'test'}, 'membership_partitions')
    for role in ('train', 'test'):
        rr = members.loc[members.partition.eq(role), 'source_row'].to_numpy(int)
        check(len(rr) == manifest['membership'][f'n_{role}'], f'membership_n:{role}')
        check(members_hash(rr) == manifest['membership'][f'{role}_members_sha256'], f'membership_hash:{role}')
    check(np.array_equal(members.y, gs.loc[members.source_row].gt(37).astype(int)), 'raw_row_mapping_labels_exact')
    # Explicit offset negatives make accidental one-based row joins fail loudly.
    offset_checks = {}
    for offset in (-1, 1):
        shifted = members.source_row.to_numpy(int) + offset
        in_bounds = (shifted >= 0) & (shifted < len(raw))
        offset_checks[str(offset)] = {'out_of_bounds': int((~in_bounds).sum()),
            'label_disagreements_in_bounds': int(np.sum(
                gs.iloc[shifted[in_bounds]].gt(37).to_numpy() != members.y.to_numpy()[in_bounds]))}
        check(offset_checks[str(offset)]['out_of_bounds'] > 0 or
              offset_checks[str(offset)]['label_disagreements_in_bounds'] > 0, f'offset_rejected:{offset}')
    pred = read_csv(PRIMARY / 'predictions.csv')
    oof = read_csv(PRIMARY / 'oof_predictions.csv')
    aligned = {}
    for kind, df, role in [('test', pred, 'test'), ('oof', oof, 'train')]:
        check(set(df.model) == set(MODELS) and set(df.scheme) == {'retuned'}, f'eight_models:{kind}')
        canonical_rows = df.loc[df.model.eq(MODELS[0]), 'source_row'].to_numpy(int)
        expected = set(members.loc[members.partition.eq(role), 'source_row'])
        check(len(canonical_rows) == len(expected) and set(canonical_rows) == expected, f'prediction_members:{kind}')
        yy = gs.loc[canonical_rows].gt(37).to_numpy(int)
        ps = []; scores = []
        for name in MODELS:
            g = df.loc[df.model.eq(name)]
            check(g.source_row.is_unique, f'unique_prediction_rows:{kind}:{name}')
            check(np.array_equal(g.source_row, canonical_rows), f'prediction_order:{kind}:{name}')
            check(np.array_equal(g.y, yy), f'prediction_y:{kind}:{name}')
            check(np.isfinite(g[['p', 'score']].to_numpy()).all() and g.p.between(0, 1).all(), f'prediction_finite:{kind}:{name}')
            if kind == 'oof':
                check(set(g.oof_fold) == set(range(5)), f'five_oof_folds:{name}')
            ps.append(g.p.to_numpy()); scores.append(g.score.to_numpy())
        aligned[kind] = {'rows': canonical_rows, 'y': yy, 'p': np.asarray(ps), 'score': np.asarray(scores)}
    saved = read_csv(Q4 / 'results/strict_primary_summary/primary_holdout_conditional.csv').set_index('model')
    for i, name in enumerate(MODELS):
        a = auc_u(aligned['test']['y'], aligned['test']['score'][i])
        b = np.mean((aligned['test']['y'] - aligned['test']['p'][i]) ** 2)
        check(abs(a - saved.loc[name, 'auc']) < 2e-14, f'auc_vs_independent_primary:{name}')
        check(abs(b - saved.loc[name, 'brier']) < 2e-14, f'brier_vs_independent_primary:{name}')
    return old, cleaned, members, aligned, corrections, offset_checks


def build_baseline(out, cs, cleaned, members):
    y = cleaned['Gensini评分'].gt(37)
    groups = {'总体': cleaned.index, 'GS≤37': cleaned.index[~y], 'GS>37': cleaned.index[y],
              '单次开发集': members.loc[members.partition.eq('train'), 'source_row'].to_numpy(),
              '单次测试集': members.loc[members.partition.eq('test'), 'source_row'].to_numpy()}
    q4 = read_csv(Q4 / 'results/cohort_audit/baseline.csv').set_index(['source_column', 'group'])
    long_rows = []; smd_rows = []; formatted = []
    def fmt_cont(x):
        return f'{x.median():.2f} ({x.quantile(.25):.2f}, {x.quantile(.75):.2f})'
    for name in cs['FEATURES_49']:
        x = cleaned[name]
        binary = name in cs['CATEGORICAL'] and name != '室壁运动评分'
        if binary:
            check(set(x.dropna().unique()).issubset({0, 1}), f'binary_coding:{name}')
        for group, idx in groups.items():
            v = x.loc[idx]; z = v.dropna()
            row = {'source_column': name, 'group': group, 'n': len(v), 'observed_n': len(z),
                   'missing_n': int(v.isna().sum()), 'mean': z.mean(), 'sd': z.std(ddof=1),
                   'median': z.median(), 'q1': z.quantile(.25), 'q3': z.quantile(.75),
                   'count_1': int(z.eq(1).sum()) if binary else np.nan,
                   'fraction_1_observed': z.mean() if binary else np.nan,
                   'category_counts_json': json.dumps({str(float(k)): int(vv) for k, vv in
                       z.value_counts().sort_index().items()}, ensure_ascii=False) if name in cs['CATEGORICAL'] else '',
                   'role': 'binary' if binary else ('ordinal_numeric' if name == '室壁运动评分' else 'continuous')}
            old = q4.loc[(name, group)]
            for field in ['n', 'observed_n', 'missing_n']:
                check(row[field] == old[field], f'baseline_q4:{name}:{group}:{field}')
            for field in ['median', 'q1', 'q3']:
                check(np.isclose(row[field], old[field], rtol=2e-13, atol=2e-12, equal_nan=True), f'baseline_q4:{name}:{group}:{field}')
            if name in cs['CATEGORICAL']:
                check(json.loads(row['category_counts_json']) == json.loads(old.category_counts_json), f'baseline_q4:{name}:{group}:categories')
            long_rows.append(row)
        a = x[~y].dropna(); b = x[y].dropna()
        p0, p1 = a.mean(), b.mean()
        if binary:
            pbar = (p0 + p1) / 2
            den = np.sqrt(max(pbar * (1 - pbar), 1e-12))  # exact original Q3 binary definition
            fmt = lambda z: f'{int(z.eq(1).sum())} ({z.mean()*100:.1f}%)'
            definition = '(p_high-p_low)/sqrt(pbar*(1-pbar)); observed denominators; Q3 definition'
        else:
            den = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            fmt = fmt_cont
            definition = '(mean_high-mean_low)/sqrt((var_high+var_low)/2); ddof=1; observed cases'
        smd = (p1 - p0) / den if den > 0 else np.nan
        smd_rows.append({'source_column': name, 'low_observed_n': len(a), 'high_observed_n': len(b),
                         'mean_low': p0, 'mean_high': p1, 'sd_low': a.std(ddof=1), 'sd_high': b.std(ddof=1),
                         'smd_high_minus_low': smd, 'smd_definition': definition})
        formatted.append({'变量': name, '总体': fmt(x.dropna()), '低GS组': fmt(a), '高GS组': fmt(b),
                          '缺失数': int(x.isna().sum()), 'SMD': smd})
    check(len(long_rows) == 245 and len(smd_rows) == 49, 'all49_baseline_complete')
    save_csv(out, 'baseline_long.csv', long_rows)
    save_csv(out, 'baseline_smd.csv', smd_rows)
    save_csv(out, 'baseline_all49.csv', formatted)
    sub = cleaned['冠心病类型']
    subtype_masks = {label: sub.eq(code) for code, label in cs['SUBTYPE'].items()}
    subtype_masks['分型未知'] = sub.isna()
    check(np.all(np.sum(np.asarray(list(subtype_masks.values())), axis=0) == 1), 'subtype_exhaustive_exclusive')
    subtype_rows = []; subtype_fmt = []
    for label, mask in subtype_masks.items():
        p0 = mask[~y].mean(); p1 = mask[y].mean(); pbar = (p0 + p1) / 2
        smd = (p1 - p0) / np.sqrt(max(pbar * (1 - pbar), 1e-12))
        subtype_fmt.append({'变量': f'冠心病分型：{label}', '总体': f'{int(mask.sum())} ({mask.mean()*100:.1f}%)',
            '低GS组': f'{int(mask[~y].sum())} ({p0*100:.1f}%)', '高GS组': f'{int(mask[y].sum())} ({p1*100:.1f}%)',
            '缺失数': 0, 'SMD': smd})
        for group, idx in groups.items():
            subtype_rows.append({'subtype': label, 'group': group, 'count': int(mask.loc[idx].sum()),
                'denominator': len(idx), 'fraction': float(mask.loc[idx].mean()), 'smd_high_minus_low': smd,
                'denominator_definition': 'all members of group; unknown is separately counted'})
    save_csv(out, 'clinical_subtype_baseline.csv', subtype_rows)
    save_csv(out, 'baseline_original_style.csv', formatted + subtype_fmt)
    aggregate_rows = []
    for group, idx in groups.items():
        s = sub.loc[idx]
        for label, mask in [('ACS', s.isin([1, 2, 3])), ('MI', s.isin([1, 2])),
                            ('nonMI', s.isin([3, 4])), ('unknown', s.isna())]:
            aggregate_rows.append({'group': group, 'category': label, 'count': int(mask.sum()),
                                   'denominator': len(s), 'fraction': float(mask.mean())})
    save_csv(out, 'clinical_category_counts.csv', aggregate_rows)
    mapping = members.copy()
    mapping['gs'] = cleaned.loc[mapping.source_row, 'Gensini评分'].to_numpy()
    mapping['subtype_code'] = cleaned.loc[mapping.source_row, '冠心病类型'].to_numpy()
    mapping['subgroup'] = np.select([mapping.subtype_code.isin([1, 2]), mapping.subtype_code.isin([3, 4])],
                                   ['MI', 'nonMI'], default='unknown')
    save_csv(out, 'nonidentifying_row_mapping.csv', mapping)
    return aggregate_rows


def complete_hl(out, test, original_hl):
    rows = []; bins = []
    for j, name in enumerate(MODELS):
        y, p = test['y'], test['p'][j]
        chi2, df, pvalue = original_hl(y, p, 10)
        order = np.argsort(p)  # exact original default sorting and frozen input order
        sy, sp = y[order], p[order]
        edges = np.linspace(0, len(y), 11).astype(int)
        summands = []; crossing_ties = 0
        for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            if b <= a:
                continue
            obs = int(sy[a:b].sum()); exp = float(sp[a:b].sum()); n = int(b - a)
            term = ((obs - exp) ** 2 / exp if exp > 0 else 0) + ((obs - exp) ** 2 / (n - exp) if n - exp > 0 else 0)
            summands.append(term)
            if b < len(p) and sp[b - 1] == sp[b]:
                crossing_ties += 1
            bins.append({'model': name, 'bin': k + 1, 'n': n, 'observed_events': obs,
                         'expected_events': exp, 'probability_min': sp[a], 'probability_max': sp[b - 1],
                         'chi2_contribution': term})
        check(abs(sum(summands) - chi2) < 1e-10, f'hl_independent_components:{name}')
        check(df == 8, f'hl_df:{name}')
        check(sum(r['n'] for r in bins if r['model'] == name) == len(y), f'hl_total_n:{name}')
        rows.append({'model': name, 'n': len(y), 'events': int(y.sum()), 'hl_chi2': chi2, 'hl_df': df,
                     'hl_p': pvalue, 'hl_p_sf_precision_check': stats.chi2.sf(chi2, df),
                     'groups': 10, 'ties_crossing_group_boundaries': crossing_ties,
                     'multiple_testing_adjustment': 'none; descriptive supplementary check'})
    save_csv(out, 'hosmer_lemeshow.csv', rows)
    save_csv(out, 'hosmer_lemeshow_bins.csv', bins)
    return rows


def complete_threshold(out, aligned, original_threshold):
    oof = aligned['oof']; test = aligned['test']
    threshold = original_threshold(oof['y'], oof['p'][0], .85)
    # Independent exhaustive unique-threshold verification (not using ROC routine).
    candidates = np.unique(oof['p'][0])
    possible = [(v, classification(oof['y'], oof['p'][0], v)) for v in candidates]
    eligible = [(v, r) for v, r in possible if r['sensitivity'] >= .85]
    max_spec = max(r['specificity'] for _, r in eligible)
    chosen = max(v for v, r in eligible if np.isclose(r['specificity'], max_spec))
    check(threshold == chosen, 'triage_threshold_matches_exhaustive_unique_probabilities')
    rows = [{'model': MODELS[0], 'sample': kind, 'threshold_rule': 'OOF sensitivity>=0.85, max specificity, highest threshold on tie',
             **classification(a['y'], a['p'][0], threshold)} for kind, a in aligned.items()]
    for row in rows:
        check(row['tp'] + row['fp'] + row['tn'] + row['fn'] == row['n'], f'triage_counts:{row["sample"]}')
    check(next(r for r in rows if r['sample'] == 'oof')['sensitivity'] >= .85, 'triage_oof_target_met')
    save_csv(out, 'lr_high_sensitivity.csv', rows)
    thresholds = json.loads((PRIMARY / 'thresholds.json').read_text(encoding='utf-8'))
    full = []
    for j, name in enumerate(MODELS):
        thr = thresholds[name]['youden_probability_threshold']
        for kind, a in aligned.items():
            r = {'model': name, 'sample': kind, 'threshold_rule': 'Youden from frozen OOF',
                 **classification(a['y'], a['p'][j], thr)}
            if kind == 'test':
                for key in ('sensitivity', 'specificity'):
                    check(abs(r[key] - thresholds[name][f'test_{key}']) < 1e-14, f'youden_saved:{name}:{key}')
            full.append(r)
    save_csv(out, 'classification_youden.csv', full)
    return rows


def complete_dca(out, test, n_boot):
    y = test['y']; n = len(y)
    thresholds = np.arange(10, 81, dtype=float) / 100
    odds = thresholds / (1 - thresholds)
    rng = np.random.default_rng(DCA_SEED)
    indices = rng.integers(0, n, size=(n_boot, n))
    counts = np.asarray([np.bincount(i, minlength=n) for i in indices], dtype=float)
    check(np.all(counts.sum(axis=1) == n), 'dca_resample_sizes')
    prevalence = y.mean(); boot_prev = counts @ y / n
    all_point = prevalence - (1 - prevalence) * odds
    all_boot = boot_prev[:, None] - (1 - boot_prev[:, None]) * odds
    none_boot = np.zeros_like(all_boot)
    boot = []; point = []
    for j, name in enumerate(MODELS):
        positive = test['p'][j, :, None] >= thresholds
        contributions = positive * np.where(y[:, None] == 1, 1., -odds)
        point.append(contributions.mean(axis=0))
        boot.append(counts @ contributions / n)
    point = np.asarray(point); boot = np.asarray(boot)
    for b in range(min(n_boot, 3)):
        ii = indices[b]
        for j in (0, 1, 4, 7):
            for k in (0, 20, 30, 40, 50, 70):
                pos = test['p'][j, ii] >= thresholds[k]
                nb = (np.sum(pos & (y[ii] == 1)) - odds[k] * np.sum(pos & (y[ii] == 0))) / n
                check(abs(nb - boot[j, b, k]) < 2e-14, f'dca_explicit_resample:{b}:{j}:{k}')
    delta_lr = boot - boot[0:1]
    check(np.array_equal(delta_lr[0], np.zeros_like(delta_lr[0])), 'lr_vs_lr_bootstrap_exact_zero')
    all_ci = np.quantile(all_boot, [.025, .975], axis=0)
    rows = []
    for j, name in enumerate(MODELS):
        ci = np.quantile(boot[j], [.025, .975], axis=0)
        ci_all = np.quantile(boot[j] - all_boot, [.025, .975], axis=0)
        ci_lr = np.quantile(delta_lr[j], [.025, .975], axis=0)
        for k, threshold in enumerate(thresholds):
            rows.append({'model': name, 'threshold': threshold, 'n': n, 'events': int(y.sum()), 'n_bootstrap': n_boot,
                         'net_benefit': point[j, k], 'nb_ci_low': ci[0, k], 'nb_ci_high': ci[1, k],
                         'treat_all': all_point[k], 'treat_all_ci_low': all_ci[0, k], 'treat_all_ci_high': all_ci[1, k],
                         'treat_none': 0., 'delta_vs_all': point[j, k] - all_point[k],
                         'delta_vs_all_ci_low': ci_all[0, k], 'delta_vs_all_ci_high': ci_all[1, k],
                         'delta_vs_lr': point[j, k] - point[0, k],
                         'delta_vs_lr_ci_low': ci_lr[0, k], 'delta_vs_lr_ci_high': ci_lr[1, k]})
    frame = pd.DataFrame(rows)
    check(np.isfinite(frame.select_dtypes('number').to_numpy()).all(), 'all_dca_values_finite')
    check(len(frame) == 568, 'dca_8_by_71')
    save_csv(out, 'dca_curve.csv', frame)
    key = frame.loc[frame.threshold.isin([.30, .40, .50, .60])].copy()
    check(len(key) == 32, 'dca_key_8_by_4')
    save_csv(out, 'dca_key_thresholds.csv', key)
    reps = pd.DataFrame({'replicate': np.arange(n_boot), 'events': (counts @ y).astype(int),
                         'n': n, 'single_class': (boot_prev == 0) | (boot_prev == 1)})
    save_csv(out, 'dca_bootstrap_replicates.csv', reps)
    np.savez_compressed(out / 'dca_bootstrap_arrays.npz', indices=indices, source_rows=test['rows'],
                        thresholds=thresholds, model_names=np.asarray(MODELS), net_benefit=boot,
                        treat_all=all_boot, treat_none=none_boot)
    return {'seed': DCA_SEED, 'generator': 'numpy.default_rng / PCG64', 'n_resamples': n_boot,
            'sample_size': n, 'paired_across_all_models_and_thresholds': True,
            'indices_sha256': hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
            'single_class_resamples': int(reps.single_class.sum()),
            'single_class_policy': 'retained; net benefit is defined with either class absent',
            'interval': '2.5/97.5 percentile; NumPy linear quantile; pointwise, conditional on frozen development; no simultaneous coverage',
            'key_thresholds': key.to_dict('records')}


def complete_subgroups(out, cleaned, test, n_boot):
    sub = cleaned.loc[test['rows'], '冠心病类型'].to_numpy()
    masks = {'nonMI': np.isin(sub, [3, 4]), 'MI': np.isin(sub, [1, 2]), 'unknown': pd.isna(sub)}
    check(np.all(np.sum(np.asarray(list(masks.values())), axis=0) == 1), 'test_subgroup_partition_exhaustive')
    rng = np.random.default_rng(SUBGROUP_SEED)
    rows = []; bootstrap_rows = []; replicate_rows = []; arrays = {}; group_meta = []
    for group in GROUPS:
        mask = masks[group]; y = test['y'][mask]; scores = test['score'][:, mask]; n = len(y)
        ii = rng.integers(0, n, size=(n_boot, n)) if n else np.empty((n_boot, 0), dtype=int)
        single_class = np.asarray([len(np.unique(y[v])) < 2 for v in ii])
        arrays[f'{group}_indices'] = ii
        arrays[f'{group}_source_rows'] = test['rows'][mask]
        for b in range(n_boot):
            replicate_rows.append({'subgroup': group, 'replicate': b, 'n': n,
                                   'events': int(y[ii[b]].sum()), 'auc_defined': not bool(single_class[b])})
        for j, name in enumerate(MODELS):
            point = auc_u(y, scores[j])
            vals = np.asarray([auc_u(y[v], scores[j, v]) for v in ii])
            finite = np.isfinite(vals)
            check(np.array_equal(finite, ~single_class), f'subgroup_undefined_only_single_class:{group}:{name}')
            ci = np.quantile(vals[finite], [.025, .975]) if finite.any() else [np.nan, np.nan]
            rows.append({'subgroup': group, 'model': name, 'n': n, 'events': int(y.sum()),
                         'non_events': int(n - y.sum()), 'auc': point, 'ci_low': ci[0], 'ci_high': ci[1],
                         'n_bootstrap_requested': n_boot, 'n_bootstrap_valid': int(finite.sum()),
                         'n_bootstrap_single_class': int(single_class.sum()),
                         'status': 'estimable' if np.isfinite(point) else 'undefined_single_class_or_empty',
                         'purpose': 'post-hoc descriptive subgroup; unknown retained separately; not interaction testing'})
            for b, value in enumerate(vals):
                bootstrap_rows.append({'subgroup': group, 'model': name, 'replicate': b, 'auc': value})
        group_meta.append({'subgroup': group, 'n': n, 'events': int(y.sum()),
                           'single_class_resamples': int(single_class.sum()),
                           'indices_sha256': hashlib.sha256(ii.astype('<i8').tobytes()).hexdigest()})
    check(len(rows) == 24, 'subgroup_all8_including_unknown')
    save_csv(out, 'subgroup_auc.csv', rows)
    save_csv(out, 'subgroup_bootstrap_auc.csv', bootstrap_rows)
    save_csv(out, 'subgroup_bootstrap_replicates.csv', replicate_rows)
    np.savez_compressed(out / 'subgroup_bootstrap_indices.npz', **arrays)
    return {'seed': SUBGROUP_SEED, 'generator': 'numpy.default_rng / PCG64', 'group_order': GROUPS,
            'groups': group_meta, 'n_resamples': n_boot, 'paired_within_subgroup_across_models': True,
            'resampling': 'ordinary within-subgroup patient bootstrap, not outcome-stratified',
            'single_class_policy': 'AUC NA, excluded from percentile calculation only; valid denominator reported; never redraw',
            'inference': 'conditional percentile intervals, post-hoc descriptive; no interaction test, multiplicity adjustment, or full-development uncertainty',
            'results': rows}


def small_unit_checks(hl, threshold):
    yy = np.asarray([0, 1, 0, 1])
    pp = np.asarray([.1, .2, .2, .4])
    pairwise = np.mean(pp[yy == 1, None] > pp[yy == 0]) + .5 * np.mean(pp[yy == 1, None] == pp[yy == 0])
    check(auc_u(yy, pp) == pairwise, 'unit_auc_tied_pairs')
    check(np.isnan(auc_u(np.zeros(4, int), pp)), 'unit_auc_single_class_is_na')
    stat, df, pv = hl(yy, pp, 10)
    check(df == 2 and np.isfinite(stat) and 0 <= pv <= 1, 'unit_hl_small_n_nonempty_groups')
    chosen = threshold(yy, pp, .85)
    check(classification(yy, pp, chosen)['sensitivity'] >= .85, 'unit_threshold_ties')
    rng = np.random.default_rng(10)
    for k in range(10):
        y = rng.integers(0, 2, 31); p = rng.choice(np.linspace(.1, .9, 9), 31)
        if len(np.unique(y)) < 2:
            continue
        direct = np.mean(p[y == 1, None] > p[y == 0]) + .5 * np.mean(p[y == 1, None] == p[y == 0])
        check(abs(auc_u(y, p) - direct) < 2e-15, f'unit_auc_random_ties:{k}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'results/original_metrics'
    if args.self_test:
        out = out / 'selftest'
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / 'run.log', mode='w', encoding='utf-8')])
    n_boot = 20 if args.self_test else 1000
    inputs = [RAW, Q3_CODE, METRIC_CODE, NESTED_CODE,
              PRIMARY.parents[1] / 'run_manifest.json', PRIMARY / 'manifest.json',
              *[PRIMARY / name for name in ['membership.csv', 'predictions.csv', 'oof_predictions.csv', 'thresholds.json', 'audit.json', 'oof_audit.json', 'metrics.json']],
              Q4 / 'results/cohort_audit/baseline.csv', Q4 / 'results/cohort_audit/summary.json',
              Q4 / 'results/strict_primary_summary/primary_holdout_conditional.csv']
    before = {str(p): file_hash(p) for p in inputs}
    started = datetime.now(timezone.utc).isoformat()
    try:
        hl, hl_source = isolated_function(METRIC_CODE, 'hosmer_lemeshow')
        threshold, threshold_source = isolated_function(Q3_CODE, 'threshold_at_min_sensitivity')
        small_unit_checks(hl, threshold)
        cs, cleaned, members, aligned, corrections, offset_checks = load_inputs()
        logging.info('Source alignment verified: cohort=%s train=%s test=%s; identity columns not loaded',
                     len(cleaned), len(aligned['oof']['y']), len(aligned['test']['y']))
        categories = build_baseline(out, cs, cleaned, members)
        hl_rows = complete_hl(out, aligned['test'], hl)
        triage = complete_threshold(out, aligned, threshold)
        dca = complete_dca(out, aligned['test'], n_boot)
        subgroup = complete_subgroups(out, cleaned, aligned['test'], n_boot)
        after = {str(p): file_hash(p) for p in inputs}
        check(before == after, 'all_sources_unchanged_after_analysis')
        save_json(out, 'source_hashes.json', {'sources_before': before, 'sources_after': after,
                                            'script_sha256': file_hash(Path(__file__)),
                                            'original_hl_function': hl_source, 'original_threshold_function': threshold_source})
        summary = {'status': 'selftest_passed_not_publication_results' if args.self_test else 'complete',
                   'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
                   'no_model_training': True, 'n_bootstrap': n_boot,
                   'runtime': {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__,
                               'pandas': pd.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
                   'clinical_subtype_scope': 'newly authorized descriptive and post-hoc subgroup use only; frozen Q4 training/stratification unchanged',
                   'source_row_convention': 'zero-based physical data-record position excluding CSV header; no identifier fields read',
                   'cohort_n': len(cleaned), 'cohort_events': int(cleaned['Gensini评分'].gt(37).sum()),
                   'unknown_subtype_n': int(cleaned['冠心病类型'].isna().sum()), 'row_offset_negative_checks': offset_checks,
                   'clinical_category_counts': categories, 'range_cleaning_counts': corrections,
                   'baseline': {'n_features': 49, 'n_long_rows': 245, 'smd_direction': 'high GS minus low GS',
                                'binary_smd_denominator': 'sqrt(pbar*(1-pbar)); original Q3 definition',
                                'missing': 'observed denominators for predictors; unknown subtype separately counted; no imputation'},
                   'hosmer_lemeshow': {'definition': 'exact original function: np.argsort(p); linspace(0,n,11).astype(int) edges; df=max(nonempty_groups-2,1); original p=1-cdf',
                                      'input_order': 'frozen predictions.csv LR row order; verified identical for all 8 models',
                                      'tie_boundary_note': 'original equal-count bins may split tied probabilities; boundary counts reported',
                                      'results': hl_rows},
                   'lr_high_sensitivity': triage, 'dca': dca, 'subgroup_auc': subgroup,
                   'cautions': ['All intervals condition on frozen development and this test sample, not complete-development uncertainty.',
                                'DCA intervals are pointwise; neither simultaneous bands nor evidence of clinical impact.',
                                'Subgroups are post-hoc; no model retuning or interaction test was performed.',
                                'Unknown subtype is never merged into MI/nonMI; its results are displayed even if imprecise.',
                                'Three low nonzero GS records remain pending author eligibility verification.']}
        save_json(out, 'summary.json', summary)
        save_json(out, 'validation.json', {'status': 'passed', 'n_checks': len(CHECKS), 'failures': 0, 'checks': CHECKS})
        outputs = {p.name: file_hash(p) for p in sorted(out.iterdir()) if p.is_file() and p.name not in ['output_hashes.json', 'run.log']}
        save_json(out, 'output_hashes.json', outputs)
        logging.info('Completed %s bootstrap replicates; %s checks passed; no source changed.', n_boot, len(CHECKS))
        logging.info('Test subgroup sizes/events: %s', [(r['subgroup'], r['n'], r['events']) for r in subgroup['groups']])
        logging.info('LR high-sensitivity threshold and performance: %s', triage)
    except Exception as exc:
        logging.exception('Stopped on validation failure; outputs are not final.')
        save_json(out, 'validation.json', {'status': 'failed', 'n_checks': len(CHECKS), 'checks': CHECKS, 'error': repr(exc)})
        save_json(out, 'summary.json', {'status': 'failed', 'error': repr(exc), 'started_utc': started})
        raise


if __name__ == '__main__':
    main()
