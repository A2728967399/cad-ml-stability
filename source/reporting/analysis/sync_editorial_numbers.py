#!/usr/bin/env python3
"""Add traceable editorial number macros without changing displayed results.

--generate writes only this script's generated macro and audit files.
--check is strictly read-only. --apply-chapter is an explicit, separate operation
for the coordinating author: exact contextual substitutions, with byte-identical
macro-expanded text required. No model fitting or changes to upstream results.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
CHINESE = WORK / 'supplementary_analysis'
Q4 = WORK / 'strict_development'
CHAPTER = ROOT / 'sections/03_results.tex'
OLD = ROOT / 'generated/numbers.tex'
MACROS = ROOT / 'generated/editorial_numbers.tex'
MAP = ROOT / 'audit/editorial_macro_map.json'
REPORT = ROOT / 'audit/editorial_macro_validation.json'
REPORT_MD = ROOT / 'audit/editorial_macro_validation.md'
BACKUP = ROOT / 'revision_history/before_editorial_polish_20260908'
V_PATTERN = re.compile(r'\\V\{([^{}]+)\}')
DEFINITION = re.compile(r'\\expandafter\\def\\csname result@([^\\]+)\\endcsname\{([^{}]*)\}')
MODELS = {'lr': 'Logistic回归', 'rf': '随机森林', 'svm': 'SVM',
          'knn': 'K近邻', 'gbdt': '梯度提升', 'xgb': 'XGBoost',
          'lgb': 'LightGBM', 'mlp': '多层感知机'}


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    return sha_bytes(Path(path).read_bytes())


def read_text(path):
    # Do not normalize newlines: mechanical equivalence is byte-level.
    return Path(path).read_bytes().decode('utf-8-sig')


def fmt(value, digits=3, signed=False):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Missing or non-finite result cannot be formatted')
    if abs(value) < 0.5 * 10 ** -digits:
        value = 0.0
    return format(value, ('+' if signed else '') + f'.{digits}f')


def definitions(text):
    matches = DEFINITION.findall(text)
    result = dict(matches)
    if len(result) != len(matches):
        raise ValueError('Duplicate macro definitions')
    return result


def expand(text, values):
    def replace(match):
        key = match.group(1)
        if key not in values:
            raise ValueError('Unresolved numeric macro: ' + key)
        return values[key]
    return V_PATTERN.sub(replace, text)


def equation(text):
    a, b = text.index(r'\['), text.index(r'\]') + 2
    return text[a:b]


def write_atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    temp.write_bytes(content.encode('utf-8'))
    temp.replace(path)


class Builder:
    def __init__(self):
        self.sources, self.values, self.rules, self.skipped = {}, {}, [], []
        self.old = definitions(self.track_text(OLD))
        if len(self.old) != 632:
            raise ValueError('Expected 632 frozen pre-existing macros')
        if OLD.read_bytes() != (BACKUP / 'generated/numbers.tex').read_bytes():
            raise ValueError('Frozen numbers.tex differs from the pre-polish backup')
        self.track_text(BACKUP / 'generated/numbers.tex')
        self.track_text(Path(__file__))
        for path in [CHINESE / 'analysis/sync_evidence.py',
                     CHINESE / 'analysis/verify_results_numbers.py',
                     CHINESE / 'analysis/complete_original_models.py',
                     Q4 / 'analysis/nested_pipeline.py']:
            self.track_text(path)

    def track_text(self, path):
        path = Path(path)
        self.sources[str(path.relative_to(WORK)).replace('\\', '/')] = sha(path)
        return read_text(path)

    def csv(self, path):
        return list(csv.DictReader(self.track_text(path).splitlines()))

    def js(self, path):
        return json.loads(self.track_text(path))

    def add(self, key, raw, path, selector, field, digits=3, signed=False,
            transform='identity', existing=False, aggregate_inputs=None):
        if key in self.values:
            raise ValueError('Duplicate editorial key: ' + key)
        display = fmt(raw, digits, signed)
        if existing:
            if self.old.get(key) != display:
                raise ValueError(f'Existing macro disagrees with raw result: {key}')
        elif key in self.old:
            raise ValueError('New macro collides with frozen key: ' + key)
        entry = {'raw': float(raw), 'display': display, 'precision': digits,
                 'signed': signed, 'is_existing': existing,
                 'source': str(path.relative_to(WORK)).replace('\\', '/'),
                 'selector': selector, 'field': field, 'transform': transform}
        if aggregate_inputs is not None:
            entry['aggregate_inputs'] = aggregate_inputs
        self.values[key] = entry
        return key

    def rule(self, rule_id, before, after, keys):
        # Each rule has enough surrounding prose to disambiguate equal numbers.
        # No global literal-number replacement is used.
        expected = collections.Counter(keys)
        added = collections.Counter(V_PATTERN.findall(after)) - collections.Counter(V_PATTERN.findall(before))
        if added != expected:
            raise ValueError('Rule key accounting failed: ' + rule_id)
        full_values = self.old | {k: v['display'] for k, v in self.values.items()}
        if expand(before, full_values) != expand(after, full_values):
            raise ValueError('Rule changes displayed text: ' + rule_id)
        self.rules.append({'id': rule_id, 'before': before, 'after': after,
                           'expected_count': 1,
                           'occurrences': [dict(macro_key=k, macro_call=r'\V{' + k + '}',
                                                old_literal=self.values[k]['display'],
                                                **self.values[k]) for k in keys]})

    def series(self, rule_id, prefix, suffix, keys, percent=False):
        def join(items):
            return ', '.join(items[:-1]) + ', and ' + items[-1]
        unit = r'\%' if percent else ''
        old = [self.values[k]['display'] + unit for k in keys]
        new = [r'\V{' + k + '}' + unit for k in keys]
        self.rule(rule_id, prefix + join(old) + suffix,
                  prefix + join(new) + suffix, keys)

    def collect(self):
        primary_path = Q4 / 'results/strict_primary_summary/primary_holdout_conditional.csv'
        primary = self.csv(primary_path)
        by_model = {r['model']: r for r in primary}
        if len(primary) != 8 or set(by_model) != set(MODELS.values()):
            raise ValueError('AUC range requires exactly all 8 unique models')
        aucs = {r['model']: float(r['auc']) for r in primary}
        if not all(math.isfinite(x) for x in aucs.values()):
            raise ValueError('Non-finite AUC in range')
        for suffix, operation in [('min', min), ('max', max)]:
            self.add('ed-primary-auc-' + suffix, operation(aucs.values()), primary_path,
                     {'models': 'ALL 8, no hardwired extrema'}, 'auc',
                     transform=suffix + ' of unrounded stored AUCs', aggregate_inputs=aucs)
        self.rule('all-eight-auc-range', '(AUC) ranged from 0.657 to 0.696 across the 8 models',
                  r'(AUC) ranged from \V{ed-primary-auc-min} to \V{ed-primary-auc-max} across the 8 models',
                  ['ed-primary-auc-min', 'ed-primary-auc-max'])
        for field, literal in [('delong_p_unadjusted', '0.773'), ('p_holm', '1.000')]:
            key = self.add('ed-rf-' + field.replace('_', '-'), by_model[MODELS['rf']][field],
                           primary_path, {'model': MODELS['rf']}, field)
            prefix = 'unadjusted $P=' if field.startswith('delong') else 'Holm-adjusted $P='
            self.rule('rf-' + field, prefix + literal + '$', prefix + r'\V{' + key + '}$', [key])

        audit_path = Q4 / 'results/strict_primary_oof/checkpoints/primary_42/audit.json'
        rep = self.js(audit_path)['development']['final_representation']
        key = self.add('ed-primary-lasso-C', rep['C'], audit_path,
                       {'json_path': 'development.final_representation.C'}, 'C', 4)
        self.rule('lasso-selected-C', 'penalty parameter ($C=0.0433$)',
                  r'penalty parameter ($C=\V{' + key + '}$)', [key])

        recovery = CHINESE / 'results/model_recovery/full'
        freq_path = recovery / 'lasso_bootstrap_frequencies.csv'
        freq = {r['feature']: r for r in self.csv(freq_path)}
        features = [('wms', '室壁运动评分'), ('ldl', '低密度脂蛋白'), ('siri', 'SIRI'),
                    ('glucose', '葡萄糖'), ('albumin', '白蛋白'), ('ctnt', '血清肌钙蛋白T'), ('ck', '肌酸激酶')]
        keys = []
        for code, feature in features:
            if int(freq[feature]['bootstrap_completed']) != 300:
                raise ValueError('Conditional LASSO frequency is not from 300 samples')
            keys.append(self.add('ed-lasso-bootstrap-' + code + '-percent',
                                 100 * float(freq[feature]['selection_frequency']), freq_path,
                                 {'feature': feature}, 'selection_frequency', 1,
                                 transform='100 * stored selection_frequency'))
        self.series('conditional-lasso-frequency-five',
                    'selection frequencies for WMS, LDL-C, SIRI, glucose, and albumin were ',
                    ', respectively', keys[:5], True)
        self.rule('conditional-lasso-frequency-two',
                  r'those for serum cTnT and creatine kinase were 68.3\% and 66.3\%',
                  r'those for serum cTnT and creatine kinase were \V{' + keys[5] + r'}\% and \V{' + keys[6] + r'}\%', keys[5:])

        metrics = CHINESE / 'results/original_metrics'
        hl_path = metrics / 'hosmer_lemeshow.csv'
        hl = {r['model']: r for r in self.csv(hl_path)}
        hl_keys = {}
        for code, dp in [('svm', 4), ('knn', 4), ('xgb', 5), ('lgb', 4), ('lr', 3), ('rf', 3), ('mlp', 3)]:
            hl_keys[code] = self.add('ed-body-' + code + '-hl-p', hl[MODELS[code]]['hl_p'],
                                     hl_path, {'model': MODELS[code]}, 'hl_p', dp)
        self.series('hl-four-body-precision', 'Hosmer--Lemeshow $P$ values were ',
                    ' for SVM, k-nearest neighbors, XGBoost, and LightGBM',
                    [hl_keys[k] for k in ['svm', 'knn', 'xgb', 'lgb']])
        self.series('hl-three-body-precision',
                    'The corresponding values for logistic regression, random forest, and multilayer perceptron were ',
                    '.', [hl_keys[k] for k in ['lr', 'rf', 'mlp']])

        hi_path = metrics / 'lr_high_sensitivity.csv'
        hi = {r['sample']: r for r in self.csv(hi_path)}
        if set(hi) != {'test', 'oof'} or any(r['model'] != MODELS['lr'] for r in hi.values()):
            raise ValueError('Ambiguous high-sensitivity source rows')
        hi_keys = []
        hi_keys.append(self.add('ed-lr-highsens-threshold', hi['test']['threshold'], hi_path,
                                {'model': MODELS['lr'], 'sample': 'test'}, 'threshold'))
        for sample in ['oof', 'test']:
            for field in ['sensitivity', 'specificity']:
                hi_keys.append(self.add(f'ed-lr-highsens-{sample}-{field}', hi[sample][field], hi_path,
                                        {'model': MODELS['lr'], 'sample': sample}, field))
        for field in ['tp', 'fp', 'tn', 'fn']:
            hi_keys.append(self.add('ed-lr-highsens-test-' + field, hi['test'][field], hi_path,
                                    {'model': MODELS['lr'], 'sample': 'test'}, field, 0))
        old = ('The high-sensitivity threshold was 0.361, with sensitivity/specificity of 0.851/0.287 '
               'for training-set out-of-fold predictions and 0.832/0.315 in the held-out test set '
               '(134 true positives, 113 false positives, 52 true negatives, and 27 false negatives;')
        calls = [r'\V{' + k + '}' for k in hi_keys]
        new = ('The high-sensitivity threshold was ' + calls[0] + ', with sensitivity/specificity of '
               + calls[1] + '/' + calls[2] + ' for training-set out-of-fold predictions and '
               + calls[3] + '/' + calls[4] + ' in the held-out test set (' + calls[5]
               + ' true positives, ' + calls[6] + ' false positives, ' + calls[7]
               + ' true negatives, and ' + calls[8] + ' false negatives;')
        self.rule('high-sensitivity-operating-point', old, new, hi_keys)

        dca_path = metrics / 'dca_key_thresholds.csv'
        dca = self.csv(dca_path)
        lookup = {(r['model'], round(float(r['threshold']) * 100)): r for r in dca}
        if len(lookup) != len(dca) or len(dca) != 32:
            raise ValueError('DCA source must contain 8 models by 4 thresholds')
        def math_negative(s):
            return '$' + s + '$' if s.startswith('-') else s
        for threshold in [30, 40, 50, 60]:
            raw_threshold = lookup[(MODELS['lr'], threshold)]['threshold']
            self.add('ed-dca-threshold-' + str(threshold), raw_threshold, dca_path,
                     {'model': MODELS['lr'], 'threshold_percent': threshold}, 'threshold', 2)
            for code in ['lr', 'rf']:
                row = lookup[(MODELS[code], threshold)]
                keys = []
                for field in ['delta_vs_all', 'delta_vs_all_ci_low', 'delta_vs_all_ci_high']:
                    keys.append(self.add(f'{code}-dca-{threshold}-{field}', row[field], dca_path,
                                         {'model': MODELS[code], 'threshold_percent': threshold},
                                         field, existing=True))
                values = [self.values[k]['display'] for k in keys]
                old = math_negative(values[0]) + r' (95\% CI, ' + math_negative(values[1]) + '--' + math_negative(values[2]) + ')'
                calls = [('$' if v.startswith('-') else '') + r'\V{' + k + '}' + ('$' if v.startswith('-') else '') for k, v in zip(keys, values)]
                new = calls[0] + r' (95\% CI, ' + calls[1] + '--' + calls[2] + ')'
                self.rule(f'dca-{code}-{threshold}-versus-all', old, new, keys)
            row = lookup[(MODELS['rf'], threshold)]
            self.add(f'rf-dca-{threshold}-delta_vs_lr', row['delta_vs_lr'], dca_path,
                     {'model': MODELS['rf'], 'threshold_percent': threshold}, 'delta_vs_lr', existing=True)
        for threshold, prefix in [(30, 'At a threshold of '), (40, 'At '), (50, 'At '), (60, 'at ')]:
            key = 'ed-dca-threshold-' + str(threshold)
            self.rule(f'dca-threshold-context-{threshold}', prefix + self.values[key]['display'] + ',',
                      prefix + r'\V{' + key + '},', [key])
        keys = [f'rf-dca-{t}-delta_vs_lr' for t in [30, 40, 50, 60]]
        threshold_keys = [f'ed-dca-threshold-{t}' for t in [30, 40, 50, 60]]
        def comma_list(items):
            return ', '.join(items[:-1]) + ', and ' + items[-1]
        old = ('Paired differences in net benefit for random forest relative to logistic regression were '
               + comma_list([math_negative(self.values[k]['display']) for k in keys]) + ' at thresholds of '
               + comma_list([self.values[k]['display'] for k in threshold_keys]) + ', respectively;')
        calls = [('$' if self.values[k]['display'].startswith('-') else '') + r'\V{' + k + '}'
                 + ('$' if self.values[k]['display'].startswith('-') else '') for k in keys]
        new = ('Paired differences in net benefit for random forest relative to logistic regression were '
               + comma_list(calls) + ' at thresholds of '
               + comma_list([r'\V{' + k + '}' for k in threshold_keys]) + ', respectively;')
        self.rule('dca-rf-versus-lr-four-thresholds', old, new, keys + threshold_keys)

        sens_path = recovery / 'sensitivity_metrics.csv'
        sens = {(r['scenario'], r['model']): r for r in self.csv(sens_path)}
        keys = []
        for scenario, name in [('P50_locked_baseline', 'primary'), ('readmit_GS0_locked', 'readmit')]:
            inputs = {code: float(sens[(scenario, MODELS[code])]['auc']) for code in ['rf', 'lr']}
            keys.append(self.add('ed-sensitivity-' + name + '-rf-minus-lr', inputs['rf'] - inputs['lr'],
                                 sens_path, {'scenario': scenario, 'models': ['随机森林', 'Logistic回归']},
                                 'auc', signed=True, transform='RF auc - LR auc, unrounded inputs',
                                 aggregate_inputs=inputs))
        self.rule('sensitivity-signed-differences',
                  'changed from +0.004 in the primary analysis to +0.008,',
                  r'changed from \V{' + keys[0] + r'} in the primary analysis to \V{' + keys[1] + '},', keys)

        # Optional straightforward final-model summaries. The display equation
        # remains literal and is protected separately by exact-byte comparison.
        coef_path = recovery / 'logistic_coefficients.csv'
        coefs = {r['feature']: r for r in self.csv(coef_path)}
        terms = [('ck', '肌酸激酶'), ('ctnt', '血清肌钙蛋白T'), ('siri', 'SIRI'),
                 ('ldl', '低密度脂蛋白'), ('wms', '室壁运动评分')]
        coef_keys, or_keys = [], []
        for code, feature in terms:
            coef_keys.append(self.add('ed-lr-coef-' + code, coefs[feature]['coefficient'], coef_path,
                                      {'feature': feature}, 'coefficient'))
            or_keys.append(self.add('ed-lr-OR-' + code, coefs[feature]['OR_per_training_SD'], coef_path,
                                    {'feature': feature}, 'OR_per_training_SD', 2))
        self.series('final-lr-five-coefficients',
                    'Standardized coefficients for creatine kinase, serum cTnT, SIRI, LDL-C, and WMS were ',
                    ', respectively', coef_keys)
        self.series('final-lr-five-ORs',
                    'The corresponding odds ratios (ORs) per 1 training-set standard deviation increase were ',
                    '.', or_keys)
        key = self.add('ed-lr-intercept-body', coefs['intercept']['coefficient'], coef_path,
                       {'feature': 'intercept'}, 'coefficient', 4)
        self.rule('final-lr-intercept-body', 'with an intercept of $-0.0009$',
                  r'with an intercept of $\V{' + key + '}$', [key])
        shap_path = recovery / 'rf_shap_importance.csv'
        shap = {r['feature']: r for r in self.csv(shap_path)}
        terms = [('ctnt', '血清肌钙蛋白T'), ('siri', 'SIRI'), ('wms', '室壁运动评分'),
                 ('ck', '肌酸激酶'), ('lvesv', '心室收缩容量')]
        keys = [self.add('ed-rf-meanabs-shap-' + code, shap[feature]['mean_absolute_shap'], shap_path,
                         {'feature': feature}, 'mean_absolute_shap') for code, feature in terms]
        self.series('rf-five-meanabs-SHAP',
                    'mean absolute SHAP values for serum cTnT, SIRI, WMS, creatine kinase, and LVESV were ',
                    ', respectively', keys)

    def plan(self, text):
        out = text
        rules = []
        for rule in self.rules:
            count = out.count(rule['before'])
            record = dict(rule, observed_count=count, status='ready' if count == 1 else 'skipped')
            rules.append(record)
            if count != 1:
                self.skipped.append({'id': rule['id'], 'reason': 'Exact context missing or ambiguous', 'count': count})
                continue
            out = out.replace(rule['before'], rule['after'], 1)
        all_values = self.old | {k: v['display'] for k, v in self.values.items()}
        old_calls, new_calls = V_PATTERN.findall(text), V_PATTERN.findall(out)
        # This is a true subsequence check, not just matching counts or sets.
        iterator = iter(new_calls)
        retained = all(any(v == old for v in iterator) for old in old_calls)
        if not retained or len(old_calls) != 89:
            raise ValueError('Original 89 V occurrences must remain in order')
        if equation(text) != equation(out):
            raise ValueError('Display equation changed')
        expanded_before = expand(text, all_values).encode('utf-8')
        expanded_after = expand(out, all_values).encode('utf-8')
        if expanded_before != expanded_after:
            raise ValueError('Expanded chapter changed')
        return out, rules, {'expanded_chapter_byte_identical': True,
                            'expanded_chapter_sha256': sha_bytes(expanded_before),
                            'display_equation_byte_identical': True,
                            'display_equation_sha256': sha_bytes(equation(text).encode('utf-8')),
                            'original_V_occurrences': len(old_calls), 'planned_V_occurrences': len(new_calls),
                            'original_V_sequence_preserved': retained,
                            'additional_V_occurrences': len(new_calls) - len(old_calls)}


def validate_saved(require_applied=False):
    manifest = json.loads(read_text(MAP))
    if manifest.get('status') != 'READY':
        raise ValueError('Mapping is incomplete; see skipped rules')
    for relative, expected in manifest['sources_sha256'].items():
        if sha(WORK / relative) != expected:
            raise ValueError('Stale source hash: ' + relative)
    if sha(MACROS) != manifest['editorial_macros_sha256']:
        raise ValueError('Editorial macro file changed after generation')
    old, new = definitions(read_text(OLD)), definitions(read_text(MACROS))
    if len(old) != 632 or set(old) & set(new):
        raise ValueError('Frozen macro count or namespace violation')
    expected_new = {k: v['display'] for k, v in manifest['values'].items() if not v['is_existing']}
    if new != expected_new:
        raise ValueError('New definitions disagree with provenance map')
    for key, value in manifest['values'].items():
        if (old | new).get(key) != value['display']:
            raise ValueError('Mapped display mismatch: ' + key)
    current = CHAPTER.read_bytes()
    digest = sha_bytes(current)
    if digest == manifest['chapter_after_sha256']:
        stage = 'APPLIED'
    elif digest == manifest['chapter_before_sha256']:
        stage = 'READY_TO_APPLY'
    else:
        raise ValueError('Chapter changed since mapping; explicit --generate is required')
    if require_applied and stage != 'APPLIED':
        raise ValueError('Editorial substitutions have not been applied')
    text = current.decode('utf-8')
    if sha_bytes(expand(text, old | new).encode('utf-8')) != manifest['checks']['expanded_chapter_sha256']:
        raise ValueError('Expanded chapter no longer matches the verified text')
    if sha_bytes(equation(text).encode('utf-8')) != manifest['checks']['display_equation_sha256']:
        raise ValueError('Display equation differs')
    return manifest, {'status': 'PASS', 'chapter_stage': stage,
                      'new_macro_definitions': len(new), 'reused_existing_keys': manifest['reused_existing_key_count'],
                      'replacement_rules': len(manifest['rules']), **manifest['checks']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--generate', action='store_true')
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--apply-chapter', action='store_true')
    parser.add_argument('--require-applied', action='store_true', help='With --check, reject an unapplied plan')
    args = parser.parse_args()
    if args.require_applied and not args.check:
        parser.error('--require-applied is only valid with --check')
    if args.check or args.apply_chapter:
        manifest, result = validate_saved(args.require_applied)
        if args.apply_chapter and result['chapter_stage'] != 'APPLIED':
            before = CHAPTER.read_bytes()
            text = before.decode('utf-8')
            for rule in manifest['rules']:
                if text.count(rule['before']) != rule['expected_count']:
                    raise ValueError('Changed context during apply: ' + rule['id'])
                text = text.replace(rule['before'], rule['after'], rule['expected_count'])
            after = text.encode('utf-8')
            values = definitions(read_text(OLD)) | definitions(read_text(MACROS))
            if expand(before.decode('utf-8'), values).encode('utf-8') != expand(text, values).encode('utf-8'):
                raise ValueError('Mechanical substitution changed expanded bytes')
            if sha_bytes(after) != manifest['chapter_after_sha256'] or CHAPTER.read_bytes() != before:
                raise ValueError('Concurrent chapter change or unexpected output')
            write_atomic(CHAPTER, text)
            _, result = validate_saved(True)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    builder = Builder()
    builder.collect()
    chapter_bytes = CHAPTER.read_bytes()
    text = chapter_bytes.decode('utf-8')
    # Explicit regeneration after an applied plan reverses only that known plan;
    # it does not expand/remove the original 89 V calls.
    if MAP.exists():
        prior = json.loads(read_text(MAP))
        if sha_bytes(chapter_bytes) == prior.get('chapter_after_sha256'):
            for rule in reversed(prior['rules']):
                if text.count(rule['after']) != 1:
                    raise ValueError('Prior applied context is ambiguous')
                text = text.replace(rule['after'], rule['before'], 1)
    after, rules, checks = builder.plan(text)
    new_values = {k: v for k, v in builder.values.items() if not v['is_existing']}
    macro_text = '% Editorial additions only; frozen numbers.tex is not rewritten.\n'
    macro_text += '\n'.join(r'\expandafter\def\csname result@' + k + r'\endcsname{' + v['display'] + '}'
                            for k, v in sorted(new_values.items())) + '\n'
    manifest = {'schema_version': '1.0', 'status': 'READY' if not builder.skipped else 'INCOMPLETE',
                'scope': 'Selected result literals only; not all chapter numbers are automated.',
                'source_results_recomputed': False, 'display_precision_changed': False,
                'new_macro_count': len(new_values),
                'reused_existing_key_count': sum(v['is_existing'] for v in builder.values.values()),
                'frozen_existing_macro_count': len(builder.old),
                'sources_sha256': builder.sources,
                'editorial_macros_sha256': sha_bytes(macro_text.encode('utf-8')),
                'chapter_before_sha256': sha_bytes(text.encode('utf-8')),
                'chapter_after_sha256': sha_bytes(after.encode('utf-8')),
                'chapter_observed_during_generation_sha256': sha_bytes(chapter_bytes),
                'values': builder.values, 'rules': rules, 'skipped': builder.skipped, 'checks': checks,
                'not_automated': ['Full displayed logistic equation remains literal.',
                                  'Protocol constants, CI levels, zero-reference values, thresholds defining plot windows, '
                                  'cohort facts, and unlisted prose numbers remain literal.',
                                  'No full-pipeline uncertainty, clinical eligibility, or causal conclusion is certified.'],
                'precision_note': 'XGBoost body H-L P uses 5 decimals (0.00015); the old 4-decimal macro remains unchanged.'}
    for relative, digest in builder.sources.items():
        if sha(WORK / relative) != digest:
            raise ValueError('Source changed during generation: ' + relative)
    if CHAPTER.read_bytes() != chapter_bytes:
        raise ValueError('Chapter changed during generation')
    # Focused negative controls exercise fail-closed formatting and expansion.
    controls = {}
    for name, operation in [('reject_nonfinite', lambda: fmt(float('nan'))),
                            ('reject_undefined_macro', lambda: expand(r'\V{not-defined}', {}))]:
        try:
            operation()
        except ValueError:
            controls[name] = 'PASS'
        else:
            raise AssertionError('Negative control failed: ' + name)
    controls['all_eight_models_range'] = 'PASS' if len(builder.values['ed-primary-auc-min']['aggregate_inputs']) == 8 else 'FAIL'
    controls['XGB_body_precision'] = 'PASS' if builder.values['ed-body-xgb-hl-p']['display'] == '0.00015' else 'FAIL'
    controls['old_632_macros_byte_identical_to_backup'] = 'PASS'
    report = {'status': manifest['status'], 'checks': checks, 'negative_and_boundary_controls': controls,
              'new_macro_count': len(new_values), 'reused_existing_key_count': manifest['reused_existing_key_count'],
              'rule_count': len(rules), 'skipped': builder.skipped,
              'chapter_modified': False, 'frozen_numbers_modified': False,
              'chapter_before_sha256': manifest['chapter_before_sha256'],
              'chapter_planned_after_sha256': manifest['chapter_after_sha256']}
    write_atomic(MACROS, macro_text)
    write_atomic(MAP, json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    write_atomic(REPORT, json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Editorial numeric macro validation', '', 'Status: ' + manifest['status'] + '.', '',
             f'{len(new_values)} added definitions; {manifest["reused_existing_key_count"]} existing keys reused; '
             f'{len(rules)} exact-context rules; {checks["additional_V_occurrences"]} additional V occurrences planned.', '',
             'The generator did not edit the chapter, wrapper, frozen results, or the existing 632 definitions. '
             'The entire planned chapter expands to exactly the same UTF-8 bytes as before substitution. '
             'The complete displayed equation is unchanged. These checks do not automate or verify all chapter numbers.', '',
             'XGBoost H-L body P retains five decimals, 0.00015, without replacing its old four-decimal macro. '
             'DCA values reuse existing macros after independent lookup and formatting checks against the stored CSV.', '',
             '## Read-only build gate', '',
             '`python analysis/sync_editorial_numbers.py --check --require-applied`', '',
             'This command writes nothing and rejects stale source/code hashes, modified definitions, unresolved keys, '
             'or a chapter not matching the exact before/after plan. Explicit --generate refreshes the plan; '
             '--apply-chapter is a separate, coordinating-author-only operation.', '', '## Skipped contexts', '']
    lines += [json.dumps(item, ensure_ascii=False) for item in builder.skipped] or ['None.']
    write_atomic(REPORT_MD, '\n'.join(lines) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if builder.skipped:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
