#!/usr/bin/env python3
"""Recover frozen seed42 models and supplement only the original paper's analyses.

No feature/C selection or repeated-split development is run. All Q3/Q4/raw inputs
are read only. Smoke is a software check, never a research bootstrap result.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import sys
sys.dont_write_bytecode = True
import argparse
import importlib.metadata
import importlib.util
import json
import platform
import time
import traceback
import warnings
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
Q4 = ROOT.parent / 'strict_development'
PIPELINE = Q4 / 'analysis/nested_pipeline.py'
PRIMARY = Q4 / 'results/strict_primary_oof'
CHECKPOINT = PRIMARY / 'checkpoints/primary_42'
RAW = ROOT.parent / 'clinical_pipeline/data/raw/source_clean.csv'
OLD_ANALYSIS = ROOT.parent / 'legacy_clinical/reanalysis/scripts/01_q3_analysis.py'
OLD_FIGURES = OLD_ANALYSIS.with_name('02_q3_figures_tables.py')
EXPECTED_PIPELINE_SHA = '0d1d41393b06f57f8a79499b0476c67b2021d28698f2106d0317e6f683693f8f'
THESIS16 = ['心率', '乳酸脱氢酶', '肌酸激酶同工酶', '血清肌钙蛋白T', 'SII', 'AISI',
            '白蛋白', '低密度脂蛋白', '高密度脂蛋白', '血氯', '血钠', '葡萄糖',
            'QTc间期', '心室收缩容量', '糖尿病史', '室壁运动评分']


def load_pipeline():
    import hashlib
    if hashlib.sha256(PIPELINE.read_bytes()).hexdigest() != EXPECTED_PIPELINE_SHA:
        raise RuntimeError('Frozen Q4 pipeline source mismatch; recovery refused')
    spec = importlib.util.spec_from_file_location('q4_recovery_readonly', PIPELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    parser.add_argument('--raw', type=Path, default=RAW)
    parser.add_argument('--resume', action='store_true', help='Resume only identical code/config/input hashes.')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    p = load_pipeline()
    import numpy as np
    import pandas as pd
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss
    if args.self_test:
        assert len(THESIS16) == len(set(THESIS16)) == 16
        assert set(THESIS16) <= set(p.FEATURES)
        assert len(p.FEATURES) == 49
        rng1, rng2 = np.random.default_rng(73), np.random.default_rng(73)
        assert np.array_equal(rng1.integers(0, 978, 978), rng2.integers(0, 978, 978))
        assert np.allclose(np.exp([0., np.log(2)]), [1., 2.])
        print(json.dumps({'self_test': 'PASS', 'assertions': 5, 'fits': 0, 'files_written': False}))
        return 0

    destination = ROOT / 'results/model_recovery' / args.mode
    destination.mkdir(parents=True, exist_ok=True)
    lock = destination / 'run.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'Existing run lock; inspect exact process before preserving a stale lock: {lock}')
    with os.fdopen(fd, 'w') as stream:
        stream.write(str(os.getpid()))
    status = {'state': 'running', 'mode': args.mode, 'publication_eligible': args.mode == 'full',
              'started_utc': p.utc(), 'pid': os.getpid(), 'events': []}
    started = time.monotonic()
    warning_rows = []

    def js(name, value):
        p.atomic_json(destination / name, value)

    def csv(name, data):
        p.atomic_csv(destination / name, data if isinstance(data, pd.DataFrame) else pd.DataFrame(data))

    def event(stage, **extra):
        record = {'utc': p.utc(), 'stage': stage, **extra}
        status['events'].append(record)
        status['current_stage'] = stage
        js('run_status.json', status)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def capture(stage, model_name, function):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            value = function()
        record = {'stage': stage, 'model': model_name, **p.warning_record(caught)}
        warning_rows.append(record)
        return value

    def fit(name, params, Z, y, stage):
        model = clone(p.MODELS[name][0]).set_params(**params)
        capture(stage, name, lambda: model.fit(Z, y))
        return model

    try:
        run = read_json(PRIMARY / 'run_manifest.json')
        config = run['config']
        actual_versions = p.runtime_versions()
        if actual_versions['packages'] != config['runtime']['packages'] or sys.version != config['runtime']['python']:
            raise RuntimeError('Python or package version differs from Q4; recovery refused')
        if p.file_hash(args.raw) != config['raw_sha256']:
            raise RuntimeError('Raw data SHA256 mismatch')
        if p.object_hash(config) != run['run_hash']:
            raise RuntimeError('Q4 run configuration hash mismatch')
        p.verify_checkpoint(CHECKPOINT, run['run_hash'])
        actual_base = {name: p.describe_params(model.get_params()) for name, (model, _) in p.MODELS.items()}
        if json.loads(p.canonical(actual_base)) != config['estimator_base_params']:
            raise RuntimeError('Estimator base defaults differ from frozen Q4')
        input_paths = [Path(__file__), PIPELINE, args.raw, OLD_ANALYSIS, OLD_FIGURES,
                       PRIMARY / 'run_manifest.json', *sorted(CHECKPOINT.glob('*'))]
        inputs = {str(path): p.file_hash(path) for path in input_paths if path.is_file()}
        recovery_config = {'analysis': 'frozen_seed42_original_scope_recovery_v1', 'mode': args.mode,
                           'n_lr_bootstrap': 1000 if args.mode == 'full' else 5,
                           'n_lasso_bootstrap': 300 if args.mode == 'full' else 5,
                           'lr_bootstrap_seed': 73, 'lasso_bootstrap_seed': 51,
                           'bootstrap_resampling': 'ordinary training-patient pairs, replacement; redraw single-class draws',
                           'bootstrap_conditioning': 'fixed training preprocessing, feature support and hyperparameters; no selection uncertainty',
                           'lasso_conditioning': 'fixed training preprocessing and final selected C; 49-variable refits without CV',
                           'sensitivity_scenarios': ['P50_locked_baseline', 'P33_locked', 'mean_locked', 'readmit_GS0_locked', 'thesis16_locked'],
                           'excluded_scope': 'No GS<2 exclusion rerun, no subtype inputs, no re-tuning or repeated-split development',
                           'shap_feature_perturbation': 'tree_path_dependent', 'shap_model_output': 'raw',
                           'shap_class': 1, 'shap_background': 'none; training node path counts',
                           'q4_run_hash': run['run_hash'], 'runtime': actual_versions,
                           'shap_version': importlib.metadata.version('shap'), 'inputs': inputs}
        run_hash = p.object_hash(recovery_config)
        manifest = {'config': recovery_config, 'run_hash': run_hash, 'created_utc': p.utc()}
        prior_path = destination / 'run_manifest.json'
        if prior_path.exists():
            prior = read_json(prior_path)
            if prior['run_hash'] != run_hash or not args.resume:
                raise RuntimeError('Existing output requires --resume and identical code/config/input hashes')
        else:
            if any(path.name != 'run.lock' for path in destination.iterdir()):
                raise RuntimeError('Untrusted nonempty output without manifest')
            js('run_manifest.json', manifest)
        status['run_hash'] = run_hash
        event('input_and_environment_verified')
        audit = read_json(CHECKPOINT / 'audit.json')
        representation = audit['development']['final_representation']
        meta, selected = representation['preprocessing'], representation['selected']
        if len(selected) != 13:
            raise RuntimeError('Expected the frozen 13-feature representation')
        X, y, cohort = p.load_cohort(args.raw)
        tr, te, split = p.split_members(X, y, 0, primary=True)
        if split != audit['membership']:
            raise RuntimeError('Reconstructed split does not match frozen membership metadata')
        stored_members = pd.read_csv(CHECKPOINT / 'membership.csv')
        expected_members = pd.DataFrame({'source_row': X.index, 'partition': np.where(np.isin(np.arange(len(X)), tr), 'train', 'test'), 'y': y})
        pd.testing.assert_frame_equal(stored_members, expected_members.reset_index(drop=True))
        # Restore learned preprocessing from stored numbers, not by selection.
        prep = p.FoldPreprocessor()
        prep.kept = list(meta['features_kept'])
        prep.impute = dict(meta['impute_values'])
        prep.logs = list(meta['log1p_features'])
        prep.meta = meta
        prep.scaler = p.StandardScaler()
        prep.scaler.mean_ = np.asarray(meta['scaler_mean'])
        prep.scaler.scale_ = np.asarray(meta['scaler_scale'])
        prep.scaler.var_ = prep.scaler.scale_ ** 2
        prep.scaler.n_features_in_ = len(prep.kept)
        prep.scaler.n_samples_seen_ = len(tr)
        # Independent training-only preprocessing check is not feature selection.
        refit_meta = p.FoldPreprocessor().fit(X.iloc[tr]).meta
        if p.object_hash(refit_meta) != p.object_hash(meta):
            raise RuntimeError('Stored preprocessing differs from exact training-only reconstruction')
        if prep.kept != p.FEATURES:
            raise RuntimeError('49-variable bootstrap requires all 49 eligible primary predictors')
        Zall = prep.transform(X)
        jj = [prep.kept.index(name) for name in selected]
        Ztr, Zte = Zall[tr][:, jj], Zall[te][:, jj]
        metrics = read_json(CHECKPOINT / 'metrics.json')
        params = {row['model']: row['params'] for row in metrics if row['scheme'] == 'retuned'}
        stored_predictions = pd.read_csv(CHECKPOINT / 'predictions.csv')
        recovered, checks, recovered_rows = {}, [], []
        # LR/RF first, and every subsequent supplemental fit is blocked unless
        # both per-patient probability and discrimination-score checks pass.
        for name in p.MODELS:
            model = fit(name, params[name], Ztr, y[tr], 'frozen_model_recovery')
            prob, score = capture('recovery_prediction', name, lambda: p.predict_pair(name, model, Zte))
            actual = pd.DataFrame({'source_row': X.index[te], 'y': y[te], 'p': prob, 'score': score}).sort_values('source_row')
            expected = stored_predictions.loc[(stored_predictions.model == name) & (stored_predictions.scheme == 'retuned')].sort_values('source_row')
            if not np.array_equal(actual.source_row, expected.source_row) or not np.array_equal(actual.y, expected.y):
                raise RuntimeError('Patient identity/outcome mismatch for ' + name)
            max_p = float(np.max(np.abs(actual.p.to_numpy() - expected.p.to_numpy())))
            max_score = float(np.max(np.abs(actual.score.to_numpy() - expected.score.to_numpy())))
            passed = max_p <= 1e-12 and max_score <= 1e-12
            checks.append({'model': name, 'n_patients_verified': len(te), 'max_absolute_probability_difference': max_p,
                           'max_absolute_score_difference': max_score, 'tolerance_absolute': 1e-12, 'passed': passed})
            if not passed:
                js('recovery_validation.json', {'state': 'failed', 'checks': checks})
                raise RuntimeError('Patient-level frozen prediction recovery failed for ' + name)
            recovered[name] = model
            for partition, index, matrix in [('train', tr, Ztr), ('test', te, Zte)]:
                pp, ss = capture('recovered_saved_prediction', name, lambda matrix=matrix: p.predict_pair(name, model, matrix))
                recovered_rows.append(pd.DataFrame({'source_row': X.index[index], 'partition': partition, 'y': y[index], 'p': pp, 'score': ss, 'model': name}))
        js('recovery_validation.json', {'state': 'passed', 'checks': checks, 'membership': split,
                                      'preprocessing_exactly_reconstructed': True, 'selected_features': selected})
        csv('recovered_predictions_CONTROLLED.csv', pd.concat(recovered_rows, ignore_index=True))
        js('preprocessing.json', {'final_representation': representation, 'model_params': params,
                                  'formula': 'z_j=(log1p(x*_j) or x*_j - training_mean_j)/training_scale_j; x* is train-only imputation',
                                  'note': 'Parentheses: transform first, then subtract mean, then divide scale. All predictors, including binary/ordinal, standardized.'})
        event('all_eight_models_recovered_per_patient', max_probability_difference=max(r['max_absolute_probability_difference'] for r in checks))

        # RF interpretation under the original no-background TreeExplainer setup.
        import shap
        shap_matrix = Zte if args.mode == 'full' else Zte[:12]
        shap_rows = X.index[te] if args.mode == 'full' else X.index[te][:12]
        explainer = capture('shap_explainer', '随机森林', lambda: shap.TreeExplainer(recovered['随机森林'], feature_perturbation='tree_path_dependent', model_output='raw'))
        values = capture('shap_values', '随机森林', lambda: explainer.shap_values(shap_matrix, check_additivity=True))
        values = values[1] if isinstance(values, list) else np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1]
        base = float(np.asarray(explainer.expected_value).reshape(-1)[1])
        shap_prediction = recovered['随机森林'].predict_proba(shap_matrix)[:, 1]
        additivity_error = float(np.max(np.abs(base + values.sum(axis=1) - shap_prediction)))
        if values.shape != shap_matrix.shape or additivity_error > 1e-8:
            raise RuntimeError('RF SHAP dimensionality/additivity mismatch')
        csv('rf_shap_importance.csv', pd.DataFrame({'feature': selected, 'mean_absolute_shap': np.abs(values).mean(axis=0)}).sort_values('mean_absolute_shap', ascending=False))
        long_shap = []
        for j, feature in enumerate(selected):
            long_shap.append(pd.DataFrame({'source_row': shap_rows, 'feature': feature, 'standardized_feature_value': shap_matrix[:, j],
                                          'shap_value': values[:, j], 'base_value': base, 'predicted_probability': shap_prediction}))
        csv('rf_shap_beeswarm_CONTROLLED.csv', pd.concat(long_shap, ignore_index=True))
        js('rf_shap_metadata.json', {'n_test': len(shap_rows), 'features': selected, 'explainer': 'TreeExplainer',
                                   'feature_perturbation': 'tree_path_dependent', 'background': 'no external background; training node counts',
                                   'model_output_argument': 'raw', 'empirically_verified_output_scale': 'class1 probability for sklearn RandomForestClassifier',
                                   'expected_value_class1': base, 'maximum_additivity_error': additivity_error,
                                   'shap_version': shap.__version__, 'interpretation': 'conditional fitted-model explanation, not causal effects or proof RF is best',
                                   'correlated_features_note': 'Attributions depend on tree-path allocation; do not interpret as uniquely identified independent feature effects.'})
        event('rf_shap_complete', n_test=len(shap_rows), max_additivity_error=additivity_error)

        # Sensitivities retain primary hyperparameters, never select again.
        gs = pd.read_csv(args.raw, encoding='utf-8-sig', usecols=['Gensini评分'])['Gensini评分'].to_numpy(float)
        p33, mean = float(np.quantile(gs[X.index], 1 / 3)), float(np.mean(gs[X.index]))
        if p33 != 23.:
            raise RuntimeError('Expected audited P33=23; source definition discrepancy')
        allowed = set(p.FEATURES + ['Gensini评分', '白细胞计数', '中性粒细胞计数'])
        whole = pd.read_csv(args.raw, encoding='utf-8-sig', usecols=lambda col: col in allowed).apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        den = whole['白细胞计数'] - whole['中性粒细胞计数']
        whole['dNLR'] = (whole['中性粒细胞计数'] / den).where((den > 0) & (whole['中性粒细胞计数'] >= 0))
        whole['TC-HDLDL'] = whole['总胆固醇'] - whole['高密度脂蛋白'] - whole['低密度脂蛋白']
        for feature, (lo, hi) in p.RANGES.items():
            bad = whole[feature].notna() & ~whole[feature].between(lo, hi)
            whole.loc[bad, feature] = np.nan
        Xwhole = whole[p.FEATURES].copy()
        pd.testing.assert_frame_equal(Xwhole.loc[X.index], X)
        ywhole = (gs > 37).astype(int)
        tr2, te2, membership2 = p.split_members(Xwhole, ywhole, 0, primary=True)
        prep2 = p.FoldPreprocessor().fit(Xwhole.iloc[tr2])
        if not set(selected) <= set(prep2.kept):
            raise RuntimeError('Fixed 13-feature specification not eligible in readmission training set')
        Z2tr = p.representation_matrix(prep2, selected, Xwhole.iloc[tr2])
        Z2te = p.representation_matrix(prep2, selected, Xwhole.iloc[te2])
        thesis_j = [prep.kept.index(f) for f in THESIS16]
        scenarios = [
            ('P50_locked_baseline', 37., Ztr, y[tr], Zte, y[te], X.index[te], selected, split, False),
            ('P33_locked', p33, Ztr, (gs[X.index[tr]] > p33).astype(int), Zte, (gs[X.index[te]] > p33).astype(int), X.index[te], selected, split, False),
            ('mean_locked', mean, Ztr, (gs[X.index[tr]] > mean).astype(int), Zte, (gs[X.index[te]] > mean).astype(int), X.index[te], selected, split, False),
            ('readmit_GS0_locked', 37., Z2tr, ywhole[tr2], Z2te, ywhole[te2], Xwhole.index[te2], selected, membership2, True),
            ('thesis16_locked', 37., Zall[tr][:, thesis_j], y[tr], Zall[te][:, thesis_j], y[te], X.index[te], THESIS16, split, False)]
        scenario_metrics, scenario_predictions, scenario_audit = [], [], []
        for scenario, cutoff, train_z, train_y, test_z, test_y, rows, features, membership, refitted_prep in scenarios:
            for name in p.MODELS:
                model = recovered[name] if scenario == 'P50_locked_baseline' else fit(name, params[name], train_z, train_y, 'sensitivity:' + scenario)
                prob, score = capture('sensitivity_prediction:' + scenario, name, lambda: p.predict_pair(name, model, test_z))
                scenario_metrics.append({'scenario': scenario, 'model': name, 'cutoff': cutoff, 'n_train': len(train_y), 'n_test': len(test_y),
                                         'events_train': int(train_y.sum()), 'events_test': int(test_y.sum()), 'n_features': len(features),
                                         'auc': float(roc_auc_score(test_y, score)), 'brier': float(brier_score_loss(test_y, prob)),
                                         **p.calibration_metrics(test_y, prob)})
                scenario_predictions.append(pd.DataFrame({'scenario': scenario, 'model': name, 'source_row': rows, 'y': test_y, 'p': prob, 'score': score}))
            scenario_audit.append({'scenario': scenario, 'cutoff': cutoff, 'features': features, 'membership': membership,
                                   'preprocessing_refit_on_scenario_training_only': refitted_prep,
                                   'feature_or_hyperparameter_reselection': False,
                                   'label': 'fixed-specification conditional sensitivity, not full redevelopment',
                                   'cutoff_provenance': 'historical full working-cohort definition, descriptive sensitivity'})
            event('sensitivity_complete', scenario=scenario)
        csv('sensitivity_metrics.csv', scenario_metrics)
        csv('sensitivity_predictions_CONTROLLED.csv', pd.concat(scenario_predictions, ignore_index=True))
        csv('readmission_membership_CONTROLLED.csv', pd.DataFrame({'source_row': Xwhole.index, 'partition': np.where(np.isin(np.arange(len(Xwhole)), tr2), 'train', 'test'), 'y': ywhole}))
        js('sensitivity_metadata.json', {'scenarios': scenario_audit, 'GS_mean': mean, 'GS_P33': p33, 'GS0_readmitted': int((gs == 0).sum()),
                                       'n_readmission_cohort': len(Xwhole), 'readmission_preprocessing': prep2.meta,
                                       'clinical_subtype_read_or_used': False,
                                       'qualification_unresolved': '3 records with 0<GS<2 remain in 1304 working cohort; no new exclusion scenario computed.'})

        # Ordinary bootstrap pairs: all preprocessing / C / support remain fixed.
        def bootstrap(kind, count, seed, matrix, base_model):
            checkpoint_path = destination / (kind + '_bootstrap_checkpoint.json')
            saved = read_json(checkpoint_path) if args.resume and checkpoint_path.exists() else None
            if saved and saved['run_hash'] != run_hash:
                raise RuntimeError('Bootstrap checkpoint identity mismatch')
            draws = saved['draws'] if saved else []
            records = saved['warning_records'] if saved else []
            rng = np.random.default_rng(seed)
            if saved:
                rng.bit_generator.state = saved['rng_state']
            attempts = saved['attempts'] if saved else 0
            while len(draws) < count:
                ii = rng.integers(0, len(y[tr]), len(y[tr]))
                attempts += 1
                if np.unique(y[tr][ii]).size != 2:
                    continue
                model = clone(base_model)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    model.fit(matrix[ii], y[tr][ii])
                coefficients = np.r_[model.intercept_[0], model.coef_[0]]
                if not np.isfinite(coefficients).all():
                    raise RuntimeError('Nonfinite bootstrap fit')
                draws.append(coefficients.tolist())
                records.append({'draw': len(draws), **p.warning_record(caught)})
                if len(draws) % 25 == 0 or len(draws) == count:
                    js(checkpoint_path.name, {'run_hash': run_hash, 'state': 'complete' if len(draws) == count else 'partial',
                                             'draws': draws, 'warning_records': records, 'rng_state': rng.bit_generator.state, 'attempts': attempts})
                    event(kind + '_bootstrap', completed=len(draws), requested=count)
            warning_rows.append({'stage': kind + '_bootstrap', 'model': type(base_model).__name__, **p.combine_warnings(records)})
            return np.asarray(draws), attempts

        lr = recovered['Logistic回归']
        lr_draws, lr_attempts = bootstrap('lr_conditional', recovery_config['n_lr_bootstrap'], 73, Ztr, lr)
        point = np.r_[lr.intercept_[0], lr.coef_[0]]
        lo, hi = np.quantile(lr_draws, [.025, .975], axis=0)
        coefficient_rows = []
        for j, name in enumerate(['intercept', *selected]):
            index = prep.kept.index(name) if j else None
            coefficient_rows.append({'feature': name, 'coefficient': float(point[j]), 'ci_lower': float(lo[j]), 'ci_upper': float(hi[j]),
                                     'OR_per_training_SD': float(np.exp(point[j])) if j else None,
                                     'OR_ci_lower': float(np.exp(lo[j])) if j else None, 'OR_ci_upper': float(np.exp(hi[j])) if j else None,
                                     'transform': ('log1p_then_standardize' if name in prep.logs else 'standardize') if j else 'none',
                                     'imputation': prep.impute[name] if j else None, 'training_mean_transformed': float(prep.scaler.mean_[index]) if j else None,
                                     'training_sd_transformed': float(prep.scaler.scale_[index]) if j else None,
                                     'bootstrap_completed': len(lr_draws), 'conditional_interval': True})
        csv('logistic_coefficients.csv', coefficient_rows)
        csv('logistic_bootstrap_draws.csv', pd.DataFrame(lr_draws, columns=['intercept', *selected]))
        js('logistic_formula.json', {'intercept': float(lr.intercept_[0]), 'coefficients': dict(zip(selected, lr.coef_[0].tolist())),
                                     'probability': 'expit(intercept + sum_j coefficient_j * z_j)',
                                     'z_j': '(training-imputed-and-prespecified-transformed x_j minus training mean_j) / training scale_j',
                                     'regularization': params['Logistic回归'], 'features': selected,
                                     'bootstrap_n': len(lr_draws), 'bootstrap_attempts': lr_attempts,
                                     'interval': 'pointwise 2.5th/97.5th empirical percentiles, conditional on selected support/preprocessing/C',
                                     'OR_caution': 'Per 1 training SD of transformed predictor; binary/ordinal predictors also standardized. Not causal or unpenalized adjusted-effect inference.'})
        lasso = LogisticRegression(C=representation['C'], l1_ratio=1., solver='liblinear', max_iter=5000, random_state=42)
        lasso_draws, lasso_attempts = bootstrap('lasso_fixed_C', recovery_config['n_lasso_bootstrap'], 51, Zall[tr], lasso)
        frequencies = pd.DataFrame({'feature': prep.kept, 'selection_count': (lasso_draws[:, 1:] != 0).sum(axis=0),
                                    'selection_frequency': (lasso_draws[:, 1:] != 0).mean(axis=0),
                                    'mean_absolute_coefficient': np.abs(lasso_draws[:, 1:]).mean(axis=0),
                                    'bootstrap_completed': len(lasso_draws), 'fixed_C': representation['C']})
        csv('lasso_bootstrap_frequencies.csv', frequencies.sort_values(['selection_frequency', 'mean_absolute_coefficient'], ascending=False))
        csv('lasso_bootstrap_draws.csv', pd.DataFrame(lasso_draws, columns=['intercept', *prep.kept]))
        js('lasso_bootstrap_metadata.json', {'n_features': 49, 'n_completed': len(lasso_draws), 'attempts': lasso_attempts,
                                           'C': representation['C'], 'preprocessing_fixed': True, 'C_reselected': False,
                                           'empty_selection_draws': int(np.sum(np.all(lasso_draws[:, 1:] == 0, axis=1))),
                                           'empty_model_fallback_in_bootstrap': False,
                                           'interpretation': 'Conditional fixed-C training bootstrap support frequency, not 200-split full-development stability.'})
        csv('warnings.csv', [{'stage': r['stage'], 'model': r['model'], 'total': r['total'], 'convergence': r['convergence'],
                              'categories': json.dumps(r['categories']), 'examples': json.dumps(r['examples'], ensure_ascii=False)} for r in warning_rows])
        js('warnings.json', {'records': warning_rows, 'aggregate': p.combine_warnings(warning_rows),
                             'counting': 'Each directly captured fit/prediction/SHAP call once; bootstrap aggregate includes each draw once; checkpoints repeat evidence, not fits.'})
        changed = [path for path, digest in inputs.items() if p.file_hash(path) != digest]
        if changed:
            raise RuntimeError('Read-only input changed during analysis: ' + str(changed))
        outputs = {path.name: p.file_hash(path) for path in sorted(destination.iterdir()) if path.is_file() and path.name not in {'run_status.json', 'run.lock', 'summary.json'}}
        summary = {'state': 'complete', 'mode': args.mode, 'publication_eligible': args.mode == 'full', 'run_hash': run_hash,
                   'n_cohort': len(X), 'n_train': len(tr), 'n_test': len(te), 'selected_features': selected,
                   'recovery_validated_models': len(checks), 'recovery_max_probability_difference': max(r['max_absolute_probability_difference'] for r in checks),
                   'lr_bootstrap_completed': len(lr_draws), 'lasso_bootstrap_completed': len(lasso_draws),
                   'sensitivity_model_rows': len(scenario_metrics), 'sensitivity_scenarios': 5,
                   'shap_test_patients': len(shap_rows), 'shap_max_additivity_error': additivity_error,
                   'GS_mean': mean, 'GS_P33': p33, 'n_readmission': len(Xwhole),
                   'warnings': p.combine_warnings(warning_rows), 'source_inputs_unchanged': True, 'output_sha256': outputs,
                   'private_outputs': 'All *_CONTROLLED.csv contain patient-level derived data. Do not publish or upload without separate authorization.',
                   'limitations': ['No feature/C reselection in any supplemental scenario or bootstrap.',
                                   'Conditional bootstrap intervals are not post-selection confidence intervals.',
                                   'Fixed-model sensitivity is not independent validation or complete redevelopment.',
                                   'No clinical-subtype predictor or stratification; low-GS qualification remains unresolved.']}
        js('summary.json', summary)
        status['state'] = 'complete'
        event('complete', elapsed_seconds=time.monotonic() - started)
        return 0
    except Exception as exc:
        status['state'] = 'failed'
        status['exception'] = f'{type(exc).__name__}: {exc}'
        status['traceback'] = traceback.format_exc()
        js('run_status.json', status)
        print(status['traceback'], file=sys.stderr)
        return 2
    finally:
        if lock.is_file() and lock.read_text().strip() == str(os.getpid()):
            lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
