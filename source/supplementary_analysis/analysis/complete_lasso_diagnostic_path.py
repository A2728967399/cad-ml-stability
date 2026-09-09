#!/usr/bin/env python3
"""S4 diagnostic only: 100 fixed-C fits on the frozen full training matrix.

No CV, no C/feature reselection, no changes to model_recovery/full or Q3/Q4.
The plotted CV scores are the original frozen fold-specific-preprocessing scores.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import sys
sys.dont_write_bytecode = True
import argparse
import json
import time
import warnings
import traceback
from pathlib import Path
from complete_original_models import ROOT, Q4, PRIMARY, CHECKPOINT, RAW, PIPELINE, load_pipeline


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true', help='Reuse an identical completed, hash-verified diagnostic run.')
    args = parser.parse_args()
    p = load_pipeline()
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    full = ROOT / 'results/model_recovery/full'
    output = ROOT / 'results/model_recovery/lasso_path'
    output.mkdir(parents=True, exist_ok=True)
    lock = output / 'run.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, 'w') as stream:
        stream.write(str(os.getpid()))
    state = {'state': 'running', 'started_utc': p.utc(), 'n_fits_completed': 0, 'pid': os.getpid()}
    started = time.monotonic()
    try:
        run = read(PRIMARY / 'run_manifest.json')
        config = run['config']
        versions = p.runtime_versions()
        if versions['packages'] != config['runtime']['packages'] or sys.version != config['runtime']['python']:
            raise RuntimeError('Frozen Q4 runtime mismatch')
        if p.file_hash(RAW) != config['raw_sha256']:
            raise RuntimeError('Raw hash mismatch')
        p.verify_checkpoint(CHECKPOINT, run['run_hash'])
        full_summary = read(full / 'summary.json')
        if full_summary['state'] != 'complete' or not full_summary['publication_eligible']:
            raise RuntimeError('Formal model recovery must already be complete')
        if not all(p.file_hash(full / name) == digest for name, digest in full_summary['output_sha256'].items()):
            raise RuntimeError('Formal recovery output hash mismatch')
        full_before = {path.name: p.file_hash(path) for path in full.iterdir() if path.is_file()}
        sources = [Path(__file__), Path(__file__).with_name('complete_original_models.py'), PIPELINE, RAW,
                   PRIMARY / 'run_manifest.json', CHECKPOINT / 'audit.json', CHECKPOINT / 'membership.csv',
                   full / 'summary.json', full / 'recovery_validation.json']
        hashes = {str(path): p.file_hash(path) for path in sources}
        identity = {'analysis': 'S4_fixed_training_LASSO_coefficient_path_v1', 'source_sha256': hashes,
                    'runtime': versions, 'n_C': 100, 'fit_rows': 'frozen seed42 training members only',
                    'preprocessing': 'restored full training values, fixed for all path fits',
                    'CV': 'no new CV; display frozen original fold-specific preprocessing CV scores',
                    'selection': 'no new selection; original C and support remain locked'}
        run_hash = p.object_hash(identity)
        manifest_path = output / 'run_manifest.json'
        if manifest_path.exists():
            if not args.resume or read(manifest_path)['run_hash'] != run_hash:
                raise RuntimeError('Existing diagnostic output requires --resume and identical identity')
            prior = read(output / 'summary.json')
            if prior['state'] == 'complete' and all(p.file_hash(output / name) == value for name, value in prior['output_sha256'].items()):
                print(json.dumps({'state': 'resumed_verified', 'run_hash': run_hash}), flush=True)
                return 0
            raise RuntimeError('Incomplete diagnostic run is not silently adopted; preserve it before restarting')
        if any(path.name != 'run.lock' for path in output.iterdir()):
            raise RuntimeError('Nonempty diagnostic directory without trusted manifest')
        p.atomic_json(manifest_path, {'run_hash': run_hash, 'created_utc': p.utc(), 'config': identity})
        state['run_hash'] = run_hash
        p.atomic_json(output / 'run_status.json', state)
        audit = read(CHECKPOINT / 'audit.json')
        rep = audit['development']['final_representation']
        meta = rep['preprocessing']
        X, y, _ = p.load_cohort(RAW)
        tr, te, split = p.split_members(X, y, 0, primary=True)
        if split != audit['membership']:
            raise RuntimeError('Membership reconstruction mismatch')
        prep = p.FoldPreprocessor()
        prep.kept, prep.impute, prep.logs = meta['features_kept'], meta['impute_values'], meta['log1p_features']
        prep.scaler = p.StandardScaler()
        prep.scaler.mean_, prep.scaler.scale_ = np.asarray(meta['scaler_mean']), np.asarray(meta['scaler_scale'])
        prep.scaler.n_features_in_, prep.scaler.n_samples_seen_ = len(prep.kept), len(tr)
        Z = prep.transform(X.iloc[tr])
        C_values = np.asarray(rep['path']['C'])
        if len(C_values) != 100 or not np.array_equal(C_values, np.asarray(config['lasso_cs'])):
            raise RuntimeError('Frozen 100-C path mismatch')
        rows, intercepts, counts, records = [], [], [], []
        selected_validation = None
        for i, c in enumerate(C_values):
            model = LogisticRegression(C=float(c), l1_ratio=1., solver='liblinear', max_iter=5000, random_state=42)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model.fit(Z, y[tr])
            rec = p.warning_record(caught)
            records.append({'C': float(c), **rec})
            coefficients = model.coef_[0]
            if not np.isfinite(coefficients).all():
                raise RuntimeError('Nonfinite path coefficients')
            rows.extend({'C': float(c), 'log10_C': float(np.log10(c)), 'feature': name,
                         'coefficient': float(value), 'is_locked_primary_C': float(c) == rep['C']}
                        for name, value in zip(prep.kept, coefficients))
            intercepts.append(float(model.intercept_[0]))
            counts.append(int(np.count_nonzero(coefficients)))
            if float(c) == rep['C']:
                expected = np.asarray([rep['coef'].get(name, 0.) for name in prep.kept])
                difference = float(np.max(np.abs(coefficients - expected)))
                actual_selected = [name for name, value in zip(prep.kept, coefficients) if value != 0]
                if difference > 1e-12 or actual_selected != rep['selected']:
                    raise RuntimeError('Locked-C coefficients/support fail original representation check')
                selected_validation = {'max_coefficient_difference': difference, 'selected_features': actual_selected,
                                       'passed': True, 'tolerance_absolute': 1e-12}
            state['n_fits_completed'] = i + 1
        if selected_validation is None:
            raise RuntimeError('Locked C not found on path')
        fold_auc = np.asarray(rep['path']['fold_auc'])
        means = fold_auc.mean(axis=0)
        sem = fold_auc.std(axis=0, ddof=1) / np.sqrt(fold_auc.shape[0])
        if not np.allclose(means, rep['path']['mean_auc'], rtol=0, atol=1e-14):
            raise RuntimeError('Saved CV means inconsistent with fold AUC')
        diagnostics = pd.DataFrame({'C': C_values, 'log10_C': np.log10(C_values), 'cv_mean_auc': means,
                                    'cv_standard_error': sem, 'cv_standard_deviation': fold_auc.std(axis=0, ddof=1),
                                    'n_nonzero_full_training': counts, 'intercept_full_training': intercepts,
                                    'is_locked_primary_C': C_values == rep['C']})
        p.atomic_csv(output / 'coefficient_path.csv', pd.DataFrame(rows))
        p.atomic_csv(output / 'cv_path.csv', diagnostics)
        p.atomic_json(output / 'warnings.json', {'per_C': records, 'aggregate': p.combine_warnings(records)})
        if full_before != {path.name: p.file_hash(path) for path in full.iterdir() if path.is_file()}:
            raise RuntimeError('Formal recovery products changed during diagnostics')
        if any(p.file_hash(path) != value for path, value in hashes.items()):
            raise RuntimeError('Diagnostic input changed during execution')
        output_hashes = {name: p.file_hash(output / name) for name in ['coefficient_path.csv', 'cv_path.csv', 'warnings.json', 'run_manifest.json']}
        summary = {'state': 'complete', 'run_hash': run_hash, 'n_model_fits': 100, 'n_training': len(tr),
                   'n_features': len(prep.kept), 'long_coefficient_rows': len(rows), 'n_C': 100,
                   'locked_C': rep['C'], 'initial_1se_C': rep['C_initial_1se'], 'selected_validation': selected_validation,
                   'frozen_cv_folds': int(fold_auc.shape[0]), 'one_se_cutoff': rep['one_se_cutoff'],
                   'CV_refit_performed': False, 'selection_repeated': False, 'test_data_used_for_fitting': False,
                   'warning_aggregate': p.combine_warnings(records), 'formal_recovery_outputs_unchanged': True,
                   'output_sha256': output_hashes,
                   'plotting_note': 'Use log10(C), or explicitly label inverse-C if displayed. C is software scaling, not a universal sample-size-independent lambda inverse.'}
        p.atomic_json(output / 'summary.json', summary)
        state.update(state='complete', finished_utc=p.utc(), elapsed_seconds=time.monotonic() - started)
        p.atomic_json(output / 'run_status.json', state)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        state.update(state='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        p.atomic_json(output / 'run_status.json', state)
        print(state['traceback'], file=sys.stderr)
        return 2
    finally:
        if lock.is_file() and lock.read_text().strip() == str(os.getpid()):
            lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
