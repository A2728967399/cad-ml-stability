"""Read-only independent recomputation of the locked 400-dataset simulation.

Does not fit any original models or write to results/simulation. Only the audit
JSON is written. Run under the recorded WSL Python environment.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
import sklearn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'simulation'
FAMILIES = ['LR', 'RF', 'SVM']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean_mcse(values):
    values = [float(x) for x in values]
    n = len(values)
    mean = math.fsum(values) / n
    sample_var = math.fsum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, math.sqrt(sample_var / n)


def independent_auc(y, score):
    """Mann--Whitney formulation, independent of sklearn's ROC integration."""
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    ranks = rankdata(score, method='average')
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def replay_generate(n, rho, signal, rng):
    """Replay RNG draws; calculate signal independently from explicit moments."""
    innovations = rng.normal(size=(n, 15))
    x = innovations.copy()
    for j in range(1, 15):
        x[:, j] = rho * x[:, j - 1] + math.sqrt(1 - rho**2) * innovations[:, j]
    if signal == 'linear':
        w = [1, -.8, .6, .4, -.4]
        variance = math.fsum(w[a] * w[b] * rho**abs(a-b) for a in range(5) for b in range(5))
        eta = sum(w[j] * x[:, j] for j in range(5)) / math.sqrt(variance)
    else:
        eta = 1.1 * (x[:, 0] * x[:, 1] - rho) / math.sqrt(1 + rho**2)
        eta += .65 * (x[:, 2]**2 - 1) / math.sqrt(2)
        eta += .45 * x[:, 4]
    y = rng.binomial(1, expit(1.25 * eta))
    return x, y, eta


def ranking_summary(records):
    scores = np.array([[next(m for m in r['models'] if m['model'] == name)['small_test_auc']
                        for name in FAMILIES] for r in records])
    ranks = rankdata(scores, axis=1, method='average')
    centered = ranks - ranks.mean(axis=1, keepdims=True)
    norms = np.sqrt((centered**2).sum(axis=1))
    if np.any(norms == 0):
        raise AssertionError('A whole-rank tie needs an explicit correlation convention')
    unit = centered / norms[:, None]
    correlations = unit @ unit.T
    n = len(records)
    estimate = float(correlations[np.triu_indices(n, 1)].mean())
    # Literal delete-one calculation; intentionally not the producer's sum shortcut.
    leave_one = []
    for i in range(n):
        retained = np.delete(np.delete(correlations, i, axis=0), i, axis=1)
        leave_one.append(float(retained[np.triu_indices(n-1, 1)].mean()))
    jack_mean = math.fsum(leave_one) / n
    mcse = math.sqrt((n-1)/n * math.fsum((v-jack_mean)**2 for v in leave_one))
    return estimate, mcse, int(sum(len(set(row)) < 3 for row in scores))


def main():
    config = json.loads((OUT / 'config.json').read_text(encoding='utf-8'))
    checkpoint = json.loads((OUT / 'COMPLETE.json').read_text(encoding='utf-8'))
    original_summary = {x['scenario']: x for x in json.loads((OUT/'summary.json').read_text())}
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((OUT/'replicates').glob('*.json'))]
    failures, checked, largest = [], 0, 0.

    def check(name, condition):
        nonlocal checked
        checked += 1
        if not bool(condition):
            failures.append(name)

    def close(name, actual, expected, tolerance=2e-12):
        nonlocal largest
        error = abs(float(actual)-float(expected))
        largest = max(largest, error)
        check(name, np.isfinite(error) and error <= tolerance)

    without_hash = {k: v for k, v in config.items() if k != 'hash'}
    check('canonical_config_hash', hashlib.sha256(json.dumps(without_hash, sort_keys=True).encode()).hexdigest() == config['hash'])
    check('locked_source_hash', sha(ROOT/'analysis'/'simulation.py') == config['source_sha256'])
    check('recorded_runtime_numpy', config['numpy'] == np.__version__)
    check('recorded_runtime_sklearn', config['sklearn'] == sklearn.__version__)
    check('recorded_runtime_python', config['python'] == platform.python_version())
    expected_design = [[n, rho, signal] for n, rho, signal in product([200, 800], [0., .6], ['linear', 'nonlinear'])]
    check('complete_factorial_design', config['scenarios'] == expected_design)
    check('replicates_50', config['replicates'] == 50)
    check('independent_evaluation_5000', config['evaluation_n'] == 5000)
    check('record_count_400', len(rows) == 400 and checkpoint['records'] == 400)
    check('complete_hash', checkpoint['config_hash'] == config['hash'])
    check('no_duplicate_keys', len({(r['scenario'], r['rep']) for r in rows}) == 400)
    check('unique_seeds', len({r['seed'] for r in rows}) == 400)
    check('exact_8_summaries', len(original_summary) == 8)
    by_scenario = defaultdict(list)
    warning_counts, chosen_param_counts = Counter(), defaultdict(Counter)
    hashes, max_bayes_error = {}, 0.
    for r in rows:
        label = f"{r['scenario']}/{r['rep']}"
        by_scenario[r['scenario']].append(r)
        design_id = config['scenarios'].index([r['n'], r['rho'], r['signal']])
        expected_seed = config['seed_base'] + 10000*design_id + r['rep']
        check(label+'/seed', r['seed'] == expected_seed)
        check(label+'/hash', r['config_hash'] == config['hash'])
        check(label+'/families', [m['model'] for m in r['models']] == FAMILIES)
        check(label+'/eval_n', r['evaluation_n'] == 5000)
        expected_cv = max(r['models'], key=lambda m: m['cv_auc'])
        expected_test = max(r['models'], key=lambda m: m['small_test_auc'])
        check(label+'/cv_choice', r['cv_selected'] == expected_cv['model'])
        check(label+'/test_choice', r['test_selected'] == expected_test['model'])
        close(label+'/cv_evaluation', r['cv_strategy_independent_auc'], expected_cv['independent_auc'])
        close(label+'/signed_optimism', r['test_selection_optimism'], expected_test['small_test_auc']-expected_test['independent_auc'])
        for m in r['models']:
            check(label+'/'+m['model']+'/valid_scores', all(np.isfinite(m[k]) and 0 <= m[k] <= 1 for k in ['cv_auc','small_test_auc','independent_auc']))
            warning_counts.update(m['warnings'])
            chosen_param_counts[m['model']].update([json.dumps(m['params'], sort_keys=True)])
        # Replay only generation, not original model fitting; two disjoint sequential RNG blocks.
        rng = np.random.default_rng(r['seed'])
        x, y, _ = replay_generate(r['n'], r['rho'], r['signal'], rng)
        xe, ye, etae = replay_generate(5000, r['rho'], r['signal'], rng)
        generated_auc = independent_auc(ye, etae)
        max_bayes_error = max(max_bayes_error, abs(generated_auc-r['bayes_auc']))
        close(label+'/replayed_bayes_auc', generated_auc, r['bayes_auc'])
        data_digest = hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
        eval_digest = hashlib.sha256(xe.tobytes()+ye.tobytes()).hexdigest()
        check(label+'/separate_data_draws', data_digest != eval_digest)
        hashes[label] = {'development_and_small_test_data_sha256': data_digest, 'independent_evaluation_sha256': eval_digest}
    check('unique_generated_datasets', len({v['development_and_small_test_data_sha256'] for v in hashes.values()}) == 400)
    check('unique_evaluation_datasets', len({v['independent_evaluation_sha256'] for v in hashes.values()}) == 400)
    recomputed = []
    for scenario, group in sorted(by_scenario.items()):
        original = original_summary[scenario]
        check(scenario+'/exact_reps', sorted(r['rep'] for r in group) == list(range(50)))
        entry = {'scenario': scenario, 'replicates': len(group)}
        for metric in ['bayes_auc','cv_strategy_independent_auc','test_selection_optimism']:
            mean, mcse = mean_mcse([r[metric] for r in group])
            close(scenario+'/'+metric+'/mean', original[metric+'_mean'], mean)
            close(scenario+'/'+metric+'/mcse', original[metric+'_mcse'], mcse)
            entry[metric+'_mean'], entry[metric+'_mcse'] = mean, mcse
        rho, se, ties = ranking_summary(group)
        close(scenario+'/rho_mean', original['ranking_rho_mean'], rho)
        close(scenario+'/rho_jackknife_mcse', original['ranking_rho_mcse'], se)
        entry.update(ranking_rho_mean=rho, ranking_rho_mcse=se, datasets_with_any_test_auc_tie=ties)
        for family in FAMILIES:
            wins = sum(r['test_selected'] == family for r in group)
            p = wins / len(group)
            mcse = math.sqrt(p*(1-p)/len(group))
            perf = [next(m for m in r['models'] if m['model']==family)['independent_auc'] for r in group]
            mean, mean_se = mean_mcse(perf)
            close(scenario+'/'+family+'/win', original[family+'_test_win_fraction'], p)
            close(scenario+'/'+family+'/win_mcse', original[family+'_test_win_mcse'], mcse)
            close(scenario+'/'+family+'/mean_auc', original[family+'_independent_auc'], mean)
            entry[family+'_test_win_fraction'], entry[family+'_test_win_mcse'] = p, mcse
            entry[family+'_independent_auc'], entry[family+'_independent_auc_mcse'] = mean, mean_se
        recomputed.append(entry)
    csv = pd.read_csv(OUT/'per_model.csv')
    check('per_model_csv_count', len(csv) == 1200)
    check('per_model_csv_unique', not csv.duplicated(['scenario','rep','model']).any())
    for r in rows:
        for m in r['models']:
            actual = csv[(csv.scenario == r['scenario']) & (csv.rep == r['rep']) & (csv.model == m['model'])]
            check('csv/key/'+r['scenario']+'/'+str(r['rep'])+'/'+m['model'], len(actual)==1)
            for key in ['cv_auc','small_test_auc','independent_auc']:
                close('csv/'+key, actual.iloc[0][key], m[key])
    csv_summary = pd.read_csv(OUT/'scenario_summary.csv').set_index('scenario')
    for original in original_summary.values():
        for key, value in original.items():
            if isinstance(value, (int,float)):
                close('summary_csv/'+key, csv_summary.loc[original['scenario'],key], value)
    # Check production generator against independently written signal equations.
    spec = importlib.util.spec_from_file_location('locked_simulation_for_audit', ROOT/'analysis'/'simulation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    moment_checks = []
    for rho, signal in product([0., .6], ['linear', 'nonlinear']):
        seed = 807 + int(100*rho) + (signal=='nonlinear')
        x0,y0,e0 = module.generate(1000,rho,signal,np.random.default_rng(seed))
        x1,y1,e1 = replay_generate(1000,rho,signal,np.random.default_rng(seed))
        check('generator_features/'+str((rho,signal)), np.array_equal(x0,x1))
        check('generator_labels/'+str((rho,signal)), np.array_equal(y0,y1))
        check('generator_signal/'+str((rho,signal)), np.allclose(e0,e1,rtol=0,atol=2e-15))
        x,y,e = replay_generate(60000,rho,signal,np.random.default_rng(4007+seed))
        theory = rho**np.abs(np.arange(15)[:,None]-np.arange(15)[None,:])
        covariance_error = float(np.max(np.abs(np.corrcoef(x,rowvar=False)-theory)))
        check('AR1_correlation_moment/'+str((rho,signal)), covariance_error < .025)
        if signal=='linear':
            check('linear_signal_variance/'+str(rho), abs(float(e.var())-1)<.035)
        moment_checks.append({'rho':rho,'signal':signal,'n':60000,'max_correlation_error':covariance_error,
                              'eta_mean':float(e.mean()),'eta_variance':float(e.var()),'event_fraction':float(y.mean()),
                              'direct_feature_indices_zero_based':[0,1,2,3,4] if signal=='linear' else [0,1,2,4]})
    audit = {'status':'PASS' if not failures else 'FAIL', 'utc':datetime.now(timezone.utc).isoformat(),
             'checks':checked,'failures':failures,'max_numeric_absolute_error':largest,
             'bayes_auc_replay_max_error':max_bayes_error,'original_models_refitted':0,
             'locked_source_sha256':config['source_sha256'],'verification_source_sha256':sha(__file__),
             'runtime':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},
             'source_files_sha256':{p.name:sha(p) for p in sorted(OUT.glob('*')) if p.is_file()},
             'replicate_warning_counts':dict(warning_counts),
             'chosen_parameter_counts':{k:dict(v) for k,v in chosen_param_counts.items()},
             'recomputed_summaries':recomputed,'generator_moment_checks':moment_checks,
             'data_generation_hashes':hashes,
             'interpretation_notes':['Binomial plug-in MCSE at observed proportions 0 or 1 is zero, not zero probability uncertainty.',
                 'Per-model independent-AUC MCSE was absent from producer summary and is supplied here without modifying producer outputs.',
                 'Original prediction scores and fitted models are not saved; fitted-model AUCs cannot be independently recomputed without refitting.',
                 'Sequential draws from each unique seeded RNG supply distinct fitting/test versus independent-evaluation data blocks.',
                 'Audit checks do not certify universal unbiasedness or externally validate the clinical study.']}
    destination = ROOT/'audit'/'simulation_verification.json'
    destination.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ['status','checks','failures','max_numeric_absolute_error','bayes_auc_replay_max_error','original_models_refitted']},ensure_ascii=False))
    raise SystemExit(bool(failures))


if __name__=='__main__':
    main()
