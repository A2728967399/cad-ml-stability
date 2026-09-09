"""ADEMP simulation: independent datasets, selection on CV, independent evaluation.

This supplementary experiment deliberately uses three representative families,
not the eight-model clinical grid. It is not an external clinical validation.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
import argparse, hashlib, json, platform, time, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import sklearn

ROOT=Path(__file__).resolve().parents[1]
MODELS=('LR','RF','SVM')

def generate(n,rho,signal,rng):
    p=15
    x=rng.normal(size=(n,p))
    for j in range(1,p): x[:,j]=rho*x[:,j-1]+np.sqrt(1-rho*rho)*x[:,j]
    if signal=='linear':
        weights=np.array([1.,-.8,.6,.4,-.4])
        cov=rho**np.abs(np.arange(5)[:,None]-np.arange(5)[None,:])
        eta=(x[:,:5]@weights)/np.sqrt(weights@cov@weights)
    else:
        # Var(X0*X1)=1+rho^2; centered square has variance 2.
        # X4 is separated from interaction block. Exact global SNR equality is
        # not assumed; Bayes AUC is measured separately in every scenario.
        eta=(1.1*(x[:,0]*x[:,1]-rho)/np.sqrt(1+rho*rho)
             +.65*(x[:,2]**2-1)/np.sqrt(2)+.45*x[:,4])
    prob=expit(1.25*eta)
    return x,rng.binomial(1,prob),eta

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def one(task):
    n,rho,signal,rep,config,out=task
    scenario=f'n{n}_rho{rho:g}_{signal}'
    dest=Path(out)/'replicates'/f'{scenario}_{rep:03}.json'
    if dest.exists():
        old=json.loads(dest.read_text())
        if old['config_hash']!=config['hash']: raise RuntimeError('Simulation checkpoint config changed')
        return old
    seed=202609080+config['scenarios'].index([n,rho,signal])*10000+rep
    rng=np.random.default_rng(seed)
    x,y,_=generate(n,rho,signal,rng)
    tr,te=train_test_split(np.arange(n),train_size=.75,stratify=y,random_state=seed)
    xe,ye,etae=generate(config['evaluation_n'],rho,signal,rng)
    cv=StratifiedKFold(3,shuffle=True,random_state=seed)
    specs={
      'LR':(LogisticRegression(max_iter=3000),{'model__C':[.1,1.,10.]}),
      'RF':(RandomForestClassifier(n_estimators=100,random_state=seed,n_jobs=1),
            {'model__max_depth':[3,None],'model__min_samples_leaf':[5,15]}),
      'SVM':(SVC(probability=False),{'model__C':[.1,1.,10.],'model__gamma':['scale',.05]})}
    rows=[]
    t=time.time()
    for model in MODELS:
        est,grid=specs[model]
        search=GridSearchCV(Pipeline([('scale',StandardScaler()),('model',est)]),grid,
                scoring='roc_auc',cv=cv,n_jobs=1,error_score='raise')
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            search.fit(x[tr],y[tr])
        f=search.decision_function if hasattr(search,'decision_function') else lambda z:search.predict_proba(z)[:,1]
        rows.append({'model':model,'cv_auc':float(search.best_score_),
                     'small_test_auc':float(roc_auc_score(y[te],f(x[te]))),
                     'independent_auc':float(roc_auc_score(ye,f(xe))),
                     'params':search.best_params_,
                     'warnings':{c:sum(type(w.message).__name__==c for w in ws) for c in sorted({type(w.message).__name__ for w in ws})}})
    chosen_cv=max(rows,key=lambda z:z['cv_auc'])
    chosen_test=max(rows,key=lambda z:z['small_test_auc'])
    record={'scenario':scenario,'n':n,'rho':rho,'signal':signal,'rep':rep,'seed':seed,
        'config_hash':config['hash'],'bayes_auc':float(roc_auc_score(ye,etae)),
        'cv_selected':chosen_cv['model'],'test_selected':chosen_test['model'],
        'cv_strategy_independent_auc':chosen_cv['independent_auc'],
        'test_selection_optimism':chosen_test['small_test_auc']-chosen_test['independent_auc'],
        'evaluation_n':config['evaluation_n'],'elapsed_seconds':time.time()-t,'models':rows}
    tmp=dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(dest)
    return record

def summarize(records,out):
    allrows=[]
    for r in records:
        for m in r['models']:
            allrows.append({k:r[k] for k in ['scenario','n','rho','signal','rep','bayes_auc','cv_selected','test_selected','cv_strategy_independent_auc','test_selection_optimism']}|m)
    d=pd.DataFrame(allrows)
    d.to_csv(out/'per_model.csv',index=False,encoding='utf-8-sig')
    summary=[]
    for key,g in d.groupby('scenario'):
        r=g.drop_duplicates('rep')
        entry={'scenario':key,'n':int(r.n.iloc[0]),'rho':float(r.rho.iloc[0]),'signal':r.signal.iloc[0],'replicates':len(r)}
        for metric in ['bayes_auc','cv_strategy_independent_auc','test_selection_optimism']:
            entry[metric+'_mean']=float(r[metric].mean())
            entry[metric+'_mcse']=float(r[metric].std(ddof=1)/np.sqrt(len(r)))
        ranks=g.pivot(index='rep',columns='model',values='small_test_auc').rank(axis=1)
        corrs=np.corrcoef(ranks.to_numpy())
        upper=np.triu_indices(len(ranks),1)
        entry['ranking_rho_mean']=float(corrs[upper].mean())
        # U-statistic jackknife MCSE across independent simulation datasets.
        loo=np.array([(corrs.sum()-len(ranks)-2*(corrs[i].sum()-1))/((len(ranks)-1)*(len(ranks)-2)) for i in range(len(ranks))])
        entry['ranking_rho_mcse']=float(np.sqrt((len(loo)-1)/len(loo)*np.sum((loo-loo.mean())**2)))
        for model in MODELS:
            p=float((r.test_selected==model).mean())
            entry[model+'_test_win_fraction']=p
            entry[model+'_test_win_mcse']=float(np.sqrt(p*(1-p)/len(r)))
            entry[model+'_independent_auc']=float(g[g.model==model].independent_auc.mean())
        summary.append(entry)
    pd.DataFrame(summary).to_csv(out/'scenario_summary.csv',index=False,encoding='utf-8-sig')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--replicates',type=int,default=50);ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--evaluation-n',type=int,default=5000);args=ap.parse_args()
    out=ROOT/'results'/'simulation';(out/'replicates').mkdir(parents=True,exist_ok=True)
    scenarios=[[n,rho,s] for n in [200,800] for rho in [0.,.6] for s in ['linear','nonlinear']]
    config={'scenarios':scenarios,'replicates':args.replicates,'evaluation_n':args.evaluation_n,'seed_base':202609080,
        'source_sha256':sha(__file__),'sklearn':sklearn.__version__,'numpy':np.__version__,'python':platform.python_version(),
        'p':15,'training_fraction':.75,'cv_folds':3,'families':list(MODELS),'status':'supplementary_simulation_not_clinical_validation'}
    config['hash']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    cp=out/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise RuntimeError('Existing simulation has different config')
    cp.write_text(json.dumps(config,indent=2),encoding='utf-8')
    tasks=[(n,rho,s,rep,config,str(out)) for n,rho,s in scenarios for rep in range(args.replicates)]
    records=[];t=time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(one,task) for task in tasks]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records)%10==0:print(f'{len(records)}/{len(tasks)} complete elapsed={time.time()-t:.1f}s',flush=True)
    summarize(records,out)
    (out/'COMPLETE.json').write_text(json.dumps({'records':len(records),'seconds':time.time()-t,'config_hash':config['hash']}),encoding='utf-8')
    print('Simulation complete',len(records),flush=True)

if __name__=='__main__':main()
