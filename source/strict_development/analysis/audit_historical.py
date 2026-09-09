#!/usr/bin/env python3
"""Read-only independent audit of archived historical (non-strict-nested) results.

The source artifacts are never edited. All summaries are descriptive across
overlapping resamples; no ordinary independent-sample inference is performed.
Figure contract: two quantitative grids, Python only, 183 mm width, vector text.
Figure 1 separates rank movement from test-maximum distance; Figure 2 relates
structural variability to the observed, split-confounded performance spread.
"""
from __future__ import annotations

import ast
import hashlib
import itertools
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import scipy
from scipy import optimize, special, stats
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
SOURCE = PROJECT / "legacy_clinical" / "reanalysis"
RES = SOURCE / "results"
OUT = ROOT / "results" / "historical_audit"
AUDIT = ROOT / "audit"
FIG = ROOT / "figures"
MODELS = ["Logistic回归", "随机森林", "K近邻", "梯度提升", "SVM", "XGBoost", "LightGBM", "多层感知机"]
LABELS = {"Logistic回归": "LR", "随机森林": "RF", "K近邻": "KNN", "梯度提升": "GBDT", "SVM": "SVM", "XGBoost": "XGB", "LightGBM": "LGBM", "多层感知机": "MLP"}
FILES = {
    "fixed": "表_排序稳定性_逐次.csv",
    "retuned": "表_排序稳定性_重新调参_逐次.csv",
    "features_fixed": "表_特征稳定性_逐次.csv",
    "features_retuned": "表_排序稳定性_重新调参_特征.csv",
    "hp": "表_排序稳定性_重新调参_超参数.csv",
    "main_hp": "表_交叉验证与超参数.csv",
    "predictions": "predictions_deidentified.csv",
    "main_table": "表_主分析模型性能.csv",
}
WARNINGS = [
    "Historical 1304-case analysis used joint outcome/diagnostic-subtype stratification; not the new outcome-only analysis.",
    "Preprocessing preceded LASSO CV and supervised selection preceded model CV. Historical results are not strict nested-CV results.",
    "200 splits share patients and 19,900 split-pair correlations are dependent. Empirical percentiles are not confidence intervals.",
    "Distance to same-test maximum is a selected random benchmark, not true selection loss. 0.01/0.02 are exploratory numerical scales, not clinical equivalence margins.",
    "Historical fixed parameters originated in one overlapping development sample; fixed strategy is a conditional descriptive sensitivity analysis.",
    "Historical repeated predictions/member manifests were not saved. Split pairing is supported by equal seeds/code/raw-data and identical feature records, not by contemporaneous member hashes.",
    "Repeated retuned calibration is unavailable. Stored fixed calibration can be summarized but cannot be independently recomputed without its prediction rows.",
    "Historical cohort includes 3 low-GS cases whose angiographic eligibility awaits author verification; 1301 sensitivity requires a newly run pipeline.",
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def canonical(value):
    obj = json.loads(value) if isinstance(value, str) else value
    def norm(v):
        if isinstance(v, dict):
            return {k: norm(v[k]) for k in sorted(v)}
        if isinstance(v, (list, tuple)):
            return [norm(x) for x in v]
        if isinstance(v, str) and v.startswith("("):
            try:
                return norm(ast.literal_eval(v))
            except (ValueError, SyntaxError):
                pass
        return v
    return json.dumps(norm(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def desc(x):
    x = np.asarray(x, dtype=float)
    assert np.isfinite(x).all()
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x)>1 else 0.,
            "min": float(x.min()), "p025": float(np.quantile(x,.025)),
            "p25": float(np.quantile(x,.25)), "median": float(np.median(x)),
            "p75": float(np.quantile(x,.75)), "p975": float(np.quantile(x,.975)), "max": float(x.max())}


def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig", float_format="%.12g")


def delong(y, scores):
    """Independent direct positive-negative comparison implementation."""
    y = np.asarray(y)
    scores = np.atleast_2d(scores)
    pos, neg = scores[:,y==1], scores[:,y==0]
    u = (pos[:,:,None] > neg[:,None,:]).astype(float)
    u += .5 * (pos[:,:,None] == neg[:,None,:])
    vpos, vneg = u.mean(axis=2), u.mean(axis=1)
    covariance = np.atleast_2d(np.cov(vpos,ddof=1))/pos.shape[1] + np.atleast_2d(np.cov(vneg,ddof=1))/neg.shape[1]
    return u.mean(axis=(1,2)), covariance


def calibration(y,p):
    p = np.clip(np.asarray(p),1e-6,1-1e-6)
    x = np.column_stack([np.ones(len(p)),special.logit(p)])
    y = np.asarray(y)
    def fun(b):
        z = x@b
        return np.logaddexp(0,z).sum()-np.dot(y,z)
    def jac(b):
        return x.T@(special.expit(x@b)-y)
    def hess(b):
        q=special.expit(x@b)
        return x.T@(x*(q*(1-q))[:,None])
    fit=optimize.minimize(fun,[0.,1.],jac=jac,hess=hess,method="trust-exact",options={"gtol":1e-10})
    assert np.max(np.abs(jac(fit.x))) < 1e-5
    return float(fit.x[1]),float(fit.x[0])


def main():
    for p in (OUT,AUDIT,FIG):
        p.mkdir(parents=True,exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    inputs = [RES/v for v in FILES.values()] + [RES/"analysis_meta.json"] + list((SOURCE/"scripts").glob("*.py"))
    raw = PROJECT/"clinical_pipeline"/"data"/"raw"/"source_clean.csv"
    inputs += [raw,PROJECT/"clinical_pipeline"/"eval"/"fl_metrics.py"]
    hashes_before={str(p.relative_to(PROJECT)):sha(p) for p in inputs}
    meta=json.loads((RES/"analysis_meta.json").read_text(encoding="utf-8"))
    data={k:pd.read_csv(RES/v) for k,v in FILES.items()}
    rawdf=pd.read_csv(raw)
    gs=pd.to_numeric(rawdf["Gensini评分"],errors="raise")
    df=rawdf.loc[gs.ne(0)].reset_index(drop=True)
    scores=pd.to_numeric(df["Gensini评分"])
    y=(scores>37).astype(int).to_numpy()
    candidates=meta["preprocessing"]["features_kept"]
    assert len(rawdf)==1313 and len(df)==1304 and sum(y)==645 and len(candidates)==49
    cohort={"n_raw":len(rawdf),"n_excluded_gs0":int(gs.eq(0).sum()),"n":len(df),"events":int(y.sum()),
            "nonevents":int(len(df)-y.sum()),"cutoff":37.,"median_gs":float(scores.median()),
            "p33":float(scores.quantile(1/3)),"mean_gs":float(scores.mean()),
            "n_low_gs_pending":int(scores.lt(2).sum()),"low_gs_pending_values":scores[scores.lt(2)].tolist(),
            "n_if_low_gs_excluded":int(scores.ge(2).sum()),"n_initial_selected":len(meta["selected_features"])}
    strat=pd.to_numeric(df["冠心病类型"],errors="coerce").fillna(0).astype(int).astype(str)+"_"+pd.Series(y).astype(str)
    seeds=[]
    # These are reconstructed, not historical contemporaneous membership hashes.
    for split,seed in [(-1,42)]+[(i,1000+i) for i in range(200)]:
        tr,te=train_test_split(np.arange(len(df)),train_size=.75,random_state=seed,stratify=strat)
        if split == -1:
            main_test_y=y[te].copy()
            main_train_y=y[tr].copy()
        seeds.append({"split":split,"seed":seed,"n_train":len(tr),"n_test":len(te),"events_train":int(y[tr].sum()),"events_test":int(y[te].sum()),
                      "train_members_sha256_reconstructed":hashlib.sha256(np.sort(tr).astype("<i8").tobytes()).hexdigest(),
                      "test_members_sha256_reconstructed":hashlib.sha256(np.sort(te).astype("<i8").tobytes()).hexdigest()})
    save(pd.DataFrame(seeds),"reconstructed_membership_manifest.csv")
    cohort.update({"main_n_train":seeds[0]["n_train"],"main_n_test":seeds[0]["n_test"],"main_events_train":seeds[0]["events_train"],"main_events_test":seeds[0]["events_test"]})
    summary={"analysis_label":"historical_non_strict_nested_1304","created_utc":started,"warnings":WARNINGS,"cohort":cohort,
             "runtime_audit_not_historical":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__,"matplotlib":matplotlib.__version__},
             "provenance":{"raw_matches_meta":sha(raw)==meta["raw_data_sha256"],"analysis_script_matches_meta":sha(SOURCE/"scripts"/"01_q3_analysis.py")==meta["analysis_script_sha256"],"reconstructed_membership_status":"Reconstructed now from archived rules; not a contemporaneous audit record."}}
    assert summary["provenance"]["raw_matches_meta"] and summary["provenance"]["analysis_script_matches_meta"]
    pivots={}
    model_rows=[]; relative_rows=[]; gap_rows=[]; corr_rows=[]; all_auc_rows=[]
    for scheme in ("fixed","retuned"):
        d=data[scheme].copy()
        assert len(d)==1600 and not d.duplicated(["split","模型"]).any()
        assert set(d.split)==set(range(200)) and set(d["模型"])==set(MODELS)
        assert (d.groupby("split").size()==8).all() and d.auc.between(0,1).all()
        computed=d.groupby("split").auc.rank(ascending=False,method="min")
        assert np.array_equal(computed.to_numpy(),d["rank"].to_numpy())
        p=d.pivot(index="split",columns="模型",values="auc")[MODELS].sort_index()
        pivots[scheme]=p
        ranks=p.rank(axis=1,ascending=False,method="average")
        c=np.corrcoef(ranks.to_numpy())
        # Verify matrix method against scipy on representative pairs including ties.
        for i,j in [(0,1),(5,80),(100,199)]:
            assert abs(c[i,j]-stats.spearmanr(p.iloc[i],p.iloc[j]).statistic)<1e-12
        pair_idx=np.triu_indices(200,1)
        rho=c[pair_idx]
        assert len(rho)==19900
        corr_rows.extend({"scheme":scheme,"split_i":int(i),"split_j":int(j),"spearman_rho":float(r)} for i,j,r in zip(*pair_idx,rho))
        top=np.sort(p.to_numpy(),axis=1)
        top_gap=top[:,-1]-top[:,-2]
        full_range=top[:,-1]-top[:,0]
        gap_rows.extend({"scheme":scheme,"split":int(i),"leading_two_auc_gap":float(g),"eight_model_auc_range":float(r)} for i,g,r in zip(p.index,top_gap,full_range))
        summary[scheme]={"splits":200,"rows":len(d),"models":8,"ranking_rho":dict(desc(rho),negative_fraction=float((rho<0).mean()),greater_05_fraction=float((rho>.5).mean())),
                         "leading_two_gap":desc(top_gap),"eight_model_full_range":desc(full_range),"tied_winner_splits":int((d.groupby("split").apply(lambda x:(x["rank"]==1).sum())>1).sum())}
        rounded_ranks=p.round(12).rank(axis=1,ascending=False,method="average")
        rounded_corr=np.corrcoef(rounded_ranks.to_numpy())[pair_idx]
        summary[scheme]["float_tie_sensitivity"]={"rounding_auc_decimals":12,"changed_rank_cells":int(np.sum(rounded_ranks.to_numpy()!=ranks.to_numpy())),"tied_winner_splits_after_rounding":int((p.round(12).eq(p.round(12).max(axis=1),axis=0).sum(axis=1)>1).sum()),"rho_mean_after_rounding":float(rounded_corr.mean()),"negative_fraction_after_rounding":float((rounded_corr<0).mean())}
        means=p.mean().rank(ascending=False,method="min")
        for nm in MODELS:
            v=d[d["模型"]==nm]
            gap=p.max(axis=1)-p[nm]
            row={"scheme":scheme,"model":nm,"model_short":LABELS[nm],"mean_auc_rank":int(means[nm]),
                 **{"auc_"+k:v for k,v in desc(p[nm]).items()},**{"rank_"+k:v for k,v in desc(v["rank"]).items()},
                 "winner_fraction":float((v["rank"]==1).mean()),"top3_fraction":float((v["rank"]<=3).mean()),
                 **{"distance_test_max_"+k:v for k,v in desc(gap).items()},"within_001_fraction":float((gap<=.01).mean()),"within_002_fraction":float((gap<=.02).mean())}
            model_rows.append(row)
            delta=p[nm]-p["Logistic回归"]
            relative_rows.append({"scheme":scheme,"model":nm,**desc(delta),"positive_fraction":float((delta>0).mean()),"zero_fraction":float((delta==0).mean())})
            all_auc_rows.extend({"scheme":scheme,"split":int(i),"model":nm,"auc":float(p.loc[i,nm]),"rank":int(v.set_index("split").loc[i,"rank"]),"distance_test_max":float(gap.loc[i]),"delta_vs_lr":float(delta.loc[i])} for i in p.index)
    paired=[]
    for nm in MODELS:
        delta=pivots["retuned"][nm]-pivots["fixed"][nm]
        paired.append({"model":nm,**desc(delta),"positive_fraction":float((delta>0).mean()),"negative_fraction":float((delta<0).mean()),"exact_zero_fraction":float((delta==0).mean())})
    save(pd.DataFrame(model_rows),"model_summary.csv")
    save(pd.DataFrame(relative_rows),"paired_model_minus_lr.csv")
    save(pd.DataFrame(paired),"paired_retuned_minus_fixed.csv")
    save(pd.DataFrame(all_auc_rows),"auc_rank_distance_long.csv")
    save(pd.DataFrame(gap_rows),"leading_gap_and_full_range.csv")
    save(pd.DataFrame(corr_rows),"pairwise_rank_correlations.csv")
    summary["model_summary"]=model_rows
    summary["paired_model_minus_lr"]=relative_rows
    summary["paired_retuned_minus_fixed"]=paired
    # Audit feature sets and pairing, preserving all 49 candidate variables.
    ff=data["features_fixed"].sort_values("split").reset_index(drop=True)
    fr=data["features_retuned"].sort_values("split").reset_index(drop=True)
    assert len(ff)==len(fr)==200 and set(ff.split)==set(fr.split)==set(range(200))
    fs=[set(x.split("|")) if pd.notna(x) and x else set() for x in ff.features]
    rt=[set(x.split("|")) if pd.notna(x) and x else set() for x in fr.features]
    assert fs==rt and np.array_equal(ff.n_selected,fr.n_selected)
    assert all(len(s)==n for s,n in zip(fs,ff.n_selected)) and set.union(*fs)<=set(candidates)
    count=Counter(itertools.chain.from_iterable(fs))
    freq=pd.DataFrame([{"feature":f,"feature_display":"左心室收缩末期容积（LVESV）" if f=="心室收缩容量" else f,"selected_count":count[f],"frequency":count[f]/200} for f in candidates]).sort_values(["frequency","feature"],ascending=[False,True])
    jrows=[{"split_i":i,"split_j":j,"jaccard":len(fs[i]&fs[j])/len(fs[i]|fs[j])} for i,j in itertools.combinations(range(200),2)]
    jac=pd.DataFrame(jrows)
    summary["features"]={"n_selected":desc(ff.n_selected),"n_candidates":49,"ever_selected":sum(v>0 for v in count.values()),"at_least_080":int((freq.frequency>=.8).sum()),"at_least_050":int((freq.frequency>=.5).sum()),"jaccard":desc(jac.jaccard),"fixed_retuned_feature_sets_identical":True,"top_features":freq.head(10).to_dict("records")}
    save(freq,"feature_selection_frequencies.csv")
    save(jac,"pairwise_feature_jaccard.csv")
    save(ff,"feature_sets_per_split.csv")
    # Parameter vectors, not marginal counts; canonicalize tuple/list differences.
    hp=data["hp"].copy()
    assert len(hp)==1600 and not hp.duplicated(["split","模型"]).any()
    hp["canonical_parameters"]=hp["最佳参数"].map(canonical)
    main_hp=data["main_hp"].set_index("模型")["最佳参数"].map(canonical)
    hp=hp.merge(data["retuned"][["split","模型","auc"]],on=["split","模型"],validate="one_to_one")
    hp_rows=[]; hp_combos=[]
    for nm in MODELS:
        h=hp[hp["模型"]==nm]
        assert len(h)==200 and set(h.split)==set(range(200))
        counts=h.canonical_parameters.value_counts()
        hp_rows.append({"model":nm,"n_unique_combinations":len(counts),"main_parameters":main_hp[nm],"main_recurrence_count":int(counts.get(main_hp[nm],0)),"main_recurrence_fraction":float(counts.get(main_hp[nm],0)/200),"modal_parameters":counts.index[0],"modal_count":int(counts.iloc[0]),"modal_fraction":float(counts.iloc[0]/200),"entropy_nats":float(stats.entropy(counts/200))})
        for params,v in h.groupby("canonical_parameters"):
            hp_combos.append({"model":nm,"canonical_parameters":params,"count":len(v),"frequency":len(v)/200,"is_main_configuration":params==main_hp[nm],**{"auc_"+k:x for k,x in desc(v.auc).items()}})
    summary["hyperparameters"]=hp_rows
    save(pd.DataFrame(hp_rows),"hyperparameter_summary.csv")
    save(pd.DataFrame(hp_combos),"hyperparameter_combinations_with_auc.csv")
    save(hp,"hyperparameters_per_split.csv")
    structure=[]
    for nm in MODELS:
        h=hp[hp["模型"]==nm].set_index("split").sort_index()
        changed=h.canonical_parameters!=main_hp[nm]
        delta=pivots["retuned"][nm]-pivots["fixed"][nm]
        for changed_value in (False,True):
            z=delta.loc[changed==changed_value]
            if len(z):
                structure.append({"model":nm,"parameter_configuration_changed":changed_value,**desc(z),"absolute_delta_le_001_fraction":float((z.abs()<=.01).mean()),"absolute_delta_le_002_fraction":float((z.abs()<=.02).mean()),"interpretation":"Same split paired difference; chosen configurations were not randomized, so this is descriptive."})
    summary["parameter_change_performance"]=structure
    save(pd.DataFrame(structure),"parameter_change_paired_performance.csv")
    compact=[]
    for scheme in ("retuned","fixed"):
        for r in sorted([r for r in model_rows if r["scheme"]==scheme],key=lambda r:r["mean_auc_rank"]):
            delta=next(x for x in relative_rows if x["scheme"]==scheme and x["model"]==r["model"])
            compact.append({"方案":"历史逐次调参" if scheme=="retuned" else "历史固定参数","模型":r["model_short"],"平均AUC（SD）":f'{r["auc_mean"]:.3f}（{r["auc_sd"]:.3f}）',"名次中位（IQR）":f'{r["rank_median"]:g}（{r["rank_p25"]:g}–{r["rank_p75"]:g}）',"居首比例":f'{100*r["winner_fraction"]:.1f}%',"对LR平均ΔAUC":f'{delta["mean"]:+.4f}',"ΔAUC经验P2.5–P97.5":f'{delta["p025"]:+.4f}–{delta["p975"]:+.4f}',"距测试最高≤0.01":f'{100*r["within_001_fraction"]:.1f}%',"距测试最高≤0.02":f'{100*r["within_002_fraction"]:.1f}%'})
    save(pd.DataFrame(compact),"manuscript_ready_repeated_table.csv")
    # Independent single-development AUC, Brier, calibration and paired DeLong.
    pred=data["predictions"]
    test=pred[pred.split=="test"].copy()
    assert len(test)==326*8 and not test.duplicated(["row","模型"]).any()
    assert not pred.duplicated(["split","row","模型"]).any()
    main_rows=[]; main_differences=[]; reference=test[test["模型"]=="Logistic回归"].set_index("row").sort_index()
    assert np.array_equal(reference.y,main_test_y)
    for nm in MODELS:
        oof=pred[(pred.split=="train_outer_oof")&(pred["模型"]==nm)].sort_values("row")
        assert len(oof)==978 and np.array_equal(oof.y,main_train_y)
    summary["provenance"]["main_saved_test_and_oof_outcomes_match_reconstructed_split"]=True
    old=data["main_table"].set_index("模型")
    for nm in MODELS:
        p=test[test["模型"]==nm].set_index("row").sort_index()
        assert p.index.equals(reference.index) and np.array_equal(p.y,reference.y)
        auc=roc_auc_score(p.y,p.score)
        brier=float(np.mean((p.p-p.y)**2))
        slope,intercept=calibration(p.y,p.p)
        au,cv=delong(p.y,p.score.to_numpy())
        assert abs(auc-au[0])<1e-12 and abs(auc-old.loc[nm,"auc"])<5.1e-7 and abs(brier-old.loc[nm,"brier"])<5.1e-7
        # Exact optimization differs slightly from the historical default solver tolerance.
        assert abs(slope-old.loc[nm,"cal_slope"])<.003 and abs(intercept-old.loc[nm,"cal_intercept"])<.003
        se=float(np.sqrt(cv[0,0])); threshold=float(p.threshold.iloc[0]); predicted=p.p>=threshold
        main_rows.append({"model":nm,"n":len(p),"events":int(p.y.sum()),"auc":auc,"auc_conditional_ci_lo":auc-1.95996398454*se,"auc_conditional_ci_hi":auc+1.95996398454*se,"brier":brier,"cal_slope_mle":slope,"cal_intercept_mle":intercept,"historical_cal_slope_stored":float(old.loc[nm,"cal_slope"]),"historical_cal_intercept_stored":float(old.loc[nm,"cal_intercept"]),"threshold_development_oof":threshold,"sensitivity":float(predicted[p.y==1].mean()),"specificity":float((~predicted[p.y==0]).mean())})
        if nm!="Logistic回归":
            au,cv=delong(p.y,np.vstack([p.score,reference.score]))
            delta=float(au[0]-au[1]); se=float(np.sqrt(cv[0,0]+cv[1,1]-2*cv[0,1]))
            main_differences.append({"model":nm,"delta_auc_vs_lr":delta,"se":se,"conditional_ci_lo":delta-1.95996398454*se,"conditional_ci_hi":delta+1.95996398454*se,"p_unadjusted":float(2*stats.norm.sf(abs(delta/se)))})
    ps=np.array([r["p_unadjusted"] for r in main_differences]); order=np.argsort(ps); adj=np.empty_like(ps)
    adj[order]=np.minimum(1,np.maximum.accumulate(ps[order]*np.arange(len(ps),0,-1)))
    for r,a in zip(main_differences,adj):r["p_holm"]=float(a)
    summary["single_development"]={"models":main_rows,"paired_vs_lr":main_differences,"calibration_note":"Exact MLE recalculation; small differences from archived sklearn default tolerance are numerical, not new patient results.","auc_ci_note":"Conditional DeLong CI given fitted models and this holdout, not development-strategy uncertainty."}
    save(pd.DataFrame(main_rows),"single_development_recomputed.csv")
    save(pd.DataFrame(main_differences),"single_development_paired_delong.csv")
    # Archived fixed-only calibration values have no recoverable repeated probabilities.
    cal=[]
    for nm in MODELS:
        v=data["fixed"][data["fixed"]["模型"]==nm]
        cal.append({"model":nm,**{"stored_brier_"+k:x for k,x in desc(v.brier).items()},**{"stored_cal_slope_"+k:x for k,x in desc(v.cal_slope).items()},"independent_prediction_recomputation":False})
    save(pd.DataFrame(cal),"fixed_calibration_stored_descriptive_only.csv")
    summary["calibration_availability"]={"single_development_independently_recomputed":True,"repeated_fixed_only_stored_aggregate":True,"repeated_retuned_unavailable":True}
    summary["provenance"]["input_hashes_unchanged_after_audit"]=all(sha(PROJECT/k)==v for k,v in hashes_before.items())
    assert summary["provenance"]["input_hashes_unchanged_after_audit"]
    summary["provenance"]["source_sha256"]=hashes_before
    summary["provenance"]["audit_script_sha256"]=sha(__file__)
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    # Exhaustive leaf-value ledger, with the source layer and metric definition.
    ledger=[]
    def source_for(key):
        if key.startswith("cohort"):return "raw source + analysis_meta.json; independently count exclusions and cutoff"
        if key.startswith("features"):return FILES["features_fixed"]+" + "+FILES["features_retuned"]+" + candidate list from analysis_meta.json"
        if key.startswith("hyperparameters"):return FILES["hp"]+" + "+FILES["main_hp"]+"; full canonical parameter combination"
        if key.startswith("single_development"):return FILES["predictions"]+"; score for AUC, p for Brier and calibration"
        return FILES["fixed"]+" + "+FILES["retuned"]+"; paired split index, 200 overlapping resamples"
    def flatten(x,key=""):
        if isinstance(x,dict):
            for k,v in x.items():flatten(v,key+"."+k if key else k)
        elif isinstance(x,list):
            for i,v in enumerate(x):flatten(v,key+f"[{i}]")
        elif isinstance(x,(int,float)) and not isinstance(x,bool):
            ledger.append({"metric_key":key,"value":x,"source":source_for(key),"analysis_status":"Historical non-strict nested; descriptive repeated-split metrics","interval_definition":"p025/p975 and p25/p75 are empirical percentiles; conditional_ci is single-development DeLong only"})
    for key,value in summary.items():
        if key not in ("provenance","runtime_audit_not_historical"):flatten(value,key)
    pd.DataFrame(ledger).to_csv(AUDIT/"historical_number_source_map.csv",index=False,encoding="utf-8-sig",float_format="%.15g")
    pd.DataFrame([{"file":k,"sha256_before":v,"sha256_after":sha(PROJECT/k),"unchanged":sha(PROJECT/k)==v} for k,v in hashes_before.items()]).to_csv(AUDIT/"historical_source_hashes.csv",index=False,encoding="utf-8-sig")
    make_figures(pivots,pd.DataFrame(model_rows),freq,pd.DataFrame(hp_rows),jac)
    result_hashes={str(p.relative_to(ROOT)):sha(p) for p in OUT.glob("*") if p.is_file()}
    (AUDIT/"historical_output_hashes.json").write_text(json.dumps(result_hashes,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"completed":True,"cohort":cohort,"fixed_rho":summary["fixed"]["ranking_rho"],"retuned_rho":summary["retuned"]["ranking_rho"],"features":summary["features"],"output":str(OUT)},ensure_ascii=False,indent=2))


def make_figures(pivots, model_table, freq, hp, jac):
    # Statistical contract: all 200 splits, all 8 models; empirical IQR/whiskers,
    # no P values. Source data are the audited CSV exports, not simulated data.
    font=Path("/mnt/c/Windows/Fonts/simhei.ttf")
    if not font.exists():font=Path("C:/Windows/Fonts/simhei.ttf")
    assert font.exists(),"Chinese font must be available; do not export missing glyphs."
    font_manager.fontManager.addfont(str(font))
    family=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":[family,"DejaVu Sans"],"font.size":7,"axes.labelsize":7,"xtick.labelsize":6.5,"ytick.labelsize":6.5,"axes.spines.right":False,"axes.spines.top":False,"axes.linewidth":.7,"legend.frameon":False,"pdf.fonttype":42,"svg.fonttype":"none","axes.unicode_minus":False})
    def export(fig,name):
        fig.savefig(FIG/(name+".svg"))
        fig.savefig(FIG/(name+".pdf"))
        fig.savefig(FIG/(name+".png"),dpi=300)
        fig.savefig(FIG/(name+".tiff"),dpi=600,pil_kwargs={"compression":"tiff_lzw"})
        plt.close(fig)
    order=["随机森林","梯度提升","Logistic回归","XGBoost","SVM","LightGBM","K近邻","多层感知机"]
    fig,axs=plt.subplots(2,2,figsize=(183/25.4,142/25.4),layout="constrained",gridspec_kw={"width_ratios":[1,1.25]})
    for row,(scheme,colour,label) in enumerate([("fixed","#507D8D","历史：固定参数"),("retuned","#A46B45","历史：逐次调参")]):
        p=pivots[scheme]
        r=p.rank(axis=1,ascending=False,method="min")
        dist=p.max(axis=1).to_numpy()[:,None]-p.to_numpy()
        dd=pd.DataFrame(dist,index=p.index,columns=p.columns)
        for col,(v,xlabel) in enumerate([(r,"测试 AUC 名次（1 为最高）"),(dd,"距同次测试最高 AUC 的差值")]):
            ax=axs[row,col]
            box=ax.boxplot([v[nm] for nm in order],orientation="horizontal",tick_labels=[LABELS[nm] for nm in order],showfliers=False,patch_artist=True,widths=.55,whis=(2.5,97.5),medianprops={"color":"#222222","linewidth":1})
            for patch in box["boxes"]:patch.set_facecolor(colour);patch.set_alpha(.5)
            ax.invert_yaxis();ax.set_xlabel(xlabel);ax.set_title(chr(97+2*row+col)+"  "+label,loc="left",fontweight="bold",fontsize=8)
            ax.grid(axis="x",color="#E6E6E6",linewidth=.5);ax.set_axisbelow(True)
            if col==0:ax.set_xticks(range(1,9));ax.set_xlim(.5,8.5)
            else:
                ax.axvline(.01,color="#555555",ls="--",lw=.6);ax.axvline(.02,color="#555555",ls=":",lw=.6)
                ax.set_xlim(-.003,.15)
    export(fig,"historical_01_rank_and_test_max_distance")
    fig=plt.figure(figsize=(183/25.4,189/25.4),layout="constrained")
    grid=fig.add_gridspec(2,2,width_ratios=[1.65,1],height_ratios=[1,1])
    ax=fig.add_subplot(grid[:,0]);top=fig.add_subplot(grid[0,1]);bottom=fig.add_subplot(grid[1,1])
    names=freq.feature_display.str.replace("左心室收缩末期容积（LVESV）","LVESV",regex=False).str.replace("TC-HDLDL","残余胆固醇（计算值）",regex=False)
    ax.hlines(np.arange(len(freq)),0,freq.frequency,color="#B1C2CA",lw=1)
    ax.scatter(freq.frequency,np.arange(len(freq)),s=8,color="#507D8D",zorder=3)
    ax.set_yticks(np.arange(len(freq)),names,fontsize=6);ax.invert_yaxis();ax.set_xlim(0,1.04);ax.axvline(.8,color="#A46B45",ls="--",lw=.6)
    ax.set_xlabel("200 次历史开发中的选择频率");ax.set_title("a  LASSO 变量选择（全部 49 项）",loc="left",fontweight="bold",fontsize=8)
    h=hp.set_index("model").loc[order]
    yy=np.arange(8)
    top.barh(yy-.16,h.main_recurrence_fraction,height=.3,color="#A46B45",label="主划分组合复现")
    top.barh(yy+.16,h.modal_fraction,height=.3,color="#507D8D",label="众数组合频率")
    top.set_yticks(yy,[LABELS[nm] for nm in order]);top.invert_yaxis();top.set_xlim(0,1)
    top.set_xlabel("完整超参数组合频率");top.set_title("b  历史逐次调参",loc="left",fontweight="bold",fontsize=8);top.legend(loc="lower right",fontsize=6)
    bottom.hist(jac.jaccard,bins=np.linspace(0,1,26),density=True,color="#507D8D",edgecolor="white",linewidth=.4)
    bottom.axvline(jac.jaccard.median(),color="#A46B45",lw=1)
    bottom.set_xlabel("两次划分特征集的 Jaccard 指数");bottom.set_ylabel("描述性密度")
    bottom.set_title("c  特征集相似度",loc="left",fontweight="bold",fontsize=8)
    bottom.text(.03,.96,"19,900 个相关划分对\n非独立样本",transform=bottom.transAxes,va="top",fontsize=6.5)
    export(fig,"historical_02_feature_and_parameter_stability")


if __name__=="__main__":
    main()
