#!/usr/bin/env python3
"""Independent checkpoint verification and descriptive strict-pipeline summary.

Reads immutable completed checkpoints only; never imports the training program.
AUC uses Mann-Whitney average ranks, calibration uses Newton/IRLS, and source-row
memberships plus preprocessing parameters are reconstructed at every CV level.
Overlapping clinical resamples are not independent inferential observations.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

for _thread_variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable]="1"

import numpy as np
import pandas as pd
from scipy import special, stats
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT.parent/"clinical_pipeline"/"data"/"raw"/"source_clean.csv"
MODEL_NAMES=["Logistic回归","随机森林","K近邻","梯度提升","SVM","XGBoost","LightGBM","多层感知机"]
FIXED="fixed_historical_conditional"
MARGINS={"auc":1e-12,"brier":1e-12,"calibration":2e-5,"preprocessing":2e-10}


def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
def objsha(x):return hashlib.sha256(canonical(x).encode("utf-8")).hexdigest()
def member_hash(x):return objsha(sorted(int(v) for v in x))
def read_json(p):return json.loads(Path(p).read_text(encoding="utf-8"))


def write_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    os.replace(temp,p)


def csv(p,rows):
    frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    temp=Path(p).with_name(Path(p).name+f".tmp.{os.getpid()}")
    frame.to_csv(temp,index=False,encoding="utf-8-sig",float_format="%.17g")
    os.replace(temp,p)


def moments(x):
    x=np.asarray(x,dtype=float)
    if not len(x):return {k:None for k in ["mean","sd","min","q025","q25","median","q75","q975","max"]}|{"n":0}
    if not np.isfinite(x).all():raise ValueError("Nonfinite values must not be silently omitted")
    return {"n":len(x),"mean":float(x.mean()),"sd":float(x.std(ddof=1)) if len(x)>1 else None,
            "min":float(x.min()),"q025":float(np.quantile(x,.025)),"q25":float(np.quantile(x,.25)),
            "median":float(np.median(x)),"q75":float(np.quantile(x,.75)),"q975":float(np.quantile(x,.975)),"max":float(x.max())}


def mann_whitney_auc(y,score):
    y=np.asarray(y,dtype=int);score=np.asarray(score,dtype=float)
    n1=int(y.sum());n0=len(y)-n1
    if n1==0 or n0==0 or not np.isfinite(score).all():raise ValueError("AUC undefined")
    ranks=stats.rankdata(score,method="average")
    return float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0))


def irls_calibration(y,p):
    y=np.asarray(y,dtype=float);z=special.logit(np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6))
    x=np.column_stack([np.ones(len(z)),z]);beta=np.array([0.,1.])
    def objective(b):
        eta=x@b
        return float(np.sum(np.logaddexp(0,eta)-y*eta))
    success=False
    for step in range(100):
        q=special.expit(x@beta);grad=x.T@(q-y);hess=x.T@(x*(q*(1-q))[:,None])
        if np.max(np.abs(grad))<1e-8:success=True;break
        delta=np.linalg.solve(hess,grad)
        rate=1.;old=objective(beta)
        while rate>2**-25 and objective(beta-rate*delta)>old+1e-12:rate/=2
        beta-=rate*delta
    grad_norm=float(np.max(np.abs(x.T@(special.expit(x@beta)-y))))
    success=success or grad_norm<1e-7
    if not success:raise ValueError(f"Independent calibration failed: gradient={grad_norm}")
    intercept=0.
    for _ in range(100):
        q=special.expit(intercept+z);gradient=float((q-y).sum())
        if abs(gradient)<1e-9:break
        intercept-=gradient/float(np.sum(q*(1-q)))
    return {"cal_intercept_joint":float(beta[0]),"cal_slope":float(beta[1]),"cal_intercept_only":float(intercept),"cal_gradient_max":grad_norm}


class Verifier:
    def __init__(self,config,raw,report):
        self.c=config;self.report=report;self.checks=Counter();self.maxdiff=Counter()
        features=config["features"]
        needed=features+["Gensini评分","白细胞计数","中性粒细胞计数"]
        f=pd.read_csv(raw,usecols=lambda col:col in set(needed)).apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
        self.check(set(f.columns)==set(needed),"source_required_columns")
        gs=f["Gensini评分"]
        self.check(gs.notna().all() and gs.ge(0).all(),"outcome_valid")
        denominator=f["白细胞计数"]-f["中性粒细胞计数"]
        f["dNLR"]=(f["中性粒细胞计数"]/denominator).where(denominator.gt(0)&f["中性粒细胞计数"].ge(0))
        f["TC-HDLDL"]=f["总胆固醇"]-f["高密度脂蛋白"]-f["低密度脂蛋白"]
        for feature,(lo,hi) in config["range_rules"].items():f.loc[~f[feature].between(lo,hi),feature]=np.nan
        keep=gs.ge(2) if config["exclude_gs_lt2"] else gs.ne(0)
        self.rows=np.flatnonzero(keep.to_numpy());self.y=gs.loc[keep].gt(config["outcome_cutoff_locked"]).astype(int).to_numpy()
        self.X=f.loc[keep,features].copy();self.X.index=self.rows
        self.lookup=pd.Series(self.y,index=self.rows)
        self.known_preprocessing={}

    def check(self,condition,key,context=""):
        self.checks[key]+=1
        if not bool(condition):raise AssertionError(f"{key}: {context}")

    def close(self,a,b,key,tol,context=""):
        a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
        self.check(a.shape==b.shape,key+"_shape",context)
        self.check(np.isfinite(a).all() and np.isfinite(b).all(),key+"_finite",context)
        d=float(np.max(np.abs(a-b))) if a.size else 0.
        self.maxdiff[key]=max(self.maxdiff[key],d)
        self.check(np.allclose(a,b,atol=tol,rtol=tol),key,context+f"; max difference {d}")

    def preprocessing(self,meta,rows,context):
        rows=np.asarray(rows,dtype=int)
        self.check(meta["fit_rows_sha256"]==member_hash(rows),"preprocessing_fit_hash",context)
        self.check(meta["n_fit"]==len(rows),"preprocessing_fit_count",context)
        identity=member_hash(rows)
        if identity in self.known_preprocessing:
            self.check(canonical(meta)==self.known_preprocessing[identity],"same_members_same_preprocessing",context)
            return
        source=self.X.loc[rows];values=source.to_numpy(dtype=float);missing=np.isnan(values).mean(axis=0)
        names=source.columns.to_numpy();kept=names[missing<=self.c["missing_fraction_limit"]].tolist();dropped=names[missing>self.c["missing_fraction_limit"]].tolist()
        self.check(kept==meta["features_kept"] and dropped==meta["features_dropped"],"missingness_exclusion",context)
        self.close(missing,[meta["missing_rate_train"][f] for f in names],"missing_rates",1e-12,context)
        a=source[kept].to_numpy(dtype=float).copy();logs=[]
        for j,feature in enumerate(kept):
            observed=a[~np.isnan(a[:,j]),j]
            if feature in self.c["categorical"]:
                u,n=np.unique(observed,return_counts=True);impute=float(u[np.argmax(n)])
            else:impute=float(np.median(observed))
            self.close(impute,meta["impute_values"][feature],"imputation",1e-12,context+"/"+feature)
            if feature not in self.c["categorical"] and observed.min()>=0 and np.unique(observed).size>1 and abs(stats.skew(observed))>2:logs.append(feature)
            a[np.isnan(a[:,j]),j]=impute
            if feature in logs:a[:,j]=np.log1p(a[:,j])
        self.check(logs==meta["log1p_features"],"training_skew_log_rule",context)
        scale=a.std(axis=0,ddof=0);scale[scale==0]=1.
        self.close(a.mean(axis=0),meta["scaler_mean"],"scaler_mean",MARGINS["preprocessing"],context)
        self.close(scale,meta["scaler_scale"],"scaler_scale",MARGINS["preprocessing"],context)
        self.known_preprocessing[identity]=canonical(meta)

    def representation(self,rep,rows,y,context):
        self.preprocessing(rep["preprocessing"],rows,context+"/final")
        selected=rep["selected"]
        self.check(bool(selected) and len(selected)==len(set(selected)) and set(selected)<=set(rep["preprocessing"]["features_kept"]),"selected_features_eligible",context)
        if self.c["all_features"]:
            self.check(rep["method"]=="all eligible features" and selected==rep["preprocessing"]["features_kept"],"all_features_rule",context)
            return
        self.check(rep["method"]=="nested LASSO 1SE","lasso_method",context)
        folds=list(StratifiedKFold(self.c["lasso_folds"],shuffle=True,random_state=self.c["cv_seed"]).split(np.zeros(len(rows)),y))
        self.check(len(rep["lasso_cv_folds"])==len(folds),"lasso_fold_count",context)
        for i,((tr,va),fold) in enumerate(zip(folds,rep["lasso_cv_folds"])):
            self.check(fold["train_members_sha256"]==member_hash(rows[tr]),"lasso_train_hash",context)
            self.check(fold["valid_members_sha256"]==member_hash(rows[va]),"lasso_valid_hash",context)
            self.check(not set(rows[tr])&set(rows[va]),"lasso_train_valid_disjoint",context)
            self.preprocessing(fold["preprocessing"],rows[tr],context+f"/lasso_{i}")
        path=rep["path"];foldauc=np.asarray(path["fold_auc"])
        self.check(foldauc.shape==(self.c["lasso_folds"],len(self.c["lasso_cs"])),"lasso_full_grid_shape",context)
        self.check(np.all((foldauc>=0)&(foldauc<=1)),"lasso_fold_auc_range",context)
        self.close(path["C"],self.c["lasso_cs"],"lasso_grid_values",1e-12,context)
        means=foldauc.mean(axis=0);sds=foldauc.std(axis=0,ddof=1);best=int(means.argmax())
        cutoff=means[best]-sds[best]/np.sqrt(len(folds));first=int(np.flatnonzero(means>=cutoff)[0]);Cs=np.array(self.c["lasso_cs"])
        chosen=int(np.argmin(abs(Cs-rep["C"])))
        self.close(means,path["mean_auc"],"lasso_means",1e-12,context)
        self.close(sds,path["sd_auc"],"lasso_sd",1e-12,context)
        self.close([cutoff,means[best],means[chosen],Cs[first],Cs[chosen]],
                   [rep["one_se_cutoff"],rep["best_cv_auc"],rep["chosen_cv_auc"],rep["C_initial_1se"],rep["C"]],"lasso_selection_rule",1e-12,context)
        self.check(chosen>=first and chosen-first==rep["empty_model_fallback_steps"],"lasso_fallback_record",context)
        self.check(set(rep["coef"])==set(selected) and all(v!=0 for v in rep["coef"].values()),"lasso_nonzero_coefficients",context)

    def development(self,audit,rows,y,context):
        self.representation(audit["final_representation"],rows,y,context+"/final_representation")
        folds=list(StratifiedKFold(self.c["model_folds"],shuffle=True,random_state=self.c["cv_seed"]).split(np.zeros(len(rows)),y))
        self.check(len(audit["model_cv_folds"])==len(folds),"model_fold_count",context)
        for i,((tr,va),fold) in enumerate(zip(folds,audit["model_cv_folds"])):
            self.check(fold["fold"]==i,"model_fold_id",context)
            self.check(fold["train_members_sha256"]==member_hash(rows[tr]),"model_train_hash",context)
            self.check(fold["valid_members_sha256"]==member_hash(rows[va]),"model_valid_hash",context)
            self.check(not set(rows[tr])&set(rows[va]),"model_train_valid_disjoint",context)
            self.representation(fold["representation"],rows[tr],y[tr],context+f"/model_{i}")
        self.check(set(audit["candidates"])==set(self.c["models"]),"candidate_model_set",context)
        best={}
        for model,records in audit["candidates"].items():
            self.check([canonical(r["params"]) for r in records]==[canonical(r) for r in self.c["grids"][model]],"complete_candidate_grid",context+"/"+model)
            for r in records:
                self.check(len(r["fold_auc"])==self.c["model_folds"],"candidate_fold_count",context)
                self.close(np.mean(r["fold_auc"]),r["mean_auc"],"candidate_mean",1e-12,context)
                self.close(np.std(r["fold_auc"],ddof=1),r["sd_auc"],"candidate_sd",1e-12,context)
            best[model]=max(records,key=lambda r:r["mean_auc"])
        return best


def independent_metrics(v):
    y=v.y.to_numpy();p=v.p.to_numpy()
    if not np.isfinite(p).all() or not np.all((p>=0)&(p<=1)):raise ValueError("Invalid probabilities")
    return {"auc":mann_whitney_auc(y,v.score),"brier":float(np.mean((p-y)**2)),**irls_calibration(y,p)}


def main(args):
    run=Path(args.run).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    report={"started_utc":datetime.now(timezone.utc).isoformat(),"run_directory":str(run),"verification_state":"running","errors":[],"warnings":[],"tolerances":MARGINS,"independent_methods":{"auc":"Mann-Whitney U from average ranks","brier":"mean squared probability error","calibration":"Newton/IRLS with line search, independently from training BFGS","preprocessing":"source-row numerical reconstruction at every recorded training fold"}}
    try:
        rm=read_json(run/"run_manifest.json");config=rm["config"]
        v=Verifier(config,Path(args.raw),report)
        v.check(objsha(config)==rm["run_hash"],"run_config_hash")
        v.check(sha(args.raw)==config["raw_sha256"],"raw_hash")
        v.check(member_hash(v.rows)==rm["cohort"]["cohort_members_sha256"],"cohort_membership_hash")
        v.check(len(v.rows)==rm["cohort"]["n_analyzed"] and int(v.y.sum())==rm["cohort"]["events"],"cohort_counts")
        v.check(rm["cohort"]["clinical_subtype_read_or_used"] is False and config["stratification"].startswith("outcome only"),"outcome_only_config")
        v.check(config["models"]==MODEL_NAMES,"eight_models")
        if config["mode"]=="full":
            v.check(config["model_folds"]==5 and config["lasso_folds"]==5,"full_cv_hierarchy")
            v.check([len(config["grids"][k]) for k in MODEL_NAMES]==[4,8,6,16,6,32,32,9],"full_published_model_grids")
            v.close(config["lasso_cs"],np.logspace(-3,3,100),"full_lasso_grid",1e-12)
        code_match=sha(ROOT/"analysis"/"nested_pipeline.py")==config["analysis_code_sha256"]
        if not code_match and args.allow_recorded_code:report["warnings"].append("Recorded training code hash differs from current file; explicitly permitted for benchmark inspection only. Do not use this check as current-code provenance validation.")
        else:v.check(code_match,"current_training_code_hash")
        historical_path=Path(config["historical_parameter_source"]) if config["historical_parameter_source"] else None
        if historical_path:
            v.check(sha(historical_path)==config["historical_params_sha256"],"historical_parameter_source_hash")
            historical={r["模型"]:json.loads(r["最佳参数"]) for r in pd.read_csv(historical_path).to_dict("records")}
        else:historical={}
        schemes=["retuned",FIXED] if historical else ["retuned"]
        expected=args.expected if args.expected is not None else (1 if config["primary_seed42"] else 200)
        folders=sorted(p for p in (run/"checkpoints").iterdir() if p.is_dir() and not p.name.startswith(".") and (p/"manifest.json").is_file())
        report.update({"completed_folders_at_snapshot":len(folders),"expected_splits":expected,"training_code_hash_matches_current":code_match,"run_config_sha256":rm["run_hash"],"run_manifest_sha256":sha(run/"run_manifest.json"),"audit_script_sha256":sha(__file__),"verification_file":str(args.verification),"cohort_n":len(v.rows),"raw_sha256":config["raw_sha256"]})
        per_split=[];features=[];hyper=[];verified=[];oof_rows=[]
        for folder in folders:
            cm=read_json(folder/"manifest.json");split=int(cm["split"]);context=folder.name
            v.check(cm["state"]=="complete" and cm["run_hash"]==rm["run_hash"],"checkpoint_complete_identity",context)
            required={"membership.csv","predictions.csv","metrics.json","audit.json"}
            v.check(required<=set(cm["output_sha256"]),"required_checkpoint_files",context)
            for name,digest in cm["output_sha256"].items():
                v.check(Path(name).name==name,"safe_manifest_file_name",context)
                v.check(sha(folder/name)==digest,"checkpoint_file_hash",context+"/"+name)
            seed=42 if config["primary_seed42"] else 1000+split
            tr,te=train_test_split(np.arange(len(v.rows)),train_size=.75,random_state=seed,stratify=v.y)
            training,testing=v.rows[tr],v.rows[te]
            audit=read_json(folder/"audit.json");member=audit["membership"]
            v.check(canonical(cm["membership"])==canonical(member),"membership_record_agrees",context)
            v.check(member["seed"]==seed and member["stratification"]=="outcome only","split_seed_and_outcome_stratification",context)
            v.check(member["train_members_sha256"]==member_hash(training) and member["test_members_sha256"]==member_hash(testing),"reconstructed_split_hashes",context)
            v.check(member["n_train"]==len(training) and member["n_test"]==len(testing) and member["events_train"]==int(v.y[tr].sum()) and member["events_test"]==int(v.y[te].sum()),"split_counts",context)
            membership=pd.read_csv(folder/"membership.csv").set_index("source_row").sort_index()
            v.check(np.array_equal(membership.index,v.rows),"membership_all_source_rows",context)
            v.check(np.array_equal(membership.y,v.y),"membership_all_outcomes",context)
            v.check(set(membership[membership.partition=="train"].index)==set(training) and set(membership[membership.partition=="test"].index)==set(testing),"membership_partitions",context)
            best=v.development(audit["development"],training,v.y[tr],context)
            rep=audit["development"]["final_representation"]
            features.append({"split":split,"features":rep["selected"],"n_selected":len(rep["selected"]),"lasso_C":rep.get("C")})
            saved=read_json(folder/"metrics.json")
            v.check(len(saved)==8*len(schemes) and len({(r["scheme"],r["model"]) for r in saved})==len(saved),"metrics_model_scheme_grid",context)
            saved={(r["scheme"],r["model"]):r for r in saved}
            prediction=pd.read_csv(folder/"predictions.csv",float_precision="round_trip")
            v.check(len(prediction)==len(testing)*8*len(schemes),"prediction_row_count",context)
            v.check(not prediction.duplicated(["source_row","scheme","model"]).any(),"prediction_unique_key",context)
            v.check(set(prediction.scheme)==set(schemes) and set(prediction.model)==set(MODEL_NAMES) and set(prediction.split)=={split},"prediction_scheme_model_split",context)
            for scheme,model in itertools.product(schemes,MODEL_NAMES):
                z=prediction[(prediction.scheme==scheme)&(prediction.model==model)].sort_values("source_row")
                v.check(np.array_equal(z.source_row,np.sort(testing)),"prediction_test_membership",context)
                v.check(np.array_equal(z.y,v.lookup.loc[z.source_row].to_numpy()),"prediction_outcomes",context)
                r=saved[(scheme,model)]
                v.check(r["split"]==split and r["selected_features"]==rep["selected"] and r["n_selected"]==len(rep["selected"]),"metric_selected_representation",context)
                params=best[model]["params"] if scheme=="retuned" else historical[model]
                v.check(canonical(params)==canonical(r["params"]),"selected_parameters",context+"/"+model)
                if scheme=="retuned":v.close(r["inner_cv_auc"],best[model]["mean_auc"],"selected_cv_auc",1e-12,context)
                recomputed=independent_metrics(z)
                for metric in ("auc","brier","cal_slope","cal_intercept_joint","cal_intercept_only"):
                    v.check(r[metric] is not None,"saved_metric_not_missing",context+"/"+metric)
                    v.close(recomputed[metric],r[metric],"metric_"+metric,MARGINS[metric] if metric in MARGINS else MARGINS["calibration"],context+"/"+model)
                per_split.append({"scheme":scheme,"model":model,"split":split,"n_test":len(z),"events_test":int(z.y.sum()),"inner_cv_auc":r["inner_cv_auc"],"saved_auc":r["auc"],"auc_quantization_half_tie_step":1/(2*int(z.y.sum())*int(len(z)-z.y.sum())),**recomputed})
                hyper.append({"scheme":scheme,"model":model,"split":split,"parameters":canonical(params)})
            if config["strict_primary_oof_requested"]:
                oof_rows.extend(verify_oof(v,folder,training,v.y[tr],prediction,config))
            verified.append({"split":split,"checkpoint":folder.name,"manifest_sha256":sha(folder/"manifest.json"),"output_sha256":cm["output_sha256"]})
            if len(verified)%10==0:print(f"Verified {len(verified)}/{len(folders)} complete checkpoints",flush=True)
        completed_ids={r["split"] for r in verified}
        planned_ids=set(range(config["start_split"],config["start_split"]+config["n_splits_planned"]))
        v.check(len(completed_ids)==len(verified),"completed_split_unique")
        v.check(completed_ids<=planned_ids,"completed_splits_within_declared_plan")
        report["verified_checkpoints"]=verified;report["checks_passed_counts"]=dict(v.checks);report["maximum_absolute_differences"]=dict(v.maxdiff)
        report["verification_state"]="passed";report["n_verified"]=len(verified);report["complete"]=len(verified)==expected==config["n_splits_planned"] and completed_ids==planned_ids
        report["publication_result_set_ready"]=bool(report["complete"] and config["publication_eligible"] and code_match)
        report["finished_utc"]=datetime.now(timezone.utc).isoformat()
        report["limitations"]=["This independently reconstructs recorded training boundaries and preprocessing but does not refit every candidate estimator.","Clinical repeated splits and all split-pair correlations share patients and must not be treated as independent samples.","Fixed historical parameters remain a conditional descriptive comparator because their source development members may overlap test sets."]
        if len(per_split):summarize(out,per_split,features,hyper,historical,expected,config,report)
        else:write_json(out/"summary.json",{"n_completed":0,"expected":expected,"complete":False,"status":"incomplete_no_completed_checkpoint","rank_correlations":None,"jaccard":None})
        if oof_rows:csv(out/"primary_oof_recomputed.csv",oof_rows)
        if config["primary_seed42"] and report["complete"]:
            primary_conditional_intervals(out,folders[0],oof_rows)
        if args.reference_summary and per_split:
            compare_reference(out,Path(args.reference_summary),report)
        write_json(args.verification,report)
        print(json.dumps({"verification":"passed","n_completed":len(verified),"expected":expected,"complete":report["complete"],"publication_result_set_ready":report["publication_result_set_ready"],"output":str(out)},ensure_ascii=False))
    except Exception as e:
        report["verification_state"]="failed";report["errors"].append(f"{type(e).__name__}: {e}");report["finished_utc"]=datetime.now(timezone.utc).isoformat()
        write_json(args.verification,report)
        raise


def verify_oof(v,folder,training,y,holdout,config):
    audit=read_json(folder/"oof_audit.json");pred=pd.read_csv(folder/"oof_predictions.csv",float_precision="round_trip");threshold=read_json(folder/"thresholds.json")
    folds=list(StratifiedKFold(5,shuffle=True,random_state=143).split(np.zeros(len(training)),y))
    v.check(len(audit)==5,"oof_fold_count",folder.name)
    for i,((tr,va),a) in enumerate(zip(folds,audit)):
        v.check(a["oof_fold"]==i and a["train_members_sha256"]==member_hash(training[tr]) and a["valid_members_sha256"]==member_hash(training[va]),"oof_membership_hash",folder.name)
        v.development(a["development"],training[tr],y[tr],folder.name+f"/oof_{i}")
        for model in MODEL_NAMES:
            z=pred[(pred.model==model)&(pred.oof_fold==i)]
            v.check(set(z.source_row)==set(training[va]) and len(z)==len(va),"oof_valid_predictions",folder.name)
    v.check(len(pred)==len(training)*8 and not pred.duplicated(["source_row","model"]).any(),"oof_predictions_complete",folder.name)
    result=[]
    for model in MODEL_NAMES:
        z=pred[pred.model==model].sort_values("source_row")
        v.check(np.array_equal(z.y,v.lookup.loc[z.source_row]),"oof_outcomes",folder.name)
        met=independent_metrics(z);t=threshold[model]["youden_probability_threshold"]
        # Exhaustive probability cuts, preserving the highest-threshold first-max tie rule.
        cuts=np.unique(z.p)[::-1];true=z.y.to_numpy();prob=z.p.to_numpy()
        J=np.array([(prob[true==1]>=c).mean()-(prob[true==0]>=c).mean() for c in cuts])
        # ROC drop_intermediate cannot remove a unique maximum; allow equivalent
        # maximal J when different cuts have the same empirical objective.
        chosen_J=float((prob[true==1]>=t).mean()-(prob[true==0]>=t).mean())
        v.close(chosen_J,J.max(),"oof_youden_optimum",1e-12,folder.name)
        v.check(t in cuts,"oof_threshold_observed_probability",folder.name)
        v.close(met["auc"],threshold[model]["oof_auc"],"oof_auc",1e-12,folder.name)
        test=holdout[(holdout.model==model)&(holdout.scheme=="retuned")]
        sens=float((test.loc[test.y==1,"p"]>=t).mean());spec=float((test.loc[test.y==0,"p"]<t).mean())
        v.close([sens,spec],[threshold[model]["test_sensitivity"],threshold[model]["test_specificity"]],"holdout_locked_threshold_metrics",1e-12,folder.name)
        result.append({"model":model,"n_oof":len(z),**met,"youden_probability_threshold":t,"test_sensitivity":sens,"test_specificity":spec})
    return result


def primary_conditional_intervals(out,folder,oof_rows):
    """Fixed-fit conditional DeLong intervals, not full-strategy uncertainty."""
    prediction=pd.read_csv(folder/"predictions.csv",float_precision="round_trip")
    prediction=prediction[prediction.scheme=="retuned"]
    scores=prediction.pivot(index="source_row",columns="model",values="score")[MODEL_NAMES].sort_index()
    outcomes=prediction[prediction.model==MODEL_NAMES[0]].set_index("source_row").loc[scores.index,"y"].to_numpy()
    a=scores.to_numpy().T;positive=a[:,outcomes==1];negative=a[:,outcomes==0]
    u=(positive[:,:,None]>negative[:,None,:]).astype(float)+.5*(positive[:,:,None]==negative[:,None,:])
    aucs=u.mean(axis=(1,2));vpos=u.mean(axis=2);vneg=u.mean(axis=1)
    covariance=np.cov(vpos,ddof=1)/positive.shape[1]+np.cov(vneg,ddof=1)/negative.shape[1]
    metrics=pd.read_csv(out/"per_split_metrics.csv",float_precision="round_trip").set_index("model")
    oof={r["model"]:r for r in oof_rows};rows=[];zcrit=float(stats.norm.ppf(.975))
    for i,model in enumerate(MODEL_NAMES):
        r=metrics.loc[model]
        if abs(aucs[i]-r.auc)>1e-12:raise AssertionError("Primary direct-comparison and rank AUC disagree")
        se=float(np.sqrt(covariance[i,i]));difference=float(aucs[i]-aucs[0]);variance=float(covariance[i,i]+covariance[0,0]-2*covariance[i,0])
        if variance < -1e-12:raise AssertionError("Negative DeLong paired variance")
        paired_se=float(np.sqrt(max(variance,0)))
        p=float(2*stats.norm.sf(abs(difference/paired_se))) if paired_se>0 else (1. if difference==0 else None)
        item={"model":model,"n_test":len(outcomes),"events_test":int(outcomes.sum()),"auc":float(aucs[i]),"auc_conditional_ci_low":float(aucs[i]-zcrit*se),"auc_conditional_ci_high":float(aucs[i]+zcrit*se),"delta_auc_vs_lr":difference,"delta_conditional_ci_low":difference-zcrit*paired_se,"delta_conditional_ci_high":difference+zcrit*paired_se,"delong_p_unadjusted":p if i else None,"brier":float(r.brier),"cal_slope":float(r.cal_slope),"cal_intercept_joint":float(r.cal_intercept_joint),"cal_intercept_only":float(r.cal_intercept_only),"inner_cv_auc":float(r.inner_cv_auc),"rank":int(r["rank"])}
        if model in oof:
            item.update({"oof_auc":oof[model]["auc"],"youden_probability_threshold":oof[model]["youden_probability_threshold"],"test_sensitivity":oof[model]["test_sensitivity"],"test_specificity":oof[model]["test_specificity"]})
        rows.append(item)
    tests=[i for i,r in enumerate(rows) if r["delong_p_unadjusted"] is not None];order=sorted(tests,key=lambda i:rows[i]["delong_p_unadjusted"]);running=0.
    rows[0]["p_holm"]=None
    for rank,i in enumerate(order):
        running=max(running,(len(order)-rank)*rows[i]["delong_p_unadjusted"]);rows[i]["p_holm"]=min(1.,running)
    audit=read_json(folder/"audit.json");rep=audit["development"]["final_representation"]
    chosen=max(MODEL_NAMES,key=lambda model:float(metrics.loc[model,"inner_cv_auc"]))
    info={"analysis":"seed42 strict primary holdout with complete development-only OOF","condition":"All intervals condition on this development sample and fitted models; they are not intervals for the full model-development or algorithm-selection strategy.","winner_caution":"All prespecified models are shown. Selecting the largest test AUC and treating its displayed CI as selection-adjusted is invalid.","multiplicity":"Holm adjustment across seven predefined model-versus-LR tests; confidence intervals are pairwise, not simultaneous.","n_selected":len(rep["selected"]),"selected_features":rep["selected"],"lasso_C":rep.get("C"),"algorithm_selected_using_development_cv":chosen,"models":rows,"source":"Verified primary checkpoint predictions.csv; independent positive-negative structural DeLong covariance."}
    csv(out/"primary_holdout_conditional.csv",rows);write_json(out/"primary_conditional_inference.json",info)
    summary=read_json(out/"summary.json");summary["primary_conditional_inference"]=info;write_json(out/"summary.json",summary)
    ledger=[]
    for i,row in enumerate(rows):
        for key,value in row.items():
            if isinstance(value,(int,float)) and not isinstance(value,bool):ledger.append({"metric_key":f"primary_conditional_inference.models[{i}].{key}","value":value,"source":"primary_holdout_conditional.csv; verified primary predictions + independent DeLong structural comparisons","n_completed":1,"expected":1,"status":"complete_single_primary","interpretation":"Conditional on fitted models/this development set; pairwise intervals are not simultaneous or strategy-level CI."})
    previous=pd.read_csv(out/"number_source_map.csv",float_precision="round_trip")
    csv(out/"number_source_map.csv",pd.concat([previous,pd.DataFrame(ledger)],ignore_index=True))


def summarize(out,per_split,features,hyper,historical,expected,config,report):
    m=pd.DataFrame(per_split);n=m.split.nunique();hp=pd.DataFrame(hyper)
    m["rank"]=m.groupby(["scheme","split"]).auc.rank(ascending=False,method="min").astype(int)
    m["gap_to_best"]=m.groupby(["scheme","split"]).auc.transform("max")-m.auc
    lr=m[m.model=="Logistic回归"][["scheme","split","auc"]].rename(columns={"auc":"lr_auc"})
    m=m.merge(lr,on=["scheme","split"],validate="many_to_one");m["delta_lr"]=m.auc-m.lr_auc
    rows=[];rho={};pairs=[];leader=[];tie_sensitivity={}
    for scheme,scheme_data in m.groupby("scheme"):
        p=scheme_data.pivot(index="split",columns="model",values="auc")[MODEL_NAMES].sort_index()
        rank=p.rank(axis=1,ascending=False,method="average")
        stored_p=scheme_data.pivot(index="split",columns="model",values="saved_auc")[MODEL_NAMES].sort_index()
        rounded_p=p.round(12)
        primary_min=p.rank(axis=1,ascending=False,method="min")
        stored_min=stored_p.rank(axis=1,ascending=False,method="min")
        rounded_min=rounded_p.rank(axis=1,ascending=False,method="min")
        tie_sensitivity[scheme]={"main_definition":"Mann-Whitney U normalized by n1*n0; same U gives identical AUC. Descending min ranks for rank/win, average ranks for Spearman.","rounding_decimals_sensitivity":12,"stored_vs_mannwhitney_changed_rank_cells":int((stored_min!=primary_min).to_numpy().sum()),"rounding_12_changed_rank_cells":int((rounded_min!=primary_min).to_numpy().sum()),"stored_vs_mannwhitney_max_abs_auc_difference":float(np.max(abs(stored_p.to_numpy()-p.to_numpy()))),"primary_tied_winner_splits":int((primary_min.eq(1).sum(axis=1)>1).sum()),"stored_tied_winner_splits":int((stored_min.eq(1).sum(axis=1)>1).sum()),"rounding_12_tied_winner_splits":int((rounded_min.eq(1).sum(axis=1)>1).sum()),"primary_win_fraction":{model:float(primary_min[model].eq(1).mean()) for model in MODEL_NAMES},"stored_win_fraction":{model:float(stored_min[model].eq(1).mean()) for model in MODEL_NAMES},"rounding_12_win_fraction":{model:float(rounded_min[model].eq(1).mean()) for model in MODEL_NAMES}}
        if n>=2:
            i,j=np.triu_indices(n,1);cor=np.corrcoef(rank.to_numpy())[i,j]
            rho[scheme]={**moments(cor),"negative_fraction":float((cor<0).mean()),"independence":"dependent split pairs; descriptive only"}
            pairs.extend({"scheme":scheme,"split_i":int(p.index[a]),"split_j":int(p.index[b]),"rho":float(c)} for a,b,c in zip(i,j,cor))
            for label,values in [("stored",stored_p),("rounding_12",rounded_p)]:
                corr=np.corrcoef(values.rank(axis=1,ascending=False,method="average").to_numpy())[i,j]
                tie_sensitivity[scheme][label+"_rho_mean"]=float(corr.mean())
                tie_sensitivity[scheme][label+"_rho_negative_fraction"]=float((corr<0).mean())
        else:rho[scheme]=None
        arr=np.sort(p.to_numpy(),axis=1)
        leader.extend({"scheme":scheme,"split":int(s),"top_two_gap":float(a[-1]-a[-2]),"eight_model_range":float(a[-1]-a[0])} for s,a in zip(p.index,arr))
        for model,z in scheme_data.groupby("model"):
            auc=moments(z.auc);delta=moments(z.delta_lr);rank_summary=moments(z["rank"])
            rows.append({"scheme":scheme,"model":model,"n_splits":len(z),"mean_auc":auc["mean"],"sd_auc":auc["sd"],"auc_q025":auc["q025"],"auc_q975":auc["q975"],"mean_delta_lr":delta["mean"],"delta_q025":delta["q025"],"delta_q975":delta["q975"],"median_rank":rank_summary["median"],"rank_q25":rank_summary["q25"],"rank_q75":rank_summary["q75"],"win_fraction":float((z["rank"]==1).mean()),"within_001":float((z.gap_to_best<=.01).mean()),"within_002":float((z.gap_to_best<=.02).mean()),"mean_gap_to_best":float(z.gap_to_best.mean()),"mean_brier":float(z.brier.mean()),"sd_brier":float(z.brier.std(ddof=1)) if len(z)>1 else None,"brier_q025":float(z.brier.quantile(.025)),"brier_q975":float(z.brier.quantile(.975)),"median_cal_slope":float(z.cal_slope.median()),"mean_cal_slope":float(z.cal_slope.mean()),"cal_slope_q025":float(z.cal_slope.quantile(.025)),"cal_slope_q975":float(z.cal_slope.quantile(.975)),"mean_cal_intercept":float(z.cal_intercept_joint.mean()),"mean_cal_intercept_only":float(z.cal_intercept_only.mean()),"cal_intercept_only_q025":float(z.cal_intercept_only.quantile(.025)),"cal_intercept_only_q975":float(z.cal_intercept_only.quantile(.975))})
    sets=[set(r["features"]) for r in sorted(features,key=lambda r:r["split"])]
    count=Counter(itertools.chain.from_iterable(sets))
    display={"心室收缩容量":"左心室收缩末期容积（LVESV）","TC-HDLDL":"残余胆固醇（计算值）"}
    freq=[{"feature":f,"display_name":display.get(f,f),"count":count[f],"denominator":n,"frequency":count[f]/n} for f in config["features"]]
    jac=[{"split_i":features[i]["split"],"split_j":features[j]["split"],"jaccard":len(a&b)/len(a|b)} for (i,a),(j,b) in itertools.combinations(enumerate(sets),2)]
    hsum=[]
    for model,z in hp[hp.scheme=="retuned"].groupby("model"):
        counts=z.parameters.value_counts();hist=canonical(historical[model]) if model in historical else None
        hsum.append({"model":model,"unique_combinations":len(counts),"modal_fraction":float(counts.iloc[0]/n),"historical_fraction":float(counts.get(hist,0)/n) if hist is not None else None,"modal_parameters":counts.index[0],"historical_parameters":hist})
    paired=[]
    if FIXED in set(m.scheme):
        for model in MODEL_NAMES:
            r=m[(m.scheme=="retuned")&(m.model==model)].set_index("split").sort_index()
            f=m[(m.scheme==FIXED)&(m.model==model)].set_index("split").sort_index()
            h=hp[(hp.scheme=="retuned")&(hp.model==model)].set_index("split").loc[r.index]
            changed=h.parameters.ne(canonical(historical[model]))
            for group,idx in [("all",np.ones(len(r),dtype=bool)),("changed",changed.to_numpy()),("unchanged",~changed.to_numpy())]:
                delta=(r.auc-f.auc).to_numpy()[idx]
                if not len(delta):continue
                paired.append({"model":model,"parameter_group":group,**moments(delta),"positive_fraction":float((delta>0).mean()),"negative_fraction":float((delta<0).mean()),"absolute_delta_le_001":float((abs(delta)<=.01).mean()),"absolute_delta_le_002":float((abs(delta)<=.02).mean()),"mean_delta_brier":float((r.brier-f.brier).to_numpy()[idx].mean())})
    strategy=[]
    for split,z in m[m.scheme=="retuned"].groupby("split"):
        z=z.set_index("model").loc[MODEL_NAMES]
        chosen=MODEL_NAMES[int(np.argmax(z.inner_cv_auc.to_numpy()))];r=z.loc[chosen]
        strategy.append({"split":int(split),"chosen_model":chosen,"selected_inner_cv_auc":float(r.inner_cv_auc),"test_auc":float(r.auc),"test_brier":float(r.brier),"delta_auc_vs_lr":float(r.auc-z.loc["Logistic回归","auc"]),"gap_to_same_test_max_descriptive":float(z.auc.max()-r.auc)})
    counts=Counter(r["chosen_model"] for r in strategy)
    strategy_summary={"selection_rule":"Highest training-only inner CV AUC among eight retuned pipelines; first in fixed MODEL_NAMES order on exact tie.","evaluation_rule":"Same split test set was not used for algorithm selection.","n_splits":n,"auc":moments([r["test_auc"] for r in strategy]),"brier":moments([r["test_brier"] for r in strategy]),"delta_auc_vs_lr":moments([r["delta_auc_vs_lr"] for r in strategy]),"selection_frequency":{model:counts[model]/n for model in MODEL_NAMES},"positive_delta_vs_lr_fraction":float(np.mean([r["delta_auc_vs_lr"]>0 for r in strategy])),"inference":"Empirical distribution across overlapping clinical splits, not independent confidence intervals."}
    summary={"analysis":"outcome_only_strict_fold_local_preprocessing_and_selection","n_completed":n,"expected":expected,"n_planned_in_config":config["n_splits_planned"],"complete":n==expected,"status":"complete" if n==expected else "incomplete","all_features":config["all_features"],"primary_seed42":config["primary_seed42"],"publication_result_set_ready":report["publication_result_set_ready"],"rank_correlations":rho,"jaccard":moments([r["jaccard"] for r in jac]) if jac else None,"n_selected":moments([len(s) for s in sets]),"features_at_least_080":sum(r["frequency"]>=.8 for r in freq),"ever_selected":sum(r["count"]>0 for r in freq),"cohort_n":report["cohort_n"],"warnings":["All clinical resamples share patients; empirical intervals are not confidence intervals.","0.01 and 0.02 are exploratory numerical scales, not clinical equivalence margins.","Same-test maximum is a selected random benchmark, not true algorithm-selection loss.","Historical fixed-parameter comparator is conditional and is not an unbiased estimate of a complete development strategy."],"verification_report":report["verification_file"],"model_summary":rows,"paired_scheme":paired,"hyperparameter_summary":hsum,"algorithm_selection_strategy":strategy_summary,"leading_gaps":{s:{"top_two_gap":moments([r["top_two_gap"] for r in leader if r["scheme"]==s]),"eight_model_range":moments([r["eight_model_range"] for r in leader if r["scheme"]==s])} for s in sorted(set(m.scheme))}}
    summary["complete"]=report["complete"];summary["status"]="complete" if report["complete"] else "incomplete"
    summary["float_tie_sensitivity"]=tie_sensitivity
    summary["ranking_definition"]="Primary AUC is independently recomputed as Mann-Whitney U/(n1*n0), eliminating integration-roundoff ordering of equal U. Descending min rank for rank/win; average tied ranks for Spearman. 12-decimal rounding is separately reported sensitivity only."
    csv(out/"per_split_metrics.csv",m);csv(out/"model_summary.csv",rows);csv(out/"feature_frequency.csv",freq);csv(out/"hyperparameter_summary.csv",hsum);csv(out/"paired_scheme.csv",paired if paired else pd.DataFrame(columns=["model","parameter_group","n","mean","q025","q975"]))
    csv(out/"feature_sets.csv",[{**r,"features":"|".join(r["features"])} for r in features]);csv(out/"hyperparameters.csv",hp);csv(out/"rank_correlations.csv",pd.DataFrame(pairs,columns=["scheme","split_i","split_j","rho"]));csv(out/"jaccard.csv",pd.DataFrame(jac,columns=["split_i","split_j","jaccard"]));csv(out/"leading_gaps.csv",leader)
    csv(out/"algorithm_selection_per_split.csv",strategy);write_json(out/"algorithm_selection_summary.json",strategy_summary)
    write_json(out/"summary.json",summary)
    ledger=[]
    def flatten(value,key=""):
        if isinstance(value,dict):
            for k,val in value.items():flatten(val,key+"."+k if key else k)
        elif isinstance(value,list):
            for i,val in enumerate(value):flatten(val,key+f"[{i}]")
        elif isinstance(value,(int,float)) and not isinstance(value,bool):
            source="per_split_metrics.csv; independently recomputed from checkpoint predictions.csv"
            if key.startswith("jaccard") or key.startswith("n_selected") or key in ("features_at_least_080","ever_selected"):source="feature_sets.csv + feature_frequency.csv + jaccard.csv; selected fields independently checked against checkpoint audits"
            elif key.startswith("rank_correlations"):source="rank_correlations.csv; pairwise Spearman on eight-model AUC vectors, dependent split pairs"
            elif key.startswith("hyperparameter_summary"):source="hyperparameters.csv; complete canonical parameter vectors, selected candidate checked against every inner CV grid"
            elif key.startswith("algorithm_selection_strategy"):source="algorithm_selection_per_split.csv; training-inner-CV selection then held-out test evaluation"
            elif key.startswith("paired_scheme"):source="paired_scheme.csv; retuned minus fixed on identical split/model keys"
            elif key.startswith("leading_gaps"):source="leading_gaps.csv; top-two gap is distinct from eight-model range"
            ledger.append({"metric_key":key,"value":value,"source":source,"n_completed":n,"expected":expected,"status":summary["status"],"interpretation":"Clinical empirical percentile ranges are descriptive, not independent confidence intervals."})
    flatten(summary);csv(out/"number_source_map.csv",ledger)


def compare_reference(out,reference,report):
    """Same membership proof is required before cross-workflow pairing."""
    summary=read_json(out/"summary.json");rs=read_json(reference/"summary.json")
    ref_report=read_json(rs["verification_report"])
    if ref_report["verification_state"]!="passed":raise AssertionError("Reference verification did not pass")
    reference_raw_hash=read_json(Path(ref_report["run_directory"])/"run_manifest.json")["config"]["raw_sha256"]
    if reference_raw_hash!=report["raw_sha256"]:raise AssertionError("Reference raw data hash differs")
    if not summary["all_features"] or rs["all_features"]:raise AssertionError("Reference comparison expects current=all-features and reference=LASSO")
    current_hash={r["split"]:r["output_sha256"]["membership.csv"] for r in report["verified_checkpoints"]}
    reference_hash={r["split"]:r["output_sha256"]["membership.csv"] for r in ref_report["verified_checkpoints"]}
    common=sorted(set(current_hash)&set(reference_hash))
    for split in common:
        if current_hash[split]!=reference_hash[split]:raise AssertionError(f"Unpaired source-row membership: split {split}")
    current=pd.read_csv(out/"per_split_metrics.csv",float_precision="round_trip");ref=pd.read_csv(reference/"per_split_metrics.csv",float_precision="round_trip")
    current=current[(current.scheme=="retuned")&current.split.isin(common)]
    ref=ref[(ref.scheme=="retuned")&ref.split.isin(common)]
    merged=current.merge(ref,on=["split","model"],suffixes=("_all_features","_lasso"),validate="one_to_one")
    rows=[]
    for model,z in merged.groupby("model"):
        delta=z.auc_all_features-z.auc_lasso
        rows.append({"model":model,**moments(delta),"positive_fraction":float((delta>0).mean()),"absolute_delta_le_001":float((abs(delta)<=.01).mean()),"mean_delta_brier":float((z.brier_all_features-z.brier_lasso).mean())})
    result={"direction":"all features minus common LASSO; retuned workflows on identical train/test membership", "n_paired_splits":len(common),"current_n":len(current_hash),"reference_n":len(reference_hash),"current_only_splits":sorted(set(current_hash)-set(reference_hash)),"reference_only_splits":sorted(set(reference_hash)-set(current_hash)),"complete":summary["complete"] and rs["complete"] and len(common)==summary["expected"]==rs["expected"],"membership_sha256_agrees_for_all_paired_splits":True,"model_results":rows}
    csv(out/"all_features_vs_lasso_paired.csv",rows)
    write_json(out/"all_features_vs_lasso_summary.json",result)
    summary["all_features_vs_lasso"]=result;write_json(out/"summary.json",summary)
    report["cross_workflow_membership_pairing"]=result
    additional=[]
    def enumerate_numbers(value,key):
        if isinstance(value,dict):
            for k,x in value.items():enumerate_numbers(x,key+"."+k)
        elif isinstance(value,list):
            for i,x in enumerate(value):enumerate_numbers(x,key+f"[{i}]")
        elif isinstance(value,(int,float)) and not isinstance(value,bool):additional.append({"metric_key":key,"value":value,"source":"all_features_vs_lasso_paired.csv + both verified per_split_metrics.csv; exact membership.csv hashes agree","n_completed":len(common),"expected":summary["expected"],"status":"complete" if result["complete"] else "incomplete","interpretation":"Same-split exploratory workflow comparison; empirical ranges, not confidence intervals."})
    enumerate_numbers(result,"all_features_vs_lasso")
    existing=pd.read_csv(out/"number_source_map.csv",float_precision="round_trip")
    csv(out/"number_source_map.csv",pd.concat([existing,pd.DataFrame(additional)],ignore_index=True))


def self_test():
    """Meaningful metric and adversarial-audit tests; never mutate a checkpoint."""
    checks=[]
    for y,s,expected in [([0,1],[0.,1.],1.),([0,1],[1.,0.],0.),([0,1],[.5,.5],.5),([0,0,1,1],[0.,1.,1.,2.],.875)]:
        assert mann_whitney_auc(y,s)==expected
    checks.append("Mann-Whitney AUC: perfect, reversed, constant and tied scores")
    y=np.array([0,0,0,1,0,1,1,1]);p=np.array([.1,.2,.8,.4,.6,.5,.8,.9])
    a=irls_calibration(y,p);b=irls_calibration(1-y,p)
    assert abs(a["cal_slope"]+b["cal_slope"])<1e-6 and abs(a["cal_intercept_joint"]+b["cal_intercept_joint"])<1e-6
    checks.append("Independent IRLS calibration obeys exact outcome-complement symmetry")
    benchmark=ROOT/"results"/"strict_full_grid_benchmark"
    config=read_json(benchmark/"run_manifest.json")["config"];v=Verifier(config,RAW,{})
    tr,_=train_test_split(np.arange(len(v.rows)),train_size=.75,random_state=1000,stratify=v.y)
    meta=read_json(benchmark/"checkpoints"/"split_000"/"audit.json")["development"]["final_representation"]["preprocessing"]
    corrupt=json.loads(canonical(meta));corrupt["fit_rows_sha256"]="0"*64
    try:v.preprocessing(corrupt,v.rows[tr],"deliberately_corrupt_member_hash")
    except AssertionError:checks.append("Tampered preprocessing membership hash rejected")
    else:raise AssertionError("Corrupted hash was accepted")
    corrupt=json.loads(canonical(meta));key=next(iter(corrupt["impute_values"]));corrupt["impute_values"][key]+=1
    try:v.preprocessing(corrupt,v.rows[tr],"deliberately_corrupt_imputation")
    except AssertionError:checks.append("Tampered imputation value rejected from original training rows")
    else:raise AssertionError("Corrupted imputation was accepted")
    assert moments([1.])["sd"] is None and moments([])["n"]==0
    checks.append("Single-split and empty empirical summaries do not create NaN uncertainty")
    result={"passed":True,"checks":checks,"audit_script_sha256":sha(__file__),"source_files_mutated":False}
    write_json(ROOT/"audit"/"strict_summary_self_tests.json",result);print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",default=str(ROOT/"results"/"strict_lasso_200"))
    parser.add_argument("--output",default=str(ROOT/"results"/"strict_summary"))
    parser.add_argument("--raw",default=str(RAW))
    parser.add_argument("--expected",type=int)
    parser.add_argument("--verification",default=str(ROOT/"audit"/"strict_verification.json"))
    parser.add_argument("--allow-recorded-code",action="store_true",help="Explicitly permit current-code mismatch for inspection; publication-ready remains false")
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--reference-summary",help="Verified common-LASSO summary for same-split all-features comparison")
    args=parser.parse_args()
    self_test() if args.self_test else main(args)
