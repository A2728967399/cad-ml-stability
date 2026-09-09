#!/usr/bin/env python3
"""Descriptive pooled-versus-fold OOF AUC audit; no fitting or primary replacement."""
from __future__ import annotations
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key]="1"
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/"results"/"strict_primary_oof"
CHECKPOINT=RUN/"checkpoints"/"primary_42"
OUT=ROOT/"results"/"strict_primary_summary"
MODELS=["Logistic回归","随机森林","K近邻","梯度提升","SVM","XGBoost","LightGBM","多层感知机"]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):h.update(chunk)
    return h.hexdigest()


def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def auc(y,score):
    y=np.asarray(y,dtype=int);score=np.asarray(score,dtype=float);n1=int(y.sum());n0=len(y)-n1
    if n1==0 or n0==0 or not np.isfinite(score).all():raise AssertionError("AUC undefined")
    return float((rankdata(score,method="average")[y==1].sum()-n1*(n1+1)/2)/(n1*n0))


def main():
    verification=load(ROOT/"audit"/"strict_primary_verification.json")
    if verification["verification_state"]!="passed" or not verification["complete"] or verification["n_verified"]!=1:raise AssertionError("Complete verified primary OOF required")
    source=CHECKPOINT/"oof_predictions.csv";manifest=load(CHECKPOINT/"manifest.json")
    if manifest["state"]!="complete" or sha(source)!=manifest["output_sha256"]["oof_predictions.csv"]:raise AssertionError("OOF checkpoint hash/status mismatch")
    verified=verification["verified_checkpoints"][0]
    if verified["output_sha256"]["oof_predictions.csv"]!=sha(source):raise AssertionError("OOF predictions differ from independently verified predictions")
    data=pd.read_csv(source,float_precision="round_trip")
    if "oof_fold" not in data:raise AssertionError("Expected explicit oof_fold column absent; recover fold membership before analysis")
    if set(data.model)!=set(MODELS) or set(data.oof_fold)!=set(range(5)):raise AssertionError("Expected eight models and five OOF folds")
    if data.duplicated(["source_row","model"]).any():raise AssertionError("Duplicate model-patient OOF prediction")
    rows=[];fold_rows=[]
    for model in MODELS:
        d=data[data.model==model].copy()
        if len(d)!=978 or int(d.y.sum())!=484:raise AssertionError("OOF cohort mismatch")
        frows=[]
        for fold,z in d.groupby("oof_fold"):
            frows.append({"model":model,"oof_fold":int(fold),"n_validation":len(z),"events_validation":int(z.y.sum()),"score_auc":auc(z.y,z.score),"probability_auc":auc(z.y,z.p),"score_mean":float(z.score.mean()),"score_sd":float(z.score.std(ddof=1)),"score_min":float(z.score.min()),"score_max":float(z.score.max()),"probability_mean":float(z.p.mean())})
        fold_rows.extend(frows);scores=np.array([r["score_auc"] for r in frows]);probs=np.array([r["probability_auc"] for r in frows])
        pooled=auc(d.y,d.score);pooled_p=auc(d.y,d.p)
        rows.append({"model":model,"n_oof":len(d),"n_folds":5,"pooled_score_auc":pooled,"mean_fold_score_auc":float(scores.mean()),"sd_fold_score_auc":float(scores.std(ddof=1)),"min_fold_score_auc":float(scores.min()),"max_fold_score_auc":float(scores.max()),"pooled_minus_mean_fold_auc":pooled-float(scores.mean()),"pooled_probability_auc":pooled_p,"mean_fold_probability_auc":float(probs.mean()),"min_fold_probability_auc":float(probs.min()),"max_fold_probability_auc":float(probs.max()),"pooled_probability_minus_score_auc":pooled_p-pooled})
    for name,records in [("oof_score_scale_audit.csv",rows),("oof_score_scale_by_fold.csv",fold_rows)]:
        pd.DataFrame(records).to_csv(OUT/name,index=False,encoding="utf-8-sig",float_format="%.17g")
    ledger=[{"model":r["model"],"metric":k,"value":value,"source":"oof_score_scale_audit.csv / oof_score_scale_by_fold.csv; verified primary oof_predictions.csv","definition":"Pooled and within-fold AUC are distinct descriptive quantities; no CI or fixed-model ROC interpretation."} for r in rows for k,value in r.items() if isinstance(value,(int,float))]
    pd.DataFrame(ledger).to_csv(OUT/"oof_score_scale_number_source_map.csv",index=False,encoding="utf-8-sig",float_format="%.17g")
    info={"completed_utc":datetime.now(timezone.utc).isoformat(),"status":"complete_descriptive_primary_oof_supplement","models":rows,"definitions":{"pooled_score_auc":"Mann-Whitney AUC on concatenated held-out scores from five separately fitted outer-fold models.","mean_fold_score_auc":"Unweighted arithmetic mean of five within-outer-fold AUCs; range is empirical min-max, not a confidence interval.","pooled_probability_auc":"Mann-Whitney AUC on concatenated OOF probabilities. SVM value is an explicitly supplementary scale check, not a replacement primary AUC."},"limitations":["SVM decision_function scores originate from different fitted models across folds; pooled ordering includes cross-fold comparisons that may be affected by model-specific score scales.","Within-fold AUC comparisons are made inside each fitted model's score scale. Their mean and a pooled AUC are different summaries and need not agree.","Probabilities also come from separately fitted/calibrated models; a common 0-1 range does not make these scores predictions of one fixed final model.","No P values, confidence intervals, causal scale-attribution claims or new model training; no original primary metrics were replaced."],"source_prediction_sha256":sha(source),"script_sha256":sha(__file__),"output_sha256":{name:sha(OUT/name) for name in ["oof_score_scale_audit.csv","oof_score_scale_by_fold.csv","oof_score_scale_number_source_map.csv"]}}
    (OUT/"oof_score_scale_audit.json").write_text(json.dumps(info,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    if sha(source)!=info["source_prediction_sha256"]:raise AssertionError("Source predictions changed during supplement")
    print(json.dumps(info["models"],ensure_ascii=False,indent=2))


if __name__=="__main__":main()
