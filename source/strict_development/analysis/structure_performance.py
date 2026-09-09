#!/usr/bin/env python3
"""Posthoc descriptive feature-Jaccard versus within-model split-pair AUC gaps.

No new model training, independent-sample inference, causal attribution, or edits
to the verified summary. Exact four-bin specification precedes computation.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from datetime import datetime, timezone
from pathlib import Path

for _name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_name]="1"
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]
MODELS=["Logistic回归","随机森林","K近邻","梯度提升","SVM","XGBoost","LightGBM","多层感知机"]
BINS=[("0<=J<0.25",0.,.25),("0.25<=J<0.50",.25,.50),("0.50<=J<0.75",.50,.75),("0.75<=J<=1",.75,1.)]
WARNINGS=[
    "Posthoc descriptive supplement; bins were fixed in the written specification before this supplement was computed.",
    "Pairs repeatedly share split endpoints, and clinical splits share patients; pair counts are not independent sample sizes.",
    "Both development and test patients change across splits, along with selected features and model parameters; no effect can be causally attributed to feature-set change.",
    "The 0.01 threshold is an exploratory numerical scale, not clinical equivalence or acceptable prediction loss.",
    "No P values or confidence intervals are computed or reported. AUC differences compare the same model across two splits, not different models on the same test sample.",
]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()


def read_json(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def write_json(p,x):
    p=Path(p);tmp=p.with_name(p.name+f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8");os.replace(tmp,p)


def write_csv(p,x,columns=None):
    frame=x if isinstance(x,pd.DataFrame) else pd.DataFrame(x,columns=columns)
    p=Path(p);tmp=p.with_name(p.name+f".tmp.{os.getpid()}")
    frame.to_csv(tmp,index=False,encoding="utf-8-sig",float_format="%.17g");os.replace(tmp,p)


def describe(x):
    x=np.asarray(x,dtype=float)
    if not len(x):return {"n":0,"mean":None,"sd":None,"min":None,"q25":None,"median":None,"q75":None,"max":None}
    if not np.isfinite(x).all():raise AssertionError("Nonfinite observations must not be silently omitted")
    return {"n":int(len(x)),"mean":float(x.mean()),"sd":float(x.std(ddof=1)) if len(x)>1 else None,
            "min":float(x.min()),"q25":float(np.quantile(x,.25)),"median":float(np.median(x)),"q75":float(np.quantile(x,.75)),"max":float(x.max())}


def group_index(j):
    if not 0<=j<=1:raise AssertionError("Jaccard outside [0,1]")
    return 0 if j<.25 else 1 if j<.5 else 2 if j<.75 else 3


def descriptive_spearman(x,y):
    if len(x)<2:return None,"Fewer than two observed split pairs"
    rx,ry=rankdata(x,method="average"),rankdata(y,method="average")
    if np.ptp(rx)==0 or np.ptp(ry)==0:return None,"Jaccard or absolute AUC difference is constant"
    return float(np.corrcoef(rx,ry)[0,1]),None


def validate_pair_table(frame,sets):
    ids=sorted(sets);expected=set(itertools.combinations(ids,2))
    if len(frame)!=len(expected):raise AssertionError("Incorrect total split-pair count")
    if frame.duplicated(["split_i","split_j"]).any():raise AssertionError("Duplicate Jaccard split-pair key")
    seen=set()
    for row in frame.itertuples(index=False):
        key=(int(row.split_i),int(row.split_j))
        if key not in expected:raise AssertionError("Unexpected or reversed Jaccard split-pair key")
        a,b=sets[key[0]],sets[key[1]];union=a|b
        if not union:raise AssertionError("Both feature sets are empty")
        observed=len(a&b)/len(union)
        if not np.isfinite(row.jaccard) or abs(observed-float(row.jaccard))>1e-12:raise AssertionError(f"Jaccard mismatch at {key}")
        seen.add(key)
    if seen!=expected:raise AssertionError("Missing Jaccard split pairs")


def run(args):
    source=Path(args.input).resolve();out=Path(args.output).resolve()
    if source==out or source in out.parents:raise ValueError("Output must be outside the verified summary directory")
    out.mkdir(parents=True,exist_ok=True)
    meta=read_json(source/"summary.json");verification_path=Path(meta["verification_report"])
    verification=read_json(verification_path)
    if verification["verification_state"]!="passed":raise AssertionError("Source summary has not passed independent verification")
    if meta["all_features"] or meta["primary_seed42"]:raise AssertionError("This specification requires repeated common-LASSO main results")
    spec=ROOT/"audit"/"structure_performance_spec.md"
    inputs=[source/n for n in ["summary.json","feature_sets.csv","jaccard.csv","per_split_metrics.csv"]]+[verification_path,spec,Path(__file__)]
    hashes={str(p):sha(p) for p in inputs}
    features=pd.read_csv(source/"feature_sets.csv",float_precision="round_trip")
    auc=pd.read_csv(source/"per_split_metrics.csv",float_precision="round_trip")
    pairs=pd.read_csv(source/"jaccard.csv",float_precision="round_trip").sort_values(["split_i","split_j"]).reset_index(drop=True)
    if features.split.duplicated().any():raise AssertionError("Duplicate feature-set split")
    sets={}
    for row in features.itertuples(index=False):
        selected=str(row.features).split("|")
        if len(set(selected))!=len(selected) or len(selected)!=int(row.n_selected):raise AssertionError("Feature-count mismatch")
        sets[int(row.split)]=set(selected)
    n=len(sets)
    if n!=int(meta["n_completed"]) or n!=int(verification["n_verified"]):raise AssertionError("Source completion counts disagree")
    if n>200:raise AssertionError("This specification is for at most 200 planned main splits")
    if set(sets)!={int(r["split"]) for r in verification["verified_checkpoints"]}:raise AssertionError("Feature splits differ from verified checkpoints")
    auc=auc[auc.scheme=="retuned"].copy()
    if len(auc)!=n*8 or auc.duplicated(["split","model"]).any() or set(auc.model)!=set(MODELS) or set(auc.split)!=set(sets):raise AssertionError("Incomplete retuned model-by-split grid")
    if not np.isfinite(auc.auc).all() or not auc.auc.between(0,1).all():raise AssertionError("Invalid retuned AUC")
    validate_pair_table(pairs,sets)
    complete=bool(n==200 and len(pairs)==19900 and meta["expected"]==200 and meta["complete"] and verification["complete"] and verification["publication_result_set_ready"])
    status="complete" if complete else "incomplete"
    p=auc.pivot(index="split",columns="model",values="auc")[MODELS].sort_index()
    pair_data=[];all_rows=[];group_rows=[];identity_rows=[]
    for model in MODELS:
        left=p.loc[pairs.split_i,model].to_numpy();right=p.loc[pairs.split_j,model].to_numpy();d=abs(left-right);j=pairs.jaccard.to_numpy()
        rho,undefined=descriptive_spearman(j,d)
        all_rows.append({"model":model,"n_splits":n,"n_pairs":len(d),"jaccard_mean":describe(j)["mean"],"jaccard_q25":describe(j)["q25"],"jaccard_median":describe(j)["median"],"jaccard_q75":describe(j)["q75"],**{"abs_delta_auc_"+k:v for k,v in describe(d).items() if k!="n"},"spearman_jaccard_abs_delta_auc":rho,"spearman_undefined_reason":undefined,"fraction_abs_delta_auc_le_001":float((d<=.01).mean()) if len(d) else None,"status":status})
        bins=np.array([group_index(v) for v in j],dtype=int)
        for i,(label,lo,hi) in enumerate(BINS):
            mask=bins==i;dd=d[mask];jj=j[mask]
            group_rows.append({"model":model,"jaccard_group":label,"n_pairs":int(mask.sum()),"median_jaccard":describe(jj)["median"],"median_abs_delta_auc":describe(dd)["median"],"q25_abs_delta_auc":describe(dd)["q25"],"q75_abs_delta_auc":describe(dd)["q75"],"fraction_abs_delta_auc_le_001":float((dd<=.01).mean()) if len(dd) else None,"status":status})
        for identity,mask in [("different_feature_sets",j<1),("identical_feature_sets",j==1)]:
            dd=d[mask];jj=j[mask]
            identity_rows.append({"model":model,"feature_set_relation":identity,"n_pairs":int(mask.sum()),"median_jaccard":describe(jj)["median"],"median_abs_delta_auc":describe(dd)["median"],"q25_abs_delta_auc":describe(dd)["q25"],"q75_abs_delta_auc":describe(dd)["q75"],"fraction_abs_delta_auc_le_001":float((dd<=.01).mean()) if len(dd) else None,"status":status})
        pair_data.extend({"model":model,"split_b":int(b),"split_c":int(c),"jaccard":float(jj),"auc_b":float(ab),"auc_c":float(ac),"abs_delta_auc":float(delta),"jaccard_group":BINS[int(group)][0],"status":status} for b,c,jj,ab,ac,delta,group in zip(pairs.split_i,pairs.split_j,j,left,right,d,bins))
    for model in MODELS:
        if sum(r["n_pairs"] for r in group_rows if r["model"]==model)!=len(pairs):raise AssertionError("Jaccard bins do not partition all observed pairs")
    summary={"analysis":"posthoc_feature_set_similarity_and_same_algorithm_split_pair_auc_difference","n_splits":n,"expected_splits":200,"n_pairs_per_model":len(pairs),"expected_pairs_per_model":19900,"n_models":8,"total_model_pair_rows":len(pair_data),"complete":complete,"status":status,"scheme":"retuned","bins":[r[0] for r in BINS],"exploratory_absolute_auc_difference_threshold":.01,"warnings":WARNINGS,"jaccard_distribution":describe(pairs.jaccard),"model_summary":all_rows,"jaccard_groups":group_rows,"same_vs_different_feature_sets":identity_rows,"source_summary":str(source),"source_verification":str(verification_path)}
    write_csv(out/"pairwise_structure_performance.csv",pair_data,columns=["model","split_b","split_c","jaccard","auc_b","auc_c","abs_delta_auc","jaccard_group","status"])
    write_csv(out/"model_summary.csv",all_rows);write_csv(out/"jaccard_group_summary.csv",group_rows);write_csv(out/"same_vs_different_feature_sets.csv",identity_rows)
    write_json(out/"summary.json",summary)
    ledger=[]
    def numbers(value,key=""):
        if isinstance(value,dict):
            for k,x in value.items():numbers(x,key+"."+k if key else k)
        elif isinstance(value,list):
            for i,x in enumerate(value):numbers(x,key+f"[{i}]")
        elif isinstance(value,(int,float)) and not isinstance(value,bool):ledger.append({"metric_key":key,"value":value,"source":"pairwise_structure_performance.csv; verified retuned AUC and recomputed feature-set Jaccard","n_splits":n,"n_pairs_per_model":len(pairs),"status":status,"interpretation":"Descriptive posthoc statistic on dependent split pairs, not causal and not independent-sample inference."})
    numbers(summary);write_csv(out/"number_source_map.csv",ledger)
    unchanged=all(sha(path)==digest for path,digest in hashes.items())
    if not unchanged:raise RuntimeError("An input changed during this supplement; rerun against a stable verified snapshot")
    manifest={"completed_utc":datetime.now(timezone.utc).isoformat(),"status":status,"source_files_unchanged":unchanged,"input_and_code_sha256":hashes,"output_sha256":{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!="input_output_manifest.json"},"n_splits":n,"n_pairs_per_model":len(pairs),"complete":complete}
    write_json(out/"input_output_manifest.json",manifest)
    print(json.dumps({"status":status,"n_splits":n,"n_pairs_per_model":len(pairs),"n_model_pair_rows":len(pair_data),"output":str(out),"source_files_unchanged":True},ensure_ascii=False))


def self_test():
    assert [group_index(j) for j in [0.,.2499,.25,.4999,.5,.7499,.75,1.]]==[0,0,1,1,2,2,3,3]
    assert describe([])["median"] is None and describe([])["n"]==0
    assert descriptive_spearman([1,2,3],[3,2,1])[0]==-1.
    assert descriptive_spearman([1,1],[.2,.3])[0] is None
    sets={0:{"a"},1:{"a","b"},2:{"c"}}
    pairs=pd.DataFrame([{"split_i":0,"split_j":1,"jaccard":.5},{"split_i":0,"split_j":2,"jaccard":0.},{"split_i":1,"split_j":2,"jaccard":0.}])
    validate_pair_table(pairs,sets)
    corrupt=pairs.copy();corrupt.loc[0,"jaccard"]=.4
    try:validate_pair_table(corrupt,sets)
    except AssertionError:pass
    else:raise AssertionError("Corrupted Jaccard accepted")
    try:validate_pair_table(pairs.iloc[:2],sets)
    except AssertionError:pass
    else:raise AssertionError("Missing split-pair accepted")
    result={"passed":True,"tests":["Exact bin boundaries","Empty-group nulls","Descriptive inverse rank correlation","Constant-variable undefined correlation","Independent Jaccard recomputation","Corrupted Jaccard rejected","Missing pair rejected"],"script_sha256":sha(__file__)}
    write_json(ROOT/"audit"/"structure_performance_tests.json",result);print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",default=str(ROOT/"results"/"strict_summary"))
    parser.add_argument("--output",default=str(ROOT/"results"/"structure_performance"))
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    self_test() if args.self_test else run(args)
