"""Non-identifying cohort checks; source data never modified."""
import ast, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
SOURCE=PROJECT/'联邦学习_冠脉病变'/'data'/'raw'/'source_clean.csv'
OLD=PROJECT/'临床篇_八模型'/'Q3_GS0重分析'/'scripts'/'01_q3_analysis.py'

def constants():
    result={}
    for node in ast.parse(OLD.read_text(encoding='utf-8')).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in {'FEATURES_49','CATEGORICAL','RANGE'}:
                    result[target.id]=ast.literal_eval(node.value)
    return result

def main():
    cs=constants()
    columns=list(dict.fromkeys(cs['FEATURES_49']+['patient_ID','Gensini评分','白细胞计数','中性粒细胞计数']))
    d=pd.read_csv(SOURCE,encoding='utf-8-sig',usecols=columns,float_precision='round_trip');gs=pd.to_numeric(d['Gensini评分'],errors='raise')
    mask=gs.ne(0);clean=d.loc[mask].copy();s=gs.loc[mask]
    summary={'source_n':len(d),'gs0_excluded':int((gs==0).sum()),'work_cohort_n':int(mask.sum()),
        'low_nonzero_gs':[float(x) for x in sorted(s[(s>0)&(s<2)].tolist())],
        'low_nonzero_eligibility':'awaiting_author_chart_verification','high_gs_gt37':int((s>37).sum()),
        'low_gs_le37':int((s<=37).sum()),'median_gs':float(s.median()),'mean_gs':float(s.mean()),
        'candidate_count':len(cs['FEATURES_49']),'uses_diagnosis_subtype':False,
        'raw_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
    ids=clean['patient_ID'].astype('string').str.strip()
    summary['record_id_missing']=int((ids.isna()|ids.eq('')).sum())
    summary['record_id_duplicate_rows']=int(ids.duplicated(keep=False).sum())
    summary['record_id_unique']=int(ids.nunique())
    summary['id_scope_note']='Checks recorded patient_ID only; unique values do not prove no patient has multiple hospital identifiers.'
    # Derived definitions apply patient-wise, not estimated using validation data.
    wbc=pd.to_numeric(clean['白细胞计数'],errors='coerce');neu=pd.to_numeric(clean['中性粒细胞计数'],errors='coerce')
    clean['dNLR']=np.where((neu>=0)&(wbc-neu>0),neu/(wbc-neu),np.nan)
    clean['TC-HDLDL']=clean['总胆固醇']-clean['高密度脂蛋白']-clean['低密度脂蛋白']
    rows=[];baseline=[]
    groups={'总体':clean.index,'GS≤37':s[s<=37].index,'GS>37':s[s>37].index}
    membership=ROOT/'results/strict_primary_oof/checkpoints/primary_42/membership.csv'
    if membership.exists():
        members=pd.read_csv(membership)
        # Membership keys are local raw-row numbers, never patient identifiers.
        source_key='source_row' if 'source_row' in members else 'source_row_index'
        for role in ['train','test']:
            groups[{'train':'单次开发集','test':'单次测试集'}[role]]=members.loc[members['partition']==role,source_key].to_numpy()
    for name in cs['FEATURES_49']:
        x=pd.to_numeric(clean[name],errors='coerce');before=int(x.isna().sum());bad=pd.Series(False,index=x.index)
        rule='无固定数值范围规则'
        if name in cs['RANGE']:
            lo,hi=cs['RANGE'][name];bad=x.notna()&((x<lo)|(x>hi));x=x.mask(bad)
            rule=f'<{lo}或>{hi}置为缺失（合理性核查界值，非绝对生理界限）'
        display={'心室收缩容量':'左心室收缩末期容积（LVESV，mL）','TC-HDLDL':'残余胆固醇计算值','血清肌钙蛋白T':'血清肌钙蛋白T（检测平台待核实）'}.get(name,name)
        rows.append({'source_column':name,'display_name':display,'n':len(clean),'missing_before':before,
            'range_flagged':int(bad.sum()),'missing_after':int(x.isna().sum()),'rule':rule,
            'median_after':float(x.median()),'q1_after':float(x.quantile(.25)),'q3_after':float(x.quantile(.75)),
            'role':'categorical' if name in cs['CATEGORICAL'] else 'continuous'})
        for group,indices in groups.items():
            values=x.loc[indices];valid=values.dropna()
            item={'source_column':name,'display_name':display,'group':group,'n':len(values),
                  'observed_n':len(valid),'missing_n':int(values.isna().sum()),
                  'median':float(valid.median()),'q1':float(valid.quantile(.25)),'q3':float(valid.quantile(.75)),
                  'category_counts_json':json.dumps({str(float(k)):int(v) for k,v in valid.value_counts().sort_index().items()},ensure_ascii=False) if name in cs['CATEGORICAL'] else '',
                  'role':'categorical' if name in cs['CATEGORICAL'] else 'continuous'}
            baseline.append(item)
    out=ROOT/'results'/'cohort_audit';out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(rows).to_csv(out/'candidates.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(baseline).to_csv(out/'baseline.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
