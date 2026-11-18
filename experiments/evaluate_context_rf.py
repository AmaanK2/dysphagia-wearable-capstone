import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from cnn import evaluate,best_threshold
from validate import select_event_settings
base=Path('_cache/zenodo_feature_build')
meta=json.loads((base/'records.json').read_text())
prev=Path('experiments/grouped_cv_event_tuned')
plan=json.loads((prev/'fold_plan.json').read_text())['folds']
with np.load(base/'features.npz') as d:
    x,y,ids,groups=(d[k] for k in ('X','y','ids','groups'))
for idx,offset in enumerate((-8,-4,-2,2,4,8),1):
    j=np.clip(np.arange(len(x))+offset,0,len(x)-1)
    j=np.where(ids[j]==ids,j,np.arange(len(x)))
    if idx==1: scaled=[x]
    scaled.append(x[j][:,[0,1,2,3,14,15,16,17,28,29,30,31,42,43,44,45]])
x=np.concatenate(scaled,axis=1)
results=[]
for i,fold in enumerate(plan,1):
    m=np.isin(groups,fold['train'])
    clf=RandomForestClassifier(n_estimators=200,max_depth=14,min_samples_leaf=5,class_weight='balanced_subsample',random_state=2026+i,n_jobs=3)
    clf.fit(x[m],y[m])
    def predict(role):
        rows=[r for r in meta if r['participant'] in fold[role]]
        m=np.isin(groups,fold[role]); p=clf.predict_proba(x[m])[:,1]; keys=ids[m]
        return rows,{r['id']:p[keys==r['id']] for r in rows}
    rows,p=predict('validation'); _,cut=best_threshold(rows,p)
    rules=select_event_settings(rows,p,cut)
    rows,p=predict('assessment')
    results.append({str(iou):evaluate(rows,p,rules['tuned_threshold'],iou,rules['merge_gap'],rules['min_duration']) for iou in (.3,.5)})
    print(i,[(k,v['f1']) for k,v in results[-1].items()],flush=True)
out=Path('output/context_rf');out.mkdir(parents=True,exist_ok=True)
summary={}
for iou in ('0.3','0.5'):
    tp,fp,fn=(sum(f[iou][k] for f in results) for k in ('tp','fp','fn'))
    summary[iou]={'f1':2*tp/(2*tp+fp+fn),'precision':tp/(tp+fp),'recall':tp/(tp+fn)}
(out/'metrics.json').write_text(json.dumps(summary,indent=2))
print(summary)
