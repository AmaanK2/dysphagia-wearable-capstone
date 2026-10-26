"""Three outer participant folds with separate inner validation; pilot test excluded."""
from ..paths import PROJECT_ROOT, BASELINE_REPORTS, CNN_REPORTS, CV_REPORTS, LABEL_REVIEW, HRCA_SIGNALS
import argparse
import copy
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from .cnn import (SwallowCNN, Records, collate, fit_normalization, loss_value,
                 predict_records, best_threshold, evaluate)
from .baseline import bootstrap_interval, ANNOTATION_NOTICE


def fold_plan(meta, original, folds=3, seed=2026):
    excluded=set(original['test'])
    development=set(original['train'])|set(original['validation'])
    assert not development & excluded
    records=[r for r in meta if r['participant'] in development]
    groups=np.array([r['participant'] for r in records])
    plan=[]
    for outer,(dev,assessment) in enumerate(GroupKFold(n_splits=folds,shuffle=True,random_state=seed).split(groups,groups=groups)):
        tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed+outer).split(dev,groups=groups[dev]))
        roles={'train':sorted(set(groups[dev[tr]])),'validation':sorted(set(groups[dev[va]])),
               'assessment':sorted(set(groups[assessment]))}
        assert all(not set(roles[a])&set(roles[b]) for a,b in [('train','validation'),('train','assessment'),('validation','assessment')])
        assert not set().union(*(set(v) for v in roles.values())) & excluded
        plan.append(roles)
    all_assessed=[p for f in plan for p in f['assessment']]
    assert len(all_assessed)==len(set(all_assessed)) and set(all_assessed)==development
    return records,plan


def cnn_fold(train,validation,assessment,cache,out,seed,epochs,patience):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    mean,std=fit_normalization(train,cache)
    dataset=Records(train,cache,mean,std)
    loader=DataLoader(dataset,batch_size=8,shuffle=True,collate_fn=collate,generator=torch.Generator().manual_seed(seed))
    val_loader=DataLoader(Records(validation,cache,mean,std),batch_size=8,collate_fn=collate)
    positive=negative=0.
    for x,y,key in dataset:
        positive+=float(y.sum());negative+=len(y)-float(y.sum())
    weight=torch.tensor(negative/positive)
    model=SwallowCNN();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    best=-1.;best_epoch=0;threshold=None;checkpoint=None;history=[]
    for epoch in range(1,epochs+1):
        started=time.time();model.train();loss_sum=0.;count=0
        for x,y,lengths,keys in loader:
            optimizer.zero_grad(set_to_none=True)
            logits,n=model(x,lengths);loss=loss_value(logits,y,n,weight)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();loss_sum+=float(loss.detach());count+=1
        pred=predict_records(model,val_loader,validation,torch.device('cpu'))
        f1,t=best_threshold(validation,pred)
        history.append({'epoch':epoch,'loss':loss_sum/count,'validation_event_f1':f1,'threshold':t,'seconds':time.time()-started})
        (out/'cnn_history.json').write_text(json.dumps(history,indent=2))
        print(out.name,'CNN epoch',epoch,'validation F1',round(f1,4),flush=True)
        if f1>best+1e-8:
            best=f1;best_epoch=epoch;threshold=t;checkpoint=copy.deepcopy(model.state_dict())
        if epoch-best_epoch>=patience:break
    model.load_state_dict(checkpoint)
    torch.save({'state_dict':checkpoint,'mean':mean.tolist(),'std':std.tolist(),'threshold':threshold,
                'epoch':best_epoch,'scope':'Grouped development CV; original pilot test excluded'},out/'cnn.pt')
    assessment_loader=DataLoader(Records(assessment,cache,mean,std),batch_size=8,collate_fn=collate)
    predictions=predict_records(model,assessment_loader,assessment,torch.device('cpu'))
    return predictions,threshold,{'best_epoch':best_epoch,'completed_epochs':len(history),
                                 'inner_validation_f1':best,'normalization_participants':sorted({r['participant'] for r in train})}


def rf_fold(features,train,validation,assessment,out,seed):
    X,y,groups,ids=features
    train_people={r['participant'] for r in train}
    mask=np.isin(groups,list(train_people))
    model=RandomForestClassifier(n_estimators=200,max_depth=14,min_samples_leaf=5,
                                  class_weight='balanced_subsample',random_state=seed,n_jobs=2)
    model.fit(X[mask],y[mask])
    predictions={}
    for role,records in [('validation',validation),('assessment',assessment)]:
        selected=np.isin(ids,[r['id'] for r in records]);score=model.predict_proba(X[selected])[:,1]
        selected_ids=ids[selected]
        predictions[role]={r['id']:score[selected_ids==r['id']] for r in records}
    f1,threshold=best_threshold(validation,predictions['validation'])
    joblib.dump({'model':model,'threshold':threshold},out/'rf.joblib')
    return predictions['assessment'],threshold,{'inner_validation_f1':f1}


def paired_ci(cnn,rf):
    people=sorted(cnn)
    assert people==sorted(rf)
    a=np.array([cnn[p] for p in people]);b=np.array([rf[p] for p in people]);rng=np.random.default_rng(42)
    differences=[]
    for _ in range(1000):
        idx=rng.integers(0,len(people),len(people))
        def f1(rows):
            tp,fp,fn,seconds=rows.sum(axis=0)
            return 2*tp/max(2*tp+fp+fn,1)
        differences.append(f1(a[idx])-f1(b[idx]))
    return np.percentile(differences,[2.5,97.5]).tolist()


def main():
    print(ANNOTATION_NOTICE, flush=True)
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',type=Path,default=BASELINE_REPORTS)
    p.add_argument('--cache',type=Path,default=PROJECT_ROOT / 'data/processed/hrca_cnn')
    p.add_argument('--output',type=Path,default=CV_REPORTS)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--patience',type=int,default=5)
    args=p.parse_args()
    if (args.output/'fold_plan.json').exists():raise ValueError('Use a fresh output directory to preserve experiments')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    meta=json.loads((args.baseline/'records.json').read_text())
    manifest=json.loads((args.cache/'manifest.json').read_text())
    if manifest['records']!={r['id']:r['sha256'] for r in meta}:
        raise ValueError('Waveform cache does not match the baseline recording manifest')
    original=json.loads((args.baseline/'participant_splits.json').read_text())
    records,plan=fold_plan(meta,original)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'config.json').write_text(json.dumps({'seed':2026,'outer_folds':3,'inner_validation_fraction':.2,
        'cnn_max_epochs':args.epochs,'cnn_patience':args.patience,'torch_version':str(torch.__version__),
        'cnn_learning_rate':.001,'cnn_weight_decay':.0001,'cnn_batch_size':8,'cnn_threads':4,
        'rf_trees':200,'rf_max_depth':14,'rf_min_samples_leaf':5,'rf_class_weight':'balanced_subsample',
        'threshold_grid':np.linspace(.05,.95,37).tolist(),'iou_thresholds':[.3,.5],
        'model_context':'CNN ~2.6s; RF 0.5s; not an isolated architecture ablation'},indent=2))
    (args.output/'fold_plan.json').write_text(json.dumps({'original_test_excluded':original['test'],'folds':plan},indent=2))
    data=np.load(args.baseline/'features.npz',allow_pickle=False)
    development=set(original['train'])|set(original['validation'])
    keep=np.isin(data['groups'],list(development))
    features=tuple(data[k][keep] for k in ['X','y','groups','ids'])
    assert not set(features[2])&set(original['test'])
    pooled={'cnn':{},'rf':{}};fold_results=[]
    # Never change preprocessing, model settings or event postprocessing after seeing outer assessments.
    for i,roles in enumerate(plan,1):
        out=args.output/f'fold_{i}';out.mkdir()
        role_records={k:[r for r in records if r['participant'] in v] for k,v in roles.items()}
        train,val,assessment=[role_records[k] for k in ['train','validation','assessment']]
        print('Starting fold',i,{k:len(v) for k,v in roles.items()},flush=True)
        fold={'fold':i,'participant_counts':{k:len(v) for k,v in roles.items()},'models':{}}
        for name in ['rf','cnn']:
            if name=='rf':pred,t,info=rf_fold(features,train,val,assessment,out,2026+i)
            else:pred,t,info=cnn_fold(train,val,assessment,args.cache,out,2026+i,args.epochs,args.patience)
            result={'threshold':t,'training_details':info,'assessment_event':evaluate(assessment,pred,t),
                    'assessment_strict_event':evaluate(assessment,pred,t,.5)}
            fold['models'][name]=result
            # Each outer fold has its own inner-validation-selected threshold.
            pooled[name].update({r['id']:{'score':pred[r['id']],'threshold':t} for r in assessment})
            ordered_ids=np.concatenate([[r['id']]*len(r['times']) for r in assessment])
            np.savez_compressed(out/(name+'_assessment_predictions.npz'),
                                ids=ordered_ids,scores=np.concatenate([pred[r['id']] for r in assessment]),threshold=t)
            print('Fold',i,name,'assessment F1',round(result['assessment_event']['f1'],4),flush=True)
        (out/'metrics.json').write_text(json.dumps(fold,indent=2));fold_results.append(fold)
    # Pool counts rather than applying a shared threshold across distinct trained folds.
    summary={'design':'Three grouped outer folds, each with 20% inner participant validation; exploratory development check',
             'original_test_participants_excluded':len(original['test']),'development_participants':len(development),
             'folds':fold_results,'models':{},
             'limits':['Same source dataset; not external clinical validation','CNN has longer context than RF',
                       'Participant bootstrap does not capture repeated-training uncertainty','No dysphagia or aspiration labels']}
    for name in ['rf','cnn']:
        events=[f['models'][name]['assessment_event'] for f in fold_results]
        strict=[f['models'][name]['assessment_strict_event'] for f in fold_results]
        by_person={p:c for e in events for p,c in e['participant_counts'].items()}
        tp,fp,fn,seconds=np.array(list(by_person.values())).sum(axis=0)
        stp=sum(e['tp'] for e in strict);sfp=sum(e['fp'] for e in strict);sfn=sum(e['fn'] for e in strict)
        summary['models'][name]={'precision':tp/(tp+fp),'recall':tp/(tp+fn),'f1':2*tp/(2*tp+fp+fn),
            'tp':int(tp),'fp':int(fp),'fn':int(fn),'false_events_per_recording_minute':fp/(seconds/60),
            'strict_iou_0_5_f1':2*stp/(2*stp+sfp+sfn),'participant_counts':by_person,
            'bootstrap_95_ci':bootstrap_interval(by_person),'fold_f1':[e['f1'] for e in events]}
    summary['paired_bootstrap_95_ci_f1_difference_cnn_minus_rf']=paired_ci(
        summary['models']['cnn']['participant_counts'],summary['models']['rf']['participant_counts'])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print('POOLED DEVELOPMENT RESULTS',json.dumps({k:{a:b for a,b in v.items() if a not in ['participant_counts','bootstrap_95_ci']} for k,v in summary['models'].items()}),flush=True)


if __name__=='__main__':main()
