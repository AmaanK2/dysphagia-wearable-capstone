"""Plot development-only outer-fold results and classify annotation-relative errors."""
from ..paths import PROJECT_ROOT, BASELINE_REPORTS, CNN_REPORTS, CV_REPORTS, LABEL_REVIEW, HRCA_SIGNALS
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from .baseline import events, ANNOTATION_NOTICE

root=PROJECT_ROOT
parser=argparse.ArgumentParser()
parser.add_argument('--cv-dir',type=Path,default=CV_REPORTS)
parser.add_argument('--baseline',type=Path,default=BASELINE_REPORTS)
args=parser.parse_args();p=args.cv_dir
s=json.loads((p/'summary.json').read_text())
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
for name,label,offset,color in [('rf','Random forest',-.18,'#d18a43'),('cnn','PyTorch CNN',.18,'#326d91')]:
    values=s['models'][name]['fold_f1'];bars=axes[0].bar(np.arange(3)+offset,values,.35,label=label,color=color)
    axes[0].bar_label(bars,labels=[f'{v:.0%}' for v in values],padding=3,fontsize=9)
    values=[s['models'][name][k] for k in ['precision','recall','f1']]
    bars=axes[1].bar(np.arange(3)+offset,values,.35,label=label,color=color)
    axes[1].bar_label(bars,labels=[f'{v:.0%}' for v in values],padding=3,fontsize=9)
axes[0].set_xticks(np.arange(3),['Fold 1','Fold 2','Fold 3']);axes[0].set_ylabel('Outer-assessment event F1')
axes[0].set_title('Different participants in each fold')
axes[1].set_xticks(np.arange(3),['Precision','Recall','F1']);axes[1].set_title('Pooled out-of-fold events')
for ax in axes:ax.set_ylim(0,1);ax.legend(fontsize=9)
fig.suptitle('148 development participants | original 27 pilot test participants excluded\nInner validation selects checkpoints and thresholds; IoU ≥ 0.30',fontsize=13)
fig.savefig(p/'grouped_validation.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True,layout='constrained')
for i,ax in enumerate(axes,1):
    h=json.loads((p/f'fold_{i}/cnn_history.json').read_text())
    ax.plot([r['epoch'] for r in h],[r['validation_event_f1'] for r in h],marker='o',ms=3)
    chosen=s['folds'][i-1]['models']['cnn']['training_details']['best_epoch']
    ax.axvline(chosen,ls=':',color='black');ax.set_title(f'Fold {i}: selected epoch {chosen}');ax.set_xlabel('Epoch');ax.set_ylim(0,1)
axes[0].set_ylabel('Inner-validation event F1');fig.suptitle('Checkpoint selection uses inner validation only')
fig.savefig(p/'inner_validation_curves.png',dpi=150);plt.close(fig)

meta={r['id']:r for r in json.loads((args.baseline/'records.json').read_text())}
errors=[];counts={};tp=fp=fn=0
for i in range(1,4):
    d=np.load(p/f'fold_{i}/cnn_assessment_predictions.npz');threshold=float(d['threshold'])
    for key in sorted(set(d['ids'])):
        r=meta[key];score=d['scores'][d['ids']==key]
        pred=events(r['times'],score,threshold,r['duration']);truth=r['intervals']
        ious=np.zeros((len(pred),len(truth)));overlap=np.zeros_like(ious)
        for a,(l,h) in enumerate(pred):
            for b,(lo,hi) in enumerate(truth):
                overlap[a,b]=max(0.,min(h,hi)-max(l,lo))
                ious[a,b]=overlap[a,b]/(h-l+hi-lo-overlap[a,b])
        pairs=[]
        if pred and truth:
            valid=ious>=.3
            ri,ci=linear_sum_assignment(-(valid*(min(ious.shape)+1)+ious*valid))
            pairs=[(a,b) for a,b in zip(ri,ci) if valid[a,b]]
        matched_pred={a for a,b in pairs};matched_truth={b for a,b in pairs};tp+=len(pairs)
        for a,(l,h) in enumerate(pred):
            if a in matched_pred:continue
            fp+=1
            kind='overlaps_annotation_boundary_or_duplicate' if len(truth) and overlap[a].max()>0 else 'no_annotation_overlap'
            counts[kind]=counts.get(kind,0)+1
            errors.append({'fold':i,'participant':r['participant'],'recording':key,'type':'extra_detection',
                           'start_seconds':l,'end_seconds':h,'category':kind})
        for b,(l,h) in enumerate(truth):
            if b in matched_truth:continue
            fn+=1;errors.append({'fold':i,'participant':r['participant'],'recording':key,'type':'missed_annotation',
                                 'start_seconds':l,'end_seconds':h,'category':'requires_signal_review'})
assert (tp,fp,fn)==tuple(s['models']['cnn'][k] for k in ['tp','fp','fn'])
with (p/'cnn_error_review.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['fold','participant','recording','type','start_seconds','end_seconds','category']);w.writeheader();w.writerows(errors)
(p/'cnn_error_categories.json').write_text(json.dumps(counts,indent=2))
lines=['# Grouped development validation', '', ANNOTATION_NOTICE, '',
       'Three outer folds across 148 development participant IDs. Each outer assessment participant appears exactly once; the original 27 pilot test participants are excluded entirely.',
       'Within each fold, 20% of the remaining participants form a separate inner-validation set. Both models select thresholds there; the CNN also selects its checkpoint there. Normalization is fitted only to inner-training participants.', '',
       '| Pooled outer-assessment metric | Random forest | CNN |','| --- | ---: | ---: |']
for metric,label in [('precision','Event precision'),('recall','Event recall'),('f1','Event F1'),('strict_iou_0_5_f1','Strict event F1, IoU >= 0.50')]:
    lines.append(f"| {label} | {s['models']['rf'][metric]:.1%} | {s['models']['cnn'][metric]:.1%} |")
lines.append(f"| Extra detections / recording minute | {s['models']['rf']['false_events_per_recording_minute']:.2f} | {s['models']['cnn']['false_events_per_recording_minute']:.2f} |")
lines+=['', '## Fold-level F1, IoU >= 0.30', '', '| Fold | RF | CNN |','| --- | ---: | ---: |']
for i,f in enumerate(s['folds'],1):lines.append(f"| {i} | {f['models']['rf']['assessment_event']['f1']:.1%} | {f['models']['cnn']['assessment_event']['f1']:.1%} |")
lines+=['', '## CNN error review', '']
for name,n in counts.items():lines.append(f'- {n} extra detections: {name}.')
lines+=['', 'These categories describe overlap with annotations, not the physiological cause of an error. Missing/incomplete labels remain a possible issue. Review cnn_error_review.csv alongside the raw signals.', '',
        '## Limits', '',
        '- Exploratory internal validation on the same public dataset, not external clinical validation.',
        '- CNN changes both temporal context and architecture compared with RF. Matched-context ablations are still needed.',
        '- This is swallow timing detection, not dysphagia or aspiration prediction.',
        '- Original pilot results and weights are unchanged. No outer-assessment results were used to change these runs.',
        '- Cluster bootstrap intervals reflect assessment-participant variability for these fitted folds; they do not capture all retraining uncertainty.',
        '- Do not select the best outer fold as a production model. Fold checkpoints exist for reproducibility; the original CNN remains the single-split candidate.', '']
(p/'SUMMARY.md').write_text('\n'.join(lines))
print('\n'.join(lines[:24]))
