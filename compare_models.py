"""Summarize frozen RF and CNN results without choosing new settings."""
from pathlib import Path
import json,io,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train import events, ANNOTATION_NOTICE

root=Path(__file__).resolve().parent
p=root/'experiments/cnn_v1'
rf=json.loads((root/'results/metrics.json').read_text())['models']['rf_combined']
cnn=json.loads((p/'metrics.json').read_text())
config=json.loads((p/'config.json').read_text())
history=json.loads((p/'history.json').read_text())
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
axes[0].plot([r['epoch'] for r in history],[r['training_loss_mean_batch'] for r in history],marker='o',color='#245b7a')
axes[0].set(xlabel='Epoch',ylabel='Weighted binary loss',title='Training loss')
axes[1].plot([r['epoch'] for r in history],[r['validation_event_f1'] for r in history],marker='o',label='CNN')
axes[1].axhline(rf['validation_event']['f1'],ls='--',color='#b65d29',label='Frozen RF baseline')
axes[1].axvline(cnn['best_epoch'],ls=':',color='black',label='Selected epoch')
axes[1].set(xlabel='Epoch',ylabel='Event F1',title='Validation participants only',ylim=(0,1))
axes[1].legend(fontsize=8)
fig.suptitle('PyTorch training: checkpoint and threshold selected on validation data')
fig.savefig(p/'training_curves.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for ax,key,title in zip(axes,['test_event','test_event_strict_iou_0_5'],['Timing overlap IoU ≥ 0.30','Stricter timing: IoU ≥ 0.50']):
    pos=np.arange(3)
    for offset,result,label,color in [(-.18,rf,'Random forest','#d18a43'),(.18,cnn,'PyTorch CNN','#326d91')]:
        values=[result[key][k] for k in ['precision','recall','f1']]
        bars=ax.bar(pos+offset,values,width=.35,label=label,color=color)
        ax.bar_label(bars,labels=[f'{v:.0%}' for v in values],padding=3,fontsize=9)
    ax.set_xticks(pos,['Precision','Recall','F1']);ax.set_ylim(0,1);ax.set_title(title)
axes[0].set_ylabel('Held-out event score');axes[1].legend(fontsize=9)
fig.suptitle('Same 27 pilot test participants | CNN also has longer temporal context')
fig.savefig(p/'rf_vs_cnn.png',dpi=150);plt.close(fig)

meta=json.loads((root/'results/records.json').read_text())
splits=json.loads((root/'results/participant_splits.json').read_text())
r=sorted((r for r in meta if r['participant'] in splits['test']),key=lambda r:r['id'])[0]
signals=root.parent/'dataset_inspection/signals.zip'
if not signals.exists():signals=root/'data/raw/signals.zip'
with zipfile.ZipFile(signals) as z:x=np.load(io.BytesIO(z.read('signals/'+r['id']+'.npy')),allow_pickle=False)
fig,axes=plt.subplots(4,1,figsize=(11,6),sharex=True,gridspec_kw={'height_ratios':[2,1,1,1]},layout='constrained')
axes[0].plot(np.arange(len(x))[::8]/4000,x[::8,0],lw=.6,color='#245b7a');axes[0].set_ylabel('Microphone')
rows=[(r['intervals'],'Annotations','#24875c')]
for path,threshold,label,color in [(root/'results/rf_combined_test_predictions.npz',rf['threshold'],'Random forest','#d18a43'),(p/'test_predictions.npz',cnn['threshold'],'PyTorch CNN','#326d91')]:
    pred=np.load(path);score=pred['scores'][pred['ids']==r['id']]
    rows.append((events(r['times'],score,threshold,r['duration']),label,color))
for ax,(spans,label,color) in zip(axes[1:],rows):
    ax.broken_barh([(a,b-a) for a,b in spans],(.15,.7),facecolors=color)
    ax.set_ylim(0,1);ax.set_yticks([]);ax.set_ylabel(label,rotation=0,ha='right',va='center',labelpad=12)
axes[-1].set_xlabel('Seconds');axes[-1].set_xlim(0,r['duration'])
fig.suptitle('Same example as the baseline: '+r['id']+' | first test recording alphabetically')
fig.savefig(p/'example_rf_vs_cnn.png',dpi=150);plt.close(fig)

lines=['# PyTorch experiment 1', '', ANNOTATION_NOTICE, '',
       'One prespecified CNN run; the existing random forest and pilot test participant split were preserved.', '',
       f"CNN parameters: {config['model_parameters']:,}. Selected epoch: {cnn['best_epoch']} of {len(history)} completed epochs.",
       'The model learns from filtered waveform sequences with about 2.6 seconds of symmetric context. The baseline uses 0.5-second feature windows. This comparison changes architecture and context together.', '',
       '| Metric | Random forest | PyTorch CNN |','| --- | ---: | ---: |']
for key,name in [('precision','Event precision'),('recall','Event recall'),('f1','Event F1')]:
    lines.append(f"| {name}, IoU >= 0.30 | {rf['test_event'][key]:.1%} | {cnn['test_event'][key]:.1%} |")
lines.append(f"| Event F1, IoU >= 0.50 | {rf['test_event_strict_iou_0_5']['f1']:.1%} | {cnn['test_event_strict_iou_0_5']['f1']:.1%} |")
lines.append(f"| Extra events / recording minute | {rf['test_event']['false_events_per_recording_minute']:.2f} | {cnn['test_event']['false_events_per_recording_minute']:.2f} |")
cm=np.array(cnn['test_window']['confusion_matrix_tn_fp_fn_tp'])
lines+=['',f"CNN window accuracy: {np.trace(cm)/cm.sum():.1%}. This is different from event F1 and is not the primary measure.",
        '', '## Interpretation limits', '',
        '- This is swallow timing detection, not normal/abnormal, dysphagia or aspiration classification.',
        '- Labels and unannotated background retain the uncertainties described in the baseline audit.',
        '- The pilot test results had already been inspected for the random forest. CNN checkpoint, normalization and threshold use training/validation data only; this is not a fresh blinded external study.',
        '- One random seed and one participant split are insufficient to claim a generally superior architecture. CNN changes both model and context; isolate these factors in later ablations.',
        '- Both preprocessing and CNN are offline/noncausal. Clinical clips do not establish continuous home-monitoring performance.',
        '- Keep these results fixed. Use grouped cross-validation within development participants for subsequent choices, then obtain external data for stronger validation.', '']
(p/'SUMMARY.md').write_text('\n'.join(lines))
print('\n'.join(lines[:22]))
