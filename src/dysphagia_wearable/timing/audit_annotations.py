"""Prepare an annotation-relative error review; never invent or change labels."""
from ..paths import PROJECT_ROOT, BASELINE_REPORTS, CNN_REPORTS, CV_REPORTS, LABEL_REVIEW, HRCA_SIGNALS
import csv,io,json,zipfile
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .baseline import events

root=PROJECT_ROOT
out=LABEL_REVIEW;out.mkdir(exist_ok=True)
queue=out/'review_queue.csv'
if queue.exists():
    existing=list(csv.DictReader(queue.open()))
    if any(r.get('review_decision','unknown')!='unknown' or any(r.get(k,'').strip() for k in
           ['reference_type','reference_locator','reviewer','notes']) for r in existing):
        raise FileExistsError('Existing queue contains review work. Preserve/version it before generating a new queue.')
meta={r['id']:r for r in json.loads((BASELINE_REPORTS/'records.json').read_text())}
cv=CV_REPORTS
rows=list(csv.DictReader((cv/'cnn_error_review.csv').open()))
original=json.loads((BASELINE_REPORTS/'participant_splits.json').read_text())
excluded=set(original['test']);review=[];locations=Counter()
predictions={i:np.load(cv/f'fold_{i}/cnn_assessment_predictions.npz') for i in [1,2,3]}
for idx,row in enumerate(rows,1):
    r=meta[row['recording']];assert r['participant'] not in excluded
    a,b=float(row['start_seconds']),float(row['end_seconds'])
    annotations=r['intervals'];location='not_applicable'
    if row['type']=='extra_detection' and row['category']=='no_annotation_overlap':
        if not annotations:location='no_annotations_in_recording'
        elif b<=annotations[0][0]:location='before_first_annotation'
        elif a>=annotations[-1][1]:location='after_last_annotation'
        else:location='between_annotations'
        locations[location]+=1
    d=predictions[int(row['fold'])];score=d['scores'][d['ids']==row['recording']]
    times=np.array(r['times']);mask=(times>=a)&(times<b)
    review.append({'review_id':idx,**row,'unlabelled_region_location':location,
                   'max_model_score_in_interval':float(score[mask].max()) if mask.any() else '',
                   'review_decision':'unknown','reference_type':'','reference_locator':'',
                   'reviewer':'','notes':''})
with (out/'review_queue.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(review[0]));w.writeheader();w.writerows(review)
with (out/'verified_coverage_TEMPLATE.csv').open('w',newline='') as f:
    csv.writer(f).writerow(['recording','start_seconds','end_seconds','reference_type','reference_locator','reviewer'])
summary={'source_paper':'https://pmc.ncbi.nlm.nih.gov/articles/PMC7251089/',
 'dataset_record':'https://zenodo.org/records/4539695',
 'source_finding':'Linked study describes imaging pauses while signals continued, unlabelled swallows, and manual video/log review. This does not identify which released files or intervals are affected.',
 'review_rows':len(review),'all_review_decisions':'unknown','verified_new_labels':0,
 'no_annotation_overlap_locations':dict(locations),'original_pilot_test_participants_in_review':0,
 'metric_status':'Existing numerical results remain unchanged and are relative to supplied annotations, not confirmed clinical truth.'}
(out/'audit_summary.json').write_text(json.dumps(summary,indent=2))
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
names=['no_annotations_in_recording','before_first_annotation','between_annotations','after_last_annotation']
labels=['File has no swallow annotations','Before first annotation','Between annotations','After last annotation']
fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
bars=ax.barh(labels,[locations[n] for n in names],color='#b57b39');ax.invert_yaxis()
ax.bar_label(bars,padding=5);ax.set_xlim(0,260)
ax.set_xlabel('Unmatched CNN detections with no annotation overlap')
ax.set_title('Where the 680 unlabelled-region detections occur\nLocation alone cannot tell us whether a detection is correct',fontsize=13)
fig.savefig(out/'unlabelled_regions.png',dpi=150);plt.close(fig)

# Deterministic review cases, selected by category rather than by favourable model behaviour.
cases=[]
for match in [lambda r:r['unlabelled_region_location']=='no_annotations_in_recording',
              lambda r:r['unlabelled_region_location']=='between_annotations',
              lambda r:r['category']=='overlaps_annotation_boundary_or_duplicate']:
    cases.append(sorted([r for r in review if match(r)],key=lambda r:(r['recording'],float(r['start_seconds'])))[0])
signals=HRCA_SIGNALS
if not signals.exists():signals=HRCA_SIGNALS
fig,axes=plt.subplots(3,2,figsize=(12,8),layout='constrained')
with zipfile.ZipFile(signals) as z:
    for pair,row in zip(axes,cases):
        r=meta[row['recording']]
        x=np.load(io.BytesIO(z.read('signals/'+r['id']+'.npy')),allow_pickle=False)
        d=predictions[int(row['fold'])];score=d['scores'][d['ids']==r['id']]
        pair[0].plot(np.arange(len(x))[::8]/4000,x[::8,0],lw=.65,color='#245b7a')
        pair[0].set_ylabel('Microphone amplitude')
        pair[1].plot(r['times'],score,color='#245b7a')
        pair[1].axhline(float(d['threshold']),ls='--',color='black');pair[1].set_ylim(0,1)
        pair[1].set_ylabel('Uncalibrated model score')
        for ax in pair:
            for a,b in r['intervals']:ax.axvspan(a,b,color='#24875c',alpha=.2)
            ax.axvspan(float(row['start_seconds']),float(row['end_seconds']),color='#bf5b36',alpha=.25)
            ax.set_xlim(0,r['duration']);ax.set_xlabel('Seconds')
        location=row['unlabelled_region_location'].replace('_',' ')
        if location=='not applicable':location='boundary / duplicate candidate'
        pair[0].set_title(r['id']+' | '+location,fontsize=10)
        pair[1].set_title('Highlighted interval requires reference review',fontsize=10)
fig.suptitle('Review examples: green = supplied annotation; orange = interval to review\nWaveform shape and model score do not establish a new ground-truth label',fontsize=13)
fig.savefig(out/'review_examples.png',dpi=150);plt.close(fig)
(out/'selected_review_cases.json').write_text(json.dumps(cases,indent=2))
print(json.dumps(summary,indent=2))
