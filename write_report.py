"""Generate a short, factual design-review summary from saved experiment outputs."""
import json
from pathlib import Path
from collections import Counter
from train import ANNOTATION_NOTICE

p = Path(__file__).resolve().parent / 'results'
m = json.loads((p / 'metrics.json').read_text())
a = json.loads((p / 'audit.json').read_text())
selected = m['validation_selected_model']
lines = ['# First swallow-detection experiment', '', ANNOTATION_NOTICE, '',
         '**Status: research baseline, not a dysphagia or aspiration classifier.**', '',
         '## Data actually used', '',
         f"- {a['included_recordings']:,} recordings from {a['included_participants']} participant IDs.",
         f"- {a['included_events']:,} annotated intervals; {a['included_seconds']/60:.1f} recording minutes.",
         f"- {a['included_background_recordings']} recordings retained as unannotated background ({a['header_only_label_files']} header-only files before exclusions).",
         f"- {len(a['excluded'])} exclusions: " + '; '.join(f'{v} {k}' for k, v in Counter(x['reason'] for x in a['excluded']).items()) + '.',
         '- Counts differ from the Zenodo description; audit.json records the archive-derived counts.', '',
         '## Participant-separated evaluation', '']
for name, size in m['split_sizes'].items():
    lines.append(f"- {name.title()}: {size['participants']} participants, {size['windows']:,} windows.")
lines += ['', 'All recordings from a person stay in one split. Thresholds and the selected model were determined on validation data before test evaluation.', '',
          '## Held-out event results', '',
          '| Method | Precision | Recall | F1 | False events / recording minute |',
          '| --- | ---: | ---: | ---: | ---: |']
for name, r in m['models'].items():
    e = r['test_event']
    lines.append(f"| {name} | {e['precision']:.1%} | {e['recall']:.1%} | {e['f1']:.1%} | {e['false_events_per_recording_minute']:.2f} |")
r = m['models'][selected]; e = r['test_event']; ci = r['test_event_bootstrap_95_ci']
lines += ['', f'**Selected on validation: {selected}.**', '',
          f"Its held-out matches: {e['tp']} detected annotations, {e['fn']} missed annotations, {e['fp']} extra detections.",
          f"Participant-bootstrap F1 interval: {ci['f1'][0]:.1%}–{ci['f1'][1]:.1%} (95%; does not capture retraining uncertainty).", '',
          f"With stricter event boundaries (IoU >= 0.50), selected-model F1 falls to {r['test_event_strict_iou_0_5']['f1']:.1%}.", '',
          'An event counts as matched only once and must overlap the annotation with IoU >= 0.30. A stricter IoU >= 0.50 analysis is in metrics.json. Window metrics are separate and must not be presented as event metrics.', '',
          '## Interpretation and next steps', '',
          '- This is an initial feasibility result on short clinical recordings, not validated home-monitoring performance.',
          '- Empty label files and unlabelled time are treated as background; confirm annotation completeness with the authors.',
          '- Review missed swallows and false detections, then use participant-grouped cross-validation for subsequent development.',
          '- Test on your eventual sensors before claiming transfer to the wearable.',
          '- Obtain suitable clinical labels before training normal/abnormal or aspiration-risk classification.', '',
          '## Files', '',
          '- README.md: methods, assumptions, commands and limitations.',
          '- train.py / predict.py: training and offline inference.',
          '- results/*.joblib: trained models; only load trusted model files.',
          '- results/metrics.json, audit.json and participant_splits.json: evidence and reproducibility.',
          '- results/model_comparison.png and held_out_example.png: design-review figures.', '',
          'Dataset source: https://zenodo.org/records/4539695. Original raw recordings are not bundled.', '']
(p.parent / 'START_HERE.md').write_text('\n'.join(lines))
print('\n'.join(lines[:35]))
