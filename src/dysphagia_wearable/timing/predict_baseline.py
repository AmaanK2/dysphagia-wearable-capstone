"""Offline prediction on a compatible 4 kHz N x 4 NumPy recording."""
from ..paths import PROJECT_ROOT, BASELINE_REPORTS, CNN_REPORTS, CV_REPORTS, LABEL_REVIEW, HRCA_SIGNALS
import argparse
import csv
import json
from pathlib import Path
import joblib
import numpy as np
from .baseline import features, events, FS

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--recording', type=Path, required=True)
parser.add_argument('--output', type=Path, default=Path('prediction'))
args = parser.parse_args()
# Only load model bundles you trust: joblib uses pickle internally.
bundle = joblib.load(args.model)
x = np.load(args.recording, allow_pickle=False)
if x.ndim != 2 or x.shape[1] != 4 or not np.isfinite(x).all():
    raise ValueError('Expected finite N x 4 array: microphone, AP, SI, ML, sampled at 4 kHz')
if bundle['fs'] != FS:
    raise ValueError('Model sampling rate does not match feature extraction')
X, times = features(x)
scores = bundle['model'].predict_proba(X[:, bundle['columns']])[:, 1]
intervals = events(times, scores, bundle['threshold'], len(x) / FS)
args.output.mkdir(parents=True, exist_ok=True)
with (args.output / 'scores.csv').open('w', newline='') as f:
    w = csv.writer(f); w.writerow(['center_seconds', 'uncalibrated_model_score'])
    w.writerows(zip(times, scores))
(args.output / 'intervals.json').write_text(json.dumps({
    'task': bundle['task'], 'threshold': bundle['threshold'],
    'intervals_seconds': intervals,
    'warning': 'Research prototype; no dysphagia or aspiration diagnosis. Score is not calibrated clinical probability.'
}, indent=2))
print('Predicted intervals:', len(intervals))
