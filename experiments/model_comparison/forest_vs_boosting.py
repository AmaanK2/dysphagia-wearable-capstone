"""Nested LOPO: forest alone, boosting alone, their average; with and without the sensor gate.
Run from the repository root:
    python experiments/model_comparison/forest_vs_boosting.py .
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO / 'data_collection'))
import train_mixed  # noqa: E402
from adapt_public import adapted_public  # noqa: E402
from live_detection import IMU_COLUMNS, MIC_COLUMNS, Model, agreement_scores, build_model  # noqa: E402
from train_live import ALL, evaluate_scores, grouped_split, pool_folds, reviewed_records, score_records, wearable_records  # noqa: E402
from train_mixed import BASE, best_settings, fit_mixed, fit_mixed_forest, fit_mixed_trees  # noqa: E402

wearable = list(wearable_records(REPO / 'data_collection' / 'recordings' / 'participants'))
cache = {}


def public_for(parts):
    key = tuple(parts)
    if key not in cache:
        cache[key] = adapted_public(REPO / 'data' / 'raw', [r for r in wearable if r['participant'] in key], verbose=False)[0]
    return cache[key]


def scorer(train, kind, gate):
    """Return a function records -> scored records for one training set."""
    parts = sorted({r['participant'] for r in train})
    public = public_for(parts)
    models = {}
    if kind in ('forest', 'average'):
        models['forest'] = build_model(fit_mixed_forest(train, public, ALL, 0.))
    if kind in ('boost', 'average'):
        models['boost'] = build_model(fit_mixed_trees(train, public, ALL, .35))
    share = 0. if kind == 'forest' else .35
    if gate:
        mic = Model(fit_mixed(train, public, np.array(MIC_COLUMNS), share, **BASE))
        imu = Model(fit_mixed(train, public, np.array(IMU_COLUMNS), share, **BASE))

    def score(records):
        out = []
        for r in records:
            usable = np.isfinite(r['features']).all(axis=1)
            f = r['features'][usable]
            s = np.zeros(len(usable))
            s[usable] = np.mean([m.score(f) for m in models.values()], axis=0)
            if gate:
                m_s, i_s = np.zeros_like(s), np.zeros_like(s)
                m_s[usable], i_s[usable] = mic.score(f), imu.score(f)
                s = agreement_scores(s, m_s, i_s, .4, .4)
            out.append(dict(r, scores=s, usable=usable))
        return out
    return score


def nested(kind, gate):
    folds = {}
    for participant, train, test in grouped_split(wearable):
        test = reviewed_records(test)
        inner = []
        for _, inner_train, inner_test in grouped_split(train):
            inner += scorer(inner_train, kind, gate)(reviewed_records(inner_test))
        chosen = best_settings(inner)
        folds[participant] = evaluate_scores(scorer(train, kind, gate)(test), **chosen)
    return pool_folds(folds)


for kind in ('boost', 'forest', 'average'):
    for gate in (False, True):
        r = nested(kind, gate)
        p = r['pooled']
        per = ' '.join(f"{k}:{v['f1']:.3f}" for k, v in r['by_participant'].items())
        print(f"{kind:8s} gate={str(gate):5s} nested F1 {p['f1']:.3f} P {p['precision']:.3f} R {p['recall']:.3f} "
              f"FP/min {p['extra_per_minute']:.2f} | {per}", flush=True)
