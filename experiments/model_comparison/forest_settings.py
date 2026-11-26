"""Random forest vs gradient boosting on the wearable swallow task, identical protocol.

Same features, same public/wearable weighting, same nested leave-one-participant-out
evaluation (threshold and cooldown chosen on the two training participants only).
No sensor-agreement gate for either model, so only the joint classifier differs.

Run from the repository root:
    python experiments/model_comparison/forest_settings.py .
"""
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO / 'data_collection'))
import train_mixed  # noqa: E402
from adapt_public import adapted_public  # noqa: E402
from train_live import ALL, wearable_records  # noqa: E402
from train_mixed import dataset_rows, nested_evaluation, fit_mixed_trees  # noqa: E402


class Wrapped:
    """Model-like object for train_live.score_records."""
    def __init__(self, classifier, columns):
        self.classifier, self.columns = classifier, np.asarray(columns)

    def score(self, features):
        return self.classifier.predict_proba(np.atleast_2d(features)[:, self.columns])[:, 1]


original_build = train_mixed.build_model
train_mixed.build_model = lambda data: data['_model'] if '_model' in data else original_build(data)


def forest_fitter(**params):
    def fit(wearable, public, columns=ALL, public_share=.5, threshold=.6, refractory=1.2, **_):
        from sklearn.ensemble import RandomForestClassifier
        columns = np.asarray(columns)
        parts = []
        if public_share < 1:
            rows, labels, _, w = dataset_rows(wearable, columns, 'clean_trainable')
            parts.append((rows, labels, (1 - public_share) * w))
        if public_share > 0:
            rows, labels, _, w = dataset_rows(public, columns, 'trainable')
            parts.append((rows, labels, public_share * w))
        X = np.vstack([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
        w = np.concatenate([p[2] for p in parts])
        rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0, **params)
        rf.fit(X, y, sample_weight=w / w.mean())
        return dict(_model=Wrapped(rf, columns), threshold=threshold, refractory=refractory)
    return fit


def main():
    wearable = list(wearable_records(REPO / 'data_collection' / 'recordings' / 'participants'))
    cache = {}

    def public_for(parts):
        key = tuple(parts)
        if key not in cache:
            cache[key] = adapted_public(REPO / 'data' / 'raw', [r for r in wearable if r['participant'] in key],
                                        verbose=False)[0]
        return cache[key]

    configs = [
        ('Gradient boosting (shipped settings)', fit_mixed_trees),
        ('Random forest: baseline settings (depth 14, leaf 5)', forest_fitter(max_depth=14, min_samples_leaf=5, max_features='sqrt')),
        ('Random forest: fully grown, leaf 20', forest_fitter(min_samples_leaf=20, max_features='sqrt')),
        ('Random forest: fully grown, leaf 40', forest_fitter(min_samples_leaf=40, max_features='sqrt')),
        ('Random forest: leaf 20, 50% of features per split', forest_fitter(min_samples_leaf=20, max_features=.5)),
    ]
    for share in (0., .35):
        for name, fitter in configs:
            started = time.perf_counter()
            r = nested_evaluation(wearable, public_for, share, {}, fitter)
            p, o = r['pooled'], r['oracle_fixed_settings']
            per = ' '.join(f"{k}:{v['f1']:.3f}" for k, v in r['by_participant'].items())
            print(f"share {share:.2f} | {name:52s} | nested F1 {p['f1']:.3f} P {p['precision']:.3f} R {p['recall']:.3f} "
                  f"FP/min {p['extra_per_minute']:.2f} | {per} | best-fixed F1 {o['f1']:.3f} "
                  f"({o['settings']['threshold']}) | {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == '__main__':
    main()
