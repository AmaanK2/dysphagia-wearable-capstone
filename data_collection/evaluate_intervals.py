"""How well do live swallow intervals match reviewed swallow labels?

Leave-one-participant-out with the recommended detector recipe (boosted trees, 35% public
weight, mic/IMU agreement gate). For each held-out session the live alert logic and
IntervalTracker run over the window scores; every alert that matches a labelled swallow is
compared with that label: onset error, offset error, duration error and overlap (IoU).

Run from the repository root:
    python data_collection/evaluate_intervals.py
"""
import json
from pathlib import Path

import numpy as np

from live_detection import WINDOW, AgreementModel, IntervalTracker, Trigger
from train_live import grouped_split, reviewed_records, score_records, wearable_records
from train_mixed import SENSOR_GATE, agreement_fit, best_settings, fit_mixed_trees
from adapt_public import adapted_public

HOME = Path(__file__).resolve().parent
REPO = HOME.parent
OUT = REPO / 'experiments' / 'intervals'
SHARE = .35
LOW_FRACTIONS = (.5, .7, .85, 1.)
LATE = .25


def replay(record, threshold, refractory, low_fraction, start_bias=0., end_bias=0.):
    trigger = Trigger(threshold, refractory)
    tracker = IntervalTracker(low_fraction * threshold, start_bias=start_bias, end_bias=end_bias)
    alerts, intervals = [], []
    for t, score in zip(record['times'], record['scores']):
        fired = trigger.update(t, score)
        if fired:
            alerts.append(t)
        intervals += tracker.update(t, score, fired)
    intervals += tracker.flush()
    return alerts, intervals


def compare(records, threshold, refractory, low_fraction, start_bias=0., end_bias=0.):
    rows = []
    for record in records:
        alerts, intervals = replay(record, threshold, refractory, low_fraction, start_bias, end_bias)
        unmatched = list(record['intervals'])
        for t in alerts:
            label = next(((a, b) for a, b in unmatched if a <= t <= b + LATE), None)
            if label is None:
                continue
            unmatched.remove(label)
            interval = next(((s, e) for s, e, _ in intervals if s - WINDOW <= t - WINDOW / 2 <= e + WINDOW), None)
            if interval is None:
                continue
            (a, b), (s, e) = label, interval
            overlap = max(0., min(b, e) - max(a, s))
            rows.append(dict(participant=record['participant'], onset=s - a, offset=e - b,
                             duration=(e - s) - (b - a), iou=overlap / (max(b, e) - min(a, s))))
    return rows


def summary(rows):
    def stats(key):
        v = np.array([r[key] for r in rows])
        return dict(median=round(float(np.median(v)), 3), mean_abs=round(float(np.mean(np.abs(v))), 3),
                    p90_abs=round(float(np.percentile(np.abs(v), 90)), 3))
    iou = np.array([r['iou'] for r in rows])
    return dict(matched=len(rows), onset_s=stats('onset'), offset_s=stats('offset'), duration_s=stats('duration'),
                iou_median=round(float(np.median(iou)), 3), iou_at_least_0_5=round(float(np.mean(iou >= .5)), 3))


def main():
    wearable = list(wearable_records(HOME / 'recordings' / 'participants'))
    public_dir = REPO / 'data' / 'raw'
    scored = {}
    for participant, train, test in grouped_split(wearable):
        test = reviewed_records(test)
        if not test:
            continue
        public = adapted_public(public_dir, train, verbose=False)[0]
        model = AgreementModel(agreement_fit(train, public, SHARE, {}, fit_mixed_trees, SENSOR_GATE))
        scored[participant] = score_records(model, test)
    every = [r for rows in scored.values() for r in rows]
    settings = best_settings(every)
    report = dict(recipe=f'boosted trees, {SHARE:.0%} public, gate {SENSOR_GATE}', alert_settings=settings,
                  low_fraction={})
    for low in LOW_FRACTIONS:
        rows = compare(every, settings['threshold'], settings['refractory'], low)
        report['low_fraction'][str(low)] = summary(rows)
        print(f'low {low}:', report['low_fraction'][str(low)], flush=True)
    # Bias calibration, leave-one-participant-out: biases from the other participants only.
    low = max(LOW_FRACTIONS, key=lambda k: report['low_fraction'][str(k)]['iou_median'])
    calibrated, biases = [], {}
    for participant, rows in scored.items():
        others = [r for p, recs in scored.items() if p != participant for r in recs]
        base = compare(others, settings['threshold'], settings['refractory'], low)
        start_bias = -float(np.median([r['onset'] for r in base]))
        end_bias = -float(np.median([r['offset'] for r in base]))
        biases[participant] = dict(start=round(start_bias, 3), end=round(end_bias, 3))
        calibrated += compare(rows, settings['threshold'], settings['refractory'], low, start_bias, end_bias)
    final = compare(every, settings['threshold'], settings['refractory'], low)
    report.update(chosen_low_fraction=low, lopo_biases=biases, calibrated_lopo=summary(calibrated),
                  deploy_biases=dict(start=round(-float(np.median([r['onset'] for r in final])), 3),
                                     end=round(-float(np.median([r['offset'] for r in final])), 3)),
                  label_duration_s=round(float(np.median([b - a for r in every for a, b in r['intervals']])), 3))
    print('chosen low fraction', low, '| calibrated LOPO:', report['calibrated_lopo'], flush=True)
    print('deploy biases', report['deploy_biases'], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'swallow_intervals.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
