"""Train and evaluate the live event classifier (swallow / cough / talking / throat clear / none).

Data:
  * Wearable sessions (recordings/participants): reviewed dry_swallow, cough and talking
    labels; unlabelled windows away from any label or marker are 'none'.
  * Throat-event dataset (Zenodo 12630148, Song et al. 2025): 0.625 s single-axis skin
    vibration clips at 6.4 kHz for cough, speech, swallow and throat clear; the test split
    is a separate group of speakers in five languages. Download with
    ``python download_data.py --throat``.

Throat clips are adapted to the wearable IMU: resampled to its measured rate, then a gain
and an 8-band equaliser are calibrated so that class-matched feature averages (swallow and
talking windows) agree with the wearable's.

Evaluation:
  1. Throat test speakers (never trained on): window accuracy and confusion matrix.
  2. Wearable leave-one-participant-out replay through the live alert logic: per-class
     matched events, confusions and false alerts per minute.

Run from the repository root:
    python data_collection/train_events.py
"""
import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from adapt_public import _filter_centred, band_fir
from events import (BANDS, CLASSES, EVENT_MODEL_PATH, FEATURE_NAMES, HOP, WINDOW, EventIntervals, EventModel, EventTrigger,
                    principal_projection, signal_features, window_event_features)

HOME = Path(__file__).resolve().parent
REPO = HOME.parent
OUT = REPO / 'experiments' / 'event_detection'
LABEL_MAP = {'dry_swallow': 'swallow', 'swallow': 'swallow', 'cough': 'cough', 'coughing': 'cough',
             'talking': 'talking', 'speech': 'talking', 'throat_clear': 'throat_clear'}
THROAT_MAP = {'cough': 'cough', 'speak': 'talking', 'swallow': 'swallow', 'throatclear': 'throat_clear'}
THROAT_RATE = 6400
THRESHOLDS = dict(cough=.5, throat_clear=.5, talking=.5)
THRESHOLD_GRID = (.5, .6, .7, .8)
TREES = dict(learning_rate=.08, max_leaf_nodes=15, max_iter=250, l2_regularization=1., min_samples_leaf=20)
MATCH_LATE = .75   # an alert may follow the labelled interval's end by up to this many seconds


# ----------------------------------------------------------------------------- wearable
def wearable_sessions(root):
    for metadata in sorted(root.glob('*/*/Session.json')):
        folder = metadata.parent
        info = json.loads(metadata.read_text())
        if info.get('simulated') or info.get('status') not in {'complete', 'stopped'}:
            continue
        with (folder / 'Events.csv').open(newline='', encoding='utf-8') as file:
            rows = list(csv.DictReader(file))
        labels = [(float(r['start_s']), float(r['end_s']), LABEL_MAP.get(r['activity']))
                  for r in rows if r['record_type'] == 'label' and r['start_s'] and r['end_s']
                  and not (r.get('notes') or '').startswith('[uncertain]')]
        markers = [float(r['time_s']) for r in rows if r['record_type'] == 'marker' and r['time_s']]
        if not labels:
            continue
        imu = pd.read_csv(folder / 'IMU.csv', usecols=['time_s', 'ax', 'ay', 'az']).to_numpy(float)
        yield dict(participant=folder.parent.name, session=str(folder.relative_to(root)), imu=imu,
                   labels=labels, markers=markers, duration=float(imu[-1, 0]))


def wearable_windows(session):
    """(times, features, class index or -1 for ambiguous) for every 0.1 s window."""
    times = np.arange(round(WINDOW / HOP), int(session['duration'] / HOP) + 1) * HOP
    feats, targets, keep = [], [], []
    for end in times:
        row = window_event_features(session['imu'], end)
        if row is None:
            continue
        centre = end - WINDOW / 2
        target = -1
        inside = [c for a, b, c in session['labels'] if a <= centre <= b]
        if inside:
            target = CLASSES.index(inside[0]) if inside[0] else -1
        elif (not any(end > a - .5 and end - WINDOW < b + .5 for a, b, _ in session['labels'])
              and not any(abs(centre - m) < 1. for m in session['markers'])):
            target = CLASSES.index('none')
        feats.append(row)
        targets.append(target)
        keep.append(end)
    return np.array(keep), np.array(feats), np.array(targets)


# ----------------------------------------------------------------------------- throat dataset
def load_throat(path):
    """{'train': [(class, clip)], 'test': [(class, language, clip)]} from the two archives."""
    out = {'train': [], 'test': []}
    for split, name in (('train', 'train.zip'), ('test', 'test.zip')):
        with zipfile.ZipFile(path / name) as archive:
            for member in archive.namelist():
                if not member.endswith('.npy'):
                    continue
                parts = Path(member).parts
                label = THROAT_MAP[Path(member).stem]
                clips = np.load(io.BytesIO(archive.read(member)), allow_pickle=False)
                language = parts[-2] if split == 'test' else 'English'
                out[split] += [(label, language, clip) for clip in clips]
    return out


def adapt_clip(clip, plan, offsets=(0.,)):
    """Resample, equalise and scale one throat clip; return windows centred at ``offsets``."""
    up, down = int(round(plan['imu_rate'] * 2)), 2 * THROAT_RATE
    real_rate = up / down * THROAT_RATE
    x = resample_poly(clip - clip.mean(), up, down)
    x = _filter_centred(x[:, None], band_fir(BANDS, plan['band_db'], real_rate, 255))[:, 0] * plan['gain']
    half = int(round(WINDOW * real_rate / 2))
    centre = len(x) // 2
    windows = []
    for offset in offsets:
        c = centre + int(round(offset * real_rate))
        if c - half >= 0 and c + half <= len(x):
            windows.append(signal_features(x[c - half:c + half], real_rate))
    return windows


def calibrate(sessions, throat_train, imu_rate, iterations=4, sample=400, seed=0):
    """Gain and band equaliser so throat swallow/talking windows match wearable ones."""
    rng = np.random.default_rng(seed)
    w_feats, w_targets = [], []
    for s in sessions:
        _, f, t = wearable_windows(s)
        w_feats.append(f)
        w_targets.append(t)
    w_feats, w_targets = np.vstack(w_feats), np.concatenate(w_targets)
    shared = [c for c in ('swallow', 'talking') if (w_targets == CLASSES.index(c)).sum() >= 20]
    by_class = {c: [clip for label, _, clip in throat_train if label == c] for c in shared}
    picks = {c: [by_class[c][i] for i in rng.choice(len(by_class[c]), min(sample, len(by_class[c])), replace=False)]
             for c in shared}
    plan = dict(imu_rate=imu_rate, gain=.02, band_db=np.zeros(len(BANDS) - 1))
    shares = slice(1, len(BANDS))
    for _ in range(iterations):
        delta, ratios = np.zeros(len(BANDS) - 1), []
        for c in shared:
            w = w_feats[w_targets == CLASSES.index(c)]
            p = np.array([f for clip in picks[c] for f in adapt_clip(clip, plan)])
            delta += (np.log(w[:, shares] + 1e-6).mean(axis=0) - np.log(p[:, shares] + 1e-6).mean(axis=0)) / len(shared)
            ratios.append(np.exp(np.median(w[:, 0]) - np.median(p[:, 0])))
        delta -= delta.mean()
        plan['band_db'] = np.clip(plan['band_db'] + .8 * 10 / np.log(10) * delta, -12, 12)
        plan['gain'] *= float(np.exp(np.mean(np.log(ratios))))
    plan['band_db'] = plan['band_db'].tolist()
    plan['calibrated_on'] = shared
    return plan


def throat_windows(clips, plan, offsets):
    feats, targets = [], []
    for label, _, clip in clips:
        for f in adapt_clip(clip, plan, offsets):
            feats.append(f)
            targets.append(CLASSES.index(label))
    return np.array(feats), np.array(targets)


# ----------------------------------------------------------------------------- model
def class_weights(parts):
    """Every class gets equal total weight, split equally between the datasets that have it."""
    weights = []
    for _, targets in parts:
        weights.append(np.zeros(len(targets)))
    for k in range(len(CLASSES)):
        owners = [i for i, (_, t) in enumerate(parts) if (t == k).any()]
        for i in owners:
            mask = parts[i][1] == k
            weights[i][mask] = 1 / len(CLASSES) / len(owners) / mask.sum()
    return weights


def fit(parts, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = np.vstack([f for f, _ in parts])
    y = np.concatenate([t for _, t in parts])
    w = np.concatenate(class_weights(parts))
    classifier = HistGradientBoostingClassifier(**TREES, random_state=seed, early_stopping=False)
    classifier.fit(X, y, sample_weight=w / w.mean())
    if list(classifier.classes_) != list(range(len(CLASSES))):
        raise ValueError('Every class needs training windows.')
    trees = []
    for k in range(len(CLASSES)):
        table = dict(feature=[], split=[], left=[], right=[], leaf=[], value=[], roots=[], depth=0)
        for predictors in classifier._predictors:
            nodes = predictors[k].nodes
            offset = len(table['feature'])
            table['roots'].append(offset)
            table['feature'] += nodes['feature_idx'].astype(int).tolist()
            table['split'] += nodes['num_threshold'].astype(float).tolist()
            table['left'] += (nodes['left'].astype(int) + offset).tolist()
            table['right'] += (nodes['right'].astype(int) + offset).tolist()
            table['leaf'] += nodes['is_leaf'].astype(bool).tolist()
            table['value'] += nodes['value'].astype(float).tolist()
            table['depth'] = max(table['depth'], int(nodes['depth'].max()))
        trees.append(table)
    return dict(classes=list(CLASSES), features=list(FEATURE_NAMES), thresholds=THRESHOLDS, trees=trees,
                baseline=np.ravel(classifier._baseline_prediction).tolist(), tree_settings=TREES)


# ----------------------------------------------------------------------------- evaluation
def confusion(model, feats, targets):
    predicted = model.probabilities(feats).argmax(axis=1)
    matrix = np.zeros((len(CLASSES), len(CLASSES)), int)
    for t, p in zip(targets, predicted):
        matrix[t, p] += 1
    present = [k for k in range(len(CLASSES)) if matrix[k].sum()]
    recall = {CLASSES[k]: round(matrix[k, k] / matrix[k].sum(), 4) for k in present}
    return dict(accuracy=round(float((predicted == targets).mean()), 4), recall=recall,
                balanced_accuracy=round(float(np.mean(list(recall.values()))), 4),
                matrix=matrix.tolist(), windows=int(len(targets)))


def replay(model, session, thresholds=None):
    """Run the live alert logic over a session; match alerts and intervals to labels."""
    times, feats, _ = session.get('_windows') or wearable_windows(session)
    thresholds = thresholds or model.thresholds
    trigger = EventTrigger(thresholds)
    tracker = EventIntervals(thresholds, trigger.talk_gap)
    probs = model.probabilities(feats) if len(feats) else np.empty((0, len(CLASSES)))
    alerts, intervals = [], []
    for t, p in zip(times, probs):
        fired = trigger.update(t, p)
        alerts += [(t, name) for name, _ in fired]
        intervals += tracker.update(t, p, {name for name, _ in fired})
    intervals += tracker.flush()
    labels = [(a, b, c) for a, b, c in session['labels'] if c in ('cough', 'talking', 'throat_clear')]
    found = {c: dict(labelled=0, matched=0, alerted_as={}, iou=[]) for c in ('cough', 'talking', 'throat_clear')}
    for a, b, c in labels:
        found[c]['labelled'] += 1
        hits = [name for t, name in alerts if a <= t <= b + MATCH_LATE]
        if c in hits:
            found[c]['matched'] += 1
            overlaps = [max(0., min(b, e) - max(a, s)) / (max(b, e) - min(a, s))
                        for name, s, e, _ in intervals if name == c and s <= b + MATCH_LATE and e >= a]
            found[c]['iou'].append(round(max(overlaps, default=0.), 3))
        for name in set(hits) - {c}:
            found[c]['alerted_as'][name] = found[c]['alerted_as'].get(name, 0) + 1
    spurious = {}
    during_swallow = 0
    for t, name in alerts:
        if any(a <= t <= b + MATCH_LATE for a, b, c in session['labels'] if c != 'swallow'):
            continue
        if any(a <= t <= b + MATCH_LATE for a, b, c in session['labels'] if c == 'swallow'):
            during_swallow += 1
        spurious[name] = spurious.get(name, 0) + 1
    return dict(found=found, false_alerts=spurious, false_alerts_during_swallows=during_swallow,
                minutes=session['duration'] / 60)


def pool(rows):
    total = dict(found={c: dict(labelled=0, matched=0, alerted_as={}, iou=[]) for c in ('cough', 'talking', 'throat_clear')},
                 false_alerts={}, false_alerts_during_swallows=0, minutes=0.)
    for r in rows:
        for c, v in r['found'].items():
            total['found'][c]['labelled'] += v['labelled']
            total['found'][c]['matched'] += v['matched']
            total['found'][c]['iou'] += v.get('iou', [])
            for k, n in v['alerted_as'].items():
                total['found'][c]['alerted_as'][k] = total['found'][c]['alerted_as'].get(k, 0) + n
        for k, n in r['false_alerts'].items():
            total['false_alerts'][k] = total['false_alerts'].get(k, 0) + n
        total['false_alerts_during_swallows'] += r['false_alerts_during_swallows']
        total['minutes'] += r['minutes']
    total['false_alerts_per_minute'] = {k: round(n / total['minutes'], 3) for k, n in total['false_alerts'].items()}
    total['minutes'] = round(total['minutes'], 2)
    return total


def imu_rate(sessions):
    return float(np.median([(len(s['imu']) - 1) / (s['imu'][-1, 0] - s['imu'][0, 0]) for s in sessions]))


def build(sessions, throat, offsets=(-.05, 0., .05)):
    plan = calibrate(sessions, throat['train'], imu_rate(sessions))
    t_feats, t_targets = throat_windows(throat['train'], plan, offsets)
    parts = [(t_feats, t_targets)]
    w = [wearable_windows(s)[1:] for s in sessions]
    w_feats = np.vstack([f for f, _ in w])
    w_targets = np.concatenate([t for _, t in w])
    keep = w_targets >= 0
    parts.append((w_feats[keep], w_targets[keep]))
    return plan, parts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--recordings', type=Path, default=HOME / 'recordings' / 'participants')
    parser.add_argument('--throat-data', type=Path, default=REPO / 'data' / 'throat')
    args = parser.parse_args()
    sessions = list(wearable_sessions(args.recordings))
    throat = load_throat(args.throat_data)
    print(f'{len(sessions)} wearable sessions; throat train {len(throat["train"])} clips, '
          f'test {len(throat["test"])} clips', flush=True)
    report = dict(classes=list(CLASSES), thresholds=THRESHOLDS, tree_settings=TREES,
                  wearable_labels={c: sum(1 for s in sessions for *_, k in s['labels'] if k == c)
                                   for c in CLASSES[1:]})

    # 1. Held-out throat speakers, with the adaptation calibrated on all wearable sessions.
    plan, parts = build(sessions, throat)
    model = EventModel(fit(parts))
    test_feats, test_targets = throat_windows(throat['test'], plan, (0.,))
    report['throat_test_speakers'] = confusion(model, test_feats, test_targets)
    by_language = {}
    for language in sorted({lang for _, lang, _ in throat['test']}):
        f, t = throat_windows([c for c in throat['test'] if c[1] == language], plan, (0.,))
        by_language[language] = confusion(model, f, t)['accuracy']
    report['throat_test_by_language'] = by_language
    print('Throat test speakers:', {k: report['throat_test_speakers'][k] for k in ('accuracy', 'balanced_accuracy', 'recall')}, flush=True)

    # 2. Wearable leave-one-participant-out live replay, with and without wearable training data.
    participants = sorted({s['participant'] for s in sessions})
    for variant in ('throat_plus_wearable', 'throat_only'):
        rows = {}
        for held in participants:
            train = [s for s in sessions if s['participant'] != held]
            test = [s for s in sessions if s['participant'] == held]
            fold_plan, fold_parts = build(train, throat)
            if variant == 'throat_only':
                fold_parts = fold_parts[:1] + [(fold_parts[1][0][fold_parts[1][1] == 0],
                                                fold_parts[1][1][fold_parts[1][1] == 0])]
            fold_model = EventModel(fit(fold_parts))
            for s in test:
                s['_windows'] = wearable_windows(s)
            rows[held] = pool([replay(fold_model, s) for s in test])
            rows[held]['threshold_sweep'] = {
                f'{level:.1f}': pool([replay(fold_model, s, dict(cough=level, throat_clear=level, talking=level))
                                      for s in test]) for level in THRESHOLD_GRID}
            _, f, t = zip(*[wearable_windows(s) for s in test])
            f, t = np.vstack(f), np.concatenate(t)
            rows[held]['windows'] = confusion(fold_model, f[t >= 0], t[t >= 0])
        pooled = pool(list(rows.values()))
        sweep = {level: pool([r['threshold_sweep'][level] for r in rows.values()]) for level in rows[held]['threshold_sweep']}
        for r in rows.values():
            r.pop('threshold_sweep')
        report[f'wearable_lopo_{variant}'] = dict(pooled=pooled, by_participant=rows, threshold_sweep=sweep)
        for level, row in sweep.items():
            print(f'  threshold {level}:', {c: f"{v['matched']}/{v['labelled']}" for c, v in row['found'].items() if v['labelled']},
                  'false alerts/min', row['false_alerts_per_minute'], flush=True)
        print(f'Wearable LOPO ({variant}):', {c: f"{v['matched']}/{v['labelled']}" for c, v in pooled['found'].items()},
              'false alerts/min', pooled['false_alerts_per_minute'],
              'median interval IoU', {c: round(float(np.median(v['iou'])), 3) for c, v in pooled['found'].items() if v['iou']},
              flush=True)

    # Final model: all wearable sessions + throat training speakers.
    data = fit(parts)
    data['adaptation'] = plan
    data['training'] = dict(wearable_sessions=[s['session'] for s in sessions],
                            throat_train_clips=len(throat['train']))
    EventModel(data)
    EVENT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVENT_MODEL_PATH.write_text(json.dumps(data) + '\n')
    report['adaptation'] = plan
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved {EVENT_MODEL_PATH.relative_to(REPO)} and {(OUT / "evaluation.json").relative_to(REPO)}')


if __name__ == '__main__':
    main()
