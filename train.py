"""Reproducible offline annotated-swallow detector. See README.md for limits."""
import argparse
import csv
import hashlib
import io
import json
import platform
import re
import zipfile
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from scipy.signal import butter, sosfiltfilt, resample_poly
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

FS = 4000
FEATURE_FS = 2000
WINDOW = 0.5
HOP = 0.1
SEED = 42
ANNOTATION_NOTICE = ('Interpretation update: the linked study reports unlabelled events and imaging pauses. '
                     'Unannotated time is not verified negative ground truth. All reported scores are relative '
                     'to supplied annotations; see label_review/README.md.')
CHANNELS = ['microphone', 'acc_AP', 'acc_SI', 'acc_ML']
STAT_NAMES = ['log_rms', 'log_std', 'crest', 'skew', 'kurtosis', 'zero_crossings',
              'centroid', 'bandwidth', 'spectral_entropy', 'peak_frequency',
              'power_0_50', 'power_50_200', 'power_200_500', 'power_500_1000']
FEATURE_NAMES = [f'{c}:{s}' for c in CHANNELS for s in STAT_NAMES]


def features(x):
    """Fixed, offline filters; no labels or fitted population statistics used."""
    out = np.empty_like(x)
    for c in range(4):
        low, high = (20, 800) if c == 0 else (5, 500)
        out[:, c] = sosfiltfilt(butter(4, [low, high], fs=FS, btype='bandpass', output='sos'), x[:, c])
    out = resample_poly(out, 1, 2, axis=0)
    width, step = int(WINDOW * FEATURE_FS), int(HOP * FEATURE_FS)
    starts = np.arange(0, len(out) - width + 1, step)
    if not len(starts):
        raise ValueError('Recording shorter than feature window')
    blocks = np.stack([out[i:i + width] for i in starts])
    freq = np.fft.rfftfreq(width, 1 / FEATURE_FS)
    collected = []
    for c in range(4):
        v = blocks[:, :, c]
        rms = np.sqrt(np.mean(v ** 2, axis=1))
        std = v.std(axis=1)
        centered = v - v.mean(axis=1, keepdims=True)
        standardized = centered / np.maximum(std[:, None], 1e-12)
        spectrum = abs(np.fft.rfft(centered * np.hanning(width), axis=1)) ** 2
        fractions = spectrum / np.maximum(spectrum.sum(axis=1, keepdims=True), 1e-24)
        centroid = (fractions * freq).sum(axis=1)
        cols = [np.log10(rms + 1e-12), np.log10(std + 1e-12),
                abs(v).max(axis=1) / np.maximum(rms, 1e-12),
                np.mean(standardized ** 3, axis=1), np.mean(standardized ** 4, axis=1),
                np.mean(np.diff(np.signbit(v), axis=1), axis=1), centroid,
                np.sqrt((fractions * (freq - centroid[:, None]) ** 2).sum(axis=1)),
                -(fractions * np.log(fractions + 1e-24)).sum(axis=1) / np.log(len(freq)),
                freq[spectrum.argmax(axis=1)]]
        for low, high in [(0, 50), (50, 200), (200, 500), (500, 1001)]:
            cols.append(fractions[:, (freq >= low) & (freq < high)].sum(axis=1))
        collected.append(np.column_stack(cols))
    return np.hstack(collected).astype(np.float32), (starts + width / 2) / FEATURE_FS


def events(times, scores, threshold, duration):
    """Cover positive grid points, merge <=0.2s gaps, discard events <0.2s."""
    times = np.asarray(times)[np.asarray(scores) >= threshold]
    spans = []
    for t in times:
        a, b = max(0., t - HOP / 2), min(duration, t + HOP / 2)
        if spans and a - spans[-1][1] <= 0.200001:
            spans[-1][1] = b
        else:
            spans.append([a, b])
    return [s for s in spans if s[1] - s[0] >= 0.199999]


def match_events(pred, truth, iou_threshold=0.3):
    if not pred or not truth:
        return 0, len(pred), len(truth), []
    ious = np.zeros((len(pred), len(truth)))
    for i, (a, b) in enumerate(pred):
        for j, (c, d) in enumerate(truth):
            inter = max(0., min(b, d) - max(a, c))
            ious[i, j] = inter / (b - a + d - c - inter)
    # First maximize number of valid matches, then IoU. Each event matches once.
    valid = ious >= iou_threshold
    ri, ci = linear_sum_assignment(-(valid * (min(ious.shape) + 1) + ious * valid))
    pairs = [(i, j) for i, j in zip(ri, ci) if valid[i, j]]
    errors = [abs(pred[i][0] - truth[j][0]) for i, j in pairs]
    tp = len(pairs)
    return tp, len(pred) - tp, len(truth) - tp, errors


def event_metrics(meta, ids, scores, threshold, iou=0.3):
    tp = fp = fn = 0
    seconds = 0.
    errors, by_participant = [], {}
    for record in meta:
        mask = ids == record['id']
        if not mask.any():
            continue
        pred = events(record['times'], scores[mask], threshold, record['duration'])
        a, b, c, err = match_events(pred, record['intervals'], iou)
        tp += a; fp += b; fn += c; errors += err; seconds += record['duration']
        row = by_participant.setdefault(record['participant'], [0, 0, 0, 0.])
        row[0] += a; row[1] += b; row[2] += c; row[3] += record['duration']
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {'precision': precision, 'recall': recall,
            'f1': 2 * tp / max(2 * tp + fp + fn, 1),
            'tp': tp, 'fp': fp, 'fn': fn,
            'false_events_per_recording_minute': fp / max(seconds / 60, 1e-12),
            'matched_onset_mae_seconds': float(np.mean(errors)) if errors else None,
            'iou_threshold': iou, 'participant_counts': by_participant}


def window_metrics(y, scores, threshold):
    pred = scores >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
    return {'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
            'average_precision': float(average_precision_score(y, scores)),
            'confusion_matrix_tn_fp_fn_tp': confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def choose_threshold(meta, ids, scores, candidates):
    candidates = sorted(set(float(x) for x in candidates))
    evaluated = [(event_metrics(meta, ids, scores, t)['f1'], t) for t in candidates]
    # Prespecified tie break: higher threshold, to reduce detections.
    return max(evaluated)[1]


def bootstrap_interval(counts):
    a = np.array(list(counts.values()))
    rng = np.random.default_rng(SEED)
    metrics = []
    for _ in range(1000):
        tp, fp, fn, seconds = a[rng.integers(0, len(a), len(a))].sum(axis=0)
        metrics.append([tp / max(tp + fp, 1), tp / max(tp + fn, 1),
                        2 * tp / max(2 * tp + fp + fn, 1), fp / (seconds / 60)])
    ci = np.percentile(metrics, [2.5, 97.5], axis=0)
    return {name: ci[:, i].tolist() for i, name in enumerate(['precision', 'recall', 'f1', 'false_events_per_recording_minute'])}


def build_dataset(signal_path, label_path, output):
    matrices, targets, groups, ids, records, excluded, hashes = [], [], [], [], [], [], {}
    empty_label_files = 0
    duplicates = []
    with zipfile.ZipFile(signal_path) as signals, zipfile.ZipFile(label_path) as labels:
        names = sorted(n for n in labels.namelist() if n.endswith('.csv') and not n.startswith('__MACOSX'))
        label_count = 0
        for i, name in enumerate(names):
            stem = Path(name).stem
            rows = list(csv.DictReader(io.StringIO(labels.read(name).decode('utf-8-sig'))))
            label_count += len(rows)
            try:
                participants = {r['participant'].upper() for r in rows}
                expected = re.match(r'(ns\d+|s\d+)', stem, re.I).group(1).upper()
                if not participants:
                    # Header-only files remain in evaluation as unannotated background.
                    empty_label_files += 1
                    participant = expected
                elif len(participants) != 1 or participants != {expected}:
                    raise ValueError('Expected exactly one participant')
                else:
                    participant = next(iter(participants))
                raw = signals.read('signals/' + stem + '.npy')
                x = np.load(io.BytesIO(raw), allow_pickle=False)
                if x.ndim != 2 or x.shape[1] != 4 or not np.isfinite(x).all():
                    raise ValueError('Invalid shape or nonfinite values')
                duration = len(x) / FS
                intervals = sorted([[float(r['start']) / 60, float(r['end']) / 60] for r in rows])
                if any(a < 0 or b <= a or b > duration for a, b in intervals):
                    raise ValueError('Label outside recording bounds')
                if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
                    raise ValueError('Overlapping annotated intervals; requires review')
                digest = hashlib.sha256(raw).hexdigest()
                if digest in hashes:
                    duplicates.append({'id': stem, 'same_as': hashes[digest]})
                    raise ValueError('Exact duplicate signal excluded')
                hashes[digest] = stem
                X, t = features(x)
                y = np.zeros(len(t), dtype=np.uint8)
                for a, b in intervals:
                    y[(t >= a) & (t < b)] = 1
                matrices.append(X); targets.append(y)
                groups.extend([participant] * len(t)); ids.extend([stem] * len(t))
                records.append({'id': stem, 'participant': participant, 'duration': duration,
                                'intervals': intervals, 'times': t.tolist(), 'sha256': digest})
            except (ValueError, KeyError) as e:
                excluded.append({'id': stem, 'reason': str(e)})
            if (i + 1) % 100 == 0:
                print(f'Processed {i + 1}/{len(names)} recordings', flush=True)
    X, y, groups, ids = np.vstack(matrices), np.concatenate(targets), np.array(groups), np.array(ids)
    audit = {'label_files': len(names), 'label_rows': label_count, 'included_recordings': len(records),
             'header_only_label_files': empty_label_files,
             'included_background_recordings': sum(not r['intervals'] for r in records),
             'included_participants': len(set(groups)), 'included_events': sum(len(r['intervals']) for r in records),
             'included_seconds': sum(r['duration'] for r in records), 'windows': len(y),
             'positive_windows': int(y.sum()), 'excluded': excluded, 'duplicates': duplicates,
             'webpage_claim': {'participants': 248, 'swallows': 3144},
             'label_units': 'VFSS frames at 60 fps, converted to seconds by division by 60',
             'channels': CHANNELS}
    (output / 'audit.json').write_text(json.dumps(audit, indent=2))
    np.savez_compressed(output / 'features.npz', X=X, y=y, groups=groups, ids=ids)
    (output / 'records.json').write_text(json.dumps(records))
    return X, y, groups, ids, records


def main():
    print(ANNOTATION_NOTICE, flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--signals', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--reuse-cache', action='store_true', help='Only use when input files and feature code are unchanged')
    args = parser.parse_args()
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    if args.reuse_cache and (out / 'features.npz').exists():
        data = np.load(out / 'features.npz', allow_pickle=False)
        X, y, groups, ids = [data[k] for k in ['X', 'y', 'groups', 'ids']]
        meta = json.loads((out / 'records.json').read_text())
    else:
        X, y, groups, ids, meta = build_dataset(args.signals, args.labels, out)
    trainval, test = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=SEED).split(X, y, groups))
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.15/.85, random_state=SEED + 1).split(X[trainval], y[trainval], groups[trainval]))
    train, val = trainval[tr], trainval[va]
    splits = {name: sorted(set(groups[idx])) for name, idx in [('train', train), ('validation', val), ('test', test)]}
    assert not (set(splits['train']) & set(splits['test']) or set(splits['train']) & set(splits['validation']) or set(splits['validation']) & set(splits['test']))
    (out / 'participant_splits.json').write_text(json.dumps(splits, indent=2))
    summary = {'task': 'Annotated-swallow interval detection, not diagnosis or aspiration prediction',
               'seed': SEED, 'sampling_rate': FS, 'feature_sampling_rate': FEATURE_FS,
               'window_seconds': WINDOW, 'hop_seconds': HOP,
               'split_sizes': {name: {'participants': len(splits[name]), 'windows': len(idx), 'positive_windows': int(y[idx].sum())}
                               for name, idx in [('train', train), ('validation', val), ('test', test)]},
               'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
               'models': {}}
    specs = [('energy_baseline', [0]), ('rf_microphone', list(range(14))),
             ('rf_acceleration', list(range(14, 56))), ('rf_combined', list(range(56)))]
    # Train and select all thresholds on validation only. Test results are not used to choose a model.
    fitted = []
    for name, columns in specs:
        print('Training', name, flush=True)
        if name == 'energy_baseline':
            clf = None
            validation_scores = X[val, 0]
            candidates = np.quantile(X[train, 0], np.linspace(.01, .99, 41))
        else:
            clf = RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=5,
                                         class_weight='balanced_subsample', random_state=SEED, n_jobs=2)
            clf.fit(X[train][:, columns], y[train])
            validation_scores = clf.predict_proba(X[val][:, columns])[:, 1]
            candidates = np.linspace(.05, .95, 37)
        threshold = choose_threshold(meta, ids[val], validation_scores, candidates)
        result = {'threshold': threshold, 'validation_event': event_metrics(meta, ids[val], validation_scores, threshold)}
        fitted.append((name, columns, clf, threshold, result))
    selected = max(fitted, key=lambda t: t[4]['validation_event']['f1'])[0]
    summary['validation_selected_model'] = selected
    for name, columns, clf, threshold, result in fitted:
        scores = X[test, 0] if clf is None else clf.predict_proba(X[test][:, columns])[:, 1]
        result['test_window'] = window_metrics(y[test], scores, threshold)
        result['test_event'] = event_metrics(meta, ids[test], scores, threshold)
        result['test_event_strict_iou_0_5'] = event_metrics(meta, ids[test], scores, threshold, .5)
        result['test_event_bootstrap_95_ci'] = bootstrap_interval(result['test_event']['participant_counts'])
        summary['models'][name] = result
        np.savez_compressed(out / f'{name}_test_predictions.npz', scores=scores, y=y[test], ids=ids[test])
        if clf is not None:
            joblib.dump({'model': clf, 'columns': columns, 'feature_names': FEATURE_NAMES,
                         'threshold': threshold, 'fs': FS, 'window': WINDOW, 'hop': HOP,
                         'channels': CHANNELS, 'task': summary['task']}, out / f'{name}.joblib')
        print(name, 'test event F1', result['test_event']['f1'], flush=True)
    (out / 'metrics.json').write_text(json.dumps(summary, indent=2))
    # Deterministic held-out example: first test recording alphabetically, not best result.
    name, columns, clf, threshold, result = next(t for t in fitted if t[0] == selected)
    record = sorted((r for r in meta if r['participant'] in splits['test']), key=lambda r: r['id'])[0]
    with zipfile.ZipFile(args.signals) as z:
        x = np.load(io.BytesIO(z.read('signals/' + record['id'] + '.npy')), allow_pickle=False)
    fx, t = features(x)
    score = fx[:, 0] if clf is None else clf.predict_proba(fx[:, columns])[:, 1]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, layout='constrained')
    axes[0].plot(np.arange(len(x))[::8] / FS, x[::8, 0], lw=.5, color='#245a81')
    axes[0].set_ylabel('Microphone\n(dataset units)')
    for c in range(1, 4):
        axes[1].plot(np.arange(len(x))[::8] / FS, x[::8, c], lw=.5, label=CHANNELS[c])
    axes[1].legend(loc='upper right', ncol=3, fontsize=8); axes[1].set_ylabel('Acceleration\n(dataset units)')
    axes[2].plot(t, score, color='#245a81', label='Model score')
    axes[2].axhline(threshold, color='black', ls='--', label='Validation-selected threshold')
    for ax in axes:
        for a, b in record['intervals']:
            ax.axvspan(a, b, color='#44aa77', alpha=.2)
    for a, b in events(t, score, threshold, record['duration']):
        axes[2].axvspan(a, b, color='#dd8833', alpha=.25)
    axes[2].set_ylabel('Score'); axes[2].set_xlabel('Seconds'); axes[2].legend(fontsize=8)
    fig.suptitle(f'Held-out participant {record["participant"]}: {record["id"]}\nGreen: annotated swallows | orange: predicted intervals | {selected}')
    fig.savefig(out / 'held_out_example.png', dpi=150); plt.close(fig)
    names = list(summary['models'])
    fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
    pos = np.arange(len(names)); width = .25
    for i, metric in enumerate(['precision', 'recall', 'f1']):
        ax.bar(pos + (i - 1) * width, [summary['models'][n]['test_event'][metric] for n in names], width, label=metric.title())
    ax.set_xticks(pos, ['Energy baseline', 'RF microphone', 'RF acceleration', 'RF combined'])
    ax.set_ylim(0, 1); ax.set_ylabel('Event-level score'); ax.legend(ncol=3)
    ax.set_title('Held-out participant evaluation | one-to-one event matching, IoU ≥ 0.30')
    fig.savefig(out / 'model_comparison.png', dpi=150); plt.close(fig)
    print('Selected on validation:', selected, flush=True)


if __name__ == '__main__':
    main()
