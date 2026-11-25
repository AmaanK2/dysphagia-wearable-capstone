"""Train and evaluate the causal wearable detector on reviewed Events.csv labels."""
import argparse
import csv
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
from time import perf_counter
import zipfile

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from live_detection import (FEATURE_COUNT, FEATURE_VERSION, HOP, IMU_COLUMNS, MIC_COLUMNS, MODEL_PATH, REFRACTORY, THRESHOLD, WINDOW,
                            AgreementModel, Model, Trigger, agreement_scores, quadratic_terms, window_features)

HOME = Path(__file__).resolve().parent
ALL = np.arange(FEATURE_COUNT)
BASE_FEATURES = np.arange(22)
SHAPE = np.delete(BASE_FEATURES, [10, 21])
LABELS = {'dry_swallow', 'swallow'}
BACKGROUND_LABELS = {'talking', 'speech', 'cough', 'coughing', 'throat_clear', 'movement', 'quiet', 'background'}
LATE_TOLERANCE = .25
THRESHOLDS = (.50, .55, .60, .65, .70, .75)
COOLDOWNS = (.8, 1.2, 1.6)
PREVIOUS_SETTINGS = dict(threshold=.60, refractory=1.2, quadratic=True,
                         background_weight=4., reviewed_weight=4.)
# Change one factor at a time from the previous model, then test a combined variant.
TUNING_VARIANTS = (
    {},
    dict(quadratic=False, penalty=.02),
    dict(penalty=.015),
    dict(penalty=.10),
    dict(background_weight=2.),
    dict(reviewed_weight=8.),
    dict(motion_weight=4.),
    dict(motion_weight=4., penalty=.10),
)
DEPLOYED_REFRACTORY = 1.2


def make_record(mic, imu, intervals, uncertain=(), *, wearable=True, reviewed_background=(), ambiguous=()):
    intervals = [tuple(bounds) for bounds in intervals]
    duration = min(mic[-1, 0], imu[-1, 0])
    times = np.arange(round(WINDOW / HOP), int(duration / HOP) + 1) * HOP
    features = np.full((len(times), len(ALL)), np.nan)
    for i, end in enumerate(times):
        row = window_features(mic, imu, end, wearable=wearable)
        if row is not None:
            features[i] = row
    centers = times - WINDOW / 2
    labels = np.zeros(len(times), dtype=bool)
    for start, end in intervals:
        labels |= (centers >= start) & (centers <= end)
    trainable = np.isfinite(features).all(axis=1)
    background = np.zeros(len(times), dtype=bool)
    overlaps_swallow = np.zeros(len(times), dtype=bool)
    for start, end in intervals:
        overlaps_swallow |= (times > start) & (times - WINDOW < end)
    for start, end in reviewed_background:
        background |= (times >= start + WINDOW) & (times <= end)
    clean_trainable = trainable & (labels | ~overlaps_swallow)
    for start, end in ambiguous:
        clean_trainable &= labels | ~((times > start) & (times - WINDOW < end))
    for marker in uncertain:
        trainable &= labels | (np.abs(centers - marker) > 1.)
        clean_trainable &= labels | background | (np.abs(centers - marker) > 1.)
    return dict(times=times, features=features, labels=labels, trainable=trainable,
                intervals=intervals, duration=duration, reviewed_background=background,
                clean_trainable=clean_trainable)


def wearable_records(root):
    for metadata in sorted(root.glob('*/*/Session.json')):
        folder = metadata.parent
        info = json.loads(metadata.read_text())
        if info.get('simulated') or info.get('status') not in {'complete', 'stopped'}:
            continue
        with (folder / 'Events.csv').open(newline='', encoding='utf-8') as file:
            rows = list(csv.DictReader(file))
        intervals = [(float(row['start_s']), float(row['end_s']), row['activity'],
                      (row.get('notes') or '').startswith('[uncertain]')) for row in rows
                     if row['record_type'] == 'label' and row['start_s'] and row['end_s']]
        reviewed = [(start, end, activity) for start, end, activity, uncertain in intervals if not uncertain]
        markers = [float(row['time_s']) for row in rows if row['record_type'] == 'marker' and row['time_s']]
        if not intervals and not markers:
            continue
        uncertain = [t for t in markers if not any(a - .5 <= t <= b + .5 for a, b, _ in reviewed)]
        mic_path = folder / 'Mic.csv'
        if not mic_path.is_file():
            mic_path = folder / 'Mic.csv.gz'
        print(f'Extracting {folder.relative_to(root)}', flush=True)
        mic = np.loadtxt(mic_path, delimiter=',', skiprows=1, usecols=(1, 2), ndmin=2)
        imu = np.loadtxt(folder / 'IMU.csv', delimiter=',', skiprows=1, usecols=(1, 2, 3, 4), ndmin=2)
        swallows = sorted((a, b) for a, b, label in reviewed if label in LABELS)
        confounders = [(a, b, label) for a, b, label in reviewed if label in BACKGROUND_LABELS]
        ambiguous = [(a, b) for a, b, label, flagged in intervals
                     if flagged or label not in LABELS | BACKGROUND_LABELS]
        record = make_record(mic, imu, swallows, uncertain,
                             reviewed_background=[(a, b) for a, b, _ in confounders], ambiguous=ambiguous)
        yield dict(record, participant=folder.parent.name, annotations=[label for _, _, label in reviewed],
                   confounders=confounders, session=str(folder.relative_to(root)))


def training_rows(records, columns, mask='trainable'):
    rows = np.concatenate([r['features'][r[mask]] for r in records])[:, columns]
    labels = np.concatenate([r['labels'][r[mask]] for r in records])
    counts = np.bincount(labels.astype(int), minlength=2)
    if np.any(counts == 0):
        raise ValueError('Training needs both reviewed swallow and background windows.')
    return rows, labels, .5 / counts[labels.astype(int)]


def fit(records, columns, supplemental=(), *, quadratic=False, threshold=THRESHOLD,
        refractory=REFRACTORY, background_weight=0., reviewed_weight=0., motion_weight=0., penalty=None):
    reweight = background_weight or reviewed_weight or motion_weight
    mask = 'clean_trainable' if reweight else 'trainable'
    rows, labels, weights = training_rows(records, columns, mask)
    mean = rows.mean(axis=0)
    scale = np.maximum(rows.std(axis=0), 1e-8)
    if supplemental:
        extra_rows, extra_labels, extra_weights = training_rows(supplemental, columns)
        rows = np.vstack((rows, extra_rows))
        labels = np.r_[labels, extra_labels]
        weights = .5 * np.r_[weights, extra_weights]
    z = (rows - mean) / scale
    regularization = (.04 if quadratic else .01) if penalty is None else penalty
    if quadratic:
        z = quadratic_terms(z)

    def loss(params):
        coeff, bias = params[:-1], params[-1]
        logits = z @ coeff + bias
        diff = weights * (expit(logits) - labels)
        return (weights @ (np.logaddexp(0, logits) - labels * logits) + regularization * (coeff @ coeff),
                np.r_[z.T @ diff + 2 * regularization * coeff, diff.sum()])

    def optimize(initial):
        result = minimize(loss, initial, method='L-BFGS-B', jac=True)
        if not result.success:
            raise RuntimeError(f'Detector training failed ({result.message}).')
        return result.x

    coeff = optimize(np.zeros(z.shape[1] + 1))
    if reweight:
        negative = ~labels
        reviewed = np.concatenate([r['reviewed_background'][r[mask]] for r in records])
        if supplemental:
            reviewed = np.r_[reviewed, np.zeros(len(extra_labels), dtype=bool)]
        scores = expit(z[negative] @ coeff[:-1] + coeff[-1])
        confusing = scores >= np.quantile(scores, .8)
        # Keep every background window while giving confusing and reviewed
        # examples a greater share of the background loss.
        factors = 1 + background_weight * confusing + reviewed_weight * reviewed[negative]
        if motion_weight:
            # Strong acceleration relative to audio is a proxy for motion
            # artifacts, independent of the model's initial confusion score.
            raw = rows[negative]
            mic_index = np.flatnonzero(columns == 10)[0]
            imu_index = np.flatnonzero(columns == 21)[0]
            mic = (raw[:, mic_index] - mean[mic_index]) / scale[mic_index]
            imu = (raw[:, imu_index] - mean[imu_index]) / scale[imu_index]
            motion = (imu >= np.median(imu)) & ((imu - mic) >= np.quantile(imu - mic, .8))
            factors += motion_weight * motion
        weights[negative] *= factors / factors.mean()
        coeff = optimize(coeff)
    return dict(feature_version=FEATURE_VERSION, columns=columns.tolist(), mean=mean.tolist(),
                scale=scale.tolist(), weights=coeff[:-1].tolist(), bias=float(coeff[-1]),
                quadratic=quadratic, threshold=threshold, refractory=refractory)


def score_records(model, records):
    scored = []
    for record in records:
        features = record['features']
        usable = np.isfinite(features).all(axis=1)
        scores = np.zeros(len(features))
        scores[usable] = model.score(features[usable])
        scored.append(dict(record, scores=scores, usable=usable))
    return scored


def evaluate(model, records):
    return evaluate_scores(score_records(model, records), model.threshold, model.refractory)


def evaluate_scores(records, threshold, refractory=REFRACTORY):
    tp = fp = fn = during = correct = positive_correct = windows = positives = 0
    seconds = gated_seconds = 0.
    delays = []
    confounder_alerts = Counter()
    for record in records:
        scores, usable = record['scores'], record['usable']
        predicted = scores >= threshold
        truth = record['labels']
        correct += int((predicted == truth).sum())
        positive_correct += int((predicted & truth).sum())
        windows += len(truth)
        positives += int(truth.sum())
        seconds += record['duration']
        gated_seconds += (~usable).sum() * HOP
        trigger = Trigger(threshold, refractory)
        unmatched = list(record['intervals'])
        for t, score in zip(record['times'], scores):
            if not trigger.update(t, score):
                continue
            match = next(((a, b) for a, b in unmatched if a <= t <= b + LATE_TOLERANCE), None)
            if match:
                unmatched.remove(match)
                delays.append(round(float(t - match[0]), 6))
                tp += 1
                during += int(t <= match[1])
            else:
                fp += 1
                confounder_alerts.update({label for a, b, label in record.get('confounders', []) if a <= t <= b})
        fn += len(unmatched)
    return dict(summarize(tp, fp, fn, during, correct, positive_correct, windows, positives, seconds,
                          gated_seconds, delays), confounder_alerts=dict(confounder_alerts))


def summarize(tp, fp, fn, during, correct, positive_correct, windows, positives, seconds, gated_seconds, delays):
    return dict(tp=tp, fp=fp, fn=fn, matched_during_interval=during,
                correct_windows=correct, positive_correct=positive_correct,
                windows=windows, positive_windows=positives, seconds=round(seconds, 3),
                gated_seconds=round(gated_seconds, 3), delays_s=delays,
                precision=tp / max(1, tp + fp), recall=tp / max(1, tp + fn),
                f1=2 * tp / max(1, 2 * tp + fp + fn), extra_per_minute=60 * fp / seconds,
                median_detection_delay_s=float(np.median(delays)) if delays else None,
                window_accuracy=correct / windows,
                background_window_accuracy=1 - positives / windows,
                window_balanced_accuracy=.5 * (positive_correct / max(1, positives) +
                                               (correct - positive_correct) / max(1, windows - positives)))


def grouped_split(records):
    for participant in sorted({r['participant'] for r in records}):
        yield (participant, [r for r in records if r['participant'] != participant],
               [r for r in records if r['participant'] == participant])


def pool_folds(folds):
    totals = [sum(row[key] for row in folds.values()) for key in
              ('tp', 'fp', 'fn', 'matched_during_interval', 'correct_windows', 'positive_correct', 'windows',
               'positive_windows', 'seconds', 'gated_seconds')]
    delays = [t for row in folds.values() for t in row['delays_s']]
    pooled = summarize(*totals, delays)
    pooled['confounder_alerts'] = dict(sum((Counter(row['confounder_alerts']) for row in folds.values()), Counter()))
    return dict(pooled=pooled, by_participant=folds)


def reviewed_records(records):
    return [r for r in records if r.get('annotations', True)]


def select_detector(records):
    splits = [(train, reviewed_records(test)) for _, train, test in grouped_split(records)
              if any(r['intervals'] for r in train) and reviewed_records(test)]
    if len(splits) < 2:
        return PREVIOUS_SETTINGS.copy()
    best_key = None
    for overrides in TUNING_VARIANTS:
        settings = dict(PREVIOUS_SETTINGS, **overrides)
        scored = []
        for train, test in splits:
            scored.extend(score_records(Model(fit(train, ALL, **settings)), test))
        if not overrides:
            reference = evaluate_scores(scored, PREVIOUS_SETTINGS['threshold'],
                                        PREVIOUS_SETTINGS['refractory'])
            recall_floor = reference['recall'] - .05
        for threshold in THRESHOLDS:
            for cooldown in COOLDOWNS:
                row = evaluate_scores(scored, threshold, cooldown)
                key = (row['recall'] >= recall_floor, row['f1'], row['tp'], -row['fp'])
                if best_key is None or key > best_key:
                    best_key = key
                    best = dict(settings, threshold=threshold, refractory=cooldown)
    return best


def grouped_evaluation(records, columns=ALL, supplemental=(), *, tune=False,
                       exclude_training=(), test_participants=None, **fixed_settings):
    folds = {}
    for participant, train, test in grouped_split(records):
        if test_participants is not None and participant not in test_participants:
            continue
        train = [r for r in train if r['participant'] not in exclude_training]
        test = reviewed_records(test)
        if not test:
            continue
        settings = select_detector(train) if tune else fixed_settings
        if not any(settings.get(key) for key in ('background_weight', 'reviewed_weight', 'motion_weight')):
            train = reviewed_records(train)
        model = Model(fit(train, columns, supplemental, **settings))
        folds[participant] = evaluate(model, test)
        folds[participant]['settings'] = settings
    return pool_folds(folds)


SENSOR_THRESHOLDS = (.45, .50, .55, .60, .65, .70)


def fit_agreement(records, settings, thresholds=(.50, .50)):
    sensor_settings = dict(settings, motion_weight=0.)
    return dict(joint=fit(records, ALL, **settings),
                mic=fit(records, np.array(MIC_COLUMNS), **dict(sensor_settings, threshold=thresholds[0])),
                imu=fit(records, np.array(IMU_COLUMNS), **dict(sensor_settings, threshold=thresholds[1])))


def sensor_score_folds(records, settings):
    for participant, train, test in grouped_split(records):
        test = reviewed_records(test)
        if not test or not any(r['intervals'] for r in train):
            continue
        model = AgreementModel(fit_agreement(train, settings))
        scored = []
        for record in score_records(model.joint, test):
            features = record['features'][record['usable']]
            mic = np.zeros(len(record['scores']))
            imu = np.zeros_like(mic)
            mic[record['usable']] = model.mic.score(features)
            imu[record['usable']] = model.imu.score(features)
            scored.append(dict(record, mic_scores=mic, imu_scores=imu))
        yield participant, scored


def evaluate_agreement(scored, settings, thresholds):
    gated = [dict(r, scores=agreement_scores(r['scores'], r['mic_scores'], r['imu_scores'], *thresholds))
             for r in scored]
    return evaluate_scores(gated, settings['threshold'], settings['refractory'])


def choose_sensor_thresholds(scored, settings):
    if not scored:
        return (.50, .50)
    baseline = evaluate_scores(scored, settings['threshold'], settings['refractory'])
    choices = ((m, i) for m in SENSOR_THRESHOLDS for i in SENSOR_THRESHOLDS)

    def quality(thresholds):
        row = evaluate_agreement(scored, settings, thresholds)
        return row['recall'] >= baseline['recall'] - .05, row['f1'], row['tp'], -row['fp']

    return max(choices, key=quality)


def agreement_evaluation(records, settings):
    scored_folds = dict(sensor_score_folds(records, settings))
    thresholds = choose_sensor_thresholds([r for rows in scored_folds.values() for r in rows], settings)
    fixed = {p: evaluate_agreement(rows, settings, thresholds) for p, rows in scored_folds.items()}
    nested = {}
    for participant, train, _ in grouped_split(records):
        if participant not in scored_folds:
            continue
        calibration = [r for _, rows in sensor_score_folds(train, settings) for r in rows]
        levels = choose_sensor_thresholds(calibration, settings)
        nested[participant] = dict(evaluate_agreement(scored_folds[participant], settings, levels),
                                   mic_threshold=levels[0], imu_threshold=levels[1])
    return thresholds, dict(selected=pool_folds(fixed), nested_thresholds=pool_folds(nested))


def public_records(path, participants):
    manifest = json.loads((HOME.parent / 'results/records.json').read_text())
    with zipfile.ZipFile(path / 'signals.zip') as signals:
        for i, record in enumerate(r for r in manifest if r['participant'] in participants):
            raw = signals.read(f'signals/{record["id"]}.npy')
            if hashlib.sha256(raw).hexdigest() != record['sha256']:
                raise ValueError(f'Public recording checksum mismatch for {record["id"]}.')
            signal = np.load(io.BytesIO(raw), allow_pickle=False)
            times = np.arange(len(signal)) / 4000.
            yield make_record(np.column_stack((times, signal[:, 0])), np.column_stack((times, signal[:, 1:])),
                              record['intervals'], wearable=False)
            if (i + 1) % 100 == 0:
                print(f'Processed {i + 1} public recordings', flush=True)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings', type=Path, default=HOME / 'recordings' / 'participants')
    parser.add_argument('--exclude-participant', action='append', default=[],
                        help='Participant code to omit from tuning and final fit')
    parser.add_argument('--compare-exclusion', help='Evaluate keeping versus dropping this participant')
    parser.add_argument('--holdout-session', action='append', default=[],
                        help='Relative participant/session path for the chronological comparison')
    parser.add_argument('--public-data', type=Path, help='Directory containing the checksum-verified public signals.zip')
    args = parser.parse_args()
    started = perf_counter()
    records = list(wearable_records(args.recordings))
    participants = {r['participant'] for r in records if r['intervals']}
    excluded = set(args.exclude_participant)
    if len(participants - excluded) < 2:
        raise ValueError('At least two included participants with reviewed swallows are needed.')
    report = {}
    if args.compare_exclusion:
        subject = args.compare_exclusion
        shared = participants - {subject}
        if subject not in participants or len(shared) < 2:
            raise ValueError('The exclusion comparison needs the named participant and two others.')
        print(f'Comparing models with and without {subject}', flush=True)
        comparison = {}
        for omitted in (set(), {subject}):
            key = 'without' if omitted else 'with'
            included = [r for r in records if r['participant'] not in omitted]
            settings = select_detector(included)
            comparison[key] = dict(
                settings=settings,
                fixed_comparison=grouped_evaluation(records, exclude_training=omitted,
                                                    test_participants=shared, **PREVIOUS_SETTINGS),
                tuned_comparison=grouped_evaluation(records, exclude_training=omitted,
                                                    test_participants=shared, **settings),
            )
            holdouts = set(args.holdout_session)
            if holdouts:
                test = [r for r in records if r['session'] in holdouts]
                if {r['session'] for r in test} != holdouts:
                    raise ValueError('Unknown hold-out session.')
                train = [r for r in included if r['session'] not in holdouts]
                chronological_model = Model(fit(train, ALL, **select_detector(train)))
                comparison[key]['chronological'] = evaluate(chronological_model, reviewed_records(test))
        report['exclusion_comparison'] = comparison

    training = [r for r in records if r['participant'] not in excluded]
    print('Tuning the deployment model', flush=True)
    settings = select_detector(training)
    report.update(
        wearable=grouped_evaluation(training, **settings),
        nested_tuning=grouped_evaluation(training, tune=True),
        previous_wearable=grouped_evaluation(training, **PREVIOUS_SETTINGS),
        deployed_settings=settings, excluded_participants=sorted(excluded),
        training_sessions=[r['session'] for r in training],
        evaluated_sessions=len(reviewed_records(training)),
        annotations=dict(Counter(label for r in training for label in r['annotations'])),
    )
    if excluded:
        audited = reviewed_records([r for r in records if r['participant'] in excluded])
        if audited:
            report['excluded_participant_audit'] = evaluate(Model(fit(training, ALL, **settings)), audited)
    thresholds, report['sensor_agreement'] = agreement_evaluation(training, settings)
    report['sensor_agreement']['thresholds'] = dict(zip(('mic', 'imu'), thresholds))
    write_json(MODEL_PATH, fit_agreement(training, settings, thresholds))
    report_path = MODEL_PATH.with_name('evaluation.json')
    write_json(report_path, report)
    print(f'Wearable model and evaluation saved to {MODEL_PATH.parent}', flush=True)
    if args.public_data:
        splits = json.loads((HOME.parent / 'results/participant_splits.json').read_text())
        public = list(public_records(args.public_data, set(splits['train'])))
        public_model = Model(fit(public, SHAPE))
        public_test = list(public_records(args.public_data, set(splits['test'])))
        report['public_transfer'] = dict(training_sessions=len(public), training_participants=len(splits['train']),
                                         public_test_sessions=len(public_test), public_test=evaluate(public_model, public_test),
                                         wearable_test=evaluate(public_model, records),
                                         wearable_shape_control=grouped_evaluation(records, SHAPE)['pooled'],
                                         wearable_with_public=grouped_evaluation(records, SHAPE, public)['pooled'])
        write_json(report_path, report)
    print(f'Finished in {perf_counter() - started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
