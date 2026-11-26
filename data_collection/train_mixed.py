"""Train wearable swallow detectors on our recordings plus the adapted public dataset.

Each model is a weighted mix: ``public_share`` is the fraction of the total training
loss that comes from adapted public windows; the rest comes from wearable windows.
Inside each dataset the swallow and background classes are balanced, so a 50/50 model
gives the 85 wearable swallows the same total influence as the ~1,600 public swallows.

Evaluation is leave-one-wearable-participant-out. The public data is always in
training (never tested here) and the held-out wearable participant never contributes
to adaptation statistics, scaling, or threshold/cooldown selection, which is picked
by an inner leave-one-participant-out loop on the remaining wearable participants.

Run from the repository root after ``adapt_public.py`` has built its cache:
    data_collection/.venv/bin/python data_collection/train_mixed.py
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from adapt_public import adapted_public
from live_detection import FEATURE_VERSION, IMU_COLUMNS, MIC_COLUMNS, AgreementModel, build_model, quadratic_terms
from train_live import (ALL, evaluate, evaluate_scores, grouped_split, pool_folds, reviewed_records,
                        score_records, wearable_records, write_json)

HOME = Path(__file__).resolve().parent
OUT = HOME / 'models' / 'mixed'
THRESHOLDS = tuple(np.round(np.arange(.30, .91, .05), 2))
COOLDOWNS = (.8, 1.2, 1.6)
SHARES = (0., .2, .35, .5, .65, .8, .9, .95, 1.)
BASE = dict(quadratic=True, penalty=.015, background_weight=4., reviewed_weight=4.)


def dataset_rows(records, columns, mask):
    rows = np.concatenate([r['features'][r[mask]] for r in records])[:, columns]
    labels = np.concatenate([r['labels'][r[mask]] for r in records])
    reviewed = np.concatenate([r.get('reviewed_background', np.zeros(len(r['labels']), bool))[r[mask]]
                               for r in records])
    counts = np.bincount(labels.astype(int), minlength=2)
    if np.any(counts == 0):
        raise ValueError('Each dataset needs swallow and background windows.')
    return rows, labels, reviewed, .5 / counts[labels.astype(int)]


def fit_mixed(wearable, public, columns=ALL, public_share=.5, *, quadratic=True, penalty=.015,
              background_weight=4., reviewed_weight=4., threshold=.6, refractory=1.2, max_public_windows=None,
              rng_seed=0):
    """Weighted logistic fit; returns a ``Model``-compatible dictionary."""
    columns = np.asarray(columns)
    parts = []
    reweight = background_weight or reviewed_weight
    if public_share < 1:
        parts.append((1 - public_share, *dataset_rows(wearable, columns, 'clean_trainable' if reweight else 'trainable')))
    if public_share > 0:
        rows, labels, reviewed, weights = dataset_rows(public, columns, 'trainable')
        if max_public_windows and len(labels) > max_public_windows:
            keep = np.random.default_rng(rng_seed).choice(len(labels), max_public_windows, replace=False)
            rows, labels, reviewed = rows[keep], labels[keep], reviewed[keep]
            counts = np.bincount(labels.astype(int), minlength=2)
            weights = .5 / counts[labels.astype(int)]
        parts.append((public_share, rows, labels, reviewed, weights))
    # Standardise with the deployment target (wearable) whenever it is in the mix.
    reference = parts[0][1]
    mean, scale = reference.mean(axis=0), np.maximum(reference.std(axis=0), 1e-8)
    z = [(p[1] - mean) / scale for p in parts]
    if quadratic:
        z = [quadratic_terms(x) for x in z]
    Z = np.vstack(z)
    labels = np.concatenate([p[2] for p in parts])
    owner = np.concatenate([np.full(len(p[2]), i) for i, p in enumerate(parts)])
    reviewed = np.concatenate([p[3] for p in parts])
    weights = np.concatenate([p[0] * p[4] for p in parts])
    regularization = penalty

    def loss(params):
        coeff, bias = params[:-1], params[-1]
        logits = Z @ coeff + bias
        diff = weights * (expit(logits) - labels)
        return (weights @ (np.logaddexp(0, logits) - labels * logits) + regularization * (coeff @ coeff),
                np.r_[Z.T @ diff + 2 * regularization * coeff, diff.sum()])

    def optimize(initial):
        result = minimize(loss, initial, method='L-BFGS-B', jac=True, options=dict(maxiter=500))
        return result.x

    params = optimize(np.zeros(Z.shape[1] + 1))
    if reweight:
        # Hard-negative emphasis, applied within each dataset so its share is unchanged.
        scores = expit(Z @ params[:-1] + params[-1])
        for i in range(len(parts)):
            negative = (owner == i) & ~labels
            if not negative.any():
                continue
            confusing = scores[negative] >= np.quantile(scores[negative], .8)
            factors = 1 + background_weight * confusing + reviewed_weight * reviewed[negative]
            weights[negative] *= factors / factors.mean()
        params = optimize(params)
    return dict(feature_version=FEATURE_VERSION, columns=columns.tolist(), mean=mean.tolist(),
                scale=scale.tolist(), weights=params[:-1].tolist(), bias=float(params[-1]),
                quadratic=quadratic, threshold=threshold, refractory=refractory)


TREES = dict(learning_rate=.05, max_leaf_nodes=15, max_iter=300, l2_regularization=1., min_samples_leaf=40)


def export_trees(classifier):
    """Flatten a fitted binary HistGradientBoostingClassifier into TreeModel arrays."""
    feature, split, left, right, leaf, value, roots = [], [], [], [], [], [], []
    depth = 0
    for (predictor,) in classifier._predictors:
        nodes = predictor.nodes
        offset = len(feature)
        roots.append(offset)
        feature += nodes['feature_idx'].astype(int).tolist()
        split += nodes['num_threshold'].astype(float).tolist()
        left += (nodes['left'].astype(int) + offset).tolist()
        right += (nodes['right'].astype(int) + offset).tolist()
        leaf += nodes['is_leaf'].astype(bool).tolist()
        value += nodes['value'].astype(float).tolist()
        depth = max(depth, int(nodes['depth'].max()))
    return dict(feature=feature, split=split, left=left, right=right, leaf=leaf, value=value, roots=roots,
                depth=depth, baseline=float(np.ravel(classifier._baseline_prediction)[0]))


def fit_mixed_trees(wearable, public, columns=ALL, public_share=.5, *, threshold=.8, refractory=1.2, seed=0,
                    **_ignored):
    """Gradient-boosted trees with the same per-dataset, class-balanced weighting."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    columns = np.asarray(columns)
    parts = []
    if public_share < 1:
        rows, labels, _, weights = dataset_rows(wearable, columns, 'clean_trainable')
        parts.append((rows, labels, (1 - public_share) * weights))
    if public_share > 0:
        rows, labels, _, weights = dataset_rows(public, columns, 'trainable')
        parts.append((rows, labels, public_share * weights))
    X = np.vstack([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    w = np.concatenate([p[2] for p in parts])
    classifier = HistGradientBoostingClassifier(**TREES, random_state=seed, early_stopping=False)
    classifier.fit(X, y, sample_weight=w / w.mean())
    return dict(feature_version=FEATURE_VERSION, columns=columns.tolist(), threshold=threshold,
                refractory=refractory, trees=export_trees(classifier), trees_settings=TREES)


FOREST = dict(n_estimators=300, max_depth=14, min_samples_leaf=5, max_features='sqrt')


def export_forest(classifier):
    """Flatten a fitted RandomForestClassifier into TreeModel arrays (mean of leaf probabilities)."""
    feature, split, left, right, leaf, value, roots = [], [], [], [], [], [], []
    depth = 0
    for estimator in classifier.estimators_:
        tree = estimator.tree_
        offset = len(feature)
        roots.append(offset)
        is_leaf = tree.children_left < 0
        counts = tree.value[:, 0, :]
        probability = counts[:, 1] / np.maximum(counts.sum(axis=1), 1e-12)
        feature += np.where(is_leaf, 0, tree.feature).astype(int).tolist()
        split += np.where(is_leaf, 0., tree.threshold).round(7).tolist()
        left += np.where(is_leaf, 0, tree.children_left + offset).astype(int).tolist()
        right += np.where(is_leaf, 0, tree.children_right + offset).astype(int).tolist()
        leaf += is_leaf.tolist()
        value += probability.round(5).tolist()
        depth = max(depth, int(tree.max_depth))
    return dict(feature=feature, split=split, left=left, right=right, leaf=leaf, value=value, roots=roots,
                depth=depth, baseline=0., aggregate='mean')


def fit_mixed_forest(wearable, public, columns=ALL, public_share=0., *, threshold=.7, refractory=1.2, seed=0,
                     **_ignored):
    """Random forest with the same per-dataset, class-balanced weighting."""
    from sklearn.ensemble import RandomForestClassifier
    columns = np.asarray(columns)
    parts = []
    if public_share < 1:
        rows, labels, _, weights = dataset_rows(wearable, columns, 'clean_trainable')
        parts.append((rows, labels, (1 - public_share) * weights))
    if public_share > 0:
        rows, labels, _, weights = dataset_rows(public, columns, 'trainable')
        parts.append((rows, labels, public_share * weights))
    X = np.vstack([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    w = np.concatenate([p[2] for p in parts])
    classifier = RandomForestClassifier(**FOREST, random_state=seed, n_jobs=-1)
    classifier.fit(X, y, sample_weight=w / w.mean())
    return dict(feature_version=FEATURE_VERSION, columns=columns.tolist(), threshold=threshold,
                refractory=refractory, trees=export_forest(classifier), forest_settings=FOREST)


FITTERS = dict(logistic=fit_mixed, trees=fit_mixed_trees, forest=fit_mixed_forest)


def best_settings(scored):
    best = None
    for threshold in THRESHOLDS:
        for cooldown in COOLDOWNS:
            row = evaluate_scores(scored, threshold, cooldown)
            key = (row['f1'], row['tp'], -row['fp'])
            if best is None or key > best[0]:
                best = (key, dict(threshold=float(threshold), refractory=cooldown))
    return best[1]


def nested_evaluation(wearable, public_for, share, settings, fitter=fit_mixed):
    """Outer LOPO on wearable participants; inner LOPO picks threshold and cooldown.

    ``public_for(train_participants)`` returns public records adapted with statistics
    from those wearable participants only.
    """
    folds, curves = {}, {}
    for participant, train, test in grouped_split(wearable):
        test = reviewed_records(test)
        if not test:
            continue
        inner_scored = []
        for _, inner_train, inner_test in grouped_split(train):
            inner_test = reviewed_records(inner_test)
            if not inner_test or (share < 1 and not any(r['intervals'] for r in inner_train)):
                continue
            public = public_for(sorted({r['participant'] for r in inner_train}))
            model = build_model(fitter(inner_train, public, public_share=share, **settings))
            inner_scored.extend(score_records(model, inner_test))
        chosen = best_settings(inner_scored)
        public = public_for(sorted({r['participant'] for r in train}))
        model = build_model(fitter(train, public, public_share=share, **settings, **chosen))
        scored = score_records(model, test)
        folds[participant] = dict(evaluate_scores(scored, **chosen), settings=chosen)
        curves[participant] = scored
    pooled = pool_folds(folds)
    # Diagnostic only (not used for selection): best fixed setting on the pooled outer scores.
    all_scored = [r for rows in curves.values() for r in rows]
    oracle = best_settings(all_scored)
    pooled['oracle_fixed_settings'] = dict(evaluate_scores(all_scored, **oracle), settings=oracle)
    for key in ('delays_s',):
        pooled['pooled'].pop(key, None)
        pooled['oracle_fixed_settings'].pop(key, None)
        for row in pooled['by_participant'].values():
            row.pop(key, None)
    return pooled


def nested_share_selection(wearable, public_for, shares, settings, fitter=fit_mixed):
    """Choose the public share inside each outer fold, using inner LOPO only.

    This estimates how well the *procedure* "pick the best share on the training
    participants" works for a new wearer, instead of reporting the share that happened to
    score best on the same three people.
    """
    folds = {}
    for participant, train, test in grouped_split(wearable):
        test = reviewed_records(test)
        if not test:
            continue
        inner = {}
        for share in shares:
            scored = []
            for _, inner_train, inner_test in grouped_split(train):
                inner_test = reviewed_records(inner_test)
                if inner_test:
                    public = public_for(sorted({r['participant'] for r in inner_train}))
                    model = build_model(fitter(inner_train, public, public_share=share, **settings))
                    scored.extend(score_records(model, inner_test))
            chosen = best_settings(scored)
            row = evaluate_scores(scored, **chosen)
            inner[share] = ((row['f1'], -row['fp']), chosen)
        share = max(inner, key=lambda s: inner[s][0])
        chosen = inner[share][1]
        public = public_for(sorted({r['participant'] for r in train}))
        model = build_model(fitter(train, public, public_share=share, **settings, **chosen))
        folds[participant] = dict(evaluate_scores(score_records(model, test), **chosen), settings=chosen,
                                  selected_share=share,
                                  inner_f1={f'{s:.2f}': round(v[0][0], 4) for s, v in inner.items()})
        print(f'  outer {participant}: inner selected share {share:.2f}', flush=True)
    pooled = pool_folds(folds)
    for row in [pooled['pooled'], *pooled['by_participant'].values()]:
        row.pop('delays_s', None)
    return pooled


def brief(row):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
            if k in ('tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'extra_per_minute', 'median_detection_delay_s')}


VARIANTS = {
    'base (quadratic, L2 0.015, hard negatives x4)': {},
    'linear': dict(quadratic=False, penalty=.02),
    'quadratic, L2 0.005': dict(penalty=.005),
    'quadratic, L2 0.05': dict(penalty=.05),
    'no hard-negative reweighting': dict(background_weight=0., reviewed_weight=0.),
    'hard negatives x8': dict(background_weight=8., reviewed_weight=8.),
}


SENSOR_GATE = (.4, .4)


def agreement_fit(wearable, public, share, settings, fitter, gate, chosen=None):
    """Joint detector plus logistic microphone-only and IMU-only checks (the app's format)."""
    sensor = dict(BASE, **{k: v for k, v in settings.items() if k in BASE})
    return dict(joint=fitter(wearable, public, ALL, share, **settings, **(chosen or {})),
                mic=fit_mixed(wearable, public, np.array(MIC_COLUMNS), share, **dict(sensor, threshold=gate[0])),
                imu=fit_mixed(wearable, public, np.array(IMU_COLUMNS), share, **dict(sensor, threshold=gate[1])))


def export_model(wearable, public_for, share, settings, name, gate=SENSOR_GATE, fitter=fit_mixed):
    """Fit the app-ready AgreementModel on every wearable participant plus adapted public data.

    Alert threshold and cooldown come from pooled leave-one-participant-out scores with the
    sensor-agreement gate applied, exactly as the app will score windows.
    """
    scored, folds = [], {}
    for participant, train, test in grouped_split(wearable):
        test = reviewed_records(test)
        if test:
            public = public_for(sorted({r['participant'] for r in train}))
            model = AgreementModel(agreement_fit(train, public, share, settings, fitter, gate))
            rows = score_records(model, test)
            scored.extend(rows)
            folds[participant] = rows
    chosen = best_settings(scored)
    lopo = pool_folds({p: evaluate_scores(rows, **chosen) for p, rows in folds.items()})
    for row in [lopo['pooled'], *lopo['by_participant'].values()]:
        row.pop('delays_s', None)
    public = public_for(sorted({r['participant'] for r in wearable}))
    data = agreement_fit(wearable, public, share, settings, fitter, gate, chosen)
    data['training'] = dict(
        model={fit_mixed_trees: 'boosted trees', fit_mixed_forest: 'random forest'}.get(fitter, 'quadratic logistic'),
        public_share=share, wearable_share=round(1 - share, 4), settings=settings, chosen=chosen,
        sensor_gate=dict(mic=gate[0], imu=gate[1]), wearable_sessions=sorted(r['session'] for r in wearable),
        public_recordings=len(public),
        leave_one_participant_out=dict(note='Threshold/cooldown chosen on these same pooled folds; '
                                            'see evaluation*.json for nested estimates.',
                                       pooled=brief(lopo['pooled']),
                                       by_participant={p: brief(r) for p, r in lopo['by_participant'].items()}))
    AgreementModel(data)  # validate the format the app loads
    path = OUT / f'{name}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(',', ':')) + '\n', encoding='utf-8')   # compact: forests are large
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--recordings', type=Path, default=HOME / 'recordings' / 'participants')
    parser.add_argument('--public-data', type=Path, default=HOME.parent / 'data' / 'raw')
    parser.add_argument('--mode', choices=('sweep', 'control', 'variants', 'select-share', 'export'),
                        default='sweep')
    parser.add_argument('--shares', type=float, nargs='*', default=SHARES)
    parser.add_argument('--export', nargs='*', default=[], metavar='NAME=SHARE',
                        help='Models to export in export mode, e.g. mixed_50_50=0.5')
    parser.add_argument('--model', choices=tuple(FITTERS), default='logistic')
    parser.add_argument('--no-gate', action='store_true',
                        help='Export without the mic/IMU agreement check (it lowers random-forest accuracy)')
    args = parser.parse_args()
    fitter = FITTERS[args.model]
    base = BASE if args.model == 'logistic' else {}
    suffix = '' if args.model == 'logistic' else f'_{args.model}'
    started = perf_counter()
    wearable = list(wearable_records(args.recordings))
    participants = sorted({r['participant'] for r in wearable if r['intervals']})
    print('Wearable participants:', participants, flush=True)

    cache = {}

    def public_for(reference_participants):
        key = tuple(reference_participants)
        if key not in cache:
            cache[key] = adapted_public(args.public_data, [r for r in wearable if r['participant'] in key])[0]
        return cache[key]

    def run(label, share, settings, source=public_for):
        result = nested_evaluation(wearable, source, share, settings, fitter)
        print(f'{label}:', brief(result['pooled']), flush=True)
        print('  per participant:', {p: brief(r) for p, r in result['by_participant'].items()}, flush=True)
        return result

    header = dict(design=__doc__.strip().split('\n\n')[1], participants=participants, model=args.model, base_settings=base if base else TREES)
    if args.mode == 'sweep':
        report = dict(header, shares={f'{s:.2f}': run(f'public share {s:.2f}', s, base) for s in args.shares})
        write_json(OUT / f'evaluation{suffix}.json', report)
    elif args.mode == 'control':
        from adapt_public import raw_public
        raw = raw_public(args.public_data)
        report = dict(header, note='Public signals used without adaptation (raw units, 4 kHz, no EQ).',
                      shares={f'{s:.2f}': run(f'UNADAPTED public share {s:.2f}', s, base, lambda _: raw)
                              for s in args.shares})
        write_json(OUT / f'evaluation_unadapted_control{suffix}.json', report)
    elif args.mode == 'variants':
        report = dict(header, variants={})
        for share in args.shares:
            for label, overrides in VARIANTS.items():
                report['variants'][f'{share:.2f} | {label}'] = run(f'share {share:.2f} | {label}', share,
                                                                    dict(BASE, **overrides))
        write_json(OUT / 'evaluation_variants.json', report)
    elif args.mode == 'select-share':
        result = nested_share_selection(wearable, public_for, args.shares, base, fitter)
        result['pooled'].pop('delays_s', None)
        print('Nested share selection:', brief(result['pooled']), flush=True)
        write_json(OUT / f'evaluation_share_selection{suffix}.json', dict(header, shares=list(args.shares), result=result))
    else:
        for item in args.export:
            name, share = item.split('=')
            data = export_model(wearable, public_for, float(share), base, name, fitter=fitter,
                                gate=(.01, .01) if args.no_gate else SENSOR_GATE)
            print(f'Exported {name}: share {share}, settings {data["training"]["chosen"]}', flush=True)
    print(f'Finished in {perf_counter() - started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
