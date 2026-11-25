"""Check how closely adapted public windows match wearable windows in detector-feature space.

For every detector feature it reports the standardised mean difference (SMD, in wearable
standard deviations) between public and wearable windows, separately for swallow and
background windows, for: the raw public signal as used by the earlier transfer test, and
the adapted signal. Lower is better; |SMD| < 0.2 is conventionally a negligible shift.
Writes models/mixed/adaptation_check.json and adaptation_check.png.

    .venv/bin/python data_collection/check_adaptation.py
"""
import json
import zipfile
from pathlib import Path

import numpy as np

from adapt_public import (HOME, PUBLIC_RATE, REPO, adapt_signal, calibrate, imu_rate_and_gravity, public_manifest,
                          read_public, training_participants)
from live_detection import FEATURE_COUNT
from train_live import make_record, wearable_records

NAMES = ([f'mic_band{i}' for i in range(4)] + [f'mic_env{i}' for i in range(5)] + ['mic_crest', 'mic_logrms'] +
         [f'imu_band{i}' for i in range(4)] + [f'imu_env{i}' for i in range(5)] + ['imu_crest', 'imu_logrms'] +
         ['mic_peak', 'mic_entropy', 'mic_rise', 'imu_peak', 'imu_entropy', 'imu_rise', 'mic_imu_corr'])


def pooled(records):
    rows, labels = [], []
    for r in records:
        ok = np.isfinite(r['features']).all(axis=1)
        rows.append(r['features'][ok])
        labels.append(r['labels'][ok])
    return np.vstack(rows), np.concatenate(labels)


def smd(public, wearable):
    sd = np.maximum(wearable.std(axis=0), 1e-9)
    return (public.mean(axis=0) - wearable.mean(axis=0)) / sd


def main():
    out = HOME / 'models' / 'mixed'
    out.mkdir(parents=True, exist_ok=True)
    wearable = list(wearable_records(HOME / 'recordings' / 'participants'))
    w_rows, w_labels = pooled(wearable)
    sessions = sorted({HOME / 'recordings' / 'participants' / r['session'] for r in wearable})
    manifest = public_manifest(training_participants())
    records = manifest[3::8]  # disjoint from the calibration sample (manifest[::8])
    public_dir = REPO / 'data' / 'raw'
    plan = calibrate(public_dir, manifest[::8], wearable, *imu_rate_and_gravity(sessions), verbose=True)
    raw, adapted = [], []
    with zipfile.ZipFile(public_dir / 'signals.zip') as signals:
        for i, record in enumerate(records):
            x = read_public(signals, record)
            t = np.arange(len(x)) / PUBLIC_RATE
            intervals = [tuple(v) for v in record['intervals']]
            raw.append(make_record(np.column_stack((t, x[:, 0])), np.column_stack((t, x[:, 1:])), intervals,
                                   wearable=False))
            mic, imu = adapt_signal(x, plan)
            adapted.append(make_record(mic, imu, intervals, wearable=True))
    result = dict(plan=plan, public_recordings=len(records), features=NAMES)
    for name, group in (('raw_public', raw), ('adapted_public', adapted)):
        rows, labels = pooled(group)
        gated = 1 - sum(np.isfinite(r['features']).all(axis=1).sum() for r in group) / sum(len(r['features']) for r in group)
        result[name] = dict(
            swallow_smd=smd(rows[labels], w_rows[w_labels]).round(3).tolist(),
            background_smd=smd(rows[~labels], w_rows[~w_labels]).round(3).tolist(),
            mean_abs_smd_swallow=float(np.abs(smd(rows[labels], w_rows[w_labels])).mean()),
            mean_abs_smd_background=float(np.abs(smd(rows[~labels], w_rows[~w_labels])).mean()),
            gated_fraction=float(gated), windows=int(len(labels)), swallow_windows=int(labels.sum()))
        print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in result[name].items()
                     if not k.endswith('smd')}, flush=True)
    (out / 'adaptation_check.json').write_text(json.dumps(result, indent=2))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        x = np.arange(FEATURE_COUNT)
        for ax, key, title in zip(axes, ('swallow_smd', 'background_smd'), ('Swallow windows', 'Background windows')):
            ax.bar(x - .2, np.clip(result['raw_public'][key], -6, 6), .4, label='Raw public (earlier transfer test)',
                   color='#c9c9c9')
            ax.bar(x + .2, np.clip(result['adapted_public'][key], -6, 6), .4, label='Adapted public', color='#2462d8')
            ax.axhspan(-.2, .2, color='#2ca02c', alpha=.12)
            ax.set_title(f'{title}: public minus wearable, in wearable SDs (clipped at ±6)')
            ax.set_ylabel('SMD')
        axes[0].legend(loc='upper right')
        axes[1].set_xticks(x, NAMES, rotation=70, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / 'adaptation_check.png', dpi=130)
    except ImportError:
        pass


if __name__ == '__main__':
    main()
