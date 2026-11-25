"""Make the public Pittsburgh recordings look like Swallow Studio wearable recordings.

Public (Zenodo 4539695): 4 kHz; columns = contact microphone, A-P, S-I, M-L acceleration
in arbitrary (voltage-like) units with gravity removed. Wearable: 44.1 kHz signed 16-bit
microphone PCM and a ~904 Hz ISM330DHCX accelerometer in m/s^2 that includes gravity.

Adaptation, using statistics from public training participants and from the wearable
participants passed in (never from a held-out wearable participant):

1. Resampling. Microphone 4 kHz -> 44.1 kHz, rounded and clipped to 16-bit PCM counts.
   Accelerometer anti-alias filtered and resampled 4 kHz -> the measured wearable IMU rate.
2. Axes and gravity. Anatomical axes are mapped onto the wearable's native axes (gravity
   sits on native X, i.e. along the neck, so S-I -> X, M-L -> Y, A-P -> Z) and the
   wearable's median gravity vector is added. (The detector's IMU features combine the
   three axes, so this mainly makes the data usable by future per-axis models.)
3. Units and spectral shape, calibrated in the detector's own feature space. Starting
   from a level-matched gain, a few iterations adjust (a) one gain per sensor so median
   swallow-window RMS equals the wearable's and (b) a smooth linear-phase equaliser so
   the mean log band shares of public windows match the wearable's (swallow and
   background windows weighted equally). The equaliser is limited to +/-12 dB.

The public background is already noisier than the wearable's relative to swallows
(swallow/background RMS ~2:1 versus ~4:1), so no noise is added. Swallow durations
already agree (median 0.80 s public vs 0.84 s wearable), so labels are used unchanged.
The detector then sees adapted public recordings through exactly the same causal
feature extractor and quality gates as live wearable data.
"""
import hashlib
import io
import json
import pickle
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import firwin2, lfilter, resample_poly

HOME = Path(__file__).resolve().parent
REPO = HOME.parent
CACHE = REPO / '_cache' / 'adapted_public'
ADAPTATION_VERSION = 2
PUBLIC_RATE = 4000
MIC_RATE = 44100
MIC_BANDS = (20, 60, 150, 350, 800)      # same as live_detection.window_features
IMU_BANDS = (5, 15, 40, 100, 300)
MIC_SHARES, MIC_LOGRMS = np.arange(0, 4), 10
IMU_SHARES, IMU_LOGRMS = np.arange(11, 15), 21
EQ_LIMIT_DB = 12.
CALIBRATION_ITERATIONS = 4
AXIS_MAP = {'A-P': 2, 'S-I': 0, 'M-L': 1}   # public column -> wearable native axis


def public_manifest(participants=None):
    manifest = json.loads((REPO / 'public_baseline' / 'results' / 'records.json').read_text())
    return [r for r in manifest if participants is None or r['participant'] in participants]


def training_participants():
    """Public participants outside the original 27-person public test split."""
    splits = json.loads((REPO / 'public_baseline' / 'results' / 'participant_splits.json').read_text())
    return set(splits['train']) | set(splits['validation'])


def read_public(zip_file, record):
    raw = zip_file.read(f'signals/{record["id"]}.npy')
    if hashlib.sha256(raw).hexdigest() != record['sha256']:
        raise ValueError(f'Public recording checksum mismatch for {record["id"]}.')
    return np.load(io.BytesIO(raw), allow_pickle=False).astype(float)


def imu_rate_and_gravity(sessions):
    rates, gravity = [], []
    for folder in sessions:
        imu = pd.read_csv(folder / 'IMU.csv', usecols=['time_s', 'ax', 'ay', 'az']).to_numpy(float)
        rates.append((len(imu) - 1) / (imu[-1, 0] - imu[0, 0]))
        gravity.append(imu[:, 1:].mean(axis=0))
    return float(np.median(rates)), np.median(gravity, axis=0)


def band_fir(bands, band_db, rate, taps):
    """Smooth linear-phase FIR with the given gain (dB) at each band's geometric centre."""
    centres = np.sqrt(np.asarray(bands[:-1], float) * np.asarray(bands[1:], float))
    freq = np.r_[0., np.geomspace(1., rate / 2, 400)]
    gain_db = np.interp(np.log(np.maximum(freq, 1.)), np.log(centres), band_db)  # held flat outside
    return firwin2(taps, freq, 10 ** (gain_db / 20), fs=rate)


def _filter_centred(x, fir):
    delay = (len(fir) - 1) // 2
    y = lfilter(fir, 1, np.vstack((x, np.zeros((delay,) + x.shape[1:]))), axis=0)
    return y[delay:]


def adapt_signal(x, plan):
    """Return wearable-format (mic[t, pcm], imu[t, ax, ay, az]) for one public N x 4 array."""
    mic = x[:, :1] - x[:, :1].mean()
    mic = _filter_centred(mic, band_fir(MIC_BANDS, plan['mic_band_db'], PUBLIC_RATE, 511))[:, 0] * plan['mic_gain']
    mic = np.clip(np.round(resample_poly(mic, 441, 40)), -32768, 32767)
    up, down = int(round(plan['imu_rate'] * 2)), 2 * PUBLIC_RATE
    rate = PUBLIC_RATE * up / down
    acc = resample_poly(x[:, 1:] - x[:, 1:].mean(axis=0), up, down, axis=0)
    acc = _filter_centred(acc, band_fir(IMU_BANDS, plan['acc_band_db'], rate, 511)) * plan['acc_gain']
    native = np.empty_like(acc)
    for column, name in enumerate(('A-P', 'S-I', 'M-L')):
        native[:, AXIS_MAP[name]] = acc[:, column]
    native += np.asarray(plan['gravity'])
    return (np.column_stack((np.arange(len(mic)) / MIC_RATE, mic)),
            np.column_stack((np.arange(len(native)) / rate, native)))


def _featurise(x, intervals, plan):
    from train_live import make_record
    mic, imu = adapt_signal(x, plan)
    return make_record(mic, imu, intervals, wearable=True)


def _pooled(records):
    rows, labels = [], []
    for r in records:
        ok = np.isfinite(r['features']).all(axis=1)
        rows.append(r['features'][ok])
        labels.append(r['labels'][ok])
    return np.vstack(rows), np.concatenate(labels)


def _log_shares(rows, columns):
    shares = rows[:, columns]
    return np.log(shares / shares.sum(axis=1, keepdims=True) + 1e-6)


def calibrate(public_dir, sample, wearable_records, imu_rate, gravity, verbose=False):
    """Fit gains and equalisers so adapted public features match wearable features."""
    w_rows, w_labels = _pooled(wearable_records)
    with zipfile.ZipFile(Path(public_dir) / 'signals.zip') as signals:
        signals_sample = [(read_public(signals, r), [tuple(i) for i in r['intervals']]) for r in sample]
    # Level-matched starting point (band-limited swallow RMS ratio measured earlier).
    plan = dict(version=ADAPTATION_VERSION, imu_rate=imu_rate, gravity=np.asarray(gravity).tolist(),
                mic_gain=5000., acc_gain=2.5, mic_band_db=np.zeros(4), acc_band_db=np.zeros(4))
    history = []
    for iteration in range(CALIBRATION_ITERATIONS + 1):
        p_rows, p_labels = _pooled([_featurise(x, iv, plan) for x, iv in signals_sample])
        step = {}
        for sensor, shares, logrms in (('mic', MIC_SHARES, MIC_LOGRMS), ('acc', IMU_SHARES, IMU_LOGRMS)):
            delta = np.zeros(len(shares))
            for label in (True, False):
                delta += .5 * (_log_shares(w_rows[w_labels == label], shares).mean(axis=0) -
                               _log_shares(p_rows[p_labels == label], shares).mean(axis=0))
            delta -= delta.mean()
            w_rms = np.median(np.expm1(w_rows[w_labels, logrms]))
            p_rms = np.median(np.expm1(p_rows[p_labels, logrms]))
            step[sensor] = dict(band_shift_db=(10 / np.log(10) * delta).round(2).tolist(),
                                rms_ratio=round(float(w_rms / p_rms), 4))
            if iteration < CALIBRATION_ITERATIONS:
                plan[f'{sensor}_band_db'] = np.clip(plan[f'{sensor}_band_db'] + .8 * 10 / np.log(10) * delta,
                                                    -EQ_LIMIT_DB, EQ_LIMIT_DB)
                plan[f'{sensor}_gain'] *= w_rms / p_rms
        history.append(step)
        if verbose:
            print(f'  calibration step {iteration}: {step}', flush=True)
    plan['calibration_history'] = history
    plan['mic_band_db'] = plan['mic_band_db'].tolist()
    plan['acc_band_db'] = plan['acc_band_db'].tolist()
    return plan


def _adapt_and_featurise(args):
    public_dir, record, plan = args
    with zipfile.ZipFile(public_dir / 'signals.zip') as signals:
        x = read_public(signals, record)
    built = _featurise(x, [tuple(i) for i in record['intervals']], plan)
    keep = ('times', 'features', 'labels', 'trainable', 'intervals', 'duration', 'reviewed_background',
            'clean_trainable')
    return dict({k: built[k] for k in keep}, participant=record['participant'], session=record['id'],
                annotations=['swallow'] * len(record['intervals']))


def _raw_featurise(args):
    public_dir, record = args
    from train_live import make_record
    with zipfile.ZipFile(public_dir / 'signals.zip') as signals:
        x = read_public(signals, record)
    t = np.arange(len(x)) / PUBLIC_RATE
    built = make_record(np.column_stack((t, x[:, 0])), np.column_stack((t, x[:, 1:])),
                        [tuple(i) for i in record['intervals']], wearable=False)
    keep = ('times', 'features', 'labels', 'trainable', 'intervals', 'duration', 'reviewed_background',
            'clean_trainable')
    return dict({k: built[k] for k in keep}, participant=record['participant'], session=record['id'],
                annotations=['swallow'] * len(record['intervals']))


def raw_public(public_dir, *, workers=8):
    """Control condition: unadapted public signals, as in the earlier transfer test."""
    path = CACHE / 'raw_unadapted.pkl'
    if path.is_file():
        with path.open('rb') as file:
            return pickle.load(file)
    records = public_manifest(training_participants())
    with ProcessPoolExecutor(workers) as pool:
        built = list(pool.map(_raw_featurise, [(Path(public_dir), r) for r in records], chunksize=4))
    CACHE.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as file:
        pickle.dump(built, file)
    return built


def adapted_public(public_dir, wearable_reference_records, *, participants=None, workers=8, verbose=True):
    """Return (adapted public records, plan) calibrated on the given wearable records (cached)."""
    public_dir = Path(public_dir)
    sessions = sorted({r['session'] for r in wearable_reference_records})
    participants = sorted(participants or training_participants())
    key_source = json.dumps(dict(v=ADAPTATION_VERSION, sessions=sessions, public=participants))
    key = hashlib.sha256(key_source.encode()).hexdigest()[:16]
    path = CACHE / f'{key}.pkl'
    if path.is_file():
        with path.open('rb') as file:
            cached = pickle.load(file)
        return cached['records'], cached['plan']
    records = public_manifest(set(participants))
    folders = [HOME / 'recordings' / 'participants' / s for s in sessions]
    if verbose:
        print(f'Adapting {len(records)} public recordings to {len(sessions)} wearable sessions '
              f'({", ".join(sorted({r["participant"] for r in wearable_reference_records}))})', flush=True)
    imu_rate, gravity = imu_rate_and_gravity(folders)
    plan = calibrate(public_dir, records[::8], wearable_reference_records, imu_rate, gravity)
    with ProcessPoolExecutor(workers) as pool:
        adapted = list(pool.map(_adapt_and_featurise, [(public_dir, r, plan) for r in records], chunksize=4))
    CACHE.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as file:
        pickle.dump(dict(records=adapted, plan=plan, key=key_source), file)
    return adapted, plan
