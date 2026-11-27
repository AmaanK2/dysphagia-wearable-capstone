"""Causal microphone/IMU features and the wearable pilot detector."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.fft import next_fast_len, rfft, rfftfreq
from scipy.special import expit

WINDOW = .5
HOP = .1
THRESHOLD = .65
REFRACTORY = .8
FEATURE_VERSION = 3
FEATURE_COUNT = 29
MIC_COLUMNS = (*range(11), 22, 23, 24)
IMU_COLUMNS = (*range(11, 22), 25, 26, 27)
MODEL_PATH = Path(__file__).with_name('models') / 'wearable.json'
MODEL_IMU_RATE = 904.5   # measured rate of the 833 Hz setting that every current model was trained on


def match_model_rate(values, rate):
    """Block-average faster IMU data (1.66/3.33 kHz settings) down to ~MODEL_IMU_RATE."""
    factor = int(round(rate / MODEL_IMU_RATE))
    if factor < 2:
        return values, rate
    usable = len(values) // factor * factor
    return values[:usable].reshape(-1, factor, values.shape[1]).mean(axis=1), rate / factor


@lru_cache(maxsize=64)
def hann(samples):
    return np.hanning(samples)[:, None]


def window_features(mic, imu, end, *, wearable=True, feature_version=FEATURE_VERSION, history_start=0.):
    """Use only samples timestamped in the trailing half-second."""
    out = []
    extras = []
    envelopes = []
    for samples, bands, min_count in (
        (mic, (20, 60, 150, 350, 800), 850),
        (imu, (5, 15, 40, 100, 300), 365),
    ):
        timestamps = samples[:, 0]
        left, right = timestamps.searchsorted([end - WINDOW, end])
        t, values = timestamps[left:right], samples[left:right, 1:]
        if (len(t) < min_count or t[0] > end - WINDOW + .015
                or t[-1] < end - .015 or np.max(np.diff(t)) > .015):
            return None
        centered = values - values.mean(axis=0)
        rate = full_rate = (len(t) - 1) / (t[-1] - t[0])
        if samples.shape[1] == 4:
            centered, rate = match_model_rate(centered, rate)
        energy = np.sum(centered ** 2, axis=1)
        rms = np.sqrt(energy.mean())
        if wearable and samples.shape[1] == 2 and (rms < 10 or np.max(np.abs(values)) >= 32760):
            return None
        # Keep the raw amplitude/quality features and average PCM in groups
        # before the low-frequency spectrum, which only uses bins below 800 Hz.
        if feature_version >= 2 and samples.shape[1] == 2:
            factor = max(1, round(rate / 5500))
            if factor > 1:
                centered = centered[:len(t) // factor * factor].reshape(-1, factor, 1).mean(axis=1)
                rate /= factor
        n = len(centered)
        fft_size = next_fast_len(n, real=True)
        freq = rfftfreq(fft_size, 1 / rate)
        power = (np.abs(rfft(centered * hann(n), n=fft_size, axis=0)) ** 2).sum(axis=1)
        edges = freq.searchsorted(bands)
        band_power = np.array([power[a:b].sum() for a, b in zip(edges[:-1], edges[1:])])
        shares = band_power / max(band_power.sum(), 1e-20)
        envelope = np.array([block.mean() for block in np.array_split(energy, 5)])
        out.extend([*shares, *(envelope / max(energy.mean(), 1e-20)),
                    np.sqrt(energy.max()) / max(rms, 1e-10), np.log1p(rms)])
        if feature_version >= 3:
            band = power[edges[0]:edges[-1]]
            probability = band / max(band.sum(), 1e-20)
            history_left = max(0, left - len(t))
            if timestamps[history_left] < history_start:
                history_left = timestamps.searchsorted(history_start)
            prior = samples[history_left:left, 1:]
            if samples.shape[1] == 4 and len(prior):
                prior = match_model_rate(prior - prior.mean(axis=0), full_rate)[0]
                enough = len(prior) > len(centered) // 2
            else:
                enough = len(prior) > len(t) // 2
            prior_rms = np.linalg.norm(prior.std(axis=0)) if enough else rms
            extras.extend([probability.max(initial=0.),
                           -np.sum(probability * np.log(probability + 1e-20)) / np.log(max(2, len(band))),
                           np.log((rms + 1e-4) / (prior_rms + 1e-4))])
            envelopes.append(envelope - envelope.mean())
    if feature_version >= 3:
        mic_envelope, imu_envelope = envelopes
        correlation = (mic_envelope @ imu_envelope) / max(
            np.linalg.norm(mic_envelope) * np.linalg.norm(imu_envelope), 1e-20)
        out.extend([*extras, correlation])
    return np.array(out)


def stack_lags(features, lags):
    """Append, to each window's features, the rows from ``lag`` windows (x 0.1 s) earlier.

    ``features`` is (windows, FEATURE_COUNT) in time order; rows without enough history are NaN.
    Column block 0 is the current window, block k the window ``lags[k-1]`` hops earlier.
    """
    if not lags:
        return features
    blocks = [features]
    for lag in lags:
        shifted = np.full_like(features, np.nan)
        shifted[lag:] = features[:-lag]
        blocks.append(shifted)
    return np.hstack(blocks)


def model_lags(data):
    lags = tuple(int(lag) for lag in data.get('lags', ()))
    if any(lag < 1 for lag in lags) or len(set(lags)) != len(lags):
        raise ValueError('Invalid detector context windows.')
    return lags


class Model:
    lags = ()

    def __init__(self, data):
        self.feature_version = data['feature_version']
        if self.feature_version not in (1, 2, FEATURE_VERSION):
            raise ValueError('Incompatible detector model. Retrain the wearable model.')
        self.columns = np.asarray(data['columns'], dtype=int)
        self.mean = np.asarray(data['mean'])
        self.scale = np.asarray(data['scale'])
        self.weights = np.asarray(data['weights'])
        self.bias = float(data['bias'])
        self.threshold = float(data.get('threshold', THRESHOLD))
        self.refractory = float(data.get('refractory', REFRACTORY))
        self.quadratic = data.get('quadratic', False)
        count = self.columns.size
        expected_weights = count + count * (count + 1) // 2 if self.quadratic else count
        if (self.columns.ndim != 1 or not len(self.columns)
                or np.any((self.columns < 0) | (self.columns >= (FEATURE_COUNT if self.feature_version >= 3 else 22)))
                or any(a.shape != self.columns.shape for a in (self.mean, self.scale))
                or self.weights.shape != (expected_weights,)
                or not 0. < self.threshold < 1.
                or not np.isfinite(self.refractory) or self.refractory < 0.
                or not np.isfinite(np.r_[self.mean, self.scale, self.weights, self.bias]).all()
                or np.any(self.scale <= 0)):
            raise ValueError('Invalid detector coefficients.')

    def score(self, features):
        inputs = (features[..., self.columns] - self.mean) / self.scale
        if self.quadratic:
            inputs = quadratic_terms(inputs)
        return expit(inputs @ self.weights + self.bias)


class TreeModel:
    """Gradient-boosted tree ensemble exported from scikit-learn, scored with NumPy only.

    All trees share one flat node table; ``roots`` holds each tree's first node. The
    score is expit(baseline + sum of reached leaf values), like the logistic ``Model``.
    A random forest (``aggregate: mean``) instead stores each leaf's swallow probability
    and the score is the average over trees.
    """

    def __init__(self, data):
        self.feature_version = data['feature_version']
        if self.feature_version != FEATURE_VERSION:
            raise ValueError('Incompatible detector model. Retrain the wearable model.')
        self.columns = np.asarray(data['columns'], dtype=int)
        self.threshold = float(data.get('threshold', THRESHOLD))
        self.refractory = float(data.get('refractory', REFRACTORY))
        trees = data['trees']
        self.feature = np.asarray(trees['feature'], dtype=int)
        self.split = np.asarray(trees['split'], dtype=float)
        self.left = np.asarray(trees['left'], dtype=int)
        self.right = np.asarray(trees['right'], dtype=int)
        self.leaf = np.asarray(trees['leaf'], dtype=bool)
        self.value = np.asarray(trees['value'], dtype=float)
        self.roots = np.asarray(trees['roots'], dtype=int)
        self.depth = int(trees['depth'])
        self.baseline = float(trees['baseline'])
        self.aggregate = trees.get('aggregate', 'sum')
        if self.aggregate not in ('sum', 'mean'):
            raise ValueError('Unknown tree aggregation.')
        # Optional context: features of windows this many hops earlier (see stack_lags).
        self.lags = model_lags(data)
        nodes = len(self.feature)
        if (self.columns.ndim != 1 or not len(self.columns)
                or np.any((self.columns < 0) | (self.columns >= FEATURE_COUNT * (1 + len(self.lags))))
                or any(len(a) != nodes for a in (self.split, self.left, self.right, self.leaf, self.value))
                or np.any((self.feature < 0) | (self.feature >= len(self.columns)))
                or np.any(~self.leaf & ((self.left < 0) | (self.left >= nodes) | (self.right < 0) | (self.right >= nodes)))
                or np.any((self.roots < 0) | (self.roots >= nodes)) or self.depth < 0
                or not 0. < self.threshold < 1. or not np.isfinite(self.refractory) or self.refractory < 0.
                or not np.isfinite(np.r_[self.value, self.baseline]).all()):
            raise ValueError('Invalid detector trees.')

    def score(self, features):
        x = np.atleast_2d(features[..., self.columns])
        rows = np.arange(len(x))[:, None]
        node = np.broadcast_to(self.roots, (len(x), len(self.roots))).copy()
        for _ in range(self.depth):
            go_left = x[rows, self.feature[node]] <= self.split[node]
            node = np.where(self.leaf[node], node, np.where(go_left, self.left[node], self.right[node]))
        leaves = self.value[node]
        scores = leaves.mean(axis=1) if self.aggregate == 'mean' else expit(self.baseline + leaves.sum(axis=1))
        return scores if np.ndim(features) > 1 else scores[0]


def build_model(data):
    return TreeModel(data) if 'trees' in data else Model(data)


@lru_cache(maxsize=4)
def interaction_indexes(count):
    return np.triu_indices(count)


def quadratic_terms(inputs):
    limited = np.clip(inputs, -4., 4.)
    left, right = interaction_indexes(limited.shape[-1])
    return np.concatenate((limited, limited[..., left] * limited[..., right]), axis=-1)


class AgreementModel:
    def __init__(self, data):
        self.joint = build_model(data['joint'])
        self.mic = Model(data['mic'])
        self.imu = Model(data['imu'])
        self.feature_version = self.joint.feature_version
        self.lags = self.joint.lags   # the sensor checks only read the current window's columns
        self.threshold = self.joint.threshold
        self.refractory = self.joint.refractory
        for model, columns in ((self.mic, MIC_COLUMNS), (self.imu, IMU_COLUMNS)):
            if model.feature_version != self.feature_version or not set(model.columns) <= set(columns):
                raise ValueError('Invalid sensor-specific detector.')

    def score(self, features):
        return agreement_scores(self.joint.score(features), self.mic.score(features), self.imu.score(features),
                                self.mic.threshold, self.imu.threshold)


def agreement_scores(joint, mic, imu, mic_threshold, imu_threshold):
    return np.where((mic >= mic_threshold) & (imu >= imu_threshold), joint, 0.)


def load_model(path=None):
    data = json.loads(Path(path or MODEL_PATH).read_text(encoding='utf-8'))
    if 'joint' not in data:
        raise ValueError('Retrain the wearable model to enable microphone and IMU agreement.')
    return AgreementModel(data)


# Trees that also see the two earlier windows (0.2 and 0.4 s back); see experiments/public_domain_gap.
RECOMMENDED_MODEL = 'mixed_context_35_65.json'


def model_label(path):
    """Short, human-readable detector name for the GUI and session metadata."""
    path = Path(path)
    if path.name == MODEL_PATH.name:
        return 'Wearable only (current)'
    try:
        training = json.loads(path.read_text(encoding='utf-8')).get('training', {})
    except (OSError, ValueError):
        training = {}
    if 'random forest' in training.get('model', ''):
        public = round(100 * training.get('public_share', 0))
        return 'Random forest · wearable data only' if public == 0 else f'Random forest {public}/{100 - public} public/wearable'
    if 'public_share' in training:
        public = round(100 * training['public_share'])
        kind = 'trees' if 'tree' in training.get('model', '') else 'logistic'
        if (training.get('settings') or {}).get('lags'):
            kind += ' + context'
        label = f'Mixed {public}/{100 - public} public/wearable · {kind}'
        return label + (' (recommended)' if path.name == RECOMMENDED_MODEL else '')
    return path.stem.replace('_', ' ')


def available_models(folder=None):
    """[(label, path)] for every detector file: the current wearable model first, then mixed models."""
    folder = Path(folder or MODEL_PATH.parent)
    found = []
    if (folder / MODEL_PATH.name).is_file():
        found.append(folder / MODEL_PATH.name)
    mixed = sorted((folder / 'mixed').glob('mixed_*.json'),
                   key=lambda p: (p.name != RECOMMENDED_MODEL, 'logistic' in p.stem, 'optimal' not in p.stem, p.stem))
    found += mixed
    # Other swallow detectors saved next to wearable.json (events.json is the cough/talking model).
    found += sorted(p for p in folder.glob('*.json') if p.name not in (MODEL_PATH.name, 'evaluation.json', 'events.json'))
    return [(model_label(p), p) for p in found]


class Trigger:
    """Emit on the second consecutive positive window, once per active burst."""

    def __init__(self, threshold=THRESHOLD, refractory=REFRACTORY):
        self.threshold = threshold
        self.refractory = refractory
        self.positive = self.negative = 0
        self.active = False
        self.last = -np.inf

    def update(self, seconds, score):
        if score >= self.threshold:
            self.positive += 1
            self.negative = 0
            if self.positive >= 2 and not self.active and seconds - self.last >= self.refractory:
                self.active = True
                self.last = seconds
                return True
        else:
            self.positive = 0
            self.negative += 1
            if self.negative >= 2:
                self.active = False
        return False


# Interval boundaries. A window ending at t covers [t - WINDOW, t]; its centre is used as
# the time of its score. Biases shift the first/last active window centre onto the
# labelled start/end (set from leave-one-participant-out analysis, see evaluate_intervals.py).
INTERVAL_LOW_FRACTION = .7      # a run continues while score >= this fraction of the alert threshold
SWALLOW_START_BIAS = 0.
SWALLOW_END_BIAS = 0.


class IntervalTracker:
    """Turn per-window scores into (start, end, peak score) intervals.

    A run starts when the score reaches ``low`` and ends after ``end_windows`` consecutive
    windows below it. A run is reported only if an alert fired inside it, so every
    interval belongs to exactly one alert, and only once the run has ended.
    """

    def __init__(self, low, end_windows=2, start_bias=0., end_bias=0.):
        self.low = low
        self.end_windows = end_windows
        self.start_bias = start_bias
        self.end_bias = end_bias
        self.first = self.last = None
        self.below = 0
        self.peak = 0.
        self.alerted = False

    def update(self, seconds, score, alerted=False):
        centre = seconds - WINDOW / 2
        done = []
        if score >= self.low:
            if self.first is None:
                self.first, self.peak, self.alerted = centre, 0., False
            self.last = centre
            self.peak = max(self.peak, score)
            self.below = 0
        elif self.first is not None:
            self.below += 1
            if self.below >= self.end_windows:
                done = self.flush()
        if alerted and self.first is not None:
            self.alerted = True
        return done

    def flush(self):
        """Close the current run (e.g. at the end of a recording)."""
        done = []
        if self.first is not None and self.alerted:
            start = self.first + self.start_bias
            end = max(start + HOP, self.last + self.end_bias)
            done.append((round(start, 3), round(end, 3), float(self.peak)))
        self.first = self.last = None
        self.below = 0
        self.alerted = False
        return done


class LiveDetector:
    def __init__(self, model):
        self.model = model
        self.step = round(WINDOW / HOP)
        self.trigger = Trigger(model.threshold, model.refractory)
        self.tracker = IntervalTracker(INTERVAL_LOW_FRACTION * model.threshold,
                                       start_bias=SWALLOW_START_BIAS, end_bias=SWALLOW_END_BIAS)
        self.intervals = []   # completed (start, end, peak); the caller drains this list
        self.signal_valid = None
        self.lags = tuple(getattr(model, 'lags', ()))
        self.history = []     # recent feature rows (None = unusable window), newest last

    def update(self, mic, imu, origin, duration):
        detections = []
        if not len(mic) or not len(imu):
            return detections
        latest = min(mic[-1, 0], imu[-1, 0], origin + duration)
        while origin + self.step * HOP <= latest:
            seconds = self.step * HOP
            features = window_features(mic, imu, origin + seconds,
                                       feature_version=self.model.feature_version, history_start=origin)
            self.signal_valid = features is not None
            if self.lags:
                self.history = (self.history + [features])[-(max(self.lags) + 1):]
                earlier = [self.history[-1 - lag] if lag < len(self.history) else None for lag in self.lags]
                features = None if features is None or any(row is None for row in earlier) else np.concatenate([features, *earlier])
            score = float(self.model.score(features)) if features is not None else 0.
            fired = self.trigger.update(seconds, score)
            if fired:
                detections.append((seconds, score))
            self.intervals += self.tracker.update(seconds, score, fired)
            self.step += 1
        return detections

    def finish(self):
        """Report an interval still open when the recording stops."""
        self.intervals += self.tracker.flush()
