"""IMU event classifier for the live session: swallow, cough, talking, throat clear or nothing.

Each 0.5 s IMU window is projected onto its main vibration direction, so a single-axis
throat sensor and the wearable's three-axis IMU yield comparable signals. Features
describe loudness, spectral shape, voicing (periodicity) and temporal shape. The model
is a multi-class gradient-boosted tree ensemble scored with NumPy only.

Swallow alerts in the app still come from the validated swallow detector
(``live_detection``); this classifier adds cough, talking and throat-clear alerts.
"""
import json
from pathlib import Path

import numpy as np
from scipy.fft import next_fast_len, rfft, rfftfreq

WINDOW = .5
HOP = .1
CLASSES = ('none', 'swallow', 'cough', 'talking', 'throat_clear')
DISPLAY = {'swallow': 'Swallow', 'cough': 'Cough', 'talking': 'Talking', 'throat_clear': 'Throat clear'}
BANDS = (5, 12, 25, 50, 90, 150, 220, 320, 450)
ENVELOPE_BLOCKS = 10
FEATURE_NAMES = (['log_rms'] + [f'band_{a}_{b}' for a, b in zip(BANDS, BANDS[1:])] +
                 ['centroid', 'flatness', 'rolloff', 'entropy', 'peak_freq'] +
                 [f'envelope_{i}' for i in range(ENVELOPE_BLOCKS)] +
                 ['crest', 'kurtosis', 'zero_cross', 'voicing', 'voicing_lag'])
EVENT_MODEL_PATH = Path(__file__).with_name('models') / 'events.json'


def principal_projection(values):
    """Project centred N x k samples onto their direction of greatest variance."""
    centred = values - values.mean(axis=0)
    if centred.shape[1] == 1:
        return centred[:, 0]
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return centred @ vt[0]


def signal_features(signal, rate):
    """Features of one centred 1-D vibration window (m/s^2) sampled at ``rate`` Hz."""
    n = len(signal)
    rms = np.sqrt(np.mean(signal ** 2))
    fft_size = next_fast_len(n, real=True)
    freq = rfftfreq(fft_size, 1 / rate)
    power = np.abs(rfft(signal * np.hanning(n), n=fft_size)) ** 2
    edges = freq.searchsorted(BANDS)
    band = np.array([power[a:b].sum() for a, b in zip(edges[:-1], edges[1:])])
    total = max(band.sum(), 1e-20)
    shares = band / total
    usable = power[edges[0]:edges[-1]]
    f = freq[edges[0]:edges[-1]]
    p = usable / max(usable.sum(), 1e-20)
    centroid = float(p @ f)
    flatness = float(np.exp(np.mean(np.log(usable + 1e-20))) / max(usable.mean(), 1e-20))
    rolloff = float(f[min(len(f) - 1, np.searchsorted(np.cumsum(p), .85))])
    entropy = float(-np.sum(p * np.log(p + 1e-20)) / np.log(max(2, len(p))))
    peak = float(f[np.argmax(usable)])
    energy = signal ** 2
    envelope = np.array([b.mean() for b in np.array_split(energy, ENVELOPE_BLOCKS)])
    envelope = envelope / max(envelope.sum(), 1e-20)
    crest = float(np.abs(signal).max() / max(rms, 1e-12))
    kurtosis = float(np.mean(signal ** 4) / max(np.mean(signal ** 2) ** 2, 1e-20))
    zero_cross = float(np.mean(np.signbit(signal[1:]) != np.signbit(signal[:-1])))
    # Voicing: strongest normalised autocorrelation for pitch periods of 80-300 Hz.
    spectrum = rfft(signal, n=next_fast_len(2 * n, real=True))
    autocorr = np.fft.irfft(np.abs(spectrum) ** 2)[:n]
    autocorr = autocorr / max(autocorr[0], 1e-20)
    low, high = int(rate / 300), min(n - 1, int(rate / 80))
    lag = low + int(np.argmax(autocorr[low:high + 1]))
    return np.r_[np.log(rms + 1e-6), shares, np.log(centroid + 1), flatness, np.log(rolloff + 1), entropy,
                 np.log(peak + 1), envelope, crest, np.log(kurtosis), zero_cross, autocorr[lag], rate / lag / 300]


def window_event_features(imu, end):
    """Trailing 0.5 s IMU window ending at ``end`` (seconds); None if the window is incomplete."""
    timestamps = imu[:, 0]
    left, right = timestamps.searchsorted([end - WINDOW, end])
    t = timestamps[left:right]
    if len(t) < 300 or t[0] > end - WINDOW + .015 or t[-1] < end - .015 or np.max(np.diff(t)) > .015:
        return None
    from live_detection import match_model_rate
    rate = (len(t) - 1) / (t[-1] - t[0])
    values, rate = match_model_rate(imu[left:right, 1:], rate)
    return signal_features(principal_projection(values), rate)


class EventModel:
    """Multi-class boosted trees: one flat node table per class, softmax over classes."""

    def __init__(self, data):
        if tuple(data['classes']) != CLASSES or data['features'] != list(FEATURE_NAMES):
            raise ValueError('Incompatible event model. Retrain with train_events.py.')
        self.thresholds = {k: float(v) for k, v in data['thresholds'].items()}
        self.tables = []
        for trees in data['trees']:
            table = {k: np.asarray(trees[k]) for k in ('feature', 'split', 'left', 'right', 'leaf', 'value', 'roots')}
            table['leaf'] = table['leaf'].astype(bool)
            table['depth'] = int(trees['depth'])
            self.tables.append(table)
        self.baseline = np.asarray(data['baseline'], dtype=float)
        if len(self.tables) != len(CLASSES) or self.baseline.shape != (len(CLASSES),):
            raise ValueError('Invalid event model.')

    def probabilities(self, features):
        x = np.atleast_2d(features)
        rows = np.arange(len(x))[:, None]
        raw = np.empty((len(x), len(CLASSES)))
        for k, table in enumerate(self.tables):
            node = np.broadcast_to(table['roots'], (len(x), len(table['roots']))).copy()
            for _ in range(table['depth']):
                go_left = x[rows, table['feature'][node]] <= table['split'][node]
                node = np.where(table['leaf'][node], node, np.where(go_left, table['left'][node], table['right'][node]))
            raw[:, k] = self.baseline[k] + table['value'][node].sum(axis=1)
        raw -= raw.max(axis=1, keepdims=True)
        probs = np.exp(raw)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs if np.ndim(features) > 1 else probs[0]


def load_event_model(path=None):
    return EventModel(json.loads(Path(path or EVENT_MODEL_PATH).read_text(encoding='utf-8')))


class EventTrigger:
    """Turn per-window class probabilities into alerts.

    Brief events (cough, throat clear) alert on two consecutive windows above their
    threshold, but not during talking, where they are mostly confusions. Talking alerts
    once per bout when most of the last eight windows (0.8 s) are talking, and a new
    bout needs ``talk_gap`` seconds without talking.
    """

    def __init__(self, thresholds, refractory=1., talk_gap=3.):
        self.thresholds = thresholds
        self.refractory = refractory
        self.talk_gap = talk_gap
        self.streak = {k: 0 for k in ('cough', 'throat_clear')}
        self.last = {k: -np.inf for k in ('cough', 'throat_clear')}
        self.talk_history = []
        self.talking = False
        self.last_talk = -np.inf

    def update(self, seconds, probs):
        alerts = []
        p = probs[CLASSES.index('talking')]
        self.talk_history = (self.talk_history + [p >= self.thresholds['talking']])[-8:]
        if p >= self.thresholds['talking']:
            self.last_talk = seconds
        if not self.talking and sum(self.talk_history) >= 5:
            self.talking = True
            alerts.append(('talking', float(p)))
        elif self.talking and seconds - self.last_talk >= self.talk_gap:
            self.talking = False
        speaking = self.talking or sum(self.talk_history) >= 3
        for name in ('cough', 'throat_clear'):
            p = probs[CLASSES.index(name)]
            if p >= self.thresholds[name] and p == probs.max() and not speaking:
                self.streak[name] += 1
                if self.streak[name] == 2 and seconds - self.last[name] >= self.refractory:
                    self.last[name] = seconds
                    alerts.append((name, float(p)))
            else:
                self.streak[name] = 0
        return alerts


class EventIntervals:
    """Start/end intervals for alerted events, built from per-window class probabilities.

    Cough and throat clear: windows where the class leads and reaches 70% of its
    threshold, ending after two windows without it. Talking: windows above the talking
    threshold, ending after the same 3 s pause that ends a talking bout.
    """

    def __init__(self, thresholds, talk_gap=3.):
        from live_detection import INTERVAL_LOW_FRACTION, IntervalTracker
        self.trackers = {name: IntervalTracker(INTERVAL_LOW_FRACTION * thresholds[name])
                         for name in ('cough', 'throat_clear')}
        self.trackers['talking'] = IntervalTracker(thresholds['talking'], end_windows=round(talk_gap / HOP))

    def update(self, seconds, probs, alerted=()):
        done = []
        top = int(np.argmax(probs))
        for name, tracker in self.trackers.items():
            k = CLASSES.index(name)
            score = probs[k] if name == 'talking' or k == top else 0.
            done += [(name, *interval) for interval in tracker.update(seconds, score, name in alerted)]
        return done

    def flush(self):
        return [(name, *interval) for name, tracker in self.trackers.items() for interval in tracker.flush()]


class LiveEventDetector:
    """Score the trailing IMU window every 0.1 s, like ``live_detection.LiveDetector``."""

    def __init__(self, model):
        self.model = model
        self.trigger = EventTrigger(model.thresholds)
        self.tracker = EventIntervals(model.thresholds, self.trigger.talk_gap)
        self.intervals = []   # completed (event, start, end, peak); the caller drains this list
        self.step = round(WINDOW / HOP)
        self.latest = None

    def update(self, imu, origin, duration):
        alerts = []
        if not len(imu):
            return alerts
        latest = min(imu[-1, 0], origin + duration)
        while origin + self.step * HOP <= latest:
            seconds = self.step * HOP
            features = window_event_features(imu, origin + seconds)
            if features is not None:
                self.latest = self.model.probabilities(features)
                fired = self.trigger.update(seconds, self.latest)
                alerts += [(seconds, name, p) for name, p in fired]
                self.intervals += self.tracker.update(seconds, self.latest, {name for name, _ in fired})
            self.step += 1
        return alerts

    def finish(self):
        self.intervals += self.tracker.flush()
