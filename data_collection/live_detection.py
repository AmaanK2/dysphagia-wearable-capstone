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
        energy = np.sum(centered ** 2, axis=1)
        rms = np.sqrt(energy.mean())
        if wearable and samples.shape[1] == 2 and (rms < 10 or np.max(np.abs(values)) >= 32760):
            return None
        rate = (len(t) - 1) / (t[-1] - t[0])
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
            prior_rms = np.linalg.norm(prior.std(axis=0)) if len(prior) > len(t) // 2 else rms
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


class Model:
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


@lru_cache(maxsize=4)
def interaction_indexes(count):
    return np.triu_indices(count)


def quadratic_terms(inputs):
    limited = np.clip(inputs, -4., 4.)
    left, right = interaction_indexes(limited.shape[-1])
    return np.concatenate((limited, limited[..., left] * limited[..., right]), axis=-1)


class AgreementModel:
    def __init__(self, data):
        self.joint = Model(data['joint'])
        self.mic = Model(data['mic'])
        self.imu = Model(data['imu'])
        self.feature_version = self.joint.feature_version
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


def load_model():
    data = json.loads(MODEL_PATH.read_text(encoding='utf-8'))
    if 'joint' not in data:
        raise ValueError('Retrain the wearable model to enable microphone and IMU agreement.')
    return AgreementModel(data)


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


class LiveDetector:
    def __init__(self, model):
        self.model = model
        self.step = round(WINDOW / HOP)
        self.trigger = Trigger(model.threshold, model.refractory)
        self.signal_valid = None

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
            score = float(self.model.score(features)) if self.signal_valid else 0.
            if self.trigger.update(seconds, score):
                detections.append((seconds, score))
            self.step += 1
        return detections
