"""Interpretable clip features in time and frequency, using the verified 6.4kHz rate."""

import numpy as np
from .data import normalize_clips, FS


def extract_features(x):
    raw = np.asarray(x, dtype=np.float32)
    v = normalize_clips(raw)
    n = v.shape[1]
    spectrum = abs(np.fft.rfft(v * np.hanning(n), axis=1)) ** 2
    power = spectrum / np.maximum(spectrum.sum(axis=1, keepdims=True), 1e-24)
    freq = np.fft.rfftfreq(n, 1 / FS)
    centroid = (power * freq).sum(axis=1)
    columns = [
        np.log10(np.sqrt((raw**2).mean(axis=1)) + 1e-12),
        abs(v).max(axis=1),
        (v**3).mean(axis=1),
        (v**4).mean(axis=1),
        np.diff(np.signbit(v), axis=1).mean(axis=1),
        centroid,
        np.sqrt((power * (freq - centroid[:, None]) ** 2).sum(axis=1)),
        -(power * np.log(power + 1e-24)).sum(axis=1),
        freq[spectrum.argmax(axis=1)],
    ]
    for lo, hi in [(0, 100), (100, 300), (300, 800), (800, 1600), (1600, 3201)]:
        columns.append(power[:, (freq >= lo) & (freq < hi)].sum(axis=1))
    blocks = v.reshape(len(v), 8, n // 8)
    columns.extend(list((blocks**2).mean(axis=-1).T))
    return np.column_stack(columns).astype(np.float32)
