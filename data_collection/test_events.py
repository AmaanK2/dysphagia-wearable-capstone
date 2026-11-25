"""Checks for the live event classifier (synthetic data only)."""
import unittest

import numpy as np

from events import CLASSES, FEATURE_NAMES, EventModel, EventTrigger, principal_projection, signal_features
from train_events import fit


class FeatureTest(unittest.TestCase):
    def test_rotated_three_axis_matches_single_axis(self):
        rate = 904.5
        t = np.arange(452) / rate
        signal = np.sin(2 * np.pi * 120 * t) * np.hanning(452) * .3
        rotation = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))[0]
        three_axis = np.outer(signal, rotation[0]) + 9.81 * rotation[1]
        single = signal_features(signal - signal.mean(), rate)
        projected = signal_features(principal_projection(three_axis), rate)
        np.testing.assert_allclose(single, projected, atol=1e-6)
        self.assertEqual(len(single), len(FEATURE_NAMES))

    def test_voiced_signal_has_higher_voicing_than_noise(self):
        rate = 904.5
        t = np.arange(452) / rate
        voiced = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in (1, 2, 3))
        noise = np.random.default_rng(1).normal(size=452)
        voicing = FEATURE_NAMES.index('voicing')
        self.assertGreater(signal_features(voiced, rate)[voicing], signal_features(noise, rate)[voicing] + .3)


class ModelTest(unittest.TestCase):
    def test_numpy_multiclass_trees_match_sklearn_argmax(self):
        rng = np.random.default_rng(2)
        parts = []
        for shift in (0., .5):
            x = rng.normal(size=(1500, len(FEATURE_NAMES)))
            y = rng.integers(0, len(CLASSES), 1500)
            x[np.arange(1500), y] += 3 + shift
            parts.append((x, y))
        model = EventModel(fit(parts))
        x, y = parts[0]
        probs = model.probabilities(x)
        np.testing.assert_allclose(probs.sum(axis=1), 1)
        self.assertGreater((probs.argmax(axis=1) == y).mean(), .9)
        np.testing.assert_allclose(model.probabilities(x[0]), probs[0])


class TriggerTest(unittest.TestCase):
    def probs(self, name, p=.9):
        out = np.full(len(CLASSES), (1 - p) / (len(CLASSES) - 1))
        out[CLASSES.index(name)] = p
        return out

    def test_cough_needs_two_windows_and_respects_refractory(self):
        trigger = EventTrigger(dict(cough=.5, throat_clear=.5, talking=.5), refractory=1.)
        alerts = [trigger.update(t / 10, self.probs('cough' if t in (5, 6, 7, 9, 10, 20, 21) else 'none'))
                  for t in range(30)]
        fired = [t for t, a in enumerate(alerts) if a]
        self.assertEqual(fired, [6, 21])

    def test_talking_alerts_once_per_bout(self):
        trigger = EventTrigger(dict(cough=.5, throat_clear=.5, talking=.5))
        names = ['talking'] * 30 + ['none'] * 25 + ['talking'] * 10
        fired = [t for t, name in enumerate(names) if trigger.update(t / 10, self.probs(name))]
        self.assertEqual(fired, [4, 59])


if __name__ == '__main__':
    unittest.main()
