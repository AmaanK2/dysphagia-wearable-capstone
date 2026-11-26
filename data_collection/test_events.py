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
        # A 2.5 s pause continues the bout; a 3.5 s pause starts a new one.
        names = ['talking'] * 30 + ['none'] * 25 + ['talking'] * 10 + ['none'] * 35 + ['talking'] * 10
        fired = [t for t, name in enumerate(names) if trigger.update(t / 10, self.probs(name))]
        self.assertEqual(fired, [4, 104])

    def test_cough_is_not_reported_while_talking(self):
        trigger = EventTrigger(dict(cough=.5, throat_clear=.5, talking=.5))
        names = ['talking'] * 20 + ['cough'] * 3 + ['talking'] * 5 + ['none'] * 40 + ['cough'] * 3
        alerts = [(t, a) for t, name in enumerate(names) for a in trigger.update(t / 10, self.probs(name))]
        self.assertEqual([(t, a[0]) for t, a in alerts], [(4, 'talking'), (69, 'cough')])


class IntervalTest(unittest.TestCase):
    def test_tracker_reports_alerted_runs_with_window_centres(self):
        from live_detection import IntervalTracker
        tracker = IntervalTracker(.5)
        scores = [0, .6, .9, .9, .7, .2, .1, 0, .6, .4, .1, .1, 0]
        alerted = {3}
        out = []
        for i, score in enumerate(scores):
            out += tracker.update(1 + i / 10, score, i in alerted)
        # Run 1 spans windows 1-4 (alert inside), run 2 (window 8) has no alert.
        self.assertEqual(len(out), 1)
        start, end, peak = out[0]
        self.assertAlmostEqual(start, 1.1 - .25)
        self.assertAlmostEqual(end, 1.4 - .25)
        self.assertEqual(peak, .9)

    def test_open_run_is_flushed(self):
        from live_detection import IntervalTracker
        tracker = IntervalTracker(.5)
        for i in range(5):
            tracker.update(i / 10, .9, i == 2)
        self.assertEqual(len(tracker.flush()), 1)
        self.assertEqual(tracker.flush(), [])

    def test_talking_interval_spans_short_pauses(self):
        from events import EventIntervals
        intervals = EventIntervals(dict(cough=.5, throat_clear=.5, talking=.5))
        talk = np.array([.05, .05, .05, .8, .05])
        quiet = np.array([.8, .05, .05, .05, .05])
        out = []
        names = ['talk'] * 20 + ['quiet'] * 15 + ['talk'] * 10 + ['quiet'] * 40
        for i, name in enumerate(names):
            out += intervals.update(i / 10, talk if name == 'talk' else quiet, {'talking'} if i == 5 else set())
        self.assertEqual(len(out), 1)
        event, start, end, _ = out[0]
        self.assertEqual(event, 'talking')
        self.assertAlmostEqual(start, -.25)
        self.assertAlmostEqual(end, 4.4 - .25)


class SampleRateTest(unittest.TestCase):
    def make(self, rate, seconds=3., seed=0):
        t = np.arange(int(seconds * rate)) / rate
        rng = np.random.default_rng(seed)
        burst = np.exp(-((t - 2.2) / .12) ** 2)
        signal = burst * (np.sin(2 * np.pi * 60 * t) + .5 * np.sin(2 * np.pi * 180 * t)) * .3
        # Same noise density (per root hertz) at both rates, as for the real sensor.
        noise = .002 * np.sqrt(rate / 904.5) * rng.normal(size=(len(t), 3))
        imu = np.column_stack((t, signal + noise[:, 0], .5 * signal + noise[:, 1], -9.81 + .2 * signal + noise[:, 2]))
        mic = np.column_stack((np.arange(int(seconds * 44100)) / 44100,
                               200 * np.sin(2 * np.pi * 120 * np.arange(int(seconds * 44100)) / 44100)))
        return mic, imu

    def test_features_at_1666_setting_match_833_setting(self):
        from live_detection import window_features
        from events import window_event_features
        mic, slow = self.make(904.5)
        _, fast = self.make(1809.)
        a, b = window_features(mic, slow, 2.45), window_features(mic, fast, 2.45)
        imu_columns = list(range(11, 22)) + [25, 26, 27]
        np.testing.assert_allclose(a[imu_columns], b[imu_columns], atol=.06)
        e1, e2 = window_event_features(slow, 2.45), window_event_features(fast, 2.45)
        np.testing.assert_allclose(e1, e2, atol=.08)


if __name__ == '__main__':
    unittest.main()
