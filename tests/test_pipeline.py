"""Checks for evaluation rules that could materially inflate reported results."""
import unittest
import numpy as np
from dysphagia_wearable.timing.baseline import events, match_events, features


class EvaluationTests(unittest.TestCase):
    def test_one_prediction_cannot_match_two_swallows(self):
        tp, fp, fn, _ = match_events([[0., 4.]], [[0., 1.5], [2., 3.5]])
        self.assertEqual((tp, fp, fn), (1, 0, 1))

    def test_no_predictions_are_misses(self):
        self.assertEqual(match_events([], [[1., 2.]])[:3], (0, 0, 1))

    def test_disjoint_events_are_not_matches(self):
        self.assertEqual(match_events([[0., 1.]], [[2., 3.]])[:3], (0, 1, 1))

    def test_threshold_and_minimum_duration(self):
        self.assertEqual(events([.25, .35, .45], [.9, .1, .1], .5, 1.), [])
        found = events([.25, .35, .45], [.9, .9, .1], .5, 1.)
        np.testing.assert_allclose(found, [[.2, .4]])

    def test_zero_signal_features_are_finite(self):
        X, t = features(np.zeros((4000, 4)))
        self.assertEqual(X.shape[1], 56)
        self.assertTrue(np.isfinite(X).all())
        self.assertAlmostEqual(t[0], .25)


if __name__ == '__main__':
    unittest.main()
