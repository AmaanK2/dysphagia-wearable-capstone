"""Prevent leakage and numerical failures in the separate activity pipeline."""

import unittest
import numpy as np
import torch
from dysphagia_wearable.activity.data import audit_and_split, normalize_clips
from dysphagia_wearable.activity.features import extract_features
from dysphagia_wearable.activity.models import ActivityCNN


def dataset():
    n = 80
    return {
        "X": np.zeros((n, 4000), dtype=np.float32),
        "y": np.repeat(np.arange(4), 20),
        "hashes": np.array([f"train_{i}" for i in range(n)]),
        "ids": np.array([f"row_{i}" for i in range(n)]),
        "files": [],
    }


class ActivityTests(unittest.TestCase):
    def test_normalization_handles_flat_clips_and_gain_offset(self):
        x = np.sin(np.arange(4000) / 11)[None].astype(np.float32)
        np.testing.assert_allclose(
            normalize_clips(x), normalize_clips(3 * x + 7), atol=2e-6
        )
        self.assertTrue(np.isfinite(extract_features(np.zeros((2, 4000)))).all())
        self.assertTrue((normalize_clips(np.ones((2, 4000))) == 0).all())

    def test_train_test_duplicate_stops_evaluation(self):
        a = dataset()
        with self.assertRaisesRegex(ValueError, "leakage"):
            audit_and_split(a, dataset())

    def test_conflicts_excluded_and_duplicates_grouped(self):
        a, b = dataset(), dataset()
        b["hashes"] = np.array([f"test_{i}" for i in range(80)])
        a["hashes"][1] = a["hashes"][0]
        a["hashes"][20] = a["hashes"][0]  # One hash has conflicting class labels.
        a["hashes"][4] = a["hashes"][3]  # Valid duplicate must stay together.
        tr, va, audit = audit_and_split(a, b)
        self.assertEqual(len(audit["excluded_ambiguous_training_ids"]), 3)
        self.assertFalse({0, 1, 20} & set(tr) | {0, 1, 20} & set(va))
        self.assertFalse(set(a["hashes"][tr]) & set(a["hashes"][va]))
        self.assertEqual(set(tr) | set(va), set(range(80)) - {0, 1, 20})

    def test_conflicting_test_labels_stop_evaluation(self):
        a, b = dataset(), dataset()
        b["hashes"] = np.array([f"test_{i}" for i in range(80)])
        b["hashes"][20] = b["hashes"][0]
        with self.assertRaisesRegex(ValueError, "Test waveform"):
            audit_and_split(a, b)

    def test_cnn_accepts_clip_and_returns_four_logits(self):
        with torch.no_grad():
            output = ActivityCNN().eval()(torch.zeros(2, 1, 4000))
        self.assertEqual(tuple(output.shape), (2, 4))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
