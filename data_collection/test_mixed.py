"""Checks for public-data adaptation and dataset-weighted training (synthetic data only)."""
import unittest

import numpy as np

from adapt_public import IMU_BANDS, MIC_RATE, PUBLIC_RATE, adapt_signal, band_fir
from train_live import make_record
from train_mixed import dataset_rows, fit_mixed
from live_detection import Model

PLAN = dict(imu_rate=904.5, gravity=[-9.0, 1.0, -3.0], mic_gain=3000., acc_gain=2.,
            mic_band_db=[0., 0., 0., 0.], acc_band_db=[0., 0., 0., 0.])


def synthetic_public(seconds=6., seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * PUBLIC_RATE)
    x = rng.normal(0, .01, (n, 4))
    t = np.arange(n) / PUBLIC_RATE
    burst = (t > 2) & (t < 3)
    x[burst, 0] += .05 * np.sin(2 * np.pi * 100 * t[burst])
    x[burst, 1:] += .2 * np.sin(2 * np.pi * 30 * t[burst])[:, None]
    return x


class AdaptationTest(unittest.TestCase):
    def test_output_format_matches_wearable(self):
        mic, imu = adapt_signal(synthetic_public(), PLAN)
        self.assertAlmostEqual(1 / np.diff(mic[:20, 0]).mean(), MIC_RATE, places=3)
        self.assertAlmostEqual(1 / np.diff(imu[:20, 0]).mean(), 904.5, delta=.01)
        self.assertAlmostEqual(mic[-1, 0], 6., delta=.01)
        self.assertAlmostEqual(imu[-1, 0], 6., delta=.01)
        self.assertTrue(np.all(mic[:, 1] == np.round(mic[:, 1])))
        self.assertLessEqual(np.abs(mic[:, 1]).max(), 32768)
        np.testing.assert_allclose(imu[:, 1:].mean(axis=0), PLAN['gravity'], atol=.05)

    def test_equaliser_is_zero_phase_aligned(self):
        x = np.zeros((2000, 4))
        x[1000, :] = 1.
        plan = dict(PLAN, mic_band_db=[6., -3., 0., 3.], acc_band_db=[-6., 3., 0., -3.])
        mic, imu = adapt_signal(x, plan)
        self.assertAlmostEqual(mic[np.argmax(np.abs(mic[:, 1])), 0], .25, delta=.002)
        self.assertAlmostEqual(imu[np.argmax(np.abs(imu[:, 1:] - plan['gravity']).sum(axis=1)), 0], .25, delta=.003)

    def test_band_fir_gain(self):
        fir = band_fir(IMU_BANDS, [0., 6., 0., 0.], 904.5, 511)
        freq = np.fft.rfftfreq(8192, 1 / 904.5)
        response = 20 * np.log10(np.abs(np.fft.rfft(fir, 8192)))
        self.assertAlmostEqual(response[np.argmin(np.abs(freq - np.sqrt(15 * 40)))], 6., delta=.5)
        self.assertAlmostEqual(response[np.argmin(np.abs(freq - np.sqrt(40 * 100)))], 0., delta=.5)

    def test_adapted_record_is_scored_by_detector_features(self):
        mic, imu = adapt_signal(synthetic_public(), PLAN)
        record = make_record(mic, imu, [(2., 3.)], wearable=True)
        self.assertGreater(np.isfinite(record['features']).all(axis=1).sum(), 40)
        self.assertTrue(record['labels'].any())


def toy_records(seed, shift=0., n=4):
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(n):
        labels = np.zeros(200, bool)
        labels[50:60] = labels[120:130] = True
        features = rng.normal(0, 1, (200, 29)) + shift
        features[labels, :3] += 2.
        ok = np.ones(200, bool)
        records.append(dict(features=features, labels=labels, trainable=ok, clean_trainable=ok,
                            reviewed_background=np.zeros(200, bool)))
    return records


class WeightingTest(unittest.TestCase):
    def test_class_balance_within_dataset(self):
        _, labels, _, weights = dataset_rows(toy_records(0), np.arange(29), 'trainable')
        self.assertAlmostEqual(weights[labels].sum(), .5)
        self.assertAlmostEqual(weights[~labels].sum(), .5)

    def test_share_extremes_follow_their_dataset(self):
        wearable, public = toy_records(1), toy_records(2, shift=3.)
        only_w = Model(fit_mixed(wearable, public, public_share=0., quadratic=False, background_weight=0.,
                                 reviewed_weight=0.))
        only_p = Model(fit_mixed(wearable, public, public_share=1., quadratic=False, background_weight=0.,
                                 reviewed_weight=0.))
        # A wearable-only model treats public-shifted background as swallow-like; a
        # public-only model does not.
        public_background = np.vstack([r['features'][~r['labels']] for r in public])
        self.assertGreater(only_w.score(public_background).mean(), only_p.score(public_background).mean())


class TreeExportTest(unittest.TestCase):
    def test_numpy_trees_match_sklearn(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from live_detection import TreeModel, AgreementModel
        from train_mixed import export_trees, fit_mixed
        rng = np.random.default_rng(3)
        x = rng.normal(size=(3000, 29))
        y = (x[:, 0] + x[:, 11] ** 2 + .5 * rng.normal(size=3000)) > 1
        classifier = HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=15, random_state=0).fit(x, y)
        model = TreeModel(dict(feature_version=3, columns=list(range(29)), threshold=.5, refractory=1.,
                               trees=export_trees(classifier)))
        np.testing.assert_allclose(model.score(x[:500]), classifier.predict_proba(x[:500])[:, 1], atol=1e-9)
        self.assertAlmostEqual(float(model.score(x[0])), classifier.predict_proba(x[:1])[0, 1], places=9)
        # The app's loader accepts a tree joint model with logistic sensor checks.
        records = toy_records(4)
        sensors = dict(feature_version=3, columns=[0], mean=[0.], scale=[1.], weights=[1.], bias=0.,
                       threshold=.01, refractory=1.)
        agreement = AgreementModel(dict(joint=dict(feature_version=3, columns=list(range(29)), threshold=.5,
                                                   refractory=1., trees=export_trees(classifier)),
                                        mic=sensors, imu=dict(sensors, columns=[11])))
        self.assertEqual(agreement.score(records[0]['features']).shape, (200,))


class ForestExportTest(unittest.TestCase):
    def test_numpy_forest_matches_sklearn_and_loads_in_app(self):
        import json, tempfile
        from pathlib import Path
        from sklearn.ensemble import RandomForestClassifier
        from live_detection import TreeModel, AgreementModel, available_models, model_label
        from train_mixed import export_forest
        rng = np.random.default_rng(5)
        x = rng.normal(size=(2000, 29))
        y = (x[:, 0] * x[:, 11] + .3 * rng.normal(size=2000)) > .2
        forest = RandomForestClassifier(n_estimators=40, max_depth=8, min_samples_leaf=5, random_state=0).fit(x, y)
        joint = dict(feature_version=3, columns=list(range(29)), threshold=.6, refractory=1., trees=export_forest(forest))
        model = TreeModel(json.loads(json.dumps(joint)))
        np.testing.assert_allclose(model.score(x[:400]), forest.predict_proba(x[:400])[:, 1], atol=1e-4)
        sensors = dict(feature_version=3, columns=[0], mean=[0.], scale=[1.], weights=[1.], bias=0.,
                       threshold=.01, refractory=1.)
        data = dict(joint=joint, mic=sensors, imu=dict(sensors, columns=[11]),
                    training=dict(model='random forest', public_share=0.))
        AgreementModel(data)
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            (folder / 'mixed').mkdir()
            (folder / 'mixed' / 'mixed_forest_wearable_only.json').write_text(json.dumps(data))
            (folder / 'events.json').write_text('{}')
            listed = available_models(folder)
            self.assertEqual([label for label, _ in listed], ['Random forest · wearable data only'])
            self.assertEqual(model_label(listed[0][1]), 'Random forest · wearable data only')


if __name__ == '__main__':
    unittest.main()
