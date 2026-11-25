"""Run from this folder: python -m unittest -v test_collection.py"""

import csv
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
try:
    from PyQt6 import QtWidgets as W, QtGui, QtCore, QtTest
except ImportError:
    from PySide6 import QtWidgets as W, QtGui, QtCore, QtTest

from session import EVENT_FIELDS, Protocol, Session, next_session, safe_name, save_events
from device import Acquisition, Demo, LivePreview, peak_envelope
from app import MainWindow, Setup, STYLE
from live_detection import FEATURE_COUNT, FEATURE_VERSION, AgreementModel, Model, Trigger
from train_live import make_record


def read(path):
    with path.open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


class DetectionTests(unittest.TestCase):
    def test_detector_requires_mic_and_imu_in_the_same_window(self):
        def detector(column):
            return dict(feature_version=FEATURE_VERSION, columns=[column], mean=[0.],
                        scale=[1.], weights=[2.], bias=0.)

        model = AgreementModel(dict(joint=dict(detector(0), weights=[0.], bias=4.),
                                    mic=detector(0), imu=detector(11)))
        windows = np.zeros((4, FEATURE_COUNT))
        windows[:, [0, 11]] = [[2, -2], [-2, 2], [-2, -2], [2, 2]]
        np.testing.assert_array_equal(model.score(windows) >= model.threshold, [False, False, False, True])

    def test_background_windows_exclude_swallow_boundaries_and_unreviewed_markers(self):
        mic = np.array([[0., 0.], [8., 0.]])
        imu = np.array([[0., 0., 0., 0.], [8., 0., 0., 0.]])
        with patch('train_live.window_features', return_value=np.zeros(FEATURE_COUNT)):
            record = make_record(mic, imu, [(1., 1.5)], uncertain=[6.],
                                 reviewed_background=[(3., 4.)], ambiguous=[(4.1, 4.4)])
        indexes = {round(t, 1): i for i, t in enumerate(record['times'])}
        self.assertTrue(record['labels'][indexes[1.3]])
        self.assertFalse(record['clean_trainable'][indexes[1.1]])
        self.assertTrue(record['reviewed_background'][indexes[3.5]])
        self.assertTrue(record['clean_trainable'][indexes[2.5]])
        self.assertFalse(record['clean_trainable'][indexes[4.2]])
        self.assertFalse(record['clean_trainable'][indexes[6.]])

    def test_model_cooldown_makes_a_second_burst_wait(self):
        trigger = Trigger(.65, 1.2)
        times_and_scores = [(0., .9), (.1, .9), (.2, .2), (.3, .2),
                            (1., .9), (1.2, .9), (1.4, .9)]
        self.assertEqual([t for t, score in times_and_scores if trigger.update(t, score)], [.1, 1.4])


class StorageTests(unittest.TestCase):
    def test_numbering_shared_across_activities_and_days(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(next_session(root, 'P01', 'dry_swallow').name, 'S01_dry_swallow')
            self.assertEqual(next_session(root, 'P01', 'talking').name, 'S02_talking')
            self.assertEqual(next_session(root, 'P01', 'mixed').name, 'S03_mixed')
            self.assertEqual(next_session(root, 'P02', 'talking').name, 'S01_talking')

    def test_codes_cannot_escape_output_or_use_reserved_names(self):
        for name in ['../P01', '', 'P/01', 'CON', 'LPT1', '.hidden', 'a' * 65]:
            with self.assertRaises(ValueError, msg=name):
                safe_name(name)

    def test_raw_values_shared_origin_duration_and_marker_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            session = Session(Path(root), 'dry_swallow', 1, 10, (1000, 100), 'test')
            session.sample('mic', 4000, 9.998, (-32768, -1, 123, 32767))
            session.sample('imu', 5, 10.001, (1.25, -2.5, 9.81))
            session.sample('imu', 6, 11.1, (0, 0, 0))
            session.marker(.25)
            session.close()
            mic, imu = read(Path(root) / 'Mic.csv'), read(Path(root) / 'IMU.csv')
            self.assertEqual([int(x['mic_raw']) for x in mic], [123, 32767])
            self.assertAlmostEqual(float(mic[0]['time_s']), 0)
            self.assertAlmostEqual(float(imu[0]['time_s']), .001)
            self.assertEqual(len(imu), 1)
            markers = read(Path(root) / 'Events.csv')
            labels = markers + [dict(record_type='label', activity='dry_swallow', time_s='', start_s='.1', end_s='.3', notes='[uncertain] unclear peak')]
            save_events(Path(root) / 'Events.csv', labels)
            self.assertEqual(read(Path(root) / 'Events.csv')[0], markers[0])
            self.assertFalse((Path(root) / 'Events.csv.tmp').exists())

    def test_partial_capture_metadata_survives(self):
        with tempfile.TemporaryDirectory() as root:
            session = Session(Path(root), 'talking', 1, 0, (44100, 833), 'COM5')
            session.sample('imu', 1, .1, (1, 2, 3))
            session.close('incomplete', 'USB disconnected')
            session.close()  # Idempotent cleanup.
            metadata = json.loads((Path(root) / 'Session.json').read_text())
            self.assertEqual(metadata['status'], 'incomplete')
            self.assertEqual(metadata['samples']['imu'], 1)

    def test_short_tail_cannot_be_reported_as_complete(self):
        with tempfile.TemporaryDirectory() as root:
            session = Session(Path(root), 'talking', 2, 0, (44100, 833), 'test')
            session.sample('mic', 0, 1, (1,) * 128)
            session.sample('imu', 0, 1.99, (1, 2, 3))
            session.close()
            self.assertEqual(session.metadata['status'], 'incomplete')


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.p = Protocol()
        self.p.parse(b'S,1000000,44100,833,m/s2,V1')

    def test_device_clock_and_pcm(self):
        packet = self.p.parse(('A,0,1010000,' + ','.join(['-123'] * 128)).encode())
        self.assertEqual(packet[:3], ('mic', 0, .01))
        self.assertEqual(packet[3], (-123.0,) * 128)
        self.assertEqual(self.p.parse(b'I,0,1010000,1,2,9.81')[2], .01)

    def test_microseconds_rollover(self):
        self.p.parse(b'S,4294967000,44100,833,m/s2,V1')
        self.assertAlmostEqual(self.p.parse(b'I,0,704,1,2,3')[2], .001)

    def test_sequence_gap_is_counted(self):
        self.p.parse(b'I,0,1001000,1,2,3')
        self.p.parse(b'I,2,1003000,1,2,3')
        self.assertEqual(self.p.gaps, 1)

    def test_malformed_nonfinite_duplicate_and_wrong_firmware_rejected(self):
        for packet in [b'I,0,1001000,nan,2,3', b'I,0,1001000,1,2', b'S,1,44100,833', b'E,audio overrun']:
            with self.assertRaises(ValueError):
                self.p.parse(packet)
        self.p.parse(b'I,0,1001000,1,2,3')
        with self.assertRaises(ValueError):
            self.p.parse(b'I,0,1002000,1,2,3')

    def test_device_clock_reset_rejected(self):
        self.p.parse(b'I,0,1001000,1,2,3')
        with self.assertRaises(ValueError):
            self.p.parse(b'I,1,999999,1,2,3')


class PreviewTests(unittest.TestCase):
    def test_envelope_is_bounded_and_preserves_each_axis_extrema(self):
        values = np.zeros((44100, 4))
        values[:, 0] = np.arange(len(values)) / 44100
        for axis in range(1, 4):
            values[100 * axis, axis] = 20 + axis
            values[500 * axis, axis] = -30 - axis
        reduced = peak_envelope(values)
        self.assertLessEqual(len(reduced), 1800)
        np.testing.assert_array_equal(reduced[:, 1:].max(axis=0), values[:, 1:].max(axis=0))
        np.testing.assert_array_equal(reduced[:, 1:].min(axis=0), values[:, 1:].min(axis=0))
        self.assertTrue(np.all(np.diff(reduced[:, 0]) > 0))

    def test_preview_window_is_bounded_without_modifying_input(self):
        preview = LivePreview()
        audio = np.arange(44100) % 200
        original = audio.copy()
        for second in range(15):
            frame = preview.update([('mic', second * 44100, second, audio),
                                    ('imu', second, second, (1, 2, 9.81))], (44100, 833))
        np.testing.assert_array_equal(audio, original)
        for arrays in (frame['raw'], frame['filtered']):
            for array in arrays:
                self.assertLessEqual(len(array), 1800)
                self.assertLessEqual(np.ptp(array[:, 0]), 10)
        self.assertLessEqual(len(preview.buffers['mic']), 11)

    def test_gui_stall_cannot_queue_old_preview_frames(self):
        with tempfile.TemporaryDirectory() as root:
            worker = Acquisition('DEMO', root, True)
            worker.start()
            try:
                # Intentionally do not pump Qt events while acquisition runs.
                deadline = time.perf_counter() + 2
                while not worker.ready and time.perf_counter() < deadline:
                    time.sleep(.01)
                self.assertTrue(worker.ready)
                time.sleep(.4)
                first = worker.take_preview()
                self.assertIsNotNone(first)
                self.assertIsNone(worker.take_preview())
                time.sleep(.4)
                newest = worker.take_preview()
                self.assertGreater(newest['raw'][0][-1, 0], first['raw'][0][-1, 0])
                self.assertLessEqual(len(newest['raw'][0]), 1800)
            finally:
                worker.command('quit')
                self.assertTrue(worker.wait(4000))


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = W.QApplication.instance() or W.QApplication([])
        QtGui.QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
        cls.app.setStyle('Fusion')
        cls.app.setFont(QtGui.QFont('Segoe UI', 10))
        cls.app.setStyleSheet(STYLE)

    def wait_for(self, condition, timeout=6):
        deadline = time.perf_counter() + timeout
        while not condition() and time.perf_counter() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.app.processEvents()
        self.assertTrue(condition(), 'Timed out waiting for Qt workflow')

    def test_demo_full_recording_and_review_workflow(self):
        with tempfile.TemporaryDirectory() as root, patch('app.message') as warning:
            window = MainWindow()
            window.show()
            window.demo.setChecked(True)
            window.connect_device()
            window.worker.root = Path(root)
            self.wait_for(lambda: window.ready)
            self.assertTrue(window.start_button.isEnabled())
            window.worker.command('start', ('P01', 'mixed', 1.5))
            self.wait_for(lambda: window.recording)
            self.wait_for(lambda: time.perf_counter() - window.started > .35)
            QtTest.QTest.keyClick(window, QtCore.Qt.Key.Key_Space)
            self.wait_for(lambda: window.latest_session is not None)
            folder = window.latest_session
            self.assertEqual(folder.name, 'S01_mixed')
            mic = read(folder / 'Mic.csv')
            imu = read(folder / 'IMU.csv')
            self.assertGreater(len(mic), 60000)
            self.assertGreater(len(imu), 1100)
            self.assertLessEqual(float(mic[-1]['time_s']), 1.5)
            self.assertLess(abs(float(mic[-1]['time_s']) - float(imu[-1]['time_s'])), .02)
            self.assertEqual(len(read(folder / 'Events.csv')), 1)
            self.assertFalse(window.recording)
            window.review_last()
            self.wait_for(lambda: window.review.folder is not None)
            review = window.review
            self.assertGreater(review.start.lineEdit().width(), 100)
            self.assertGreater(review.end.lineEdit().width(), 100)
            review.table.setCurrentRow(0)
            review.start.setValue(.1)
            review.end.setValue(.7)
            review.activity.setCurrentText('dry_swallow')
            review.save()
            events = read(folder / 'Events.csv')
            self.assertEqual([x['record_type'] for x in events], ['marker', 'label'])
            review.start.setValue(.2)
            review.uncertain.setChecked(True)
            review.notes.setText('Boundary unclear')
            review.save()
            events = read(folder / 'Events.csv')
            self.assertEqual(len(events), 2)
            self.assertTrue(events[1]['notes'].startswith('[uncertain]'))
            self.assertAlmostEqual(float(events[1]['start_s']), .2)
            review.delete()
            self.assertEqual([x['record_type'] for x in read(folder / 'Events.csv')], ['marker'])
            if os.environ.get('COLLECTION_SCREENSHOT'):
                window.tabs.setCurrentIndex(0)
                window.grab().save(os.environ['COLLECTION_SCREENSHOT'])
            window.close()
            self.app.processEvents()
            warning.assert_not_called()

    def test_manual_stop_saves_shorter_recording(self):
        with tempfile.TemporaryDirectory() as root:
            worker = Acquisition('DEMO', root, True)
            errors, saved = [], []
            worker.failed.connect(errors.append)
            worker.saved.connect(lambda folder, state: saved.append((folder, state)))
            worker.start()
            try:
                self.wait_for(lambda: worker.ready)
                worker.command('start', ('P02', 'talking', 60))
                self.wait_for(lambda: worker.session is not None)
                self.wait_for(lambda: time.perf_counter() - worker.started > .4)
                worker.command('stop')
                self.wait_for(lambda: bool(saved))
                self.assertEqual(saved[0][1], 'stopped')
                metadata = json.loads((Path(saved[0][0]) / 'Session.json').read_text())
                self.assertLess(metadata['saved_duration_s'], 1)
                self.assertEqual(errors, [])
            finally:
                worker.command('quit')
                self.assertTrue(worker.wait(4000))

    def test_setup_custom_and_presets(self):
        window = MainWindow()
        dialog = Setup(window)
        dialog.activity.setCurrentText('Custom')
        dialog.custom.setText('throat clear')
        dialog.seconds.setValue(42.5)
        dialog.validate()
        self.assertEqual(dialog.task, 'throat_clear')
        self.assertEqual(dialog.seconds.value(), 42.5)
        window.close()

    def test_live_microphone_range_tracks_visible_values(self):
        window = MainWindow()
        window.origin = 20.
        imu = np.empty((0, 4))
        try:
            for graph in (window.graph, window.live_graph):
                for samples, expected in (
                    (np.array([[19., 30000.], [20., -1000.], [21., 1000.]]), [-1100., 1100.]),
                    (np.array([[22., -10.], [23., 10.]]), [-11., 11.]),
                    (np.array([[24., 400.], [25., 400.]]), [399., 401.]),
                ):
                    frame = dict(raw=(samples, imu), filtered=(samples, imu), clipping=False)
                    window.receive(frame, graph)
                    np.testing.assert_allclose(graph.plots[0].viewRange()[1], expected)
                    np.testing.assert_array_equal(graph.curves[0][0].yData, samples[:, 1])
        finally:
            window.close()

    def test_live_session_displays_and_saves_detector_alert(self):
        source = Demo()
        model = Model(dict(feature_version=FEATURE_VERSION, columns=[0], mean=[0.], scale=[1.],
                           weights=[0.], bias=4.))
        with (tempfile.TemporaryDirectory() as root,
              patch('device.serial.Serial') as serial, patch('device.load_model', return_value=model),
              patch('app.ports', return_value=[('TEST', 'Test USB', True)])):
            port = serial.return_value
            port.in_waiting = 1000
            port.read.side_effect = lambda count: source.read()
            window = MainWindow()
            window.show()
            try:
                window.connect_device()
                window.worker.root = Path(root)
                self.wait_for(lambda: window.ready)
                window.tabs.setCurrentWidget(window.live)
                self.wait_for(lambda: window.live_log.count() > 0)
                self.wait_for(lambda: window.live_graph.curves[0][0].xData is not None)
                self.assertTrue(window.live_start.isEnabled())
                self.assertFalse(window.live_mark.isEnabled())
                self.assertIn('Not recording', window.live_name.text())
                self.assertEqual(list(Path(root).iterdir()), [])
                window.worker.command('start', ('P01', 'mixed', 1.3))
                self.wait_for(lambda: window.recording)
                self.assertEqual(window.live_log.count(), 0)
                self.wait_for(lambda: window.live_log.count() > 0)
                self.assertIs(window.tabs.currentWidget(), window.live)
                self.assertTrue(window.recording)
                self.assertIn('Swallow detected', window.live_alert.text())
                self.assertEqual(len(window.live_graph.overlays), 2)
                self.wait_for(lambda: window.latest_session is not None)
                folder = window.latest_session
                self.assertEqual(read(folder / 'Events.csv'), [])
                detections = read(folder / 'Detections.csv')
                self.assertEqual(len(detections), 1)
                self.assertEqual(detections[0]['event'], 'swallow')
                self.assertGreaterEqual(float(detections[0]['time_s']), .6)
                self.wait_for(lambda: window.live_log.count() > 0)
                self.assertFalse(window.recording)
                self.assertIn('Not recording', window.live_name.text())
                self.assertEqual(read(folder / 'Detections.csv'), detections)
            finally:
                window.close()
                self.app.processEvents()

    def test_fragmented_usb_and_disconnect_preserve_partial_session(self):
        class FakeSerial:
            def __init__(self, *args, **kwargs):
                self.source = Demo()
                self.remaining = b''
                self.in_waiting = 800

            def read(self, count):
                if time.perf_counter() - self.source.start > 1:
                    raise OSError('USB cable removed')
                if not self.remaining:
                    self.remaining = self.source.read()
                part, self.remaining = self.remaining[:713], self.remaining[713:]
                return part

            def reset_input_buffer(self):
                pass

            def write(self, data):
                return len(data)

            def flush(self):
                pass

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as root, patch('device.serial.Serial', FakeSerial):
            worker = Acquisition('COM_TEST', root)
            errors, saved = [], []
            worker.failed.connect(errors.append)
            worker.saved.connect(lambda folder, state: saved.append((folder, state)))
            worker.start()
            try:
                self.wait_for(lambda: worker.ready)
                worker.command('start', ('P03', 'dry_swallow', 120))
                self.wait_for(lambda: worker.session is not None)
                worker.command('marker')
                self.wait_for(lambda: bool(saved))
                self.wait_for(lambda: bool(errors))
                self.assertEqual(saved[0][1], 'incomplete')
                self.assertIn('USB cable removed', errors[0])
                folder = Path(saved[0][0])
                self.assertGreater(len(read(folder / 'Mic.csv')), 1000)
                self.assertGreater(len(read(folder / 'IMU.csv')), 10)
                self.assertEqual(len(read(folder / 'Events.csv')), 1)
            finally:
                worker.command('quit')
                self.assertTrue(worker.wait(4000))


if __name__ == '__main__':
    unittest.main()
