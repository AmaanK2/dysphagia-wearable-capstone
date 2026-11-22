"""Recorder interval and output checks without connecting a sensor."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import record_teensy as recorder


class IntervalMarkerTests(unittest.TestCase):
    def test_v1_packets_parse_without_dropping_sensor_samples(self):
        with TemporaryDirectory() as temp:
            raw = Path(temp) / "recording.txt"
            raw.write_text(
                "S,1000000,44100.000,833.000,m/s2,V1\n"
                + "A,0,1000100," + ",".join(str(i) for i in range(128)) + "\n"
                + "I,0,1000200,1.25,2.5,3.75\n")
            mic, imu, rate, imu_times = recorder.parse_recording(raw)
            self.assertEqual(mic, list(range(128)))
            self.assertEqual(imu, [(1.25, 2.5, 3.75)])
            self.assertEqual(rate, 44100)
            self.assertAlmostEqual(imu_times[0], 0.0002)

    def test_port_detection_accepts_teensy_on_windows(self):
        ports = [SimpleNamespace(device="COM7", vid=0x16C0,
                                 description="USB Serial", manufacturer="PJRC")]
        with patch("serial.tools.list_ports.comports", return_value=ports):
            self.assertEqual(recorder.find_teensy_port(), "COM7")

    def test_two_presses_complete_an_interval_and_open_one_is_preserved(self):
        marks = recorder.IntervalMarkers(3)
        self.assertIsNone(marks.toggle(100.1))
        marks.start(100.0)
        self.assertEqual(marks.toggle(100.25), "started")
        self.assertEqual(marks.toggle(100.75), "ended")
        self.assertEqual(marks.toggle(101.2), "started")
        self.assertIsNone(marks.toggle(103.1))
        marks.stop()
        self.assertEqual(len(marks.snapshot()), 2)
        self.assertAlmostEqual(marks.snapshot()[0][0], 0.25)
        self.assertAlmostEqual(marks.snapshot()[0][1], 0.75)
        self.assertAlmostEqual(marks.snapshot()[1][0], 1.2)
        self.assertIsNone(marks.snapshot()[1][1])
        self.assertEqual(marks.counts(), (1, 1))

    def test_test_and_dataset_paths_and_event_rows(self):
        marks = recorder.IntervalMarkers(2)

        def fake_capture(port, seconds, raw_path, on_start=None, on_stop=None):
            if on_start:
                on_start(100.0)
            marks.toggle(100.25)
            marks.toggle(100.75)
            marks.toggle(101.2)
            raw_path.write_text(
                "S,1000000,44100.000,1000\n"
                + "A,0," + ",".join(["2"] * 128) + "\n"
                + "I,1001000,1,2,3\nI,1002000,2,3,4\n")
            if on_stop:
                on_stop()

        with TemporaryDirectory() as temp:
            with patch.object(recorder, "__file__", str(Path(temp) / "record_teensy.py")), \
                 patch.object(recorder, "capture", fake_capture):
                for test, subfolder in ((True, "test"), (False, "pilot")):
                    marks = recorder.IntervalMarkers(2)
                    graph, complete, incomplete = recorder.record(
                        2, "P01", "dry_swallow", test, "fake",
                        marks.start, marks.stop, marks.snapshot)
                    self.assertEqual(graph.parent.parts[-3], subfolder)
                    self.assertGreater(graph.stat().st_size, 1000)
                    self.assertEqual((complete, incomplete), (1, 1))
                    with (graph.parent / "events.csv").open() as file:
                        rows = list(csv.DictReader(file))
                    self.assertEqual(len(rows), 2)
                    self.assertEqual(rows[0]["record_type"], "marker")
                    self.assertEqual(rows[0]["activity"], "dry_swallow")
                    self.assertEqual(rows[0]["start_s"], "0.250000")
                    self.assertEqual(rows[0]["end_s"], "0.750000")
                    self.assertEqual(rows[1]["end_s"], "")
                    metadata = json.loads((graph.parent / "metadata.json").read_text())
                    self.assertEqual(metadata["mode"], "test" if test else "dataset")


if __name__ == "__main__":
    unittest.main()
