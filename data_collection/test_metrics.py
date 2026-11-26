"""Checks for swallow metrics and the water swallow test."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from swallow_metrics import clusters, post_swallow_coughs, session_events, swallow_metrics, water_test


def events(swallows, coughs=(), markers=(), duration=60.):
    return dict(swallow=list(swallows), cough=list(coughs), talking=[], throat_clear=[],
                markers=list(markers), duration=duration)


class MetricsTest(unittest.TestCase):
    def test_clusters_and_post_swallow_coughs(self):
        swallows = [(1, 1.6), (2.5, 3.0), (10, 10.7)]
        self.assertEqual([len(g) for g in clusters(swallows)], [2, 1])
        self.assertEqual(post_swallow_coughs(swallows, [(4.0, 4.4), (20, 20.5)]), [(4.0, 4.4)])
        m = swallow_metrics(events(swallows, [(4.0, 4.4)]))
        self.assertEqual((m['swallows'], m['multi_swallow_clusters'], m['post_swallow_coughs']), (3, 1, 1))
        self.assertEqual(m['swallows_per_minute'], 3.0)

    def test_water_test_capacity_and_cues(self):
        swallows = [(2.0 + 1.5 * i, 2.6 + 1.5 * i) for i in range(10)]   # last ends at 16.1 s
        normal = water_test(events(swallows, markers=[1.0]), 150)
        self.assertEqual(normal['swallows'], 10)
        self.assertAlmostEqual(normal['seconds'], 15.1)
        self.assertAlmostEqual(normal['capacity_ml_per_s'], 9.93, places=2)
        self.assertTrue(normal['refer_for_assessment'])        # 9.9 ml/s is below 10 ml/s
        fast = water_test(events(swallows[:5], coughs=[(30, 30.4)], markers=[1.0]), 150)
        self.assertAlmostEqual(fast['capacity_ml_per_s'], 150 / 7.6, places=2)   # 1.0 s to 8.6 s
        self.assertEqual(fast['coughs_after'], 1)
        self.assertEqual(len(fast['screening_cues']), 1)

    def test_second_marker_bounds_the_test(self):
        swallows = [(2, 2.5), (4, 4.5), (30, 30.5)]
        result = water_test(events(swallows, markers=[1.0, 5.0]), 100)
        self.assertEqual(result['swallows'], 2)
        self.assertEqual(result['end_s'], 5.0)

    def test_water_test_needs_start_marker(self):
        with self.assertRaises(ValueError):
            water_test(events([(2, 2.5)]), 100)

    def test_session_events_reads_detected_intervals_and_labels(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            (folder / 'Session.json').write_text(json.dumps(dict(saved_duration_s=30.)))
            with (folder / 'Events.csv').open('w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['record_type', 'activity', 'time_s', 'start_s', 'end_s', 'notes'])
                w.writerow(['marker', '', '1.0', '', '', ''])
                w.writerow(['label', 'dry_swallow', '', '2.0', '2.6', ''])
                w.writerow(['label', 'cough', '', '5.0', '5.4', '[uncertain] maybe'])
            with (folder / 'Intervals.csv').open('w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['event', 'start_s', 'end_s', 'duration_s', 'peak_score'])
                w.writerow(['swallow', '2.1', '2.7', '0.6', '0.9'])
                w.writerow(['cough', '5.0', '5.3', '0.3', '0.8'])
            labels = session_events(folder)
            detected = session_events(folder, 'detected')
            self.assertEqual((labels['swallow'], labels['cough'], labels['markers']), ([(2.0, 2.6)], [], [1.0]))
            self.assertEqual((detected['swallow'], detected['cough']), ([(2.1, 2.7)], [(5.0, 5.3)]))
            self.assertEqual(detected['duration'], 30.)


if __name__ == '__main__':
    unittest.main()
