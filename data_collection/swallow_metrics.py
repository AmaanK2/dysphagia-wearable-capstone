"""Per-swallow metrics, normal ranges, and the timed water swallow test.

Swallows and coughs are (start, end) intervals in seconds, either from reviewed labels
(Events.csv) or from live detection (Intervals.csv).

Metrics per session: swallow count and rate, swallow duration, time between swallows
(onset to onset), swallow clusters (several swallows in quick succession, e.g. piecemeal
swallowing of one mouthful) and coughs shortly after a swallow.

Water swallow test (``water_test``): the person drinks a known volume as quickly as is
comfortable. The operator presses Space when the cup touches the lips (and optionally
again when drinking ends). Reported: number of swallows, time, volume per swallow, time
per swallow and swallowing capacity (ml/s). Screening cues follow the timed water swallow
test (capacity below 10 ml/s; Nathadwarawala et al. 1992, Hughes & Wiles 1996) and the
3-oz water swallow test (cough or choking during or within a minute after drinking;
Suiter & Leder 2008). This is a research tool, not a diagnosis.

Normal ranges from reviewed labels in our recordings, or re-score one water test:
    python data_collection/swallow_metrics.py
    python data_collection/swallow_metrics.py --water-test <session folder> --volume 100 [--labels]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

HOME = Path(__file__).resolve().parent
OUT = HOME.parent / 'experiments' / 'swallow_metrics'
CLUSTER_GAP = 1.5          # s between one swallow's end and the next start to count as one cluster
POST_SWALLOW_COUGH = 3.    # s after a swallow ends in which a cough counts as post-swallow
CAPACITY_CUTOFF = 10.      # ml/s; below this the timed water swallow test is usually called abnormal
WATER_TEST_COUGH_WINDOW = 60.   # s after drinking ends during which a cough is still reported
SWALLOW = {'dry_swallow', 'swallow', 'water_swallow'}


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def session_events(folder, source='labels'):
    """{'swallow': [(start, end)], 'cough': [...], 'talking': [...], 'markers': [t], 'duration': s}."""
    folder = Path(folder)
    out = {'swallow': [], 'cough': [], 'talking': [], 'throat_clear': [], 'markers': []}
    events = read_csv(folder / 'Events.csv') if (folder / 'Events.csv').is_file() else []
    out['markers'] = sorted(float(r['time_s']) for r in events if r['record_type'] == 'marker' and r['time_s'])
    if source == 'labels':
        for r in events:
            if r['record_type'] == 'label' and r['start_s'] and r['end_s'] and not (r.get('notes') or '').startswith('[uncertain]'):
                kind = 'swallow' if r['activity'] in SWALLOW else r['activity']
                if kind in out:
                    out[kind].append((float(r['start_s']), float(r['end_s'])))
    else:
        path = folder / 'Intervals.csv'
        for r in read_csv(path) if path.is_file() else []:
            if r['event'] in out:
                out[r['event']].append((float(r['start_s']), float(r['end_s'])))
    metadata = json.loads((folder / 'Session.json').read_text()) if (folder / 'Session.json').is_file() else {}
    out['duration'] = float(metadata.get('saved_duration_s') or metadata.get('requested_duration_s') or 0.)
    for kind in ('swallow', 'cough', 'talking', 'throat_clear'):
        out[kind].sort()
    return out


def clusters(swallows, gap=CLUSTER_GAP):
    """Group swallows whose start follows the previous end by at most ``gap`` seconds."""
    groups = []
    for start, end in sorted(swallows):
        if groups and start - groups[-1][-1][1] <= gap:
            groups[-1].append((start, end))
        else:
            groups.append([(start, end)])
    return groups


def post_swallow_coughs(swallows, coughs, window=POST_SWALLOW_COUGH):
    return [c for c in coughs if any(0 <= c[0] - e <= window for _, e in swallows)]


def describe(values):
    v = np.asarray(values, float)
    if not len(v):
        return None
    return dict(n=int(len(v)), median=round(float(np.median(v)), 3),
                p25=round(float(np.percentile(v, 25)), 3), p75=round(float(np.percentile(v, 75)), 3),
                p5=round(float(np.percentile(v, 5)), 3), p95=round(float(np.percentile(v, 95)), 3))


def swallow_metrics(events):
    swallows, coughs = events['swallow'], events['cough']
    onsets = [s for s, _ in swallows]
    groups = clusters(swallows)
    minutes = events['duration'] / 60 if events['duration'] else None
    return dict(
        swallows=len(swallows),
        swallows_per_minute=round(len(swallows) / minutes, 2) if minutes else None,
        duration_s=describe([e - s for s, e in swallows]),
        onset_interval_s=describe(np.diff(onsets)) if len(onsets) > 1 else None,
        clusters=len(groups),
        multi_swallow_clusters=sum(len(g) > 1 for g in groups),
        coughs=len(coughs),
        post_swallow_coughs=len(post_swallow_coughs(swallows, coughs)))


def water_test(events, volume_ml):
    """Timed water swallow test from one session's swallows, coughs and operator markers."""
    markers = events['markers']
    if not markers:
        raise ValueError('Press Space when the cup touches the lips so the test has a start time.')
    start = markers[0]
    drinking = [(s, e) for s, e in events['swallow'] if e >= start and (len(markers) < 2 or s <= markers[1] + 1.)]
    if not drinking:
        raise ValueError('No swallows were detected after the start marker.')
    end = max(markers[1] if len(markers) > 1 else 0., drinking[-1][1])
    seconds = end - start
    n = len(drinking)
    during = [c for c in events['cough'] if start <= c[0] <= end]
    after = [c for c in events['cough'] if end < c[0] <= end + WATER_TEST_COUGH_WINDOW]
    capacity = volume_ml / seconds if seconds > 0 else None
    cues = []
    if capacity is not None and capacity < CAPACITY_CUTOFF:
        cues.append(f'Swallowing capacity {capacity:.1f} ml/s is below {CAPACITY_CUTOFF:.0f} ml/s')
    if during or after:
        cues.append(f'{len(during) + len(after)} cough(s) during or within {WATER_TEST_COUGH_WINDOW:.0f} s after drinking')
    return dict(volume_ml=volume_ml, start_s=round(start, 3), end_s=round(end, 3), seconds=round(seconds, 2),
                swallows=n, ml_per_swallow=round(volume_ml / n, 1), seconds_per_swallow=round(seconds / n, 2),
                capacity_ml_per_s=round(capacity, 2) if capacity else None,
                coughs_during=len(during), coughs_after=len(after),
                screening_cues=cues, refer_for_assessment=bool(cues),
                note='Research screening aid only; not a diagnosis. Confirm with a clinician.')


def normal_ranges(root):
    sessions, pooled = [], {'duration': [], 'interval': [], 'cluster_size': []}
    for metadata in sorted(Path(root).glob('*/*/Session.json')):
        folder = metadata.parent
        events = session_events(folder)
        if not events['swallow']:
            continue
        row = dict(session=str(folder.relative_to(root)), **swallow_metrics(events))
        sessions.append(row)
        pooled['duration'] += [e - s for s, e in events['swallow']]
        onsets = [s for s, _ in events['swallow']]
        pooled['interval'] += list(np.diff(onsets))
        pooled['cluster_size'] += [len(g) for g in clusters(events['swallow'])]
    by_participant = {}
    for row in sessions:
        p = row['session'].split('/')[0]
        by_participant.setdefault(p, []).append(row['duration_s']['median'])
    return dict(sessions=sessions,
                pooled=dict(swallow_duration_s=describe(pooled['duration']),
                            onset_interval_s=describe(pooled['interval']),
                            cluster_size=describe(pooled['cluster_size'])),
                median_duration_by_participant={p: round(float(np.median(v)), 3) for p, v in by_participant.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--water-test', type=Path, help='Session folder of a water swallow test')
    parser.add_argument('--volume', type=float, default=100., help='Water volume in ml')
    parser.add_argument('--labels', action='store_true', help='Use reviewed labels instead of live detections')
    args = parser.parse_args()
    if args.water_test:
        events = session_events(args.water_test, 'labels' if args.labels else 'detected')
        print(json.dumps(water_test(events, args.volume), indent=2))
        return
    report = normal_ranges(HOME / 'recordings' / 'participants')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'normal_ranges.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['pooled'], indent=1))
    print('median duration by participant:', report['median_duration_by_participant'])


if __name__ == '__main__':
    main()
