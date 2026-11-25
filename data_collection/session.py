"""Device packet validation and lossless, session-relative CSV storage."""

import csv
import json
import math
import os
import re
from pathlib import Path

EVENT_FIELDS = ['record_type', 'activity', 'time_s', 'start_s', 'end_s', 'notes']


def safe_name(value):
    value = value.strip()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', value):
        raise ValueError('Use 1–64 letters, numbers, hyphens or underscores, starting with a letter or number.')
    if value.upper().split('.')[0] in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(10)), *(f'LPT{i}' for i in range(10))}:
        raise ValueError('That code is reserved by Windows. Choose another code.')
    return value


def next_session(root, participant, activity):
    """Allocate atomically; numbering is shared across activities and days."""
    parent = Path(root) / safe_name(participant)
    activity = safe_name(activity).lower()
    parent.mkdir(parents=True, exist_ok=True)
    numbers = [int(m.group(1)) for p in parent.iterdir()
               if (m := re.match(r'^S(\d+)_', p.name))]
    number = max(numbers, default=0) + 1
    while True:
        folder = parent / f'S{number:02d}_{activity}'
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            number += 1


class Protocol:
    """V1 packets use device microseconds for both sensors, never USB arrival times."""

    def __init__(self):
        self.epoch = None
        self.rates = (0, 0)
        self.last = {'mic': None, 'imu': None}
        self.expected = {'mic': None, 'imu': None}
        self.gaps = 0

    def parse(self, line):
        row = line.decode('ascii').strip().split(',')
        if row[0] == 'E':
            raise ValueError('Device: ' + ', '.join(row[1:]))
        if row[0] == 'S':
            if len(row) != 6 or row[5] != 'V1':
                raise ValueError('Upload the included V1 Teensy firmware before connecting.')
            epoch, mic_fs, imu_fs = int(row[1]), float(row[2]), float(row[3])
            if not (0 <= epoch < 2**32 and 1000 <= mic_fs <= 192000 and 1 <= imu_fs <= 10000):
                raise ValueError('Invalid device clock or sample rates.')
            if row[4] != 'm/s2':
                raise ValueError('Expected acceleration in m/s2.')
            self.__init__()
            self.epoch, self.rates = epoch, (mic_fs, imu_fs)
            return ('header', self.rates)
        if row[0] not in ('A', 'I'):
            return None  # Boot messages do not imply readiness.
        if self.epoch is None:
            raise ValueError('No V1 start header received. Upload the included firmware.')
        kind = 'mic' if row[0] == 'A' else 'imu'
        expected_length = 131 if kind == 'mic' else 6
        if len(row) != expected_length:
            raise ValueError(f'Malformed {kind} packet: expected {expected_length} fields.')
        index, timestamp = int(row[1]), int(row[2])
        if index < 0 or not 0 <= timestamp < 2**32:
            raise ValueError('Invalid sample index or device timestamp.')
        seconds = ((timestamp - self.epoch) % 2**32) / 1e6
        if seconds > 3600:
            raise ValueError('Device clock reset or stream exceeds one hour. Reconnect.')
        values = tuple(float(x) for x in row[3:])
        if not all(math.isfinite(x) for x in values):
            raise ValueError('Non-finite sensor value received.')
        if kind == 'mic' and any(x != int(x) or not -32768 <= x <= 32767 for x in values):
            raise ValueError('Microphone samples must be signed 16-bit PCM.')
        count = len(values) if kind == 'mic' else 1
        previous = self.expected[kind]
        if previous is not None:
            if index < previous:
                raise ValueError('Duplicate/out-of-order samples or device restart.')
            self.gaps += index - previous
        last = self.last[kind]
        if last is not None and seconds <= last:
            raise ValueError('Device timestamps moved backwards.')
        self.expected[kind] = index + count
        self.last[kind] = seconds + ((count - 1) / self.rates[0] if kind == 'mic' else 0)
        return kind, index, seconds, values


def save_events(path, rows):
    """Replace atomically so an interrupted review cannot truncate the old file."""
    path = Path(path)
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


class Session:
    def __init__(self, folder, activity, duration, origin, rates, port, demo=False):
        if not math.isfinite(duration) or not 0 < duration <= 1800:
            raise ValueError('Duration must be between 0 and 1800 seconds.')
        self.folder, self.activity = Path(folder), activity
        self.duration, self.origin, self.rates = duration, origin, rates
        self.counts = {'mic': 0, 'imu': 0}
        self.first_index = {'mic': None, 'imu': None}
        self.last_time = 0.0
        self.last_times = {'mic': 0.0, 'imu': 0.0}
        self.events = []
        self.detection_writer = None
        self.closed = False
        self.files = {}
        self.writers = {}
        try:
            for kind, filename, columns in (
                ('mic', 'Mic.csv', ['sample_idx', 'time_s', 'mic_raw']),
                ('imu', 'IMU.csv', ['sample_idx', 'time_s', 'ax', 'ay', 'az']),
            ):
                file = (self.folder / filename).open('x', newline='', encoding='utf-8')
                self.files[kind] = file
                self.writers[kind] = csv.writer(file)
                self.writers[kind].writerow(columns)
            self.event_file = (self.folder / 'Events.csv').open('x', newline='', encoding='utf-8')
            self.event_writer = csv.DictWriter(self.event_file, EVENT_FIELDS)
            self.event_writer.writeheader()
        except Exception:
            for file in self.files.values():
                file.close()
            raise
        self.metadata = dict(activity=activity, requested_duration_s=duration,
                             device_origin_s=origin, mic_rate_hz=rates[0], imu_rate_hz=rates[1],
                             port=port, simulated=demo, status='recording', protocol='V1',
                             time_note='Common device clock; observer markers include USB and human reaction delay.',
                             display_filter='Mic 20–2000 Hz; acceleration 0.5–100 Hz; raw CSV unchanged.')
        self.write_metadata()

    def write_metadata(self):
        path = self.folder / 'Session.json'
        temp = path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(self.metadata, indent=2) + '\n', encoding='utf-8')
        temp.replace(path)

    def sample(self, kind, index, device_time, values):
        if self.closed:
            return
        samples = [(index + n, device_time + n / self.rates[0], (int(value),))
                   for n, value in enumerate(values)] if kind == 'mic' else [(index, device_time, values)]
        for idx, timestamp, data in samples:
            relative = timestamp - self.origin
            if 0 <= relative <= self.duration:
                if self.first_index[kind] is None:
                    self.first_index[kind] = idx
                self.writers[kind].writerow([idx - self.first_index[kind], f'{relative:.9f}', *data])
                self.counts[kind] += 1
                self.last_time = max(self.last_time, relative)
                self.last_times[kind] = relative

    def marker(self, seconds):
        if self.closed or not 0 <= seconds <= self.duration:
            return
        row = dict.fromkeys(EVENT_FIELDS, '')
        row.update(record_type='marker', activity=self.activity, time_s=f'{seconds:.6f}')
        self.event_writer.writerow(row)
        self.event_file.flush()
        self.events.append(row)

    def enable_detection(self):
        file = (self.folder / 'Detections.csv').open('x', newline='', encoding='utf-8')
        self.files['detections'] = file
        self.detection_writer = csv.writer(file)
        self.detection_writer.writerow(['time_s', 'event', 'model_score'])
        self.metadata['live_detector'] = 'wearable'
        self.write_metadata()

    def detection(self, seconds, score):
        self.detection_writer.writerow([f'{seconds:.3f}', 'swallow', f'{score:.6f}'])

    def flush(self):
        for file in (*self.files.values(), self.event_file):
            file.flush()

    def close(self, status='complete', note=''):
        if self.closed:
            return
        self.closed = True
        if status == 'complete' and any(self.last_times[kind] < self.duration - .05 for kind in ('mic', 'imu')):
            status = 'incomplete'
            note = 'A sensor stream ended before the requested recording duration. ' + note
        for file in (*self.files.values(), self.event_file):
            try:
                file.flush()
                os.fsync(file.fileno())
            finally:
                file.close()
        self.metadata.update(status=status, note=note, samples=self.counts, saved_duration_s=self.last_time,
                             stream_end_s=self.last_times)
        self.write_metadata()
