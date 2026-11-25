import math
import queue
import threading
import time
from collections import deque
from pathlib import Path

import serial
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
try:
    from PyQt6.QtCore import QThread, pyqtSignal as Signal
except ImportError:
    from PySide6.QtCore import QThread, Signal
from serial.tools import list_ports

from events import EVENT_MODEL_PATH, LiveEventDetector, load_event_model
from live_detection import MODEL_PATH, LiveDetector, load_model, model_label
from session import Protocol, Session, next_session


def filter_sos(kind, rate):
    low, high = (20, 2000) if kind == 'mic' else (.5, 100)
    return butter(2, [low, min(high, rate * .45)], btype='bandpass', fs=rate, output='sos')


def peak_envelope(array, max_points=1800):
    """Keep each channel's local extremes, in time order, for rendering only."""
    if len(array) <= max_points:
        return array.copy()
    channels = array.shape[1] - 1
    bins = max(1, (max_points - 2 - 2 * channels) // (2 * channels))
    size = int(np.ceil(len(array) / bins))
    count = len(array) // size
    values = array[:count * size, 1:].reshape(count, size, channels)
    offsets = np.arange(count)[:, None] * size
    indices = np.unique(np.r_[0, (values.argmin(axis=1) + offsets).ravel(),
                              (values.argmax(axis=1) + offsets).ravel(), len(array) - 1])
    # Include extrema in the final partial bin as well.
    tail = count * size
    if tail < len(array):
        indices = np.unique(np.r_[indices, tail + array[tail:, 1:].argmin(axis=0),
                                  tail + array[tail:, 1:].argmax(axis=0)])
    return array[indices]


class LivePreview:
    """Worker-owned ten-second window. Full-rate samples go to Session first."""

    def __init__(self):
        self.buffers = {'mic': deque(), 'imu': deque()}
        self.filters = {}

    def update(self, packets, rates, detect=None):
        mic, imu = [], []
        for kind, index, timestamp, values in packets:
            if kind == 'mic':
                values = np.asarray(values)
                mic.append(np.column_stack((timestamp + np.arange(len(values)) / rates[0], values)))
            else:
                imu.append((timestamp, *values))
        clipping = False
        for kind, pieces, rate in (('mic', mic, rates[0]), ('imu', imu, rates[1])):
            if not len(pieces):
                continue
            array = np.concatenate(pieces) if kind == 'mic' else np.asarray(pieces)
            if kind not in self.filters:
                sos = filter_sos(kind, rate)
                zi = sosfilt_zi(sos)[:, :, None] * array[0, 1:][None, None, :]
                self.filters[kind] = sos, zi
            sos, zi = self.filters[kind]
            filtered, zi = sosfilt(sos, array[:, 1:], axis=0, zi=zi)
            self.filters[kind] = sos, zi
            self.buffers[kind].append(np.column_stack((array, filtered)))
            cutoff = array[-1, 0] - 10
            while self.buffers[kind] and self.buffers[kind][0][-1, 0] < cutoff:
                self.buffers[kind].popleft()
            if kind == 'mic':
                clipping = bool(np.any(np.abs(array[:, 1]) >= 32760))
        raw, filtered, full_rate = [], [], []
        for kind, width in (('mic', 2), ('imu', 4)):
            if not self.buffers[kind]:
                full_rate.append(np.empty((0, width)))
                raw.append(np.empty((0, width)))
                filtered.append(np.empty((0, width)))
                continue
            array = np.concatenate(self.buffers[kind])
            array = array[array[:, 0] >= array[-1, 0] - 10]
            full_rate.append(array[:, :width])
            raw.append(peak_envelope(array[:, :width]))
            filtered.append(peak_envelope(np.column_stack((array[:, 0], array[:, width:]))))
        if detect is not None:
            detect(*full_rate)
        return {'raw': raw, 'filtered': filtered, 'clipping': clipping}


def ports():
    return [(p.device, f'{p.device}: {p.description}', p.vid == 0x16C0)
            for p in list_ports.comports()]


class Demo:
    """Deterministic synthetic signals, never presented as participant data."""

    def __init__(self):
        self.start = time.perf_counter()
        self.mic = self.imu = 0
        self.header = True

    def read(self):
        if self.header:
            self.header = False
            return b'S,1000000,44100,833,m/s2,V1\n'
        now = time.perf_counter() - self.start
        rows = []
        for _ in range(40):
            if (self.mic + 128) / 44100 > now:
                break
            values = []
            for n in range(128):
                t = (self.mic + n) / 44100
                burst = math.exp(-((t % 5 - 2.5) / .22)**2)
                values.append(str(int(140 * math.sin(2 * math.pi * 73 * t) + 9000 * burst * math.sin(2 * math.pi * 350 * t))))
            rows.append(f'A,{self.mic},{1000000 + round(self.mic / 44100 * 1e6)},' + ','.join(values))
            self.mic += 128
        for _ in range(100):
            t = self.imu / 833
            if t > now:
                break
            burst = math.exp(-((t % 5 - 2.5) / .22)**2)
            axes = [0.06 * math.sin(t * 6 + a) + burst * math.sin(t * 100 + a) + (9.81 if a == 2 else 0) for a in range(3)]
            rows.append(f'I,{self.imu},{1000000 + round(t * 1e6)},' + ','.join(f'{v:.6f}' for v in axes))
            self.imu += 1
        time.sleep(.003)
        return ('\n'.join(rows) + ('\n' if rows else '')).encode('ascii')


class Acquisition(QThread):
    status = Signal(str, bool)
    begun = Signal(object)
    saved = Signal(str, str)
    failed = Signal(str)
    detected = Signal(float, str, float)
    detection_status = Signal(str)

    def __init__(self, port, root, demo=False, model_path=None):
        super().__init__()
        self.model_path = Path(model_path or MODEL_PATH)
        self.detector_label = model_label(self.model_path)
        self.events_enabled = True
        self.event_detector = None
        self.port, self.root, self.demo = port, Path(root), demo
        self.commands = queue.Queue()
        self.session = None
        self.ready = False
        self.started = None
        self.protocol = Protocol()
        self.fresh = {'mic': 0.0, 'imu': 0.0}
        self.reference = None
        self._preview_lock = threading.Lock()
        self._latest_preview = None
        self.detector = None
        self.signal_valid = None

    def take_preview(self):
        """A single-slot mailbox: a slow/minimized UI never queues old frames."""
        with self._preview_lock:
            frame, self._latest_preview = self._latest_preview, None
            return frame

    def command(self, name, value=None):
        self.commands.put((name, value, time.perf_counter()))

    def finish(self, status='complete', note=''):
        if self.session:
            saved = self.session
            self.session = None
            if not all(saved.counts.values()):
                status, note = 'incomplete', 'One or both sensor streams contained no saved samples. ' + note
            saved.close(status, note)
            self.saved.emit(str(saved.folder), saved.metadata['status'])

    def active_text(self):
        events = ' · cough, talking, throat clear' if self.event_detector else ''
        swallow = f'{self.detector_label} active · swallows' if self.detector else 'Swallow detector off'
        return swallow + events

    def start_detection(self):
        self.preview_origin = None
        self.detector = None
        self.event_detector = None
        self.signal_valid = None
        if self.demo:
            self.detection_status.emit('Demo signals · live detection disabled')
            return
        label = self.detector_label = model_label(self.model_path)
        problem = None
        try:
            self.detector = LiveDetector(load_model(self.model_path))
        except FileNotFoundError:
            problem = (f'Detector unavailable ({self.model_path.name} not found). '
                       'Run train_live.py or train_mixed.py to create it.')
        except (OSError, ValueError, KeyError, TypeError):
            problem = f'Could not load {self.model_path.name}. Retrain or choose another detector.'
        if self.events_enabled:
            try:
                self.event_detector = LiveEventDetector(load_event_model())
            except (OSError, ValueError, KeyError, TypeError):
                problem = (problem + ' ' if problem else '') + 'Event model unavailable (run train_events.py).'
        if self.session and (self.detector or self.event_detector):
            self.session.enable_detection(label if self.detector else 'off', self.model_path.name if self.detector else None,
                                          EVENT_MODEL_PATH.name if self.event_detector else None)
        self.detection_status.emit(problem if problem and not (self.detector or self.event_detector)
                                   else self.active_text() + (f' · {problem}' if problem else ''))

    def detect(self, mic, imu):
        try:
            if self.session:
                origin, duration = self.session.origin, self.session.duration
            else:
                if not len(mic) or not len(imu):
                    return
                if self.preview_origin is None:
                    self.preview_origin = min(mic[-1, 0], imu[-1, 0])
                origin, duration = self.preview_origin, math.inf
            events = [(t, 'swallow', p) for t, p in self.detector.update(mic, imu, origin, duration)] if self.detector else []
            if self.event_detector:
                events += self.event_detector.update(imu, origin, duration)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            self.detector = self.event_detector = None
            self.detection_status.emit(f'Detector stopped: {error}')
            return
        valid = self.detector.signal_valid if self.detector else None
        if valid is not None and valid != self.signal_valid:
            self.signal_valid = valid
            self.detection_status.emit(self.active_text() if valid
                                       else 'Swallow detection paused · Weak, clipped, or missing signal')
        for seconds, event, score in sorted(events):
            if self.session:
                self.session.detection(seconds, score, event)
            else:
                seconds += origin
            self.detected.emit(seconds, event, score)

    def handle_commands(self):
        while True:
            try:
                name, value, when = self.commands.get_nowait()
            except queue.Empty:
                return True
            if name == 'quit':
                if self.session:
                    self.session.duration = min(self.session.duration, max(0, when - self.started))
                self.finish('stopped', 'Application closed or device disconnected by tester.')
                return False
            if name == 'start':
                if not self.ready or self.session or self.reference is None:
                    self.failed.emit('Both sensor streams must be receiving before starting.')
                    continue
                participant, activity, seconds = value
                device_s, received_at = self.reference
                # Extrapolate device acquisition time from the latest IMU packet.
                # USB latency remains an uncertainty of observer timing, documented in metadata.
                origin = device_s + max(0, when - received_at)
                folder = next_session(self.root, participant, activity)
                self.session = Session(folder, activity, seconds, origin, self.protocol.rates, self.port, self.demo)
                self.started = when
                self.stop_at = when + seconds + .25  # Allow queued USB packets to drain.
                self.begun.emit(dict(folder=str(folder), started=when, seconds=seconds, origin=origin))
                self.start_detection()
            elif name == 'stop' and self.session:
                self.session.duration = min(self.session.duration, max(0, when - self.started))
                self.stop_at = min(self.stop_at, when + .25)
            elif name == 'marker' and self.session:
                self.session.marker(when - self.started)
            elif name == 'events':
                self.events_enabled = bool(value)
                if not self.session:
                    self.start_detection()
            elif name == 'model':
                # The GUI locks the choice during recording; this guard keeps each
                # session's Detections.csv from a single detector.
                self.model_path = Path(value)
                if not self.session:
                    self.start_detection()

    def run(self):
        port = None
        output = []
        preview = LivePreview()
        last_plot = last_flush = time.perf_counter()
        connecting_at = last_plot
        buffer = b''
        last_status = None
        try:
            if self.demo:
                source = Demo()
            else:
                port = serial.Serial(self.port, 115200, timeout=.03, write_timeout=.5)
                port.reset_input_buffer()
                port.write(b'X')
                port.flush()
                time.sleep(.15)  # Let the Teensy process stop before starting a new stream.
                port.reset_input_buffer()
                port.write(b'S')
            while self.handle_commands():
                chunk = source.read() if self.demo else port.read(max(1, min(port.in_waiting, 32768)))
                buffer += chunk
                if len(buffer) > 1000000:
                    raise ValueError('USB packet exceeds the allowed length.')
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    packet = self.protocol.parse(line)
                    if not packet:
                        continue
                    if packet[0] == 'header':
                        if self.session:
                            raise ValueError('Device restarted during recording.')
                        self.fresh = {'mic': 0.0, 'imu': 0.0}
                        self.reference = None
                        output.clear()
                        preview = LivePreview()
                        self.start_detection()
                        continue
                    kind, index, seconds, values = packet
                    now = time.perf_counter()
                    self.fresh[kind] = now
                    if kind == 'imu':
                        self.reference = (seconds, now)
                    if self.protocol.gaps:
                        raise ValueError(f'Sample sequence gap detected ({self.protocol.gaps}). Check the USB cable and reconnect.')
                    if self.session:
                        self.session.sample(kind, index, seconds, values)
                    output.append(packet)
                now = time.perf_counter()
                self.ready = all(now - t < .75 for t in self.fresh.values())
                status = ('Demo ready (simulated signals)' if self.demo else 'Ready: microphone and acceleration receiving') if self.ready else 'Waiting for both sensor streams...'
                if status != last_status:
                    self.status.emit(status, self.ready)
                    last_status = status
                if self.session and not self.ready:
                    raise ValueError('Sensor stream stalled. Partial recording was saved.')
                if not self.ready and now - connecting_at > 5:
                    raise ValueError('No complete sensor stream. Check firmware, sensor wiring and USB data cable.')
                finishing = self.session and now >= self.stop_at
                if finishing or now - last_plot >= .1:
                    if output:
                        detect = self.detect if self.ready and (self.detector or self.event_detector) else None
                        frame = preview.update(output, self.protocol.rates, detect)
                        with self._preview_lock:
                            self._latest_preview = frame
                        output = []
                    last_plot = now
                if finishing:
                    self.finish('stopped' if self.session.duration < self.session.metadata['requested_duration_s'] else 'complete')
                    self.start_detection()
                if self.session and now - last_flush >= 1:
                    self.session.flush()
                    last_flush = now
        except Exception as error:
            self.ready = False
            try:
                self.finish('incomplete', str(error))
            except Exception as save_error:
                self.failed.emit(f'{error}\nCould not finish saving: {save_error}')
            else:
                self.failed.emit(str(error))
        finally:
            self.ready = False
            if port:
                try:
                    port.write(b'X')
                except serial.SerialException:
                    pass
                port.close()
