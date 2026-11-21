"""USB acquisition runs outside the GUI thread. Demo uses the same packet parser."""

import math
import queue
import time
from pathlib import Path

import serial
from PySide6.QtCore import QThread, Signal
from serial.tools import list_ports

from session import Protocol, Session, next_session


def ports():
    return [(p.device, f'{p.device} · {p.description}', p.vid == 0x16C0)
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
    batch = Signal(object)
    begun = Signal(object)
    saved = Signal(str, str)
    failed = Signal(str)

    def __init__(self, port, root, demo=False):
        super().__init__()
        self.port, self.root, self.demo = port, Path(root), demo
        self.commands = queue.Queue()
        self.session = None
        self.ready = False
        self.started = None
        self.protocol = Protocol()
        self.fresh = {'mic': 0.0, 'imu': 0.0}
        self.reference = None

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
            elif name == 'stop' and self.session:
                self.session.duration = min(self.session.duration, max(0, when - self.started))
                self.stop_at = min(self.stop_at, when + .25)
            elif name == 'marker' and self.session:
                self.session.marker(when - self.started)

    def run(self):
        port = None
        output = []
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
                status = ('DEMO · simulated signals' if self.demo else 'Ready · microphone + X / Y / Z receiving') if self.ready else 'Waiting for microphone and all acceleration channels…'
                if status != last_status:
                    self.status.emit(status, self.ready)
                    last_status = status
                if self.session and not self.ready:
                    raise ValueError('Sensor stream stalled. Partial recording was saved.')
                if not self.ready and now - connecting_at > 5:
                    raise ValueError('No complete sensor stream. Check firmware, sensor wiring and USB data cable.')
                if self.session and now >= self.stop_at:
                    self.finish('stopped' if self.session.duration < self.session.metadata['requested_duration_s'] else 'complete')
                if now - last_plot >= .05:
                    if output:
                        self.batch.emit((output, self.protocol.rates))
                        output = []
                    last_plot = now
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
