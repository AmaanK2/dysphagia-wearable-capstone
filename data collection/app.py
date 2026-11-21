"""Run: python app.py. Use --demo to try the complete workflow without hardware."""

import argparse
import csv
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt
from PySide6 import QtCore, QtGui, QtWidgets as W

from device import Acquisition, ports
from session import EVENT_FIELDS, safe_name, save_events

HOME = Path(__file__).resolve().parent
ACTIVITIES = {'Dry swallow': 'dry_swallow', 'Talking': 'talking', 'Mixed activities': 'mixed'}
COLORS = ['#18b5a4', '#658bff', '#f2ab53']


def button(text, action, primary=False):
    item = W.QPushButton(text)
    item.setMinimumHeight(38)
    if primary:
        item.setObjectName('primary')
    item.clicked.connect(action)
    return item


def message(parent, text):
    W.QMessageBox.warning(parent, 'Swallow Studio', str(text))


def filter_sos(kind, rate):
    low, high = (20, 2000) if kind == 'mic' else (.5, 100)
    return butter(2, [low, min(high, rate * .45)], btype='bandpass', fs=rate, output='sos')


class SignalPlots(W.QWidget):
    def __init__(self):
        super().__init__()
        layout = W.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plots, self.curves, self.overlays = [], [], []
        for title, label, colors in (
            ('CONTACT MICROPHONE', 'PCM counts', [COLORS[0]]),
            ('ACCELERATION · X / Y / Z', 'm/s²', COLORS),
        ):
            plot = pg.PlotWidget(background='#111e2c')
            plot.setTitle(title, color='#c7d5e4', size='10pt')
            plot.setLabel('left', label, color='#a8b9cc')
            plot.setLabel('bottom', 'Time', units='s', color='#a8b9cc')
            plot.showGrid(x=True, y=True, alpha=.15)
            plot.setMenuEnabled(False)
            plot.getAxis('left').setWidth(68)
            plot.getPlotItem().setContentsMargins(10, 10, 14, 5)
            curves = []
            if len(colors) == 3:
                plot.addLegend(offset=(12, 10))
            for n, color in enumerate(colors):
                curve = plot.plot(pen=pg.mkPen(color, width=1.3), name='XYZ'[n] if len(colors) == 3 else None)
                curve.setDownsampling(auto=True, method='peak')
                curve.setClipToView(True)
                curves.append(curve)
            layout.addWidget(plot, 1)
            self.plots.append(plot)
            self.curves.append(curves)
        self.plots[1].setXLink(self.plots[0])

    def draw(self, mic, imu):
        self.curves[0][0].setData(mic[:, 0], mic[:, 1]) if len(mic) else self.curves[0][0].clear()
        for axis, curve in enumerate(self.curves[1]):
            curve.setData(imu[:, 0], imu[:, axis + 1]) if len(imu) else curve.clear()

    def clear_overlays(self):
        for plot, item in self.overlays:
            plot.removeItem(item)
        self.overlays = []

    def marker(self, seconds):
        for plot in self.plots:
            line = pg.InfiniteLine(seconds, pen=pg.mkPen('#f2ab53', width=1.3, style=QtCore.Qt.PenStyle.DashLine))
            plot.addItem(line)
            self.overlays.append((plot, line))

    def interval(self, start, end, uncertain=False):
        for plot in self.plots:
            item = pg.LinearRegionItem((start, end), movable=False,
                                       brush=pg.mkBrush(242, 171, 83, 40) if uncertain else pg.mkBrush(24, 181, 164, 35),
                                       pen=pg.mkPen('#f2ab53' if uncertain else '#18b5a4'))
            item.setZValue(-5)
            plot.addItem(item)
            self.overlays.append((plot, item))


class Setup(W.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle('New recording')
        self.setMinimumWidth(430)
        form = W.QFormLayout(self)
        form.setContentsMargins(26, 22, 26, 24)
        form.setSpacing(16)
        heading = W.QLabel('Set up a session')
        heading.setObjectName('heading')
        form.addRow(heading)
        self.participant = W.QLineEdit(parent.last_participant)
        self.participant.setPlaceholderText('P01')
        form.addRow('Participant ID', self.participant)
        self.activity = W.QComboBox()
        self.activity.addItems([*ACTIVITIES, 'Custom'])
        form.addRow('Activity', self.activity)
        self.custom = W.QLineEdit()
        self.custom.setPlaceholderText('e.g. throat_clear')
        self.custom.setEnabled(False)
        self.activity.currentTextChanged.connect(lambda t: self.custom.setEnabled(t == 'Custom'))
        form.addRow('Custom activity', self.custom)
        presets = W.QWidget()
        row = W.QHBoxLayout(presets)
        row.setContentsMargins(0, 0, 0, 0)
        self.seconds = W.QDoubleSpinBox()
        self.seconds.setRange(1, 1800)
        self.seconds.setDecimals(1)
        self.seconds.setValue(120)
        self.seconds.setSuffix(' seconds')
        for label, value in [('1 minute', 60), ('2 minutes', 120), ('3 minutes', 180)]:
            row.addWidget(button(label, lambda checked=False, v=value: self.seconds.setValue(v)))
        form.addRow('Duration', presets)
        form.addRow('Custom duration', self.seconds)
        hint = W.QLabel('Use the same participant ID on return visits.\nThe next session number is assigned automatically.')
        hint.setObjectName('muted')
        form.addRow(hint)
        self.error = W.QLabel('')
        self.error.setWordWrap(True)
        self.error.setStyleSheet('color: #f2ab53')
        form.addRow(self.error)
        controls = W.QDialogButtonBox(W.QDialogButtonBox.StandardButton.Cancel)
        controls.addButton('Begin recording', W.QDialogButtonBox.ButtonRole.AcceptRole)
        controls.accepted.connect(self.validate)
        controls.rejected.connect(self.reject)
        form.addRow(controls)

    def validate(self):
        try:
            self.code = safe_name(self.participant.text())
            custom = self.custom.text().strip().replace(' ', '_').lower()
            self.task = safe_name(custom) if self.activity.currentText() == 'Custom' else ACTIVITIES[self.activity.currentText()]
        except ValueError as error:
            self.error.setText(str(error))
            return
        self.accept()


class ReviewLoader(QtCore.QThread):
    loaded = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, folder):
        super().__init__()
        self.folder = folder

    def run(self):
        try:
            metadata_path = self.folder / 'Session.json'
            if metadata_path.exists() and json.loads(metadata_path.read_text(encoding='utf-8')).get('status') == 'recording':
                raise ValueError('Stop the recording before opening it for review.')
            mic = np.loadtxt(self.folder / 'Mic.csv', delimiter=',', skiprows=1, usecols=(1, 2), ndmin=2)
            imu = np.loadtxt(self.folder / 'IMU.csv', delimiter=',', skiprows=1, usecols=(1, 2, 3, 4), ndmin=2)
            if not len(mic) or not len(imu):
                raise ValueError('This session has no complete pair of sensor streams.')
            with (self.folder / 'Events.csv').open(newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                if reader.fieldnames != EVENT_FIELDS:
                    raise ValueError('Events.csv has a different schema.')
                events = list(reader)
            for array in (mic, imu):
                if not np.isfinite(array).all() or np.any(np.diff(array[:, 0]) <= 0) or array[0, 0] < 0:
                    raise ValueError('Session contains invalid or non-increasing signal times.')
            self.loaded.emit((self.folder, mic, imu, events))
        except Exception as error:
            self.failed.emit(str(error))


class Review(W.QWidget):
    def __init__(self):
        super().__init__()
        self.folder = None
        self.loader = None
        self.events = []
        self.selected = None
        self.syncing = False
        layout = W.QVBoxLayout(self)
        top = W.QHBoxLayout()
        self.open_button = button('Open session', self.choose)
        top.addWidget(self.open_button)
        self.path = W.QLabel('Choose a recorded session folder')
        self.path.setObjectName('muted')
        top.addWidget(self.path, 1)
        self.filtered = W.QCheckBox('Filtered view')
        self.filtered.setChecked(True)
        self.filtered.toggled.connect(self.redraw)
        top.addWidget(self.filtered)
        layout.addLayout(top)
        body = W.QSplitter()
        layout.addWidget(body, 1)
        self.graph = SignalPlots()
        body.addWidget(self.graph)
        self.region = pg.LinearRegionItem((0, 1), brush=pg.mkBrush(101, 139, 255, 40), pen=pg.mkPen('#658bff'))
        self.region.setZValue(10)
        self.graph.plots[0].addItem(self.region)
        self.region.sigRegionChanged.connect(self.from_region)
        self.region.hide()
        self.imu_region = pg.LinearRegionItem((0, 1), movable=False, brush=pg.mkBrush(101, 139, 255, 40), pen=pg.mkPen('#658bff'))
        self.graph.plots[1].addItem(self.imu_region)
        self.imu_region.hide()
        sidebar = W.QWidget()
        sidebar.setMinimumWidth(300)
        side = W.QVBoxLayout(sidebar)
        title = W.QLabel('Event review')
        title.setObjectName('heading')
        side.addWidget(title)
        hint = W.QLabel('Select a marker, then drag the blue region\nto the estimated event start and end.')
        hint.setObjectName('muted')
        side.addWidget(hint)
        self.table = W.QListWidget()
        self.table.currentRowChanged.connect(self.select)
        side.addWidget(self.table, 1)
        side.addWidget(button('New event', self.new))
        form = W.QFormLayout()
        self.activity = W.QComboBox()
        self.activity.setEditable(True)
        self.activity.addItems(['dry_swallow', 'talking', 'cough', 'throat_clear', 'normal_breathing'])
        form.addRow('Activity', self.activity)
        self.start, self.end = W.QDoubleSpinBox(), W.QDoubleSpinBox()
        for widget in (self.start, self.end):
            widget.setRange(0, 1800)
            widget.setDecimals(4)
            widget.setSuffix(' s')
            widget.valueChanged.connect(self.to_region)
        form.addRow('Start', self.start)
        form.addRow('End', self.end)
        self.uncertain = W.QCheckBox('Uncertain event')
        form.addRow(self.uncertain)
        self.notes = W.QLineEdit()
        self.notes.setPlaceholderText('Optional review notes')
        form.addRow('Notes', self.notes)
        side.addLayout(form)
        self.save_button = button('Save event', self.save, True)
        self.delete_button = button('Delete selected label', self.delete)
        side.addWidget(self.save_button)
        side.addWidget(self.delete_button)
        self.notice = W.QLabel('Original observer markers are preserved.')
        self.notice.setWordWrap(True)
        self.notice.setObjectName('muted')
        side.addWidget(self.notice)
        body.addWidget(sidebar)
        body.setStretchFactor(0, 4)
        body.setStretchFactor(1, 1)
        self.save_button.setEnabled(False)
        self.delete_button.setEnabled(False)

    def choose(self):
        folder = W.QFileDialog.getExistingDirectory(self, 'Open session folder', str(HOME / 'recordings'))
        if folder:
            self.load(Path(folder))

    def load(self, folder):
        if self.loader and self.loader.isRunning():
            return
        self.open_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.delete_button.setEnabled(False)
        self.path.setText('Loading signals…')
        self.loader = ReviewLoader(Path(folder))
        self.loader.loaded.connect(self.loaded)
        self.loader.failed.connect(self.load_failed)
        self.loader.finished.connect(lambda: self.open_button.setEnabled(True))
        self.loader.start()

    def load_failed(self, error):
        self.path.setText('Could not open session')
        message(self, error)
        self.save_button.setEnabled(self.folder is not None)

    def loaded(self, data):
        self.folder, self.mic, self.imu, self.events = data
        self.duration = max(self.mic[-1, 0], self.imu[-1, 0])
        self.path.setText(f'{self.folder.parent.name} / {self.folder.name}')
        self.path.setToolTip(str(self.folder))
        self.filtered_data = []
        for kind, array in [('mic', self.mic), ('imu', self.imu)]:
            filtered = array.copy()
            if len(array) > 30:
                rate = 1 / np.median(np.diff(array[:, 0]))
                try:
                    filtered[:, 1:] = sosfiltfilt(filter_sos(kind, rate), array[:, 1:], axis=0)
                except ValueError:
                    pass
            self.filtered_data.append(filtered)
        self.redraw()
        self.graph.plots[0].setXRange(0, self.duration, padding=.01)
        self.graph.plots[0].setLimits(xMin=0, xMax=self.duration)
        self.refresh()
        self.new()
        self.region.show()
        self.imu_region.show()
        self.save_button.setEnabled(True)

    def redraw(self):
        if self.folder is None:
            return
        self.graph.draw(*self.filtered_data if self.filtered.isChecked() else (self.mic, self.imu))
        self.graph.clear_overlays()
        for event in self.events:
            if event['record_type'] == 'marker':
                self.graph.marker(float(event['time_s']))
            elif event['record_type'] == 'label':
                self.graph.interval(float(event['start_s']), float(event['end_s']), event['notes'].startswith('[uncertain]'))

    def refresh(self):
        self.table.blockSignals(True)
        self.table.clear()
        for event in self.events:
            if event['record_type'] == 'marker':
                text = f'○  Marker · {float(event["time_s"]):.3f} s'
            else:
                text = f'{"?" if event["notes"].startswith("[uncertain]") else "✓"}  {event["activity"]} · {event["start_s"]}–{event["end_s"]} s'
            self.table.addItem(text)
        self.table.blockSignals(False)

    def select(self, index):
        if not 0 <= index < len(self.events):
            return
        event = self.events[index]
        label = event['record_type'] == 'label'
        self.selected = index if label else None
        self.delete_button.setEnabled(label)
        self.save_button.setText('Update event' if label else 'Save event')
        activity = event['activity'] if event['activity'] != 'mixed' else 'dry_swallow'
        self.activity.setCurrentText(activity)
        if label:
            start, end = float(event['start_s']), float(event['end_s'])
        else:
            t = float(event['time_s'])
            start, end = max(0, t - .6), min(self.duration, t + .6)
        self.region.setRegion((start, end))
        self.uncertain.setChecked(event['notes'].startswith('[uncertain]'))
        self.notes.setText(event['notes'].removeprefix('[uncertain]').strip())
        self.graph.plots[0].setXRange(max(0, start - 2), min(self.duration, end + 2), padding=0)

    def new(self):
        if not self.folder:
            return
        self.selected = None
        self.table.setCurrentRow(-1)
        self.delete_button.setEnabled(False)
        self.save_button.setText('Save event')
        self.notes.clear()
        self.uncertain.setChecked(False)
        self.region.setBounds((0, self.duration))
        self.imu_region.setBounds((0, self.duration))
        self.region.setRegion((0, min(1, self.duration)))

    def from_region(self):
        if self.syncing:
            return
        self.syncing = True
        start, end = self.region.getRegion()
        self.imu_region.setRegion((start, end))
        self.start.setValue(start)
        self.end.setValue(end)
        self.syncing = False

    def to_region(self):
        if not self.syncing:
            self.syncing = True
            self.region.setRegion((self.start.value(), self.end.value()))
            self.imu_region.setRegion((self.start.value(), self.end.value()))
            self.syncing = False

    def save(self):
        if not self.folder:
            return
        try:
            activity = safe_name(self.activity.currentText().strip().replace(' ', '_').lower())
            start, end = self.start.value(), self.end.value()
            if not 0 <= start < end <= self.duration:
                raise ValueError('Select a start before the end, within the recorded signals.')
            notes = self.notes.text().strip()
            if self.uncertain.isChecked():
                if not notes:
                    raise ValueError('Add a short note explaining why the event is uncertain.')
                notes = '[uncertain] ' + notes
            event = dict(record_type='label', activity=activity, time_s='', start_s=f'{start:.6f}', end_s=f'{end:.6f}', notes=notes)
            proposed = list(self.events)
            index = self.selected if self.selected is not None else len(proposed)
            if self.selected is None:
                proposed.append(event)
            else:
                proposed[index] = event
            save_events(self.folder / 'Events.csv', proposed)
            self.events = proposed
            self.refresh()
            self.redraw()
            self.table.setCurrentRow(index)
            self.notice.setText('Saved to Events.csv · observer markers preserved')
        except (OSError, ValueError) as error:
            message(self, error)

    def delete(self):
        if self.selected is None:
            return
        proposed = self.events.copy()
        proposed.pop(self.selected)
        try:
            save_events(self.folder / 'Events.csv', proposed)
        except OSError as error:
            message(self, error)
            return
        self.events = proposed
        self.refresh()
        self.redraw()
        self.new()


class MainWindow(W.QMainWindow):
    def __init__(self, demo=False):
        super().__init__()
        self.setWindowTitle('Swallow Studio · Data Collection')
        self.resize(1300, 850)
        self.setMinimumSize(1040, 720)
        self.worker = None
        self.recording = False
        self.pending = False
        self.ready = False
        self.last_participant = 'P01'
        self.latest_session = None
        self.data_root = HOME / 'recordings' / ('demo' if demo else 'participants')
        self.markers = []
        self.origin = 0
        self.filter_state = {}
        self.buffers = {'mic': deque(maxlen=240), 'imu': deque(maxlen=240)}
        central = W.QWidget()
        self.setCentralWidget(central)
        layout = W.QVBoxLayout(central)
        layout.setContentsMargins(26, 22, 26, 22)
        layout.setSpacing(16)
        header = W.QHBoxLayout()
        title = W.QLabel('Swallow Studio')
        title.setObjectName('title')
        header.addWidget(title)
        subtitle = W.QLabel('RESEARCH DATA COLLECTION')
        subtitle.setObjectName('muted')
        header.addWidget(subtitle, 1)
        self.connection_status = W.QLabel('○  Device disconnected')
        self.connection_status.setObjectName('pill')
        header.addWidget(self.connection_status)
        layout.addLayout(header)
        self.tabs = W.QTabWidget()
        layout.addWidget(self.tabs, 1)
        record_tab = W.QWidget()
        self.tabs.addTab(record_tab, 'Collect')
        self.review = Review()
        self.tabs.addTab(self.review, 'Review and label')
        record_layout = W.QVBoxLayout(record_tab)
        record_layout.setSpacing(16)
        connection_row = W.QHBoxLayout()
        self.port = W.QComboBox()
        self.port.setMinimumWidth(270)
        connection_row.addWidget(self.port, 1)
        self.refresh_button = button('Refresh USB', self.refresh_ports)
        connection_row.addWidget(self.refresh_button)
        self.demo = W.QCheckBox('Demo mode')
        self.demo.setChecked(demo)
        connection_row.addWidget(self.demo)
        self.connect_button = button('Connect device', self.connect_device)
        connection_row.addWidget(self.connect_button)
        record_layout.addLayout(connection_row)
        body = W.QHBoxLayout()
        record_layout.addLayout(body, 1)
        side_widget = W.QWidget()
        side_widget.setFixedWidth(265)
        side = W.QVBoxLayout(side_widget)
        side.setContentsMargins(0, 4, 16, 0)
        eyebrow = W.QLabel('SESSION')
        eyebrow.setObjectName('muted')
        side.addWidget(eyebrow)
        self.session_name = W.QLabel('Ready when you are')
        self.session_name.setObjectName('heading')
        self.session_name.setWordWrap(True)
        side.addWidget(self.session_name)
        self.timer = W.QLabel('00:00.0')
        self.timer.setObjectName('timer')
        side.addWidget(self.timer)
        self.remaining = W.QLabel('Set a duration to begin')
        self.remaining.setObjectName('muted')
        side.addWidget(self.remaining)
        self.progress = W.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        side.addWidget(self.progress)
        side.addSpacing(20)
        self.start_button = button('Start recording', self.setup, True)
        self.start_button.setEnabled(False)
        side.addWidget(self.start_button)
        self.stop_button = button('Stop recording', self.stop)
        self.stop_button.setEnabled(False)
        side.addWidget(self.stop_button)
        side.addSpacing(16)
        self.mark_button = button('Mark event  ·  Space', self.mark)
        self.mark_button.setEnabled(False)
        side.addWidget(self.mark_button)
        self.count = W.QLabel('0 observer markers')
        self.count.setObjectName('muted')
        side.addWidget(self.count)
        hint = W.QLabel('One press for a swallow.\nFor talking, mark the start and end.\nRefine each event after recording.')
        hint.setWordWrap(True)
        hint.setObjectName('muted')
        side.addWidget(hint)
        side.addStretch()
        self.filtered = W.QCheckBox('Filtered live view')
        self.filtered.setChecked(True)
        side.addWidget(self.filtered)
        raw_note = W.QLabel('Raw samples are always saved.\nMic: 20–2000 Hz · IMU: 0.5–100 Hz')
        raw_note.setObjectName('muted')
        side.addWidget(raw_note)
        self.review_button = button('Review last session', self.review_last)
        self.review_button.setEnabled(False)
        side.addWidget(self.review_button)
        body.addWidget(side_widget)
        self.graph = SignalPlots()
        body.addWidget(self.graph, 1)
        self.status = W.QLabel('Connect the Teensy 4.1 using a USB data cable.')
        self.status.setObjectName('muted')
        self.status.setWordWrap(True)
        record_layout.addWidget(self.status)
        self.refresh_ports()
        self.usb_timer = QtCore.QTimer(self)
        self.usb_timer.timeout.connect(self.refresh_ports)
        self.usb_timer.start(1500)
        self.clock = QtCore.QTimer(self)
        self.clock.timeout.connect(self.tick)
        self.clock.start(100)
        self.shortcut = QtGui.QShortcut(QtGui.QKeySequence('Space'), self)
        self.shortcut.setAutoRepeat(False)
        self.shortcut.activated.connect(self.mark)
        self.shortcut.setEnabled(False)
        if demo:
            QtCore.QTimer.singleShot(0, self.connect_device)

    def refresh_ports(self):
        if self.worker and self.worker.isRunning():
            return
        current = self.port.currentData()
        available = ports()
        signature = [(p[0], p[1]) for p in available]
        if signature == getattr(self, 'port_signature', None):
            return
        self.port_signature = signature
        self.port.clear()
        for code, label, teensy in sorted(available, key=lambda p: not p[2]):
            self.port.addItem(label, code)
        if self.port.count() == 0:
            self.port.addItem('No USB serial device found', None)
        index = self.port.findData(current)
        if index >= 0:
            self.port.setCurrentIndex(index)

    def connect_device(self):
        if self.worker and self.worker.isRunning():
            if self.recording or self.pending:
                return
            self.worker.command('quit')
            self.connect_button.setEnabled(False)
            return
        demo = self.demo.isChecked()
        port = self.port.currentData()
        if not demo and not port:
            message(self, 'No USB serial device found. Use a USB data cable and upload the included firmware.')
            return
        self.origin = 0
        self.filter_state.clear()
        for buffer in self.buffers.values():
            buffer.clear()
        self.graph.clear_overlays()
        self.graph.draw(np.empty((0, 2)), np.empty((0, 4)))
        self.data_root = HOME / 'recordings' / ('demo' if demo else 'participants')
        self.worker = Acquisition(port or 'DEMO', self.data_root, demo)
        self.worker.status.connect(self.device_status)
        self.worker.batch.connect(self.receive)
        self.worker.begun.connect(self.begun)
        self.worker.saved.connect(self.saved)
        self.worker.failed.connect(self.failed)
        self.worker.finished.connect(self.disconnected)
        self.worker.start()
        self.port.setEnabled(False)
        self.demo.setEnabled(False)
        self.connect_button.setText('Disconnect')
        self.status.setText('Connecting and checking both sensor streams…')

    def device_status(self, text, ready):
        self.ready = ready
        self.connection_status.setText(('●  ' if ready else '○  ') + text)
        self.start_button.setEnabled(ready and not self.recording and not self.pending)
        if not self.recording:
            self.status.setText(text)

    def disconnected(self):
        self.ready = False
        self.connection_status.setText('○  Device disconnected')
        self.start_button.setEnabled(False)
        self.connect_button.setText('Connect device')
        self.connect_button.setEnabled(True)
        self.port.setEnabled(True)
        self.demo.setEnabled(True)

    def setup(self):
        dialog = Setup(self)
        if dialog.exec() != W.QDialog.DialogCode.Accepted:
            return
        if not self.ready or not self.worker.isRunning():
            message(self, 'Device is no longer ready. Reconnect before recording.')
            return
        self.last_participant = dialog.code
        self.pending = True
        self.start_button.setEnabled(False)
        self.connect_button.setEnabled(False)
        self.worker.command('start', (dialog.code, dialog.task, dialog.seconds.value()))

    def begun(self, data):
        self.pending = False
        self.recording = True
        self.tabs.setTabEnabled(1, False)
        self.started, self.seconds, self.origin = data['started'], data['seconds'], data['origin']
        self.markers = []
        self.graph.clear_overlays()
        self.session_name.setText(f'{Path(data["folder"]).parent.name}\n{Path(data["folder"]).name}')
        self.count.setText('0 observer markers')
        self.stop_button.setEnabled(True)
        self.mark_button.setEnabled(True)
        self.shortcut.setEnabled(True)
        self.review_button.setEnabled(False)
        self.status.setText('Recording · begin with normal breathing; keep breathing between activity attempts.')

    def stop(self):
        if self.recording:
            self.worker.command('stop')
            self.stop_button.setEnabled(False)
            self.mark_button.setEnabled(False)
            self.shortcut.setEnabled(False)
            self.status.setText('Finishing recording and saving CSV files…')

    def mark(self):
        if not self.recording or not self.mark_button.isEnabled():
            return
        seconds = time.perf_counter() - self.started
        if seconds > self.seconds:
            return
        self.worker.command('marker')
        self.markers.append(seconds)
        self.count.setText(f'{len(self.markers)} observer markers')
        self.graph.marker(seconds)

    def tick(self):
        if not self.recording:
            return
        elapsed = min(time.perf_counter() - self.started, self.seconds)
        self.timer.setText(f'{int(elapsed // 60):02d}:{elapsed % 60:04.1f}')
        self.remaining.setText(f'{max(0, self.seconds - elapsed):.1f} seconds remaining')
        self.progress.setValue(round(elapsed / self.seconds * 1000))
        if elapsed >= self.seconds:
            self.stop_button.setEnabled(False)
            self.mark_button.setEnabled(False)
            self.shortcut.setEnabled(False)

    def receive(self, payload):
        packets, rates = payload
        grouped = {'mic': [], 'imu': []}
        for kind, index, t, values in packets:
            if kind == 'mic':
                grouped[kind].extend((t + n / rates[0], value) for n, value in enumerate(values))
            else:
                grouped[kind].append((t, *values))
        clipping = False
        for kind, rows in grouped.items():
            if not rows:
                continue
            array = np.asarray(rows, dtype=float)
            rate = rates[0 if kind == 'mic' else 1]
            if kind not in self.filter_state:
                sos = filter_sos(kind, rate)
                zi = sosfilt_zi(sos)[:, :, None] * array[0, 1:][None, None, :]
                self.filter_state[kind] = sos, zi
            sos, zi = self.filter_state[kind]
            filtered, zi = sosfilt(sos, array[:, 1:], axis=0, zi=zi)
            self.filter_state[kind] = sos, zi
            # Keep raw and filtered values; view changes never affect CSV writing.
            combined = np.column_stack([array, filtered])
            self.buffers[kind].append(combined)
            if kind == 'mic':
                clipping = np.any(np.abs(array[:, 1]) >= 32760)
        arrays = []
        for kind in ('mic', 'imu'):
            array = np.concatenate(self.buffers[kind]) if self.buffers[kind] else np.empty((0, 1))
            width = 2 if kind == 'mic' else 4
            if not len(array):
                arrays.append(np.empty((0, width)))
                continue
            latest = array[-1, 0]
            array = array[array[:, 0] >= latest - 10]
            values = array[:, width:] if self.filtered.isChecked() else array[:, 1:width]
            arrays.append(np.column_stack([array[:, 0] - self.origin, values]))
        self.graph.draw(*arrays)
        if len(arrays[0]):
            latest = arrays[0][-1, 0]
            self.graph.plots[0].setXRange(max(0, latest - 10), max(10, latest), padding=0)
            # Retain all markers in Events.csv; only keep recent live overlays.
            for plot, item in self.graph.overlays[:]:
                if isinstance(item, pg.InfiniteLine) and item.value() < latest - 10:
                    plot.removeItem(item)
                    self.graph.overlays.remove((plot, item))
        if clipping:
            self.status.setText('Microphone clipping detected · check the line-input gain before the next session.')

    def saved(self, folder, status):
        self.recording = self.pending = False
        self.tabs.setTabEnabled(1, True)
        self.latest_session = Path(folder)
        self.stop_button.setEnabled(False)
        self.mark_button.setEnabled(False)
        self.shortcut.setEnabled(False)
        self.review_button.setEnabled(True)
        self.connect_button.setEnabled(True)
        self.start_button.setEnabled(self.ready)
        self.status.setText(f'Saved ({status}) · {folder}')
        self.remaining.setText('Recording saved · ready for review')

    def failed(self, error):
        self.pending = False
        self.connect_button.setEnabled(True)
        self.status.setText(error)
        message(self, error)

    def review_last(self):
        if self.latest_session:
            self.tabs.setCurrentWidget(self.review)
            self.review.load(self.latest_session)

    def closeEvent(self, event):
        if self.review.loader and self.review.loader.isRunning():
            message(self, 'Wait for the session to finish loading before closing.')
            event.ignore()
            return
        if self.worker and self.worker.isRunning():
            self.worker.command('quit')
            if not self.worker.wait(4000):
                self.status.setText('Waiting for acquisition to finish saving…')
                event.ignore()
                return
        event.accept()


STYLE = '''
QWidget { background: #0b1521; color: #deE8f3; font-family: 'Segoe UI'; font-size: 13px; }
QLabel#title { font-size: 29px; font-weight: 700; }
QLabel#heading { font-size: 19px; font-weight: 600; }
QLabel#muted { color: #91a6bd; font-size: 12px; line-height: 1.4; }
QLabel#pill { background: #152b37; color: #6edbcb; border-radius: 12px; padding: 9px 14px; }
QLabel#timer { font-size: 43px; font-weight: 600; padding: 12px 0; }
QPushButton { background: #1b2c40; border: 1px solid #2c4259; border-radius: 7px; padding: 7px 12px; font-weight: 600; }
QPushButton:hover { background: #263d55; }
QPushButton#primary { background: #18b5a4; color: #071c23; border-color: #18b5a4; }
QPushButton#primary:hover { background: #57cfbf; }
QPushButton:disabled { background: #142130; color: #52677d; border-color: #1c3043; }
QLineEdit, QComboBox, QDoubleSpinBox { background: #132234; border: 1px solid #30465d; border-radius: 6px; padding: 8px; }
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus { border-color: #18b5a4; }
QComboBox QAbstractItemView { background: #172a3f; selection-background-color: #245462; }
QTabWidget::pane { border: none; }
QTabBar::tab { color: #94a9c0; padding: 12px 24px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #6edbcb; border-color: #18b5a4; }
QProgressBar { border: none; border-radius: 3px; background: #213347; }
QProgressBar::chunk { background: #18b5a4; border-radius: 3px; }
QListWidget { background: #111e2c; border: 1px solid #263c52; border-radius: 7px; }
QListWidget::item { padding: 8px 5px; }
QListWidget::item:selected { background: #234754; }
QCheckBox { spacing: 8px; padding: 6px 0; }
QSplitter::handle { background: #203249; width: 2px; }
QToolTip { color: #e1eaf4; background: #233c53; border: none; }
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()
    app = W.QApplication(sys.argv[:1])
    app.setStyle('Fusion')
    app.setFont(QtGui.QFont('Segoe UI', 10))
    app.setStyleSheet(STYLE)
    pg.setConfigOptions(antialias=False, foreground='#a8b9cc')
    window = MainWindow(args.demo)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
