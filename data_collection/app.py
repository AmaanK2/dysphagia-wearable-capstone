"""Run: python app.py. Use --demo to try the complete workflow without hardware."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from string import Template

import numpy as np
from scipy.signal import sosfiltfilt
try:
    from PyQt6 import QtCore, QtGui, QtWidgets as W
    Signal = QtCore.pyqtSignal
except ImportError:
    from PySide6 import QtCore, QtGui, QtWidgets as W
    Signal = QtCore.Signal
import pyqtgraph as pg

from device import Acquisition, ports, filter_sos
from live_detection import RECOMMENDED_MODEL, available_models
from session import EVENT_FIELDS, safe_name, save_events

HOME = Path(__file__).resolve().parent
ACTIVITIES = {'Dry swallow': 'dry_swallow', 'Talking': 'talking', 'Mixed activities': 'mixed'}
THEME = {
    'paper': '#f7f9fc', 'paper_2': '#eef2f7', 'paper_3': '#e3e9f1',
    'ink': '#151c2a', 'ink_2': '#3a4556', 'muted': '#627086',
    'rule': '#dce2ec', 'edge': '#9caabc', 'key': '#fcfdff',
    'accent': '#2462d8', 'accent_hover': '#1d53be', 'accent_edge': '#174592',
    'accent_wash': '#eaf0fc', 'accent_ink': '#f7f9fc',
    'graphite': '#161c29', 'graphite_ink': '#e3eaf5', 'graphite_muted': '#a6b6d0',
    'series_1': '#91bcff', 'series_2': '#d7e6ff', 'series_3': '#6792d7',
    'error': '#ad343b', 'overlay': '#91bcff', 'selection': '#91bcff',
    'font': 'Segoe UI',
}
COLORS = [THEME['series_1'], THEME['series_2'], THEME['series_3']]
# Live alert colours on the graphite plots; swallow keeps its original green.
EVENT_COLORS = {'swallow': '#70e2b0', 'cough': '#f2a25c', 'talking': '#c8a6ff', 'throat_clear': '#f3d36b'}
EVENT_TEXT = {'swallow': 'Swallow detected', 'cough': 'Cough detected', 'talking': 'Talking started',
              'throat_clear': 'Throat clear detected'}


def button(text, action, primary=False):
    item = W.QPushButton(text)
    item.setMinimumHeight(38)
    if primary:
        item.setObjectName('primary')
    item.clicked.connect(action)
    return item


def message(parent, text):
    W.QMessageBox.warning(parent, 'Swallow Studio', str(text))


class TimeInput(W.QDoubleSpinBox):
    def __init__(self):
        super().__init__()
        self.setButtonSymbols(W.QAbstractSpinBox.ButtonSymbols.NoButtons)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Qt 6.12 can leave a styled NoButtons editor only one pixel wide.
        self.lineEdit().setGeometry(self.rect().adjusted(8, 2, -8, -2))


class SignalPlots(W.QWidget):
    def __init__(self):
        super().__init__()
        layout = W.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plots, self.curves, self.overlays = [], [], []
        for title, label, colors in (
            ('Microphone', 'PCM counts', [COLORS[0]]),
            ('Acceleration', 'm/s\u00b2', COLORS),
        ):
            plot = pg.PlotWidget(background=THEME['graphite'])
            plot.setTitle(title, color=THEME['graphite_ink'], size='10pt')
            plot.setLabel('left', label, color=THEME['graphite_muted'])
            plot.setLabel('bottom', 'Time', units='s', color=THEME['graphite_muted'])
            for axis in ('left', 'bottom'):
                plot.getAxis(axis).setTextPen(THEME['graphite_muted'])
                plot.getAxis(axis).setPen(THEME['graphite_muted'])
            plot.showGrid(x=True, y=True, alpha=.15)
            plot.setMenuEnabled(False)
            plot.getAxis('left').setWidth(68)
            plot.getAxis('left').enableAutoSIPrefix(False)
            plot.getAxis('left').setTickDensity(2)
            plot.getAxis('left').setStyle(textFillLimits=[(0, .95)])
            plot.getPlotItem().setContentsMargins(10, 10, 14, 5)
            curves = []
            if len(colors) == 3:
                plot.addLegend(offset=(-14, 6), colCount=3, verticalSpacing=0,
                               horizontalSpacing=12, labelTextColor=THEME['graphite_ink'])
            for n, color in enumerate(colors):
                curve = plot.plot(pen=pg.mkPen(color, width=1.3), name='XYZ'[n] if len(colors) == 3 else None)
                curve.setDownsampling(auto=True, method='peak')
                curve.setClipToView(True)
                curves.append(curve)
            layout.addWidget(plot, 1)
            self.plots.append(plot)
            self.curves.append(curves)
        self.plots[1].setXLink(self.plots[0])
        self.plots[0].setLabel('bottom', '')

    def draw(self, mic, imu):
        self.curves[0][0].setData(mic[:, 0], mic[:, 1]) if len(mic) else self.curves[0][0].clear()
        for axis, curve in enumerate(self.curves[1]):
            curve.setData(imu[:, 0], imu[:, axis + 1]) if len(imu) else curve.clear()

    def clear_overlays(self):
        for plot, item in self.overlays:
            plot.removeItem(item)
        self.overlays = []

    def marker(self, seconds, detected=False):
        # ``detected`` is False for observer marks, or the detected event name.
        color = EVENT_COLORS.get(detected if isinstance(detected, str) else 'swallow') if detected else THEME['overlay']
        style = QtCore.Qt.PenStyle.SolidLine if detected else QtCore.Qt.PenStyle.DashLine
        for plot in self.plots:
            line = pg.InfiniteLine(seconds, pen=pg.mkPen(color, width=1.3, style=style))
            plot.addItem(line)
            self.overlays.append((plot, line))

    def interval(self, start, end, uncertain=False):
        fill = QtGui.QColor(THEME['overlay'])
        fill.setAlpha(35)
        for plot in self.plots:
            item = pg.LinearRegionItem((start, end), movable=False,
                                       brush=pg.mkBrush(fill),
                                       pen=pg.mkPen(THEME['series_3'] if uncertain else THEME['series_1']))
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
        form.addRow('Custom activity', self.custom)
        form.setRowVisible(self.custom, False)
        self.activity.currentTextChanged.connect(lambda t: form.setRowVisible(self.custom, t == 'Custom'))
        presets = W.QWidget()
        row = W.QHBoxLayout(presets)
        row.setContentsMargins(0, 0, 0, 0)
        self.seconds = TimeInput()
        self.seconds.setRange(1, 1800)
        self.seconds.setDecimals(1)
        self.seconds.setValue(120)
        self.seconds.setSuffix(' seconds')
        for label, value in [('1 minute', 60), ('2 minutes', 120), ('3 minutes', 180)]:
            row.addWidget(button(label, lambda checked=False, v=value: self.seconds.setValue(v)))
        form.addRow('Duration', presets)
        form.addRow('Custom duration', self.seconds)
        self.participant.setToolTip('Reuse this ID on return visits. Session numbers are assigned automatically.')
        self.error = W.QLabel('')
        self.error.setWordWrap(True)
        self.error.setStyleSheet(f"color: {THEME['error']}")
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
    loaded = Signal(object)
    failed = Signal(str)

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
            filtered_data = []
            for kind, array in [('mic', mic), ('imu', imu)]:
                filtered = array.copy()
                if len(array) > 30:
                    rate = 1 / np.median(np.diff(array[:, 0]))
                    filtered[:, 1:] = sosfiltfilt(filter_sos(kind, rate), array[:, 1:], axis=0)
                filtered_data.append(filtered)
            self.loaded.emit((self.folder, mic, imu, events, filtered_data))
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
        selection = QtGui.QColor(THEME['selection'])
        selection.setAlpha(40)
        self.region = pg.LinearRegionItem((0, 1), brush=pg.mkBrush(selection), pen=pg.mkPen(THEME['selection']))
        self.region.setZValue(10)
        self.graph.plots[0].addItem(self.region)
        self.region.sigRegionChanged.connect(self.from_region)
        self.region.hide()
        self.imu_region = pg.LinearRegionItem((0, 1), movable=False, brush=pg.mkBrush(selection), pen=pg.mkPen(THEME['selection']))
        self.graph.plots[1].addItem(self.imu_region)
        self.imu_region.hide()
        sidebar = W.QWidget()
        sidebar.setMinimumWidth(300)
        side = W.QVBoxLayout(sidebar)
        title = W.QLabel('Events')
        title.setObjectName('heading')
        side.addWidget(title)
        hint = W.QLabel('Select a marker, then adjust the blue region.')
        hint.setWordWrap(True)
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
        self.start, self.end = TimeInput(), TimeInput()
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
        self.notice = W.QLabel('')
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
        self.folder, self.mic, self.imu, self.events, self.filtered_data = data
        self.duration = max(self.mic[-1, 0], self.imu[-1, 0])
        self.path.setText(f'{self.folder.parent.name} / {self.folder.name}')
        self.path.setToolTip(str(self.folder))
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
                text = f'Marker at {float(event["time_s"]):.3f} s'
            else:
                text = f'{"Uncertain: " if event["notes"].startswith("[uncertain]") else ""}{event["activity"]}: {event["start_s"]} to {event["end_s"]} s'
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
            self.notice.setText('Saved to Events.csv')
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
        self.setWindowTitle('Swallow Studio')
        self.resize(1200, 820)
        self.setMinimumSize(960, 660)
        self.worker = None
        self.recording = False
        self.pending = False
        self.ready = False
        self.last_participant = 'P01'
        self.latest_session = None
        self.data_root = HOME / 'recordings' / ('demo' if demo else 'participants')
        self.markers = []
        self.origin = 0
        central = W.QWidget()
        self.setCentralWidget(central)
        layout = W.QVBoxLayout(central)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)
        header = W.QHBoxLayout()
        title = W.QLabel('Swallow Studio')
        title.setObjectName('title')
        header.addWidget(title)
        header.addStretch()
        self.connection_status = W.QLabel('Disconnected')
        self.connection_status.setObjectName('muted')
        header.addWidget(self.connection_status)
        layout.addLayout(header)
        self.tabs = W.QTabWidget()
        layout.addWidget(self.tabs, 1)
        record_tab = W.QWidget()
        self.tabs.addTab(record_tab, 'Record')
        self.review = Review()
        self.tabs.addTab(self.review, 'Review')
        record_layout = W.QVBoxLayout(record_tab)
        record_layout.setContentsMargins(0, 12, 0, 0)
        record_layout.setSpacing(12)
        connection_row = W.QHBoxLayout()
        self.port = W.QComboBox()
        self.port.setAccessibleName('USB device')
        connection_row.addWidget(self.port, 1)
        self.demo = W.QCheckBox('Demo')
        self.demo.setChecked(demo)
        self.demo.toggled.connect(self.show_source)
        connection_row.addWidget(self.demo)
        self.connect_button = button('Connect device', self.connect_device)
        self.connect_button.setFixedWidth(148)
        connection_row.addWidget(self.connect_button)
        record_layout.addLayout(connection_row)
        session_row = W.QHBoxLayout()
        self.session_name = W.QLabel('No recording')
        self.session_name.setObjectName('heading')
        self.session_name.setWordWrap(True)
        session_row.addWidget(self.session_name, 1)
        timing = W.QVBoxLayout()
        timing.setSpacing(0)
        self.timer = W.QLabel('00:00.0')
        self.timer.setObjectName('timer')
        self.remaining = W.QLabel('')
        self.remaining.setObjectName('muted')
        timing.addWidget(self.timer)
        timing.addWidget(self.remaining)
        session_row.addLayout(timing)
        session_row.addSpacing(16)
        self.start_button = button('Start recording', self.setup, True)
        self.start_button.setEnabled(False)
        self.stop_button = button('Stop', self.stop)
        self.stop_button.setEnabled(False)
        session_row.addWidget(self.start_button)
        session_row.addWidget(self.stop_button)
        record_layout.addLayout(session_row)
        self.progress = W.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(3)
        record_layout.addWidget(self.progress)
        self.graph = SignalPlots()
        record_layout.addWidget(self.graph, 1)
        actions = W.QHBoxLayout()
        self.mark_button = button('Mark event   [Space]', self.mark)
        self.mark_button.setEnabled(False)
        actions.addWidget(self.mark_button)
        self.count = W.QLabel('0 markers')
        self.count.setObjectName('muted')
        actions.addWidget(self.count)
        actions.addStretch()
        self.filtered = W.QCheckBox('Filter signals')
        self.filtered.setChecked(True)
        self.filtered.setToolTip('Display only. Full-rate raw samples are saved unchanged.')
        actions.addWidget(self.filtered)
        self.review_button = button('Review recording', self.review_last)
        self.review_button.setEnabled(False)
        actions.addWidget(self.review_button)
        record_layout.addLayout(actions)
        self.status = W.QLabel('Connect a device or select Demo.')
        self.status.setObjectName('muted')
        self.status.setWordWrap(True)
        record_layout.addWidget(self.status)
        self.live = W.QWidget()
        self.tabs.insertTab(1, self.live, 'Live Session')
        live_layout = W.QVBoxLayout(self.live)
        live_layout.setContentsMargins(0, 12, 0, 0)
        live_top = W.QHBoxLayout()
        self.live_name = W.QLabel('No recording')
        self.live_name.setObjectName('heading')
        self.live_time = W.QLabel('00:00.0')
        self.live_time.setObjectName('timer')
        self.live_mark = button('Mark event   [Space]', self.mark)
        self.live_stop = button('Stop', self.stop)
        self.live_start = button('Start recording', self.setup, True)
        self.live_start.setEnabled(False)
        self.live_mark.setEnabled(False)
        self.live_stop.setEnabled(False)
        live_top.addWidget(self.live_name, 1)
        for widget in (self.live_time, self.live_start, self.live_mark, self.live_stop):
            live_top.addWidget(widget)
        live_layout.addLayout(live_top)
        model_row = W.QHBoxLayout()
        detector_label = W.QLabel('Detector')
        detector_label.setObjectName('muted')
        self.model_choice = W.QComboBox()
        self.model_choice.setAccessibleName('Swallow detector model')
        self.model_choice.setToolTip('Which trained model raises live swallow alerts. '
                                     'Locked while recording; saved in Session.json.')
        self.model_choice.setMinimumWidth(340)
        detector_label.setBuddy(self.model_choice)
        self.model_refresh = button('Refresh list', self.populate_models)
        model_row.addWidget(detector_label)
        model_row.addWidget(self.model_choice)
        model_row.addWidget(self.model_refresh)
        model_row.addSpacing(16)
        self.events_toggle = W.QCheckBox('Also detect cough, talking, throat clear')
        self.events_toggle.setChecked(True)
        self.events_toggle.setToolTip('IMU event classifier (models/events.json). Locked while recording.')
        self.events_toggle.toggled.connect(self.toggle_events)
        model_row.addWidget(self.events_toggle)
        legend = W.QLabel('  '.join(f'<span style="color:{EVENT_COLORS[k]}">■</span> {k.replace("_", " ").capitalize()}'
                                    for k in EVENT_COLORS))
        legend.setObjectName('muted')
        model_row.addWidget(legend)
        model_row.addStretch()
        live_layout.addLayout(model_row)
        self.populate_models()
        self.model_choice.currentIndexChanged.connect(self.choose_model)
        live_body = W.QHBoxLayout()
        self.live_graph = SignalPlots()
        live_body.addWidget(self.live_graph, 1)
        events = W.QVBoxLayout()
        self.live_alert = W.QLabel('No detections yet')
        self.live_alert.setObjectName('heading')
        self.live_alert.setWordWrap(True)
        self.live_log = W.QListWidget()
        self.live_log.setAccessibleName('Swallow detection log')
        self.live_log.setMinimumWidth(230)
        self.live_log.setMaximumWidth(280)
        events.addWidget(self.live_alert)
        events.addWidget(self.live_log, 1)
        live_body.addLayout(events)
        live_layout.addLayout(live_body, 1)
        self.live_status = W.QLabel('Connect a device in Record to preview the live signals.')
        self.live_status.setObjectName('muted')
        self.live_status.setWordWrap(True)
        live_layout.addWidget(self.live_status)
        self.refresh_ports()
        self.show_source()
        self.usb_timer = QtCore.QTimer(self)
        self.usb_timer.timeout.connect(self.refresh_ports)
        self.usb_timer.start(1500)
        self.clock = QtCore.QTimer(self)
        self.clock.timeout.connect(self.tick)
        self.clock.start(100)
        # Pull the newest bounded frame at 10 Hz. Nothing is queued in Qt
        # when painting is slow, the review tab is open, or the app is minimized.
        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.timeout.connect(self.refresh_plot)
        self.plot_timer.start(100)
        self.shortcut = QtGui.QShortcut(QtGui.QKeySequence('Space'), self)
        self.shortcut.setAutoRepeat(False)
        self.shortcut.activated.connect(self.mark)
        self.shortcut.setEnabled(False)
        if demo:
            QtCore.QTimer.singleShot(0, self.connect_device)

    def populate_models(self):
        current = self.model_choice.currentData()
        self.model_choice.blockSignals(True)
        self.model_choice.clear()
        for label, path in available_models():
            self.model_choice.addItem(label, str(path))
        if not self.model_choice.count():
            self.model_choice.addItem('No detector models found', None)
        index = self.model_choice.findData(current)
        if index < 0:
            # Default to the current wearable model when present, then the recommended mixed model.
            index = next((i for i in range(self.model_choice.count())
                          if str(self.model_choice.itemData(i) or '').endswith(('wearable.json', RECOMMENDED_MODEL))), 0)
        self.model_choice.setCurrentIndex(index)
        self.model_choice.blockSignals(False)
        if self.model_choice.currentData() != current:
            self.choose_model()

    def choose_model(self):
        path = self.model_choice.currentData()
        if path and self.worker and self.worker.isRunning() and not self.recording and not self.pending:
            self.worker.command('model', path)

    def toggle_events(self, enabled):
        if self.worker and self.worker.isRunning() and not self.recording and not self.pending:
            self.worker.command('events', enabled)

    def lock_model_choice(self, locked):
        self.model_choice.setEnabled(not locked)
        self.model_refresh.setEnabled(not locked)
        self.events_toggle.setEnabled(not locked)

    def show_source(self):
        self.port.setVisible(not self.demo.isChecked())

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
        self.reset_live_view('Connecting…')
        self.graph.clear_overlays()
        self.graph.draw(np.empty((0, 2)), np.empty((0, 4)))
        self.data_root = HOME / 'recordings' / ('demo' if demo else 'participants')
        self.worker = Acquisition('DEMO' if demo else port, self.data_root, demo, self.model_choice.currentData())
        self.worker.events_enabled = self.events_toggle.isChecked()
        self.worker.status.connect(self.device_status)
        self.worker.begun.connect(self.begun)
        self.worker.saved.connect(self.saved)
        self.worker.detected.connect(self.detected)
        self.worker.detection_status.connect(self.live_status.setText)
        self.worker.failed.connect(self.failed)
        self.worker.finished.connect(self.disconnected)
        self.worker.start()
        self.port.setEnabled(False)
        self.demo.setEnabled(False)
        self.connect_button.setText('Disconnect')
        self.status.setText('Connecting and checking both sensor streams…')

    def device_status(self, text, ready):
        self.ready = ready
        self.connection_status.setText('Demo ready' if self.demo.isChecked() and ready else ('Ready' if ready else 'Connecting'))
        can_start = ready and not self.recording and not self.pending
        self.start_button.setEnabled(can_start)
        self.live_start.setEnabled(can_start)
        if ready and not self.recording:
            self.live_name.setText('Live preview · Not recording')
        if not self.recording:
            self.status.setText(text)

    def disconnected(self):
        self.ready = False
        self.connection_status.setText('Disconnected')
        self.reset_live_view('Disconnected')
        self.live_status.setText('Connect a device in Record to preview the live signals.')
        self.live_start.setEnabled(False)
        self.start_button.setEnabled(False)
        self.connect_button.setText('Connect device')
        self.connect_button.setEnabled(True)
        self.port.setEnabled(True)
        self.demo.setEnabled(True)
        self.lock_model_choice(False)

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
        self.live_start.setEnabled(False)
        self.connect_button.setEnabled(False)
        self.lock_model_choice(True)
        self.worker.command('start', (dialog.code, dialog.task, dialog.seconds.value()))

    def reset_live_view(self, title):
        self.live_graph.clear_overlays()
        self.live_graph.draw(np.empty((0, 2)), np.empty((0, 4)))
        self.live_name.setText(title)
        self.live_time.setText('00:00.0')
        self.live_log.clear()
        self.live_alert.setText('No detections yet')

    def begun(self, data):
        self.pending = False
        self.recording = True
        self.start_button.setEnabled(False)
        self.connect_button.setEnabled(False)
        self.tabs.setTabEnabled(self.tabs.indexOf(self.review), False)
        self.started, self.seconds, self.origin = data['started'], data['seconds'], data['origin']
        self.markers = []
        self.graph.clear_overlays()
        self.session_name.setText(f'{Path(data["folder"]).parent.name} / {Path(data["folder"]).name}')
        self.count.setText('0 markers')
        self.stop_button.setEnabled(True)
        self.mark_button.setEnabled(True)
        self.shortcut.setEnabled(True)
        self.review_button.setEnabled(False)
        self.status.setText('Recording. Press Space when you observe an event.')
        self.reset_live_view(self.session_name.text())
        self.live_start.setEnabled(False)
        self.lock_model_choice(True)
        self.live_status.setText('Starting detector…')
        self.live_mark.setEnabled(True)
        self.live_stop.setEnabled(True)
        self.tabs.setCurrentWidget(self.live)

    def stop(self):
        if self.recording:
            self.worker.command('stop')
            self.stop_button.setEnabled(False)
            self.mark_button.setEnabled(False)
            self.shortcut.setEnabled(False)
            self.status.setText('Finishing recording and saving CSV files…')
            self.live_mark.setEnabled(False)
            self.live_stop.setEnabled(False)

    def mark(self):
        if not self.recording or not self.mark_button.isEnabled():
            return
        seconds = time.perf_counter() - self.started
        if seconds > self.seconds:
            return
        self.worker.command('marker')
        self.markers.append(seconds)
        self.count.setText(f'{len(self.markers)} markers')
        self.graph.marker(seconds)
        self.live_graph.marker(seconds)

    def detected(self, seconds, event, score):
        text = EVENT_TEXT.get(event, event)
        self.live_alert.setText(f'{text} at {seconds:.2f} s')
        self.live_log.addItem(f'{seconds:7.2f} s   ·   {text}')
        item = self.live_log.item(self.live_log.count() - 1)
        item.setToolTip(f'Model score {score:.2f}')
        item.setForeground(QtGui.QColor(EVENT_COLORS.get(event, THEME['ink'])).darker(140))
        if not self.recording and self.live_log.count() > 100:
            self.live_log.takeItem(0)
        self.live_log.scrollToBottom()
        self.live_graph.marker(seconds, detected=event)
        self.prune_markers(self.live_graph, seconds - 10)

    def tick(self):
        if not self.recording:
            return
        elapsed = min(time.perf_counter() - self.started, self.seconds)
        self.timer.setText(f'{int(elapsed // 60):02d}:{elapsed % 60:04.1f}')
        self.remaining.setText(f'{max(0, self.seconds - elapsed):.1f} seconds remaining')
        self.live_time.setText(self.timer.text())
        self.progress.setValue(round(elapsed / self.seconds * 1000))
        if elapsed >= self.seconds:
            self.live_mark.setEnabled(False)
            self.live_stop.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.mark_button.setEnabled(False)
            self.shortcut.setEnabled(False)

    def refresh_plot(self):
        if not self.worker:
            return
        frame = self.worker.take_preview()
        if frame is None or self.isMinimized():
            return
        current = self.tabs.currentWidget()
        if current is self.live:
            self.receive(frame, self.live_graph)
            if not self.recording and len(frame['raw'][0]):
                elapsed = frame['raw'][0][-1, 0]
                self.live_time.setText(f'{int(elapsed // 60):02d}:{elapsed % 60:04.1f}')
        elif self.tabs.currentIndex() == 0:
            self.receive(frame, self.graph)

    def receive(self, frame, graph):
        # These arrays contain at most 1800 points per plot, prepared by
        # acquisition. Never rebuild the full microphone window in the GUI.
        arrays = []
        for data in frame['filtered' if self.filtered.isChecked() else 'raw']:
            data = data.copy()
            if len(data):
                data[:, 0] -= self.origin
            arrays.append(data)
        graph.draw(*arrays)
        if len(arrays[0]):
            mic = arrays[0]
            latest = mic[-1, 0]
            left, right = max(0, latest - 10), max(10, latest)
            mic_plot = graph.plots[0]
            mic_plot.setXRange(left, right, padding=0)
            # Range the displayed PCM envelope directly. Clip-to-view downsampling
            # can otherwise leave the linked live plot with stale Y bounds.
            visible = mic[(mic[:, 0] >= left) & (mic[:, 0] <= right), 1]
            visible = visible[np.isfinite(visible)]
            if len(visible):
                low, high = float(visible.min()), float(visible.max())
                margin = max(1., .05 * (high - low))
                mic_plot.setYRange(low - margin, high + margin, padding=0)
            self.prune_markers(graph, latest - 10)
        if frame['clipping']:
            self.status.setText('Microphone clipping detected. Lower the input gain before the next recording.')

    def prune_markers(self, graph, cutoff):
        for plot, item in graph.overlays[:]:
            if isinstance(item, pg.InfiniteLine) and item.value() < cutoff:
                plot.removeItem(item)
                graph.overlays.remove((plot, item))

    def saved(self, folder, status):
        self.recording = self.pending = False
        self.tabs.setTabEnabled(self.tabs.indexOf(self.review), True)
        self.latest_session = Path(folder)
        self.stop_button.setEnabled(False)
        self.mark_button.setEnabled(False)
        self.shortcut.setEnabled(False)
        self.review_button.setEnabled(True)
        self.connect_button.setEnabled(True)
        self.start_button.setEnabled(self.ready)
        self.status.setText(f'Saved ({status}): {self.latest_session.parent.name} / {self.latest_session.name}')
        self.status.setToolTip(folder)
        self.remaining.setText('Saved')
        self.live_mark.setEnabled(False)
        self.live_stop.setEnabled(False)
        self.origin = 0
        self.graph.clear_overlays()
        self.reset_live_view('Live preview · Not recording')
        self.live_start.setEnabled(self.ready)
        self.live_status.setText(self.status.text())
        self.lock_model_choice(False)

    def failed(self, error):
        self.pending = False
        self.lock_model_choice(self.recording)
        self.connect_button.setEnabled(True)
        self.start_button.setEnabled(self.ready and not self.recording)
        self.live_start.setEnabled(self.ready and not self.recording)
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


STYLE = Template("""
/* Hallmark v2.0.0; shape: app screen; theme: Cobalt (by request); scope: app; redesign: full */
/* Native Qt adaptation: role tokens, system face, live signals are the signature; no decorative field. */
QWidget { background: $paper; color: $ink; font-family: '$font'; font-size: 13px; }
QLabel#title { font-size: 24px; font-weight: 600; }
QLabel#heading { font-size: 18px; font-weight: 600; }
QLabel#muted { color: $muted; font-size: 12px; }
QLabel#timer { font-size: 34px; font-weight: 600; }
QPushButton { background: $key; border: 1px solid $edge; border-bottom: 3px solid $edge; border-radius: 6px; padding: 6px 14px; font-weight: 600; }
QPushButton:hover { background: $paper_2; }
QPushButton:pressed { background: $paper_3; border-bottom-width: 1px; padding-top: 8px; }
QPushButton:focus { border-color: $accent; }
QPushButton#primary { background: $accent; color: $accent_ink; border-color: $accent_edge; }
QPushButton#primary:hover { background: $accent_hover; }
QPushButton#primary:focus { border-color: $ink; }
QPushButton:disabled, QPushButton#primary:disabled { background: $paper_2; color: $muted; border-color: $rule; }
QLineEdit, QComboBox, QDoubleSpinBox { background: $key; border: 1px solid $edge; border-radius: 6px; padding: 7px; selection-background-color: $accent; selection-color: $accent_ink; }
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus { border-color: $accent; }
QAbstractSpinBox QLineEdit, QComboBox QLineEdit { border: none; padding: 0; background: transparent; }
QLineEdit:disabled, QComboBox:disabled { background: $paper_2; color: $muted; border-color: $rule; }
QComboBox QAbstractItemView { background: $key; color: $ink; selection-background-color: $accent_wash; selection-color: $ink; }
QTabWidget::pane { border: none; border-top: 1px solid $rule; }
QTabBar::tab { color: $muted; padding: 10px 20px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: $accent; border-bottom-color: $accent; }
QTabBar::tab:focus { background: $accent_wash; }
QTabBar::tab:disabled { color: $edge; }
QProgressBar { border: none; background: $paper_3; }
QProgressBar::chunk { background: $accent; }
QListWidget { background: $key; color: $ink; border: 1px solid $rule; border-radius: 6px; }
QListWidget::item { padding: 8px 6px; border-bottom: 1px solid $rule; }
QListWidget::item:selected { background: $accent_wash; color: $ink; }
QListWidget:focus { border-color: $accent; }
QCheckBox { spacing: 8px; padding: 8px 0; }
QCheckBox:focus { color: $accent; }
QCheckBox::indicator { width: 16px; height: 16px; }
QSplitter::handle { background: $rule; width: 1px; }
QToolTip { color: $ink; background: $key; border: 1px solid $rule; padding: 6px; }
""").substitute(THEME)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()
    app = W.QApplication(sys.argv[:1])
    app.setStyle('Fusion')
    app.setFont(QtGui.QFont(THEME['font'], 10))
    app.setStyleSheet(STYLE)
    pg.setConfigOptions(antialias=False, foreground=THEME['graphite_muted'])
    window = MainWindow(args.demo)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
