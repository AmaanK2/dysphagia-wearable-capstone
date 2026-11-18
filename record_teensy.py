"""Record microphone and motion signals with observer-marked activity intervals."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import time
import json
import re
import threading
import webbrowser
from queue import Empty, Queue
from math import sqrt


ACTIVITIES = (
    ("dry_swallow", "Dry swallow (saliva)"),
    ("liquid_swallow", "Liquid swallow"),
    ("food_swallow", "Food swallow"),
    ("cough", "Cough"),
    ("throat_clear", "Throat clear"),
    ("speech", "Speech"),
    ("rest", "Quiet rest"),
    ("bench", "Equipment check"),
)
ACTIVITY_LABELS = dict(ACTIVITIES)


class IntervalMarkers:
    """Keep keypress timing on the same monotonic clock as serial capture."""

    def __init__(self, seconds):
        self.seconds = seconds
        self.lock = threading.Lock()
        self.started_at = None
        self.active = False
        self.open_start = None
        self.intervals = []

    def start(self, started_at):
        with self.lock:
            self.started_at = started_at
            self.active = True

    def toggle(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            if not self.active:
                return None
            offset = now - self.started_at
            if not 0 <= offset < self.seconds:
                return None
            if self.open_start is None:
                self.open_start = offset
                return "started"
            self.intervals.append((self.open_start, offset))
            self.open_start = None
            return "ended"

    def stop(self):
        with self.lock:
            self.active = False
            if self.open_start is not None:
                self.intervals.append((self.open_start, None))
                self.open_start = None

    def snapshot(self):
        with self.lock:
            return list(self.intervals)

    def counts(self):
        with self.lock:
            completed = sum(end is not None for _, end in self.intervals)
            incomplete = sum(end is None for _, end in self.intervals)
            return completed, incomplete + int(self.open_start is not None)


def capture(port_name, seconds, raw_path, on_start=None, on_stop=None):
    import serial

    with serial.Serial(port_name, 115200, timeout=0.2) as port:
        time.sleep(0.5)
        port.reset_input_buffer()
        with raw_path.open("wb") as raw:
            port.write(b"S")
            started_at = time.monotonic()
            try:
                if on_start:
                    on_start(started_at)
                print(f"Recording for {seconds:g} seconds.", flush=True)
                deadline = started_at + seconds
                while time.monotonic() < deadline:
                    raw.write(port.readline())
            finally:
                try:
                    port.write(b"X")
                finally:
                    if on_stop:
                        on_stop()


def save_events(path, intervals, activity):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("timestamp_s", "record_type", "activity", "start_s", "end_s"))
        for start, end in intervals:
            start_text = f"{start:.6f}"
            writer.writerow((start_text, "marker", activity, start_text,
                             f"{end:.6f}" if end is not None else ""))


def parse_recording(raw_path):
    mic, imu, imu_times = [], [], []
    device_start_us = mic_fs = None
    with raw_path.open(encoding="ascii", errors="replace") as raw:
        for line in raw:
            if not line.endswith("\n"):
                continue
            row = line.strip().split(",")
            try:
                if row[0] == "S" and len(row) >= 4:
                    device_start_us = int(row[1])
                    mic_fs = float(row[2])
                elif row[0] == "A" and len(row) == 130:
                    mic.extend([int(value) for value in row[2:]])
                elif row[0] == "I" and len(row) == 5:
                    imu.append(tuple(float(value) for value in row[2:]))
                    imu_times.append(((int(row[1]) - device_start_us) % (2 ** 32)) / 1e6
                                     if device_start_us is not None else None)
            except ValueError:
                continue
    return mic, imu, mic_fs, imu_times


def find_teensy_port():
    from serial.tools import list_ports

    matches = [port.device for port in list_ports.comports()
               if port.vid == 0x16C0
               or "teensy" in (port.description or "").lower()
               or "pjrc" in (port.manufacturer or "").lower()
               or port.device.startswith("/dev/cu.usbmodem")]
    if len(matches) != 1:
        raise ValueError("Connect one Teensy, or enter its port in Device settings.")
    return matches[0]


def record(seconds, participant, activity, test, port_name=None,
           on_start=None, on_stop=None, get_markers=None, show_plot=False):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", participant):
        raise ValueError("Participant code must use letters, numbers, underscores or hyphens.")
    if activity not in ACTIVITY_LABELS:
        raise ValueError(f"Choose an activity from: {', '.join(ACTIVITY_LABELS)}")
    if activity == "bench" and not test:
        raise ValueError("Equipment check is available in Test mode only.")
    if not 0 < seconds < float("inf"):
        raise ValueError("Duration must be positive and finite.")
    if port_name is None:
        port_name = find_teensy_port()

    import matplotlib
    if not show_plot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = (Path(__file__).resolve().parent / "data" / ("test" if test else "pilot")
              / participant / f"{stamp}_{activity}")
    folder.mkdir(parents=True)
    raw_path = folder / "recording.txt"
    metadata = {"participant": participant, "activity": activity,
                "activity_label": ACTIVITY_LABELS[activity],
                "requested_seconds": seconds, "port": port_name,
                "mode": "test" if test else "dataset",
                "event_schema": "observer_start_end_v1"}
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    capture(port_name, seconds, raw_path, on_start, on_stop)
    intervals = get_markers() if get_markers else []
    save_events(folder / "events.csv", intervals, activity)

    mic, imu, mic_fs, imu_times = parse_recording(raw_path)
    print(f"Saved {len(mic)} mic samples and {len(imu)} IMU readings.")
    print(f"Raw recording: {raw_path}")
    completed = sum(end is not None for _, end in intervals)
    incomplete = len(intervals) - completed
    print(f"Activity intervals: {completed} complete, {incomplete} incomplete.")
    if not mic or not imu:
        raise ValueError("One or both sensor streams were missing. The raw capture is saved.")

    fig, (audio_plot, motion_plot) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    block_size = 1024
    blocks = [mic[start:start + block_size] for start in range(0, len(mic), block_size)]
    rms = [sqrt(sum(value * value for value in block) / len(block)) for block in blocks]
    mic_fs = mic_fs if mic_fs and mic_fs > 0 else len(mic) / seconds
    rms_times = [(start + len(block) / 2) / mic_fs
                 for start, block in zip(range(0, len(mic), block_size), blocks)]
    audio_plot.plot(rms_times, rms, linewidth=1)
    audio_plot.set(title="Contact microphone (RMS over 1024 samples)", ylabel="RMS")
    if any(t is None for t in imu_times):
        imu_times = [i * seconds / len(imu) for i in range(len(imu))]
    for axis, values in zip("XYZ", zip(*imu)):
        motion_plot.plot(imu_times, values, label=axis, linewidth=0.8)
    motion_plot.set(title="IMU", xlabel="Seconds from recording start", ylabel="Acceleration")
    motion_plot.legend()
    for index, (start, end) in enumerate(intervals, 1):
        for plot in (audio_plot, motion_plot):
            if end is None:
                plot.axvline(start, color="#ba4c35", linestyle="--", linewidth=1.5)
            else:
                plot.axvspan(start, end, color="#48a883", alpha=0.25)
        audio_plot.annotate(f"{index}. {ACTIVITY_LABELS[activity]}{' (start only)' if end is None else ''}",
                            (start, 1), xycoords=("data", "axes fraction"),
                            xytext=(3, -12), textcoords="offset points", fontsize=8,
                            color="#8a3527" if end is None else "#1c6950")
    fig.suptitle(f"{ACTIVITY_LABELS[activity]} — observer-marked intervals (* start only)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    graph_path = folder / "recording.png"
    fig.savefig(graph_path, dpi=150)
    print(f"Graph: {graph_path}")
    if show_plot:
        plt.show()
    plt.close(fig)
    return graph_path, completed, incomplete


def open_recorder():
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Swallow recording")
    width, height = 760, 610
    x = max(0, (root.winfo_screenwidth() - width) // 2)
    y = max(0, (root.winfo_screenheight() - height) // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    root.minsize(660, 540)
    root.configure(bg="#eef5f6")
    hero = tk.Canvas(root, height=120, bg="#123445", highlightthickness=0)
    hero.pack(fill="x")
    hero.create_text(30, 42, anchor="w", text="Swallow Recorder",
                     fill="#ffffff", font=("Helvetica", 27, "bold"))
    hero.create_text(32, 83, anchor="w", text="Contact microphone + motion sensor",
                     fill="#acdcd6", font=("Helvetica", 15))
    hero.create_line(535, 65, 564, 65, 575, 54, 586, 76, 598, 32,
                     610, 94, 622, 54, 633, 69, 647, 65, 708, 65,
                     fill="#68e3cc", width=3)
    root.after_idle(root.lift)
    style = ttk.Style(root)
    style.configure("Recorder.TFrame", background="#eef5f6")
    style.configure("Recorder.TLabel", background="#eef5f6",
                    foreground="#123445", font=("Helvetica", 14))
    style.configure("Recorder.TButton", font=("Helvetica", 15, "bold"), padding=8)
    form = ttk.Frame(root, padding=(30, 18), style="Recorder.TFrame")
    form.pack(fill="both", expand=True)
    form.columnconfigure(1, weight=1)
    ttk.Label(form, text="Participant code (no name)", style="Recorder.TLabel").grid(
        row=0, column=0, sticky="w", pady=5)
    participant = ttk.Entry(form, width=28, font=("Helvetica", 14))
    participant.grid(row=0, column=1, sticky="ew")

    ttk.Label(form, text="Activity", style="Recorder.TLabel").grid(
        row=1, column=0, sticky="w", pady=5)
    activity_label = tk.StringVar(value=ACTIVITY_LABELS["dry_swallow"])
    activity_field = ttk.Combobox(form, textvariable=activity_label, state="readonly",
                                  font=("Helvetica", 14))
    activity_field.grid(row=1, column=1, sticky="ew")

    ttk.Label(form, text="Recording length (seconds)", style="Recorder.TLabel").grid(
        row=2, column=0, sticky="w", pady=5)
    duration = ttk.Entry(form, width=28, font=("Helvetica", 14))
    duration.insert(0, "20")
    duration.grid(row=2, column=1, sticky="ew")

    mode = tk.StringVar(value="Test")
    ttk.Label(form, text="Save as", style="Recorder.TLabel").grid(
        row=3, column=0, sticky="w", pady=8
    )
    mode_field = ttk.Combobox(form, textvariable=mode, values=("Test", "Dataset"),
                              state="readonly", font=("Helvetica", 14))
    mode_field.grid(row=3, column=1, sticky="ew")
    destination = tk.StringVar()

    def show_destination(*_):
        folder = "test" if mode.get() == "Test" else "pilot"
        destination.set(f"Saves in data/{folder}/<participant code>/")
        choices = [(code, label) for code, label in ACTIVITIES
                   if mode.get() == "Test" or code != "bench"]
        activity_field["values"] = [label for _, label in choices]
        if activity_label.get() not in activity_field["values"]:
            activity_label.set(ACTIVITY_LABELS["dry_swallow"])

    mode.trace_add("write", show_destination)
    show_destination()
    ttk.Label(form, textvariable=destination, style="Recorder.TLabel").grid(
        row=4, column=0, columnspan=2, sticky="w", pady=6
    )

    advanced = ttk.Frame(form, style="Recorder.TFrame")
    advanced.grid(row=5, column=0, columnspan=2, sticky="ew")
    advanced.columnconfigure(1, weight=1)
    port = ttk.Entry(advanced, font=("Helvetica", 13))
    ttk.Label(advanced, text="Device port (leave blank for automatic)",
              style="Recorder.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 10))
    port.grid(row=0, column=1, sticky="ew")
    advanced.grid_remove()

    def toggle_advanced():
        if advanced.winfo_ismapped():
            advanced.grid_remove()
        else:
            advanced.grid()

    ttk.Button(form, text="Device settings", command=toggle_advanced).grid(
        row=6, column=0, columnspan=2, sticky="w", pady=(0, 4))

    ttk.Label(form, text="During recording: press SPACE at the start and again at the end.",
              wraplength=680, style="Recorder.TLabel").grid(
        row=7, column=0, columnspan=2, sticky="w", pady=5)
    marker_count = tk.StringVar(value="Completed: 0  •  In progress: no")
    status = tk.StringVar(value="Connect the sensor, then start a recording.")
    markers = None
    space_down = False
    last_space = float("-inf")
    updates = Queue()

    def refresh_count():
        completed, incomplete = markers.counts() if markers else (0, 0)
        marker_count.set(f"Completed: {completed}  •  In progress/incomplete: {incomplete}")
        event_button.configure(text="Mark activity end" if incomplete else "Mark activity start")

    def mark(event=None):
        nonlocal space_down, last_space
        if markers is None:
            return None
        now = time.monotonic()
        if event is not None:
            if space_down or now - last_space < 0.15:
                return "break"
            space_down = True
            last_space = now
        action = markers.toggle(now)
        if action:
            refresh_count()
            status.set("Interval started — press SPACE again at the end." if action == "started"
                       else "Interval saved — press SPACE to start the next one.")
        root.focus_set()
        return "break" if event is not None else None

    def release_space(event):
        nonlocal space_down
        space_down = False

    def start():
        nonlocal markers, space_down, last_space
        try:
            seconds = float(duration.get())
        except ValueError:
            status.set("Enter a numeric recording length.")
            return
        if not 0 < seconds < float("inf"):
            status.set("Recording length must be positive.")
            return
        participant_code = participant.get().strip()
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", participant_code):
            status.set("Use a participant code with letters, numbers, - or _. Do not enter a name.")
            return
        activity = next((code for code, label in ACTIVITIES if label == activity_label.get()), None)
        if activity is None:
            status.set("Choose an activity from the list.")
            return
        test = mode.get() == "Test"
        if activity == "bench" and not test:
            status.set("Equipment check is available in Test mode only.")
            return
        port_name = port.get().strip() or None
        markers = IntervalMarkers(seconds)
        space_down = False
        last_space = float("-inf")
        refresh_count()
        button.state(["disabled"])
        for field in (participant, activity_field, duration, mode_field, port):
            field.state(["disabled"])
        status.set("Connecting to the sensor…")

        def on_start(started_at):
            markers.start(started_at)
            updates.put(("started", None))

        def on_stop():
            markers.stop()
            updates.put(("stopped", None))

        def worker():
            try:
                result = record(seconds, participant_code, activity, test, port_name,
                                on_start, on_stop, markers.snapshot)
                updates.put(("done", result))
            except Exception as error:
                markers.stop()
                updates.put(("error", str(error)))

        threading.Thread(target=worker, daemon=True).start()
        root.focus_set()

    button = ttk.Button(form, text="Start recording", command=start, style="Recorder.TButton")
    button.grid(row=8, column=0, columnspan=2, sticky="ew", pady=6)
    event_button = ttk.Button(form, text="Mark activity start", command=mark,
                              style="Recorder.TButton", takefocus=False)
    event_button.grid(row=9, column=0, columnspan=2, sticky="ew", pady=6)
    event_button.state(["disabled"])
    ttk.Label(form, textvariable=marker_count, style="Recorder.TLabel").grid(
        row=10, column=0, columnspan=2, sticky="w", pady=5)
    ttk.Label(form, textvariable=status, wraplength=680, style="Recorder.TLabel").grid(
        row=11, column=0, columnspan=2, sticky="w", pady=8)

    def check():
        nonlocal markers
        try:
            while True:
                kind, value = updates.get_nowait()
                if kind == "started":
                    event_button.state(["!disabled"])
                    status.set("Recording — press SPACE to start an interval.")
                elif kind == "stopped":
                    event_button.state(["disabled"])
                    refresh_count()
                    status.set("Preparing graph…")
                else:
                    button.state(["!disabled"])
                    event_button.state(["disabled"])
                    for field in (participant, activity_field, duration, mode_field, port):
                        field.state(["!disabled"])
                    if kind == "done":
                        graph_path, completed, incomplete = value
                        status.set(f"Saved {completed} intervals; {incomplete} need an end time. Graph: {graph_path}")
                        webbrowser.open(graph_path.as_uri())
                    else:
                        status.set(value)
                    markers = None
        except Empty:
            pass
        root.after(100, check)

    def bind_space_first(widget):
        widget.bindtags(("RecorderSpace",) + tuple(tag for tag in widget.bindtags()
                                               if tag != "RecorderSpace"))
        for child in widget.winfo_children():
            bind_space_first(child)

    bind_space_first(root)
    root.bind_class("RecorderSpace", "<KeyPress-space>", mark)
    root.bind_class("RecorderSpace", "<KeyRelease-space>", release_space)
    root.after(100, check)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="Teensy USB serial port")
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--gui", action="store_true", help="Open the recording window")
    parser.add_argument("--participant", default="P01")
    parser.add_argument("--activity", default="dry_swallow", choices=ACTIVITY_LABELS)
    parser.add_argument("--save-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--test", action="store_true", help="Save to data/test")
    args = parser.parse_args()
    if args.gui:
        open_recorder()
        return
    try:
        from serial import SerialException
    except ImportError:
        parser.exit(1, "Install dependencies with python3 -m pip install pyserial matplotlib\n")
    try:
        record(args.seconds, args.participant, args.activity, args.test,
               args.port, show_plot=not args.save_only)
    except (ValueError, SerialException, ImportError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
