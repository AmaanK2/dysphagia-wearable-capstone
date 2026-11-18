"""Capture a Teensy bench recording and plot the mic and IMU streams."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import time
import json
import re
import subprocess
import threading
from queue import Empty, Queue
from math import sqrt


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


def save_events(path, marker_times):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("timestamp_s", "record_type", "activity", "start_s", "end_s"))
        for timestamp in marker_times:
            writer.writerow((f"{timestamp:.6f}", "marker", "dry_swallow", "", ""))


def parse_recording(raw_path):
    mic, imu = [], []
    with raw_path.open(encoding="ascii", errors="replace") as raw:
        for line in raw:
            if not line.endswith("\n"):
                continue
            row = line.strip().split(",")
            try:
                if row[0] == "A" and len(row) == 130:
                    mic.extend([int(value) for value in row[2:]])
                elif row[0] == "I" and len(row) == 5:
                    imu.append(tuple(float(value) for value in row[2:]))
            except ValueError:
                continue
    return mic, imu


def record(seconds, participant, activity, test, port_name=None,
           on_start=None, on_stop=None, get_markers=None, show_plot=False):
    for label in (participant, activity):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", label):
            raise ValueError("Use letters, numbers, underscores or hyphens for participant and activity.")
    if not 0 < seconds < float("inf"):
        raise ValueError("Duration must be positive and finite.")
    if port_name is None:
        ports = sorted(Path("/dev").glob("cu.usbmodem*"))
        if len(ports) != 1:
            raise ValueError("Pass a Teensy port shown by ls /dev/cu.*")
        port_name = str(ports[0])

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
                "requested_seconds": seconds, "port": port_name,
                "mode": "test" if test else "dataset"}
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    capture(port_name, seconds, raw_path, on_start, on_stop)
    marker_times = get_markers() if get_markers else []
    save_events(folder / "events.csv", marker_times)

    mic, imu = parse_recording(raw_path)
    print(f"Saved {len(mic)} mic samples and {len(imu)} IMU readings.")
    print(f"Raw recording: {raw_path}")
    print(f"Markers: {len(marker_times)}")
    if not mic or not imu:
        raise ValueError("One or both sensor streams were missing. The raw capture is saved.")

    fig, (audio_plot, motion_plot) = plt.subplots(2, 1, figsize=(10, 6))
    block_size = 1024
    blocks = [mic[start:start + block_size] for start in range(0, len(mic), block_size)]
    rms = [sqrt(sum(value * value for value in block) / len(block)) for block in blocks]
    audio_plot.plot(range(0, len(mic), block_size), rms, linewidth=1)
    audio_plot.set(title="Contact microphone (RMS over 1024 samples)", xlabel="Mic sample index", ylabel="RMS")
    for axis, values in zip("XYZ", zip(*imu)):
        motion_plot.plot(values, label=axis, linewidth=0.8)
    motion_plot.set(title="IMU", xlabel="IMU sample index", ylabel="Acceleration")
    motion_plot.legend()
    fig.tight_layout()
    graph_path = folder / "recording.png"
    fig.savefig(graph_path, dpi=150)
    print(f"Graph: {graph_path}")
    if show_plot:
        plt.show()
    plt.close(fig)
    return graph_path, len(marker_times)


def open_recorder():
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Swallow recording")
    width, height = 760, 540
    x = max(0, (root.winfo_screenwidth() - width) // 2)
    y = max(0, (root.winfo_screenheight() - height) // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    root.minsize(660, 480)
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
    fields = {}
    for label, default in (("Participant", "P01"), ("Activity", "dry_swallow"),
                           ("Duration (seconds)", "20"), ("Port (optional)", "")):
        row = len(fields)
        ttk.Label(form, text=label, style="Recorder.TLabel").grid(row=row, column=0, sticky="w", pady=5)
        field = ttk.Entry(form, width=28, font=("Helvetica", 14))
        field.insert(0, default)
        field.grid(row=row, column=1, sticky="ew")
        fields[label] = field

    mode = tk.StringVar(value="Test")
    ttk.Label(form, text="Save to", style="Recorder.TLabel").grid(
        row=4, column=0, sticky="w", pady=8
    )
    ttk.Combobox(form, textvariable=mode, values=("Test", "Dataset"),
                 state="readonly", font=("Helvetica", 14)).grid(
        row=4, column=1, sticky="ew"
    )
    destination = tk.StringVar()
    def show_destination(*_):
        folder = "test" if mode.get() == "Test" else "pilot"
        destination.set(f"Project folder / data / {folder} / participant ID")
    mode.trace_add("write", show_destination)
    show_destination()
    ttk.Label(form, textvariable=destination, style="Recorder.TLabel").grid(
        row=5, column=0, columnspan=2, sticky="w", pady=6
    )

    status = tk.StringVar(value="Connect the Teensy, then start a trial.")
    ttk.Label(form, textvariable=status, wraplength=660, style="Recorder.TLabel").grid(
        row=8, column=0, columnspan=2, sticky="w", pady=8
    )
    marker_count = tk.StringVar(value="Markers: 0  •  Press Space during recording")
    ttk.Label(form, textvariable=marker_count, style="Recorder.TLabel").grid(
        row=7, column=0, columnspan=2, sticky="w", pady=8
    )
    state_lock = threading.Lock()
    state = {"started_at": None, "deadline": None, "active": False, "markers": []}
    space_down = False
    updates = Queue()

    def mark(event):
        nonlocal space_down
        if space_down:
            return "break"
        space_down = True
        now = time.monotonic()
        with state_lock:
            if not state["active"] or now >= state["deadline"]:
                return "break"
            state["markers"].append(now - state["started_at"])
            count = len(state["markers"])
        marker_count.set(f"Markers: {count}  •  Press Space during recording")
        return "break"

    def release_space(event):
        nonlocal space_down
        space_down = False

    root.bind("<KeyPress-space>", mark)
    root.bind("<KeyRelease-space>", release_space)

    def start():
        nonlocal space_down
        try:
            seconds = float(fields["Duration (seconds)"].get())
        except ValueError:
            status.set("Duration must be a number of seconds.")
            return
        participant = fields["Participant"].get()
        activity = fields["Activity"].get()
        test = mode.get() == "Test"
        port = fields["Port (optional)"].get().strip() or None
        with state_lock:
            state.update(started_at=None, deadline=None, active=False, markers=[])
        space_down = False
        marker_count.set("Markers: 0  •  Press Space during recording")
        button.state(["disabled"])
        status.set("Connecting to Teensy…")

        def on_start(started_at):
            with state_lock:
                state.update(started_at=started_at, deadline=started_at + seconds,
                             active=True)
            updates.put(("started", None))

        def on_stop():
            with state_lock:
                state["active"] = False
            updates.put(("stopped", None))

        def get_markers():
            with state_lock:
                return list(state["markers"])

        def worker():
            try:
                graph_path, count = record(seconds, participant, activity, test, port,
                                           on_start, on_stop, get_markers)
                updates.put(("done", (graph_path, count)))
            except Exception as error:
                updates.put(("error", str(error)))

        threading.Thread(target=worker, daemon=True).start()

    def check():
        try:
            while True:
                kind, value = updates.get_nowait()
                if kind == "started":
                    status.set("Recording… press Space to mark each swallow.")
                elif kind == "stopped":
                    status.set("Preparing graph…")
                elif kind == "done":
                    graph_path, count = value
                    button.state(["!disabled"])
                    status.set(f"Saved {count} markers. Graph: {graph_path}")
                    subprocess.Popen(["open", str(graph_path)])
                else:
                    button.state(["!disabled"])
                    status.set(value)
        except Empty:
            pass
        root.after(100, check)

    button = ttk.Button(form, text="Start recording", command=start, style="Recorder.TButton")
    button.grid(row=6, column=0, columnspan=2, sticky="ew", pady=8)
    root.after(100, check)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="Teensy USB serial port")
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--gui", action="store_true", help="Open the recording window")
    parser.add_argument("--participant", default="P01")
    parser.add_argument("--activity", default="bench")
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
