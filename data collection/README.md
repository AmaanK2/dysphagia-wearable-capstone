# Swallow Studio

One folder for Teensy acquisition, live microphone/acceleration plots, observer markers, and event review. Existing training and recorder code elsewhere in the repository is unchanged. Password access is deferred as requested.

## Start here

Use Python 3.12 or newer. Open a terminal **inside this folder**:

```powershell
py -3 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python app.py
```

After setup, double-click `start.bat` on Windows. On macOS/Linux, use `python3` and `.venv/bin/python` instead.

To try recording and reviewing without a device:

```powershell
.venv\Scripts\python app.py --demo
```

Demo mode uses synthetic signals and saves under `recordings/demo/`. Participant sessions use `recordings/participants/`. Neither directory is committed to Git.

## Prepare the Teensy once

The included firmware targets the supplied parts list:

| Component | Connection / setting |
| --- | --- |
| Teensy 4.1 | Native USB port; a USB **data** cable |
| Teensy Audio Shield Rev D | Stacked with the headers; SGTL5000 line input |
| TE CM-01B contact microphone | Amplified signal to **left LINE IN**, ground to common ground, supply per the microphone's pinout/specification; the supplied plan specifies 5 V VIN |
| Adafruit ISM330DHCX | 3.3 V, GND, SDA = pin 18, SCL = pin 19; default I²C address 0x6A |

Use the microphone manufacturer's pinout, not guessed wire colours. The audio shield and IMU share I²C; their addresses differ. No extra interrupt wire is required by this implementation.

1. Install Arduino IDE with [PJRC Teensy support](https://www.pjrc.com/teensy/td_download.html). The Boards Manager URL is `https://www.pjrc.com/teensy/package_teensy_index.json`.
2. In Library Manager, install **Adafruit LSM6DS** and accept its dependencies: Adafruit Unified Sensor and Adafruit BusIO. The Audio library comes with Teensy support.
3. Open `firmware/SwallowRecorder/SwallowRecorder.ino`. Select **Teensy 4.1**, **USB Type: Serial**, and **600 MHz**. Upload it.
4. Close Arduino Serial Monitor so the GUI can open the port. Launch the GUI and click **Connect device**. USB ports refresh automatically; Teensy ports are listed first.

This app requires the included **V1 firmware**. The older repository recorder's packets do not include enough documented timing information to assume compatibility. Uploading this sketch replaces the Teensy's previous program; keep a copy of that program if you need it later.

Firmware defaults: microphone signed 16-bit PCM at 44.1 kHz; acceleration at nominal 833 Hz, ±2 g, reported in m/s². Axes are the sensor's native X/Y/Z; use the same placement and orientation every session. Gyroscope data is not collected. `MIC_CHANNEL` selects left/right input, `IMU_ADDRESS` selects 0x6A/0x6B, and `lineInLevel(5)` sets fixed input gain. Change these only when matching the actual wiring or after a bench gain check.

References: [PJRC audio shield](https://www.pjrc.com/store/teensy3_audio.html), [Adafruit sensor setup](https://learn.adafruit.com/lsm6dsox-and-ism330dhc-6-dof-imu/arduino).

## Collect a session

1. Place the sensors consistently. Connect the device. **Ready** requires valid microphone samples and three finite acceleration values, with both streams received within the last 0.75 seconds. USB enumeration alone does not enable recording.
2. Click **Start recording**. Enter the participant ID, choose **Dry swallow**, **Talking**, **Mixed activities**, or **Custom**, and set the duration. Use a minute preset or enter a custom value in seconds, up to 30 minutes.
3. Confirm **Begin recording**. The timer and plots run live. Begin with normal breathing; leave normal breathing between attempts.
4. Press **Space** once per observed swallow. For longer activities, press at the apparent start and end. Every press is an independent approximate marker; no automatic pairing or swallow classification is applied.
5. Wait for the timer or click **Stop recording**. The app drains pending packets briefly, finishes the three CSVs, and enables **Review last session**. Disconnecting or closing retains available samples and marks interrupted sessions in `Session.json`.

Session numbering is participant-wide, across all activities and days. Reuse the participant ID on return visits. For example:

| Recording | Folder |
| --- | --- |
| First dry-swallow session | `recordings/participants/P01/S01_dry_swallow/` |
| Next talking session | `recordings/participants/P01/S02_talking/` |
| Next mixed session | `recordings/participants/P01/S03_mixed/` |

Choose a participant code without names. Codes and custom activities accept letters, numbers, underscores and hyphens. Custom activity spaces become underscores. Existing session folders are never overwritten.

## Files and timing

| File | Columns / meaning |
| --- | --- |
| `Mic.csv` | `sample_idx,time_s,mic_raw` — signed 16-bit PCM counts, unchanged |
| `IMU.csv` | `sample_idx,time_s,ax,ay,az` — unfiltered acquisition acceleration in m/s², including gravity |
| `Events.csv` | `record_type,activity,time_s,start_s,end_s,notes` |
| `Session.json` | Rates, device origin, requested/saved duration, counts, demo flag and completion/interruption status |

All CSV times are seconds relative to **one session origin**, not independent zeroes per signal. The first saved microphone and IMU samples may begin slightly after zero because of their different sample rates. `sample_idx` starts at zero for each saved stream and preserves subsequent sequence gaps. Session time zero is the estimated device time when the tester confirms recording.

Live markers use `record_type=marker`, with `time_s` populated. Reviewed events use `record_type=label`, with `start_s` and `end_s` populated. Uncertain labels retain the activity and use a `[uncertain]` prefix in notes. CSV headers and filenames use the exact capitalization above.

Samples and markers are written during recording and flushed at least once per second; markers flush immediately. An abrupt power loss can lose the latest unflushed samples. A `recording` status left in `Session.json` indicates an unclean interruption; preserve that folder for investigation. A completed status means software recording completed, not that signal quality or physiological labels have been verified.

## Review and label

Open a saved session with **Review last session** or **Open session**. The microphone and X/Y/Z signals share the timeline. Orange dashed lines are original observer presses; shaded bands are reviewed labels.

Select a marker, zoom using the mouse wheel, and drag the blue selection region to the estimated start and end. Exact times can also be typed. Choose the activity for that event and click **Save event**. This works for both single-activity and mixed sessions. Each swallow needs its own label. For talking, select the interval between the two approximate presses and save one talking label.

Select an existing label to update its boundaries, activity or notes. Mark **Uncertain event** and add a note when confidence is low. **Delete selected label** removes only reviewed labels. Every successful add, update and deletion saves atomically to `Events.csv`; original marker rows are preserved. Unsaved selection changes are not labels. Review cannot open an actively recording session. Review one session in one app instance at a time.

## Filtering and accuracy

CSV values are never smoothed, resampled or replaced by filtered values. Live view uses stateful second-order Butterworth bandpass designs: microphone 20–2000 Hz; acceleration 0.5–100 Hz. This suppresses microphone DC/low-frequency movement and acceleration gravity/slow drift while retaining a broad vibration band. These are initial visualization settings, not validated optimal filters for this device or trained models. Turn off the filter checkbox to inspect raw signals.

The live filter is causal and has frequency-dependent delay. Review uses forward/backward filtering to avoid that phase shift; edge transients remain possible. Inspect the raw view around uncertain boundaries. Plot downsampling preserves visible peaks and affects rendering only.

Both sensors are timestamped on the Teensy's microsecond clock. Audio sample times are reconstructed from the I²S sample clock and an audio-block callback anchor. IMU timestamps mark the acquisition read after data-ready polling; they are not hardware-latched sensor conversion timestamps. Codec group delay, sensor filtering, callback latency, USB delay and observer reaction are **not calibrated out**. Observer times use a host clock mapped to the latest device timestamp, so they remain approximate, exactly as in the collection plan. Do not interpret displayed decimal places as measured synchronization precision.

The firmware explicitly reports audio-buffer overflow, I²C failures and excessive IMU read delays. The host rejects malformed packets, non-finite samples, clock reversals, device restarts and sequence gaps; during a recording it preserves available data as incomplete. Sequence checks do not prove that the sensor never overwrote an internal sample. The firmware polls data-ready without an IMU FIFO; delay checks reduce, but cannot eliminate, that risk.

Before participant collection, bench-check a known common mechanical/acoustic event on both sensors, measure timing offset/drift, confirm sample counts and a stable 1–2 minute capture, verify axis orientation, and inspect for clipping/flat signals. Structural **Ready** confirms packets, not microphone contact, correct placement or useful physiological signal quality. No physical hardware validation has been performed in this development environment. The public-dataset models elsewhere in this repo are not applied to this wearable stream.

## Check the software

```powershell
.venv\Scripts\python -m unittest -v test_collection.py
```

Tests exercise device clocks and rollover, malformed packets, dropped/duplicate indices, participant-wide numbering, raw values, common CSV timing, early stop, timed demo acquisition, session setup, and review edits while preserving markers. Demo runs through the same parser and CSV writer as USB acquisition. It cannot validate physical wiring, sensor noise or synchronization error.

Development validation: the automated desktop suite passes. The complete sketch and its core/library dependencies compiled and linked to a Teensy 4.1 ARM firmware image using PJRC Teensy 1.62.0 / GCC 15.2.1, Adafruit LSM6DS 4.7.4, Adafruit BusIO 1.17.4 and Adafruit Unified Sensor 1.1.15. No firmware was flashed and no physical sensors were tested here.

Files are deliberately separated by purpose: `app.py` is the GUI, plots and review; `device.py` handles USB acquisition; `session.py` validates packets and writes sessions; the sketch is the device-side acquisition code. No clinician authentication is implemented yet.

## V1 USB protocol

ASCII newline-terminated packets. Host commands are the single bytes `S` (start preview) and `X` (stop). Default audio blocks contain exactly 128 samples.

| Packet | Format |
| --- | --- |
| Start | `S,epoch_us,mic_rate_hz,imu_rate_hz,m/s2,V1` |
| Microphone | `A,first_sample_idx,first_sample_us,pcm0,...,pcm127` |
| Acceleration | `I,sample_idx,read_us,ax,ay,az` |
| Fault | `E,message` |

Timestamps are unsigned 32-bit microseconds. Audio and acceleration indices reset on `S`. The host handles microsecond rollover relative to the stream epoch and limits a preview stream to one hour; reconnect to reset the preview clock. Recording start/stop does not restart preview, so there is no second USB start handshake during session setup.
