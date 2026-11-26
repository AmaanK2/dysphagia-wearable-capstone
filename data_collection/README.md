# Swallow Studio

The desktop app for the wearable: it records the Teensy's contact microphone and IMU, shows live signals, detects swallows (plus coughs, talking and throat clears) in real time, and lets you review and label sessions.

## Run it

Python 3.12+. From this folder:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt     # Windows: .venv\Scripts\python
.venv/bin/python app.py                                 # add --demo to try it without hardware
```

On Windows, after setup, you can double-click `start.bat`.

To see live detection without the device, replay a saved session in real time. It loops, and anything you record during a replay is saved as a simulated session under `recordings/demo/`:

```sh
.venv/bin/python app.py --replay recordings/participants/P01/S03_mixed
```

## Using the app

1. **Record tab:** pick the USB port and click **Connect device**. **Ready** means both sensor streams are arriving. It doesn't check sensor placement or contact.
2. **Start recording:** enter a participant code (never a name), choose an activity (Dry swallow, Talking, Mixed, Water swallow test or Custom) and a duration of up to 30 minutes.
3. **Mark events:** press **Space** for each swallow you see, or at the start and end of longer activities. Presses are approximate observer marks.
4. **Live Session tab:** shows the rolling signals and detections.
   - **Detector** chooses the swallow model: your current `models/wearable.json` appears first, then the mixed public+wearable models (35/65 trees is recommended).
   - **Also detect cough, talking, throat clear** turns the IMU event classifier (`models/events.json`) on or off.
   - Each event type has its own colour. An alert first appears as a line; once the event ends, its full interval is shaded and its duration is added to the log. Both settings lock during a recording.
5. **Water swallow test:** enter the water volume, press **Space** when the cup touches the lips (and optionally when drinking ends), and keep recording ~60 s afterwards. When the recording is saved, the app reports swallow count, ml/s, ml and seconds per swallow, and coughs, and saves `WaterTest.json`. [Details](../experiments/swallow_metrics/README.md)
6. **Review tab:** open a session, drag the selection over each event, choose its activity and click **Save event**. Mark **Uncertain** when unsure. Reviewed labels are what training uses.

Sessions are saved to `recordings/participants/<code>/Sxx_<activity>/`. Numbering continues across a participant's visits, and existing folders are never overwritten.

| File | Contents |
| --- | --- |
| `Mic.csv` | `sample_idx,time_s,mic_raw`: raw 16-bit PCM at 44.1 kHz |
| `IMU.csv` | `sample_idx,time_s,ax,ay,az`: raw acceleration in m/s² (gravity included), ~904 Hz at the 833 Hz setting or ~1.8 kHz at 1666 Hz |
| `Events.csv` | Observer markers (`record_type=marker`) and reviewed labels (`record_type=label`, `start_s`/`end_s`) |
| `Detections.csv` | Live alerts: `time_s,event,model_score` (`event` is swallow, cough, talking or throat_clear) |
| `Intervals.csv` | Live event intervals: `event,start_s,end_s,duration_s,peak_score` |
| `WaterTest.json` | Water swallow test result (water swallow test sessions only) |
| `Session.json` | Rates, timing origin, status, and which detector files were used |

Every time is in seconds from one shared session start. CSVs store raw values; the filtering you see on screen is for display only. Large `Mic.csv` files can be stored as `Mic.csv.gz`.

## Models

| Model | Train with | Notes |
| --- | --- | --- |
| `models/wearable.json` | `python train_live.py` | Wearable-only logistic detector (git-ignored, machine-specific) |
| `models/mixed/mixed_*.json` | `python train_mixed.py --model trees --mode export ...` | Wearable + adapted public swallow data. [Results](../experiments/mixed_public_wearable/SUMMARY.md) |
| `models/mixed/mixed_forest_wearable_only.json` | `python train_mixed.py --mode export --model forest --no-gate --export mixed_forest_wearable_only=0` | Random forest on our recordings only; ties the recommended model. [Comparison](../experiments/model_comparison/README.md) |
| `models/events.json` | `python train_events.py` | Cough / talking / throat-clear classifier. [Results](../experiments/event_detection/README.md) |

Training needs the public datasets (`python ../download_data.py` and `python ../download_data.py --throat`, both run from the repo root) and `requirements-training.txt`. The app itself only needs `requirements.txt`.

| Code | Purpose |
| --- | --- |
| `app.py` | GUI: record, live session, review |
| `device.py`, `session.py` | USB acquisition and session files |
| `live_detection.py` | Swallow features, models (logistic and trees) and alert logic |
| `events.py` | Cough / talking / throat-clear features, model and alert logic |
| `adapt_public.py`, `check_adaptation.py` | Convert public swallow recordings to the wearable's format and check the match |
| `swallow_metrics.py` | Per-swallow metrics, normal ranges and water swallow test scoring |
| `train_live.py`, `train_mixed.py`, `train_events.py` | Model training and evaluation |
| `evaluate_intervals.py` | How well live intervals match labelled swallow boundaries |
| `test_*.py` | Tests: `python -m unittest test_collection test_mixed test_events test_metrics` |

## Hardware (one-time setup)

| Part | Connection |
| --- | --- |
| Teensy 4.1 | USB **data** cable |
| Teensy Audio Shield Rev D | Stacked; SGTL5000 line input |
| TE CM-01B contact microphone | Amplified output to **left LINE IN**, common ground, power per its datasheet (5 V VIN) |
| Adafruit ISM330DHCX | 3.3 V, GND, SDA pin 18, SCL pin 19, I²C address 0x6A |

1. Install the Arduino IDE with [Teensy support](https://www.pjrc.com/teensy/td_download.html), then the **Adafruit LSM6DS** library (with its dependencies).
2. Open `firmware/SwallowRecorder/SwallowRecorder.ino`, select **Teensy 4.1**, **USB Type: Serial**, **600 MHz**, and upload it.
3. Close the Serial Monitor before connecting from the app.

Firmware defaults: mic at 44.1 kHz with fixed gain `lineInLevel(5)`; accelerometer at ±2 g.

**IMU rate:** `IMU_ODR_HZ` at the top of the sketch selects 833 (default), 1666 or 3332 Hz nominal; the sensor's clock runs ~8% fast, so 833 gives ~904 Hz.
- **833 (default):** runs exactly the hardware-tested read path that every current model was trained on.
- **1666:** doubles the vibration bandwidth, which should help separate cough from throat clear. The app averages faster data back to ~900 Hz for the current models and saves the full-rate data for training future ones. Above 833 Hz the firmware runs I²C at 1 MHz and reads only the accelerometer registers.
- **Faster rates are untested on hardware.** Before collecting data at 1666 Hz, record a minute and check that the measured rate in `Session.json` is ~1.8 kHz, that no "IMU service delay" error appears, and that detections look normal.

Mount the sensor in the same place and orientation every session. Check for clipping before collecting participant data.

## USB protocol (V1)

Newline-terminated ASCII. The host sends `S` to start streaming and `X` to stop.

| Packet | Format |
| --- | --- |
| Start | `S,epoch_us,mic_rate_hz,imu_rate_hz,m/s2,V1` |
| Microphone | `A,first_sample_idx,first_sample_us,pcm0,...,pcm127` |
| Acceleration | `I,sample_idx,read_us,ax,ay,az` |
| Fault | `E,message` |

Timestamps come from the Teensy's microsecond clock. The host rejects malformed packets, clock resets and sequence gaps, and keeps whatever was recorded if a session is interrupted.
