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

## Using the app

1. **Record tab:** pick the USB port and click **Connect device**. **Ready** means both sensor streams are arriving. It doesn't check sensor placement or contact.
2. **Start recording:** enter a participant code (never a name), choose an activity (Dry swallow, Talking, Mixed or Custom) and a duration of up to 30 minutes.
3. **Mark events:** press **Space** for each swallow you see, or at the start and end of longer activities. Presses are approximate observer marks.
4. **Live Session tab:** shows the rolling signals and detections.
   - **Detector** chooses the swallow model: your current `models/wearable.json` appears first, then the mixed public+wearable models (35/65 trees is recommended).
   - **Also detect cough, talking, throat clear** turns the IMU event classifier (`models/events.json`) on or off.
   - Each event type has its own colour on the plots and in the log. Both settings lock during a recording.
5. **Review tab:** open a session, drag the selection over each event, choose its activity and click **Save event**. Mark **Uncertain** when unsure. Reviewed labels are what training uses.

Sessions are saved to `recordings/participants/<code>/Sxx_<activity>/`. Numbering continues across a participant's visits, and existing folders are never overwritten.

| File | Contents |
| --- | --- |
| `Mic.csv` | `sample_idx,time_s,mic_raw`: raw 16-bit PCM at 44.1 kHz |
| `IMU.csv` | `sample_idx,time_s,ax,ay,az`: raw acceleration in m/s² (gravity included), ~904 Hz |
| `Events.csv` | Observer markers (`record_type=marker`) and reviewed labels (`record_type=label`, `start_s`/`end_s`) |
| `Detections.csv` | Live alerts: `time_s,event,model_score` (`event` is swallow, cough, talking or throat_clear) |
| `Session.json` | Rates, timing origin, status, and which detector files were used |

Every time is in seconds from one shared session start. CSVs store raw values; the filtering you see on screen is for display only. Large `Mic.csv` files can be stored as `Mic.csv.gz`.

## Models

| Model | Train with | Notes |
| --- | --- | --- |
| `models/wearable.json` | `python train_live.py` | Wearable-only logistic detector (git-ignored, machine-specific) |
| `models/mixed/mixed_*.json` | `python train_mixed.py --model trees --mode export ...` | Wearable + adapted public swallow data. [Results](../experiments/mixed_public_wearable/SUMMARY.md) |
| `models/events.json` | `python train_events.py` | Cough / talking / throat-clear classifier. [Results](../experiments/event_detection/README.md) |

Training needs the public datasets (`python ../download_data.py` and `python ../download_data.py --throat`, both run from the repo root) and `requirements-training.txt`. The app itself only needs `requirements.txt`.

| Code | Purpose |
| --- | --- |
| `app.py` | GUI: record, live session, review |
| `device.py`, `session.py` | USB acquisition and session files |
| `live_detection.py` | Swallow features, models (logistic and trees) and alert logic |
| `events.py` | Cough / talking / throat-clear features, model and alert logic |
| `adapt_public.py`, `check_adaptation.py` | Convert public swallow recordings to the wearable's format and check the match |
| `train_live.py`, `train_mixed.py`, `train_events.py` | Model training and evaluation |
| `test_*.py` | Tests: `python -m unittest test_collection test_mixed test_events` |

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

Firmware defaults: mic at 44.1 kHz with fixed gain `lineInLevel(5)`; accelerometer at ±2 g (nominal 833 Hz; ~904 Hz measured). Mount the sensor in the same place and orientation every session. Check for clipping before collecting participant data.

## USB protocol (V1)

Newline-terminated ASCII. The host sends `S` to start streaming and `X` to stop.

| Packet | Format |
| --- | --- |
| Start | `S,epoch_us,mic_rate_hz,imu_rate_hz,m/s2,V1` |
| Microphone | `A,first_sample_idx,first_sample_us,pcm0,...,pcm127` |
| Acceleration | `I,sample_idx,read_us,ax,ay,az` |
| Fault | `E,message` |

Timestamps come from the Teensy's microsecond clock. The host rejects malformed packets, clock resets and sequence gaps, and keeps whatever was recorded if a session is interrupted.
