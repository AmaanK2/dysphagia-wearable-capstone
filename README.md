# Dysphagia wearable capstone

This is a neck-worn wearable (contact microphone + IMU on a Teensy 4.1) that detects **swallows** in real time, plus **coughs, talking and throat clears**. Detecting abnormal swallowing (dysphagia) is the long-term goal. The current models detect events only and make no clinical judgement.

## Repository layout

| Folder | What's inside |
| --- | --- |
| [`data_collection/`](data_collection/README.md) | **Swallow Studio**: the recording app, live detection, labelling tool, firmware, model training and our recordings |
| [`experiments/`](experiments/) | Results for the wearable models: [mixed public + wearable swallow detector](experiments/mixed_public_wearable/SUMMARY.md), [event detection](experiments/event_detection/README.md), [wearable-only pilot](experiments/wearable_pilot/RESULTS.md) |
| [`public_baseline/`](public_baseline/README.md) | Earlier RF and CNN swallow segmentation on the public Pittsburgh dataset |
| [`docs/FIGURES.md`](docs/FIGURES.md) | Presentation figures with captions |
| `download_data.py` | Fetches the public datasets into the git-ignored `data/` folder |

## Quick start

```sh
cd data_collection
python -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py --demo            # without hardware; drop --demo when the Teensy is connected
```

## Current results

Every score below is measured on a wearable participant the model never trained on (3 participants, 14 sessions: 154 reviewed swallows, 5 coughs, 14 talking intervals).

| Detector | Result |
| --- | --- |
| Swallows, wearable data only | 76% event F1, ~1.6 false alerts/min |
| **Swallows, wearable + adapted public data (35/65, recommended)** | **86% event F1 (90% with the exported model's tuned threshold), ~0.8 false alerts/min** |
| Swallows, live in the app on new sessions | 37 of 38 swallows caught, 5 false alerts in 4 minutes |
| Cough / talking / throat clear | 9/14 talking bouts and 3/5 coughs caught, ~1.1 false alerts/min; 81% accuracy on public test speakers. [Details](experiments/event_detection/README.md) |

With 3 participants, differences of a few points are within noise. More participants, more varied swallows (water, food) and more labelled coughs and talking are the biggest next steps.

## Datasets

- **Our recordings:** `data_collection/recordings/participants/` (P01–P03).
- **Pittsburgh swallow dataset:** [Zenodo 4539695](https://zenodo.org/records/4539695), 4 kHz mic + 3-axis accelerometer with VFSS swallow labels. Fetch with `python download_data.py`.
- **Throat-event dataset:** [Zenodo 12630148](https://zenodo.org/records/12630148) (Song et al., *npj Digital Medicine* 2025), skin vibration clips of cough, speech, swallow and throat clear. Fetch with `python download_data.py --throat`.
