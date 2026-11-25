# Cough, talking and throat-clear detection

Swallow Studio's Live Session runs a second detector next to the swallow detector. Every 0.1 s, an IMU-only classifier labels the last 0.5 s as **swallow, cough, talking, throat clear or nothing**. Swallow alerts still come from the swallow detector; this classifier adds the other three event types. It uses only the IMU, so it keeps working through microphone dropouts.

Code: `data_collection/events.py` (features, model, alert logic) and `data_collection/train_events.py` (training and evaluation). Model: `data_collection/models/events.json`.

## Data

| Source | Swallow | Cough | Talking | Throat clear |
| --- | ---: | ---: | ---: | ---: |
| Our recordings (reviewed labels) | 85 | 5 | 5 intervals | 0 |
| Throat dataset ([Zenodo 12630148](https://zenodo.org/records/12630148)), training speakers | 2,028 | 1,984 | 2,107 | 2,089 |
| Throat dataset, test speakers (5 languages) | 161 | 150 | 333 | 148 |

Our unlabelled windows, away from any label or marker, serve as the "nothing" class.

**Adapting the throat clips** (0.625 s, single-axis, 6.4 kHz):
- **Resampled** to our IMU's measured 904.5 Hz.
- **Rescaled and equalised:** a gain and an 8-band equaliser are fitted so that the average swallow and talking windows match ours.
- **Matched to our 3-axis IMU:** each of our windows is reduced to its main vibration direction, so a single-axis clip and our 3-axis IMU produce the same kind of signal.
- **Weighting:** each class carries equal weight in training, split evenly between the two datasets where both have it.

**Features:** loudness, 8 spectral bands (5–450 Hz), spectral shape, a 10-step energy envelope, crest factor, kurtosis, zero-crossing rate and voicing (periodicity at 80–300 Hz).

## Results

**Throat-dataset test speakers** (never trained on), one window per clip: **80.9% accuracy** (79.1% balanced).

| True ↓ / predicted → | Swallow | Cough | Talking | Throat clear | Nothing |
| --- | ---: | ---: | ---: | ---: | ---: |
| Swallow | **93%** | 4% | 0% | 1% | 2% |
| Cough | 13% | **75%** | 6% | 3% | 3% |
| Talking | 3% | 4% | **86%** | 6% | 1% |
| Throat clear | 1% | 12% | 22% | **63%** | 3% |

The dataset's authors report about 96% with their full 6.4 kHz signal. Our IMU only captures vibration below ~450 Hz, so cough versus throat clear is the hardest pair.

**Our recordings, leave one participant out**, replayed through the live alert logic (two consecutive windows for cough/throat clear, one alert per talking bout):

| Training data | Coughs caught | Talking bouts caught | False alerts/min (cough / talking / throat clear) |
| --- | ---: | ---: | --- |
| Throat dataset only | 4 / 5 | 2 / 5 | 0.67 / 0.33 / 0.33 |
| **Throat dataset + our recordings (shipped)** | **5 / 5** | **4 / 5** | **0.83 / 0.67 / 0.28** |

**Where the false alerts happen:**
- **Quiet use:** in P03's dry-swallow sessions there was 1 false alert in 6 minutes (~0.2/min).
- **Mixed sessions:** nearly all the rest fall in unlabelled stretches of P01's and P02's "mixed" sessions, which may contain talking or coughing nobody labelled.
- **Swallows:** none of the false alerts land on a labelled swallow.

Raising the alert threshold from 0.5 to 0.8 only cuts false alerts to ~1.2/min and loses detections, so the shipped model keeps 0.5.

## Limits and next steps

- **Few labels of our own:** 5 coughs and 5 talking intervals from 2 people, and no throat clears. Treat the wearable numbers as a smoke test.
- **Label everything in mixed sessions,** so false alerts can be told apart from real unlabelled events. Then record dedicated cough, talking and throat-clear sessions on several more people.
- **The mic could help:** adding mic features would likely separate cough from throat clear better, but the throat dataset has no microphone channel to train them from.

## Reproduce

```sh
python download_data.py --throat
python data_collection/train_events.py     # ~10 min; writes models/events.json and evaluation.json
```
