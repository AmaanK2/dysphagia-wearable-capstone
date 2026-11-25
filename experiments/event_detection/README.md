# Cough, talking and throat-clear detection

Swallow Studio's Live Session runs a second detector next to the swallow detector. Every 0.1 s, an IMU-only classifier labels the last 0.5 s as **swallow, cough, talking, throat clear or nothing**. Swallow alerts still come from the swallow detector; this classifier adds the other three event types. It uses only the IMU, so it keeps working through microphone dropouts.

Code: `data_collection/events.py` (features, model, alert logic) and `data_collection/train_events.py` (training and evaluation). Model: `data_collection/models/events.json`.

## Data

| Source | Swallow | Cough | Talking | Throat clear |
| --- | ---: | ---: | ---: | ---: |
| Our recordings (reviewed labels, 14 sessions) | 154 | 5 | 14 intervals | 0 |
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

Updated 25 Nov 2026 with 5 new sessions, and with the alert logic changed so cough and throat-clear alerts are held back while someone is talking.

**Throat-dataset test speakers** (never trained on), one window per clip: **81.3% accuracy** (78.8% balanced).

| True ↓ / predicted → | Swallow | Cough | Talking | Throat clear | Nothing |
| --- | ---: | ---: | ---: | ---: | ---: |
| Swallow | **93%** | 6% | 0% | 0% | 2% |
| Cough | 13% | **75%** | 6% | 3% | 4% |
| Talking | 4% | 3% | **89%** | 4% | 1% |
| Throat clear | 1% | 11% | 26% | **59%** | 3% |

The dataset's authors report about 96% with their full 6.4 kHz signal. Our IMU only captures vibration below ~450 Hz, so throat clear is the hardest class.

**Our recordings, leave one participant out** (28 minutes), replayed through the live alert logic:

| Training data | Coughs caught | Talking bouts caught | False alerts/min (cough / talking / throat clear) |
| --- | ---: | ---: | --- |
| Throat dataset only | 4 / 5 | 3 / 14 | 0.32 / 0.18 / 0.32 |
| **Throat dataset + our recordings (shipped)** | **3 / 5** | **9 / 14** | **0.46 / 0.54 / 0.14** |

- **Our recordings matter for talking:** 9/14 bouts with them vs 3/14 without, because the throat dataset's speech doesn't look like talking measured at our sensor.
- **Coughs are too few to judge:** all 5 come from P01 and P02, so when P01 is held out only one of our coughs is left to train on. 3/5 versus 4/5 is a one-event difference.
- **False alerts total 1.1/min,** down from 1.8 before the retrain. Only 2 land on a labelled swallow.
- **Many "false" alerts may be real events.** In P03's dry-swallow session S05, most alerts fall where the microphone is 5–40× louder than the session's typical level, which suggests unlabelled talking or noise.
- **The first model was optimistic.** Before retraining, it caught 6 of the 9 new talking bouts but raised 2.2 false alerts/min on the new sessions. My earlier "~0.2/min in quiet sessions" claim was based on too little data.

Raising the alert threshold to 0.8 lowers false alerts (to ~0.7/min) but drops talking to 4/14, so the shipped model keeps 0.5.

## Limits and next steps

- **Few labels of our own:** 5 coughs from 2 people, 14 talking intervals, and no throat clears. Treat the wearable numbers as a smoke test.
- **Label everything in mixed sessions,** so false alerts can be told apart from real unlabelled events. Then record dedicated cough, talking and throat-clear sessions on several more people.
- **The mic could help:** adding mic features would likely separate cough from throat clear better, but the throat dataset has no microphone channel to train them from.

## Reproduce

```sh
python download_data.py --throat
python data_collection/train_events.py     # ~10 min; writes models/events.json and evaluation.json
```
