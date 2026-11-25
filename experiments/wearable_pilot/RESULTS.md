# Wearable pilot: wearable-only swallow detector

These are the first wearable results, before public data was added (that's in [`../mixed_public_wearable`](../mixed_public_wearable/SUMMARY.md)). The model is a class-balanced logistic model on causal mic + IMU features (`data_collection/train_live.py`). It scores a trailing 0.5 s window every 0.1 s and alerts after two positive windows.

**Data:** 3 participants (P01–P03), 116 reviewed dry swallows, 14 talking and 5 cough intervals. Only 9 of the 12 sessions are in Git, so the in-repo data has 85 swallows. An alert counts as a match if it falls between the labelled onset and 0.25 s after the labelled end.

## Leave-one-participant-out results

| Configuration | Matched | Extra alerts | Precision | Recall | Event F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original detector (85 swallows) | 73/85 | 125 | 36.9% | 85.9% | 51.6% |
| Background-focused retraining | 70/85 | 68 | 50.7% | 82.4% | 62.8% |
| Expanded data, selected settings (116 swallows) | 102/116 | 94 | 52.0% | 87.9% | 65.4% |
| + mic/IMU agreement check (both must reach 0.5) | 89/116 | 44 | 66.9% | 76.7% | 71.5% |
| Nested tuning (settings chosen without the test person) | 98/116 | 129 | 43.2% | 84.5% | 57.1% |

**Per person:** recall in the original detector ranged from 21% (P01) to 100% (P02, P03). P01's sessions have long microphone dropouts.

**Honest estimate:** most settings were chosen on these same three people. The nested row (57%) is the fairer guide for a new wearer.

## Raw public data didn't transfer

| Training data | Test | Precision | Recall | Event F1 | Extra alerts/min |
| --- | --- | ---: | ---: | ---: | ---: |
| Wearable, full features | Wearable | 36.9% | 85.9% | 51.6% | 6.94 |
| Public (unadapted) | Wearable | 9.2% | 57.6% | 15.9% | 26.78 |
| Wearable + public (unadapted) | Wearable | 25.4% | 37.6% | 30.3% | 5.22 |

Using the public recordings as-is failed: units, sample rates and sensors all differ. Adapting them first fixed that; see [`../mixed_public_wearable`](../mixed_public_wearable/SUMMARY.md).

## Runtime

Feature extraction, scoring and alert logic take about 0.2–0.5 ms per 100 ms tick on a laptop. Rolling (live) and batch replay produce identical alerts.
