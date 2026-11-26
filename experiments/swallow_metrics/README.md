# Swallow metrics and the water swallow test

These are the first steps toward normal-vs-abnormal screening. They measure **how** a person swallows, not only **whether** they swallowed. Code: `data_collection/swallow_metrics.py`.

## Per-session metrics

The metrics are computed from reviewed labels (`Events.csv`) or live detections (`Intervals.csv`):

- **Swallow count and rate** (per minute).
- **Swallow duration.**
- **Time between swallows** (onset to onset).
- **Swallow clusters:** swallows starting within 1.5 s of the previous one ending. Several swallows for one mouthful (piecemeal swallowing) is a dysphagia sign.
- **Post-swallow coughs:** coughs starting within 3 s after a swallow ends.

## Normal ranges so far (`normal_ranges.json`)

These come from 154 reviewed dry swallows by 3 healthy volunteers:

| Metric | Median | Middle 50% | 5th–95th percentile |
| --- | ---: | ---: | ---: |
| Swallow duration | 0.74 s | 0.57–0.99 s | 0.35–1.20 s |
| Time between swallows | 9.1 s | 6.3–10.9 s | 4.0–16.6 s |
| Swallows per cluster | 1 | 1 | 1 |

**These are not clinical norms yet.**
- **Cued swallows:** these are dry swallows done on request, so the time between swallows reflects the protocol, not natural swallowing.
- **Labelling style:** median durations differ by person (P01 0.74 s, P02 1.17 s, P03 0.53 s), more than physiology would explain. Labelling boundaries consistently matters before using duration (see [`../intervals`](../intervals/README.md)).
- **No water or food swallows yet**, and no post-swallow coughs or clusters were observed.

## Timed water swallow test (in Swallow Studio)

Choose **Water swallow test** as the activity and enter the volume (100 ml by default). Then:
1. **Start recording** (90 s by default).
2. Ask the person to **drink the water as quickly as is comfortable**.
3. **Press Space when the cup touches the lips**, and optionally again when they finish.
4. **Keep recording ~60 s afterwards** to catch late coughs.

When the recording is saved, the app scores the test from the live detections and saves `WaterTest.json`:

| Output | Meaning |
| --- | --- |
| Swallows, seconds | Swallows from the first marker until drinking ends (the second marker or the last swallow) |
| ml per swallow, s per swallow | Volume and time per swallow |
| Capacity (ml/s) | Volume ÷ time. Below **10 ml/s** is the usual abnormal cut-off (Nathadwarawala et al. 1992; Hughes & Wiles 1996) |
| Coughs during / after | Coughs while drinking or within 60 s after. Coughing fails the 3-oz water swallow test (Suiter & Leder 2008) |
| Screening cues | Which of the above was flagged; any cue sets "review recommended" |

After correcting labels in Review, rescore from them with:

```sh
python data_collection/swallow_metrics.py --water-test data_collection/recordings/participants/P01/S07_water_swallow_test --volume 100 --labels
```

**This is a research screening aid, not a diagnosis.**
- **The cut-offs come from clinician-run tests.** They haven't been validated with this device.
- **Missed swallows inflate the result.** The device counts swallows with ~86% F1, so a missed swallow raises ml/swallow; check the detected swallows in Review.
- **Next:** run the test on many healthy volunteers to get device-specific normal values. Then, with a clinical partner, run it on people with suspected dysphagia alongside the clinician's own test.

## Reproduce

```sh
python data_collection/swallow_metrics.py      # writes experiments/swallow_metrics/normal_ranges.json
```
