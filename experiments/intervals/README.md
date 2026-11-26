# Event intervals (start, end, duration)

Live detections are no longer just a moment in time. Each alert now gets a **start, end and duration**, shown as a shaded band in Live Session and saved to `Intervals.csv`. Duration and timing are what a later normal-vs-abnormal screen will need.

## How intervals are built

The detectors score a 0.5 s window every 0.1 s; a window's score is assigned to its centre.
- **Start and end:** an interval starts at the first window whose score reaches **70% of the alert threshold**, and ends at the last such window before the score drops below it.
- **Cough, throat clear and swallow intervals** close after 2 windows below that level. **Talking** closes after a 3 s pause, the same pause that ends a talking bout.
- **Only alerted runs count:** an interval is reported only if an alert fired inside it, so every alert has exactly one interval. It appears about 0.2 s after the event ends.

Code: `IntervalTracker` in `data_collection/live_detection.py` and `EventIntervals` in `data_collection/events.py`.

## Accuracy against reviewed labels

Each participant is held out in turn, and every detection that matched a labelled event is compared with that label.

**Swallows** (recommended 35/65 tree model, 144 matched swallows; labelled swallows last a median 0.74 s):

| Continue-while score ≥ | Median overlap (IoU) | Overlap ≥ 0.5 | Start error (mean) | End error (mean) |
| --- | ---: | ---: | ---: | ---: |
| 50% of threshold | 0.62 | 69% | 0.18 s | 0.23 s |
| **70% (used)** | **0.65** | **72%** | **0.15 s** | **0.22 s** |
| 85% | 0.61 | 67% | 0.15 s | 0.24 s |
| 100% | 0.55 | 60% | 0.14 s | 0.28 s |

Starts and ends have no consistent bias (median errors of +0.005 s and −0.02 s). Shifting boundaries by a fixed amount learned from the other participants made results worse (median IoU 0.54). The best shift differs by person (P03's ends need +0.18 s, P01's −0.12 s), which points to labelling style. So no correction is applied.

**Coughs and talking** (event classifier): median overlap **0.84 for coughs** (3 matched) and **0.86 for talking bouts** (9 matched).

## What limits accuracy

- **The labels themselves.** Boundaries are drawn by hand in Review, and different reviewers mark swallows differently. Median labelled duration is 0.74 s for P01, 1.17 s for P02 and 0.53 s for P03.
- **A written labelling rule would help.** For example: "the swallow starts at the first visible acceleration deflection and ends when the signal returns to baseline". That would make both labels and measured durations more trustworthy.
- **The 0.5 s window** blurs boundaries by about ±0.1–0.2 s. A model trained to locate boundaries directly could do better once there is more data.

Reproduce: `python data_collection/evaluate_intervals.py` (swallows; writes `swallow_intervals.json`) and `python data_collection/train_events.py` (events).
