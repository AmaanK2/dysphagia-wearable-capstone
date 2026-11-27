# Public dataset vs our recordings: gap analysis and adaptation study

**Question:** the public Pittsburgh dataset has ~10x more swallows than we can collect, so why does giving it more weight not help, and can a better adaptation fix that?

**Short answer:**
- The public data is already used about as much as it can be. A model trained only on adapted public data separates our swallow and non-swallow windows almost as well as one trained on our own recordings (window AUC 0.935 vs 0.94).
- Above ~35–50% public loss weight, false alerts rise at every setting tried, because public swallows are weaker relative to their background than ours.
- None of the 10 adaptation and weighting methods tried here removed that limit.
- What did help was giving the model the two earlier windows as context (0.2 and 0.4 s back). The 1,500 public swallows make that larger model learnable.
- The remaining errors are mostly alerts in unlabelled stretches of our own recordings. The public data can't fix those.

All numbers are nested leave-one-participant-out event F1 (alert within the labelled swallow, or up to 0.25 s after it). The held-out participant never influences adaptation, weighting or the alert threshold.

## 1. How the datasets differ

| | Our recordings (frozen study set) | Public (adapted, training participants) |
| --- | --- | --- |
| People | 3 | 148 |
| Recording | 17 sessions, 34.5 min | 1,044 clips, 153.5 min (clips are 3–20 s around swallows) |
| Swallows | 159 dry swallows, observer-labelled | 1,513 food/liquid swallows from X-ray swallow studies (videofluoroscopy), labelled from the X-ray video |
| Windows labelled swallow | 6.5% | 15.5% |
| Background | quiet, 1,355 talking and 158 cough windows | almost no talking or coughing |

Source: `analysis.json`.

**Features.** Adaptation already removes most of the sensor difference. The mean absolute standardised gap over the 29 features drops from 0.63 to 0.12 for swallow windows and from 0.41 to 0.11 for background windows. A classifier can still tell the datasets apart (AUC 0.72 on swallow windows, 0.85 on background windows). The features that remain different describe context, not the sensor:

- **Onset sharpness:** the rise in IMU and mic energy from the previous half-second (`imu_rise`, `mic_rise`), plus the late-window envelope (`mic_env_3`, `mic_env_4`) and `mic_crest`.
- **Contrast:** in log RMS, a public swallow stands out from its background by 0.52 (mic) and 0.10 (IMU), against 0.95 and 0.13 in ours. Public swallows sit in noisier surroundings (patients drinking during an X-ray study, swallows close together).
- **Label timing:** relative to the labelled onset, energy peaks 0.3–0.4 s after onset in public clips and 0.5 s in ours. So public labels start about 0.1–0.2 s later in the swallow.
- **Duration:** about the same (median 0.83 s public vs 0.74 s ours). Ours has more short swallows (5th percentile 0.36 vs 0.53 s).

**Transfer.**

| Train on | Test on | Window AUC |
| --- | --- | --- |
| Adapted public only | Our recordings | 0.935 |
| Our recordings, leave one person out | Our recordings | 0.94 |
| Our recordings | Public | 0.79 |

The public-only model rarely mistakes talking (0.4% of talking windows) or coughs (0%) for swallows. It misses swallows that our model catches and fires on small events in our background.

**Within-class direction.** Every feature but one points the same way, swallow vs background, in both datasets (correlation 0.81). The exception is the IMU 15–40 Hz band share.

## 2. What was tried

All runs use the 17 sessions present when the study began (frozen, because recording continued in parallel) and boosted trees. "Original 14" excludes P03's three new talking/mixed sessions, whose gaps between planned events were never labelled. All runs are logged in `experiments.json`.

| Method (35% public loss weight unless noted) | All 17 | Original 14 | Verdict |
| --- | --- | --- | --- |
| No public data | 75.7 | 79.3 | public data is worth ~6–8 points |
| **Current method** (gain + EQ calibration, 35% public) | 81.7 | 87.6 | baseline |
| Same, 20 / 50 / 65 / 80% public | 81.2 / 80.8 / 79.9 / 76.6 | 86.5 / 85.7 / 87.3 / 84.7 | above ~50% precision falls; recall flat |
| Class-specific shares (e.g. 50–70% of the swallow class public) | 78.0–81.4 | 81.7–86.4 | more public swallows → more false alerts |
| Per-feature quantile mapping public → ours | 78.8–79.2 | 85.3–87.2 | worse; erases useful variety |
| Importance weighting by a domain classifier | 79.9–82.1 | 82.8–87.2 | no gain |
| Shift public labels 0.1–0.3 s earlier | 81.0–83.4 | 85.0–85.9 | small, unstable |
| Pre-train on public, then fine-tune on ours | 75.8–79.8 | 82.5–83.1 | worse |
| Up-weight our reviewed talking/cough windows | 81.1–82.1 | 84.8–87.7 | no gain |
| Choose 1- or 2-window alert rule in the inner loop | 82.5 | 88.5 | ~+1 |
| Finer threshold grid (scores bunch up near 1) | 83.0 | 87.6 | +1 |
| Average of 5 models (seeds or shares) | 78.5–82.5 | 85.4–87.2 | no gain |
| **Context: also the windows 0.2 and 0.4 s earlier** | 85.4 | 90.0 | best |
| **Context + finer thresholds** (shipped) | **85.9** | **89.7** | best |

**How to read these numbers.** The test has only 159 swallows, and each fold's threshold is chosen from two people. Neighbouring settings often differ by 3–5 points, and with this many configurations the top score is somewhat optimistic. Context is the only change that beat the current method at every public weight:

| Public weight | 20% | 35% | 50% | 65% |
| --- | --- | --- | --- | --- |
| Current method | 81.2 | 81.7 | 80.8 | 79.9 |
| With context | 82.2 | 85.4 | 84.4 | 81.3 |

More context (3 or 4 earlier windows) or bigger trees was worse or unstable.

## 3. Check on recordings made during the study

The final comparison was re-run with `train_mixed.py` on all 24 sessions recorded by the end of the study (201 swallows). The 7 newer sessions add 55 labelled coughs and were never used to choose settings.

| 35% public | Nested F1 | Precision | Recall | False alerts/min | Best fixed threshold (hindsight) |
| --- | --- | --- | --- | --- | --- |
| Current method | 75.4 | 69.0 | 83.1 | 1.59 | 76.9 |
| **Context** | **77.1** | 73.4 | 81.1 | 1.25 | 81.5 |

20% and 50% public weight were lower for both models, so **35% public / 65% ours is still the best ratio**. Raw output: `data_collection/models/mixed/evaluation_trees*.json`.

**Label-aware scoring.** Sessions were labelled one event type at a time. In P02/S05 and P03/S06, S07 and S09 only coughs or talking were labelled, so an alert there may be a real swallow. Re-scored so those sessions only count alerts inside a labelled cough or talking bout (the threshold is also chosen this way), with the finer threshold grid:

| Public weight | 20% | 35% | 50% |
| --- | --- | --- | --- |
| Current method | 76.5 | 76.8 | 76.0 |
| Context | 76.4 | 74.4 | 74.1 |

On all 24 sessions the two models are effectively tied: the gap is within the 3–5 point run-to-run noise. The context model's advantage shows clearly only on the original sessions. Under this scoring, 1 false alert remains in the talking/cough-only sessions. The other 69 are in sessions that also have swallow labels, mostly P03/S08 (14), P02/S07 (14), P02/S02 (14) and P02/S06 (6). 20 of the 70 fall within 3 s of a labelled cough or talking bout, 11 of them in P02/S07. `alerts_to_review.csv` lists all 70 with their times.

**Where the context model still goes wrong** (all 24 sessions):
- **59 false alerts:** 48 in stretches with no labelled event, 8 within 3 s after a cough, 3 next to talking.
- **38 missed swallows:** 29 are P01's, 14 of them in P01's newest mixed sessions (S08 and S09).

In the frozen set, the alerts in unlabelled stretches look like weak swallows. Both independent models score 97% of them at 0.5 or above, against 14% of random background windows. But they are much quieter: their mic energy rises 0.56 log units above the session median, against 1.42 for labelled swallows. They could be spontaneous swallows that were never labelled (people swallow between planned events and after coughing) or small throat movements.

## 4. Why 90%+ isn't reachable yet, and what would get there

- **The public data isn't the bottleneck any more.** On its own it gets close to our-data performance, and none of these methods made it add more above ~35–50% weight.
- **Label completeness is the bottleneck.** If even half of the unlabelled-gap alerts are real swallows, F1 on the original sessions is already above 90%. Check those times in the Review tab, and in new sessions label every swallow, not just the planned ones.
- **More people.** Each person is a third of the test, and P01's newest sessions show how much one person's day-to-day variation moves the score.

## Reproduce

```sh
# Nested evaluation (writes data_collection/models/mixed/evaluation_trees_context2_4.json)
.venv/bin/python data_collection/train_mixed.py --model trees --lags 2 4 --mode sweep --shares 0.2 0.35 0.5
# App model
.venv/bin/python data_collection/train_mixed.py --model trees --lags 2 4 --mode export --export mixed_context_35_65=0.35
```
