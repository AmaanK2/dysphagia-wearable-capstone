# Live swallow detection pilot

Eight usable wearable sessions from three participants contained 85 reviewed dry swallows, five cough intervals, and five talking intervals. The ninth session contained a marker with no reviewed interval and was excluded from scoring. The original pilot fit a class-balanced, L2-regularized logistic model to microphone and three-axis acceleration features. The cough and talking labels serve as examples of background for the binary detector. Each class appeared in recordings from two participants, which limits evaluation of dedicated cough or speech outputs.

## Microphone and IMU agreement

The 12-session training set now fits three scores: acoustic evidence using microphone features alone, motion evidence using IMU features alone, and the existing combined score. To issue an alert, the microphone and IMU scores must each reach 0.50 and the combined score must reach 0.60 on each of two consecutive 0.5-second windows. The windows are evaluated every 0.1 seconds. Missing or unusable sensor data blocks a detection.

| Expanded wearable replay | Matched | Extra alerts | Precision | Recall | Event F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Combined score alone | 102 of 116 | 94 | 52.0% | 87.9% | 65.4% |
| Combined score with both sensor checks | 89 of 116 | 44 | 66.9% | 76.7% | 71.5% |
| Both sensor thresholds tuned inside each outer training fold | 90 of 116 | 54 | 62.5% | 77.6% | 69.2% |

The selected thresholds reduced extra alerts from 3.92 to 1.83 per minute and increased median matched-alert delay from 0.42 s to 0.46 s. The third row holds the evaluated participant outside sensor-threshold selection, with the same shared classifier settings as the first two rows. There are three participants and no specifically reviewed slight Adam’s apple movements, so the analysis measures overall alert behavior without estimating a motion-specific error rate.

## Retraining with the three new recordings

P03 added six minutes across two dry-swallow sessions and one mixed session. Together they add 31 reviewed swallows and nine talking intervals. The final fit uses all 12 wearable sessions, including one marker-only session that supplies usable background windows. Eleven reviewed sessions supply the replay results below. Across those reviewed sessions there are 116 swallows, 14 talking intervals, and five cough intervals.

The search compared eight settings for the classifier, its L2 penalty, and background weighting, with six score thresholds and three cooldowns for each. One weighting candidate emphasized high-acceleration, low-relative-audio background windows as a proxy for small motion. It was tested because there are no reviewed Adam’s apple movement intervals. It was not selected for deployment. The final fit uses quadratic logistic regression, L2 coefficient 0.015, a 0.60 score threshold, and a 1.6-second cooldown. The cooldown can suppress closely spaced alerts.

| Expanded wearable replay | Swallows matched | Extra alerts | Precision | Recall | Event F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous settings refit on each training fold | 99 of 116 | 109 | 47.6% | 85.3% | 61.1% |
| Selected settings refit on each training fold | 102 of 116 | 94 | 52.0% | 87.9% | 65.4% |
| Nested tuning with the test participant held outside model selection | 98 of 116 | 129 | 43.2% | 84.5% | 57.1% |

The selected settings scored 3.92 extra alerts per minute with a median matched delay of 0.42 s. The 65.4% row uses the same participant folds that guided selection. Nested tuning selects settings using the other two participants inside each outer fold and scored lower than the previous fixed settings. The pilot does not establish that the tuning gains carry over to new wearers.

### Keep or exclude P01

The matched comparison used the same P02 and P03 test recordings, keeping the previous model settings fixed. Retaining P01 in the training folds matched **94 of 102** swallows with **88** extra alerts (66.2% F1). Dropping P01 matched **80 of 102** with **106** extra alerts (55.6% F1). The chronological comparison withheld all three new P03 sessions during training and tuning. Keeping P01 gave 22 matches and 12 extra alerts (67.7% F1). Excluding P01 gave 23 matches and 16 extra alerts (65.7% F1). The previously deployed checkpoint, scored on those same three new sessions, matched 25 of 31 swallows with 17 extra alerts (68.5% F1). P01 was retained for the final refit.

Reproduce the tuning and comparisons from the repository root:

```sh
data_collection/.venv/bin/python data_collection/train_live.py \
  --compare-exclusion P01 \
  --holdout-session P03/S03_dry_swallow \
  --holdout-session P03/S04_mixed \
  --holdout-session P03/S05_dry_swallow
```

## Background-focused detector

The previous detector already trained on full-recording windows. The new fit keeps eligible unmarked windows and emphasizes two groups within the background class: the highest-scoring 20% of background windows from an initial fit and windows fully inside reviewed non-swallow intervals such as cough or talking. A second fit uses these weights while keeping total class weights balanced. Windows that straddle a swallow boundary, overlap ambiguous annotations, or fall near an unmatched observer marker are excluded from background training. This includes usable unmarked windows in the marker-only ninth recording, which remains outside the scored eight sessions.

The model also uses spectral concentration, changes in energy relative to the preceding half-second, and how closely microphone and IMU energy rise and fall together. A regularized quadratic logistic model can combine these features without adding a runtime dependency. Training used 717 swallow windows and 8,371 background windows, including 294 fully inside reviewed non-swallow intervals. The windows overlap, so those counts represent training examples rather than independent events.

| Held-out replay metric | Previous model | Background-focused model |
| --- | ---: | ---: |
| Reviewed swallows matched | 73 of 85 | 70 of 85 |
| Extra alerts | 111 | 68 |
| Extra alerts during reviewed talking | 3 | 2 |
| Extra alerts during reviewed coughing | 6 | 6 |
| Precision | 39.7% | 50.7% |
| Recall | 85.9% | 82.4% |
| Event F1 | 54.3% | 62.8% |
| Extra alerts per minute | 6.17 | 3.78 |
| Median matched-alert delay | 0.45 s | 0.47 s |

Each fold kept the test participant's signals out of training and background weighting. Model settings were chosen after comparing candidates on this same three-person pilot. P01 still had only 3 of 14 swallows detected, and the recordings contain just five reviewed talking and five reviewed cough intervals. Most unmarked windows are weak background labels and can contain unreviewed events. All 21 tests passed. Rolling and batch replay agreed on the updated alerts. The updated detector used 0.32 ms median CPU time per 100 ms replay tick on the same Mac.

## Previous detector optimization

The running model still scores a trailing half-second of both sensors every 100 ms. Microphone quality and amplitude use the full-rate PCM. For the spectrum, the extractor averages samples in groups to reduce the rate from 44.1 kHz to about 5.5 kHz before measuring bands below 800 Hz. Hann windows are reused across calls. Both versions use a linear model and a fixed 0.65 score cutoff.

| Held-out replay configuration | Matched | Extra alerts | Missed | Precision | Recall | Event F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original detector | 73 | 125 | 12 | 36.9% | 85.9% | 51.6% |
| Faster features with original 0.8 s cooldown | 73 | 124 | 12 | 37.1% | 85.9% | 51.8% |
| Faster features with deployed 1.2 s cooldown | 73 | 111 | 12 | 39.7% | 85.9% | 54.3% |

The previous optimized detector produced **6.17 extra alerts per minute** and matched alerts came a median **0.45 seconds** after reviewed onset. The 1.2-second cooldown was selected after inspecting this pilot cohort and can suppress another nearby swallow. The gains therefore need confirmation on new participants. A nested comparison that selected a linear or quadratic model and score cutoff using only the outer-fold training participants yielded 44.2% F1 and 49.4% recall. The deployed fit kept the simpler linear model and original cutoff.

## Original wearable benchmark

Each fold trained on two participants and replayed the third participant's recordings in timestamp order. The model consumed a trailing 0.5-second window every 0.1 seconds. The alert used a fixed 0.65 score cutoff, two consecutive positive windows, and a 0.8-second refractory period. Each alert could match one reviewed swallow from its annotated onset through 0.25 seconds after its annotated end.

| Wearable fold | Matched | Extra alerts | Missed | Precision | Recall | Event F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P01 | 3 | 17 | 11 | 15.0% | 21.4% | 17.6% |
| P02 | 32 | 46 | 0 | 41.0% | 100.0% | 58.2% |
| P03 | 38 | 62 | 1 | 38.0% | 97.4% | 54.7% |
| Pooled | 73 | 125 | 12 | 36.9% | 85.9% | 51.6% |

- The model produced **6.94 extra alerts per recording minute** across 18 minutes. Review coverage was incomplete, so some extra alerts could correspond to unlabelled activity. Unmatched observer-marker regions were excluded from background training but remained in replay evaluation.
- **Window accuracy was 88.0%**, compared with **93.3%** for always predicting background. Balanced window accuracy, which gives swallow and background windows equal weight, was **76.6%**.
- The detector alerted during **71 of 85** reviewed swallows. Two more matched alerts fell within the 0.25-second grace period. Median matched-alert delay was **0.45 seconds** from reviewed onset on the recording clock. This is an offline replay measurement against observer-reviewed boundaries. USB transport and GUI rendering can add latency.
- Signal gates suppressed scoring for approximately 97.6 seconds of window hops, nearly all in P01. The pilot's recall varied from 21.4% to 100% across the three held-out people. The sensor quality notes document intermittent microphone dropout in two P01 sessions.

The GUI uses coefficients refit on all eight wearable sessions. The numbers above come from the held-out folds. They describe dry-swallow performance on this pilot corpus and carry substantial uncertainty for a new wearer.

## Software verification

All 19 recorder and GUI tests passed. The GUI test used simulated V1 serial input and verified that the tab drew an alert before the recording ended and saved its timestamp. On the same 60-second hardware recording, the original and optimized detectors each produced identical alerts in batch and rolling replay. The original model file also reproduced its original alerts when loaded by the updated code.

With a 10-second rolling buffer on this Mac, median detector runtime fell from **0.456 ms to 0.214 ms** per 100 ms tick. The 95th percentile fell from **0.501 ms to 0.233 ms**. The comparison measures feature extraction, scoring, and alert logic. Acquisition, GUI rendering, and disk writes were outside the timed operation.

## Original public-data transfer benchmark

The University of Pittsburgh public dataset provides a 4 kHz neck microphone channel, three anatomically aligned acceleration channels, and videofluoroscopy labels. The wearable records microphone PCM at 44.1 kHz and timestamps its IMU separately on the same device clock. Sensor gain, mounting, and the activities in the recordings can all differ. [Public dataset and channel layout](https://zenodo.org/records/4539695)

The repository's existing random forest and CNN achieved 50.2% and 67.3% event F1, respectively, on 27 held-out public participants using a 0.30 interval-overlap criterion. Their checkpoints require calibrated public-format signals, and neither checkpoint was available in this checkout. Those scores describe the public recordings and use a different matching rule from this live experiment. [Original public benchmark](../README.md#public-dataset-model-results)

The transfer comparison uses the causal pilot features, with amplitude terms removed to reduce dependence on gain and units. It pools the acceleration axes for relative band energy and temporal shape. Public training uses the original public training split. The combined fit gives wearable and public data equal weight, uses scaling fitted on the wearable training participants, and holds out the same wearable participant in each fold. All rows use the same alert and matching rule described above.

The public model trained on **849 recordings from 121 participants**. The public test used **232 recordings from 27 different participants**.

| Training data and input | Test recordings | Precision | Recall | Event F1 | Extra alerts/min |
| --- | --- | ---: | ---: | ---: | ---: |
| Wearable, full features (GUI model) | Wearable | 36.9% | 85.9% | 51.6% | 6.94 |
| Wearable, shape features | Wearable | 24.1% | 65.9% | 35.3% | 9.78 |
| Public, shape features | Public | 40.2% | 58.5% | 47.7% | 8.13 |
| Public, shape features | Wearable | 9.2% | 57.6% | 15.9% | 26.78 |
| Wearable + public, shape features | Wearable | 25.4% | 37.6% | 30.3% | 5.22 |

Direct transfer produced 482 extra alerts on the wearable recordings. Adding public data reduced alert frequency relative to wearable-only shape features, while detecting 32 of 85 labelled swallows instead of 56. The full-feature wearable model had the highest event F1 in these pilot comparisons and was the starting point for the optimized GUI detector.

To compare public and wearable data with the current feature extractor, run from the repository root:

```sh
data_collection/.venv/bin/python download_data.py
data_collection/.venv/bin/python data_collection/train_live.py --public-data data/raw
```
