# First swallow-detection experiment

Interpretation update: the linked study reports unlabelled events and imaging pauses. Unannotated time is not verified negative ground truth. All reported scores are relative to supplied annotations; see label_review/README.md.

**Status: research baseline, not a dysphagia or aspiration classifier.**

## Data actually used

- 1,276 recordings from 175 participant IDs.
- 1,862 annotated intervals; 190.8 recording minutes.
- 191 recordings retained as unannotated background (193 header-only files before exclusions).
- 9 exclusions: 2 Overlapping annotated intervals; requires review; 2 Recording shorter than feature window; 5 Label outside recording bounds.
- Counts differ from the Zenodo description; audit.json records the archive-derived counts.

## Participant-separated evaluation

- Train: 121 participants, 73,996 windows.
- Validation: 27 participants, 13,529 windows.
- Test: 27 participants, 21,334 windows.

All recordings from a person stay in one split. Thresholds and the selected model were determined on validation data before test evaluation.

## Held-out event results

| Method | Precision | Recall | F1 | False events / recording minute |
| --- | ---: | ---: | ---: | ---: |
| energy_baseline | 33.4% | 58.5% | 42.5% | 10.89 |
| rf_microphone | 33.6% | 62.8% | 43.8% | 11.59 |
| rf_acceleration | 33.3% | 74.8% | 46.1% | 14.03 |
| rf_combined | 37.6% | 75.4% | 50.2% | 11.70 |

**Selected on validation: rf_combined.**

Its held-out matches: 263 detected annotations, 86 missed annotations, 436 extra detections.
Participant-bootstrap F1 interval: 44.6%–56.7% (95%; does not capture retraining uncertainty).

With stricter event boundaries (IoU >= 0.50), selected-model F1 falls to 31.1%.

An event counts as matched only once and must overlap the annotation with IoU >= 0.30. A stricter IoU >= 0.50 analysis is in metrics.json. Window metrics are separate and must not be presented as event metrics.

## Interpretation and next steps

- This is an initial feasibility result on short clinical recordings, not validated home-monitoring performance.
- Empty label files and unlabelled time are treated as background; confirm annotation completeness with the authors.
- Review missed swallows and false detections, then use participant-grouped cross-validation for subsequent development.
- Test on your eventual sensors before claiming transfer to the wearable.
- Obtain suitable clinical labels before training normal/abnormal or aspiration-risk classification.

## Files

- README.md: methods, assumptions, commands and limitations.
- train.py / predict.py: training and offline inference.
- results/*.joblib: trained models; only load trusted model files.
- results/metrics.json, audit.json and participant_splits.json: evidence and reproducibility.
- results/model_comparison.png and held_out_example.png: design-review figures.

Dataset source: https://zenodo.org/records/4539695. Original raw recordings are not bundled.
