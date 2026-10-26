# Swallow detection: first capstone experiment

This is an offline research prototype for locating **annotated swallow intervals** in microphone and neck-acceleration recordings. It does not classify normal versus abnormal swallowing, diagnose dysphagia, or detect aspiration. Model scores are uncalibrated and are not clinical risk probabilities.

## Data

Source: https://zenodo.org/records/4539695 (Khalifa et al., 2021).

Use the original `signals.zip` and `labels.zip` without extracting them. Each recording is an N x 4 NumPy array, ordered microphone, anterior-posterior acceleration, superior-inferior acceleration, medial-lateral acceleration, sampled at 4,000 Hz. The labels use VFSS frame numbers at 60 fps: divide start/end by 60 for seconds, or multiply by 4000/60 for sample indices. Intervals are treated as half-open [start, end).

The webpage's participant and swallow counts differ from the downloaded archive. `reports/swallow_timing/baseline/audit.json` records the actual files, annotations, participants, exclusions and their reasons. Recordings with invalid/out-of-bounds labels or overlapping annotations are excluded conservatively rather than silently corrected. Exact duplicate arrays are excluded after their first occurrence.

Dataset 12630148 is a separate throat-event classification dataset. It is intentionally not mixed into this experiment because its sensor and labels differ.

## Run

Python 3.12 was used; exact package versions are in requirements.txt.

```bash
python -m pip install -r requirements.txt
python -m unittest test_pipeline.py
python -m dysphagia_wearable.timing.baseline --signals /path/to/signals.zip --labels /path/to/labels.zip --output results
```

Features are rebuilt by default. Add `--reuse-cache` only when the input files, preprocessing and features are unchanged. Use a new output directory for a new experiment to preserve previous results. Cached features are not automatically fingerprint-checked.

To predict on one extracted, compatible recording:

```bash
python -m dysphagia_wearable.timing.predict_baseline --model reports/swallow_timing/baseline/rf_combined.joblib --recording /path/to/recording.npy --output prediction
```

Choose the model named in reports/swallow_timing/baseline/metrics.json as `validation_selected_model`, rather than selecting whichever performs best on the test set. Only load trusted .joblib files: they use pickle internally.

## Experiment

- Fixed fourth-order Butterworth band-pass filters: 20–800 Hz for microphone and 5–500 Hz for acceleration. These are initial engineering choices, not optimized or clinically validated settings.
- Zero-phase offline filtering followed by anti-aliased downsampling to 2,000 Hz. This uses future samples and cannot be called a real-time implementation.
- 0.5-second windows with a 0.1-second hop. A window is positive if its center falls within an annotated interval.
- Fourteen features per channel: log RMS/std, crest factor, skewness, kurtosis, zero-crossing fraction, spectral centroid/bandwidth/entropy, peak frequency and power fractions in four fixed bands.
- Energy threshold baseline, plus random forests using microphone only, acceleration only and both. RF settings: 200 trees, maximum depth 14, minimum leaf size 5, balanced subsample class weights, seed 42.
- Approximately 70/15/15 participant split, with all recordings and overlapping windows for a participant confined to one split. Exact assignments are saved. Random forests use no population normalization or feature selection fitted on test data.
- Model threshold chosen solely by validation event F1. Final model selected by validation event F1 before test evaluation. No test-set tuning or hyperparameter search.

## Evaluation

Window precision/recall/F1, average precision and confusion matrices are reported separately from event performance. Positive grid points cover +/-0.05 seconds; gaps up to 0.2 seconds are joined, and events under 0.2 seconds are discarded. No predictions cover the extreme recording edges without a full analysis window.

Event matching is one-to-one and requires intersection-over-union (IoU) >= 0.30. A stricter IoU >= 0.50 check is also reported. Extra detections count as false positives; unmatched labels count as misses. False events per minute use total recording duration, not exclusively non-swallow time. Onset error is computed only for matched events and must be read alongside recall.

The selected model also has participant-bootstrap 95% intervals (1,000 resamples) at IoU 0.30. These reflect test-participant variability, not uncertainty from repeated training or dataset selection. The example plot is the alphabetically first test recording, not a handpicked success.

## What these results cannot establish

Unlabelled time is treated as background relative to the supplied annotations; annotation completeness must be confirmed with the dataset authors. Short clinical clips do not represent continuous home monitoring, and do not establish false alarm rates during everyday meals. Clinical collection context, sensor differences, placement, demographic coverage and recording protocols may limit transfer. There are no safe/unsafe or aspiration labels in these CSVs. Neither the `S` nor `NS` participant prefix is interpreted as a clinical class.

This first split is a pilot experiment. Once its test results are inspected, preserve them as reported. Later development should use participant-grouped cross-validation or a fresh external dataset, rather than repeatedly tuning to this same test set. Before hardware use, check sampling rate, channel order, bandwidth, units, timing, mounting and signal quality. No performance on your proposed hardware has been measured.

## Deliverables

`timing.baseline`, `timing.predict_baseline`, meaningful evaluation tests, exact dependency versions, trained model(s), data audit, participant split, metrics, test predictions, a model-comparison plot and a held-out recording plot. Original patient signals are not included in the deliverable ZIP.

The ZIP includes the validation-selected combined model. Metrics and predictions for all four methods are included; rerunning training regenerates the other trained random forests.
