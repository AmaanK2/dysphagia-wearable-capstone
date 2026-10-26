# Why we added grouped development validation

The first experiment compared a random forest and CNN on one 121/27/27 participant split. The CNN improved event F1, but a single split cannot establish that it works consistently across participants.

This experiment uses the original training and validation participants together as a development pool of 148 people. The original 27 pilot test participants are excluded from every fold and from pooled evaluation. The first experiment and weights remain unchanged.

## Design

Three outer folds divide development participants into assessment groups of 50, 49 and 49. Every development participant is assessed exactly once. Within the other roughly 99 participants, a separate 20% participant split is used for inner validation. The remainder is used for training.

Both models are trained from scratch for each fold. RF architecture and CNN architecture match the earlier experiment. Each model's threshold is selected using inner-validation event F1. The CNN also uses inner validation for early stopping and checkpoint selection, with at most 20 epochs and patience 5. Input normalization and positive loss weighting use only inner-training data. No original pilot test results or outer-assessment results select these settings.

CNN and RF see the same participant roles and the same event evaluation grid and matching rule. CNN has approximately 2.6 seconds of context; RF uses 0.5-second feature windows, so this does not isolate architecture from temporal context.

## Reading the output

- `experiments/grouped_cv_v1/SUMMARY.md`: readable results and limitations.
- `summary.json`: pooled counts, fold-level results and participant-bootstrap intervals.
- `fold_plan.json`: exactly who is used in each role, plus the excluded pilot test list.
- `grouped_validation.png`: fold F1 and pooled precision/recall/F1.
- `cnn_error_review.csv`: extra detections and missed annotations to review against raw signals.
- Each fold folder: training history, scores and model checkpoints (RF fold weights omitted from the downloadable release to keep it small; train again to regenerate them).

Pooled F1 is calculated from summed event counts, not the simple average of the three fold F1 values. Each fold retains its own validation-selected threshold. IoU >= 0.30 and stricter IoU >= 0.50 are reported separately. There is no event-level true-negative count because continuous background is not a discrete set of events.

## Reproduction

Install dependencies as described in README.md. Raw data and derived caches are not distributed in the project ZIP or committed to Git.

```bash
# Regenerate feature arrays and deterministic baseline manifests.
python train.py --signals data/raw/signals.zip --labels data/raw/labels.zip --output experiments/rf_reproduction

# Generates the waveform cache too; use a fresh output directory.
python cnn.py --signals data/raw/signals.zip --baseline experiments/rf_reproduction --output experiments/cnn_reproduction --cache _cache/cnn_v1

python validate.py --baseline experiments/rf_reproduction --cache _cache/cnn_v1 --output experiments/cv_reproduction
python plot_validation.py --baseline experiments/rf_reproduction --cv-dir experiments/cv_reproduction

python -m unittest test_pipeline.py test_cnn.py test_validation.py
```

## What this still cannot establish

These are exploratory internal results on one source dataset, not an external clinical evaluation. The dataset-label completeness and sensor-transfer uncertainties remain. Correlations between fitted CV folds and variation from repeating training are not fully represented by participant-bootstrap intervals. Do not choose the best-performing outer-fold model as the deployed model or tune further using these assessment results.

The next controlled experiments should separate context from architecture, investigate annotation-relative errors using inner development data, and test recordings from the actual hardware. Normal/abnormal or aspiration prediction still requires clinical outcome labels that are absent from the inspected CSVs.
