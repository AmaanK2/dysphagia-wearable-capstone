# Public-dataset baseline

These are the first experiments: swallow segmentation on the public University of Pittsburgh dataset ([Zenodo 4539695](https://zenodo.org/records/4539695), Khalifa et al.). This folder is self-contained. It doesn't feed the app, except that `results/records.json` and `results/participant_splits.json` are reused when training the wearable models.

## Data

- **Signals:** 1,276 recordings (after 9 exclusions) from 175 participant IDs, with 1,862 VFSS-annotated swallows.
- **Channels:** each recording is an N × 4 array at 4 kHz: microphone, then acceleration in three directions (front-to-back, up-down, side-to-side).
- **Labels:** VFSS frame numbers at 60 fps.
- **Audit:** `results/audit.json` records the counts and why recordings were excluded.
- **Unlabelled time:** treat it with caution. The source study reports swallows missing from labels during imaging pauses (see `label_review/`).

## Methods

- **Random forest:** 0.5 s windows every 0.1 s, 14 spectral and statistical features per channel. Variants: mic-only, accelerometer-only, and combined.
- **CNN:** a 36k-parameter temporal model with ~2.6 s of context.
- **Shared settings:** both use the same filters, 2 kHz resampling and event matching (one-to-one, IoU ≥ 0.30; IoU ≥ 0.50 also reported).
- **Single split:** 121/27/27 participants for train/validation/test. Thresholds were chosen on validation only.
- **Grouped validation:** 3 outer participant folds over the 148 non-test participants, each with its own inner validation set.

## Results

| Held-out test (27 participants) | Precision | Recall | Event F1 | Extra detections/min |
| --- | ---: | ---: | ---: | ---: |
| Random forest (combined) | 37.6% | 75.4% | 50.2% | 11.70 |
| CNN | 61.5% | 74.2% | 67.3% | 4.35 |

Grouped validation (148 participants) gives a pooled F1 of 52.2% for the RF and 68.0% for the CNN. An **accelerometer-only CNN scores the same as mic + accelerometer** (68–70% per fold, `experiments/grouped_cv_imu_only`). That matches the literature, where the accelerometer carries most of the swallow information. All of these measure swallow *timing*, not dysphagia or aspiration.

## Files

| Path | What it is |
| --- | --- |
| `train.py`, `predict.py` | Random forest training and offline prediction |
| `cnn.py`, `predict_cnn.py` | CNN training and prediction |
| `validate.py`, `plot_validation.py` | Grouped cross-validation and plots |
| `compare_models.py`, `evaluate_public_rf.py` | RF vs CNN summary; re-check the frozen RF |
| `audit_annotations.py`, `label_review/` | Annotation audit and review queue |
| `results/` | Frozen RF results, data audit, participant split, recording manifest |
| `experiments/` | CNN run (`cnn_v1`) and grouped-validation runs |
| `test_*.py` | Evaluation, CNN and fold-plan checks |

## Reproduce

Run from this folder after `python ../download_data.py` (run from the repo root, it puts data in `../data/raw`):

```sh
python -m pip install -r requirements.txt
python -m pip install torch==2.8.0     # CPU wheel: add --index-url https://download.pytorch.org/whl/cpu on Linux/Windows
python train.py --signals ../data/raw/signals.zip --labels ../data/raw/labels.zip --output experiments/rf_reproduction
python cnn.py --signals ../data/raw/signals.zip --baseline experiments/rf_reproduction --output experiments/cnn_reproduction
python validate.py --baseline experiments/rf_reproduction --output experiments/cv_reproduction
python -m unittest test_pipeline test_cnn test_validation
```

Trained weights (`*.joblib`, `*.pt`) are git-ignored. Retrain them, or get them from the team's project archive.
