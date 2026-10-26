# Swallow detection capstone

Research code for detecting **when annotated swallows occur** in neck microphone and three-axis acceleration recordings. Includes a feature-based random forest and a small PyTorch temporal CNN.

This project does **not** diagnose dysphagia or predict aspiration. The second throat-activity dataset has been inspected but has not been used to train these models.

## Annotation interpretation update

The linked study reports unlabelled events during imaging pauses. Our scores compare predictions with supplied annotations; unannotated time is not verified negative ground truth. See [the annotation audit](label_review/README.md) before interpreting accuracy or extra detections.

## Current pilot results

Same 27 held-out participants; checkpoints and thresholds selected using validation data only. The existing test cohort had already been inspected for the baseline and is not a new blinded external validation.

| Event metric | Random forest | PyTorch CNN |
| --- | ---: | ---: |
| Precision, IoU >= 0.30 | 37.6% | 61.5% |
| Recall, IoU >= 0.30 | 75.4% | 74.2% |
| F1, IoU >= 0.30 | 50.2% | 67.3% |
| F1, stricter IoU >= 0.50 | 31.1% | 56.1% |
| Extra detections / recording minute | 11.70 | 4.35 |

The CNN also uses longer temporal context, so this experiment changes architecture and context together. These results do not establish superiority across new datasets or performance during daily meals.

See [the CNN summary](experiments/cnn_v1/SUMMARY.md), [training curves](experiments/cnn_v1/training_curves.png), and [comparison plot](experiments/cnn_v1/rf_vs_cnn.png).

## Grouped development validation

We also compare both methods in three outer participant folds across the 148 development participants, excluding all 27 original pilot test participants. Each fold uses separate inner validation for checkpoint/threshold selection. The original models and pilot results remain unchanged.

See [VALIDATION.md](VALIDATION.md) for the design and reproduction commands, and [the grouped-validation results](experiments/grouped_cv_v1/SUMMARY.md) for pooled metrics, fold results and error review. This is an internal robustness check on the same dataset, not external clinical validation.

## Setup

Use Python 3.12 in a virtual environment.

```bash
python -m venv .venv
# Activate the environment using the command appropriate for your OS.
python -m pip install -r requirements.txt
```

For CPU-only Linux/Windows training:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

For macOS:

```bash
python -m pip install torch==2.8.0
```

`requirements-cnn.txt` is an alternative combined dependency list. A default PyTorch installation on Linux can also install large GPU dependencies; use the CPU command above if you do not need them.

## Data

Source: https://zenodo.org/records/4539695

Place the original `signals.zip` and `labels.zip` inside `data/raw/`. Do not extract them. If you do not have them locally:

```bash
python download_data.py --output data/raw
```

The signals archive is about 1.4 GB. The script verifies both archive checksums. Raw data, caches, model binaries and NumPy arrays are excluded from Git by .gitignore.

Each signal is N x 4, sampled at 4 kHz: microphone, anterior-posterior acceleration, superior-inferior acceleration, medial-lateral acceleration. Annotation start/end values are 60 fps VFSS frames and are divided by 60 to get seconds.

The inspected archive has 1,285 signal/label pairs. After nine conservative exclusions, experiments use 1,276 recordings from 175 participant IDs, including 1,862 annotated events and 191 recordings with no annotations. Whether empty files indicate fully annotated background needs confirmation with the dataset authors. The archive counts differ from the dataset webpage. See results/audit.json.

## Reproduce the random forest

```bash
python train.py --signals data/raw/signals.zip --labels data/raw/labels.zip --output experiments/rf_reproduction
```

This rebuilds features by default. Reuse a feature cache only with `--reuse-cache` and unchanged data/feature code. Use a fresh output directory to preserve old runs.

The frozen `results/` folder contains the original participant split, dataset manifest and baseline evidence. Keep it unchanged for the CNN comparison.

## Train a new CNN run

```bash
python cnn.py --signals data/raw/signals.zip --baseline results --output experiments/cnn_reproduction --cache _cache/cnn_v1 --epochs 20 --patience 5 --threads 4
```

The CNN refuses to overwrite an existing checkpoint. The `experiments/cnn_v1` directory is the completed first run; choose another output directory for a reproduction.

The CNN has 36,681 parameters, with strided convolutions followed by dilated temporal convolutions and a binary output. Its receptive field is about 2.6 seconds of symmetric context. It produces scores at 50 Hz, interpolated onto the baseline's exact evaluation grid. Both methods use the same fixed filters and 2 kHz intermediate sampling rate.

Channel normalization uses training participants only. Loss uses training-derived positive class weight and excludes padding. AdamW, dropout, gradient clipping, deterministic CPU settings and validation-based early stopping are included. The first run completed 11 epochs and selected epoch 6. The threshold was selected on validation event F1. No new settings were selected using CNN test results.

## Predict on one recording

For a finite N x 4 NumPy array in the original 4 kHz/channel format:

```bash
python predict_cnn.py --model experiments/cnn_v1/best.pt --recording /path/to/recording.npy --output prediction_cnn
python predict.py --model results/rf_combined.joblib --recording /path/to/recording.npy --output prediction_rf
```

This GitHub source package excludes trained weights. Train the models or obtain the matching weights from the full project ZIP separately; keep those weights out of Git. Only load trusted model files. The CNN loader uses PyTorch's weights-only loading mode; the RF joblib file uses pickle internally.

## Checks

```bash
python -m unittest test_pipeline.py test_cnn.py test_validation.py
```

Eleven checks cover one-to-one event matching, zero-signal feature handling, output timing, padding invariance, loss masking and agreement between the two evaluation implementations, and exclusion of original pilot test participants from grouped validation. Reloaded-model inference is also compared with saved held-out predictions when producing the release.

## Files

- train.py / predict.py: random forest baseline and inference.
- cnn.py / predict_cnn.py: PyTorch training and inference.
- test_pipeline.py / test_cnn.py: evaluation and model checks.
- compare_models.py: regenerate first-run comparison figures and summary.
- validate.py / plot_validation.py: grouped development validation and error review.
- results/: frozen baseline audit, split, metrics, predictions and figures.
- experiments/cnn_v1/: CNN config, history, checkpoint, scores, metrics and figures.
- BASELINE_METHODS.md: detailed first-baseline methodology and assumptions.

## Interpretation limits and next steps

F1 is not accuracy. Window accuracy is also reported, but an always-background classifier scores highly because most windows are background. Event matching is one-to-one with IoU >= 0.30; a stricter >= 0.50 result is reported separately. Extra predictions can include duplicate fragments and boundary errors. False-event rates use total clinical recording minutes, not specifically non-swallow exposure.

All processing is offline/noncausal. No performance has been measured on the proposed wearable hardware. Participant prefixes are identifiers, not interpreted as clinical labels. There are no normal/abnormal or safe/unsafe labels in the inspected CSVs.

Next: confirm annotation meaning, review development-set errors, test grouped cross-validation and matched-context ablations, assess temporal postprocessing using development data only, and collect independently labelled recordings from the actual hardware. Avoid repeatedly tuning to this pilot test cohort. Use new external data for stronger final validation.

## Before putting this in GitHub

Copy the project contents into your repository, including .gitignore. Keep raw signals, caches and generated model binaries out of ordinary commits. Decide the repository visibility and software license with your team; no license or public release is assumed here.
