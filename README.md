# Swallow detection capstone

Research code for detecting **when annotated swallows occur** in neck microphone and three-axis acceleration recordings. Includes a feature-based random forest, a small PyTorch temporal CNN, and a Teensy recording interface for collecting new research signals.

This project does **not** diagnose dysphagia or predict aspiration. The second throat-activity dataset has been inspected but has not been used to train these models.

## Find your way around

| Location | Purpose |
| --- | --- |
| [data_collection/](data_collection/README.md) | Swallow Studio acquisition app, live plots, event review and its firmware |
| [record_teensy.py](record_teensy.py) | Earlier standalone recorder with interval markers |
| Root Python scripts | Training, prediction and evaluation commands documented below |
| [experiments/](experiments/) | Versioned model comparisons, scores and experiment scripts |
| [results/](results/) | Original baseline results and participant split |
| [docs/](docs/README.md) | Setup, baseline methods, results and validation guide |
| [label_review/](label_review/README.md) | Annotation coverage and interpretation |
| [presentation/](presentation/) | Presentation figures and captions |

Swallow Studio uses the V1 USB protocol documented in its README. Check the firmware protocol before switching from the standalone recorder. Recorder values and public-model inputs require conversion and calibration before wearable inference.

## Annotation interpretation update

The linked study reports unlabelled events during imaging pauses. Our scores compare predictions with supplied annotations; unannotated time is not verified negative ground truth. See [the annotation audit](label_review/README.md) before interpreting accuracy or extra detections.

## Public-dataset model results

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

See [VALIDATION.md](docs/VALIDATION.md) for the design and reproduction commands, and [the grouped-validation results](experiments/grouped_cv_v1/SUMMARY.md) for pooled metrics, fold results and error review. This is an internal robustness check on the same dataset, not external clinical validation.

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

### Recorder-only setup

On a computer prepared for data collection, install Python with Tk support, then install the smaller recorder dependency set:

```bash
python -m pip install -r requirements-recorder.txt
python record_teensy.py --gui
```

The project `teensy` Conda environment can also launch it with `conda run -n teensy python record_teensy.py --gui`. A site coordinator should install and check the app and sensor before a clinician uses it.

### Clinician recording guide

1. Connect the Teensy sensor. Enter a **participant code**, never a patient name.
2. Choose one activity from the dropdown. Choose **Test** for practice or equipment checks; choose **Dataset** for a study recording.
3. Set the recording length and click **Start recording**. The device port is detected automatically; **Device settings** is for setup staff if detection fails.
4. Press **Space** once when the observed activity starts and again when it ends. The large on-screen interval button does the same thing. Repeat for each event.
5. Wait for the recording to finish. Review the numbered, activity-labeled shaded intervals on the graph. A red dashed line labeled "start only" means an interval was started but not ended; review its `end_s` later.

The initial activity choices are dry swallow, liquid swallow, food swallow, cough, throat clear, speech, and quiet rest. **Equipment check** appears only in Test mode. These task names are an initial collection menu; the study team should approve the final task list and instructions before participant collection. The interface currently uses English labels.

Each trial is saved below `data/test/<participant code>/` or `data/pilot/<participant code>/`. The folder contains `recording.txt` (raw microphone and X/Y/Z acceleration packets), `metadata.json`, `events.csv`, and `recording.png`. In `events.csv`, each row has the selected activity, `record_type=marker`, a `start_s` and `end_s` measured from recording start, and `timestamp_s` equal to `start_s`. An unfinished interval has a blank `end_s`. The `.gitignore` rule excludes both recording folders from Git.

These keypresses are observer marks, not instrument-verified physiological boundaries. The plotted time alignment also depends on the host-to-device start delay, so review event boundaries against the signals before using them as training labels.

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

### Recheck the trained model on the public dataset before new collection

The frozen random forest can be re-evaluated on the original **27 held-out public-dataset participants**, using the original validation-selected threshold. This does not retrain or tune the model. Obtain the checksum-verified `signals.zip` and `labels.zip` described above and a trusted `SwallowDetection_v1.zip` containing `results/rf_combined.joblib`:

```bash
python evaluate_public_rf.py \
  --signals data/raw/signals.zip \
  --labels data/raw/labels.zip \
  --model-archive /path/to/SwallowDetection_v1.zip
```

The script checks archive hashes, signal hashes, label intervals, the participant split, and agreement with the frozen event counts. It writes its report under ignored `data/public_eval/`. The published source dataset has 4 kHz microphone/AP/SI/ML channels and VFSS-derived swallow boundaries; [see its Zenodo record](https://zenodo.org/records/4539695). The Teensy records its own microphone and IMU streams with different sampling rates and sensor placement. Public-dataset scores do not establish performance on this wearable, and its raw recordings cannot be fed directly to `predict.py` or `predict_cnn.py` without a validated conversion and channel mapping.

Recheck on November 18, 2026: the frozen combined random forest reproduced the saved counts on 232 recordings from the 27 held-out participants (263 matched swallows, 436 extra detections, 86 missed annotations; event F1 50.2% at IoU 0.30). The original CNN report above remains available, but its checkpoint is not in this source repository or the local model archive used for this recheck, so this run does not independently reproduce the CNN result.

## Checks

```bash
python -m unittest test_pipeline.py test_cnn.py test_validation.py test_recorder.py
```

These checks cover one-to-one event matching, zero-signal feature handling, output timing, padding invariance, loss masking, agreement between the two evaluation implementations, exclusion of original pilot test participants from grouped validation, and recorder interval saving. Reloaded-model inference is also compared with saved held-out predictions when producing the release.

## Files

- train.py / predict.py: random forest baseline and inference.
- cnn.py / predict_cnn.py: PyTorch training and inference.
- record_teensy.py: clinician-facing research recorder with observer interval marks.
- evaluate_public_rf.py: recheck the frozen random forest on held-out public recordings.
- requirements-recorder.txt / test_recorder.py: recorder dependencies and interval checks.
- test_pipeline.py / test_cnn.py: evaluation and model checks.
- compare_models.py: regenerate first-run comparison figures and summary.
- validate.py / plot_validation.py: grouped development validation and error review.
- results/: frozen baseline audit, split, metrics, predictions and figures.
- experiments/cnn_v1/: CNN config, history, checkpoint, scores, metrics and figures.
- docs/BASELINE_METHODS.md: detailed first-baseline methodology and assumptions.

## Interpretation limits and next steps

F1 is not accuracy. Window accuracy is also reported, but an always-background classifier scores highly because most windows are background. Event matching is one-to-one with IoU >= 0.30; a stricter >= 0.50 result is reported separately. Extra predictions can include duplicate fragments and boundary errors. False-event rates use total clinical recording minutes, not specifically non-swallow exposure.

All processing is offline/noncausal. No performance has been measured on the proposed wearable hardware. Participant prefixes are identifiers, not interpreted as clinical labels. There are no normal/abnormal or safe/unsafe labels in the inspected CSVs.

Next: confirm annotation meaning, review development-set errors, perform matched-context ablations, and validate sensor placement, timing, and labels on the actual hardware before evaluating on independently labelled wearable recordings. Avoid repeatedly tuning to this pilot test cohort. Use new external data for stronger final validation.
