# Team setup and workflow

## Get the code

```bash
git clone https://github.com/AmaanK2/dysphagia-wearable-capstone.git
cd dysphagia-wearable-capstone
```

Follow the environment installation in the [main README](../README.md). Run all commands below from the repository root with that environment active. No dataset is needed for the unit tests.

## Run swallow timing

Download HRCA archives, or place your existing copies under `data/raw/hrca/`:

```bash
python scripts/download_hrca.py
python -m dysphagia_wearable.timing.baseline --signals data/raw/hrca/signals.zip --labels data/raw/hrca/labels.zip --output reports/swallow_timing/rf_reproduction
python -m dysphagia_wearable.timing.cnn --signals data/raw/hrca/signals.zip --baseline reports/swallow_timing/rf_reproduction --output reports/swallow_timing/cnn_reproduction
python -m dysphagia_wearable.timing.validate --baseline reports/swallow_timing/rf_reproduction --output reports/swallow_timing/cv_reproduction
python -m dysphagia_wearable.timing.plot_validation --baseline reports/swallow_timing/rf_reproduction --cv-dir reports/swallow_timing/cv_reproduction
```

CNN and validation share `data/processed/hrca_cnn` by default. Dataset size and CPU training can take time. For offline prediction on a compatible HRCA recording after training:

```bash
python -m dysphagia_wearable.timing.predict_cnn --model reports/swallow_timing/cnn_reproduction/best.pt --recording /path/to/recording.npy --output output/prediction
```

Check the checkpoint filename produced by your run. Only load trusted model files. These models use future signal context and are not a real-time streaming implementation.

## Run the separate activity benchmark

Download `Training dataset.zip` and `Test dataset.zip` from [STVS](https://zenodo.org/records/12630148), leaving the archives zipped under `data/raw/stvs/`:

```bash
python -m dysphagia_wearable.activity.train --training "data/raw/stvs/Training dataset.zip" --test "data/raw/stvs/Test dataset.zip" --output reports/activity/my_run
python -m dysphagia_wearable.activity.report --run reports/activity/my_run
```

Use a fresh output directory each time. Both random forest and CNN use the same split. The CNN checkpoint and model choice are selected on inner validation before the supplied test set is evaluated. The benchmark has no background class and cannot locate swallows in a continuous stream. Do not repeatedly tune against its test score.

## Work with teammates

```bash
git switch main
git pull
git switch -c feature/describe-your-change
# implement and run relevant tests
python -m unittest discover -s tests -v
git add .
git commit -m "Describe the change"
git push -u origin feature/describe-your-change
```

Open a pull request and ask a teammate to review it. Put reusable Python code under `src/dysphagia_wearable/`, tests under `tests/`, explanations under `docs/`, and experiment evidence under a new `reports/` subfolder. Keep raw recordings, caches, weights and environments out of commits. Keep historical metrics unchanged and preserve the test split when developing a model.
