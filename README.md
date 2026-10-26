# Dysphagia wearable capstone

Software research for a neck-worn swallow screening prototype. Start with the reports to understand the results; you do not need to install anything to read them.

## What works so far?

| Pipeline | Question | Current evidence |
| --- | --- | --- |
| Swallow timing, HRCA dataset | When does a swallow happen in a recording? | CNN event F1 **68.0%**, compared with random forest **52.2%**, pooled across three participant-grouped development folds |
| Throat activity, STVS dataset | Is an already-cut clip a cough, speech, swallow or throat clearing? | CNN test accuracy **81.6%**, random forest **78.4%**; [benchmark details](reports/activity/v1/SUMMARY.md) |

These tasks have different metrics. Neither model diagnoses dysphagia or aspiration. Testing on our eventual wearable sensors is still needed. HRCA annotations are incomplete, so its event scores measure agreement with available annotations.

## Where to look

```text
src/dysphagia_wearable/
  timing/       Swallow timing models, prediction and evaluation
  activity/     Separate four-activity models and evaluation
scripts/        Dataset download utilities
tests/         Checks for splitting, signal handling and evaluation
docs/          Setup, dataset explanations and methods
reports/       Saved metrics, plots and experiment summaries
data/          Local data and caches; raw files stay out of Git
hardware/      Integration plans; no hardware selected in software
```

- **New teammate:** [setup and commands](docs/TEAM_SETUP.md).
- **Understand our progress:** [project status](docs/PROJECT_STATUS.md).
- **Understand the input data:** [datasets](docs/DATASETS.md).
- **See the timing evidence:** [grouped validation summary](reports/swallow_timing/grouped_cv_v1/SUMMARY.md) and [plot](reports/swallow_timing/grouped_cv_v1/grouped_validation.png).
- **Understand label limitations:** [annotation review](docs/LABEL_REVIEW.md).

## Quick setup

Use Python 3.12 or 3.13. In a terminal at the repository root:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-cnn.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
```

Raw data, caches and trained weights are excluded from Git. Training produces local weights; the checked-in reports let teammates inspect results without those files. See setup for training and inference commands.
