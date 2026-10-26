from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_REPORTS = PROJECT_ROOT / "reports/swallow_timing/baseline"
CNN_REPORTS = PROJECT_ROOT / "reports/swallow_timing/cnn_v1"
CV_REPORTS = PROJECT_ROOT / "reports/swallow_timing/grouped_cv_v1"
LABEL_REVIEW = PROJECT_ROOT / "reports/label_review"
HRCA_SIGNALS = PROJECT_ROOT / "data/raw/hrca/signals.zip"
