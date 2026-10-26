"""Render an existing activity run; this does not retrain or select a model."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import CLASSES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.run / "metrics.json").read_text())
    audit = json.loads((args.run / "audit.json").read_text())
    history = json.loads((args.run / "history.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for ax, name in zip(axes[:2], ["random_forest", "cnn"]):
        cm = np.array(
            result["models"][name]["test"][
                "confusion_matrix_rows_true_columns_predicted"
            ]
        )
        ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] / cm[i].sum() > 0.6 else "black",
                )
        ax.set(
            xticks=range(4),
            yticks=range(4),
            xticklabels=CLASSES,
            yticklabels=CLASSES,
            xlabel="Predicted activity",
            ylabel="Actual activity",
            title=name.replace("_", " ").title(),
        )
        ax.tick_params(axis="x", labelrotation=35)
    axes[2].plot(
        [row["epoch"] for row in history],
        [row["validation_macro_f1"] for row in history],
        marker="o",
        label="CNN inner validation",
    )
    axes[2].axvline(
        result["best_cnn_epoch"], color="gray", linestyle=":", label="Selected epoch"
    )
    axes[2].axhline(
        result["models"]["random_forest"]["validation"]["macro_f1"],
        color="orange",
        linestyle="--",
        label="Random forest validation",
    )
    axes[2].set(
        xlabel="Epoch",
        ylabel="Macro F1",
        title="Clip-based inner validation",
        ylim=(0, 1),
    )
    axes[2].legend(fontsize=8)
    fig.suptitle(
        "STVS activity benchmark: 792 presegmented test clips; no background class"
    )
    fig.savefig(args.run / "evaluation.png", dpi=150)
    plt.close(fig)
    lines = [
        "# Four-activity benchmark, v1",
        "",
        "This asks whether an already-segmented vibration clip is cough, speech, swallow or throat clearing. "
        "It does not detect swallow timing in a continuous recording or diagnose dysphagia.",
        "",
        "| Model | Inner validation macro F1 | Supplied test accuracy | Supplied test macro F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, scores in result["models"].items():
        lines.append(
            f"| {name} | {scores['validation']['macro_f1']:.1%} | {scores['test']['accuracy']:.1%} | {scores['test']['macro_f1']:.1%} |"
        )
    lines += [
        "",
        f"Selected on inner validation: **{result['selected_on_inner_validation']}**, CNN epoch {result['best_cnn_epoch']}.",
        "",
        "## Audit and protocol",
        "",
        f"- Original release: {audit['training_clips']:,} training and {audit['test_clips']} test clips; 6,400 Hz, 4,000 samples per clip (0.625 s).",
        f"- Excluded {len(audit['excluded_ambiguous_training_ids'])} ambiguous training rows across {audit['conflicting_training_hashes']} waveform hashes with conflicting activity labels. No replacement labels were invented.",
        f"- Found {audit['duplicate_training_clips']} duplicate training rows; eligible duplicates stay in the same inner partition. No exact train/test overlap was found.",
        "- Inner validation is class-stratified and grouped by waveform hash, not by participant. Arrays have no per-row participant IDs.",
        "- The authors describe 12 training and 20 separate test participants. We preserve their supplied test partition; that separation cannot be reconstructed independently from these arrays.",
        "- Both models are trained only on the fitting partition. The CNN checkpoint and model choice use inner validation macro F1. Test evaluation follows freezing both models.",
        "- Random forest uses 22 time/frequency features and 300 trees. The small CNN uses a normalized waveform and temporal convolutions. Neither reproduces the paper ensemble.",
        "",
        "## Interpretation",
        "",
        "The gap between inner validation and supplied-test performance cautions against relying on clip-based validation alone. "
        "Do not describe the inner-validation result as unseen-person performance. The test set contains multiple clips per participant, "
        "so clips are not independent patient-level samples.",
        "",
        "These results are separate from HRCA event F1. There is no quiet/background class, continuous onset/offset reference, "
        "or clinical risk label here. Transfer to our wearable has not been measured. Freeze this test result and use new "
        "development data for further choices.",
        "",
        "![Evaluation plots](evaluation.png)",
        "",
        "See `audit.json`, `config.json`, `history.json`, `split.json`, `provenance.json` and `metrics.json` for reproducibility. "
        "Weights and prediction arrays are generated locally and excluded from Git.",
        "",
    ]
    (args.run / "SUMMARY.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
