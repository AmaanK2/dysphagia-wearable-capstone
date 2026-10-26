"""Separate STVS four-activity benchmark. Do not compare its accuracy to event F1."""

import argparse, copy, json, random, time, hashlib, sys
from importlib.metadata import version
from pathlib import Path
import joblib, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from .data import CLASSES, FS, SAMPLES, read_archive, audit_and_split, normalize_clips
from .features import extract_features
from .models import ActivityCNN
from ..paths import PROJECT_ROOT


def metrics(y, p):
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro")),
        "classification_report": classification_report(
            y,
            p,
            labels=range(4),
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix_rows_true_columns_predicted": confusion_matrix(
            y, p, labels=range(4)
        ).tolist(),
    }


@torch.no_grad()
def predict(model, x, batch_size=128):
    model.eval()
    parts = []
    for start in range(0, len(x), batch_size):
        parts.append(
            model(torch.from_numpy(x[start : start + batch_size, None]))
            .softmax(dim=1)
            .numpy()
        )
    return np.vstack(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--training", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/activity/v1")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=5)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError(
            "Use a fresh output directory; completed reports are preserved"
        )
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    training = read_archive(args.training, "training")
    test = read_archive(args.test, "test")
    tr, va, audit = audit_and_split(training, test)
    args.output.mkdir(parents=True)
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2))
    provenance = {
        "python": sys.version,
        "packages": {
            name: version(name)
            for name in ["numpy", "scipy", "scikit-learn", "torch", "joblib"]
        },
        "dataset_source": "https://zenodo.org/records/12630148",
        "archives_md5": {"training": training["md5"], "test": test["md5"]},
        "training_code_sha256": {
            str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in Path(__file__).parent.glob("*.py")
        },
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (args.output / "split.json").write_text(
        json.dumps(
            {
                "training_ids": training["ids"][tr].tolist(),
                "validation_ids": training["ids"][va].tolist(),
                "test_ids": test["ids"].tolist(),
            },
            indent=2,
        )
    )
    print(
        "Data audit:",
        audit["training_clips"],
        "training clips;",
        audit["test_clips"],
        "test clips;",
        len(tr),
        "fit;",
        len(va),
        "inner validation",
        flush=True,
    )
    config = {
        "seed": 42,
        "torch_version": str(torch.__version__),
        "fs": FS,
        "samples": SAMPLES,
        "classes": CLASSES,
        "epochs_max": args.epochs,
        "patience": args.patience,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 64,
        "normalization": "per-clip mean removal and RMS scaling; no fitted test-set statistics",
        "augmentation": "none in first benchmark",
        "rf_trees": 300,
        "rf_min_samples_leaf": 3,
        "scope": "Presegmented four-activity classification; no quiet/background class, onset/offset detection or clinical risk labels",
    }
    X = training["X"]
    y = training["y"]
    features = extract_features(X)
    rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=2,
    )
    rf.fit(features[tr], y[tr])
    rf_val = metrics(y[va], rf.predict(features[va]))
    joblib.dump(
        {"model": rf, "classes": CLASSES, "fs": FS},
        args.output / "random_forest.joblib",
    )
    normalized = normalize_clips(X)
    model = ActivityCNN()
    config["cnn_parameters"] = sum(p.numel() for p in model.parameters())
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(normalized[tr, None]), torch.from_numpy(y[tr]).long()
        ),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    history = []
    best = -1.0
    best_epoch = 0
    checkpoint = None
    cnn_val = None
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        losses = []
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        v = metrics(y[va], predict(model, normalized[va]).argmax(axis=1))
        row = {
            "epoch": epoch,
            "training_loss": float(np.mean(losses)),
            "validation_accuracy": v["accuracy"],
            "validation_macro_f1": v["macro_f1"],
            "seconds": time.time() - started,
        }
        history.append(row)
        (args.output / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(row), flush=True)
        if v["macro_f1"] > best + 1e-8:
            best = v["macro_f1"]
            best_epoch = epoch
            checkpoint = copy.deepcopy(model.state_dict())
            cnn_val = v
            torch.save(
                {"state_dict": checkpoint, "config": config, "epoch": epoch},
                args.output / "cnn.pt",
            )
        if epoch - best_epoch >= args.patience:
            break
    selected = "cnn" if cnn_val["macro_f1"] > rf_val["macro_f1"] else "random_forest"
    model.load_state_dict(checkpoint)
    # Evaluate the authors' supplied test partition only after both candidates are frozen.
    rf_scores = rf.predict_proba(extract_features(test["X"]))
    cnn_scores = predict(model, normalize_clips(test["X"]))
    result = {
        "selected_on_inner_validation": selected,
        "best_cnn_epoch": best_epoch,
        "models": {},
        "limitations": [
            audit["inner_validation"],
            audit["test_policy"],
            config["scope"],
            "Pilot results on the authors sensor; no transfer to the capstone wearable has been measured",
        ],
    }
    for name, scores, v in [
        ("random_forest", rf_scores, rf_val),
        ("cnn", cnn_scores, cnn_val),
    ]:
        pred = scores.argmax(axis=1)
        result["models"][name] = {
            "validation": v,
            "test": metrics(test["y"], pred),
            "test_by_language": {
                lang: metrics(
                    test["y"][test["language"] == lang], pred[test["language"] == lang]
                )
                for lang in sorted(set(test["language"]))
            },
        }
        np.savez_compressed(
            args.output / (name + "_test_predictions.npz"),
            scores=scores,
            y=test["y"],
            ids=test["ids"],
            language=test["language"],
        )
        print(
            "FINAL",
            name,
            result["models"][name]["test"]["accuracy"],
            result["models"][name]["test"]["macro_f1"],
            flush=True,
        )
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
