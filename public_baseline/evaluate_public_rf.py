"""Recheck the frozen random forest on held-out public-dataset participants."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import joblib
import numpy as np

from train import FS, event_metrics, features


def md5_file(path):
    digest = hashlib.md5()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path, help="Trusted rf_combined.joblib")
    source.add_argument("--model-archive", type=Path,
                        help="Trusted SwallowDetection_v1.zip containing rf_combined.joblib")
    parser.add_argument("--output", type=Path,
                        default=Path("data/public_eval/rf_combined_heldout.json"))
    args = parser.parse_args()

    provenance = json.loads(Path("results/provenance.json").read_text())
    archive_hashes = {"signals_md5": md5_file(args.signals),
                      "labels_md5": md5_file(args.labels)}
    for name, actual in archive_hashes.items():
        if actual != provenance[name]:
            parser.error(f"{name} does not match the original public archive")

    if args.model_archive:
        with zipfile.ZipFile(args.model_archive) as archive:
            candidates = [name for name in archive.namelist()
                          if name.endswith("/results/rf_combined.joblib")]
            if len(candidates) != 1:
                parser.error("Expected one results/rf_combined.joblib in the model archive")
            model_bytes = archive.read(candidates[0])
        bundle = joblib.load(io.BytesIO(model_bytes))
        model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    else:
        bundle = joblib.load(args.model)
        model_sha256 = hashlib.sha256(args.model.read_bytes()).hexdigest()
    if bundle["fs"] != FS or len(bundle["columns"]) != 56:
        parser.error("Checkpoint is not the 4 kHz, four-channel combined RF model")

    splits = json.loads(Path("results/participant_splits.json").read_text())
    if set(splits["test"]) & (set(splits["train"]) | set(splits["validation"])):
        parser.error("Participant split is not disjoint")
    records = [record for record in json.loads(Path("results/records.json").read_text())
               if record["participant"] in splits["test"]]
    scores, ids = [], []
    with zipfile.ZipFile(args.signals) as signals, zipfile.ZipFile(args.labels) as labels:
        for index, record in enumerate(records, 1):
            stem = record["id"]
            raw = signals.read(f"signals/{stem}.npy")
            if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError(f"Signal hash differs from frozen manifest: {stem}")
            x = np.load(io.BytesIO(raw), allow_pickle=False)
            rows = list(csv.DictReader(io.StringIO(
                labels.read(f"labels/{stem}.csv").decode("utf-8-sig"))))
            intervals = sorted([[float(row["start"]) / 60, float(row["end"]) / 60]
                                for row in rows])
            if intervals != record["intervals"]:
                raise ValueError(f"Labels differ from frozen manifest: {stem}")
            matrix, times = features(x)
            if not np.allclose(times, record["times"], atol=1e-9):
                raise ValueError(f"Feature grid differs from frozen manifest: {stem}")
            score = bundle["model"].predict_proba(matrix[:, bundle["columns"]])[:, 1]
            scores.extend(score)
            ids.extend([stem] * len(score))
            if index % 25 == 0 or index == len(records):
                print(f"Evaluated {index}/{len(records)} held-out recordings", flush=True)

    ids = np.array(ids)
    scores = np.array(scores)
    measured = event_metrics(records, ids, scores, bundle["threshold"])
    strict = event_metrics(records, ids, scores, bundle["threshold"], iou=0.5)
    expected = json.loads(Path("results/metrics.json").read_text())["models"]["rf_combined"]
    for key in ("tp", "fp", "fn"):
        if measured[key] != expected["test_event"][key]:
            raise ValueError(f"Recomputed {key} differs from the saved report")
        if strict[key] != expected["test_event_strict_iou_0_5"][key]:
            raise ValueError(f"Recomputed strict {key} differs from the saved report")

    report = {"task": "Annotated-swallow interval detection on the public dataset",
              "model": "rf_combined", "model_sha256": model_sha256,
              "archive_md5": archive_hashes, "test_participants": len(splits["test"]),
              "test_recordings": len(records), "test_windows": len(scores),
              "threshold_from_validation": bundle["threshold"],
              "event_iou_0_30": {key: measured[key] for key in
                                 ("precision", "recall", "f1", "tp", "fp", "fn",
                                  "false_events_per_recording_minute")},
              "event_iou_0_50": {key: strict[key] for key in
                                 ("precision", "recall", "f1", "tp", "fp", "fn")},
              "matches_frozen_report": True,
              "scope": "Public-dataset held-out split; not a test of wearable recordings or clinical diagnosis."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Held-out event F1 (IoU 0.30): {measured['f1']:.3f}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
