"""Read the supplied STVS activity archives without mixing them with HRCA."""

import hashlib
import io
import zipfile
from collections import Counter
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

CLASSES = ["cough", "speak", "swallow", "throatclear"]
FS = 6400
SAMPLES = 4000
EXPECTED_MD5 = {
    "training": "73ea26bc7bde041ff0690094f7477421",
    "test": "bc9aebdee8a00d21d1da968ab3bb8027",
}


def read_archive(path, split):
    digest = hashlib.md5()
    with path.open("rb") as f:
        while b := f.read(4194304):
            digest.update(b)
    if digest.hexdigest() != EXPECTED_MD5[split]:
        raise ValueError(
            f"{split} archive checksum does not match the supplied release"
        )
    arrays = []
    labels = []
    ids = []
    languages = []
    hashes = []
    files = []
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            if not name.endswith(".npy") or name.startswith("__MACOSX"):
                continue
            activity = name.split("/")[-1].removesuffix(".npy")
            if activity not in CLASSES:
                raise ValueError(f"Unknown activity: {name}")
            x = np.load(io.BytesIO(z.read(name)), allow_pickle=False)
            if x.ndim != 2 or x.shape[1] != SAMPLES or not np.isfinite(x).all():
                raise ValueError(f"Invalid recording array: {name}")
            files.append({"file": name, "shape": list(x.shape), "class": activity})
            arrays.append(x.astype(np.float32))
            labels.extend([CLASSES.index(activity)] * len(x))
            ids.extend([f"{name}:{i}" for i in range(len(x))])
            languages.extend(
                [name.split("/")[-2] if split == "test" else "training_English"]
                * len(x)
            )
            hashes.extend(
                [hashlib.sha256(row.astype("<f8").tobytes()).hexdigest() for row in x]
            )
    return {
        "X": np.vstack(arrays),
        "y": np.array(labels),
        "ids": np.array(ids),
        "language": np.array(languages),
        "hashes": np.array(hashes),
        "files": files,
        "md5": digest.hexdigest(),
    }


def audit_and_split(training, test, seed=42):
    overlap = set(training["hashes"]) & set(test["hashes"])
    if overlap:
        raise ValueError(
            f"Exact waveform leakage across supplied train/test: {len(overlap)} hashes"
        )
    conflicts = []
    for dataset in [training, test]:
        seen = {}
        bad = set()
        for h, y in zip(dataset["hashes"], dataset["y"]):
            if h in seen and seen[h] != int(y):
                bad.add(h)
            seen[h] = int(y)
        if dataset is test and bad:
            raise ValueError(
                "Test waveform has conflicting labels; external evaluation requires review"
            )
        if dataset is training:
            conflicts = sorted(bad)
    excluded = np.flatnonzero(np.isin(training["hashes"], conflicts))
    eligible = np.flatnonzero(~np.isin(training["hashes"], conflicts))
    train, val = next(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(
            training["X"][eligible],
            training["y"][eligible],
            training["hashes"][eligible],
        )
    )
    train, val = eligible[train], eligible[val]
    assert not set(training["hashes"][train]) & set(training["hashes"][val])
    audit = {
        "training_clips": len(training["y"]),
        "test_clips": len(test["y"]),
        "samples_per_clip": SAMPLES,
        "sample_rate_hz": FS,
        "clip_duration_seconds": SAMPLES / FS,
        "classes": CLASSES,
        "training_counts": dict(Counter(CLASSES[int(y)] for y in training["y"])),
        "test_counts": dict(Counter(CLASSES[int(y)] for y in test["y"])),
        "exact_train_test_overlap": len(overlap),
        "conflicting_training_hashes": len(conflicts),
        "excluded_ambiguous_training_ids": training["ids"][excluded].tolist(),
        "conflict_policy": "Exclude all training rows sharing a hash with conflicting class labels; do not choose a replacement label",
        "duplicate_training_clips": len(training["hashes"])
        - len(set(training["hashes"])),
        "duplicate_test_clips": len(test["hashes"]) - len(set(test["hashes"])),
        "supplied_split_subject_counts_reported_by_authors": {
            "training": 12,
            "test": 20,
        },
        "subject_ids_present_in_arrays": False,
        "inner_validation": "20% waveform-hash-grouped, class-stratified clip split; NOT subject-disjoint",
        "test_policy": "Supplied external test partition evaluated after model selection; participant separation follows authors documentation and cannot be independently reconstructed from arrays",
        "training_files": training["files"],
        "test_files": test["files"],
    }
    return train, val, audit


def normalize_clips(x):
    x = np.asarray(x, dtype=np.float32)
    centered = x - x.mean(axis=1, keepdims=True)
    return centered / np.maximum(
        np.sqrt((centered**2).mean(axis=1, keepdims=True)), 1e-8
    )
