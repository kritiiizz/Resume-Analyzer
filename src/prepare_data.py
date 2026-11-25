"""Validate, deduplicate, and freeze stratified Train/Val/Test splits.

Important: split on CVs, not on text chunks, and never fit a vectorizer before
this stage. Exact-text conflicts with different labels are quarantined.
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .common import load_config, sha256_file, sha256_text, utc_now, write_json


def canonical_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def prepare_frame(raw: pd.DataFrame, text_column: str = "Resume_str",
                  label_column: str = "Category") -> tuple[pd.DataFrame, dict]:
    needed = {text_column, label_column}
    missing = needed.difference(raw.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    work = raw[[text_column, label_column]].copy()
    missing_rows = int(
        work[text_column].isna().sum() + work[label_column].isna().sum()
    )
    work = work.dropna(subset=[text_column, label_column])
    work[text_column] = work[text_column].astype(str)
    work[label_column] = work[label_column].astype(str).str.strip()
    work = work.loc[
        work[text_column].str.strip().ne("") & work[label_column].ne("")
    ].copy()
    invalid_count = int(len(raw) - len(work))
    work["text_hash"] = work[text_column].map(lambda t: sha256_text(canonical_for_dedup(t)))
    conflict_hashes = (
        work.groupby("text_hash")[label_column].nunique().loc[lambda s: s > 1].index
    )
    conflicts = int(work["text_hash"].isin(conflict_hashes).sum())
    work = work.loc[~work["text_hash"].isin(conflict_hashes)].copy()
    duplicate_count = int(work.duplicated("text_hash", keep="first").sum())
    work = work.drop_duplicates("text_hash", keep="first")
    work = work.sort_values("text_hash").reset_index(drop=True)
    work.insert(0, "sample_id", work["text_hash"])
    report = {
        "raw_rows": int(len(raw)),
        "invalid_or_empty_rows_removed": invalid_count,
        "raw_missing_cells": missing_rows,
        "conflicting_label_rows_quarantined": conflicts,
        "duplicate_text_rows_removed": duplicate_count,
        "unique_rows": int(len(work)),
        "n_classes": int(work[label_column].nunique()),
        "class_distribution": {str(k): int(v) for k, v in work[label_column].value_counts().sort_index().items()},
        "deduplication": "NFKC + casefold + whitespace-folded exact text",
        "near_duplicates": "not automatically removed; manually audit if needed",
    }
    return work, report


def split_frame(frame: pd.DataFrame, label_column: str, *,
                seed: int = 42, train_fraction: float = 0.8,
                val_fraction: float = 0.1, test_fraction: float = 0.1) -> dict[str, pd.DataFrame]:
    if abs(train_fraction + val_fraction + test_fraction - 1) > 1e-8:
        raise ValueError("Split fractions must sum to 1")
    if min(train_fraction, val_fraction, test_fraction) <= 0:
        raise ValueError("All splits must be non-empty")
    counts = frame[label_column].value_counts()
    if counts.min() < 3:
        raise ValueError("At least 3 samples per class required; check rare classes")
    train, holdout = train_test_split(
        frame, test_size=val_fraction + test_fraction, random_state=seed,
        stratify=frame[label_column],
    )
    val, test = train_test_split(
        holdout, test_size=test_fraction / (val_fraction + test_fraction),
        random_state=seed, stratify=holdout[label_column],
    )
    result = {"train": train, "val": val, "test": test}
    for name, part in result.items():
        if set(part[label_column]) != set(frame[label_column]):
            raise ValueError(f"{name} does not cover every category; need more per-class samples")
    ids = [set(part["sample_id"]) for part in result.values()]
    if any(ids[i] & ids[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("Cross-split leakage detected")
    return {k: v.sort_values("sample_id").reset_index(drop=True) for k, v in result.items()}


def prepare_dataset(config_path: str = "configs/task_a.yaml") -> dict:
    cfg = load_config(config_path)
    raw_path = Path(cfg["dataset"]["raw_path"])
    if not raw_path.is_file():
        raise FileNotFoundError(
            f"{raw_path} not found. Download it first with python -m src.download_data"
        )
    raw = pd.read_csv(raw_path)
    text_col, label_col = cfg["dataset"]["text_column"], cfg["dataset"]["label_column"]
    frame, audit = prepare_frame(raw, text_col, label_col)
    labels = sorted(frame[label_col].unique().tolist())
    split_cfg = cfg["split"]
    splits = split_frame(
        frame, label_col, seed=int(split_cfg["seed"]),
        train_fraction=float(split_cfg["train_fraction"]),
        val_fraction=float(split_cfg["val_fraction"]),
        test_fraction=float(split_cfg["test_fraction"]),
    )
    split_dir = Path(cfg["paths"]["split_dir"])
    meta_dir = Path(cfg["paths"]["metadata_dir"])
    split_dir.mkdir(parents=True, exist_ok=True)
    for split, data in splits.items():
        data.to_csv(split_dir / f"task_a_{split}.csv", index=False)
    label2id = {label: i for i, label in enumerate(labels)}
    write_json(meta_dir / "label2id.json", label2id)
    write_json(meta_dir / "id2label.json", {str(i): label for label, i in label2id.items()})
    write_json(meta_dir / "data_audit.json", audit)
    manifest = {
        "dataset_source": cfg["dataset"]["source"],
        "raw_file_sha256": sha256_file(raw_path),
        "generated_at_utc": utc_now(),
        "random_seed": int(split_cfg["seed"]),
        "split_fractions": {k: float(split_cfg[k + "_fraction"]) for k in ("train", "val", "test")},
        "n_classes": len(labels),
        "split_class_counts": {
            k: {str(c): int(n) for c, n in df[label_col].value_counts().sort_index().items()}
            for k, df in splits.items()
        },
        "split_sample_ids": {k: df["sample_id"].tolist() for k, df in splits.items()},
        "label2id": label2id,
    }
    write_json(meta_dir / "split_manifest.json", manifest)
    print(f"Cleaned {len(frame)} CVs across {len(labels)} categories")
    print("Split counts:", {name: len(part) for name, part in splits.items()})
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task_a.yaml")
    args = parser.parse_args()
    prepare_dataset(args.config)


if __name__ == "__main__":
    main()
