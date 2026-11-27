"""EDA on the frozen Task A dataset; aggregate figures contain no raw CV text."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import load_config, read_json, write_json


def run_eda(config_path: str = "configs/task_a.yaml") -> dict:
    cfg = load_config(config_path)
    label_col = cfg["dataset"]["label_column"]
    text_col = cfg["dataset"]["text_column"]
    splits = [
        pd.read_csv(
            Path(cfg["paths"]["split_dir"]) / f"task_a_{split}.csv"
        )
        for split in ("train", "val", "test")
    ]
    data = pd.concat(splits, ignore_index=True)
    data[text_col] = data[text_col].fillna("").astype(str)
    data["word_count"] = data[text_col].map(lambda t: len(t.split()))
    data["char_count"] = data[text_col].str.len()
    class_counts = data[label_col].value_counts().sort_values()
    output = Path(cfg["paths"]["report_dir"]) / "eda"
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.barh(class_counts.index, class_counts.values)
    ax.set(xlabel="Number of resumes", title="Class distribution")
    fig.tight_layout()
    fig.savefig(output / "class_distribution.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(data["word_count"], bins=40)
    ax.set(xlabel="Word count", ylabel="Resumes", title="Resume length distribution")
    fig.tight_layout()
    fig.savefig(output / "word_length_hist.png", dpi=170)
    plt.close(fig)
    stats = {
        "n_samples": int(len(data)),
        "n_classes": int(len(class_counts)),
        "class_distribution": {str(k): int(v) for k, v in class_counts.items()},
        "word_count": {str(k): float(v) for k, v in
                       data["word_count"].describe(percentiles=[0.5, 0.75, 0.9, 0.95]).items()},
        "character_count": {str(k): float(v) for k, v in
                            data["char_count"].describe(percentiles=[0.5, 0.75, 0.9, 0.95]).items()},
        "note": "Word count is not DistilBERT token count; no BERT tokenizer used.",
    }
    write_json(output / "summary.json", stats)
    print(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task_a.yaml")
    args = parser.parse_args()
    run_eda(args.config)


if __name__ == "__main__":
    main()
