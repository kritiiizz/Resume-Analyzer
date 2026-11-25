"""Summarize the historical seed 42 and two added with-title Task A seeds."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import read_json

SEEDS = (42, 123, 2026)
MODELS = ("dummy", "bow_nb", "tfidf_lr", "tfidf_svm", "distilbert_head_tail")
METRICS = (
    "accuracy", "macro_precision", "macro_recall", "macro_f1",
    "weighted_f1", "top3_accuracy",
)


def main() -> None:
    output = Path("reports/task_a_with_title")
    manifest = read_json("data/metadata/split_manifest.json")
    rows = []
    epochs = []
    for seed in SEEDS:
        report = Path("reports/task_a") if seed == 42 else output / f"seed_{seed}"
        for model in MODELS:
            matches = []
            for path in (report / "runs" / "test").glob("*/metrics.json"):
                metrics = read_json(path)
                if metrics.get("model_name") == model:
                    matches.append(metrics)
            if len(matches) != 1:
                raise ValueError(f"Expected one test report for seed {seed}, {model}")
            test = matches[0]
            val = read_json(report / "runs" / "val" / test["run_id"] / "metrics.json")
            if test["n_samples"] != 249 or val["n_samples"] != 248:
                raise ValueError(f"Split size mismatch: seed {seed}, {model}")
            if test["run_metadata"].get("dataset_sha256") != manifest["raw_file_sha256"]:
                raise ValueError(f"Dataset mismatch: seed {seed}, {model}")
            rows.append({
                "seed": seed, "model": model, "run_id": test["run_id"],
                "historical": seed == 42,
                "dataset_sha256": manifest["raw_file_sha256"],
                **{f"val_{key}": val.get(key) for key in METRICS},
                **{f"test_{key}": test.get(key) for key in METRICS},
            })
        if seed != 42:
            epoch_file = report / "bert_epoch_metrics.csv"
            frame = pd.read_csv(epoch_file)
            required = {
                "epoch", "seed", "train_loss", "eval_loss", "eval_macro_f1",
                "test_loss", "test_macro_f1", "eval_accuracy", "test_accuracy",
            }
            if required - set(frame) or frame[sorted(required)].isna().any().any():
                raise ValueError(f"Incomplete epoch history for seed {seed}")
            if frame.epoch.tolist() != list(range(1, len(frame) + 1)) or len(frame) > 4:
                raise ValueError(f"Unexpected epoch sequence for seed {seed}")
            bert_row = next(row for row in rows if row["seed"] == seed and
                            row["model"] == "distilbert_head_tail")
            best = frame.loc[frame.eval_macro_f1.idxmax()]
            if abs(best.eval_macro_f1 - bert_row["val_macro_f1"]) > 1e-8:
                raise ValueError(f"Best validation metric mismatch for seed {seed}")
            if abs(best.test_macro_f1 - bert_row["test_macro_f1"]) > 1e-8:
                raise ValueError(f"Selected checkpoint test mismatch for seed {seed}")
            epochs.append(frame)
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "test_metrics_by_seed.csv", index=False)
    long = summary.melt(
        id_vars=["seed", "model", "run_id", "historical", "dataset_sha256"],
        value_vars=[f"test_{name}" for name in METRICS],
        var_name="metric", value_name="value",
    )
    long.to_csv(output / "test_metrics_long.csv", index=False)
    long.groupby(["model", "metric"], as_index=False).value.agg(
        mean="mean", std="std", min="min", max="max"
    ).to_csv(output / "test_metrics_mean_std.csv", index=False)
    pd.concat(epochs, ignore_index=True).sort_values(["seed", "epoch"]).to_csv(
        output / "bert_epoch_metrics_new_seeds.csv", index=False
    )
    no_title_file = Path("reports/task_a_no_title/test_metrics_by_seed.csv")
    if no_title_file.exists():
        no_title = pd.read_csv(no_title_file)
        comparison = summary.merge(
            no_title[["seed", "model", *[f"test_{name}" for name in METRICS]]],
            on=["seed", "model"], suffixes=("_with_title", "_no_title"),
            validate="one_to_one",
        )
        if len(comparison) != len(summary):
            raise ValueError("No-title results do not cover every model/seed pair")
        for name in METRICS:
            comparison[f"test_{name}_title_gain"] = (
                comparison[f"test_{name}_with_title"]
                - comparison[f"test_{name}_no_title"]
            )
        comparison.to_csv(output / "paired_with_vs_no_title.csv", index=False)
    print(summary[["seed", "model", "val_macro_f1", "test_macro_f1", "test_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
