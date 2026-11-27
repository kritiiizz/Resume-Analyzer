"""Consolidate completed no-title runs without reading resume text."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import dataset_fingerprint, read_json, write_json
from .prepare_no_title import SEEDS

MODELS = ("dummy", "bow_nb", "tfidf_lr", "tfidf_svm", "distilbert_head_tail")
METRICS = (
    "accuracy", "macro_precision", "macro_recall", "macro_f1",
    "weighted_f1", "top3_accuracy",
)


def main() -> None:
    root = Path("reports/task_a_no_title")
    manifest = read_json("data/metadata_no_title/split_manifest.json")
    rows = []
    epochs = []
    for seed in SEEDS:
        report = root / f"seed_{seed}"
        for model in MODELS:
            test_paths = list((report / "runs" / "test").glob("*/metrics.json"))
            tests = [read_json(path) for path in test_paths]
            matching = [item for item in tests if item.get("model_name") == model]
            if len(matching) != 1:
                raise ValueError(f"Expected one completed test run for seed {seed}, {model}")
            test = matching[0]
            val_path = report / "runs" / "val" / test["run_id"] / "metrics.json"
            val = read_json(val_path)
            if test["n_samples"] != 249 or val["n_samples"] != 248:
                raise ValueError(f"Unexpected evaluation size for seed {seed}, {model}")
            if val["model_name"] != model:
                raise ValueError(f"Validation model mismatch for seed {seed}, {model}")
            test_hash = test["run_metadata"].get("dataset_sha256")
            if test_hash != dataset_fingerprint(manifest):
                raise ValueError(f"Input data fingerprint mismatch for seed {seed}, {model}")
            rows.append({
                "seed": seed, "model": model, "run_id": test["run_id"],
                "input_data_sha256": test_hash,
                **{f"val_{key}": val.get(key) for key in METRICS},
                **{f"test_{key}": test.get(key) for key in METRICS},
            })
        epoch_file = (
            Path(f"models/task_a_no_title/seed_{seed}/distilbert_head_tail")
            / "runs" / f"no_title_seed_{seed}" / "epoch_metrics.csv"
        )
        if not epoch_file.exists():
            epoch_file = report / "bert_epoch_metrics.csv"
        frame = pd.read_csv(epoch_file)
        required = {
            "epoch", "seed", "train_loss", "eval_loss", "eval_macro_f1",
            "test_loss", "test_macro_f1", "eval_accuracy", "test_accuracy",
        }
        if required - set(frame) or frame[sorted(required)].isna().any().any():
            raise ValueError(f"Incomplete epoch history for seed {seed}")
        if list(frame.epoch) != list(range(1, len(frame) + 1)) or len(frame) > 4:
            raise ValueError(f"Unexpected epoch sequence for seed {seed}")
        bert = next(row for row in rows if row["seed"] == seed and
                    row["model"] == "distilbert_head_tail")
        best = frame.loc[frame.eval_macro_f1.idxmax()]
        if abs(best.eval_macro_f1 - bert["val_macro_f1"]) > 1e-8:
            raise ValueError(f"Best validation metric mismatch for seed {seed}")
        if abs(best.test_macro_f1 - bert["test_macro_f1"]) > 1e-8:
            raise ValueError(f"Selected checkpoint test metric mismatch for seed {seed}")
        epochs.append(frame)
    summary = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(root / "test_metrics_by_seed.csv", index=False)
    selection = (
        summary.loc[summary.model == "distilbert_head_tail",
                    ["seed", "val_macro_f1"]]
        .sort_values(["val_macro_f1", "seed"], ascending=[False, True])
        .reset_index(drop=True)
    )
    selection["validation_rank"] = range(1, len(selection) + 1)
    selection["retain_model"] = selection.validation_rank <= 2
    selection.to_csv(root / "seed_selection_by_validation.csv", index=False)
    write_json(root / "retained_model_seeds.json", {
        "rule": "top two DistilBERT validation macro-F1 seeds; test not used",
        "seeds": selection.loc[selection.retain_model, "seed"].astype(int).tolist(),
    })
    long = summary.melt(
        id_vars=["seed", "model", "run_id", "input_data_sha256"],
        value_vars=[f"test_{key}" for key in METRICS],
        var_name="metric", value_name="value",
    )
    long.to_csv(root / "test_metrics_long.csv", index=False)
    aggregate = long.groupby(["model", "metric"], as_index=False).value.agg(
        mean="mean", std="std", min="min", max="max"
    )
    aggregate.to_csv(root / "test_metrics_mean_std.csv", index=False)
    pd.concat(epochs, ignore_index=True).sort_values(["seed", "epoch"]).to_csv(
        root / "bert_epoch_metrics_all_seeds.csv", index=False
    )
    print(summary[["seed", "model", "val_macro_f1", "test_macro_f1", "test_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
