"""Evaluate saved pipelines or external models on the frozen Task A splits.

--final-test must be explicitly supplied before reading the test CSV.
External models should export CSV with sample_id and pred_category.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .common import dataset_fingerprint, load_config, read_json
from .evaluation import export_evaluation, update_comparison
from .train_classical import SUPPORTED_MODELS


def evaluate_saved_model(name: str, cfg: dict, frame: pd.DataFrame,
                         labels: list[str]) -> tuple[list[str], np.ndarray | None, str | None, dict]:
    model_dir = Path(cfg["paths"]["model_dir"]) / name
    metadata = read_json(model_dir / "metadata.json")
    manifest = read_json(Path(cfg["paths"]["metadata_dir"]) / "split_manifest.json")
    if metadata.get("dataset_sha256") != dataset_fingerprint(manifest):
        raise ValueError("Dataset fingerprint differs from the trained pipeline")
    model = joblib.load(model_dir / "pipeline.joblib")  # load only trusted artifacts
    x = frame[cfg["dataset"]["text_column"]].fillna("").tolist()
    started = time.perf_counter()
    preds = model.predict(x).tolist()
    predict_seconds = time.perf_counter() - started
    classifier = model.named_steps["classifier"]
    model_labels = classifier.classes_.tolist()
    if set(model_labels) != set(labels):
        raise ValueError("Model's classes differ from the frozen label mapping")
    score_type = None
    scores = None
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(x)
        score_type = "probability"
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(x)
        score_type = "decision_score"
    else:
        raw = None
    if raw is not None:
        raw = np.asarray(raw)
        scores = raw[:, [model_labels.index(label) for label in labels]]
    extra = {
        "training_run": metadata.get("run_id"),
        "validation_macro_f1_at_selection": metadata.get("validation_macro_f1"),
        "dataset_sha256": metadata.get("dataset_sha256"),
        "prediction_seconds": predict_seconds,
        "batch_predict_ms_per_cv": 1000 * predict_seconds / len(frame),
        "batch_size": len(frame),
    }
    return preds, scores, score_type, extra


def load_external_predictions(path: str | Path, frame: pd.DataFrame
                              ) -> list[str]:
    pred = pd.read_csv(path, dtype={"sample_id": str})
    required = {"sample_id", "pred_category"}
    if required - set(pred.columns):
        raise ValueError(f"Prediction file requires {sorted(required)}")
    if pred["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id in external predictions")
    if len(pred) != len(frame) or set(pred["sample_id"]) != set(frame["sample_id"]):
        raise ValueError("External predictions do not cover exactly the frozen split")
    ordered = frame[["sample_id"]].merge(
        pred[["sample_id", "pred_category"]],
        on="sample_id", how="left", validate="one_to_one", sort=False
    )
    if ordered["pred_category"].isna().any():
        raise ValueError("Missing predictions after sample_id alignment")
    return ordered["pred_category"].astype(str).tolist()


def evaluate(config: str, split: str, model_names: list[str],
             *, final_test: bool = False,
             predictions_file: str | None = None,
             external_model: str | None = None) -> list[dict]:
    if split == "test" and not final_test:
        raise PermissionError("Test evaluation requires --final-test")
    cfg = load_config(config)
    directory = Path(cfg["paths"]["split_dir"])
    frame = pd.read_csv(
        directory / f"task_a_{split}.csv", dtype={"sample_id": str}
    ).fillna("")
    label2id = read_json(Path(cfg["paths"]["metadata_dir"]) / "label2id.json")
    labels = [name for name, _ in sorted(label2id.items(), key=lambda x: x[1])]
    if not set(frame[cfg["dataset"]["label_column"]]).issubset(set(labels)):
        raise ValueError("Unknown labels in evaluation split")
    summary_rows = []
    if predictions_file is not None:
        if not external_model or model_names:
            raise ValueError("External CSV requires --external-model and no --models")
        preds = load_external_predictions(predictions_file, frame)
        metadata = {"prediction_source": str(predictions_file)}
        summary_rows.append(export_evaluation(
            frame, labels=labels, predictions=preds, scores=None, score_type=None,
            model=external_model, run_id=external_model + "_external_" + split,
            split=split, report_root=cfg["paths"]["report_dir"], metadata=metadata,
        ))
    else:
        for name in model_names:
            preds, scores, score_type, details = evaluate_saved_model(
                name, cfg, frame, labels
            )
            run_id = details["training_run"] or name
            summary_rows.append(export_evaluation(
                frame, labels=labels, predictions=preds, scores=scores,
                score_type=score_type, model=name, run_id=run_id,
                split=split, report_root=cfg["paths"]["report_dir"], metadata=details,
            ))
    output = update_comparison(cfg["paths"]["report_dir"], summary_rows, split)
    for row in summary_rows:
        print(f"{row['model_name']}: {split} Macro-F1={row['macro_f1']:.4f}; Accuracy={row['accuracy']:.4f}")
    print(f"Comparison saved to {output}")
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task_a.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_MODELS, default=None)
    parser.add_argument("--predictions-file", default=None)
    parser.add_argument("--external-model", default=None)
    args = parser.parse_args()
    names = args.models if args.models is not None else (
        [] if args.predictions_file else list(SUPPORTED_MODELS)
    )
    evaluate(args.config, args.split, names, final_test=args.final_test,
             predictions_file=args.predictions_file, external_model=args.external_model)


if __name__ == "__main__":
    main()
