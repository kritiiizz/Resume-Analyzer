"""Model-agnostic metric computation and artifact export.

Ground truth comes from the frozen split, never from external prediction files.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .common import utc_now, write_json


def compute_metrics(y_true: list[str], y_pred: list[str],
                    labels: list[str]) -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray]:
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("y_true and y_pred must have equal nonzero lengths")
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate labels in canonical mapping")
    if set(y_true) - set(labels) or set(y_pred) - set(labels):
        raise ValueError("Found labels outside frozen label mapping")
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    raw = confusion_matrix(y_true, y_pred, labels=labels)
    normalized = np.divide(
        raw, raw.sum(axis=1, keepdims=True),
        out=np.zeros_like(raw, dtype=float),
        where=raw.sum(axis=1, keepdims=True) != 0,
    )
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    per_class = pd.DataFrame([
        {"category": label, **{k: report[label][k] for k in
         ("precision", "recall", "f1-score", "support")}}
        for label in labels
    ])
    summary = {
        "n_samples": len(y_true),
        "n_classes": len(labels),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f),
        "weighted_f1": float(weighted_f1),
    }
    assert int(raw.sum()) == len(y_true)
    return summary, per_class, raw, normalized


def plot_confusion(matrix: np.ndarray, labels: list[str],
                   path: Path, normalized: bool) -> None:
    fig, ax = plt.subplots(figsize=(16, 13))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    ax.set(
        xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
        xticklabels=labels, yticklabels=labels,
        xlabel="Predicted category", ylabel="True category",
        title="Row-normalized confusion matrix" if normalized else "Confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=80, ha="right", rotation_mode="anchor")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def export_evaluation(
    frame: pd.DataFrame,
    *,
    labels: list[str],
    predictions: list[str],
    scores: np.ndarray | None,
    score_type: str | None,
    model: str,
    run_id: str,
    split: str,
    report_root: str | Path,
    metadata: dict | None = None,
) -> dict:
    if split not in ("val", "test"):
        raise ValueError("Only val or test evaluation supported")
    if len(frame) != len(predictions):
        raise ValueError("Prediction count does not match reference split")
    if frame["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id in reference split")
    y_true = frame["Category"].astype(str).tolist()
    summary, per_class, cm, cm_norm = compute_metrics(y_true, list(predictions), labels)
    root = Path(report_root)
    out = root / "runs" / split / run_id
    out.mkdir(parents=True, exist_ok=True)
    summary.update({
        "model_name": model,
        "run_id": run_id,
        "split": split,
        "evaluated_at_utc": utc_now(),
        "score_type": score_type,
        "run_metadata": metadata or {},
    })
    result = frame[["sample_id"]].copy()
    result["true_category"] = y_true
    result["pred_category"] = list(predictions)
    result["is_correct"] = result["true_category"] == result["pred_category"]
    if scores is not None:
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (len(frame), len(labels)):
            raise ValueError(f"Scores shape must be {(len(frame), len(labels))}")
        if not np.isfinite(scores).all():
            raise ValueError("Scores contain NaN or infinite values")
        result["scores_json"] = [json.dumps(row.tolist()) for row in scores]
        positions = np.argsort(-scores, axis=1, kind="stable")[:, : min(3, len(labels))]
        result["top3_categories_json"] = [
            json.dumps([labels[i] for i in row]) for row in positions
        ]
        summary["top3_accuracy"] = float(np.mean([
            y_true[index] in [labels[i] for i in row]
            for index, row in enumerate(positions)
        ]))
    else:
        result["scores_json"] = ""
        result["top3_categories_json"] = ""
    write_json(out / "metrics.json", summary)
    per_class.to_csv(out / "classification_report.csv", index=False)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out / "confusion_matrix.csv")
    pd.DataFrame(cm_norm, index=labels, columns=labels).to_csv(
        out / "confusion_matrix_normalized.csv"
    )
    result.to_csv(out / "predictions.csv", index=False)
    result.loc[~result["is_correct"]].to_csv(out / "errors.csv", index=False)
    plot_confusion(cm, labels, out / "confusion_matrix.png", normalized=False)
    plot_confusion(
        cm_norm, labels, out / "confusion_matrix_normalized.png", normalized=True
    )
    return summary


def update_comparison(report_root: str | Path, rows: list[dict], split: str) -> Path:
    root = Path(report_root)
    root.mkdir(parents=True, exist_ok=True)
    file = root / f"comparison_{split}.csv"
    flattened = []
    for row in rows:
        flattened.append({
            key: row.get(key) for key in (
                "run_id", "model_name", "split", "n_samples", "accuracy",
                "macro_precision", "macro_recall", "macro_f1",
                "weighted_f1", "top3_accuracy", "score_type", "evaluated_at_utc"
            )
        })
    fresh = pd.DataFrame(flattened)
    if file.exists():
        old = pd.read_csv(file)
        fresh = pd.concat([old, fresh], ignore_index=True)
        fresh = fresh.drop_duplicates(subset=["run_id", "split"], keep="last")
    fresh.sort_values(["macro_f1", "model_name"], ascending=[False, True]).to_csv(
        file, index=False
    )
    subset = fresh.loc[fresh["split"] == split].sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * len(subset) + 1)))
    ax.barh(subset["model_name"], subset["macro_f1"])
    ax.set(xlabel="Macro-F1", xlim=(0, 1), title=f"Task A model comparison ({split})")
    fig.tight_layout()
    fig.savefig(root / f"comparison_{split}.png", dpi=170)
    plt.close(fig)
    return file
