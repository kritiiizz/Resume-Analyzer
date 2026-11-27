"""Transformers Trainer callback metrics for the canonical Task A label order."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def compute_bert_metrics(eval_pred, num_labels: int) -> dict[str, float]:
    logits = eval_pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits)
    if logits.ndim != 2 or logits.shape[1] != num_labels:
        raise ValueError("Unexpected logits shape")
    gold = np.asarray(eval_pred.label_ids)
    if gold.ndim != 1 or len(gold) != len(logits):
        raise ValueError("Gold labels shape mismatch")
    pred = np.argmax(logits, axis=-1)
    labels = list(range(num_labels))
    p, r, f1, _ = precision_recall_fscore_support(
        gold, pred, labels=labels, average="macro", zero_division=0
    )
    return {
        "macro_f1": float(f1),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "accuracy": float(accuracy_score(gold, pred)),
        "weighted_f1": float(
            f1_score(gold, pred, labels=labels, average="weighted", zero_division=0)
        ),
    }


def softmax_probabilities(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or not np.isfinite(logits).all():
        raise ValueError("Logits must be a finite 2D array")
    centered = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(centered)
    return exp / exp.sum(axis=1, keepdims=True)
