"""Helpers so the baselines and DistilBERT plug into the demo and the error analysis.

Baselines (scikit-learn): save the whole fitted pipeline (vectoriser + classifier)
    import joblib; joblib.dump(pipeline, "models/tfidf_linearsvc.joblib")

DistilBERT: save model + tokenizer in one folder, with label names in the config
    model.config.id2label = {i: name for i, name in enumerate(label_names)}
    model.config.label2id = {name: i for i, name in enumerate(label_names)}
    model.save_pretrained("models/distilbert_head_tail"); tokenizer.save_pretrained("models/distilbert_head_tail")

Test predictions of any model (for src/error_analysis.py):
    from src.export_for_demo import save_predictions
    save_predictions("reports/predictions/distilbert.csv", test_df, y_true, y_pred, proba, label_names)
"""
import os

import numpy as np
import pandas as pd


def save_predictions(path, test_df, y_true, y_pred, proba=None, labels=None, text_col="Resume_str"):
    """y_true / y_pred: label names (or indices into `labels`). proba: [n, n_classes] array, optional."""
    def names(v):
        v = np.asarray(v)
        return [labels[i] for i in v] if labels is not None and np.issubdtype(v.dtype, np.integer) else list(v)
    out = pd.DataFrame({"row": test_df.index})
    if "ID" in test_df.columns:
        out["ID"] = test_df["ID"].values
    out["true"], out["pred"] = names(y_true), names(y_pred)
    out["correct"] = out["true"] == out["pred"]
    if proba is not None and labels is not None:
        proba = np.asarray(proba)
        out["confidence"] = proba.max(1).round(4)
        top3 = np.argsort(-proba, axis=1)[:, :3]
        out["top3"] = ["; ".join(f"{labels[j]} {proba[r, j]:.2f}" for j in top3[r]) for r in range(len(proba))]
    out["n_words"] = test_df[text_col].astype(str).str.split().str.len().values
    out["text_start"] = test_df[text_col].astype(str).str[:300].values
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.to_csv(path, index=False)
    return out
