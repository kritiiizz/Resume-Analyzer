"""Minimal non-UI inference utility for trustworthy saved classical pipelines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from .common import load_config, read_json
from .train_classical import SUPPORTED_MODELS


def predict_resume(text: str, model_name: str, config_path: str = "configs/task_a.yaml",
                   top_k: int = 3) -> dict:
    if not text or not text.strip():
        raise ValueError("Resume text cannot be blank")
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model {model_name}")
    cfg = load_config(config_path)
    model_dir = Path(cfg["paths"]["model_dir"]) / model_name
    # joblib/pickle must never load artifacts supplied by untrusted parties.
    model = joblib.load(model_dir / "pipeline.joblib")
    metadata = read_json(model_dir / "metadata.json")
    labels = model.named_steps["classifier"].classes_.tolist()
    predicted = str(model.predict([text])[0])
    if hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba([text])[0], dtype=float)
        score_type = "probability"
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function([text])[0], dtype=float)
        score_type = "decision_score"
    else:
        scores = None
        score_type = None
    result = {
        "category": predicted,
        "model_name": model_name,
        "run_id": metadata["run_id"],
        "score_type": score_type,
        "top_predictions": [],
    }
    if scores is not None:
        best_indices = np.argsort(-scores, kind="stable")[:top_k]
        result["top_predictions"] = [
            {"category": str(labels[i]), "score": float(scores[i])} for i in best_indices
        ]
    else:
        result["top_predictions"] = [{"category": predicted, "score": None}]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--file", required=True, help="UTF-8 resume text file")
    parser.add_argument("--config", default="configs/task_a.yaml")
    args = parser.parse_args()
    text = Path(args.file).read_text(encoding="utf-8")
    print(json.dumps(predict_resume(text, args.model, args.config), indent=2))


if __name__ == "__main__":
    main()
