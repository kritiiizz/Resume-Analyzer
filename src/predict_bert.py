"""Headless inference from an exported Task A DistilBERT checkpoint (no UI)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bert_runtime import load_model, predict_texts, resolve_best_dir
from .common import load_config


def predict_resume_bert(
    text: str,
    *,
    config_path: str = "configs/task_a.yaml",
    model_dir: str | None = None,
    device: str = "auto",
    top_k: int = 3,
) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Resume text must be nonempty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    cfg = load_config(config_path)
    best = resolve_best_dir(
        Path(cfg["paths"]["model_dir"]) / "distilbert_head_tail", model_dir
    )
    model, tokenizer, meta, labels, chosen_device = load_model(best, device)
    proba, elapsed = predict_texts(
        [text], model, tokenizer, meta, batch_size=1, device=chosen_device
    )
    scores = proba[0]
    ranking = scores.argsort()[::-1][:top_k]
    return {
        "category": labels[int(ranking[0])],
        "model_name": "distilbert_head_tail",
        "run_id": meta["run_id"],
        "score_type": "uncalibrated_softmax_probability",
        "top_predictions": [
            {"category": labels[int(i)], "score": float(scores[i])}
            for i in ranking
        ],
        "inference_seconds": elapsed,
        "device": chosen_device,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, help="UTF-8 resume text")
    parser.add_argument("--config", default="configs/task_a.yaml")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    result = predict_resume_bert(
        Path(args.file).read_text(encoding="utf-8"),
        config_path=args.config, model_dir=args.model_dir,
        device=args.device, top_k=args.top_k,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
