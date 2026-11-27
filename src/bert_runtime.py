"""Load a trusted fine-tuned DistilBERT and run consistent Head+Tail inference."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .bert_data import encode_head_tail
from .bert_metrics import softmax_probabilities
from .common import read_json


def resolve_best_dir(model_root: str | Path, model_dir: str | Path | None = None) -> Path:
    root = Path(model_root)
    if model_dir:
        path = Path(model_dir)
        if (path / "best" / "config.json").is_file():
            path = path / "best"
    else:
        latest = read_json(root / "latest.json")
        relative = Path(latest["best_relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid latest model relative path")
        path = root / relative
    if not (path / "config.json").is_file() or not (path / "metadata.json").is_file():
        raise FileNotFoundError(f"Not a complete exported checkpoint: {path}")
    return path


def load_model(best_dir: str | Path, device: str = "auto"):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    best_dir = Path(best_dir)
    meta = read_json(best_dir / "metadata.json")
    if meta.get("is_smoke"):
        raise ValueError("Refusing to evaluate/publish a debug-only smoke checkpoint")
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if device not in ("cuda", "mps", "cpu"):
        raise ValueError("device must be auto/cuda/mps/cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(best_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(best_dir))
    labels = sorted(meta["label2id"], key=meta["label2id"].get)
    if len(labels) != model.config.num_labels:
        raise ValueError("Saved label mapping does not match model output dimension")
    config_labels = [
        model.config.id2label[i] for i in range(model.config.num_labels)
    ]
    if labels != config_labels:
        raise ValueError("Model labels differ from metadata")
    model.to(torch.device(device))
    model.eval()
    return model, tokenizer, meta, labels, device


def predict_texts(
    texts: list[str],
    model,
    tokenizer,
    metadata: dict,
    *,
    batch_size: int = 4,
    device: str = "cpu",
) -> tuple[np.ndarray, float]:
    import torch
    from transformers import DataCollatorWithPadding

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not texts:
        raise ValueError("No texts supplied")
    examples = [
        encode_head_tail(
            text, tokenizer, strategy=metadata["truncation"],
            preprocessing=metadata["preprocessing"],
        )[0]
        for text in texts
    ]
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer, pad_to_multiple_of=int(
            metadata.get("training", {}).get("pad_to_multiple_of", 64)
        )
    )
    batches = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index in range(0, len(examples), batch_size):
            features = collator(examples[index : index + batch_size])
            features = {key: tensor.to(device) for key, tensor in features.items()}
            outputs = model(**features)
            batches.append(outputs.logits.detach().float().cpu().numpy())
    elapsed = time.perf_counter() - started
    return softmax_probabilities(np.concatenate(batches, axis=0)), elapsed
