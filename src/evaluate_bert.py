"""Evaluate frozen DistilBERT Head+Tail on validation or gated final test.

No hyperparameter selection is performed here. Results use the SAME
evaluation.export_evaluation/update_comparison functions as classical baselines.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .bert_data import assert_frozen_split, split_fingerprint
from .bert_runtime import load_model, predict_texts, resolve_best_dir
from .common import dataset_fingerprint, load_config, read_json
from .evaluation import export_evaluation, update_comparison


def evaluate_bert(
    config_path: str = "configs/task_a.yaml",
    *,
    split: str = "val",
    final_test: bool = False,
    model_dir: str | None = None,
    batch_size: int = 4,
    device: str = "auto",
) -> dict:
    if split not in ("val", "test"):
        raise ValueError("Split must be val or test")
    if split == "test" and not final_test:
        raise PermissionError("Test access requires explicit --final-test")
    cfg = load_config(config_path)
    meta_root = Path(cfg["paths"]["metadata_dir"])
    manifest = read_json(meta_root / "split_manifest.json")
    frozen_label2id = read_json(meta_root / "label2id.json")
    best_dir = resolve_best_dir(
        Path(cfg["paths"]["model_dir"]) / "distilbert_head_tail", model_dir
    )
    model, tokenizer, meta, labels, chosen_device = load_model(best_dir, device)
    if meta["dataset_sha256"] != dataset_fingerprint(manifest):
        raise ValueError("Dataset hash differs from the trained checkpoint")
    if meta["split_fingerprint"] != split_fingerprint(manifest):
        raise ValueError("Split fingerprint differs from the trained checkpoint")
    if meta["label2id"] != frozen_label2id:
        raise ValueError("Label mapping differs from the trained checkpoint")

    frame = pd.read_csv(
        Path(cfg["paths"]["split_dir"]) / f"task_a_{split}.csv",
        dtype={"sample_id": str},
    ).fillna("")
    assert_frozen_split(frame, manifest, split)
    data = frame[cfg["dataset"]["text_column"]].astype(str).tolist()
    probabilities, seconds = predict_texts(
        data, model, tokenizer, meta, batch_size=batch_size, device=chosen_device
    )
    pred_labels = [labels[index] for index in probabilities.argmax(axis=1)]
    result = export_evaluation(
        frame, labels=labels, predictions=pred_labels, scores=probabilities,
        score_type="uncalibrated_softmax_probability",
        model="distilbert_head_tail", run_id=meta["run_id"], split=split,
        report_root=cfg["paths"]["report_dir"],
        metadata={
            "dataset_sha256": meta["dataset_sha256"],
            "split_fingerprint": meta["split_fingerprint"],
            "training_seconds": meta["training_seconds"],
            "inference_seconds": seconds,
            "batch_predict_ms_per_cv": 1000 * seconds / len(frame),
            "batch_size": batch_size,
            "device": chosen_device,
            "model_dir": str(best_dir),
        },
    )
    comparison = update_comparison(cfg["paths"]["report_dir"], [result], split)
    print(f"DistilBERT {split} Macro-F1={result['macro_f1']:.4f}, Accuracy={result['accuracy']:.4f}")
    print(f"Results: {comparison}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task_a.yaml")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    evaluate_bert(
        args.config, split=args.split, final_test=args.final_test,
        model_dir=args.model_dir, batch_size=args.batch_size, device=args.device,
    )


if __name__ == "__main__":
    main()
