"""Fine-tune DistilBERT for Task A using Sun et al. (2019) head+tail (128+382).

By default this command loads TRAIN and VALIDATION only. With --epoch-test,
it also reports TEST metrics at each epoch; checkpoint selection remains based
only on validation Macro-F1.
Full fine-tuning runs on CUDA, MPS (Mac M3), or CPU. Install:
    pip install -r requirements-bert.txt

Do not change the historical frozen split or select a checkpoint on test.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .bert_data import assert_frozen_split, prepare_dataset, split_fingerprint
from .bert_metrics import compute_bert_metrics, softmax_probabilities
from .common import dataset_fingerprint, load_config, read_json, runtime_info, utc_now, write_json
from .evaluation import export_evaluation, update_comparison


def make_run_id() -> str:
    return "distilbert_head_tail_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _load_frozen_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    root = Path(cfg["paths"]["split_dir"])
    train = pd.read_csv(root / "task_a_train.csv", dtype={"sample_id": str}).fillna("")
    val = pd.read_csv(root / "task_a_val.csv", dtype={"sample_id": str}).fillna("")
    meta = Path(cfg["paths"]["metadata_dir"])
    manifest = read_json(meta / "split_manifest.json")
    label2id = read_json(meta / "label2id.json")
    assert_frozen_split(train, manifest, "train")
    assert_frozen_split(val, manifest, "val")
    labels = sorted(label2id, key=label2id.get)
    if [label2id[name] for name in labels] != list(range(len(labels))):
        raise ValueError("Label IDs must be contiguous 0..num_labels-1")
    label_col = cfg["dataset"]["label_column"]
    if set(train[label_col]) != set(labels):
        raise ValueError("Train classes differ from canonical label mapping")
    if not set(val[label_col]).issubset(set(labels)):
        raise ValueError("Validation has unknown classes")
    return train, val, manifest, label2id


def training_arguments(config: dict, output_dir: str, seed: int,
                       *, max_steps: int | None = None):
    from transformers import TrainingArguments

    hp = config["training"]
    kwargs = dict(
        output_dir=output_dir,
        overwrite_output_dir=False,
        num_train_epochs=float(hp["epochs"]),
        per_device_train_batch_size=int(hp["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(hp["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(hp["gradient_accumulation_steps"]),
        learning_rate=float(hp["learning_rate"]),
        weight_decay=float(hp["weight_decay"]),
        warmup_ratio=float(hp["warmup_ratio"]),
        lr_scheduler_type="linear",
        optim="adamw_torch",
        max_grad_norm=float(hp.get("max_grad_norm", 1.0)),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=int(hp["save_total_limit"]),
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=int(hp.get("logging_steps", 20)),
        dataloader_num_workers=int(hp.get("dataloader_num_workers", 0)),
        report_to=[],
        remove_unused_columns=False,
        save_safetensors=True,
        fp16=hp["precision"] == "fp16",
        bf16=hp["precision"] == "bf16",
        gradient_checkpointing=bool(hp.get("gradient_checkpointing", False)),
        seed=seed,
        data_seed=seed,
    )
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("--max-steps must be >0")
        kwargs["max_steps"] = max_steps
    return TrainingArguments(**kwargs)


def run_training(
    config_path: str = "configs/task_a.yaml",
    bert_config_path: str = "configs/task_a_bert.yaml",
    *,
    run_id: str | None = None,
    resume_checkpoint: str | None = None,
    max_steps: int | None = None,
    train_limit: int | None = None,
    val_limit: int | None = None,
    token_audit_only: bool = False,
    batch_size: int | None = None,
    gradient_accumulation: int | None = None,
    precision: str | None = None,
    epoch_test: bool = False,
) -> dict:
    cfg = load_config(config_path)
    bert = copy.deepcopy(load_config(bert_config_path))
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("--batch-size must be >0")
        bert["training"]["per_device_train_batch_size"] = batch_size
    if gradient_accumulation is not None:
        if gradient_accumulation <= 0:
            raise ValueError("--gradient-accumulation must be >0")
        bert["training"]["gradient_accumulation_steps"] = gradient_accumulation
    if precision is not None:
        bert["training"]["precision"] = precision
    if bert["training"]["precision"] not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32, fp16, or bf16")

    if bert["truncation"] != {
        "strategy": "head_tail", "first_tokens": 128,
        "last_tokens": 382, "max_length": 512,
    }:
        raise ValueError("This project freezes Sun et al. head+tail at 128+382+2=512")

    train_frame, val_frame, manifest, label2id = _load_frozen_data(cfg)
    if (train_limit is not None or val_limit is not None) and not (max_steps or token_audit_only):
        raise ValueError("Data limits are for smoke/debug only; also pass --max-steps")
    if train_limit is not None:
        if train_limit <= 0:
            raise ValueError("--train-limit must be >0")
        train_frame = train_frame.head(train_limit)
    if val_limit is not None:
        if val_limit <= 0:
            raise ValueError("--val-limit must be >0")
        val_frame = val_frame.head(val_limit)
    is_smoke = bool(max_steps or train_limit or val_limit)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        bert["model_name"], revision=bert.get("model_revision", "main"), use_fast=True
    )
    train_data, train_stats = prepare_dataset(
        train_frame, tokenizer, label2id, text_column=cfg["dataset"]["text_column"],
        label_column=cfg["dataset"]["label_column"],
        strategy=bert["truncation"], preprocessing=bert["preprocessing"],
    )
    val_data, val_stats = prepare_dataset(
        val_frame, tokenizer, label2id, text_column=cfg["dataset"]["text_column"],
        label_column=cfg["dataset"]["label_column"],
        strategy=bert["truncation"], preprocessing=bert["preprocessing"],
    )
    test_frame = None
    test_data = None
    if epoch_test:
        test_frame = pd.read_csv(
            Path(cfg["paths"]["split_dir"]) / "task_a_test.csv",
            dtype={"sample_id": str},
        ).fillna("")
        assert_frozen_split(test_frame, manifest, "test")
        test_data, _ = prepare_dataset(
            test_frame, tokenizer, label2id,
            text_column=cfg["dataset"]["text_column"],
            label_column=cfg["dataset"]["label_column"],
            strategy=bert["truncation"], preprocessing=bert["preprocessing"],
        )
    audit = {
        "method": "Sun2019_head128_tail382",
        "tokenizer_name": bert["model_name"],
        "train": train_stats,
        "val": val_stats,
        "split_fingerprint": split_fingerprint(manifest),
        "is_smoke": is_smoke,
    }
    report_root = Path(cfg["paths"]["report_dir"])
    if token_audit_only:
        path = report_root / "bert_head_tail_token_audit.json"
        write_json(path, audit)
        print(json.dumps(audit, indent=2))
        return audit

    from transformers import (
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        set_seed,
    )
    import torch
    import transformers

    seed = int(bert["training"]["seed"])
    set_seed(seed)
    if run_id is None:
        run_id = make_run_id()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id must contain only letters, numbers, _ and -")
    model_root = Path(cfg["paths"]["model_dir"]) / "distilbert_head_tail"
    run_dir = model_root / "runs" / run_id
    if run_dir.exists() and not resume_checkpoint:
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "dataset_sha256": dataset_fingerprint(manifest),
        "split_fingerprint": split_fingerprint(manifest),
        "label2id": label2id,
        "bert_config": bert,
        "is_smoke": is_smoke,
        "epoch_test_diagnostics": epoch_test,
    }
    contract_file = run_dir / "run_contract.json"
    if resume_checkpoint:
        if not contract_file.exists() or read_json(contract_file) != contract:
            raise ValueError("Resume settings, labels or split differ from frozen run contract")
    else:
        write_json(contract_file, contract)
    checkpoint_dir = run_dir / "checkpoints"
    best_dir = run_dir / "best"

    labels = sorted(label2id, key=label2id.get)
    model = AutoModelForSequenceClassification.from_pretrained(
        bert["model_name"],
        revision=bert.get("model_revision", "main"),
        num_labels=len(labels),
        label2id=label2id,
        id2label={index: label for label, index in label2id.items()},
        ignore_mismatched_sizes=True,
    )
    args = training_arguments(bert, str(checkpoint_dir), seed, max_steps=max_steps)
    if epoch_test:
        args.logging_strategy = "epoch"
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=int(bert["training"].get("pad_to_multiple_of", 64)),
    )
    callbacks = []
    if int(bert["training"].get("early_stopping_patience", 0)) > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=int(bert["training"]["early_stopping_patience"])
        ))
    if epoch_test:
        class EpochTestTrainer(Trainer):
            """Report test metrics each epoch without using them for selection."""

            def evaluate(self, eval_dataset=None, ignore_keys=None,
                         metric_key_prefix="eval"):
                metrics = super().evaluate(
                    eval_dataset=eval_dataset, ignore_keys=ignore_keys,
                    metric_key_prefix=metric_key_prefix,
                )
                if eval_dataset is None and metric_key_prefix == "eval":
                    callbacks = self.callback_handler.callbacks
                    self.callback_handler.callbacks = [
                        callback for callback in callbacks
                        if not isinstance(callback, EarlyStoppingCallback)
                    ]
                    try:
                        super().evaluate(
                            eval_dataset=test_data, ignore_keys=ignore_keys,
                            metric_key_prefix="test",
                        )
                    finally:
                        self.callback_handler.callbacks = callbacks
                return metrics

        trainer_type = EpochTestTrainer
    else:
        trainer_type = Trainer
    trainer = trainer_type(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda output: compute_bert_metrics(output, len(labels)),
        callbacks=callbacks,
    )
    print(f"Device: {args.device}; train CVs: {len(train_data)}; val CVs: {len(val_data)}")
    started = time.perf_counter()
    train_out = trainer.train(resume_from_checkpoint=resume_checkpoint)
    train_seconds = time.perf_counter() - started
    if trainer.state.best_model_checkpoint is None:
        raise RuntimeError("No best validation checkpoint was selected")

    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    prediction = trainer.predict(val_data)
    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    probabilities = softmax_probabilities(logits)
    pred_labels = [labels[i] for i in np.argmax(probabilities, axis=1)]
    selection_metrics = compute_bert_metrics(prediction, len(labels))
    best_events = [h for h in trainer.state.log_history if "eval_macro_f1" in h]
    best_event = max(best_events, key=lambda h: h["eval_macro_f1"]) if best_events else {}
    full_manifest = {
        "run_id": run_id,
        "model_name": "distilbert_head_tail",
        "base_model": bert["model_name"],
        "base_model_revision": bert.get("model_revision", "main"),
        "task": "task_a_resume_category",
        "created_utc": utc_now(),
        "dataset_source": cfg["dataset"]["source"],
        "dataset_sha256": dataset_fingerprint(manifest),
        "split_fingerprint": split_fingerprint(manifest),
        "train_count": len(train_data),
        "val_count": len(val_data),
        "is_smoke": is_smoke,
        "epoch_test_diagnostics": epoch_test,
        "label2id": label2id,
        "preprocessing": bert["preprocessing"],
        "truncation": bert["truncation"],
        "training": bert["training"],
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_epoch": best_event.get("epoch"),
        "completed_epoch": trainer.state.epoch,
        "best_validation_macro_f1": trainer.state.best_metric,
        "selected_model_val_metrics": selection_metrics,
        "training_seconds": train_seconds,
        "training_output": train_out.metrics,
        "token_audit": audit,
        "hardware": {
            "device": str(args.device),
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available()
                else ("Apple MPS" if torch.backends.mps.is_available() else "CPU")
            ),
            "precision": bert["training"]["precision"],
        },
        "runtime": {
            **runtime_info(), "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "git_commit": _git_commit(),
        "model_path": str(best_dir),
        "notes": "This is not Sun et al. BERT-base; it is DistilBERT with their truncation method.",
    }
    write_json(best_dir / "metadata.json", full_manifest)
    write_json(run_dir / "training_history.json", trainer.state.log_history)
    if epoch_test:
        epoch_rows = {}
        for event in trainer.state.log_history:
            epoch = event.get("epoch")
            if epoch is None or int(epoch) != epoch:
                continue
            row = epoch_rows.setdefault(int(epoch), {"epoch": int(epoch), "seed": seed})
            for key, value in event.items():
                if key in ("loss", "grad_norm", "learning_rate", "eval_loss",
                           "eval_macro_f1", "eval_macro_precision", "eval_macro_recall",
                           "eval_accuracy", "eval_weighted_f1", "test_loss",
                           "test_macro_f1", "test_macro_precision", "test_macro_recall",
                           "test_accuracy", "test_weighted_f1"):
                    row["train_loss" if key == "loss" else key] = value
        pd.DataFrame([epoch_rows[k] for k in sorted(epoch_rows)]).to_csv(
            run_dir / "epoch_metrics.csv", index=False
        )
    write_json(run_dir / "token_audit.json", audit)
    write_json(run_dir / "resolved_training_config.json", bert)
    summary = export_evaluation(
        val_frame, labels=labels, predictions=pred_labels, scores=probabilities,
        score_type="uncalibrated_softmax_probability",
        model="distilbert_head_tail", run_id=run_id, split="val",
        report_root=report_root,
        metadata={
            "dataset_sha256": dataset_fingerprint(manifest),
            "split_fingerprint": split_fingerprint(manifest),
            "training_seconds": train_seconds,
            "is_smoke": is_smoke,
        },
    )
    if not is_smoke:
        update_comparison(report_root, [summary], "val")
        write_json(model_root / "latest.json", {
            "run_id": run_id, "best_relative_path": f"runs/{run_id}/best",
            "split_fingerprint": split_fingerprint(manifest),
        })
    print(f"Best checkpoint exported to {best_dir}")
    print(f"Validation Macro-F1: {summary['macro_f1']:.4f}")
    if is_smoke:
        print("SMOKE RUN: no latest pointer updated and no official comparison row added.")
    return full_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task_a.yaml")
    parser.add_argument("--bert-config", default="configs/task_a_bert.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--token-audit-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation", type=int, default=None)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default=None)
    parser.add_argument("--epoch-test", action="store_true",
                        help="Log test metrics after each validation epoch; never select on test")
    arguments = parser.parse_args()
    run_training(
        arguments.config, arguments.bert_config, run_id=arguments.run_id,
        resume_checkpoint=arguments.resume_checkpoint, max_steps=arguments.max_steps,
        train_limit=arguments.train_limit, val_limit=arguments.val_limit,
        token_audit_only=arguments.token_audit_only,
        batch_size=arguments.batch_size,
        gradient_accumulation=arguments.gradient_accumulation,
        precision=arguments.precision, epoch_test=arguments.epoch_test,
    )


if __name__ == "__main__":
    main()
