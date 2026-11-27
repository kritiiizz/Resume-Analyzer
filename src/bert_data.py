"""Paper-exact 128-head + 382-tail WordPiece selection for Task A DistilBERT.

Sun et al. (2019), section 5.3.1: head+tail preserves first 128 and last
382 *content* tokens, with two separate [CLS]/[SEP] model special tokens.

This module has no torch/transformers imports; tokenizer is injected for
unit testing and to ensure training/evaluation/inference share the same code.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .preprocess import clean_text


def validate_strategy(strategy: dict[str, Any], tokenizer) -> None:
    if strategy.get("strategy") != "head_tail":
        raise ValueError("Only head_tail strategy is implemented")
    head = int(strategy["first_tokens"])
    tail = int(strategy["last_tokens"])
    maximum = int(strategy["max_length"])
    specials = len(tokenizer.build_inputs_with_special_tokens([]))
    if head <= 0 or tail <= 0 or head + tail + specials != maximum:
        raise ValueError(
            f"Invalid token budget: {head}+{tail}+{specials} != {maximum}"
        )
    if maximum != 512 or specials != 2:
        raise ValueError(
            "This Task A implementation requires 510 content + 2 special tokens = 512. "
            "Use a BERT/DistilBERT single-sequence tokenizer."
        )


def clean_for_bert(text: str, preprocessing: dict[str, Any]) -> str:
    if not preprocessing.get("normalize_unicode", True):
        raise ValueError("Unicode normalization is required for reproducibility")
    return clean_text(
        text,
        mask_pii=bool(preprocessing.get("mask_pii", True)),
        lowercase=bool(preprocessing.get("lowercase", True)),
        remove_title_line=bool(preprocessing.get("remove_title_line", False)),
        normalize_skill_synonyms=False,
    )


def encode_head_tail(
    text: str,
    tokenizer,
    *,
    strategy: dict[str, Any],
    preprocessing: dict[str, Any],
) -> tuple[dict[str, list[int]], dict[str, int | bool]]:
    """Encode once without specials, slice disjoint token spans, add specials.

    For <=510 tokens, preserve the entire text *once*. Never pad here;
    DataCollatorWithPadding pads dynamically per batch during train/eval.
    """
    validate_strategy(strategy, tokenizer)
    cleaned = clean_for_bert(text, preprocessing)
    encoded = tokenizer(cleaned, add_special_tokens=False, truncation=False)
    token_ids = encoded["input_ids"]
    if not isinstance(token_ids, list) or (token_ids and not isinstance(token_ids[0], int)):
        raise ValueError("Expected single unbatched list of token IDs")
    n = len(token_ids)
    limit = int(strategy["first_tokens"]) + int(strategy["last_tokens"])
    if n > limit:
        kept = token_ids[: int(strategy["first_tokens"])] + token_ids[-int(strategy["last_tokens"]) :]
    else:
        kept = token_ids
    ids = tokenizer.build_inputs_with_special_tokens(kept)
    if len(ids) > int(strategy["max_length"]):
        raise AssertionError("Token budget exceeded max length")
    return (
        {"input_ids": ids, "attention_mask": [1] * len(ids)},
        {
            "original_content_tokens": n,
            "kept_content_tokens": len(kept),
            "discarded_content_tokens": n - len(kept),
            "was_truncated": n > limit,
        },
    )


class TokenizedResumeDataset:
    """Minimal map-style dataset accepted by PyTorch DataLoader/Trainer.

    No raw text is retained in the resulting dataset. sample IDs are retained
    separately by the caller for joining predictions to the frozen split.
    """

    def __init__(self, examples: list[dict[str, Any]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


def prepare_dataset(
    frame,
    tokenizer,
    label2id: dict[str, int],
    *,
    text_column: str,
    label_column: str,
    strategy: dict[str, Any],
    preprocessing: dict[str, Any],
) -> tuple[TokenizedResumeDataset, dict[str, Any]]:
    validate_strategy(strategy, tokenizer)
    examples = []
    lengths = []
    dropped = []
    for text, label in zip(frame[text_column], frame[label_column]):
        if label not in label2id:
            raise ValueError(f"Unknown label: {label}")
        tokens, detail = encode_head_tail(
            str(text), tokenizer, strategy=strategy, preprocessing=preprocessing
        )
        tokens["labels"] = label2id[label]
        examples.append(tokens)
        lengths.append(detail["original_content_tokens"])
        dropped.append(detail["discarded_content_tokens"])
    return TokenizedResumeDataset(examples), length_report(lengths, dropped)


def length_report(lengths: list[int], dropped: list[int]) -> dict[str, Any]:
    if not lengths or len(lengths) != len(dropped):
        raise ValueError("Length audit requires equal-length nonempty lists")
    values = np.array(lengths)
    lost = np.array(dropped)
    return {
        "n_samples": int(len(values)),
        "content_tokens_mean": float(np.mean(values)),
        "content_tokens_min": int(np.min(values)),
        "content_tokens_p50": float(np.percentile(values, 50)),
        "content_tokens_p75": float(np.percentile(values, 75)),
        "content_tokens_p90": float(np.percentile(values, 90)),
        "content_tokens_p95": float(np.percentile(values, 95)),
        "content_tokens_max": int(np.max(values)),
        "n_truncated": int(np.count_nonzero(lost)),
        "fraction_truncated": float(np.mean(lost > 0)),
        "mean_tokens_discarded": float(np.mean(lost)),
        "total_tokens_discarded": int(np.sum(lost)),
    }


def assert_frozen_split(frame, manifest: dict, split: str) -> None:
    if split not in ("train", "val", "test"):
        raise ValueError("Unknown split")
    expected = set(manifest["split_sample_ids"][split])
    actual = frame["sample_id"].astype(str).tolist()
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(
            f"{split} does not match the frozen split_manifest.json (wrong dataset/version?)"
        )
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample IDs in {split}")


def split_fingerprint(manifest: dict) -> str:
    """Stable fingerprint of all three splits without reading test CV text."""
    canonical = "\n".join(
        split + ":" + ",".join(sorted(manifest["split_sample_ids"][split]))
        for split in ("train", "val", "test")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
