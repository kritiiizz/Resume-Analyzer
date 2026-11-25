"""Train Task A classical models, tune ONLY on validation, save fitted pipelines.

Train/test isolation: this command never reads the test CSV.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .common import dataset_fingerprint, load_config, read_json, runtime_info, utc_now, write_json
from .preprocess import ResumeCleaner, SKILL_TOKEN_PATTERN

SUPPORTED_MODELS = ("dummy", "bow_nb", "tfidf_lr", "tfidf_svm")


def build_pipeline(model_name: str, cfg: dict, *, c: float = 1.0,
                   class_weight: str | None = None, alpha: float = 1.0) -> Pipeline:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    pre = cfg["preprocessing"]
    cleaner = ResumeCleaner(
        mask_pii=bool(pre.get("mask_pii", True)),
        lowercase=bool(pre.get("lowercase", True)),
        normalize_skill_synonyms=bool(pre.get("normalize_skill_synonyms", False)),
        remove_title_line=bool(pre.get("remove_title_line", False)),
        lemmatize=bool(pre.get("lemmatize", False)),
    )
    features = cfg["features"]
    vectorizer_args = {
        "ngram_range": tuple(features["ngram_range"]),
        "min_df": features["min_df"],
        "max_df": features["max_df"],
        "token_pattern": SKILL_TOKEN_PATTERN,
        "lowercase": False,
        "stop_words": "english" if pre.get("remove_stopwords", False) else None,
    }
    if model_name == "dummy":
        return Pipeline([
            ("clean", cleaner),
            ("vectorizer", CountVectorizer(**vectorizer_args)),
            ("classifier", DummyClassifier(strategy="most_frequent")),
        ])
    if model_name == "bow_nb":
        vectorizer = CountVectorizer(**vectorizer_args)
        classifier = MultinomialNB(alpha=alpha)
    else:
        vectorizer = TfidfVectorizer(
            **vectorizer_args, sublinear_tf=bool(features["sublinear_tf"])
        )
        if model_name == "tfidf_lr":
            classifier = LogisticRegression(
                C=c, class_weight=class_weight, max_iter=2000, random_state=cfg["split"]["seed"]
            )
        else:
            classifier = LinearSVC(
                C=c, class_weight=class_weight, random_state=cfg["split"]["seed"]
            )
    return Pipeline([
        ("clean", cleaner), ("vectorizer", vectorizer), ("classifier", classifier)
    ])


def candidate_grid(model_name: str, cfg: dict) -> list[dict]:
    if model_name == "dummy":
        return [{}]
    if model_name == "bow_nb":
        return [{"alpha": float(a)} for a in cfg["training"]["nb_alpha_grid"]]
    return [
        {"c": float(c), "class_weight": weight}
        for c in cfg["training"]["c_grid"]
        for weight in cfg["training"]["class_weights"]
    ]


def train_model(name: str, cfg: dict, train: pd.DataFrame, val: pd.DataFrame,
                label2id: dict[str, int], *, dataset_hash: str) -> dict:
    text_col = cfg["dataset"]["text_column"]
    label_col = cfg["dataset"]["label_column"]
    expected_labels = list(label2id)
    trials = []
    best_score = -1.0
    best_pipeline = None
    best_params = None
    train_seconds = 0.0
    for params in candidate_grid(name, cfg):
        model = build_pipeline(name, cfg, **params)
        started = time.perf_counter()
        model.fit(train[text_col].tolist(), train[label_col].tolist())
        elapsed = time.perf_counter() - started
        train_seconds += elapsed
        predictions = model.predict(val[text_col].tolist())
        macro_f1 = float(
            f1_score(val[label_col], predictions, labels=expected_labels,
                     average="macro", zero_division=0)
        )
        trials.append({
            "model": name,
            "params_json": json.dumps(params, sort_keys=True),
            "validation_macro_f1": macro_f1,
            "fit_seconds": elapsed,
        })
        if macro_f1 > best_score:
            best_score = macro_f1
            best_pipeline = model
            best_params = params

    assert best_pipeline is not None
    timestamp = utc_now().replace(":", "").replace("-", "").replace("+0000", "Z")
    run_id = f"{name}_{timestamp}"
    model_dir = Path(cfg["paths"]["model_dir"]) / name
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_dir / "pipeline.joblib", compress=3)
    metadata = {
        "run_id": run_id,
        "model": name,
        "task": cfg["task"],
        "trained_at_utc": utc_now(),
        "training_samples": int(len(train)),
        "validation_samples": int(len(val)),
        "dataset_sha256": dataset_hash,
        "best_params": best_params,
        "validation_macro_f1": best_score,
        "total_tuning_fit_seconds": train_seconds,
        "class_labels": expected_labels,
        "n_features": int(len(best_pipeline.named_steps["vectorizer"].vocabulary_)),
        "runtime": runtime_info(),
        "pipeline_path": str(model_dir / "pipeline.joblib"),
        "selection_rule": "maximum validation macro F1; first candidate wins ties",
        "test_data_used": False,
    }
    write_json(model_dir / "metadata.json", metadata)
    tune_dir = Path(cfg["paths"]["report_dir"]) / "tuning"
    tune_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trials).to_csv(tune_dir / f"{name}_candidates.csv", index=False)
    print(f"{name}: val Macro-F1={best_score:.4f}, best={best_params}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task_a.yaml")
    parser.add_argument(
        "--models", nargs="+", choices=SUPPORTED_MODELS,
        default=["dummy", "bow_nb", "tfidf_lr", "tfidf_svm"],
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    directory = Path(cfg["paths"]["split_dir"])
    train = pd.read_csv(directory / "task_a_train.csv").fillna("")
    val = pd.read_csv(directory / "task_a_val.csv").fillna("")
    label2id = read_json(Path(cfg["paths"]["metadata_dir"]) / "label2id.json")
    manifest = read_json(Path(cfg["paths"]["metadata_dir"]) / "split_manifest.json")
    if set(train[cfg["dataset"]["label_column"]]) != set(label2id):
        raise ValueError("Train labels do not match the frozen label mapping")
    summaries = []
    for name in args.models:
        summaries.append(train_model(
            name, cfg, train, val, label2id, dataset_hash=dataset_fingerprint(manifest)
        ))
    write_json(Path(cfg["paths"]["report_dir"]) / "training_summary.json", summaries)


if __name__ == "__main__":
    main()
