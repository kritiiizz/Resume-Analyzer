"""Validate the team's no-title split and create isolated experiment configs.

Usage: python -m src.prepare_no_title
The original split, metadata, configs, models, and reports are untouched.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pandas as pd
import yaml

from .common import read_json, sha256_file, utc_now, write_json

SPLITS = ("train", "val", "test")
SEEDS = (42, 123, 2026)


def main() -> None:
    source = Path("splits_for_team/splits_no_title")
    aligned = Path("splits_for_team/splits_aligned")
    destination = Path("data/splits_no_title")
    metadata = Path("data/metadata_no_title")
    old_manifest = read_json("data/metadata/split_manifest.json")
    label2id = read_json("data/metadata/label2id.json")
    old_config = yaml.safe_load(Path("configs/task_a.yaml").read_text())
    old_bert = yaml.safe_load(Path("configs/task_a_bert.yaml").read_text())
    destination.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256()
    file_hashes = {}
    for split in SPLITS:
        name = f"task_a_{split}.csv"
        frame = pd.read_csv(source / name, dtype={"sample_id": str}).fillna("")
        original = pd.read_csv(
            Path(old_config["paths"]["split_dir"]) / name,
            dtype={"sample_id": str},
        ).fillna("")
        team = pd.read_csv(aligned / f"{split}.csv", dtype={"sample_id": str}).fillna("")
        if list(frame.columns) != ["sample_id", "Resume_str", "Category", "text_hash"]:
            raise ValueError(f"Unexpected columns in {name}")
        if frame.sample_id.duplicated().any() or not frame.Resume_str.astype(bool).all():
            raise ValueError(f"Duplicate IDs or empty text in {name}")
        expected = set(old_manifest["split_sample_ids"][split])
        if set(frame.sample_id) != expected or len(frame) != len(expected):
            raise ValueError(f"Split IDs changed in {name}")
        by_id = frame.set_index("sample_id")
        old_by_id = original.set_index("sample_id")
        team_by_id = team.set_index("sample_id")
        if set(team_by_id.index) != expected:
            raise ValueError(f"Aligned IDs changed in {split}")
        if not by_id.Category.sort_index().equals(old_by_id.Category.sort_index()):
            raise ValueError(f"Labels changed in {name}")
        if not by_id.Category.sort_index().equals(team_by_id.Category.sort_index()):
            raise ValueError(f"Aligned labels changed in {split}")
        if not by_id.Resume_str.sort_index().equals(team_by_id.text_no_header.sort_index()):
            raise ValueError(f"No-title text differs from aligned text in {split}")
        if not by_id.text_hash.sort_index().equals(old_by_id.text_hash.sort_index()):
            raise ValueError(f"Original text_hash changed in {name}")
        if set(frame.Category) - set(label2id):
            raise ValueError(f"Unknown label in {name}")

        path = source / name
        file_hashes[split] = sha256_file(path)
        digest.update(name.encode("utf-8") + b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        shutil.copy2(path, destination / name)

    manifest = dict(old_manifest)
    manifest.update({
        "dataset_source": "splits_for_team/splits_no_title",
        "input_data_sha256": digest.hexdigest(),
        "input_file_sha256": file_hashes,
        "source_raw_file_sha256": old_manifest["raw_file_sha256"],
        "text_hash_semantics": "hash of original Resume_str, carried over from original split",
        "no_title_verified_against": "splits_for_team/splits_aligned/text_no_header",
        "prepared_at_utc": utc_now(),
    })
    write_json(metadata / "split_manifest.json", manifest)
    write_json(metadata / "label2id.json", label2id)

    base = dict(old_config)
    base["paths"] = {
        **base["paths"],
        "split_dir": str(destination),
        "metadata_dir": str(metadata),
        "model_dir": "models/task_a_no_title",
        "report_dir": "reports/task_a_no_title",
    }
    Path("configs/task_a_no_title.yaml").write_text(
        yaml.safe_dump(base, sort_keys=False), encoding="utf-8"
    )
    for seed in SEEDS:
        config = yaml.safe_load(yaml.safe_dump(base))
        config["split"]["seed"] = seed
        config["paths"]["model_dir"] += f"/seed_{seed}"
        config["paths"]["report_dir"] += f"/seed_{seed}"
        Path(f"configs/task_a_no_title_seed_{seed}.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        bert = yaml.safe_load(yaml.safe_dump(old_bert))
        bert["training"]["seed"] = seed
        Path(f"configs/task_a_bert_no_title_seed_{seed}.yaml").write_text(
            yaml.safe_dump(bert, sort_keys=False), encoding="utf-8"
        )
    print(f"Prepared {destination}; input_data_sha256={digest.hexdigest()}")


if __name__ == "__main__":
    main()
