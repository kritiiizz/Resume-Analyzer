"""Join verified historical baselines with current DistilBERT evaluations.

Safeguard: historical baseline rows can only be combined with a run using the
same frozen raw dataset SHA-256 and train/val/test split fingerprint.
This command only combines *already evaluated* runs; it does not tune models.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .bert_data import split_fingerprint
from .common import load_config, read_json


def combine_comparisons(report_root: str | Path, metadata_dir: str | Path,
                        split: str = "val") -> Path:
    if split not in ("val", "test"):
        raise ValueError("Only val or test comparisons are supported")
    root = Path(report_root)
    manifest = read_json(Path(metadata_dir) / "split_manifest.json")
    historic = read_json(root / "published" / "baseline_provenance.json")
    if manifest["raw_file_sha256"] != historic["raw_file_sha256"]:
        raise ValueError("Historical baseline dataset SHA-256 differs")
    if split_fingerprint(manifest) != historic["split_fingerprint"]:
        raise ValueError("Historical baseline split fingerprint differs")

    published_path = root / "published" / f"comparison_{split}.csv"
    old = pd.read_csv(published_path)
    old["result_origin"] = "published_classical_baseline"
    local_path = root / f"comparison_{split}.csv"
    tables = [old]
    if local_path.is_file():
        new = pd.read_csv(local_path)
        new["result_origin"] = "current_run"
        tables.append(new)
    combined = pd.concat(tables, ignore_index=True)
    combined = combined.drop_duplicates(subset=["run_id", "split"], keep="last")
    combined = combined.loc[combined["split"] == split].copy()
    if combined.empty:
        raise ValueError("No matching rows")
    combined = combined.sort_values(["macro_f1", "model_name"], ascending=[False, True])
    output = root / f"combined_comparison_{split}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(combined) + 2)))
    labels = [f"{n} [{o}]" for n, o in zip(combined["model_name"], combined["result_origin"])]
    ax.barh(labels[::-1], combined["macro_f1"].to_numpy()[::-1])
    ax.set(xlim=(0, 1), xlabel="Macro-F1",
           title=f"Task A verified model comparison — {split}")
    fig.tight_layout()
    fig.savefig(root / f"combined_comparison_{split}.png", dpi=170)
    plt.close(fig)
    print(f"Combined {len(combined)} runs: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task_a.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    cfg = load_config(args.config)
    combine_comparisons(cfg["paths"]["report_dir"], cfg["paths"]["metadata_dir"], args.split)


if __name__ == "__main__":
    main()
