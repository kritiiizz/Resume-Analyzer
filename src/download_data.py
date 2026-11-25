"""Download the Kaggle CSV; never commit resumes to source control."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .common import sha256_file, utc_now, write_json


def download_csv(target: str | Path, local_source: str | Path | None = None) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if local_source:
        source = Path(local_source)
    else:
        try:
            import kagglehub
        except ImportError as exc:
            raise RuntimeError(
                "Install requirements-download.txt, or pass --from-path /path/to/Resume.csv"
            ) from exc
        root = Path(kagglehub.dataset_download("snehaanbhawal/resume-dataset"))
        matches = list(root.rglob("Resume.csv"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one Resume.csv in {root}; found {len(matches)}")
        source = matches[0]
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.resolve() != target.resolve():
        shutil.copyfile(source, target)
    write_json(
        target.parent.parent / "metadata" / "download_manifest.json",
        {
            "dataset": "snehaanbhawal/resume-dataset",
            "file": target.name,
            "sha256": sha256_file(target),
            "downloaded_at_utc": utc_now(),
            "source_kind": "local" if local_source else "kagglehub",
        },
    )
    print(f"CSV ready: {target} (SHA256: {sha256_file(target)})")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="data/raw/Resume.csv")
    parser.add_argument("--from-path", default=None, help="Use a CSV already downloaded")
    args = parser.parse_args()
    download_csv(args.target, args.from_path)


if __name__ == "__main__":
    main()
