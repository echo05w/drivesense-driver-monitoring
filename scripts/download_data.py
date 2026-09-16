#!/usr/bin/env python3
"""Download DriveSense source datasets via the Kaggle API.

Requires a Kaggle API token at ~/.kaggle/kaggle.json (or KAGGLE_USERNAME /
KAGGLE_KEY env vars) — works identically locally or in a Google Colab
runtime. See docs/Dataset_Research.md for dataset details and why
acquisition is designed to be Colab-first.

Usage:
    python scripts/download_data.py --dataset distraction
    python scripts/download_data.py --dataset drowsiness
    python scripts/download_data.py --dataset all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"

DATASETS = {
    "distraction": {
        "kind": "competition",
        "ref": "state-farm-distracted-driver-detection",
        "dest": RAW_DIR / "distraction",
    },
    "drowsiness": {
        "kind": "dataset",
        # Third-party Kaggle mirror of UTA-RLDD — verify current size/contents
        # before relying on it; see docs/Dataset_Research.md for the official
        # source as a fallback if this mirror is unavailable or changes.
        "ref": "rishab260/uta-reallife-drowsiness-dataset",
        "dest": RAW_DIR / "drowsiness",
    },
}


def _check_kaggle_available() -> None:
    try:
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            "The `kaggle` CLI is not available or not authenticated.\n"
            "Install it with `pip install kaggle` and place your API token at "
            "~/.kaggle/kaggle.json (see https://www.kaggle.com/docs/api).\n"
            "This is expected to be run either locally (once configured) or "
            "inside a Colab notebook using your own Kaggle credentials."
        ) from exc


def download(name: str) -> None:
    spec = DATASETS[name]
    dest: Path = spec["dest"]
    dest.mkdir(parents=True, exist_ok=True)

    if spec["kind"] == "competition":
        cmd = ["kaggle", "competitions", "download", "-c", spec["ref"], "-p", str(dest)]
    else:
        cmd = ["kaggle", "datasets", "download", "-d", spec["ref"], "-p", str(dest), "--unzip"]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Downloaded '{name}' to {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        required=True,
        help="Which dataset to download.",
    )
    args = parser.parse_args()

    _check_kaggle_available()

    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for name in targets:
        download(name)


if __name__ == "__main__":
    sys.exit(main())
