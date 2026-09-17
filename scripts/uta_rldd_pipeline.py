#!/usr/bin/env python3
"""Selective, storage-aware UTA-RLDD acquisition.

Storage strategy (see docs/Master_Plan_Status.md): this Kaggle mirror
(rishab260/uta-reallife-drowsiness-dataset) is NOT a monolithic archive on
Kaggle's side - `kaggle datasets files` lists 145 individually-addressable
video files across 48 subjects (verified via the Kaggle API, not assumed),
totalling ~96.6GB. `kaggle datasets download -d <ref>` (no -f) bundles
everything into one big zip client-side, which is what produced the ~89GB
partial download this project deliberately abandoned to protect local disk
space. Downloading by `-f <path>` instead fetches one raw video file
directly, with no zip wrapping - this lets us acquire a small, named subset
of subjects locally (for pipeline development/smoke-testing) while the full
48-subject set is intended to be acquired inside Google Colab for the actual
reported drowsiness experiments, never accumulating permanently on this
laptop.

Usage:
    python scripts/uta_rldd_pipeline.py index
    python scripts/uta_rldd_pipeline.py plan --n-subjects 6
    python scripts/uta_rldd_pipeline.py fetch --subjects 45 17 31 16 27 44
    python scripts/uta_rldd_pipeline.py validate
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "drowsiness"
INDEX_CACHE = RAW_DIR / "_file_index.json"
DATASET_REF = "rishab260/uta-reallife-drowsiness-dataset"

FILE_RE = re.compile(r"(Fold\d+_part\d+)/\1/(\d+)/(.+)")


def fetch_index(force: bool = False) -> List[dict]:
    """Fetch (and cache) the real per-file listing from Kaggle's API.

    Cached under data/raw/ (gitignored) - this is metadata about the
    dataset, not the dataset itself, but it's still treated as raw-data
    territory to keep the repo boundary simple.
    """
    if INDEX_CACHE.exists() and not force:
        return json.loads(INDEX_CACHE.read_text())

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "kaggle",
            "datasets",
            "files",
            "-d",
            DATASET_REF,
            "--page-size",
            "200",
            "--format",
            "json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    INDEX_CACHE.write_text(json.dumps(data, indent=2))
    return data


def group_by_subject(index: List[dict]) -> Dict[str, List[dict]]:
    by_subject: Dict[str, List[dict]] = {}
    unmatched = []
    for f in index:
        m = FILE_RE.match(f["name"])
        if not m:
            unmatched.append(f["name"])
            continue
        by_subject.setdefault(m.group(2), []).append(f)
    if unmatched:
        print(f"WARNING: {len(unmatched)} files did not match the expected path pattern: {unmatched[:5]}")
    return by_subject


def cmd_index(args: argparse.Namespace) -> None:
    index = fetch_index(force=args.force)
    by_subject = group_by_subject(index)
    total_gb = sum(f["size"] for f in index) / 1e9
    print(f"Indexed {len(index)} files across {len(by_subject)} subjects, {total_gb:.1f} GB total.")
    print(f"Cached at {INDEX_CACHE}")


def cmd_plan(args: argparse.Namespace) -> None:
    index = fetch_index(force=False)
    by_subject = group_by_subject(index)
    sizes = {s: sum(f["size"] for f in files) for s, files in by_subject.items()}
    chosen = sorted(sizes, key=lambda s: sizes[s])[: args.n_subjects]
    total = sum(sizes[s] for s in chosen)

    free_bytes = shutil.disk_usage(REPO_ROOT).free
    print(f"Smallest {args.n_subjects} subjects by size: {chosen}")
    print(f"Planned download size: {total / 1e9:.2f} GB")
    print(f"Free disk space available: {free_bytes / 1e9:.1f} GB")
    if total > 0.5 * free_bytes:
        print("WARNING: planned download exceeds 50% of free disk space - reconsider n-subjects.")
    print(
        "\nThis is a LOCAL SMOKE/PIPELINE-VERIFICATION sample, not the full "
        "experiment - the real drowsiness experiments should acquire a "
        "larger subject set (ideally all 48 available on this mirror) inside "
        "Google Colab, per the storage strategy in docs/Master_Plan_Status.md."
    )
    print(f"\nSuggested command: python scripts/uta_rldd_pipeline.py fetch --subjects {' '.join(chosen)}")


def cmd_fetch(args: argparse.Namespace) -> None:
    index = fetch_index(force=False)
    by_subject = group_by_subject(index)

    missing = [s for s in args.subjects if s not in by_subject]
    if missing:
        raise SystemExit(f"Unknown subject IDs (not in index): {missing}. Available: {sorted(by_subject)}")

    files_to_get = [f for s in args.subjects for f in by_subject[s]]
    total_bytes = sum(f["size"] for f in files_to_get)
    free_bytes = shutil.disk_usage(REPO_ROOT).free
    print(f"Subjects: {args.subjects}")
    print(f"Files to download: {len(files_to_get)}, total {total_bytes / 1e9:.2f} GB")
    print(f"Free disk space: {free_bytes / 1e9:.1f} GB")
    if total_bytes > free_bytes - 2e9:  # keep a 2GB safety margin
        raise SystemExit("Not enough free disk space for this selection (with a 2GB safety margin). Aborting.")

    for f in files_to_get:
        m = FILE_RE.match(f["name"])
        subject, fname = m.group(2), m.group(3)
        dest_dir = RAW_DIR / subject
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / fname
        if dest_path.exists() and dest_path.stat().st_size == f["size"]:
            print(f"SKIP (already present, size matches): {dest_path}")
            continue
        print(f"Downloading {f['name']} -> {dest_path} ({f['size'] / 1e9:.2f} GB)")
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                DATASET_REF,
                "-f",
                f["name"],
                "-p",
                str(dest_dir),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        # Kaggle's `-f` single-file download still wraps the file in a zip
        # named "<basename>.zip" (verified empirically - not documented
        # behavior we assumed in advance). Unzip it to the real filename,
        # then remove the wrapper.
        zip_path = dest_dir / f"{Path(f['name']).name}.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
                if len(names) != 1:
                    raise RuntimeError(f"Expected exactly one file inside {zip_path}, found {names}")
                zf.extract(names[0], path=dest_dir)
                extracted = dest_dir / names[0]
                if extracted != dest_path:
                    extracted.rename(dest_path)
            zip_path.unlink()
        elif (dest_dir / Path(f["name"]).name).exists():
            downloaded = dest_dir / Path(f["name"]).name
            if downloaded != dest_path:
                downloaded.rename(dest_path)

        actual_size = dest_path.stat().st_size if dest_path.exists() else -1
        if actual_size != f["size"]:
            print(f"WARNING: size mismatch for {dest_path}: expected {f['size']}, got {actual_size}")
        else:
            print(f"OK: {dest_path} ({actual_size / 1e9:.2f} GB, size matches index)")


def cmd_validate(args: argparse.Namespace) -> None:
    import cv2

    video_paths = sorted(RAW_DIR.glob("*/*.mov")) + sorted(RAW_DIR.glob("*/*.MOV")) + sorted(
        RAW_DIR.glob("*/*.mp4")
    )
    if not video_paths:
        print(f"No downloaded videos found under {RAW_DIR}/<subject>/. Run `fetch` first.")
        return

    report = []
    for p in video_paths:
        cap = cv2.VideoCapture(str(p))
        opened = cap.isOpened()
        info = {"path": str(p), "opened": opened}
        if opened:
            info["fps"] = cap.get(cv2.CAP_PROP_FPS)
            info["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            info["duration_sec"] = (
                info["frame_count"] / info["fps"] if info["fps"] else None
            )
        cap.release()
        report.append(info)
        status = "OK" if opened else "FAILED TO OPEN"
        print(f"{status}: {p} -> {info}")

    n_failed = sum(1 for r in report if not r["opened"])
    print(f"\n{len(report)} videos checked, {n_failed} failed to open.")
    out_path = RAW_DIR / "_local_sample_validation.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote validation report to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Fetch and cache the real per-file listing from Kaggle.")
    p_index.add_argument("--force", action="store_true")
    p_index.set_defaults(func=cmd_index)

    p_plan = sub.add_parser("plan", help="Pick the smallest N subjects and report planned size.")
    p_plan.add_argument("--n-subjects", type=int, default=6, dest="n_subjects")
    p_plan.set_defaults(func=cmd_plan)

    p_fetch = sub.add_parser("fetch", help="Download specific subjects' videos (not the whole archive).")
    p_fetch.add_argument("--subjects", nargs="+", required=True)
    p_fetch.set_defaults(func=cmd_fetch)

    p_validate = sub.add_parser("validate", help="Open every locally-downloaded video and report real metadata.")
    p_validate.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
