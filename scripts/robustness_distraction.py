#!/usr/bin/env python3
"""Robustness check for a trained distraction model: re-evaluate the same
untouched test set under a few realistic perturbations (brightness,
blur, noise, JPEG-quality loss) and report the metric drop versus the
clean baseline. Real re-evaluation, not a synthetic/invented estimate.

Usage:
    python scripts/robustness_distraction.py --run-id transfer_20260917_194003
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from drivesense.data.datasets import DistractionImageDataset  # noqa: E402
from drivesense.models.cnn import SimpleCNN, TransferLearningCNN  # noqa: E402

import pandas as pd  # noqa: E402

EXPERIMENTS_DIR = REPO_ROOT / "experiments" / "distraction"
SPLIT_CACHE = EXPERIMENTS_DIR / "split_assignment.csv"

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def darken(img: Image.Image, factor: float) -> Image.Image:
    arr = np.asarray(img).astype(np.float32) * factor
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def brighten(img: Image.Image, factor: float) -> Image.Image:
    arr = np.asarray(img).astype(np.float32) * factor
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def gaussian_blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    noise = np.random.default_rng(0).normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def jpeg_recompress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


PERTURBATIONS = {
    "clean": lambda img: img,
    "darker_50pct": lambda img: darken(img, 0.5),
    "brighter_50pct": lambda img: brighten(img, 1.5),
    "gaussian_blur_r2": lambda img: gaussian_blur(img, 2.0),
    "gaussian_noise_sigma25": lambda img: gaussian_noise(img, 25.0),
    "jpeg_quality_10": lambda img: jpeg_recompress(img, 10),
}


def build_model(name: str, num_classes: int) -> torch.nn.Module:
    if name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    if name == "transfer":
        return TransferLearningCNN(num_classes=num_classes)
    raise ValueError(name)


def make_transform(perturb_fn, image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Lambda(perturb_fn),
            transforms.ToTensor(),
            NORMALIZE,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    record = json.loads((EXPERIMENTS_DIR / f"{args.run_id}.json").read_text())
    checkpoint = torch.load(record["checkpoint_path"], map_location="cpu", weights_only=False)

    full = pd.read_csv(SPLIT_CACHE)
    test_df = full[full["split"] == "test"].reset_index(drop=True)
    if args.max_samples:
        test_df = test_df.iloc[: args.max_samples].reset_index(drop=True)

    model = build_model(checkpoint["model_name"], checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    results = {}
    for name, perturb_fn in PERTURBATIONS.items():
        tf = make_transform(perturb_fn, checkpoint["image_size"])
        ds = DistractionImageDataset(test_df, transform=tf)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in loader:
                outputs = model(images)
                all_preds.extend(outputs.argmax(dim=1).tolist())
                all_labels.extend(labels.tolist())

        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        results[name] = {"accuracy": acc, "macro_f1": macro_f1}
        print(f"{name}: accuracy={acc:.4f} macro_f1={macro_f1:.4f}")

    clean_acc = results["clean"]["accuracy"]
    clean_f1 = results["clean"]["macro_f1"]
    print("\nDrop from clean baseline:")
    for name, r in results.items():
        if name == "clean":
            continue
        print(
            f"  {name}: accuracy {clean_acc:.4f} -> {r['accuracy']:.4f} "
            f"(Δ{r['accuracy'] - clean_acc:+.4f}), macro_f1 {clean_f1:.4f} -> {r['macro_f1']:.4f} "
            f"(Δ{r['macro_f1'] - clean_f1:+.4f})"
        )

    out_path = EXPERIMENTS_DIR / f"{args.run_id}_robustness.json"
    out_path.write_text(json.dumps({"run_id": args.run_id, "n_test_samples": len(test_df), "results": results}, indent=2))
    print(f"\nWrote robustness results to {out_path}")


if __name__ == "__main__":
    main()
