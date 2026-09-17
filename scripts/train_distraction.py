#!/usr/bin/env python3
"""End-to-end State Farm distraction pipeline: EDA -> split -> leakage check
-> train -> validate. Designed to run stage-by-stage on real data the moment
it exists, and to fail loudly rather than silently produce partial results.

This machine has no GPU, 4 CPU cores, and ~7.5GB RAM shared with the desktop
session (see docs/Master_Plan_Status.md environment notes). Defaults below
are chosen to make local CPU training actually finish in a reasonable time
without OOMing, not to represent final reported numbers - full-scale
training is expected to happen in Colab per CLAUDE.md.

Usage:
    python scripts/train_distraction.py eda
    python scripts/train_distraction.py smoke-train
    python scripts/train_distraction.py train --model simple_cnn --epochs 5
    python scripts/train_distraction.py train --model transfer --epochs 5 --unfreeze
    python scripts/train_distraction.py evaluate --run-id <run_id>
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from drivesense.data.datasets import DistractionImageDataset  # noqa: E402
from drivesense.data.splits import subject_independent_split  # noqa: E402
from drivesense.data.state_farm import CLASS_ORDER, load_metadata, validate_images  # noqa: E402
from drivesense.models.cnn import SimpleCNN, TransferLearningCNN  # noqa: E402

RAW_DIR = REPO_ROOT / "data" / "raw" / "distraction"
EXPERIMENTS_DIR = REPO_ROOT / "experiments" / "distraction"
MODELS_DIR = REPO_ROOT / "models" / "distraction"
SPLIT_CACHE = EXPERIMENTS_DIR / "split_assignment.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def build_transforms(image_size: int):
    # No horizontal flip: classes c1-c4 are left/right-hand-specific
    # (see drivesense.data.state_farm.HORIZONTAL_FLIP_SAFE) - flipping would
    # silently invert those labels.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tf, eval_tf


def cmd_eda(args: argparse.Namespace) -> None:
    df = load_metadata(RAW_DIR)
    print(f"Total labeled images: {len(df)}")
    print(f"Unique drivers: {df['subject'].nunique()}")
    print("\nImages per class:")
    print(df["classname"].value_counts().sort_index())
    print("\nImages per driver:")
    print(df["subject"].value_counts().sort_index())

    print("\nValidating every image (this opens each file once)...")
    t0 = time.time()
    report = validate_images(df["image_path"].tolist())
    print(f"Checked {report.total_checked} images in {time.time() - t0:.1f}s")
    print(f"Corrupt/unreadable: {report.n_corrupt}")
    if report.corrupt_paths:
        for p in report.corrupt_paths[:10]:
            print("  CORRUPT:", p)

    dims = pd.Series(report.dimensions)
    unique_dims = dims.value_counts()
    print("\nImage dimension distribution (width, height):")
    print(unique_dims)

    counts = df["classname"].value_counts()
    imbalance_ratio = counts.max() / counts.min()
    print(f"\nClass imbalance ratio (max/min): {imbalance_ratio:.3f}")

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "total_images": int(len(df)),
        "n_drivers": int(df["subject"].nunique()),
        "images_per_class": df["classname"].value_counts().sort_index().to_dict(),
        "images_per_driver": df["subject"].value_counts().sort_index().to_dict(),
        "n_corrupt": report.n_corrupt,
        "corrupt_paths": report.corrupt_paths,
        "unique_image_dimensions": {str(k): int(v) for k, v in unique_dims.items()},
        "class_imbalance_ratio_max_over_min": float(imbalance_ratio),
    }
    out_path = EXPERIMENTS_DIR / "eda_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote real EDA summary to {out_path} (gitignored; copy real numbers into docs by hand)")

    plots_dir = EXPERIMENTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    counts.sort_index().plot(kind="bar", ax=ax)
    ax.set_title("State Farm: images per class (real counts)")
    ax.set_xlabel("class code")
    ax.set_ylabel("image count")
    fig.tight_layout()
    fig.savefig(plots_dir / "class_distribution.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    df["subject"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("State Farm: images per driver (real counts)")
    ax.set_xlabel("driver (subject) ID")
    ax.set_ylabel("image count")
    fig.tight_layout()
    fig.savefig(plots_dir / "driver_distribution.png", dpi=120)
    plt.close(fig)

    from PIL import Image as PILImage

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, classname in enumerate(CLASS_ORDER):
        sample_row = df[df["classname"] == classname].iloc[0]
        img = PILImage.open(sample_row["image_path"])
        ax = axes[i // 5, i % 5]
        ax.imshow(img)
        ax.set_title(classname)
        ax.axis("off")
    fig.suptitle("State Farm: one real sample image per class")
    fig.tight_layout()
    fig.savefig(plots_dir / "sample_grid.png", dpi=120)
    plt.close(fig)

    print(f"Wrote real EDA plots to {plots_dir} (gitignored; reference findings in docs, not the images themselves)")


def get_or_create_split(test_size: float, val_size: float, seed: int) -> pd.DataFrame:
    df = load_metadata(RAW_DIR)
    train_df, val_df, test_df = subject_independent_split(
        df, subject_col="subject", test_size=test_size, val_size=val_size, random_state=seed
    )
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    full = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Explicit leakage assertions beyond what subject_independent_split already
    # raises internally, printed for defensibility per the user's requirement.
    train_subj, val_subj, test_subj = (
        set(train_df["subject"]),
        set(val_df["subject"]),
        set(test_df["subject"]),
    )
    assert not (train_subj & val_subj), "train/val subject leakage"
    assert not (train_subj & test_subj), "train/test subject leakage"
    assert not (val_subj & test_subj), "val/test subject leakage"
    train_paths, val_paths, test_paths = (
        set(train_df["image_path"]),
        set(val_df["image_path"]),
        set(test_df["image_path"]),
    )
    assert not (train_paths & val_paths) and not (train_paths & test_paths) and not (
        val_paths & test_paths
    ), "image-path leakage across splits"

    print(f"TRAIN drivers ({len(train_subj)}): {sorted(train_subj)}")
    print(f"VAL   drivers ({len(val_subj)}): {sorted(val_subj)}")
    print(f"TEST  drivers ({len(test_subj)}): {sorted(test_subj)}")
    print(f"TRAIN samples: {len(train_df)}, VAL samples: {len(val_df)}, TEST samples: {len(test_df)}")
    print("Leakage assertions passed: no subject or image path appears in more than one split.")

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    full.to_csv(SPLIT_CACHE, index=False)
    return full


def build_model(name: str, num_classes: int, unfreeze: bool) -> nn.Module:
    if name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    if name == "transfer":
        model = TransferLearningCNN(num_classes=num_classes)
        if unfreeze:
            model.unfreeze_backbone()
        return model
    raise ValueError(f"Unknown model {name}")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss = 0.0
    all_preds, all_labels = [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1


def cmd_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cpu")

    full = get_or_create_split(args.test_size, args.val_size, args.seed)
    train_df = full[full["split"] == "train"].reset_index(drop=True)
    val_df = full[full["split"] == "val"].reset_index(drop=True)

    if args.subsample_frac < 1.0:
        train_df = (
            train_df.groupby("classname", group_keys=False)
            .apply(lambda g: g.sample(frac=args.subsample_frac, random_state=args.seed))
            .reset_index(drop=True)
        )
        print(
            f"Subsampled TRAIN to {len(train_df)} images ({args.subsample_frac:.0%} per class) "
            "for local CPU feasibility - see docs/Master_Plan_Status.md compute notes."
        )

    if args.max_train_samples is not None:
        train_df = train_df.iloc[: args.max_train_samples].reset_index(drop=True)
    if args.max_val_samples is not None:
        val_df = val_df.iloc[: args.max_val_samples].reset_index(drop=True)

    train_tf, eval_tf = build_transforms(args.image_size)
    train_ds = DistractionImageDataset(train_df, transform=train_tf)
    val_ds = DistractionImageDataset(val_df, transform=eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = build_model(args.model, num_classes=len(CLASS_ORDER), unfreeze=args.unfreeze).to(device)
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    run_id = f"{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / f"{run_id}.pt"

    history = []
    best_val_f1 = -1.0
    epochs_without_improvement = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        epoch_time = time.time() - t0
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} train_f1={train_f1:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
            f"| {epoch_time:.1f}s"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_macro_f1": train_f1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
                "epoch_time_sec": epoch_time,
            }
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model,
                    "num_classes": len(CLASS_ORDER),
                    "class_order": CLASS_ORDER,
                    "image_size": args.image_size,
                    "epoch": epoch,
                    "val_macro_f1": val_f1,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if args.patience and epochs_without_improvement >= args.patience:
                print(f"Early stopping: no val_macro_f1 improvement for {args.patience} epochs.")
                break

    total_time = time.time() - t_start

    record = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit_hash(),
        "seed": args.seed,
        "model": args.model,
        "unfrozen_backbone": args.unfreeze,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "optimizer": "Adam",
        "lr": args.lr,
        "epochs_requested": args.epochs,
        "epochs_run": len(history),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "subsample_frac": args.subsample_frac,
        "test_size_frac": args.test_size,
        "val_size_frac": args.val_size,
        "best_val_macro_f1": best_val_f1,
        "total_runtime_sec": total_time,
        "checkpoint_path": str(checkpoint_path),
        "history": history,
    }
    record_path = EXPERIMENTS_DIR / f"{run_id}.json"
    record_path.write_text(json.dumps(record, indent=2))
    print(f"\nSaved run record to {record_path}")
    print(f"Saved best checkpoint to {checkpoint_path}")
    print(f"Best val macro F1: {best_val_f1:.4f}")

    try:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'experiments' / 'mlflow.db'}")
        mlflow.set_experiment("drivesense_distraction")
        with mlflow.start_run(run_name=run_id):
            mlflow.log_params(
                {
                    "model": args.model,
                    "unfrozen_backbone": args.unfreeze,
                    "image_size": args.image_size,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "seed": args.seed,
                    "subsample_frac": args.subsample_frac,
                    "git_commit": git_commit_hash(),
                }
            )
            for h in history:
                mlflow.log_metrics(
                    {k: v for k, v in h.items() if k != "epoch"}, step=h["epoch"]
                )
            mlflow.log_metric("best_val_macro_f1", best_val_f1)
            mlflow.log_artifact(str(record_path))
    except Exception as exc:  # pragma: no cover - tracking is best-effort
        print(f"(mlflow logging skipped: {exc})")


def cmd_evaluate(args: argparse.Namespace) -> None:
    record_path = EXPERIMENTS_DIR / f"{args.run_id}.json"
    if not record_path.exists():
        raise SystemExit(f"No run record at {record_path}")
    record = json.loads(record_path.read_text())
    checkpoint = torch.load(record["checkpoint_path"], map_location="cpu", weights_only=False)

    full = pd.read_csv(SPLIT_CACHE)
    test_df = full[full["split"] == "test"].reset_index(drop=True)
    if args.max_test_samples is not None:
        test_df = test_df.iloc[: args.max_test_samples].reset_index(drop=True)

    _, eval_tf = build_transforms(checkpoint["image_size"])
    test_ds = DistractionImageDataset(test_df, transform=eval_tf)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    model = build_model(checkpoint["model_name"], checkpoint["num_classes"], unfreeze=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    t0 = time.time()
    n_images = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            all_preds.extend(outputs.argmax(dim=1).tolist())
            all_labels.extend(labels.tolist())
            n_images += images.size(0)
    inference_time = time.time() - t0
    fps = n_images / inference_time if inference_time > 0 else float("nan")

    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(CLASS_ORDER))), zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_ORDER))))

    results = {
        "run_id": args.run_id,
        "test_size": len(test_df),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": {
            CLASS_ORDER[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1_per_class[i]),
                "support": int(support[i]),
            }
            for i in range(len(CLASS_ORDER))
        },
        "confusion_matrix": cm.tolist(),
        "inference_fps": fps,
        "n_images_evaluated": n_images,
    }
    out_path = EXPERIMENTS_DIR / f"{args.run_id}_test_eval.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nWrote test evaluation to {out_path}")


def cmd_smoke_train(args: argparse.Namespace) -> None:
    """Tiny end-to-end run: few images, few batches, checkpoint save+load
    round trip. Purpose: catch pipeline bugs cheaply before real training."""
    args.model = "simple_cnn"
    args.image_size = 64
    args.batch_size = 4
    args.epochs = 1
    args.lr = 1e-3
    args.seed = 0
    args.unfreeze = False
    args.subsample_frac = 1.0
    args.test_size = 0.2
    args.val_size = 0.2
    args.num_workers = 0
    args.patience = 0
    args.max_train_samples = 16
    args.max_val_samples = 8
    print("=== SMOKE TRAIN: tiny subset, 1 epoch, checkpoint round-trip ===")
    cmd_train(args)

    # Verify checkpoint load round-trip explicitly.
    ckpts = sorted(MODELS_DIR.glob("simple_cnn_*.pt"))
    assert ckpts, "smoke train produced no checkpoint"
    latest = ckpts[-1]
    checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model_name"], checkpoint["num_classes"], unfreeze=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Checkpoint save/load round-trip OK: {latest}")
    print("=== SMOKE TRAIN PASSED ===")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("eda")

    p_smoke = sub.add_parser("smoke-train")
    p_smoke.set_defaults(func=cmd_smoke_train)

    p_train = sub.add_parser("train")
    p_train.add_argument("--model", choices=["simple_cnn", "transfer"], default="simple_cnn")
    p_train.add_argument("--unfreeze", action="store_true")
    p_train.add_argument("--image-size", type=int, default=96, dest="image_size")
    p_train.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--test-size", type=float, default=0.2, dest="test_size")
    p_train.add_argument("--val-size", type=float, default=0.15, dest="val_size")
    p_train.add_argument(
        "--subsample-frac",
        type=float,
        default=1.0,
        dest="subsample_frac",
        help="Fraction of TRAIN images per class to use (local CPU feasibility).",
    )
    p_train.add_argument("--num-workers", type=int, default=2, dest="num_workers")
    p_train.add_argument("--patience", type=int, default=3)
    p_train.add_argument("--max-train-samples", type=int, default=None, dest="max_train_samples")
    p_train.add_argument("--max-val-samples", type=int, default=None, dest="max_val_samples")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--run-id", required=True, dest="run_id")
    p_eval.add_argument("--max-test-samples", type=int, default=None, dest="max_test_samples")
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    if args.command == "eda":
        cmd_eda(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
