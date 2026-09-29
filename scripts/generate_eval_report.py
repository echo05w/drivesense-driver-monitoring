#!/usr/bin/env python3
"""Render an already-produced real distraction evaluation JSON
(`experiments/distraction/<run_id>_test_eval.json`, from
`scripts/train_distraction.py evaluate`) into `reports/`: a confusion-matrix
plot and a per-class metrics table. Does not re-run evaluation or compute
any new numbers - every figure here is copied from that real JSON.

Usage:
    python scripts/generate_eval_report.py --run-id transfer_20260921_154816
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys_path_src = REPO_ROOT / "src"
import sys  # noqa: E402

sys.path.insert(0, str(sys_path_src))

from drivesense.data.state_farm import CLASS_NAMES, CLASS_ORDER  # noqa: E402

EXPERIMENTS_DIR = REPO_ROOT / "experiments" / "distraction"
REPORTS_DIR = REPO_ROOT / "reports"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    eval_path = EXPERIMENTS_DIR / f"{args.run_id}_test_eval.json"
    if not eval_path.exists():
        raise SystemExit(
            f"No evaluation JSON at {eval_path}. Run "
            f"`python scripts/train_distraction.py evaluate --run-id {args.run_id}` first "
            "(never fabricate this report from invented numbers)."
        )
    data = json.loads(eval_path.read_text())

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    cm = np.array(data["confusion_matrix"])
    labels = [CLASS_NAMES[c] for c in CLASS_ORDER]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Distraction confusion matrix — {args.run_id}\n(real test-set evaluation, n={data['n_images_evaluated']})")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    cm_path = REPORTS_DIR / f"{args.run_id}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    lines = [
        f"# Distraction evaluation report — `{args.run_id}`",
        "",
        f"Real held-out test-set evaluation (never re-run/estimated for this report; "
        f"source: `experiments/distraction/{args.run_id}_test_eval.json`).",
        "",
        f"- Test set size: {data['test_size']} images",
        f"- Accuracy: {data['accuracy']:.4f}",
        f"- Balanced accuracy: {data['balanced_accuracy']:.4f}",
        f"- Macro F1: {data['macro_f1']:.4f}",
        f"- Weighted F1: {data['weighted_f1']:.4f}",
        f"- Inference speed (CPU): {data['inference_fps']:.1f} FPS",
        "",
        "## Per-class metrics",
        "",
        "| Class | Name | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|---|",
    ]
    for code in CLASS_ORDER:
        m = data["per_class"][code]
        lines.append(
            f"| {code} | {CLASS_NAMES[code]} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['f1']:.3f} | {m['support']} |"
        )
    lines.append("")
    lines.append(f"![Confusion matrix]({cm_path.name})")
    lines.append("")

    md_path = REPORTS_DIR / f"{args.run_id}_report.md"
    md_path.write_text("\n".join(lines))

    print(f"Wrote {cm_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
