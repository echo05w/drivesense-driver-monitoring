#!/usr/bin/env python3
"""Distraction-only real-time inference demo, fused through the temporal
risk engine.

Honesty note: State Farm provides labeled still images, not video, so
there is no real dashcam video to run this on. This demo simulates a video
feed from a sequence of real, labeled test images (each held for a few
frames) - the model inference, risk fusion, FPS measurement, and on-screen
overlay are all genuinely real; only the "video" is synthesized from real
stills rather than being an actual continuous recording. This is NOT a
substitute for a real video demo and should not be presented as one -
it verifies the inference/fusion/overlay pipeline mechanics end-to-end with
real, labeled data before a real Colab webcam/video demo is built.

Drowsiness is not yet available (no trained model), so the risk engine only
ever receives a distraction label; every risk level shown here reflects
distraction alone.

Usage:
    python scripts/demo_distraction.py --run-id transfer_20260917_194003 --n-frames 90
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from drivesense.data.state_farm import CLASS_NAMES, CLASS_ORDER  # noqa: E402
from drivesense.inference.risk_engine import RiskEngineConfig, TemporalRiskEngine  # noqa: E402
from drivesense.models.cnn import SimpleCNN, TransferLearningCNN  # noqa: E402

EXPERIMENTS_DIR = REPO_ROOT / "experiments" / "distraction"
SPLIT_CACHE = EXPERIMENTS_DIR / "split_assignment.csv"
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Every non-"safe driving" class in this dataset represents a distraction
# category by the dataset's own definition, so all of them count toward
# risk - this is a simple, defensible choice, not a subjective severity
# ranking the data doesn't support.
DISTRACTION_CLASSES_OF_CONCERN = frozenset(
    name for code, name in CLASS_NAMES.items() if code != "c0"
)


def build_model(name: str, num_classes: int) -> torch.nn.Module:
    if name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    if name == "transfer":
        return TransferLearningCNN(num_classes=num_classes)
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--n-frames", type=int, default=90, help="number of real test images to simulate as frames")
    parser.add_argument("--hold-frames", type=int, default=1, help="how many output video frames per image")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=str(REPO_ROOT / "experiments" / "distraction" / "demo_output.mp4"))
    args = parser.parse_args()

    record = json.loads((EXPERIMENTS_DIR / f"{args.run_id}.json").read_text())
    checkpoint = torch.load(record["checkpoint_path"], map_location="cpu", weights_only=False)

    model = build_model(checkpoint["model_name"], checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_size = checkpoint["image_size"]
    transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor(), NORMALIZE]
    )

    full = pd.read_csv(SPLIT_CACHE)
    test_df = full[full["split"] == "test"].sample(n=args.n_frames, random_state=args.seed).reset_index(drop=True)

    engine = TemporalRiskEngine(RiskEngineConfig(distraction_classes_of_concern=DISTRACTION_CLASSES_OF_CONCERN))

    display_size = (640, 480)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, 5.0, display_size)

    correct = 0
    latencies = []
    risk_history = []

    for _, row in test_df.iterrows():
        img_pil = Image.open(row["image_path"]).convert("RGB")
        tensor = transform(img_pil).unsqueeze(0)

        t0 = time.time()
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        latency = time.time() - t0
        latencies.append(latency)

        pred_idx = int(probs.argmax())
        pred_code = CLASS_ORDER[pred_idx]
        pred_name = CLASS_NAMES[pred_code]
        confidence = float(probs[pred_idx])
        true_correct = pred_code == row["classname"]
        correct += int(true_correct)

        risk = engine.update(distraction_label=pred_name, drowsiness_label=None)
        risk_history.append(risk.name)

        frame_bgr = cv2.cvtColor(np.array(img_pil.resize(display_size)), cv2.COLOR_RGB2BGR)
        overlay_lines = [
            f"State: {pred_name} ({confidence:.2f})",
            f"True: {CLASS_NAMES[row['classname']]}",
            f"Risk: {risk.name}",
            f"Inference: {1.0 / latency:.0f} FPS" if latency > 0 else "Inference: n/a",
        ]
        for i, line in enumerate(overlay_lines):
            color = (0, 0, 255) if risk.name == "WARNING" else (0, 165, 255) if risk.name == "CAUTION" else (0, 200, 0)
            cv2.putText(frame_bgr, line, (10, 30 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        for _ in range(args.hold_frames):
            writer.write(frame_bgr)

    writer.release()

    avg_fps = 1.0 / (sum(latencies) / len(latencies))
    accuracy_on_demo_sample = correct / len(test_df)
    summary = {
        "run_id": args.run_id,
        "n_frames_simulated": len(test_df),
        "accuracy_on_simulated_sequence": accuracy_on_demo_sample,
        "avg_inference_fps": avg_fps,
        "risk_level_counts": {lvl: risk_history.count(lvl) for lvl in set(risk_history)},
        "output_video": args.out,
        "note": (
            "Simulated feed from real labeled State Farm test images (no real "
            "dashcam video exists in this dataset) - inference/fusion/overlay "
            "pipeline is real; the video continuity is not."
        ),
    }
    print(json.dumps(summary, indent=2))
    out_json = EXPERIMENTS_DIR / f"{args.run_id}_demo_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote demo video to {args.out}")
    print(f"Wrote demo summary to {out_json}")


if __name__ == "__main__":
    main()
