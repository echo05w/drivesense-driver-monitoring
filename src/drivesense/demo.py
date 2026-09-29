"""Real-time driver-awareness demo: distraction CNN + drowsiness PERCLOS
fused through `DriverAwarenessEngine` into a single overlay.

    python demo.py --source 0                  # webcam
    python demo.py --source path/to/video.mp4  # video file
    python demo.py --source path/to/image.jpg  # single image
    python -m drivesense.demo --source 0        # equivalent, as a module

Honesty notes (see CLAUDE.md "Never fabricate"):
- Distraction: a real trained `TransferLearningCNN` checkpoint
  (`models/distraction/*.pt`) if one is found, loaded and run for real. If
  none is found, distraction is reported as "unavailable" rather than
  faked, and the risk fusion falls back to drowsiness-only.
- Drowsiness: no trained temporal model exists yet (see
  `docs/Master_Plan_Status.md` row 12) - this demo drives a real, rule-based
  PERCLOS score (`drivesense.inference.drowsiness.PerclosDrowsinessScorer`)
  from a real, lightweight local landmark extractor
  (`drivesense.features.landmarks.HaarCascadeLandmarkExtractor`), which is a
  genuine but coarse approximation of true landmark-based EAR/MAR - see that
  extractor's module docstring for exactly what is and is not measured. If
  no face is ever detected, drowsiness is reported as "unavailable", never
  fabricated as 0.
- `opencv-python-headless` is installed (no GUI backend), so this demo never
  calls `cv2.imshow` - it always writes an annotated output video (or image)
  to disk, the same pattern already used by `scripts/demo_distraction.py`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = REPO_ROOT / "models" / "distraction"

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

RISK_COLORS = {
    "LOW": (0, 200, 0),
    "MEDIUM": (0, 200, 255),
    "HIGH": (0, 140, 255),
    "CRITICAL": (0, 0, 255),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_default_checkpoint() -> Optional[Path]:
    """Pick the demo's default distraction checkpoint, if any exist.

    Prefers the robustness-augmented fine-tuned transfer-learning run
    (`transfer_20260921_154816`) over the original fine-tuned run
    (`transfer_20260917_194003`): real before/after re-evaluation
    (`docs/Master_Plan_Status.md` row 17) showed it trades a small clean-set
    macro-F1 cost (0.697 -> 0.680) for a large robustness gain under blur/
    noise/JPEG degradation (each +0.16 to +0.30 macro-F1) - the right
    trade-off for a system meant to run on real, imperfect camera footage.
    Falls back to the newest checkpoint present if that exact run is
    missing, and to None (no fabricated prediction) if no checkpoint exists
    at all.
    """
    preferred = MODELS_DIR / "transfer_20260921_154816.pt"
    if preferred.exists():
        return preferred
    candidates = sorted(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


class DistractionModel:
    """Wraps a loaded distraction checkpoint for single-frame inference."""

    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        from drivesense.data.state_farm import CLASS_NAMES, CLASS_ORDER
        from drivesense.models.cnn import SimpleCNN, TransferLearningCNN

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_name = checkpoint["model_name"]
        if model_name == "simple_cnn":
            model = SimpleCNN(num_classes=checkpoint["num_classes"])
        elif model_name == "transfer":
            model = TransferLearningCNN(num_classes=checkpoint["num_classes"])
        else:
            raise ValueError(f"Unknown model_name in checkpoint: {model_name}")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()

        self.model = model
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.class_order = CLASS_ORDER
        self.class_names = CLASS_NAMES
        image_size = checkpoint["image_size"]
        self.transform = transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor(), NORMALIZE]
        )

    def predict(self, frame_rgb: np.ndarray) -> dict:
        img = Image.fromarray(frame_rgb)
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]
        idx = int(probs.argmax())
        code = self.class_order[idx]
        name = self.class_names[code]
        confidence = float(probs[idx])
        # Distraction score: 1 - P(safe_driving), i.e. total probability mass
        # on every non-safe class, not just the top prediction's confidence -
        # a more stable continuous signal for the risk-fusion EMA than a
        # single argmax swap would give.
        safe_idx = self.class_order.index("c0")
        distraction_score = 1.0 - float(probs[safe_idx])
        return {
            "class_code": code,
            "class_name": name,
            "confidence": confidence,
            "distraction_score": distraction_score,
        }


def open_source(source: str):
    """Open a webcam index, video file, or single image. Returns
    (kind, capture_or_frame) where kind is "video" or "image"."""
    path = Path(source)
    if path.suffix.lower() in IMAGE_EXTENSIONS and path.exists():
        frame_bgr = cv2.imread(str(path))
        if frame_bgr is None:
            raise RuntimeError(f"Could not read image: {source}")
        return "image", frame_bgr

    cam_index: Optional[int] = None
    if source.isdigit():
        cam_index = int(source)
    cap = cv2.VideoCapture(cam_index if cam_index is not None else source)
    if not cap.isOpened():
        cap.release()
        if cam_index is not None:
            raise RuntimeError(
                f"Could not open webcam index {cam_index}. No camera device is available "
                "in this environment. Use --source path/to/video.mp4 (or an image) instead."
            )
        raise RuntimeError(f"Could not open video source: {source}")
    return "video", cap


def draw_overlay(frame_bgr: np.ndarray, state: dict, fps: Optional[float]) -> np.ndarray:
    lines = []
    if state["distraction_class"] is not None:
        lines.append(f"Distraction class: {state['distraction_class']} ({state['distraction_confidence']:.2f})")
    else:
        lines.append("Distraction model: unavailable")
    lines.append(
        f"Distraction: {state['distraction_pct']}"
        if state["distraction_pct"] is not None
        else "Distraction: n/a"
    )
    lines.append(
        f"Drowsiness: {state['drowsiness_pct']}" if state["drowsiness_pct"] is not None else "Drowsiness: unavailable (no face)"
    )
    lines.append(f"Awareness: {state['awareness_pct']}")
    lines.append(f"Risk: {state['risk_level']}")
    if fps is not None:
        lines.append(f"FPS: {fps:.1f}")

    color = RISK_COLORS.get(state["risk_level"], (255, 255, 255))
    for i, line in enumerate(lines):
        cv2.putText(frame_bgr, line, (10, 26 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3)
        cv2.putText(frame_bgr, line, (10, 26 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1)
    return frame_bgr


def run(args: argparse.Namespace) -> dict:
    from drivesense.features.landmarks import HaarCascadeLandmarkExtractor
    from drivesense.inference.drowsiness import PerclosDrowsinessScorer
    from drivesense.inference.risk_engine import DriverAwarenessEngine

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    distraction_model: Optional[DistractionModel] = None
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else find_default_checkpoint()
    if checkpoint_path is not None and checkpoint_path.exists():
        distraction_model = DistractionModel(checkpoint_path, device)
        print(f"[demo] Distraction model loaded: {checkpoint_path.name} (device={device})")
        if args.source == "0" or args.source.isdigit():
            print(
                "[demo] NOTE: the distraction model was trained on State Farm's "
                "side-angle dashcam images, not frontal webcam footage - live "
                "webcam distraction predictions are genuinely out-of-domain and "
                "should not be trusted as accurate; the drowsiness signal (a "
                "frontal-camera task) is the meaningful one on webcam input. "
                "See README 'Known limitations'."
            )
    else:
        print("[demo] No distraction checkpoint found under models/distraction/ - "
              "distraction reported as unavailable (see README 'Training' section).")

    landmark_extractor = None
    drowsiness_scorer = PerclosDrowsinessScorer()
    if not args.no_drowsiness:
        try:
            landmark_extractor = HaarCascadeLandmarkExtractor()
            print("[demo] Drowsiness fallback extractor ready (Haar cascades, local approximation).")
        except Exception as exc:  # pragma: no cover - network/env dependent
            print(f"[demo] Could not initialize drowsiness extractor ({exc}); drowsiness disabled.")

    awareness_engine = DriverAwarenessEngine()

    try:
        kind, handle = open_source(args.source)
    except RuntimeError as exc:
        print(f"[demo] ERROR: {exc}")
        return {"status": "error", "message": str(exc)}

    writer = None
    frames_out = []
    n_frames = 0
    last_state = None
    t_start = time.time()
    fps_estimate = None

    def process_frame(frame_bgr: np.ndarray) -> np.ndarray:
        nonlocal last_state, fps_estimate
        t0 = time.time()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        distraction_score = None
        distraction_class = None
        distraction_confidence = None
        if distraction_model is not None:
            pred = distraction_model.predict(frame_rgb)
            distraction_score = pred["distraction_score"]
            distraction_class = pred["class_name"]
            distraction_confidence = pred["confidence"]

        drowsiness_score = None
        if landmark_extractor is not None:
            features = landmark_extractor.extract(frame_rgb)
            drowsiness_score = drowsiness_scorer.update(features)

        fused = awareness_engine.update(
            distraction_score=distraction_score if distraction_score is not None else 0.0,
            drowsiness_score=drowsiness_score,
        )

        latency = time.time() - t0
        fps_estimate = 1.0 / latency if latency > 0 else fps_estimate

        state = {
            "distraction_class": distraction_class,
            "distraction_confidence": distraction_confidence,
            "distraction_pct": f"{distraction_score * 100:.0f}%" if distraction_score is not None else None,
            "drowsiness_pct": f"{fused['drowsiness'] * 100:.0f}%" if fused["drowsiness"] is not None else None,
            "awareness_pct": f"{fused['awareness'] * 100:.0f}%",
            "risk_score": fused["risk_score"],
            "risk_level": fused["risk_level"].value,
        }
        last_state = state
        return draw_overlay(frame_bgr, state, fps_estimate)

    if kind == "image":
        annotated = process_frame(handle)
        out_path = args.out or str(REPO_ROOT / "experiments" / "demo_output.jpg")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, annotated)
        n_frames = 1
        print(f"[demo] Wrote annotated image to {out_path}")
    else:
        cap = handle
        out_path = args.out or str(REPO_ROOT / "experiments" / "demo_output_live.mp4")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        display_size = (640, 480)
        writer = cv2.VideoWriter(out_path, fourcc, 15.0, display_size)
        if args.max_frames is None and (args.source == "0" or args.source.isdigit()):
            print("[demo] Running continuously on the live webcam - press Ctrl+C to stop "
                  "(the output video is finalized cleanly on interrupt). Use --max-frames "
                  "N for a bounded/automated run.")
        last_print = time.time()
        try:
            while args.max_frames is None or n_frames < args.max_frames:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                # Run detection/inference at the source's native resolution -
                # resizing beforehand would distort aspect ratio (e.g. a
                # portrait phone video squashed into a landscape box) and
                # measurably hurts Haar cascade face/eye detection. Only the
                # already-annotated frame is resized, purely for a
                # consistent output video size.
                annotated = process_frame(frame_bgr)
                annotated = cv2.resize(annotated, display_size)
                writer.write(annotated)
                n_frames += 1
                # Live progress feedback (a presenter watching stdout should
                # see this is actually running, not just a silent hang until
                # the process exits/is interrupted).
                if time.time() - last_print >= 1.0 and last_state is not None:
                    print(
                        f"[demo] frame {n_frames} | {last_state['distraction_class'] or 'n/a'} "
                        f"({last_state['distraction_pct'] or 'n/a'}) | "
                        f"drowsiness {last_state['drowsiness_pct'] or 'n/a'} | "
                        f"risk {last_state['risk_level']} | {fps_estimate:.1f} FPS"
                    )
                    last_print = time.time()
        except KeyboardInterrupt:
            print(f"\n[demo] Stopped by user after {n_frames} frames.")
        finally:
            cap.release()
            writer.release()
        print(f"[demo] Processed {n_frames} frames -> {out_path}")

    elapsed = time.time() - t_start
    summary = {
        "status": "ok",
        "source": args.source,
        "n_frames": n_frames,
        "elapsed_sec": elapsed,
        "avg_fps": (n_frames / elapsed) if elapsed > 0 and kind == "video" else fps_estimate,
        "distraction_model": str(checkpoint_path) if checkpoint_path else None,
        "drowsiness_available": landmark_extractor is not None,
        "final_state": last_state,
    }
    print(json.dumps(summary, indent=2, default=str))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0", help="webcam index (e.g. 0), or a video/image file path")
    parser.add_argument("--checkpoint", default=None, help="distraction .pt checkpoint (default: best available)")
    parser.add_argument("--out", default=None, help="output annotated video/image path")
    parser.add_argument("--max-frames", type=int, default=None, help="stop after N frames (for smoke tests)")
    parser.add_argument("--no-drowsiness", action="store_true", help="disable the local drowsiness fallback")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
