"""DriveSense – Driver Awareness Monitoring: a presenter-friendly GUI.

    streamlit run app.py

Reuses the exact same real pipeline as `demo.py` / `src/drivesense/demo.py`
(distraction checkpoint loading, the Haar-cascade drowsiness fallback,
PERCLOS scoring, and `DriverAwarenessEngine` fusion) — no model logic is
reimplemented here, only presentation.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from drivesense.demo import DistractionModel, find_default_checkpoint  # noqa: E402
from drivesense.features.landmarks import HaarCascadeLandmarkExtractor  # noqa: E402
from drivesense.inference.drowsiness import PerclosDrowsinessScorer  # noqa: E402
from drivesense.inference.risk_engine import DriverAwarenessEngine  # noqa: E402

import torch  # noqa: E402

st.set_page_config(page_title="DriveSense", page_icon="🚗", layout="wide")

RISK_COLORS = {
    "LOW": "#2ecc71",
    "MEDIUM": "#f1c40f",
    "HIGH": "#e67e22",
    "CRITICAL": "#e74c3c",
}


# --- One-time setup, cached across reruns within the session -------------


@st.cache_resource(show_spinner="Loading distraction model...")
def load_distraction_model():
    checkpoint_path = find_default_checkpoint()
    if checkpoint_path is None or not checkpoint_path.exists():
        return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistractionModel(checkpoint_path, device)
    return model, checkpoint_path


@st.cache_resource(show_spinner="Loading local drowsiness extractor...")
def load_landmark_extractor():
    try:
        return HaarCascadeLandmarkExtractor(), None
    except Exception as exc:  # pragma: no cover - network/env dependent
        return None, str(exc)


def init_state() -> None:
    if "running" not in st.session_state:
        st.session_state.running = False
    if "cap" not in st.session_state:
        st.session_state.cap = None
    if "drowsiness_scorer" not in st.session_state:
        st.session_state.drowsiness_scorer = PerclosDrowsinessScorer()
    if "awareness_engine" not in st.session_state:
        st.session_state.awareness_engine = DriverAwarenessEngine()
    if "n_frames" not in st.session_state:
        st.session_state.n_frames = 0
    if "last_fps" not in st.session_state:
        st.session_state.last_fps = None
    if "last_frame_rgb" not in st.session_state:
        st.session_state.last_frame_rgb = None
    if "last_state" not in st.session_state:
        st.session_state.last_state = None


init_state()
distraction_model, checkpoint_path = load_distraction_model()
landmark_extractor, extractor_error = load_landmark_extractor()


def _brighten_for_display(frame_rgb: np.ndarray) -> np.ndarray:
    """Auto-brighten a dark frame for the live preview only — never used for
    inference, which always runs on the untouched original frame, so this
    cannot change what the models predict."""
    mean_brightness = float(frame_rgb.mean())
    if mean_brightness >= 60:
        return frame_rgb
    lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    gain = min(2.2, 90.0 / max(mean_brightness, 1.0))
    l = cv2.convertScaleAbs(l, alpha=gain, beta=0)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def process_frame(frame_bgr: np.ndarray) -> dict:
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
        drowsiness_score = st.session_state.drowsiness_scorer.update(features)

    fused = st.session_state.awareness_engine.update(
        distraction_score=distraction_score if distraction_score is not None else 0.0,
        drowsiness_score=drowsiness_score,
    )

    latency = time.time() - t0
    st.session_state.last_fps = (1.0 / latency) if latency > 0 else st.session_state.last_fps
    st.session_state.n_frames += 1
    st.session_state.last_frame_rgb = frame_rgb

    state = {
        "distraction_class": distraction_class,
        "distraction_confidence": distraction_confidence,
        "distraction_pct": distraction_score * 100 if distraction_score is not None else None,
        "drowsiness_pct": fused["drowsiness"] * 100 if fused["drowsiness"] is not None else None,
        "awareness_pct": fused["awareness"] * 100,
        "risk_score": fused["risk_score"],
        "risk_level": fused["risk_level"].value,
    }
    st.session_state.last_state = state
    return state


# --- Header ----------------------------------------------------------------

st.markdown(
    """
    <style>
    .ds-card {border-radius: 12px; padding: 18px 20px; background: #1e222b;
               border: 1px solid #333944; margin-bottom: 12px;}
    .ds-card h3 {margin: 0 0 6px 0; font-size: 0.85rem; letter-spacing: .04em;
                 color: #9aa4b2; text-transform: uppercase;}
    .ds-card .val {font-size: 1.9rem; font-weight: 700; color: #f2f4f7;}
    .ds-card .sub {font-size: 0.85rem; color: #9aa4b2; margin-top: 2px;}
    .ds-risk-badge {border-radius: 12px; padding: 22px; text-align: center;
                     font-size: 2.1rem; font-weight: 800; color: #111; margin-bottom: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚗 DriveSense — Driver Awareness Monitoring")
st.caption(
    "Live frontal webcam distraction predictions are out-of-domain (the model was "
    "trained on side-angle dashcam images) — shown for demonstration purposes only."
)

# --- Controls ----------------------------------------------------------------

control_col, source_col = st.columns([2, 3])
with control_col:
    start_col, stop_col, reset_col = st.columns(3)
    if start_col.button("▶ Start camera", width="stretch", disabled=st.session_state.running):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam (index 0). No camera device is available. "
                      "Use the video/image upload option below instead.")
        else:
            st.session_state.cap = cap
            st.session_state.running = True
            st.rerun()
    if stop_col.button("⏹ Stop camera", width="stretch", disabled=not st.session_state.running):
        st.session_state.running = False
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
        st.rerun()
    if reset_col.button("↺ Reset", width="stretch"):
        st.session_state.drowsiness_scorer.reset()
        st.session_state.awareness_engine.reset()
        st.session_state.n_frames = 0
        st.session_state.last_state = None
        st.session_state.last_frame_rgb = None
        st.rerun()

with source_col:
    uploaded = st.file_uploader(
        "…or analyze an uploaded image/video instead of the live webcam",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
    )
    if uploaded is not None and st.button("Analyze upload"):
        suffix = Path(uploaded.name).suffix.lower()
        tmp_path = REPO_ROOT / "experiments" / f"_upload{suffix}"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(uploaded.getvalue())
        if suffix in (".jpg", ".jpeg", ".png"):
            frame_bgr = cv2.imread(str(tmp_path))
            process_frame(frame_bgr)
        else:
            cap = cv2.VideoCapture(str(tmp_path))
            n = 0
            with st.spinner("Analyzing video (sampling frames)..."):
                while n < 150:  # bounded so a long upload doesn't hang the UI
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break
                    process_frame(frame_bgr)
                    n += 1
            cap.release()
        st.rerun()

if distraction_model is None:
    st.warning(
        "No distraction checkpoint found under `models/distraction/` — distraction "
        "reported as unavailable rather than fabricated. Run "
        "`python scripts/train_distraction.py train ...` first."
    )
if landmark_extractor is None:
    st.info(f"Drowsiness fallback unavailable ({extractor_error}). Drowsiness will show 'unavailable'.")

st.divider()

# --- Live camera panel + status cards ---------------------------------------

cam_col, status_col = st.columns([3, 2])

with cam_col:
    st.subheader("Live camera")
    frame_placeholder = st.empty()
    if st.session_state.last_frame_rgb is not None:
        display_frame = _brighten_for_display(st.session_state.last_frame_rgb)
        frame_placeholder.image(display_frame, width="stretch")
        if display_frame is not st.session_state.last_frame_rgb:
            st.caption("Preview auto-brightened for visibility (low-light room) — "
                       "predictions still run on the original, unmodified frame.")
    else:
        frame_placeholder.info("Camera not running — click **Start camera**, or analyze an upload above.")

with status_col:
    st.subheader("Driver state")
    state = st.session_state.last_state
    badge_ph = st.empty()
    cards_ph = st.empty()
    fps_ph = st.empty()

    def render_status(state, fps):
        if state is None:
            badge_ph.markdown(
                '<div class="ds-risk-badge" style="background:#3a3f4b;color:#cfd4dc;">No data yet</div>',
                unsafe_allow_html=True,
            )
            cards_ph.empty()
            fps_ph.empty()
            return

        color = RISK_COLORS.get(state["risk_level"], "#888")
        badge_ph.markdown(
            f'<div class="ds-risk-badge" style="background:{color};">RISK: {state["risk_level"]}</div>',
            unsafe_allow_html=True,
        )

        dist_class = state["distraction_class"] or "unavailable"
        dist_conf = f"{state['distraction_confidence']*100:.0f}% confidence" if state["distraction_confidence"] is not None else ""
        dist_pct = f"{state['distraction_pct']:.0f}%" if state["distraction_pct"] is not None else "n/a"
        drowsy_pct = f"{state['drowsiness_pct']:.0f}%" if state["drowsiness_pct"] is not None else "unavailable"
        aware_pct = f"{state['awareness_pct']:.0f}%"

        cards_ph.markdown(
            f"""
            <div class="ds-card"><h3>Distraction class</h3>
                <div class="val">{dist_class}</div><div class="sub">{dist_conf}</div></div>
            <div class="ds-card"><h3>Distraction</h3><div class="val">{dist_pct}</div></div>
            <div class="ds-card"><h3>Drowsiness</h3><div class="val">{drowsy_pct}</div>
                <div class="sub">{"real fallback signal (Haar/PERCLOS)" if state['drowsiness_pct'] is not None else "no face detected"}</div></div>
            <div class="ds-card"><h3>Awareness</h3><div class="val">{aware_pct}</div></div>
            """,
            unsafe_allow_html=True,
        )
        fps_text = f"{fps:.1f} FPS" if fps else "n/a"
        fps_ph.caption(f"Frames processed: {st.session_state.n_frames} · {fps_text}")

    render_status(state, st.session_state.last_fps)

with st.expander("About this system (architecture)"):
    st.markdown(
        """
        - **Distraction:** `TransferLearningCNN` (MobileNetV3-Small, ImageNet-pretrained,
          fine-tuned) trained on the real State Farm Distracted Driver dataset —
          real trained checkpoint, real inference.
        - **Drowsiness:** trained temporal `DrowsinessGRU`/`DrowsinessTemporalCNN`
          architectures exist but are **not yet trained** (blocked locally by a
          MediaPipe memory constraint, deferred to Google Colab). This demo instead
          uses a real, working local fallback: `HaarCascadeLandmarkExtractor`
          (OpenCV face/eye detection) feeding a rolling **PERCLOS** (percentage of
          eye closure) score — a real, established drowsiness metric, but a coarser
          approximation than true landmark-based EAR/MAR. Never fabricated: if no
          face is found, drowsiness shows "unavailable", not a fake number.
        - **Risk fusion:** rule-based `DriverAwarenessEngine`
          (`risk_score = 0.55·drowsiness + 0.35·distraction + 0.10·sustained-evidence`,
          EMA-smoothed) — not a trained model, by design (interpretability for a
          safety-alert system).
        - Full details: `README.md`, `docs/Master_Plan_Status.md`.
        """
    )

# --- Live loop: process one frame, then immediately rerun -------------------

if st.session_state.running and st.session_state.cap is not None:
    ret, frame_bgr = st.session_state.cap.read()
    if not ret:
        st.session_state.running = False
        st.session_state.cap.release()
        st.session_state.cap = None
        st.error("Lost the camera feed — stopped.")
    else:
        process_frame(frame_bgr)
        time.sleep(0.03)
        st.rerun()
