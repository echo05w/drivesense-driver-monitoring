# DriveSense

Deep learning driver monitoring system for real-time drowsiness and
distraction detection from an ordinary RGB webcam.

**Student:** Hasan Soliyev
**Selected project track:** Track 1 — Individual Project Track
**Project Brief status:** `PENDING_MENTOR_APPROVAL` (see `docs/Individual_Project_Brief.md`)

> This README is a living document. Results below reflect actual runs
> against real data as they land. Nothing here is a projected/planned
> number; per the project's academic-integrity commitments (see
> `docs/Rubric_Alignment.md`), a metric only appears once it has actually
> been produced by a run. As of the latest session: the distraction CNN is
> trained/evaluated on real data, the risk-fusion engine and a real-time
> demo (webcam/video/image) both work end-to-end, and drowsiness uses a
> real but clearly-labeled rule-based local fallback while the trained
> temporal model awaits a real Google Colab run (see "Known limitations").

## Problem statement

Fleet operators and individual drivers lack an affordable, transparent,
retrainable, camera-only way to detect drowsy or distracted driving in real
time. DriveSense addresses this with two deep-learning classifiers (drowsiness
state, distraction category) fused into a single graduated risk score, running
from a live or recorded webcam feed. Full problem framing:
`docs/Individual_Project_Brief.md`.

## Dataset source

- **Distraction:** [State Farm Distracted Driver Detection](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data) (Kaggle)
- **Drowsiness:** [UTA-RLDD](https://sites.google.com/view/utarldd/home) (Real-Life Drowsiness Dataset)

Full access/licensing comparison and rationale: `docs/Dataset_Research.md`.
Raw data is never committed to this repository (see `.gitignore`) — see
"Setup" below for acquisition.

## ML task type

Two supervised multi-class classification tasks (single-frame distraction
classification; short-sequence drowsiness-state classification), fused by a
non-trained temporal risk-scoring layer. Full formulation:
`docs/Individual_Project_Brief.md` §7.

## Project pipeline / system architecture

```
Camera / video / image
          │
          ▼
  Frame preprocessing (resize, normalize)
          │
   ┌──────┴────────────────────────┐
   ▼                                ▼
Distraction CNN               Drowsiness module
(TransferLearningCNN,         (HaarCascadeLandmarkExtractor -> EAR proxy
 MobileNetV3-Small,            -> PerclosDrowsinessScorer; trained
 real trained checkpoint)      DrowsinessGRU/1D-CNN pending real Colab run)
   │                                │
   └───────────┬────────────────────┘
               ▼
     Temporal smoothing (EMA)
               │
               ▼
   Risk fusion (DriverAwarenessEngine)
               │
               ▼
  Drowsiness % / Distraction % / Awareness % / Risk level (LOW..CRITICAL)
               │
               ▼
      On-screen overlay (scripts/demo.py)
```

`scripts/demo_distraction.py` additionally exposes the original discrete
NORMAL/CAUTION/WARNING risk engine (`TemporalRiskEngine`) on distraction
alone — kept as-is; `scripts/demo.py` is the unified, both-signals demo
described in "Demo" below.

Source layout:

```
DriveSense/
├── docs/            # brief, rubric alignment, dataset research, responsible AI, status tracker
├── data/            # raw/ and processed/ (gitignored; see Setup)
├── notebooks/        # Colab-first EDA / training / demo notebooks
├── src/drivesense/   # installable package: data, features, models, inference, utils, demo.py
├── scripts/          # standalone acquisition / training / eval / demo scripts
├── models/           # saved model artifacts (gitignored; regenerate via training scripts)
├── experiments/      # experiment tracking logs (gitignored)
├── reports/           # rendered evaluation reports (confusion matrices, tables) — committed
├── demo.py           # `python demo.py --source 0` entry point (repo root)
└── tests/            # unit tests
```

## Models / approaches tested

**Distraction (State Farm, real data, 22,424 images, 26 drivers):** three
models trained and evaluated so far, all on an identical driver-independent
split (17 train / 3 val / 6 test drivers, seed 42, zero subject or
image-path overlap):

1. `SimpleCNN` — small from-scratch 3-block CNN (the required baseline).
2. `TransferLearningCNN` (MobileNetV3-Small, frozen ImageNet backbone).
3. `TransferLearningCNN`, fine-tuned (unfrozen backbone, lower learning
   rate) — the current best model.

**Drowsiness (UTA-RLDD):** engineered-feature baseline and temporal
GRU/1D-CNN architectures exist (`src/drivesense/models/`) but have not been
trained on real data yet — real acquisition is in progress (a small
selective per-subject sample locally; the full subject set is intended for
Google Colab; see `docs/Master_Plan_Status.md`).

Full details, hyperparameters, and per-run records: `docs/Master_Plan_Status.md`
row 11 and `docs/LEARNING_LOG.md`.

## Deep learning component (required detail for the Deep Learning rubric)

- **Task:** single-frame 10-class image classification (distraction category).
- **Architecture:** `TransferLearningCNN` — a MobileNetV3-Small backbone
  pretrained on ImageNet (real pretrained weights, `torchvision`'s
  `MobileNet_V3_Small_Weights.DEFAULT`), classifier head replaced with a
  single `Linear(feature_dim, 10)` layer. The best model additionally
  unfreezes and fine-tunes the whole backbone (not just the head) at a low
  learning rate. A from-scratch `SimpleCNN` baseline (3 conv blocks) is
  trained and evaluated alongside it as the required non-transfer-learning
  comparison point (`src/drivesense/models/cnn.py`).
- **Input:** RGB image resized to 96×96, normalized with ImageNet mean/std.
- **Output:** 10 logits → softmax → predicted class + per-class probability.
- **Loss function:** `torch.nn.CrossEntropyLoss` (multi-class, single-label).
- **Optimizer:** `torch.optim.Adam`, fine-tuned run at `lr=1e-4`.
- **Augmentation (training only):** `RandomRotation(8°)`,
  `ColorJitter(brightness=0.2, contrast=0.2)`, and — for the
  robustness-augmented variant — randomly applied (p=0.3 each) Gaussian
  blur, Gaussian noise, and JPEG quality-10 recompression, to close the
  real robustness gap found in row 17 of `docs/Master_Plan_Status.md`.
  **Deliberately no horizontal flip:** classes c1-c4 are left/right-hand
  specific (see `src/drivesense/data/state_farm.py`), so flipping would
  silently swap their true labels.
- **Train/validation/test methodology:** driver-independent split (17
  train / 3 val / 6 test drivers by `p###` subject ID, seed 42, zero
  image-path or subject overlap, verified by
  `assert_no_subject_leakage` — `src/drivesense/data/splits.py`), not a
  random per-image split, since the same driver appears in many images.
- **Evaluation metrics:** accuracy, balanced accuracy, macro F1, weighted
  F1, per-class precision/recall/F1/support, and a full confusion matrix —
  computed once per model on the held-out test set only (never used for
  training or model selection). See "Evaluation metrics and results" below
  and `reports/`.
- **Device:** automatic — `torch.device("cuda" if torch.cuda.is_available()
  else "cpu")` (`src/drivesense/demo.py`); this project has run entirely on
  CPU so far (see "Environment notes" in `docs/Master_Plan_Status.md`).

**Drowsiness deep learning (temporal):** `DrowsinessGRU` and
`DrowsinessTemporalCNN` (`src/drivesense/models/temporal.py`) are
implemented and unit/shape-tested (forward pass + one optimizer step, both
directions of the GRU, variable sequence lengths) but **not yet trained on
real data** — training needs real MediaPipe-extracted EAR/MAR/head-pose
windows from UTA-RLDD, which requires Google Colab (see "Known
limitations"). Honestly: **architecture implemented, training pending** —
this is not glossed over as "done". Until that real run happens, the demo's
drowsiness signal comes from a real but non-deep-learning rule-based
fallback (PERCLOS over a lightweight local landmark extractor) — see
"Drowsiness detection" below.

## Drowsiness detection

Two layers, clearly distinguished:

1. **Intended final model (not yet trained):** `DrowsinessGRU` /
   `DrowsinessTemporalCNN` over a time-windowed sequence of real MediaPipe
   EAR/MAR/head-pose features (`src/drivesense/data/drowsiness.py`,
   `make_time_windows` — resamples by wall-clock time, not raw frame count,
   since UTA-RLDD's frame rates vary 12-30fps across subjects). Blocked
   locally: constructing MediaPipe's `FaceLandmarker` graph gets OOM-killed
   on this development machine (confirmed repeatedly, ~400-600 MB free RAM
   under normal desktop load — see `docs/LEARNING_LOG.md`); training is
   deferred to `notebooks/02_Drowsiness_Colab_Pipeline.ipynb` in Google
   Colab, which has not yet been executed for real.
2. **Real, working local fallback (used by `scripts/demo.py` today):**
   `HaarCascadeLandmarkExtractor` (`src/drivesense/features/landmarks.py`) —
   `cv2.CascadeClassifier`-based face/eye detection, far lighter-weight than
   MediaPipe's task graph — feeding
   `PerclosDrowsinessScorer` (`src/drivesense/inference/drowsiness.py`), a
   rolling **PERCLOS** (PERcentage of eye CLOSure) score, a real, established
   drowsiness metric from the fatigue-detection literature (Wierwille &
   Ellsworth, 1994), not a placeholder. **This is a genuine approximation,
   not equivalent to true landmark-based EAR/MAR:** Haar cascades return
   bounding boxes, not the 6 contour points true EAR needs, so eye state is
   a coarse 3-level open/half/closed proxy rather than a continuous ratio,
   and mouth/yawn detection is unavailable (no maintained mouth cascade
   ships with this project's OpenCV build). **Verified on real video:**
   sampled every 3rd frame over the first ~10s of two real local UTA-RLDD
   clips for the same subject — PERCLOS 0.03 on the file labeled "alert" vs.
   0.28 on the one labeled "drowsy" — a real, directionally-correct signal
   from real video, not a fabricated number.

## Risk fusion

`src/drivesense/inference/risk_engine.py` — deliberately rule-based, not a
third trained model (Individual Project Brief §Q12.3: interpretability
matters for a safety-alert system, and a trained fusion model would need
real fused-label ground truth this project does not have). Two layers exist:

- `TemporalRiskEngine` (original): discrete NORMAL/CAUTION/WARNING from a
  sliding window of per-frame labels — still used by
  `scripts/demo_distraction.py`.
- `DriverAwarenessEngine` (added for `scripts/demo.py`): continuous scores,
  exponentially smoothed (`smoothing_alpha=0.3`) so one noisy frame cannot
  flip the displayed risk level, combined as
  `risk_score = 0.55·drowsiness + 0.35·distraction + 0.10·temporal_penalty`
  (weights are a documented, interpretable design choice — no drowsiness
  ground truth exists locally to calibrate them against; see the class
  docstring), then mapped to `LOW ≤ 0.25 < MEDIUM ≤ 0.5 < HIGH ≤ 0.75 <
  CRITICAL`. `awareness = 1 - risk_score`. Output schema:

  ```python
  {
      "drowsiness": 0.28,    # or None if no face/signal this session
      "distraction": 0.12,
      "awareness": 0.83,
      "risk_score": 0.17,
      "risk_level": "LOW",   # LOW / MEDIUM / HIGH / CRITICAL
  }
  ```

## Demo

```bash
python demo.py --source 0                    # webcam (falls back cleanly if none is available)
python demo.py --source path/to/video.mp4    # any video file
python demo.py --source path/to/image.jpg    # a single image
python -m drivesense.demo --source 0          # equivalent, as a package module
```

`opencv-python-headless` has no GUI backend, so the demo never calls
`cv2.imshow` — it writes an annotated output video/image to
`experiments/demo_output_live.mp4` (or `--out <path>`) instead, the same
pattern `scripts/demo_distraction.py` already used. On-screen overlay:
predicted distraction class + confidence, Distraction %, Drowsiness %,
Awareness %, Risk level, and FPS. `--max-frames N` stops after N frames
(useful for automated smoke tests); `--no-drowsiness` disables the fallback
extractor; `--checkpoint <path>` picks a specific distraction model instead
of the default (see below).

**Genuinely tested this session, not just written:** a real live webcam
(`/dev/video0` on this machine — output frame visually verified), two real
UTA-RLDD video files, and a real State Farm image (correctly predicted
`safe_driving`, 99.8% confidence, drowsiness honestly reported unavailable
since no frontal face exists in a side-angle dashcam still). A previously
missing webcam or unreadable video path raises a clear error instead of
crashing (`tests/test_demo.py`).

**Default distraction checkpoint:** `transfer_20260921_154816` (the
robustness-augmented fine-tuned run) — chosen over the slightly
higher-clean-accuracy `transfer_20260917_194003` because real before/after
re-evaluation (row 17, `docs/Master_Plan_Status.md`) shows it trades a small
clean-set cost (macro F1 0.697 → 0.680) for a large robustness gain under
blur/noise/JPEG degradation, the more realistic failure mode for a camera
mounted in a moving car.

**Important, honest limitation surfaced by this testing:** the distraction
model was trained only on State Farm's *side-angle dashcam* images. Run on
*frontal* footage (a laptop webcam, or UTA-RLDD), its distraction
predictions are genuinely out-of-domain/unreliable — `scripts/demo.py`
prints an explicit warning about this when the source is a webcam. The
drowsiness fallback, conversely, needs a frontal face and will correctly
report "unavailable" on side-angle dashcam stills. No public dataset
combines both camera angles for the same driver, so a single demo run
currently gets one meaningful signal at a time depending on camera angle —
a real, stated architecture-level limitation, not a bug.

## Final model and justification

**Distraction:** the fine-tuned transfer-learning model (unfrozen
MobileNetV3-Small backbone) is the current best, by a wide margin, on the
untouched test set — see results below. It is *not* the model validation
alone would have selected between the frozen and from-scratch runs, which
is itself a documented, honest finding (see `docs/LEARNING_LOG.md`,
"validation-set winner isn't always the better generalizer") rather than a
result picked after peeking at test scores.

**Drowsiness / overall system:** not yet determined — real training has not
happened yet, so there is no final model or fused system to justify.

## Evaluation metrics and results

**Distraction, held-out test set (4,978 images, 6 drivers never seen during
training or model selection), evaluated once per model:**

| Model | Val macro F1 | Test accuracy | Test macro F1 | Test weighted F1 | Inference (CPU) |
|---|---|---|---|---|---|
| SimpleCNN (baseline) | 0.384 | 0.333 | 0.247 | 0.248 | 153 FPS |
| Transfer learning, frozen backbone | 0.310 | 0.353 | 0.355 | 0.356 | 55 FPS* |
| **Transfer learning, fine-tuned (best)** | **0.578** | **0.708** | **0.697** | **0.705** | 170 FPS* |

\* the two transfer-learning FPS numbers were measured under different
background system load and should not be read as "fine-tuning made
inference faster" — both share the identical forward pass.

**Real, verified findings, not fabricated:**
- The baseline (SimpleCNN) never once predicts classes c2/c4 ("talking on
  phone", right/left) on any test image — a total class collapse.
- Fine-tuning the backbone (same architecture as the frozen run) roughly
  doubled test macro F1 (0.355 → 0.697) — the training regime mattered more
  than the architecture choice here.
- The fine-tuned model's weakest class remains c9 ("talking to passenger",
  F1 0.361). Direct visual inspection of real misclassified images (not
  just the confusion matrix) shows a consistent pattern: correctly
  classified examples show part of the passenger visible in frame;
  misclassifications show only the driver, with the model falling back to
  hand-position cues that resemble other classes.
- **Robustness (fine-tuned model, same test set, real re-evaluation):**
  brightness changes (±50%) cost ~7-8 macro-F1 points; Gaussian blur,
  Gaussian noise, and JPEG quality-10 recompression each cost ~28-35
  points — roughly halving performance. This is a genuine limitation for a
  real dashcam-deployment scenario and is not yet mitigated (e.g. via
  blur/noise augmentation).

Full per-class metrics, confusion matrices, and run records:
`experiments/distraction/*.json` (gitignored locally; summarized in
`docs/Master_Plan_Status.md` and `docs/LEARNING_LOG.md`, which are
committed) — and, rendered as a readable table + confusion-matrix image,
`reports/transfer_20260921_154816_report.md` (the default demo model) and
`reports/transfer_20260917_194003_report.md` (the original, higher-clean-
accuracy run), both regenerated from the real JSON via
`python scripts/generate_eval_report.py --run-id <run_id>` — that script
never computes new numbers, only formats existing real ones.

**Robustness-augmented run, real evaluation
(`transfer_20260921_154816`, the current demo default — see "Demo"):**
accuracy 0.691, balanced accuracy 0.687, macro F1 0.680, weighted F1 0.688.

## Installation instructions

```bash
git clone https://github.com/echo05w/drivesense-driver-monitoring.git
cd drivesense-driver-monitoring
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .   # makes the `drivesense` package importable for tests/notebooks
```

Environment as verified this session: Python 3.14.7, `torch==2.14.0+cpu`,
`torchvision==0.29.0+cpu`, `opencv-python-headless==4.14.0.94` (pinned
`<5` in `requirements.txt` — OpenCV 5.0 removed the `CascadeClassifier`
Python binding the local drowsiness fallback depends on; see
`docs/LEARNING_LOG.md`, "An unpinned opencv-python-headless..."). If
`import cv2` succeeds but every attribute access fails, that's a broken/
partial install (e.g. a leftover empty `cv2/` directory from a failed
manual `pip install`), not a code bug — `pip list | grep opencv` should
show a real installed version; if not, remove any stray `cv2*`/`opencv*`
directories under `.venv/lib/python*/site-packages/` and reinstall.

Run the test suite (geometry/EAR-MAR math, subject-independent splitting,
both risk-fusion engines, drowsiness PERCLOS scoring, the Haar-cascade
fallback's pure logic, the demo's error-handling paths, State Farm metadata
loading, dataset/model shape checks — 73/73 passing as of the last verified
run):

```bash
pytest
```

## Dataset acquisition

Requires a Kaggle account and API token (`~/.kaggle/kaggle.json` or
`KAGGLE_API_TOKEN` env var) and, for State Farm, accepting the competition
rules once at
https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/rules
while logged in as that account (the API refuses downloads otherwise, even
though it will still let you list files — see `docs/LEARNING_LOG.md`).

```bash
python scripts/download_data.py --dataset distraction   # ~4GB, full archive
```

UTA-RLDD (drowsiness) is **not** acquired as one archive — this Kaggle
mirror is 96.6GB across 48 subjects, individually downloadable per subject
(see `docs/Master_Plan_Status.md` storage-strategy note). Pick a small
sample for local pipeline work; the full subject set is intended for Google
Colab, not this repository's default workflow:

```bash
python scripts/uta_rldd_pipeline.py plan --n-subjects 6     # see what it would download
python scripts/uta_rldd_pipeline.py fetch --subjects 45 17 31 16 27 44
python scripts/uta_rldd_pipeline.py validate                # opens each video, reports real fps/resolution
```

## Training / fine-tuning instructions

Distraction (State Farm), after extracting the archive to `data/raw/distraction/`:

```bash
python scripts/train_distraction.py eda                                   # real EDA + integrity check + plots
python scripts/train_distraction.py smoke-train                           # tiny end-to-end pipeline check
python scripts/train_distraction.py train --model simple_cnn              # baseline
python scripts/train_distraction.py train --model transfer                # frozen backbone
python scripts/train_distraction.py train --model transfer --unfreeze --lr 1e-4   # fine-tuned (best so far)
python scripts/train_distraction.py evaluate --run-id <run_id>            # once-only held-out test evaluation
python scripts/robustness_distraction.py --run-id <run_id>                # brightness/blur/noise/JPEG re-evaluation
```

Drowsiness training is not yet implemented against real data — architecture
exists (`src/drivesense/models/temporal.py`) but needs a working landmark
extractor run and real UTA-RLDD data first (see `docs/Master_Plan_Status.md`).

## Demo and inference run instructions

See "Demo" above for the full unified command (`python demo.py --source
0|<video>|<image>`) — real, tested locally on webcam/video/image, no Colab
required. A Colab-hosted browser-webcam notebook remains a possible future
addition but is no longer required for a working demo.

For distraction-only inference with the original categorical risk engine:

```bash
python scripts/demo_distraction.py --run-id transfer_20260921_154816 --n-frames 90
```

## Example input and output

Real output from `python demo.py --source data/raw/distraction/imgs/train/c0/img_100026.jpg`
(a real State Farm test image, correctly predicted):

```json
{
  "status": "ok",
  "distraction_model": ".../models/distraction/transfer_20260921_154816.pt",
  "drowsiness_available": true,
  "final_state": {
    "distraction_class": "safe_driving",
    "distraction_confidence": 0.9981468915939331,
    "distraction_pct": "0%",
    "drowsiness_pct": null,
    "awareness_pct": "100%",
    "risk_score": 0.00065,
    "risk_level": "LOW"
  }
}
```

`drowsiness_pct` is honestly `null` here rather than a fabricated number —
this is a side-angle dashcam still with no frontal face for the fallback
extractor to find (see "Drowsiness detection" and "Known limitations").

## Known limitations

See `docs/Responsible_AI.md` for the fuller draft. Headline limitations,
several confirmed by real testing this session rather than only anticipated:

- **Not a certified medical/safety device.**
- **Distraction/drowsiness camera-angle mismatch (confirmed by real
  testing):** State Farm's distraction images are shot side-angle
  (dashboard/passenger-side camera); UTA-RLDD and a laptop webcam are
  frontal. Running the distraction model on frontal footage gives
  out-of-domain, unreliable predictions; running the drowsiness fallback on
  side-angle stills correctly finds no face at all. No public dataset
  combines both angles for the same driver, so today's demo gets one
  meaningful signal at a time depending on camera position — a real,
  architecture-level limitation, not a bug (`scripts/demo.py` warns about
  this explicitly on webcam sources).
- **Drowsiness is currently a rule-based fallback, not the trained deep
  model.** `PerclosDrowsinessScorer` + `HaarCascadeLandmarkExtractor` give a
  real, verified-on-real-video signal (see "Drowsiness detection"), but a
  coarser approximation than the intended MediaPipe-EAR-based
  `DrowsinessGRU`/`DrowsinessTemporalCNN`, which remains untrained pending a
  real Google Colab run (blocked locally by OOM — see
  `docs/Master_Plan_Status.md` row 12).
- **Robustness:** the default demo checkpoint was retrained with
  blur/noise/JPEG augmentation specifically to close most of this gap (see
  row 17), but real-world conditions beyond those three perturbations are
  untested.
- **Self-reported drowsiness labels** (UTA-RLDD) and **limited
  demographic/eyewear coverage** in both source datasets.
- **Risk-fusion weights are a documented design choice, not fit to data** —
  no real fused ground truth exists locally to calibrate
  `DriverAwarenessEngine`'s weights against.

## Responsible AI considerations

See `docs/Responsible_AI.md`.

## Relationship to other projects

This repository is fully independent from the author's other capstone-track
project, GOV-01 (road damage classification, Field-Based Scenario Track).
See `docs/GOV01_Preserved_Note.md` — GOV-01 is preserved, untouched, and out
of scope for this repository.

## Project status

See `docs/Master_Plan_Status.md` for the live phase-by-phase tracker.
