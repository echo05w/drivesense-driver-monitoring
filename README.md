# DriveSense

Deep learning driver monitoring system for real-time drowsiness and
distraction detection from an ordinary RGB webcam.

**Student:** Hasan Soliyev
**Selected project track:** Track 1 — Individual Project Track
**Project Brief status:** `PENDING_MENTOR_APPROVAL` (see `docs/Individual_Project_Brief.md`)

> This README is a living document. Results below reflect actual runs
> against real data as they land — sections still marked *(to be filled in)*
> genuinely have not happened yet (currently: drowsiness training, temporal
> risk fusion, and the demo). Nothing here is a projected/planned number;
> per the project's academic-integrity commitments (see
> `docs/Rubric_Alignment.md`), a metric only appears once it has actually
> been produced by a run.

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
                 ┌─────────────────────┐
 webcam / video →│ frame capture        │
                 └─────────┬───────────┘
                           │
              ┌────────────┴─────────────┐
              ▼                          ▼
   ┌─────────────────────┐   ┌───────────────────────────┐
   │ Distraction model    │   │ Face landmarks (MediaPipe) │
   │ (single-frame CNN)   │   │ → EAR / MAR / head pose    │
   └──────────┬───────────┘   └────────────┬───────────────┘
              │                            ▼
              │                 ┌───────────────────────────┐
              │                 │ Drowsiness model            │
              │                 │ (temporal: GRU / 1D-CNN)     │
              │                 └────────────┬───────────────┘
              ▼                              ▼
        ┌────────────────────────────────────────┐
        │   Temporal risk-fusion engine (rules)    │
        │   → Normal / Caution / Warning            │
        └────────────────────┬───────────────────┘
                              ▼
                     on-screen / audible alert
```

Source layout:

```
DriveSense/
├── docs/            # brief, rubric alignment, dataset research, responsible AI, status tracker
├── data/            # raw/ and processed/ (gitignored; see Setup)
├── notebooks/        # Colab-first EDA / training / demo notebooks
├── src/drivesense/   # installable package: data, features, models, inference, utils
├── scripts/          # standalone acquisition / utility scripts
├── models/           # saved model artifacts (gitignored; regenerate via training scripts)
├── experiments/      # experiment tracking logs (gitignored)
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
committed).

## Installation instructions

```bash
git clone https://github.com/echo05w/drivesense-driver-monitoring.git
cd drivesense-driver-monitoring
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .   # makes the `drivesense` package importable for tests/notebooks
```

Run the test suite (geometry/EAR-MAR math, subject-independent splitting,
risk-fusion engine, State Farm metadata loading, dataset/model shape checks —
42/42 passing as of the last verified run):

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

## Demo and inference run instructions (Colab-first)

*(to be filled in once the Colab demo notebook exists — planned:
`notebooks/DriveSense_Demo.ipynb`, browser-webcam capture, no local state
required)*

## Example input and output

*(to be filled in alongside the working demo)*

## Known limitations

See `docs/Responsible_AI.md` for the current draft; will be revised with
dataset-specific evidence after EDA. Headline limitations already known by
design: not a certified medical/safety device; cross-dataset camera mismatch;
self-reported drowsiness labels; limited demographic/eyewear coverage in
source datasets.

## Responsible AI considerations

See `docs/Responsible_AI.md`.

## Relationship to other projects

This repository is fully independent from the author's other capstone-track
project, GOV-01 (road damage classification, Field-Based Scenario Track).
See `docs/GOV01_Preserved_Note.md` — GOV-01 is preserved, untouched, and out
of scope for this repository.

## Project status

See `docs/Master_Plan_Status.md` for the live phase-by-phase tracker.
