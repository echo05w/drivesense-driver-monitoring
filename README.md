# DriveSense

Deep learning driver monitoring system for real-time drowsiness and
distraction detection from an ordinary RGB webcam.

**Student:** Hasan Soliyev
**Selected project track:** Track 1 — Individual Project Track
**Project Brief status:** `PENDING_MENTOR_APPROVAL` (see `docs/Individual_Project_Brief.md`)

> This README is a living document. Sections marked *(to be filled in as work
> lands)* are intentionally not yet populated — no results, metrics, or
> screenshots are written here until they are produced by an actual run
> against real data, per the project's academic-integrity commitments (see
> `docs/Rubric_Alignment.md`).

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

*(to be filled in as experiments run)* — planned: engineered-feature
baseline, from-scratch CNN, transfer-learning CNN, temporal
GRU/1D-CNN model. See `docs/Individual_Project_Brief.md` §7 and
`docs/Master_Plan_Status.md` for current status.

## Final model and justification

*(to be filled in after model comparison — not yet determined)*

## Evaluation metrics and results

*(to be filled in after evaluation on held-out, subject-independent data)*

## Installation instructions

```bash
git clone https://github.com/echo05w/drivesense-driver-monitoring.git
cd drivesense-driver-monitoring
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .   # makes the `drivesense` package importable for tests/notebooks
```

Run the test suite (currently covers geometry/EAR-MAR math, subject-independent splitting, and the risk-fusion engine — 16/16 passing as of the last verified run):

```bash
pytest
```

## Dataset acquisition

Requires a Kaggle account and API token (`~/.kaggle/kaggle.json`) — either
locally or, preferably, inside the Colab notebook using your own Kaggle
credentials (see `docs/Dataset_Research.md` for why this is Colab-first):

```bash
python scripts/download_data.py --dataset distraction
python scripts/download_data.py --dataset drowsiness
```

## Training / fine-tuning instructions

*(to be filled in once training scripts exist and have been run — see
`docs/Master_Plan_Status.md` for current phase)*

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
