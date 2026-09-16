# DriveSense Master Plan & Status Tracker

Single source of truth for "what's done, what's next" across sessions. Update
the Status column as work actually lands — never mark something Done before
it is verifiably true in the repository. This tracks the 31-phase autonomous
plan the project is being carried through.

| # | Phase | Status | Notes / Artifact |
|---|---|---|---|
| 1 | Project/rubric analysis | **Done** | `docs/Rubric_Alignment.md` |
| 2 | Individual Project Brief | **Drafted, PENDING_MENTOR_APPROVAL** | `docs/Individual_Project_Brief.md` |
| 3 | Dataset research and acquisition | **Research done; notebook ready; execution blocked on Kaggle credentials** | `docs/Dataset_Research.md`, `scripts/download_data.py`, `notebooks/01_Data_Acquisition_and_EDA.ipynb` §0-1 — needs a real `kaggle.json` (local or Colab) to actually pull data |
| 4 | EDA | **Notebook structure ready; not yet executed** | `notebooks/01_Data_Acquisition_and_EDA.ipynb` §2-4 — class balance, subject counts, sample grids, corruption checks scaffolded; drowsiness section intentionally raises `NotImplementedError` until real file layout is confirmed (no guessed schema) |
| 5 | Preprocessing | **Mostly implemented, partially verified** | `src/drivesense/features/landmarks.py` (EAR/MAR/head-pose-proxy extraction against the verified-correct MediaPipe Tasks API) + `src/drivesense/data/datasets.py` (`DistractionImageDataset`, `make_feature_windows`/`FeatureWindowDataset` for temporal windowing). Pure-logic parts fully unit-tested (12/12 passing: `tests/test_landmarks.py` 5, `tests/test_datasets.py` 7 — includes a synthetic-data test proving windows never cross a subject/group boundary). Full `MediaPipeLandmarkExtractor.extract()` graph construction could NOT be run to completion locally (OOM-killed, exit 137, due to desktop memory pressure on this machine — see `docs/LEARNING_LOG.md`); needs verification in Colab or under more local headroom via `scripts/verify_landmarks_extractor.py`. NOT marked done overall — real-data run still pending #3. |
| 6 | Subject-independent train/val/test splitting | **Implemented & verified** | `src/drivesense/data/splits.py`, `assert_no_subject_leakage`; 4/4 tests passing on synthetic data (`tests/test_splits.py`). Real dataset run still depends on #3. |
| 7 | Baseline | **Code implemented, not yet run on real data** | `src/drivesense/models/baseline.py` (logistic regression + random forest pipelines). Not trained — depends on #3. |
| 8 | Deep learning models | **Architectures implemented & smoke-tested, NOT trained** | `src/drivesense/models/cnn.py` (`SimpleCNN`); forward pass + single optimizer step verified on synthetic tensors (`tests/test_models_cnn.py`, 3/3 passing). No training/evaluation has happened — do not treat as a completed model. |
| 9 | Transfer learning | **Architecture implemented & manually verified to construct/forward, NOT trained** | `src/drivesense/models/cnn.py` (`TransferLearningCNN`, MobileNetV3-Small backbone). Manually verified: downloads real pretrained ImageNet weights, forward pass produces correct output shape, `unfreeze_backbone()` correctly increases trainable-parameter count (932,778 vs. 5,770 frozen). Excluded from the default fast test suite (downloads weights over the network) — verified once interactively, not on every `pytest` run. Not fine-tuned on any real data yet. |
| 10 | Temporal modeling | **Architectures implemented & smoke-tested, NOT trained** | `src/drivesense/models/temporal.py` (`DrowsinessGRU`, `DrowsinessTemporalCNN`); forward pass + single optimizer step verified for both, including bidirectional GRU and variable sequence lengths (`tests/test_models_temporal.py`, 6/6 passing). No training/evaluation has happened. |
| 11 | Distraction model | Not started (training) | Architecture ready (#8/#9); needs dataset (#3) before real training can start |
| 12 | Drowsiness model | Not started (training) | Architecture ready (#10); needs dataset (#3) and a working landmark extractor (#5) before real training can start |
| 13 | Experiments | Not started | |
| 14 | Experiment tracking | Not started | Plan: MLflow (local, file-based) or equivalent structured run log |
| 15 | Evaluation on unseen data | Not started | |
| 16 | Error analysis | Not started | |
| 17 | Robustness testing | Not started | |
| 18 | Saved model artifacts | Not started | |
| 19 | Temporal risk engine | Not started | Fuses both model outputs; not a trained model itself |
| 20 | Real-time webcam/video inference | Not started | |
| 21 | Classroom demo | Not started | Colab-first per rubric |
| 22 | Tests | **37/37 passing (verified)** | `tests/test_geometry.py`, `tests/test_splits.py`, `tests/test_risk_engine.py`, `tests/test_models_cnn.py`, `tests/test_models_temporal.py`, `tests/test_landmarks.py`, `tests/test_datasets.py` — run via `.venv` + `pytest`; all synthetic fixtures/shape checks (not real dataset results) |
| 23 | Google Colab workflow/training where available | Planned | Required given no local GPU — see environment note below |
| 24 | Documentation | In progress | This tracker + brief + rubric alignment + dataset research + responsible AI stub |
| 25 | README | Initial draft done | `README.md` — will be expanded as results land |
| 26 | Responsible AI | Initial draft done | `docs/Responsible_AI.md`, to be revised with EDA evidence |
| 27 | Git/GitHub | **Done (verified)** | Independent local repo initialized (branch `main`), initial commit `d12d194`, pushed to `https://github.com/echo05w/drivesense-driver-monitoring` (private). Verified separate from GOV-01's remote and history; GOV-01 re-checked clean/unchanged after this work. |
| 28 | Defense guide | Not started | |
| 29 | Presentation preparation | Not started | |
| 30 | Final rubric audit | Not started | |
| 31 | Clean reproducibility verification | Not started | |

## Environment notes (recorded 2026-09-16)

- Local machine: no GPU (`nvidia-smi` not present), Python 3.14.7, Arch Linux
  (externally-managed system Python — project uses its own `.venv`).
  `torch`, `torchvision`, `opencv-python-headless`, and `mediapipe` are now
  installed in `.venv` and verified importable/runnable (CNN/temporal model
  forward+backward passes confirmed; MediaPipe Tasks API confirmed present).
  Still **not** suitable for actual full-dataset model training at
  reasonable speed (no GPU), and memory-constrained for MediaPipe's graph
  construction alongside normal desktop usage (see blocker #4 below).
- `gh` CLI is authenticated as `echo05w` (same GitHub account as GOV-01),
  internet access confirmed.
- No Kaggle CLI/credentials configured locally (`~/.kaggle/kaggle.json`
  absent) — dataset downloads must happen via Colab (student's own Kaggle
  auth) or after the student adds local credentials.

## Known blockers (do not let these stop other work)

1. **Dataset download** needs Kaggle credentials not present in this
   environment. Mitigation: acquisition script + Colab notebook are written
   to work as soon as credentials exist; all non-data-dependent work
   continues in parallel.
2. **Mentor approval** of the Project Brief has not happened and must never
   be fabricated. Implementation proceeds on the *proposed* scope, but the
   brief stays marked `PENDING_MENTOR_APPROVAL` until a real mentor fills in
   Section 14.
3. **No local GPU** — actual training runs must happen in Google Colab;
   local work is limited to CPU-feasible tasks (code, light EDA on small
   samples, unit tests).
4. **Local memory pressure blocks the MediaPipe FaceLandmarker graph.**
   Verified 2026-09-16: constructing `MediaPipeLandmarkExtractor` was
   OS-killed (exit 137) with `free -h` showing ~490 MB free RAM, consumed by
   the user's own desktop session (multiple Chrome renderer processes,
   confirmed via `ps aux --sort=-%mem` — not a leak in this project's code,
   and not something this project should try to kill/manage). Mitigation:
   the pure-logic parts of the feature-extraction code are unit-tested
   separately (`tests/test_landmarks.py`); full extractor verification is
   deferred to Google Colab or a lower-memory-pressure moment, via
   `scripts/verify_landmarks_extractor.py`. Does not block model
   architecture work, dataset research, or documentation.

## Next incomplete highest-priority task

As of this note: **Phase 3 (finish dataset acquisition)** — the acquisition
script (`scripts/download_data.py`) and Colab notebook
(`notebooks/01_Data_Acquisition_and_EDA.ipynb`) are ready; the remaining work
is either (a) placing a real `kaggle.json` in this environment, or (b)
running that notebook in Colab with the student's own Kaggle login — both
are genuine external dependencies (a Kaggle account/token only the student
can provide), not something further local scaffolding can resolve. All
architecture-level work not blocked on real data (models, splitting, risk
engine, feature-extraction code) has been completed and verified in the
meantime — see phases 5–10, 19, 22 above.

## Recovery note (2026-09-16, session interruption)

A mid-session API disconnect occurred right after Phases 1–2 (rubric
analysis, brief) and the initial docs (dataset research, GOV-01 note,
Responsible AI stub) were written, but **before** README/.gitignore/LICENSE/
requirements.txt/src-scaffolding/git-init/GitHub-repo-creation actually ran.
This tracker had incorrectly pre-marked Phase 27 (Git/GitHub) as "Done" —
that was wrong and has been corrected above. Lesson encoded here so a future
session doesn't trust status text over `git status`/`find` output: **always
verify a phase against the actual filesystem/git state before trusting a
status table**, including one this project itself wrote earlier in the same
session.
