# DriveSense Master Plan & Status Tracker

Single source of truth for "what's done, what's next" across sessions. Update
the Status column as work actually lands — never mark something Done before
it is verifiably true in the repository. This tracks the 31-phase autonomous
plan the project is being carried through.

| # | Phase | Status | Notes / Artifact |
|---|---|---|---|
| 1 | Project/rubric analysis | **Done** | `docs/Rubric_Alignment.md` |
| 2 | Individual Project Brief | **Drafted, PENDING_MENTOR_APPROVAL** | `docs/Individual_Project_Brief.md` |
| 3 | Dataset research and acquisition | **State Farm: still BLOCKED_EXTERNAL after a reported rules-acceptance (re-verified, same error). UTA-RLDD: real download resumed and in progress (resumed a second time after a network timeout).** | After being told rules acceptance was complete, re-tested: same exact `RulesAcceptanceRequired` error as before, byte-for-byte. To rule out a token/scope problem, sanity-checked the same token against an always-open competition (`titanic`) — download succeeded immediately, proving the token itself is valid and download-capable in general. This isolates the problem specifically to State Farm's rules gate not being satisfied for the Kaggle account behind this token — likely causes: the acceptance happened on a different Kaggle account than the one that generated this token, or the accept step wasn't fully completed (e.g. viewed the rules page without clicking through, or a required "Join Competition" step wasn't reached). **UTA-RLDD:** progressed to 4.6 GB/89.2 GB, then hit a transient `storage.googleapis.com` read-timeout; resumed correctly from 4.66 GB (not from 0) per the user's explicit instruction not to restart from zero. Still downloading, not yet complete or integrity-checked. |
| 4 | EDA | **Notebook structure ready; not yet executed** | `notebooks/01_Data_Acquisition_and_EDA.ipynb` §2-4 — class balance, subject counts, sample grids, corruption checks scaffolded; drowsiness section intentionally raises `NotImplementedError` until real file layout is confirmed (no guessed schema) |
| 5 | Preprocessing | **Mostly implemented, partially verified** | `src/drivesense/features/landmarks.py` (EAR/MAR/head-pose-proxy extraction against the verified-correct MediaPipe Tasks API) + `src/drivesense/data/datasets.py` (`DistractionImageDataset`, `make_feature_windows`/`FeatureWindowDataset` for temporal windowing). Pure-logic parts fully unit-tested (12/12 passing: `tests/test_landmarks.py` 5, `tests/test_datasets.py` 7 — includes a synthetic-data test proving windows never cross a subject/group boundary). Full `MediaPipeLandmarkExtractor.extract()` graph construction could NOT be run to completion locally (OOM-killed, exit 137, due to desktop memory pressure on this machine — see `docs/LEARNING_LOG.md`); needs verification in Colab or under more local headroom via `scripts/verify_landmarks_extractor.py`. NOT marked done overall — real-data run still pending #3. |
| 6 | Subject-independent train/val/test splitting | **Implemented & verified** | `src/drivesense/data/splits.py`, `assert_no_subject_leakage`; 4/4 tests passing on synthetic data (`tests/test_splits.py`). Real dataset run still depends on #3. |
| 7 | Baseline | **Code implemented, not yet run on real data** | `src/drivesense/models/baseline.py` (logistic regression + random forest pipelines). Not trained — depends on #3. |
| 8 | Deep learning models | **SimpleCNN: REAL TRAINED on real State Farm data (see #11). Transfer learning: architecture verified, not yet trained.** | `src/drivesense/models/cnn.py` (`SimpleCNN`). |
| 9 | Transfer learning | **Architecture implemented & manually verified to construct/forward, NOT trained** | `src/drivesense/models/cnn.py` (`TransferLearningCNN`, MobileNetV3-Small backbone). Manually verified: downloads real pretrained ImageNet weights, forward pass produces correct output shape, `unfreeze_backbone()` correctly increases trainable-parameter count (932,778 vs. 5,770 frozen). Excluded from the default fast test suite (downloads weights over the network) — verified once interactively, not on every `pytest` run. Not fine-tuned on any real data yet. |
| 10 | Temporal modeling | **Architectures implemented & smoke-tested, NOT trained** | `src/drivesense/models/temporal.py` (`DrowsinessGRU`, `DrowsinessTemporalCNN`); forward pass + single optimizer step verified for both, including bidirectional GRU and variable sequence lengths (`tests/test_models_temporal.py`, 6/6 passing). No training/evaluation has happened. |
| 11 | Distraction model | **Experiment A (SimpleCNN baseline) REAL TRAINED & test-evaluated once.** Experiment B (transfer learning) not yet run. | Run `simple_cnn_20260917_100118` (`experiments/distraction/simple_cnn_20260917_100118.json` + `_test_eval.json`, git commit `c1453d8`): real 22,424-image State Farm dataset, subject-independent split (17 train / 3 val / 6 test drivers, zero subject or image-path overlap, assertions passed), image_size=96, batch=32, Adam lr=1e-3, no horizontal flip (c1-c4 are left/right-specific). Early-stopped after 5 epochs (patience=3); best checkpoint from epoch 2 (val macro F1 0.384) — epochs 3-5 overfit/destabilized (val loss rose to 3.5-4.5 while train loss kept falling). **Untouched test set (4,978 images, 6 held-out drivers), evaluated once:** accuracy 0.333, balanced accuracy 0.335, macro F1 0.247, weighted F1 0.248, inference 153 FPS (CPU). **Real error-analysis finding:** the model never once predicts c2 (talking phone - right) or c4 (talking phone - left) on any test image — total collapse on those two classes — while over-predicting c1 (texting-right), c5 (radio), and especially c7 (reaching behind, predicted 1513 times vs. 480 true) as dumping grounds. Plausible cause: those "phone near ear" poses are visually close to several other classes in this shallow architecture's limited feature space, and the best checkpoint (epoch 2) hadn't yet specialized past the more visually-distinct classes before validation loss started diverging. Not yet compared against Experiment B - do not present this as the final model. One data point (epoch 5's logged time) is a system-suspend artifact, not real compute time - see the run record's `notes` field and `docs/LEARNING_LOG.md`. |
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

## Storage strategy (recorded 2026-09-17)

State Farm rules were accepted for real this time (re-verified: the download
call itself succeeds now, not just file listing) and the full competition
archive (~4GB) downloaded successfully to `data/raw/distraction/` via
`scripts/train_distraction.py`'s pipeline.

UTA-RLDD's full-dataset acquisition strategy changed mid-session on the
user's instruction to protect local laptop storage: the original approach
(`kaggle datasets download -d rishab260/uta-reallife-drowsiness-dataset`,
no `-f`) bundles the *entire* mirror into one client-side zip, which turned
out to be **96.6 GB across 48 subjects** (verified via `kaggle datasets files
--format json` - not the ~89.2GB estimate from the in-progress download bar,
which was apparently imprecise) - abandoned at ~13GB partial, deliberately
not resumed further, deliberately not deleted (disk was never actually tight
- 350GB+ free throughout - so there was no need to choose between keeping it
and asking permission to delete it).

Real, verified finding: this Kaggle mirror exposes **145 individually
downloadable files** (3 videos per subject, addressable by exact path, e.g.
`Fold4_part2/Fold4_part2/45/0.mp4`), not just one monolithic archive -
`kaggle datasets download -d <ref> -f <path>` fetches one raw video file
directly. This enables genuine selective-subject acquisition without
weakening subject independence: whole subjects (all their sessions) are
selected as a unit, never partial subjects. New tool:
`scripts/uta_rldd_pipeline.py` (`index` / `plan` / `fetch` / `validate`
subcommands) queries the real file index, picks subjects by total size, and
downloads only those - discovered along the way that Kaggle still wraps even
a single `-f` file in a `<name>.zip` container, handled by unzipping and
verifying the extracted size against the index.

Adopted split: **local laptop** keeps only a small selective subject sample
(a handful of the smallest subjects, a few GB) for pipeline development,
feature-extraction verification, and smoke-scale drowsiness experiments -
never the full 96.6GB. **Google Colab** is where the full-scale subject set
(ideally most/all 48 available subjects) should be acquired and where the
real, reported drowsiness experiments (temporal vs. single-frame comparison)
should run, using the same `uta_rldd_pipeline.py fetch` mechanism with a
larger `--subjects` list and much more available disk/compute. This has not
yet been executed in Colab - no Colab/browser automation is available in
this environment, so it remains an instruction for the next Colab session,
not a claimed execution.

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
is either (a) placing a real `kaggle.json`/`KAGGLE_API_TOKEN` in this
environment, or (b) running that notebook in Colab with the student's own
Kaggle login — both are genuine external dependencies (a Kaggle account/
token only the student can provide), not something further local scaffolding
can resolve. All architecture-level work not blocked on real data (models,
splitting, risk engine, feature-extraction code) has been completed and
verified in the meantime — see phases 5–10, 19, 22 above. **Do not add more
scaffolding/docs while this is blocked** — the honest next step is to wait
for credentials, not generate more architecture around already-complete
architecture.

## Recovery note (2026-09-16, second session — verification of prior session's claims)

This session's start-of-session checklist caught two things the previous
session's status text got ahead of reality on, consistent with the
`LEARNING_LOG.md` lesson about not trusting status docs without spot-checks:

1. `docs/Master_Plan_Status.md`/`docs/Dataset_Research.md` had uncommitted
   changes claiming UTA-RLDD "downloading"/"proceeded successfully" — the
   actual file on disk is a 1.5 GB partial that fails zip validation
   (session was interrupted mid-download). Corrected in both files; the
   partial file itself was kept (not deleted) since it's resumable.
2. The same docs implied Kaggle auth was live; this session verified no
   `KAGGLE_API_TOKEN` env var and no `~/.kaggle/` directory exist here —
   auth was session-scoped and did not carry over.

Also this session: re-ran the full test suite (37/37 pass; one transient
network-timing failure on first run, passed on isolated re-run), re-attempted
`scripts/verify_landmarks_extractor.py` with more free memory than the prior
OOM (902 MiB vs. ~490 MiB free) — still OOM-killed (exit 137), so phase 5's
full-extractor verification remains genuinely blocked locally, not a flake.
Checked for browser/Colab automation availability (not connected in this
environment) and for alternate local credential sources (no secret-manager
CLIs, no stray `kaggle.json` anywhere under `$HOME`) — confirmed there is no
local path around the credential/consent blockers other than the user
providing them.

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
