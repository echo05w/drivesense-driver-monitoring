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
| 5 | Preprocessing | Not started | Depends on #3 |
| 6 | Subject-independent train/val/test splitting | Not started | Design decided in Brief §7; implementation depends on #3 |
| 7 | Baseline | Not started | Design decided in Brief §7 |
| 8 | Deep learning models | Not started | |
| 9 | Transfer learning | Not started | |
| 10 | Temporal modeling | Not started | For drowsiness sequence model |
| 11 | Distraction model | Not started | |
| 12 | Drowsiness model | Not started | |
| 13 | Experiments | Not started | |
| 14 | Experiment tracking | Not started | Plan: MLflow (local, file-based) or equivalent structured run log |
| 15 | Evaluation on unseen data | Not started | |
| 16 | Error analysis | Not started | |
| 17 | Robustness testing | Not started | |
| 18 | Saved model artifacts | Not started | |
| 19 | Temporal risk engine | Not started | Fuses both model outputs; not a trained model itself |
| 20 | Real-time webcam/video inference | Not started | |
| 21 | Classroom demo | Not started | Colab-first per rubric |
| 22 | Tests | **16/16 passing (verified)** | `tests/test_geometry.py`, `tests/test_splits.py`, `tests/test_risk_engine.py` — run via `.venv` + `pytest`; covers EAR/MAR math, subject-independent split leakage checks, and the risk-fusion state machine, all with synthetic fixtures (not real dataset results) |
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

- Local machine: no GPU (`nvidia-smi` not present), no `torch`/`opencv`/`mediapipe`
  installed, Python 3.14.7. Suitable for scaffolding, docs, lightweight EDA
  (once core packages are installed), and writing training/inference code —
  **not** suitable for actual model training at reasonable speed.
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

## Next incomplete highest-priority task

As of this note: **Phase 3 (finish dataset acquisition)** — the acquisition
script (`scripts/download_data.py`) and docs are ready; the remaining work is
either (a) placing a real `kaggle.json` in this environment, or (b) building
`notebooks/01_Data_Acquisition.ipynb` to run the same acquisition from Colab
with the student's own Kaggle login, so Phase 4 (EDA) can start on real data.

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
