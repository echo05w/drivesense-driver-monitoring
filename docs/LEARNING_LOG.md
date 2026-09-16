# DriveSense — Learning Log

Durable, cross-session lessons for this project. Append, don't rewrite
history. This is not a duplicate of `docs/Master_Plan_Status.md` (which
tracks phase completion) — this file records *why* something was done a
particular way, or a mistake worth not repeating.

## 2026-09-16 — Status docs can lie; verify against the filesystem first

Early in the project, `docs/Master_Plan_Status.md` briefly marked the
Git/GitHub phase "Done" before `git init` had actually been run in this
directory — the repo was, at that point, just an untracked folder sitting
inside an unrelated pre-existing git repository at `/home/hasan`. Caught by
running `git status` / `git rev-parse --show-toplevel` and comparing against
the doc, not by trusting the doc. **Lesson: at the start of every session,
verify status-tracker claims against `git status`, `git log`, and actual
file existence before building on them — including this project's own
status docs.** `CLAUDE.md` and `scripts/project_health.sh` now bake this
check in.

## 2026-09-16 — This project intentionally does not use PROJECT_STATUS.md / DECISIONS.md / docs/RUBRIC_CHECKLIST.md

Some instructions during setup referenced those filenames. Their role is
already covered here: `docs/Master_Plan_Status.md` (phase-by-phase status)
and `docs/Rubric_Alignment.md` (rubric-criterion mapping/checklist). Rather
than fork a second, easily-inconsistent set of tracker files, this project
keeps one tracker per concern. Future sessions should extend the existing
files rather than create parallel ones with overlapping purpose.

## 2026-09-16 — requirements.txt correctness must be checked by actually installing in a clean venv

`torchvision` was listed in `requirements.txt` from the start, but was not
actually installed in the local `.venv` in one pass — this was only caught
because `TransferLearningCNN` was actually imported and run (raised
`ModuleNotFoundError`), not by reading the requirements file. **Lesson:
"it's in requirements.txt" is not evidence a module works — run the actual
import/instantiation.**

## 2026-09-16 — MediaPipe's API changed; verify against the installed version, not memory

The installed `mediapipe==1.0.1` does not expose the older
`mp.solutions.face_mesh` API that most existing tutorials/training data
assume — only the newer Tasks API (`mediapipe.tasks.python.vision.FaceLandmarker`,
which needs a separately downloaded `.task` model bundle). This was caught by
running `dir(mediapipe)` / `dir(mediapipe.tasks.python.vision)` directly
before writing `src/drivesense/features/landmarks.py`, rather than writing
code against a remembered API and finding out later. **Lesson: for any
library whose API may have moved on since training data was collected,
inspect the actually-installed version's real surface before writing code
against it.**

## 2026-09-16 — "Download command returned success" is not "the file is valid"; verify the artifact itself

A prior session recorded UTA-RLDD as "proceeded successfully" after issuing
`kaggle datasets download`, but the session was interrupted mid-transfer
before that claim was checked. The next session found a 1.5 GB file that
fails `zipfile` integrity validation, plus Kaggle's own `.kaggle-partial`
resume marker still sitting next to it — i.e. the download never finished,
despite the doc saying it had. **Lesson: a command starting successfully
(no immediate error) is not evidence it finished — for downloads
specifically, validate the resulting artifact (file size against the
expected dataset size, archive integrity, checksum) before writing
"acquired"/"succeeded" anywhere.** Also: don't delete a partial download
just because it's currently invalid — Kaggle's `.kaggle-partial` marker
means the CLI can resume it, so deleting it would discard real progress.

## 2026-09-16 — Constructing MediaPipe's FaceLandmarker graph gets OOM-killed on this desktop machine

`MediaPipeLandmarkExtractor()` construction was killed by the OS (exit 137)
on this machine, with `free -h` showing ~490 MB free RAM at the time —
consumed by the user's own Chrome/desktop session (confirmed via
`ps aux --sort=-%mem`), not a bug in this project's code. This is a genuine
external resource constraint on a shared desktop machine, not something to
route around by e.g. killing the user's browser processes. **Lesson: when a
subprocess is SIGKILL'd (exit 137) rather than raising a Python exception,
suspect OOM first (check `free -h` / `ps aux --sort=-%mem`) before assuming
a code bug — and don't put a test that can OOM-kill the whole interpreter
into the automated suite; verify it manually instead** (see
`scripts/verify_landmarks_extractor.py`).
