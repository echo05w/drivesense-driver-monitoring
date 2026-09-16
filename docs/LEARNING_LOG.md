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
