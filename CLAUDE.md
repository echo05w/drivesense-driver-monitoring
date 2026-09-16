# CLAUDE.md — DriveSense

Instructions for any Claude Code session working in this repository.

## Priority

DriveSense is **Priority 1**. Treat this directory
(`/home/hasan/Projects/DriveSense`) as the active project. The sibling
project **GOV-01** (`~/projects/road-damage-classification`, note the
lowercase `projects/`) is a separate, preserved capstone project — do not
read it as a template, do not write to it, do not touch its git history, and
do not resume active work on it until DriveSense reaches its Definition of
Done (see `docs/Master_Plan_Status.md`) or the user explicitly says to switch
projects. See `docs/GOV01_Preserved_Note.md`.

## Start-of-session checklist (do this before writing any code)

1. `cd /home/hasan/Projects/DriveSense`
2. Read, in this order: `docs/Master_Plan_Status.md` (the phase-by-phase
   tracker — the actual source of truth for "what's done"), then
   `docs/Rubric_Alignment.md`, `docs/Individual_Project_Brief.md`,
   `docs/Dataset_Research.md`, `docs/Responsible_AI.md`, and
   `docs/LEARNING_LOG.md` if present.
   - Note: this project does **not** use separate `PROJECT_STATUS.md`,
     `DECISIONS.md`, or `docs/RUBRIC_CHECKLIST.md` files — their role is
     covered by `docs/Master_Plan_Status.md` and `docs/Rubric_Alignment.md`
     respectively, to avoid maintaining duplicate/forkable status trackers.
     If one of those filenames is referenced by an instruction and doesn't
     exist, that's expected — don't recreate a parallel tracker; use the
     files above.
3. Run `git status` and `git log --oneline` in this repo specifically
   (confirm `git rev-parse --show-toplevel` prints this directory, not
   `/home/hasan`, before trusting the result).
4. Run `scripts/project_health.sh` for a fast combined snapshot (git state,
   test results, key artifacts, latest tracked phase).
5. Treat the filesystem, git history, and actual test output as ground
   truth — never a prior response's claims, and never this project's own
   status docs without spot-checking them against reality first (a status
   doc can be stale or, as happened once in this project's history, written
   slightly ahead of the actual work — see `docs/LEARNING_LOG.md`).

## Working rules

- **Never redo verified, completed work** just because a previous session or
  response was interrupted. Confirm via git/filesystem/tests first; only
  redo what is actually missing or actually broken.
- **Resume from the first incomplete highest-priority phase** in
  `docs/Master_Plan_Status.md`'s table, unless the user says otherwise.
- **Test before marking anything complete.** A model architecture existing
  is not "the model is done" — it must be trained and evaluated on held-out
  data before that phase is marked done. Shape/smoke tests are enough to
  mark *code* as verified-working; they are not evaluation results.
- **Never fabricate** training runs, metrics, dataset access, Colab
  execution, or test results. If something hasn't actually been run, say so
  and mark it accordingly (e.g., "notebook structure ready, not yet
  executed") rather than implying it happened.
- **Update `docs/Master_Plan_Status.md`** (and `docs/Rubric_Alignment.md`
  where relevant, and `docs/LEARNING_LOG.md` for durable lessons) after each
  meaningful phase actually lands — not before.
- **Continue autonomously** between normal implementation steps without
  asking for confirmation, per standing project instructions — but still use
  judgment: stop and flag genuine external blockers (missing credentials,
  license/access gates, anything destructive or irreversible) rather than
  working around them silently.
- **This repository must never contain**: raw dataset files, trained model
  weight files, Kaggle/API credentials, `~/.cache` contents, Claude/MCP
  session logs, cliphist or weather-cache data, or any other file unrelated
  to DriveSense. Check `.gitignore` covers these; check `git status`/
  `git ls-files` before every commit, not just once.
- **No local GPU** in this environment — actual training happens in Google
  Colab. Local work is scaffolding, docs, lightweight EDA/tests, and code
  that will run for real in Colab.

## Definition of Done

See the full checklist in `docs/Master_Plan_Status.md` and the rubric
criteria in `docs/Rubric_Alignment.md`. Both models trained and evaluated on
subject-independent held-out data, a working Colab demo, complete README,
Responsible AI section grounded in real EDA findings, and a final rubric
self-audit.
