# Note: GOV-01 Road Damage Classification (Separate, Preserved Project)

This repository (**DriveSense**) is completely separate from the existing
**GOV-01 / road-damage-classification** capstone project. That project is
**not modified, not merged, not reused, and not touched** by any work in this
repository.

## GOV-01 current state (recorded, not altered)

- **Location:** `/home/hasan/projects/road-damage-classification`
- **GitHub remote:** `https://github.com/echo05w/road-damage-classification`
- **Branch:** `main`, up to date with `origin/main`
- **Working tree:** clean, nothing to commit (verified at DriveSense project start)
- **Recent history:** initial scaffold, MIT license, dataset-path bug fix,
  sample-image preview cell (4 commits as of this note)
- **Structure:** `data/`, `docs/` (including its own `GOV-01_Project_Learning_Plan.md`),
  `models/`, `notebooks/`, `src/`, own `requirements.txt`, own `.gitignore`, own `LICENSE`

GOV-01 uses the **Field-Based Scenario Track** (Track 2, GovTech domain,
scenario `GOV-01`), a different track from DriveSense's Individual Project
Track (Track 1). It has its own dataset, environment, and dependency set,
fully independent from DriveSense's.

## Separation guarantees

| Aspect | GOV-01 | DriveSense |
|---|---|---|
| Git repository | `road-damage-classification` (own history) | `DriveSense` (own history) |
| GitHub repository | `echo05w/road-damage-classification` | separate new repository (see README) |
| Local path | `~/projects/road-damage-classification` (lowercase `projects/`) | `~/Projects/DriveSense` (capitalized `Projects/`) |
| Python environment | its own `requirements.txt` / virtualenv | its own `requirements.txt` / virtualenv |
| Datasets | road damage imagery | driver drowsiness/distraction imagery+video |
| Models | road-damage classifier artifacts | drowsiness/distraction classifier artifacts |
| Documentation | its own `docs/` | this `docs/` |

## Status and next action

GOV-01 is **complete-enough-to-pause and preserved as-is**. Per explicit
project-priority instructions, active development returns to GOV-01 only
after DriveSense reaches its Definition of Done, or on explicit instruction
to switch projects. Until then, GOV-01 should only ever be read (e.g., to
check its status), never written to, from within any DriveSense-related
session.
