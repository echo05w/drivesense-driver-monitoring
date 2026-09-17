# DriveSense — Learning Log

Durable, cross-session lessons for this project. Append, don't rewrite
history. This is not a duplicate of `docs/Master_Plan_Status.md` (which
tracks phase completion) — this file records *why* something was done a
particular way, or a mistake worth not repeating.

## 2026-09-17 — First real trained model: SimpleCNN on State Farm (defense notes)

**What it is:** `SimpleCNN` (`src/drivesense/models/cnn.py`) is a small
from-scratch convolutional network - three conv+batchnorm+ReLU+maxpool
blocks, then global average pooling and a linear classifier - trained for
real on the actual State Farm Distracted Driver Detection dataset (22,424
labeled dashcam images, 26 drivers, 10 distraction classes).

**Why we used it:** the Brief specifies a from-scratch CNN as the honest
baseline distraction model, to be compared against a transfer-learning model
(MobileNetV3-Small) later. A shallow network also trains fast enough to be
feasible on this CPU-only laptop (no GPU) for a first real result.

**How it works here:** `scripts/train_distraction.py` builds the image
list from `driver_imgs_list.csv`, splits by *driver* (not by image) using
`GroupShuffleSplit` so the same person's images can never appear in more
than one of train/val/test, trains with Adam (lr=1e-3), and deliberately
does **not** use horizontal-flip augmentation - classes c1-c4 encode
left/right hand position, so flipping would silently swap their meaning.
Checkpointing keeps only the epoch with the best validation macro F1, and
the held-out test set (6 drivers, never touched during training or model
selection) is evaluated exactly once, at the end.

**What the real result was:** best validation macro F1 was 0.384 (epoch 2);
training continued to epoch 5 before early-stopping because validation loss
got worse even as training loss kept improving (train macro F1 reached 0.85)
- a textbook overfitting signature for a small model with limited
regularization. On the untouched test set: accuracy 0.333, macro F1 0.247.
Real error analysis: the model **never once predicts** class c2 or c4
("talking on phone", right/left) on any of the 4,978 test images - total
collapse on those two classes - while over-using c1, c5, and especially c7
as catch-all guesses. A plausible explanation (not yet verified against
actual images) is that "phone near the ear" poses look visually similar to
several other poses to a shallow 3-block CNN, and training stopped (at the
best-validation epoch) before the model had specialized past the more
visually distinct classes.

**What to say if the professor asks:**
- "Why macro F1 and not just accuracy?" - the classes are only mildly
  imbalanced (max/min ratio 1.3), but macro F1 still matters because it
  weights every class equally, so a model that ignores two classes entirely
  (like this one does) gets penalized even though overall accuracy looks
  survivable.
- "Why split by driver instead of randomly?" - randomly splitting images
  would let near-duplicate frames of the same driver, seat, and camera
  angle leak between train and test, inflating the reported score with
  information the model wouldn't have on a genuinely new driver. Splitting
  by driver (`subject_independent_split`, with an explicit leakage
  assertion that's checked and passes every run) measures generalization to
  people the model has never seen.
- "Is this the final result?" - no. This is Experiment A (the baseline)
  from a single run, on a CPU-only laptop, with fairly aggressive early
  stopping. Experiment B (transfer learning from ImageNet) is the actual
  comparison point, and neither should be treated as final without more
  runs/tuning, ideally on Colab GPU.
- "What would you try next?" - inspect actual misclassified c2/c4 images to
  see if the visual-similarity theory holds, try class-weighted loss or more
  epochs before early stopping, and compare against the transfer-learning
  model before picking a "best" model.

## 2026-09-17 — Experiment A vs B: validation-set "winner" isn't always the better generalizer

**What it is:** Experiment B, `TransferLearningCNN` with a frozen
MobileNetV3-Small ImageNet backbone and a new linear head, trained on the
same State Farm split as Experiment A (SimpleCNN).

**Why we used it:** transfer learning is the Brief's second required
architecture, and freezing the backbone first (training only a small head)
is the standard first stage before optionally fine-tuning the backbone -
cheaper, and a fair test of whether generic ImageNet features are already
useful for this task before spending compute unfreezing them.

**How it works here:** identical pipeline to Experiment A (same driver
split, same image size, same no-flip augmentation rule) so the only real
difference between the two runs is the architecture - a deliberate
controlled comparison, not two runs on different data.

**What the real result was:** on validation, Experiment A scored *higher*
macro F1 (0.384 vs. 0.310). But on the untouched test set, Experiment B
scored clearly *higher* (macro F1 0.355 vs. 0.247) and - more importantly -
never collapsed on any class the way A collapsed on c2/c4. B's own weakness:
it heavily over-predicts c9 (talking to passenger), including for 227 of 564
genuinely safe-driving (c0) test images. B is also much slower per image on
this CPU (55 FPS vs. A's 153 FPS), because a MobileNetV3 forward pass costs
more than a 3-layer CNN even with the backbone frozen.

**What to say if the professor asks:**
- "Which model is better?" - by the numbers, B generalizes better to unseen
  drivers (test macro F1) and fails more gracefully (no class collapse), but
  A is 3x faster and simpler. Neither is the final choice yet - fine-tuning
  B's backbone (already supported via `--unfreeze`) is the natural next
  experiment before deciding.
- "Then why does the Brief say to pick the model using validation results
  only?" - that rule exists so the untouched test set stays a fair,
  once-only final check, not so test results get ignored entirely. Here
  validation would have picked A, and that's a legitimate real finding
  worth reporting honestly (rather than switching to "pick whichever wins
  on test" after peeking, which would defeat the point of holding a test
  set out) - it also shows a real limitation of this project's setup: only
  3 drivers in the validation split is a small, high-variance sample, and a
  larger validation split (more subjects, i.e. the full dataset or more
  cross-validation folds) would make that selection signal more trustworthy.
- "Why does B confuse things into c9 so much?" - not yet verified against
  actual misclassified images (that's the natural follow-up), but a
  plausible reason is that "talking to a passenger" involves a head turn
  that's visually intermediate between several other poses, and frozen
  ImageNet features weren't trained to distinguish fine-grained driver poses
  in the first place - which is exactly the argument for trying the
  unfrozen/fine-tuned variant next.

## 2026-09-17 — `time.time()` deltas are not safe across a system suspend

A background training run's epoch-5 duration was logged as 31,710 seconds
(~8.8 hours) versus 265-349 seconds for every other epoch in the same run.
Investigated via `journalctl --list-boots` and the kernel log
(`ACPI: PM: ... system sleep state S3` / `PM: suspend exit` entries) rather
than guessing - the laptop genuinely suspended for about 9 hours mid-epoch,
and `time.time() - t0` includes that wall-clock gap even though the process
was frozen, not computing. **Lesson: on a personal machine that can sleep,
don't trust a `time.time()`-based duration measurement without sanity-
checking it against neighboring measurements first - and when reporting an
anomalous number, verify the actual cause (here, checked system suspend/
resume logs) rather than either silently keeping a misleading value or
inventing a "corrected" one.** The run record's `notes` field documents this
so the number isn't misread later as a real 8.8-hour epoch.

## 2026-09-17 — "I can list a competition's files" is not "I'm allowed to download them"

Told this session that State Farm competition access had been "handled/
checked," the Kaggle token did successfully authenticate and successfully
listed competition files (real filenames/sizes came back), which looked like
access was granted. The actual download call still failed, with Kaggle's API
returning an explicit `RulesAcceptanceRequired` reason in its JSON error
body. **Lesson: for gated APIs, test the specific permission-bearing action
itself (here, the download call), not an adjacent read-only action (here,
listing files) that happens to require the same auth token — a lesser scope
can succeed while the actual gate is still closed.** Also: when a user
reports an external blocker as resolved, verify it against the service's own
authoritative response before updating status to reflect that, especially
when the service (like Kaggle here) returns a machine-readable, unambiguous
reason code rather than a generic error.

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
