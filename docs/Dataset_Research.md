# Dataset Research — Drowsiness & Distraction Detection

Researched 2026-09-16. Status column reflects what was actually verified via
web search this session, not assumption. Re-verify before final submission —
dataset hosting/terms can change.

## Distraction detection

| Dataset | Content | Access | Notes |
|---|---|---|---|
| **State Farm Distracted Driver Detection** (Kaggle) — **selected primary** | ~22k labeled training images (10 classes: safe driving, texting, talking on phone, operating radio, drinking, reaching behind, hair/makeup, talking to passenger, etc.) + ~79k unlabeled test images; `driver_imgs_list.csv` provides subject (driver) ID per image | Kaggle account + competition rules acceptance required; free, no separate license request | Subject IDs make subject-independent splitting straightforward (rubric requirement). Widely used and well documented, lowest access friction of the candidates. [Kaggle page](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data) |
| AUC Distracted Driver Dataset v2 | Similar 10-class distraction taxonomy, multiple camera views | Requires emailing the authors / signed request form | Higher access friction — kept as a documented alternative, not primary |
| 100-Driver | Larger, multi-camera-view distraction dataset | Requires registration/request | Considered too large for a Colab-first workflow within capstone timeline |

## Drowsiness detection

| Dataset | Content | Access | Notes |
|---|---|---|---|
| **UTA-RLDD** (Real-Life Drowsiness Dataset) — **selected primary** | 60 subjects × 3 self-labeled videos each (alert / low-vigilant / drowsy), ~30 hours total, ~10 min/video | Official site (`sites.google.com/view/utarldd/home`, mirrored at `vlm1.uta.edu/~athitsos/projects/drowsiness/`) is free/open; also mirrored on Kaggle by third parties (`rishab260/uta-reallife-drowsiness-dataset`, `minhngt02/uta-rldd`) | Full raw set is ~111 GB — too large to fully download in Colab's default disk quota. Plan: use a Kaggle mirror (verify size/preprocessing) and/or subsample subjects, extracting frames/landmarks rather than storing raw video in the repo. Labels are self-reported, a documented limitation. |
| NTHU Driver Drowsiness Detection (DDD) | IR-camera videos, drowsy/non-drowsy, day/night, glasses/no-glasses subsets | Requires signing an academic-use agreement with NTHU CVLab and waiting for approval | Documented as a fallback option; **anticipated blocker** — the agreement/approval step is outside the student's control and may not complete before the project deadline, so it is not the primary choice |
| YawDD | Yawning-focused videos for drowsiness proxy | Requires emailing the dataset authors for access | Kept as a documented fallback for yawning-specific augmentation only |

## Decision

- **Distraction model:** State Farm Distracted Driver Detection (Kaggle competition dataset).
- **Drowsiness model:** UTA-RLDD, acquired via a Kaggle mirror to fit Colab disk/time constraints; frame/landmark extraction only, raw video not committed to the repository.
- Both choices prioritize (a) lawful, documented licensing for academic use,
  (b) subject-level metadata to support subject-independent splitting, and
  (c) low access friction so a mentor-approval delay on a request-form
  dataset does not block the whole project.

## Acquisition attempt log (2026-09-16, real Kaggle credentials)

- Kaggle CLI 2.2.4 authenticated successfully via `KAGGLE_API_TOKEN` (verified
  with a real `kaggle quota` call — returned actual GPU/TPU quota, not
  fabricated).
- **State Farm Distracted Driver Detection: `BLOCKED_EXTERNAL`.**
  `kaggle competitions download -c state-farm-distracted-driver-detection`
  returned `403 Forbidden`. This is Kaggle's standard competition-rules gate:
  the account owner must visit
  `https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/rules`
  and click "I Understand and Accept the Rules" while logged in, before the
  API will serve files. This is a one-time, account-bound legal-consent
  action — deliberately not automatable via API, and not something this
  session decided to click through via browser automation on the user's
  behalf without being asked to.
  - A third-party Kaggle "dataset" re-upload of the same imagery
    (`rightway11/state-farm-distracted-driver-detection`, no rules gate) was
    found but **deliberately not used as a substitute**: the competition
    rules gate exists specifically to bind the downloader to State Farm's
    usage/redistribution terms, and a third-party re-upload's own
    redistribution license is unverified. Routing around consent to reach
    the same data would conflict with this project's own "lawful and
    ethically acceptable data sources" requirement (Brief §11, rubric
    Criterion 7). Left documented here rather than silently used.
- **UTA-RLDD (drowsiness):** `kaggle datasets download -d
  rishab260/uta-reallife-drowsiness-dataset` — no rules gate (it's a
  dataset, not a competition) — started successfully but was **interrupted
  mid-transfer** when the session disconnected (see recovery note in
  `docs/Master_Plan_Status.md`): `data/raw/drowsiness/uta-reallife-drowsiness-dataset.zip`
  is 1.5 GB on disk but fails `zipfile` integrity validation
  (`BadZipFile: File is not a zip file`), and a `.kaggle-partial` resume
  marker (Kaggle's own resumable-download bookkeeping, plain JSON, no
  credentials in it) is still present next to it. **Not treated as acquired
  data** — corrected here after the next session verified the file rather
  than trusting the "proceeded successfully" claim written before
  verification. The partial file was deliberately left in place (not
  deleted) since Kaggle's CLI can resume a `.kaggle-partial`-marked download
  rather than restart from zero, once valid credentials are available again.

## Known blocker and mitigation

Neither dataset can be downloaded from this local machine without a Kaggle
account and API token (`kaggle.json`), which is not currently configured in
this environment. This does not block the rest of the project:

- Acquisition is designed to happen from **Google Colab**, where the student
  authenticates with their own Kaggle account (per the plan's "Google Colab
  workflow where available" phase) — this is the actual acquisition path,
  not a workaround.
- Locally, the repository ships an acquisition script (`scripts/download_data.py`)
  that works identically once a `kaggle.json` is placed at `~/.kaggle/kaggle.json`
  either locally or in a Colab runtime.
- All other DriveSense tasks not dependent on having the raw pixels in hand
  (brief, rubric alignment, repo scaffold, preprocessing/model code,
  responsible-AI drafting, README structure) proceed without waiting on this.
