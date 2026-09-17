# Rubric Alignment — DriveSense vs. Official Capstone Evaluation Criteria

Source: `Capstone_Project_Evaluation_Criteria_AI_ML_Fundamentals_UPDATED.docx`
(AI/ML Fundamentals, v1.1, 2026-07-18). This document maps DriveSense's plan
to each graded criterion so progress can be tracked against the actual
100-point rubric rather than an informal task list. Update the **Status**
column as work lands; do not mark anything "Done" until it is verifiably true
in the repository.

Passing score: 60/100 overall, plus all "Essential Requirements" in Section 4
of the rubric (real trained model, unseen-data evaluation, working demo,
reproducible docs, attended defense).

| # | Criterion | Max | Bare Min | DriveSense Plan | Status |
|---|---|---|---|---|---|
| 1 | Problem Definition & Project Alignment | 10 | 6 | Individual Project Brief (`Individual_Project_Brief.md`) defines problem, stakeholder, ML task framing (2 classification tasks + fusion), measurable success criteria (Section 10 acceptance criteria) | Brief drafted, PENDING_MENTOR_APPROVAL |
| 2 | Data & Preprocessing Pipeline | 15 | 9 | Dataset research (`Dataset_Research.md`), subject-independent split by driver ID, EDA notebook, face-landmark feature extraction (EAR/MAR/head-pose), leakage checks | **Distraction: done with real data** — real EDA (22,424 images, 0 corrupt, dimensions/imbalance checked), driver-independent split with passing leakage assertions on both subject and image path. **Drowsiness: data acquired (6-subject/18-video real selective sample, validated), pipeline PREPARED for Colab, NOT YET EXECUTED.** Local MediaPipe extraction is a confirmed (3x) memory boundary on this machine, not a code defect — see `docs/LEARNING_LOG.md`. `notebooks/02_Drowsiness_Colab_Pipeline.ipynb` carries the full extraction/windowing/split pipeline; its non-MediaPipe logic (windowing, splitting, leakage assertions) was actually run against synthetic data to prove correctness before a real Colab run. |
| 3 | Modeling & Experiments | 20 | 12 | Baseline (engineered features + simple classifier), CNN from scratch, transfer-learning CNN, temporal model (GRU/1D-CNN) for drowsiness; ≥2 approaches compared per task | **Distraction: 3 real trained/evaluated models** (from-scratch CNN, frozen transfer learning, fine-tuned transfer learning) on real data, real comparison written up including an honest validation-vs-test nuance. **Drowsiness: architectures exist, smoke-tested, and additionally proven correct end-to-end (training/checkpointing/resume/evaluation) against synthetic data — real training PREPARED in `notebooks/02_Drowsiness_Colab_Pipeline.ipynb`, NOT YET EXECUTED.** No trained-model claim will be made until a real Colab run produces real numbers. |
| 4 | Evaluation & Error Analysis | 15 | 9 | Macro F1 / per-class recall / confusion matrices on held-out subject-independent test set; comparison vs. baseline; qualitative error analysis (misclassified frames, edge cases: glasses, low light) | **Distraction: done** — real per-class precision/recall/F1, confusion matrices, and balanced accuracy for all 3 models on the untouched test set; qualitative error analysis backed by actually viewing real misclassified images (not just the matrix); real robustness re-evaluation under lighting/blur/noise/JPEG perturbations. **Drowsiness: not started** |
| 5 | End-to-End Implementation & Delivery | 20 | 12 | Full frame-in → risk-out pipeline; Colab-first reproducible demo (webcam + video upload); saved model/preprocessing artifacts; input validation (no-face-detected case) | Not started — needs a trained drowsiness model and temporal risk fusion before an end-to-end demo is meaningful; distraction inference alone could be wired up sooner |
| 6 | Documentation & Reproducibility | 10 | 6 | README with all rubric-required sections (see checklist below); clear setup/run/Colab instructions; logical repo layout | README now contains real distraction results, real dataset/training/evaluation instructions matching the actual scripts. Drowsiness sections still pending real training |
| 7 | Responsible AI & Limitations | 5 | 3 | `Responsible_AI.md`: bias/fairness (skin tone, eyewear, camera angle coverage in datasets), privacy (biometric data handling, no third-party footage), stated non-medical-device limitation | Stub created, to be filled in with dataset-specific findings after EDA |
| 8 | Presentation, Demo & Q&A | 5 | 3 | Slide deck + live Colab demo at defense | Not started (final phase) |

## README.md required-sections checklist (rubric Section 1)

The rubric requires the README to contain, verbatim as a checklist:

- [x] Project title
- [x] Problem statement
- [x] Selected project track (Track 1 — Individual)
- [x] Dataset source
- [x] ML task type
- [x] Project pipeline / system architecture
- [x] Models or approaches tested — real, for distraction; drowsiness pending
- [x] Final model and justification — distraction only; overall system pending
- [x] Evaluation metrics and results — real, for distraction; drowsiness pending
- [x] Installation instructions
- [x] Training / fine-tuning instructions — real commands, distraction only
- [ ] Demo and inference run instructions (Colab-first) — pending drowsiness + fusion
- [ ] Example input and output — pending demo
- [ ] Known limitations
- [ ] Responsible AI considerations
- [x] Student's full name

## Essential (non-numeric, pass/fail) requirements — Section 4 of rubric

| Requirement | How DriveSense satisfies it |
|---|---|
| Real trained/fine-tuned model | Both the distraction CNN and drowsiness temporal model are trained/fine-tuned by the student — not a wrapper around a hosted API |
| Evaluated on unseen data | Subject-independent held-out test split, never used in training or hyperparameter selection |
| Working end-to-end demo | Google Colab notebook: raw frame/video in → risk output out |
| Reproducibility documentation | README + `docs/` instructions sufficient for another person to rerun the main workflow from a clean Colab runtime |
| Attend final defense | Outside the scope of this repository — scheduling responsibility of the student |

## Academic integrity notes specific to this project

- No metrics, logs, screenshots, or evaluation numbers are to be written into
  this repository until they are produced by an actual run against real data.
  Any placeholder/synthetic numbers used for pipeline smoke-testing must be
  clearly labeled `SYNTHETIC / SMOKE-TEST — NOT A RESULT` in code comments and
  commit messages, and must never appear in the README or the Brief as if
  they were findings.
- Mentor approval status is tracked in `Individual_Project_Brief.md` and must
  never be marked "Approved" except by the mentor.
