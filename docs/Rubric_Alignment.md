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
| 2 | Data & Preprocessing Pipeline | 15 | 9 | Dataset research (`Dataset_Research.md`), subject-independent split by driver ID, EDA notebook, face-landmark feature extraction (EAR/MAR/head-pose), leakage checks | Not started — blocked on dataset acquisition |
| 3 | Modeling & Experiments | 20 | 12 | Baseline (engineered features + simple classifier), CNN from scratch, transfer-learning CNN, temporal model (GRU/1D-CNN) for drowsiness; ≥2 approaches compared per task | Not started |
| 4 | Evaluation & Error Analysis | 15 | 9 | Macro F1 / per-class recall / confusion matrices on held-out subject-independent test set; comparison vs. baseline; qualitative error analysis (misclassified frames, edge cases: glasses, low light) | Not started |
| 5 | End-to-End Implementation & Delivery | 20 | 12 | Full frame-in → risk-out pipeline; Colab-first reproducible demo (webcam + video upload); saved model/preprocessing artifacts; input validation (no-face-detected case) | Not started |
| 6 | Documentation & Reproducibility | 10 | 6 | README with all rubric-required sections (see checklist below); clear setup/run/Colab instructions; logical repo layout | README drafted (initial) |
| 7 | Responsible AI & Limitations | 5 | 3 | `Responsible_AI.md`: bias/fairness (skin tone, eyewear, camera angle coverage in datasets), privacy (biometric data handling, no third-party footage), stated non-medical-device limitation | Stub created, to be filled in with dataset-specific findings after EDA |
| 8 | Presentation, Demo & Q&A | 5 | 3 | Slide deck + live Colab demo at defense | Not started (final phase) |

## README.md required-sections checklist (rubric Section 1)

The rubric requires the README to contain, verbatim as a checklist:

- [ ] Project title
- [ ] Problem statement
- [ ] Selected project track (Track 1 — Individual)
- [ ] Dataset source
- [ ] ML task type
- [ ] Project pipeline / system architecture
- [ ] Models or approaches tested
- [ ] Final model and justification
- [ ] Evaluation metrics and results
- [ ] Installation instructions
- [ ] Training / fine-tuning instructions
- [ ] Demo and inference run instructions (Colab-first)
- [ ] Example input and output
- [ ] Known limitations
- [ ] Responsible AI considerations
- [ ] Student's full name

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
