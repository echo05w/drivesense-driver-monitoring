# INDIVIDUAL CAPSTONE PROJECT — Project Brief

**Track:** 1 — Individual Project Track
**Project Brief Status: PENDING_MENTOR_APPROVAL**

> This brief has not yet been reviewed or approved by a mentor. No implementation
> beyond exploratory scaffolding, research, and non-final code should be treated
> as "approved scope" until Section 14 below is completed by a mentor. This
> status must not be changed to "Approved" by anyone other than the mentor.

## 1. Project Identification

| Decision / Question | Student Response |
|---|---|
| Student full name | *(to be filled in by student — not fabricated by tooling)* |
| Project title | DriveSense — Deep Learning Driver Monitoring System for Real-Time Drowsiness and Distraction Detection |
| Industry / field | Automotive safety / Computer Vision / Applied Deep Learning |
| Intended client / user / stakeholder | Fleet safety operators, rideshare/logistics companies, and individual drivers who want an in-cabin camera system that warns them (or a fleet manager) when a driver shows signs of drowsiness or visual/manual distraction |
| One-sentence project summary | DriveSense is a webcam-based deep learning system that monitors a driver's face and behavior in real time to detect drowsiness and distraction, and raises a graduated risk alert before an unsafe event occurs |

## 2. Client / User Background

Driver fatigue and distraction are leading contributing factors in road traffic
crashes worldwide. Fleet operators (logistics, delivery, rideshare) and
individual drivers currently rely mostly on self-reporting, mandated rest
breaks, or expensive OEM-installed driver monitoring systems (DMS) bundled
into new vehicles. There is a gap for a low-cost, camera-only, software-based
monitoring layer that can run on a phone, dash-mounted webcam, or embedded
board without proprietary hardware.

The intended user is either (a) a fleet safety manager who wants a
retrofittable monitoring tool for existing vehicles, or (b) an individual
driver who wants a personal safety aid. Both need a system that works with an
ordinary RGB webcam, runs in real time, and gives an interpretable,
graduated warning rather than a single noisy alarm.

## 3. Business Problem

Existing low-cost driver-monitoring options are either purely rule-based
(e.g., simple eye-closure timers) and prone to false alarms, or trained on
small, narrow datasets that fail to generalize across drivers, lighting
conditions, and camera angles. Fleet operators cannot easily audit or retrain
these closed systems, and individual drivers cannot afford OEM DMS hardware.
This matters because drowsy and distracted driving are preventable causes of
injury and death, and a transparent, reproducible, retrainable open system has
practical value for both safety and research/education purposes.

## 4. Requested Solution

DriveSense should take a live webcam/video feed of a driver's face and upper
body and produce, in real time:

- A **drowsiness state** (e.g., alert / low-vigilance / drowsy), derived from
  facial cues (eye closure duration, blink rate, yawning, head nodding).
- A **distraction state** (e.g., attentive / visually distracted / manually
  distracted — texting, talking on phone, reaching, looking away), derived
  from head pose, gaze direction, and hand/object cues.
- A combined, temporally smoothed **risk score** that escalates gradually
  rather than firing a single-frame alarm, with a lightweight on-screen or
  audible alert.

The system should be evaluable offline on held-out video/image data, and
demoable live from a webcam or from a pre-recorded driving video, primarily
via Google Colab (webcam capture through the browser) as the reproducible
demo interface.

## 5. Available / Expected Information

Realistically available data types, all from public, lawfully licensed
research/education datasets — no private or self-collected footage of
identifiable third parties will be used:

- Public **distraction** image datasets: e.g., the State Farm Distracted
  Driver Detection dataset (Kaggle competition; 10 classes of driver
  behavior, ~22k labeled dashboard-camera images of drivers, subject IDs
  provided for subject-independent splitting).
- Public **drowsiness** video datasets: e.g., UTA-RLDD (Real-Life Drowsiness
  Dataset — 60 subjects, alert / low-vigilant / drowsy self-labeled videos)
  and/or NTHU-DDD (drowsy driver detection benchmark), subject to each
  dataset's own access/license terms.
- Derived signals computed from these datasets using a pretrained face-mesh
  landmark model (e.g., MediaPipe Face Mesh / Face Landmarker): eye aspect
  ratio (EAR), mouth aspect ratio (MAR), head pose (pitch/yaw/roll), and
  blink/yawn event sequences, used as engineered features alongside raw-frame
  CNN features.

No dataset requiring a data-sharing agreement that forbids redistribution of
derived features/code will be committed to the repository; only dataset
*references, access instructions, and licenses* are stored in-repo, per the
rubric's dataset-source-acknowledgement requirement.

## 6. Data & Problem Discovery

| Decision / Question | Student Response |
|---|---|
| Selected dataset and source | Primary: State Farm Distracted Driver Detection (Kaggle) for the distraction model. Secondary: UTA-RLDD (Kaggle mirror / official site) for the drowsiness model. See `docs/Dataset_Research.md` for full comparison and access status. |
| What does one record / sample represent? | Distraction: one dashboard-camera still image of a driver, labeled with one of 10 behavior classes and a subject (driver) ID. Drowsiness: one short video clip / extracted frame sequence from one of three self-labeled vigilance states for one subject. |
| Proposed target or ML objective | Two related but separately trained classification objectives: (1) distraction-class classification from a single frame; (2) drowsiness-state classification from a short temporal window of frames/derived signals. A downstream risk engine fuses both outputs over time — it is a rule-based/temporal-smoothing layer, not a third trained model, so it does not count toward the "trained model" requirement. |
| Key information available at prediction / inference time | Live or recorded RGB video frames only (no depth, no IR, no vehicle telemetry) — matching what a consumer webcam can supply. |
| Main data quality issues | Class imbalance (some distraction classes and drowsiness states are rarer); dataset-specific camera angle/lighting bias; small number of unique subjects relative to number of frames (risk of the model memorizing subjects rather than behavior); self-reported drowsiness labels are subjective. |
| Potential leakage risks | Frame-level leakage if frames from the same subject/clip appear in both train and validation/test splits — mitigated by **subject-independent** splitting (split by driver ID, never by frame). Also, near-duplicate consecutive video frames must not straddle a split boundary. |
| Privacy / fairness / licensing concerns | Faces are biometric data. Only public research datasets with documented licenses for academic/research use will be used; no third-party footage will be scraped or recorded without consent. Fairness risk: datasets may under-represent certain skin tones, facial hair, eyewear (e.g., dark glasses defeat eye-based cues), and camera placements — documented explicitly in `docs/Responsible_AI.md`. |

## 7. Technical Proposal

| Decision / Question | Student Response |
|---|---|
| ML problem formulation | Two supervised multi-class image/video classification problems (distraction: single-frame; drowsiness: short-sequence), fused by a lightweight temporal risk-scoring layer. |
| Proposed baseline | Classical baseline using engineered features (MediaPipe-derived EAR/MAR/head-pose) fed into a simple classifier (e.g., logistic regression / random forest), plus a small from-scratch CNN baseline for the distraction task. |
| Main modeling approach(es) to investigate | (1) From-scratch CNN, (2) transfer learning with a pretrained ImageNet backbone (e.g., MobileNetV3/EfficientNet/ResNet, fine-tuned), (3) a temporal model over frame sequences or landmark-signal sequences (e.g., small GRU/LSTM or 1D-CNN over EAR/MAR/head-pose time series) for the drowsiness task. |
| Data splitting / validation strategy | Subject-independent train/validation/test split (split by driver/subject ID, stratified by class where possible), so no subject's frames appear in more than one split. |
| Primary evaluation metric(s) and why | Macro F1 and per-class recall (classes are imbalanced and false negatives on "drowsy"/"distracted" are the costly error); confusion matrix and ROC/PR curves for the drowsiness binary-risk decision; latency (ms/frame) for the real-time-feasibility requirement. |
| Expected inference input | A single video frame (distraction) or a rolling window of the last N frames / derived landmark signals (drowsiness), from a live webcam or a video file. |
| Expected inference output | Per-frame distraction class + confidence; rolling drowsiness state + confidence; a fused, temporally smoothed risk level (e.g., Normal / Caution / Warning) with the signal(s) that triggered it. |
| Main technical risks / assumptions | Dataset access may require account/license approval (Kaggle login, or a dataset-specific request form) that is outside the student's control and may delay acquisition of one dataset — mitigated by using Kaggle-hosted mirrors and by prioritizing the dataset with the lowest access friction first. Real-time performance on CPU-only hardware may require model size trade-offs. Cross-dataset generalization (drowsiness dataset vs. distraction dataset were filmed with different cameras/setups) is a known limitation, not assumed away. |

## 8. Functional Requirements

1. The system must classify a single driver-facing video frame into a distraction category with reported per-class precision/recall on a held-out, subject-independent test set.
2. The system must classify a short temporal window of driver video into a drowsiness state with reported metrics on a held-out, subject-independent test set.
3. The system must fuse the two model outputs over time into a single, graduated risk level, rather than alerting on a single noisy frame.
4. The system must run a live or recorded-video demo end-to-end (frame in → risk level + explanation out) inside a clean Google Colab runtime, using the browser's webcam or an uploaded video file.
5. The system must save and reload its trained model(s) and any preprocessing/feature-extraction artifacts (e.g., scalers, label encoders) without retraining.
6. The system must document its accuracy/latency trade-offs and known failure modes (e.g., glasses, low light, extreme head angles) rather than presenting only best-case results.
7. The system must handle invalid/degenerate input (no face detected, corrupted frame) safely, without crashing the pipeline.

## 9. Expected Deliverables

(Per official Capstone Evaluation Criteria — see `docs/Rubric_Alignment.md`.)

- A working ML solution addressing the approved problem.
- Trained model(s)/pipeline with documented methodology.
- Documented data sources and assumptions.
- Evaluation on held-out, subject-independent, unseen data.
- A usable inference interface (real-time webcam demo + batch video demo).
- A reproducible repository with clear run instructions (Colab-first).
- Documented limitations, risks, and recommended next steps.

## 10. Acceptance Criteria

- [ ] Distraction classifier trained and evaluated on a subject-independent held-out test split, with macro F1 reported and compared against a documented baseline.
- [ ] Drowsiness classifier trained and evaluated on a subject-independent held-out test split, with macro F1 reported and compared against a documented baseline.
- [ ] A temporal risk-fusion layer combines both model outputs into a single graduated alert, demonstrated on at least one continuous video.
- [ ] A clean-runtime Google Colab notebook reproduces the full demo (data → preprocessing → inference → risk output) without hidden local state.
- [ ] README documents setup, training, and demo/run instructions sufficient for an independent person to reproduce the main workflow.
- [ ] Responsible AI section documents bias/fairness, privacy, and misuse-limitation considerations specific to biometric driver-monitoring data.
- [ ] GOV-01 project is verifiably untouched (separate repo, separate history) throughout DriveSense development.

## 11. Constraints — In Scope / Out of Scope

**IN SCOPE**
- RGB-camera-only driver drowsiness and distraction classification.
- Public, lawfully licensed research datasets.
- Real-time or near-real-time inference on commodity CPU/GPU (webcam or video file).
- Google Colab as the primary reproducible runtime.
- A temporal risk-scoring/alerting layer built on top of the two trained classifiers.

**OUT OF SCOPE**
- In-vehicle hardware integration, CAN-bus telemetry, or OEM DMS hardware.
- Infrared/depth camera modalities (not available in the target consumer use case).
- Collecting new footage of identifiable individuals without consent.
- Any claim of medical-grade fatigue diagnosis — this is a safety-aid, not a certified medical device.
- Modifying, merging into, or reusing the GOV-01 road-damage-classification repository in any way.

## 12. Questions You Must Resolve

1. Which drowsiness dataset (UTA-RLDD vs. NTHU-DDD vs. another public alternative) has acceptable access terms and file sizes for a Colab-based workflow, and can that decision be finalized before the mentor review?
2. What temporal window length and feature representation (raw frames vs. landmark time-series) gives the best accuracy/latency trade-off for the drowsiness model on CPU-only inference?
3. How should the two independently trained models' confidences be fused into one risk score in a way that is simple enough to defend and justify (vs. training a third, unnecessary fusion model that the rubric would not additionally reward)?
4. What is the minimum acceptable frame rate/latency for the live Colab webcam demo to be convincing during a defense, given Colab's own webcam-capture round-trip overhead?

## 13. Optional Directions

- Extend the risk engine with an audible/voice alert and a simple driver-facing dashboard (Streamlit/Gradio) as an optional application layer beyond the required Colab demo.
- Explore lightweight on-device deployment (e.g., ONNX/TFLite export) as a stretch goal for real-time embedded feasibility, clearly marked optional and non-required.
- Explore cross-dataset generalization testing (train on one distraction dataset, evaluate zero-shot on a second, different one) as an additional robustness experiment.

## 14. Mentor Review & Approval

*This section is completed by the mentor after reviewing the proposed scope and technical direction. It has not been completed. No approval is implied or fabricated by this document.*

**Decision:** ☐ Approved  ☐ Approved with revisions  ☐ Revision required

**Required revisions / comments:**


**Approved scope / special conditions:**


**Mentor name:**

**Date:**

---
**Project Brief Status: PENDING_MENTOR_APPROVAL**
