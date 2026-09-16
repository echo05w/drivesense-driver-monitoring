# Responsible AI & Limitations — DriveSense

*Status: initial draft, to be revised with dataset-specific evidence after EDA
(rubric Criterion 7 requires this to reflect the actual data/model, not just
generic boilerplate).*

## Bias & fairness considerations

- Both candidate datasets (State Farm Distracted Driver Detection, UTA-RLDD)
  were collected from a limited, documented pool of subjects and camera
  setups. Skin tone, facial hair, eyewear, head coverings, and camera
  angle/lighting diversity are not guaranteed to be representative of a
  general driver population. This must be checked quantitatively during EDA
  (e.g., manual/automated audit of demographic proxies visible in a sample)
  and reported honestly, including where it cannot be fully assessed from
  the metadata provided.
- A model trained on these datasets may perform worse for drivers wearing
  dark glasses (eye-based cues become unreliable), for facial hair patterns
  or head coverings that differ from the training distribution, or for
  camera placements/angles not represented in training data.

## Privacy & safety considerations

- Faces are biometric data. Only public datasets with documented academic/
  research-use licenses are used; no footage of identifiable individuals is
  collected without consent, and no new footage of the student or third
  parties is committed to the repository.
- A real-time driver-monitoring system, if deployed beyond this coursework
  demo, raises surveillance and consent questions (e.g., a fleet driver being
  continuously monitored). This project is a research/education prototype,
  not a deployment-ready product, and this limitation is stated explicitly
  wherever the system's real-time capability is demonstrated.
- Data at rest (any cached frames/features during development) is kept out
  of version control (see `.gitignore`) and out of any shared artifact.

## Limitations and proper use

- **Not a certified medical or safety device.** DriveSense estimates
  behavioral proxies for drowsiness/distraction from a webcam feed; it does
  not diagnose sleep disorders, measure physiological drowsiness directly
  (e.g., no EEG/heart-rate), and must not be relied upon as the sole safety
  mechanism in a real vehicle.
- **Dataset-camera mismatch.** The two source datasets were filmed under
  different camera setups than each other and than an arbitrary end-user's
  webcam; cross-domain generalization is a known, tested-for limitation, not
  an assumption.
- **Self-reported drowsiness labels** (UTA-RLDD) are subjective, which
  bounds the achievable/expected ceiling on drowsiness-classification metrics
  and must be discussed when interpreting results, not hidden.
- Appropriate use is as a coursework capstone demonstration and a research
  prototype; inappropriate use includes any life-safety-critical deployment
  without independent validation, regulatory review, and consent frameworks
  well beyond this project's scope.
