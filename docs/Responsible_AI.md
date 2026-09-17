# Responsible AI & Limitations — DriveSense

*Status: revised 2026-09-17 with real evidence from the distraction model's
EDA, error analysis, and robustness testing (see below); the drowsiness
sections below remain the original pre-EDA draft since that dataset hasn't
been trained on yet.*

## Real evidence from the distraction model (2026-09-17)

- **Robustness to image-quality degradation is a genuine, measured
  weakness, not a theoretical concern.** Re-evaluating the best distraction
  model (fine-tuned transfer learning) on the same held-out test set under
  Gaussian blur, Gaussian noise, and JPEG quality-10 recompression each cut
  macro F1 by roughly 28-35 points (from 0.697 down to 0.35-0.42) - close to
  halving performance. A real vehicle's camera/lighting/compression
  conditions vary far more than State Farm's controlled dashcam captures, so
  this system should not be presented as reliable under real-world image
  quality without further mitigation (e.g. blur/noise augmentation during
  training) and re-testing.
- **One class's accuracy appears to depend on an incidental framing
  choice, not the intended behavioral signal.** Direct visual inspection of
  real misclassified images for the "talking to passenger" class showed
  correct classifications consistently included a visible passenger in
  frame, while errors did not. If true at scale (only checked on a small
  sample so far - a systematic check is a documented next step), the model
  may be partly keying off "is another person visible" rather than the
  driver's own head/mouth behavior - a spurious-correlation risk worth
  disclosing rather than presenting the reported per-class score as purely
  measuring driver behavior recognition.
- **Reported metrics are from one CPU-only laptop run per model
  configuration**, not repeated across multiple seeds/hardware - the exact
  numbers in `README.md` should be read as one real, verified data point
  per configuration, not as a tightly-bounded estimate.

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
