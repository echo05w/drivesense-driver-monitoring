# Models

STATUS: not yet implemented — no dataset acquired yet (see
`docs/Master_Plan_Status.md`). This directory will hold model definitions
once training begins:

- `baseline.py` — engineered-feature classifier (EAR/MAR/head-pose →
  logistic regression / random forest) for the drowsiness task; simple
  from-scratch CNN baseline for the distraction task.
- `cnn.py` — transfer-learning CNN (e.g., MobileNetV3/EfficientNet backbone,
  fine-tuned) for single-frame distraction classification.
- `temporal.py` — sequence model (GRU or 1D-CNN) over frame/landmark
  time-series for drowsiness-state classification.

Intentionally left unimplemented rather than stubbed with placeholder
architectures, to avoid committing untested/unused code — see project
convention against half-finished implementations. Will be filled in as part
of Phases 7–12 in `docs/Master_Plan_Status.md`.
