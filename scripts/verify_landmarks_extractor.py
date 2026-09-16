#!/usr/bin/env python3
"""Manual verification for MediaPipeLandmarkExtractor.

Not part of the automated pytest suite: constructing the FaceLandmarker
graph was observed on this development machine to be killed by the OS
under desktop memory pressure (exit 137 / OOM), which would take down the
whole test process rather than fail one test. Run this manually — in Google
Colab, or locally when `free -h` shows more headroom — to actually verify
the extractor end-to-end before marking it done in
docs/Master_Plan_Status.md.

Usage:
    python scripts/verify_landmarks_extractor.py [path/to/image_with_a_face.jpg]

With no argument, runs only the "no face detected" case on a blank frame.
"""

from __future__ import annotations

import sys

import numpy as np


def main() -> None:
    sys.path.insert(0, "src")
    from drivesense.features.landmarks import MediaPipeLandmarkExtractor

    print("Constructing MediaPipeLandmarkExtractor (downloads model if needed)...")
    extractor = MediaPipeLandmarkExtractor()
    print("Construction OK.")

    if len(sys.argv) > 1:
        import cv2

        img_bgr = cv2.imread(sys.argv[1])
        if img_bgr is None:
            raise SystemExit(f"Could not read image: {sys.argv[1]}")
        frame_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = extractor.extract(frame_rgb)
        print("Result on provided image:", result)
    else:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        result = extractor.extract(blank)
        print("Result on blank frame (expect face_detected=False):", result)
        assert result.face_detected is False, "Expected no face on a blank frame"
        print("OK: no-face edge case handled correctly.")


if __name__ == "__main__":
    main()
