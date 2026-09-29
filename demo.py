#!/usr/bin/env python3
"""Thin entry point: `python demo.py --source 0` (or a video/image path).

Real logic lives in `drivesense.demo` (also runnable as
`python -m drivesense.demo`) so both invocation styles share one
implementation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from drivesense.demo import main  # noqa: E402

if __name__ == "__main__":
    main()
