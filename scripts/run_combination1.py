#!/usr/bin/env python3
"""Run the combination1 EPIC-KITCHENS pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipelines.combination1 import main


if __name__ == "__main__":
    main()
