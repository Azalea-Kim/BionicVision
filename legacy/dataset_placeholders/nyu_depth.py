"""NYU Depth V2 loading helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .frames import load_gray


def load_depth(path: str | Path) -> np.ndarray:
    """Load a depth map from `.npy` or image-like storage."""

    depth_path = Path(path)
    if depth_path.suffix.lower() == ".npy":
        return np.load(depth_path)
    return load_gray(depth_path).astype(np.float32)

