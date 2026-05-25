"""Generic cached segmentation-mask loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from datasets.frames import list_frames, load_gray


def load_mask(path: str | Path) -> np.ndarray:
    """Load a mask from `.npy` or an image file."""

    mask_path = Path(path)
    if mask_path.suffix.lower() == ".npy":
        return np.load(mask_path)
    return load_gray(mask_path)


def load_mask_sequence(folder: str | Path) -> list[np.ndarray]:
    """Load sorted cached masks from a folder."""

    root = Path(folder)
    paths = sorted(root.glob("*.npy")) or list_frames(root)
    return [load_mask(path) for path in paths]

