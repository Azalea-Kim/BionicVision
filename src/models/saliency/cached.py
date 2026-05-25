"""Cached saliency-map loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from datasets.frames import list_frames, load_gray


def load_saliency(path: str | Path) -> np.ndarray:
    """Load a cached saliency map from `.npy` or grayscale image."""

    saliency_path = Path(path)
    if saliency_path.suffix.lower() == ".npy":
        return np.load(saliency_path)
    return load_gray(saliency_path).astype(np.float32)


def load_saliency_sequence(folder: str | Path) -> list[np.ndarray]:
    """Load sorted cached saliency maps."""

    root = Path(folder)
    paths = sorted(root.glob("*.npy")) or list_frames(root)
    return [load_saliency(path) for path in paths]

