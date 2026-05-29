"""Cached depth-map loading and normalization helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from datasets.frames import list_frames, load_gray


def load_depth(path: str | Path, *, mmap: bool = False) -> np.ndarray:
    """Load a cached depth map from `.npy` or grayscale image."""

    depth_path = Path(path)
    if depth_path.suffix.lower() == ".npy":
        return np.load(depth_path, mmap_mode="r" if mmap else None)
    return load_gray(depth_path).astype(np.float32)


def list_depth_paths(folder: str | Path) -> list[Path]:
    """Return sorted cached depth-map paths from a folder."""
    root = Path(folder)
    return sorted(root.glob("*.npy")) or list_frames(root)


def load_depth_sequence(folder: str | Path, *, mmap: bool = False) -> list[np.ndarray]:
    """Load sorted cached depth maps."""

    return [load_depth(path, mmap=mmap) for path in list_depth_paths(folder)]


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Normalize depth/disparity-like values into [0, 1]."""

    values = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.float32)
    min_value = float(values[finite].min())
    max_value = float(values[finite].max())
    if max_value <= min_value:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip((values - min_value) / (max_value - min_value), 0, 1).astype(np.float32)
