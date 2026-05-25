"""Gaze-map utilities used by saliency models."""

from __future__ import annotations

import numpy as np


def center_of_mass(heatmap: np.ndarray) -> tuple[float, float] | None:
    """Return `(x, y)` center of mass for a heatmap."""

    values = np.asarray(heatmap, dtype=np.float32)
    total = float(values.sum())
    if total <= 0:
        return None
    ys, xs = np.indices(values.shape[:2])
    x = float((xs * values).sum() / total)
    y = float((ys * values).sum() / total)
    return x, y

