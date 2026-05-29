"""Mask helpers shared by evaluation and model adapters."""

from __future__ import annotations

import cv2
import numpy as np


def as_binary(mask: np.ndarray) -> np.ndarray:
    """Convert a mask-like array to 0/255 uint8."""

    return np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)


def resize_like(mask: np.ndarray, reference: np.ndarray, *, interpolation: int = cv2.INTER_NEAREST) -> np.ndarray:
    """Resize `mask` to match the height/width of `reference`."""

    height, width = reference.shape[:2]
    return cv2.resize(mask, (width, height), interpolation=interpolation)
