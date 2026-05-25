"""Mask cleanup and instance-mask helpers."""

from __future__ import annotations

import cv2
import numpy as np


def as_binary(mask: np.ndarray) -> np.ndarray:
    """Convert a mask-like array to 0/255 uint8."""

    return np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)


def connected_component_filter(mask: np.ndarray, *, min_area: int) -> np.ndarray:
    """Remove connected components smaller than `min_area`."""

    binary = as_binary(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    output = np.zeros_like(binary)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            output[labels == label] = 255
    return output


def split_instance_masks(instance_mask: np.ndarray, *, include_background: bool = False) -> list[np.ndarray]:
    """Split an integer instance map into one binary mask per instance ID."""

    ids = np.unique(instance_mask)
    if not include_background:
        ids = ids[ids != 0]
    return [(instance_mask == instance_id).astype(np.uint8) for instance_id in ids]


def mask_area(mask: np.ndarray) -> int:
    """Count active pixels."""

    return int(np.count_nonzero(mask))


def resize_like(mask: np.ndarray, reference: np.ndarray, *, interpolation: int = cv2.INTER_NEAREST) -> np.ndarray:
    """Resize `mask` to match the height/width of `reference`."""

    height, width = reference.shape[:2]
    return cv2.resize(mask, (width, height), interpolation=interpolation)


def overlay_max(base: np.ndarray, mask: np.ndarray, value: int | float) -> np.ndarray:
    """Apply `value` where mask is active using max composition."""

    output = np.asarray(base).copy()
    active = np.asarray(mask) > 0
    output[active] = np.maximum(output[active], value)
    return output

