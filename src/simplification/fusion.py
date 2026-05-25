"""Fusion logic for saliency, segmentation, and depth outputs."""

from __future__ import annotations

import cv2
import numpy as np

from .masks import resize_like


def threshold_saliency(saliency: np.ndarray, *, keep_fraction: float = 0.05) -> np.ndarray:
    """Keep the strongest saliency pixels as a binary mask."""

    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must be in (0, 1].")
    saliency_float = np.asarray(saliency, dtype=np.float32)
    threshold = float(np.quantile(saliency_float, 1.0 - keep_fraction))
    return np.where(saliency_float >= threshold, 255, 0).astype(np.uint8)


def depth_weighted_mask(mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Use normalized depth as brightness within an active mask."""

    depth_float = np.asarray(depth, dtype=np.float32)
    if depth_float.shape[:2] != mask.shape[:2]:
        depth_float = resize_like(depth_float, mask, interpolation=cv2.INTER_LINEAR)
    active = np.asarray(mask) > 0
    if depth_float.max() > depth_float.min():
        depth_norm = (depth_float - depth_float.min()) / (depth_float.max() - depth_float.min())
    else:
        depth_norm = np.zeros_like(depth_float)
    output = np.zeros(mask.shape[:2], dtype=np.uint8)
    output[active] = np.clip(depth_norm[active] * 255.0, 0, 255).astype(np.uint8)
    return output


def baseline_fusion(
    segmentation: np.ndarray,
    saliency: np.ndarray,
    depth: np.ndarray,
    *,
    saliency_keep_fraction: float = 0.10,
) -> np.ndarray:
    """Han-style fusion: OR segmentation with thresholded saliency, then depth-weight."""

    saliency_mask = threshold_saliency(saliency, keep_fraction=saliency_keep_fraction)
    if saliency_mask.shape[:2] != segmentation.shape[:2]:
        saliency_mask = resize_like(saliency_mask, segmentation)
    combined = np.maximum(segmentation.astype(np.uint8), saliency_mask)
    return depth_weighted_mask(combined, depth)

