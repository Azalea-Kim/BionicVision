"""Fusion logic for saliency, segmentation, and depth outputs."""

from __future__ import annotations

import cv2
import numpy as np

from .masks import resize_like


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
    saliency_threshold_fraction: float = 0.90,
) -> np.ndarray:
    """Han-style fusion: OR segmentation with thresholded saliency, then depth-weight."""

    threshold = float(np.max(saliency)) * saliency_threshold_fraction
    saliency_mask = np.asarray(saliency).copy()
    saliency_mask[saliency_mask <= threshold] = 0
    saliency_mask[saliency_mask > 0] = 255
    saliency_mask = saliency_mask.astype(np.uint8)
    if saliency_mask.shape[:2] != segmentation.shape[:2]:
        saliency_mask = resize_like(saliency_mask, segmentation)
    combined = np.maximum(segmentation.astype(np.uint8), saliency_mask)
    return depth_weighted_mask(combined, depth)
