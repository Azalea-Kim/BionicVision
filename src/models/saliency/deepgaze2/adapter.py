"""Minimal DeepGaze/FASA saliency adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from models.base import ModelSpec, ModelUnavailableError


DEEPGAZE2_SPEC = ModelSpec(
    name="deepgaze2",
    task="saliency",
    required_packages=("deepgaze", "opencv-python"),
)


def compute_saliency(image_bgr: np.ndarray, *, total_bins: int = 8) -> np.ndarray:
    """Compute a saliency map for one BGR image."""

    import cv2

    try:
        from deepgaze.saliency_map import FasaSaliencyMapping
    except ImportError as exc:
        raise ModelUnavailableError("DeepGaze/FASA dependencies are not installed.") from exc

    saliency = FasaSaliencyMapping(image_bgr.shape[0], image_bgr.shape[1])
    saliency_map = saliency.returnMask(image_bgr, tot_bins=total_bins, format="BGR2LAB")
    return cv2.GaussianBlur(saliency_map, (3, 3), 1)


def predict_image(image_path: str | Path) -> np.ndarray:
    """Load an image and return its saliency map."""

    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return compute_saliency(image)
