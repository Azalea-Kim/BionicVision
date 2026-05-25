"""OpenCV/Numpy spectral-residual saliency baseline."""

from __future__ import annotations

import cv2
import numpy as np


def compute_spectral_residual_saliency(image_bgr: np.ndarray) -> np.ndarray:
    """Compute a normalized saliency map in `[0, 1]`.

    This is a lightweight non-DL baseline for visual sanity checks when
    DeepGaze-style dependencies are unavailable.
    """

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    fft = np.fft.fft2(resized)
    amplitude = np.abs(fft)
    phase = np.angle(fft)
    log_amplitude = np.log(amplitude + 1e-8)
    average = cv2.blur(log_amplitude, (3, 3))
    residual = log_amplitude - average
    saliency = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
    saliency = cv2.GaussianBlur(saliency.astype(np.float32), (9, 9), 2.5)
    saliency = cv2.resize(saliency, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
    max_value = float(saliency.max())
    if max_value <= 0:
        return np.zeros(image_bgr.shape[:2], dtype=np.float32)
    return np.clip(saliency / max_value, 0, 1).astype(np.float32)
