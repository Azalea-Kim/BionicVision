"""DeepGaze III saliency adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import zoom
from scipy.special import logsumexp

from models.base import ModelSpec


DEEPGAZE3_SPEC = ModelSpec(
    name="deepgaze3",
    task="saliency",
    required_packages=("torch", "deepgaze_pytorch"),
)


def centerbias_path() -> Path:
    """Return the bundled MIT1003 center-bias prior path."""

    return Path(__file__).with_name("centerbias_mit1003.npy")


def load_centerbias() -> np.ndarray:
    """Load the bundled MIT1003 center-bias prior."""

    return np.load(centerbias_path())


@dataclass
class DeepGaze3SaliencyEstimator:
    """Frame-by-frame DeepGaze III wrapper.

    DeepGaze III is a scanpath model, so it expects a short fixation history.
    For this video pipeline we use a deterministic center-biased pseudo-history;
    that keeps inference stable and avoids injecting eye-tracking data that the
    dataset does not provide.
    """

    device: str = "cuda"

    def __post_init__(self) -> None:
        import torch
        import deepgaze_pytorch

        self._torch = torch
        self.device_obj = torch.device(self.device if self.device != "cuda" or torch.cuda.is_available() else "cpu")
        self.model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(self.device_obj).eval()
        self.centerbias_template = load_centerbias()

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return an 8-bit saliency map for one BGR image."""

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        centerbias = zoom(
            self.centerbias_template,
            (height / self.centerbias_template.shape[0], width / self.centerbias_template.shape[1]),
            order=0,
            mode="nearest",
        )
        centerbias -= logsumexp(centerbias)

        image_tensor = self._torch.tensor(
            np.expand_dims(image_rgb.transpose(2, 0, 1), axis=0),
            dtype=self._torch.float32,
            device=self.device_obj,
        )
        centerbias_tensor = self._torch.tensor(
            np.expand_dims(centerbias, axis=0),
            dtype=self._torch.float32,
            device=self.device_obj,
        )
        x_history, y_history = self._fixation_history(width, height)

        with self._torch.no_grad():
            log_density = self.model(image_tensor, centerbias_tensor, x_history, y_history)
        density = np.exp(log_density.detach().cpu().numpy()[0, 0]).astype(np.float32)
        return _normalize_to_uint8(density)

    def _fixation_history(self, width: int, height: int):
        fixation_x = np.array([width // 2, width // 3, (2 * width) // 3, width // 2, width // 4, (3 * width) // 4])
        fixation_y = np.array([height // 2, height // 3, height // 3, (2 * height) // 3, height // 2, height // 2])
        return (
            self._torch.tensor(
                np.expand_dims(fixation_x[self.model.included_fixations], axis=0),
                dtype=self._torch.float32,
                device=self.device_obj,
            ),
            self._torch.tensor(
                np.expand_dims(fixation_y[self.model.included_fixations], axis=0),
                dtype=self._torch.float32,
                device=self.device_obj,
            ),
        )


def _normalize_to_uint8(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros(array.shape, dtype=np.uint8)
    lo = float(array[finite].min())
    hi = float(array[finite].max())
    if hi <= lo:
        return np.zeros(array.shape, dtype=np.uint8)
    return np.clip((array - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
