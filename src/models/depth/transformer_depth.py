"""Transformer depth-estimation adapter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from models.base import ModelSpec
from models.depth.cached import normalize_depth


DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
TRANSFORMER_DEPTH_SPEC = ModelSpec(
    name="transformer_depth",
    task="depth",
    required_packages=("torch", "transformers"),
)


@dataclass
class TransformerDepthEstimator:
    model_name: str = DEFAULT_DEPTH_MODEL
    device: str = "cpu"

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._torch = torch
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def predict(self, image: Image.Image) -> np.ndarray:
        """Return a normalized depth map in `[0, 1]`."""

        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            outputs = self.model(**inputs)
            prediction = self._torch.nn.functional.interpolate(
                outputs.predicted_depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )
        depth = prediction.squeeze().detach().cpu().numpy().astype(np.float32)
        return normalize_depth(depth)
