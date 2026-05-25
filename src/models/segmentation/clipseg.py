"""CLIPSeg text-conditioned segmentation adapter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from models.base import ModelSpec


DEFAULT_CLIPSEG_MODEL = "CIDAS/clipseg-rd64-refined"
CLIPSEG_SPEC = ModelSpec(
    name="clipseg",
    task="text_conditioned_segmentation",
    required_packages=("torch", "transformers"),
)


@dataclass
class ClipSegSegmenter:
    model_name: str = DEFAULT_CLIPSEG_MODEL
    device: str = "cpu"

    def __post_init__(self) -> None:
        import torch
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        self._torch = torch
        self.processor = CLIPSegProcessor.from_pretrained(self.model_name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def predict(self, image: Image.Image, labels: list[str]) -> dict[str, np.ndarray]:
        """Return one probability mask per text label."""

        if not labels:
            return {}
        rgb = image.convert("RGB")
        prompts = [f"a photo of {label}" for label in labels]
        inputs = self.processor(
            text=prompts,
            images=[rgb] * len(prompts),
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            logits = self.model(**inputs).logits
            if logits.ndim == 2:
                logits = logits.unsqueeze(0)
            masks = self._torch.nn.functional.interpolate(
                logits.unsqueeze(1),
                size=rgb.size[::-1],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            probs = masks.sigmoid().detach().cpu().numpy().astype(np.float32)
        return {label: probs[index] for index, label in enumerate(labels)}
