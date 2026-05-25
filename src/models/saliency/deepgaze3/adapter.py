"""Minimal DeepGaze III adapter metadata and assets."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from models.base import ModelSpec


DEEPGAZE3_SPEC = ModelSpec(
    name="deepgaze3",
    task="saliency",
    required_packages=("torch", "pysaliency"),
)


def centerbias_path() -> Path:
    """Return the bundled MIT1003 center-bias prior path."""

    return Path(__file__).with_name("centerbias_mit1003.npy")


def load_centerbias() -> np.ndarray:
    """Load the bundled MIT1003 center-bias prior."""

    return np.load(centerbias_path())
