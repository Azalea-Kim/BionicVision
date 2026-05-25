"""Minimal ADE20K/MIT scene-parsing adapter metadata."""

from __future__ import annotations

from dataclasses import dataclass

from models.base import ModelSpec


MIT_SCENE_PARSING_SPEC = ModelSpec(
    name="mit_scene_parsing",
    task="segmentation",
    required_packages=("torch", "torchvision", "mit_semseg"),
)


@dataclass(frozen=True)
class SceneParsingConfig:
    """Classes used by the baseline to render indoor structure."""

    structure_class_ids: tuple[int, ...] = (0, 14)
    min_region_area: int = 16000
    min_line_length: int = 30
    max_line_gap: int = 1
