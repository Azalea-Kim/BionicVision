"""Shared pipeline configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentPaths:
    output_dir: Path
    frames_dir: Path | None = None
    segmentation_dir: Path | None = None
    depth_dir: Path | None = None
    saliency_dir: Path | None = None
    deva_arms_dir: Path | None = None
    deva_scenes_dir: Path | None = None
    deva_objects_dir: Path | None = None


@dataclass(frozen=True)
class VideoOutputConfig:
    fps: float = 20
    write_video: bool = True
    video_name: str = "simplified.mp4"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON or YAML configuration.

    YAML support is optional and only used if PyYAML is installed.
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            return yaml.safe_load(handle)
        return json.load(handle)


def path_or_none(value: str | Path | None) -> Path | None:
    return Path(value) if value is not None else None

