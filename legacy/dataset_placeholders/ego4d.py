"""Ego4D convenience helpers used by the project demos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .video import extract_frames


@dataclass(frozen=True)
class Ego4DClip:
    video_path: Path
    frames_dir: Path
    fps: float = 20


def prepare_clip(clip: Ego4DClip) -> list[Path]:
    """Extract a fixed-FPS Ego4D clip into numbered frames."""

    return extract_frames(clip.video_path, clip.frames_dir, target_fps=clip.fps)

