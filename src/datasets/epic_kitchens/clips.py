"""EPIC-KITCHENS clip/window loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from datasets.frames import list_frames

from .annotations import EpicFrame, ObjectTierFrame, build_clip_tiers, load_visor_annotations


@dataclass(frozen=True)
class EpicClip:
    annotation_path: Path
    frames_dir: Path | None = None
    frames: tuple[EpicFrame, ...] = ()
    tiers: tuple[ObjectTierFrame, ...] = ()


def load_epic_clip(annotation_path: str | Path, *, frames_dir: str | Path | None = None) -> EpicClip:
    """Load VISOR annotations and optional RGB frame paths for one clip/window."""

    frames = tuple(load_visor_annotations(annotation_path))
    tiers = tuple(build_clip_tiers(frames))
    return EpicClip(
        annotation_path=Path(annotation_path),
        frames_dir=Path(frames_dir) if frames_dir else None,
        frames=frames,
        tiers=tiers,
    )


def clip_frame_paths(clip: EpicClip) -> list[Path]:
    """Return sorted RGB frame paths for a loaded clip."""

    if clip.frames_dir is None:
        return []
    return list_frames(clip.frames_dir)


def select_frame_window(frames: tuple[EpicFrame, ...], *, start: int, end: int) -> tuple[EpicFrame, ...]:
    """Select annotations by inclusive frame-index window."""

    return tuple(frame for frame in frames if start <= frame.frame_index <= end)

