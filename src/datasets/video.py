"""Video/frame conversion helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import imageio.v2 as imageio
import numpy as np

from .frames import list_frames, load_gray, normalize_to_uint8


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    target_fps: float | None = None,
    prefix: str = "frame",
    start_index: int = 1,
    extension: str = "jpg",
) -> list[Path]:
    """Extract frames from a video, optionally downsampling to target FPS."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if target_fps and source_fps > 0:
        frame_interval = max(int(round(source_fps / target_fps)), 1)
    else:
        frame_interval = 1

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    frame_index = 0
    output_index = start_index

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % frame_interval == 0:
            path = output_root / f"{prefix}_{output_index:03d}.{extension}"
            cv2.imwrite(str(path), frame)
            written.append(path)
            output_index += 1
        frame_index += 1

    cap.release()
    return written


def frames_to_video(
    frames: str | Path | Iterable[str | Path],
    output_path: str | Path,
    *,
    fps: float = 20,
    as_gray: bool = True,
    codec: str = "libx264",
) -> Path:
    """Write a sorted frame directory or explicit frame paths to a video."""

    if isinstance(frames, (str, Path)) and Path(frames).is_dir():
        frame_paths = list_frames(frames)
    else:
        frame_paths = [Path(path) for path in frames]  # type: ignore[arg-type]

    if not frame_paths:
        raise ValueError("No frames were provided for video creation.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}:
        _frames_to_video_cv2(frame_paths, output, fps=fps, as_gray=as_gray)
        return output

    with imageio.get_writer(output, fps=fps, codec=codec) as writer:
        for path in frame_paths:
            if as_gray:
                frame = load_gray(path)
            else:
                frame = imageio.imread(path)
            writer.append_data(_video_safe_frame(frame))

    return output


def _frames_to_video_cv2(frame_paths: list[Path], output: Path, *, fps: float, as_gray: bool) -> None:
    first = load_gray(frame_paths[0]) if as_gray else imageio.imread(frame_paths[0])
    first = _video_safe_frame(first)
    height, width = first.shape[:2]
    safe_width = width - (width % 2)
    safe_height = height - (height % 2)
    if (safe_height, safe_width) != (height, width):
        first = cv2.resize(first, (safe_width, safe_height), interpolation=cv2.INTER_NEAREST)
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if output.suffix.lower() == ".mp4" else "XVID"))
    writer = cv2.VideoWriter(str(output), fourcc, fps, (safe_width, safe_height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output}")
    try:
        writer.write(cv2.cvtColor(first, cv2.COLOR_RGB2BGR))
        for path in frame_paths[1:]:
            frame = load_gray(path) if as_gray else imageio.imread(path)
            frame = _video_safe_frame(frame)
            if frame.shape[:2] != (safe_height, safe_width):
                frame = cv2.resize(frame, (safe_width, safe_height), interpolation=cv2.INTER_NEAREST)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _video_safe_frame(frame: np.ndarray) -> np.ndarray:
    image = normalize_to_uint8(frame)
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    return image
