"""Baseline cached-output fusion pipeline."""

from __future__ import annotations

from pathlib import Path

from datasets.frames import save_gray
from datasets.video import frames_to_video
from models.depth.cached import load_depth_sequence
from models.saliency.cached import load_saliency_sequence
from models.segmentation.cached import load_mask_sequence
from simplification.fusion import baseline_fusion

from .run_config import ExperimentPaths, VideoOutputConfig


def run_baseline_pipeline(
    paths: ExperimentPaths,
    *,
    saliency_keep_fraction: float = 0.10,
    video: VideoOutputConfig = VideoOutputConfig(),
) -> list[Path]:
    """Run Han-style segmentation/saliency/depth fusion from cached outputs."""

    if paths.segmentation_dir is None or paths.depth_dir is None or paths.saliency_dir is None:
        raise ValueError("Baseline pipeline requires segmentation_dir, depth_dir, and saliency_dir.")

    segmentations = load_mask_sequence(paths.segmentation_dir)
    depths = load_depth_sequence(paths.depth_dir)
    saliencies = load_saliency_sequence(paths.saliency_dir)

    output_frames = paths.output_dir / "frames"
    output_frames.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for index, (segmentation, saliency, depth) in enumerate(zip(segmentations, saliencies, depths), start=1):
        simplified = baseline_fusion(
            segmentation,
            saliency,
            depth,
            saliency_keep_fraction=saliency_keep_fraction,
        )
        written.append(save_gray(output_frames / f"frame_{index:05d}.png", simplified))

    if video.write_video and written:
        frames_to_video(written, paths.output_dir / video.video_name, fps=video.fps)

    return written
