"""Combination1 pipeline: Han-style fusion with temporal model inputs.

The fusion rule intentionally stays the same as the Han baseline:
saliency and segmentation create the support mask, then depth supplies
brightness only inside that support. Combination1 changes the sources of those
three inputs to DeepGaze III, DEVA, and TCMonoDepth.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import shutil

import cv2
import numpy as np

from datasets.frames import save_gray
from datasets.epic_kitchens.annotations import build_clip_tiers, load_visor_annotations
from models.depth.tc_monodepth import TCMonoDepthEstimator
from models.saliency.deepgaze3.adapter import DeepGaze3SaliencyEstimator
from models.segmentation.deva.run_manual import GROUP_COLORS_RGB, label_group, run_deva_manual
from simplification.fusion import baseline_fusion

from .han_baseline import (
    HanBaselineConfig,
    ensure_cuda_if_requested,
    extract_video_frames,
    gen_image_brightness,
    write_video,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMBINATION1_CLIP_DIR = ROOT / "data" / "epic_kitchens" / "video_snippets" / "first_10s"
DEFAULT_COMBINATION1_OUTPUT = ROOT / "outputs" / "combination1_epic10"
DEFAULT_DATA_ROOT = ROOT / "data" / "epic_kitchens"


@dataclass(frozen=True)
class Combination1Config:
    """Configuration for the first temporal-model baseline improvement."""

    baseline: HanBaselineConfig = HanBaselineConfig()
    source_fps: float = 50.0
    deva_size: int = 360
    deva_detection_every: int = 1
    deva_memory_reset_interval: int = 4
    tc_output_is_inverse_depth: bool = True
    require_nonempty_deva_layers: bool = True


def run_combination1(
    clip_dir: Path = DEFAULT_COMBINATION1_CLIP_DIR,
    output_root: Path = DEFAULT_COMBINATION1_OUTPUT,
    data_root: Path = DEFAULT_DATA_ROOT,
    config: Combination1Config = Combination1Config(),
) -> list[dict[str, str | int]]:
    """Run combination1 on every MP4 clip in `clip_dir`."""

    ensure_cuda_if_requested(config.baseline.device)
    clips = sorted(clip_dir.glob("*.mp4"))
    if not clips:
        raise FileNotFoundError(f"No MP4 clips found in {clip_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for clip_path in clips:
        print(f"processing {clip_path.name}", flush=True)
        summaries.append(run_combination1_on_clip(clip_path, output_root, data_root, config))
    return summaries


def run_combination1_on_clip(
    clip_path: Path,
    output_root: Path,
    data_root: Path,
    config: Combination1Config = Combination1Config(),
) -> dict[str, str | int]:
    """Run combination1 for one EPIC clip named `VIDEO_frames_START_END.mp4`."""

    video_id, start_frame, end_frame = parse_clip_name(clip_path.stem)
    clip_root = output_root / clip_path.stem
    frames = extract_video_frames(clip_path, clip_root / "frames", config.baseline.target_fps, config.baseline.max_frames)
    sampled_indices = sampled_source_indices(
        start_frame,
        end_frame,
        len(frames),
        target_fps=config.baseline.target_fps,
        source_fps=config.source_fps,
    )

    saliency = build_deepgaze3_saliency_frames(frames, clip_root / "saliency_frames", config)
    depth = build_tc_monodepth_frames(frames, clip_root / "depth" / "frames", config)
    semantic_annotations = build_deva_semantic_annotations(
        frames_dir=clip_root / "frames",
        clip_root=clip_root,
        data_root=data_root,
        video_id=video_id,
        sampled_indices=sampled_indices,
        config=config,
    )
    segmentation = build_combination1_segmentation_frames(semantic_annotations, clip_root / "segmentation_frames")
    combination = combine_combination1_frames(saliency, segmentation, depth, clip_root / "combination_frames", config)

    videos_dir = clip_root / "videos"
    write_video(depth, videos_dir / "depth_tc_monodepth.mp4", config.baseline.target_fps, is_color=True)
    write_video(saliency, videos_dir / "saliency_deepgaze3.mp4", config.baseline.target_fps, is_color=False)
    write_video(segmentation, videos_dir / "segmentation_deva.mp4", config.baseline.target_fps, is_color=False)
    write_video(combination, videos_dir / "combination1.mp4", config.baseline.target_fps, is_color=False)

    return {
        "clip": str(clip_path),
        "frames": len(frames),
        "depth_frames": len(depth),
        "saliency_frames": len(saliency),
        "segmentation_frames": len(segmentation),
        "combination_frames": len(combination),
        "output": str(clip_root),
    }


def build_deepgaze3_saliency_frames(
    frame_paths: list[Path],
    output_dir: Path,
    config: Combination1Config,
) -> list[Path]:
    """Run DeepGaze III and write grayscale saliency maps."""

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    estimator = DeepGaze3SaliencyEstimator(device=config.baseline.device)
    output_paths = []
    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path))
        if image is None:
            raise FileNotFoundError(frame_path)
        output_paths.append(save_gray(output_dir / f"{frame_path.stem}.png", estimator.predict(image)))
    return output_paths


def build_tc_monodepth_frames(
    frame_paths: list[Path],
    output_dir: Path,
    config: Combination1Config,
) -> list[Path]:
    """Run TCMonoDepth and write Han-compatible depth brightness frames."""

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    estimator = TCMonoDepthEstimator(device=config.baseline.device)
    output_paths = []
    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path))
        if image is None:
            raise FileNotFoundError(frame_path)
        depth = estimator.predict(image)
        if config.tc_output_is_inverse_depth:
            depth = 1.0 - depth
        depth_frame = gen_image_brightness(
            depth,
            np.ones(depth.shape, dtype=bool),
            mode=config.baseline.depth_mode,
            min_brightness=config.baseline.depth_min_brightness,
            max_brightness=config.baseline.depth_max_brightness,
        )
        out = output_dir / f"{frame_path.stem}.png"
        cv2.imwrite(str(out), depth_frame)
        output_paths.append(out)
    return output_paths


def build_deva_semantic_annotations(
    *,
    frames_dir: Path,
    clip_root: Path,
    data_root: Path,
    video_id: str,
    sampled_indices: list[int],
    config: Combination1Config,
) -> list[Path]:
    """Run manual class-guided DEVA and return three-color semantic labels."""

    raw_dir = clip_root / "deva_raw"
    annotation_path = data_root / "visor" / "dense_annotations" / video_id / f"{video_id}_interpolations.json"
    manual_outputs = run_deva_manual(
        frames_dir=frames_dir,
        visor_annotation_path=annotation_path,
        output_dir=raw_dir,
        sampled_source_indices=sampled_indices,
        size=config.deva_size,
        detection_every=config.deva_detection_every,
        memory_reset_interval=config.deva_memory_reset_interval,
    )
    semantic_paths = sorted(manual_outputs.raw_annotation_dir.glob("*.png"))
    if config.require_nonempty_deva_layers:
        validate_deva_semantic_annotations(
            semantic_paths,
            required_by_frame=expected_deva_layers_by_frame(annotation_path, sampled_indices),
        )
    return semantic_paths


def build_combination1_segmentation_frames(
    semantic_annotation_paths: list[Path],
    output_dir: Path,
) -> list[Path]:
    """Render DEVA semantic labels into one transparent segmentation support mask."""

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for index, annotation_path in enumerate(semantic_annotation_paths):
        annotation = cv2.imread(str(annotation_path), cv2.IMREAD_COLOR)
        if annotation is None:
            raise FileNotFoundError(annotation_path)
        segmentation = np.where(np.any(annotation > 0, axis=2), 255, 0).astype(np.uint8)
        output_paths.append(save_gray(output_dir / f"frame{index:05d}.png", segmentation))
    return output_paths


def combine_combination1_frames(
    saliency_paths: list[Path],
    segmentation_paths: list[Path],
    depth_paths: list[Path],
    output_dir: Path,
    config: Combination1Config,
) -> list[Path]:
    """Fuse saliency/segmentation/depth with the Han support-mask relationship."""

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for index, (sal_path, seg_path, dep_path) in enumerate(zip(saliency_paths, segmentation_paths, depth_paths)):
        saliency = cv2.imread(str(sal_path), cv2.IMREAD_GRAYSCALE)
        segmentation = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(str(dep_path), cv2.IMREAD_GRAYSCALE)
        if saliency is None or segmentation is None or depth is None:
            raise FileNotFoundError(f"Could not load fusion inputs at index {index}")
        fused = baseline_fusion(
            segmentation=segmentation,
            saliency=saliency,
            depth=depth,
            saliency_threshold_fraction=config.baseline.saliency_threshold_fraction,
        )
        output_paths.append(save_gray(output_dir / f"frame{index:05d}.png", fused))
    return output_paths


def validate_deva_semantic_annotations(
    semantic_annotation_paths: list[Path],
    required_by_frame: list[set[str]] | None = None,
) -> None:
    """Fail fast when a required semantic DEVA color is missing."""

    if required_by_frame is None:
        required_by_frame = [set(GROUP_COLORS_RGB) for _ in semantic_annotation_paths]
    empty_layers = []
    for frame_index, required_layers in enumerate(required_by_frame):
        if frame_index >= len(semantic_annotation_paths):
            empty_layers.extend(f"{name}@{frame_index:05d}:missing" for name in sorted(required_layers))
            continue
        image = cv2.imread(str(semantic_annotation_paths[frame_index]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(semantic_annotation_paths[frame_index])
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for name in sorted(required_layers):
            color = np.asarray(GROUP_COLORS_RGB[name], dtype=np.uint8)
            if not np.any(np.all(image_rgb == color, axis=2)):
                empty_layers.append(f"{name}@{frame_index:05d}")
    if empty_layers:
        raise RuntimeError(f"DEVA semantic annotations are missing required labels: {', '.join(empty_layers)}")


def expected_deva_layers_by_frame(annotation_path: Path, sampled_indices: list[int]) -> list[set[str]]:
    """Infer per-frame split layers that should be nonempty from VISOR labels."""

    frames_list = load_visor_annotations(annotation_path)
    frames = {frame.frame_index: frame for frame in frames_list}
    tiers = {tier.frame.frame_index: tier for tier in build_clip_tiers(frames_list)}
    required_by_frame = []
    for source_index in sampled_indices:
        required = set()
        frame = frames.get(source_index)
        tier = tiers.get(source_index)
        if frame is not None:
            for annotation in frame.annotations:
                required.add(label_group(annotation, tier))
        required_by_frame.append(required)
    return required_by_frame


def parse_clip_name(name: str) -> tuple[str, int, int]:
    match = re.match(r"(?P<video>P\d+_\d+)_frames_(?P<start>\d+)_(?P<end>\d+)$", name)
    if not match:
        raise ValueError(f"Could not parse EPIC clip name: {name}")
    return match.group("video"), int(match.group("start")), int(match.group("end"))


def sampled_source_indices(
    start_frame: int,
    end_frame: int,
    output_count: int,
    *,
    target_fps: float,
    source_fps: float,
) -> list[int]:
    source_count = end_frame - start_frame + 1
    expected_count = int(round((source_count / source_fps) * target_fps))
    if output_count != expected_count:
        expected_count = output_count
    local_indices = np.linspace(0, max(source_count - 1, 0), expected_count, dtype=int)
    return [start_frame + int(index) for index in local_indices[:output_count]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run combination1 on EPIC-KITCHENS clips.")
    parser.add_argument("--clip-dir", type=Path, default=DEFAULT_COMBINATION1_CLIP_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_COMBINATION1_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--target-fps", type=float, default=20.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--deva-detection-every", type=int, default=1)
    parser.add_argument("--deva-memory-reset-interval", type=int, default=4)
    parser.add_argument("--deva-size", type=int, default=360)
    args = parser.parse_args()
    baseline = HanBaselineConfig(target_fps=args.target_fps, max_frames=args.max_frames, device=args.device)
    config = Combination1Config(
        baseline=baseline,
        deva_detection_every=args.deva_detection_every,
        deva_memory_reset_interval=args.deva_memory_reset_interval,
        deva_size=args.deva_size,
    )
    for summary in run_combination1(args.clip_dir.resolve(), args.output_root.resolve(), args.data_root.resolve(), config):
        print(summary)


if __name__ == "__main__":
    main()
