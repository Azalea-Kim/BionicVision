"""DEVA + priority-table temporal simplification pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from datasets.frames import save_gray
from datasets.video import frames_to_video
from models.saliency.cached import load_saliency_sequence
from models.segmentation.deva.cached import load_deva_prompt_set
from simplification.edges import EdgeConfig, extract_hough_edges, large_mask_to_edges
from simplification.fusion import threshold_saliency
from simplification.hands import infer_hand_circle, intersection_percentage
from simplification.masks import as_binary, mask_area, overlay_max, split_instance_masks
from simplification.priority import PriorityConfig, object_brightness, scene_brightness
from simplification.temporal import PersistenceTracker, TemporalMaskBuffer, WeightedAverageConfig

from .run_config import ExperimentPaths, VideoOutputConfig


@dataclass(frozen=True)
class TemporalPriorityConfig:
    large_object_area: int = 80000
    no_hand_frame_threshold: int = 30
    saliency_keep_fraction: float = 0.05
    object_average: WeightedAverageConfig = WeightedAverageConfig(window=5, decay=1.2, threshold=0.1)
    arm_average: WeightedAverageConfig = WeightedAverageConfig(window=5, decay=0.8, threshold=0.3)
    scene_average: WeightedAverageConfig = WeightedAverageConfig(window=10, decay=1.0, threshold=0.3)
    persistence_frames: int = 10
    priority: PriorityConfig = PriorityConfig()
    edges: EdgeConfig = EdgeConfig()


def run_temporal_priority_pipeline(
    paths: ExperimentPaths,
    *,
    config: TemporalPriorityConfig = TemporalPriorityConfig(),
    video: VideoOutputConfig = VideoOutputConfig(video_name="temporal_priority.mp4"),
) -> list[Path]:
    """Run the proposed DEVA priority-table simplification from cached masks."""

    if paths.deva_arms_dir is None or paths.deva_scenes_dir is None or paths.deva_objects_dir is None:
        raise ValueError("Temporal priority pipeline requires deva_arms_dir, deva_scenes_dir, and deva_objects_dir.")

    deva = load_deva_prompt_set(
        arms_dir=paths.deva_arms_dir,
        scenes_dir=paths.deva_scenes_dir,
        objects_dir=paths.deva_objects_dir,
    )
    saliency_sequence = load_saliency_sequence(paths.saliency_dir) if paths.saliency_dir else []

    object_buffer = TemporalMaskBuffer(config.object_average)
    arm_buffer = TemporalMaskBuffer(config.arm_average)
    scene_buffer = TemporalMaskBuffer(config.scene_average)
    object_persistence = PersistenceTracker(max_missing_frames=config.persistence_frames)

    output_frames = paths.output_dir / "frames"
    output_frames.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    no_hand_frames = 0
    most_recent_hand_region: np.ndarray | None = None
    most_recent_saliency: np.ndarray | None = None

    for frame_index in range(len(deva)):
        arm_mask = deva.arms[frame_index]
        scene_mask = deva.scenes[frame_index]
        object_mask = object_persistence.apply(deva.objects[frame_index], frame_index)
        saliency = saliency_sequence[frame_index] if frame_index < len(saliency_sequence) else None

        rendered, hand_region, saliency_region, hand_present = simplify_deva_frame(
            arm_mask=arm_mask,
            scene_mask=scene_mask,
            object_mask=object_mask,
            saliency=saliency,
            previous_hand_region=most_recent_hand_region,
            previous_saliency_region=most_recent_saliency,
            hand_recently_seen=most_recent_hand_region is not None and no_hand_frames < config.no_hand_frame_threshold,
            no_hand_for_long=no_hand_frames >= config.no_hand_frame_threshold,
            config=config,
        )

        if hand_present:
            no_hand_frames = 0
        else:
            no_hand_frames += 1
        if hand_region is not None:
            most_recent_hand_region = hand_region
        if saliency_region is not None:
            most_recent_saliency = saliency_region

        smoothed_objects = object_buffer.update(rendered.objects)
        smoothed_arms = arm_buffer.update(rendered.arms)
        smoothed_scene = scene_buffer.update(rendered.scene)
        simplified = np.maximum.reduce([smoothed_scene, smoothed_objects, smoothed_arms])
        written.append(save_gray(output_frames / f"frame_{frame_index + 1:05d}.png", simplified))

    if video.write_video and written:
        frames_to_video(written, paths.output_dir / video.video_name, fps=video.fps)

    return written


@dataclass(frozen=True)
class RenderedLayers:
    scene: np.ndarray
    objects: np.ndarray
    arms: np.ndarray


def simplify_deva_frame(
    *,
    arm_mask: np.ndarray,
    scene_mask: np.ndarray,
    object_mask: np.ndarray,
    saliency: np.ndarray | None,
    previous_hand_region: np.ndarray | None,
    previous_saliency_region: np.ndarray | None,
    hand_recently_seen: bool,
    no_hand_for_long: bool,
    config: TemporalPriorityConfig,
) -> tuple[RenderedLayers, np.ndarray | None, np.ndarray | None, bool]:
    """Simplify one DEVA frame into scene/object/arm grayscale layers."""

    arm_layer = np.zeros_like(as_binary(arm_mask), dtype=np.uint8)
    object_layer = np.zeros_like(arm_layer)
    scene_layer = np.zeros_like(arm_layer)

    arm_instances = split_instance_masks(arm_mask)
    hand_region = _combined_hand_region(arm_instances)
    hand_present = hand_region is not None
    effective_hand_region = hand_region if hand_region is not None else previous_hand_region

    for arm in arm_instances:
        arm_layer = overlay_max(arm_layer, arm, config.priority.hand_brightness)

    saliency_region = threshold_saliency(saliency, keep_fraction=config.saliency_keep_fraction) if saliency is not None else None
    effective_saliency = saliency_region if saliency_region is not None else previous_saliency_region

    object_instances = split_instance_masks(object_mask)
    for obj in object_instances:
        near_hand = intersection_percentage(obj, effective_hand_region) if effective_hand_region is not None else 0.0
        near_gaze = bool(effective_saliency is not None and np.any((obj > 0) & (effective_saliency > 0)))
        brightness = object_brightness(
            near_hand_percent=near_hand,
            near_gaze=near_gaze,
            hand_recently_seen=hand_present or hand_recently_seen,
            no_hand_for_long=no_hand_for_long,
            config=config.priority,
        )
        if mask_area(obj) > config.large_object_area:
            object_layer = np.maximum(object_layer, large_mask_to_edges(obj, brightness=brightness))
        else:
            object_layer = overlay_max(object_layer, obj, brightness)

    if np.any(scene_mask):
        edges = extract_hough_edges(scene_mask, config.edges)
        brightness = scene_brightness(objects_present=bool(object_instances), config=config.priority)
        scene_layer = overlay_max(scene_layer, edges, brightness)

    return RenderedLayers(scene=scene_layer, objects=object_layer, arms=arm_layer), hand_region, saliency_region, hand_present


def _combined_hand_region(arm_instances: list[np.ndarray]) -> np.ndarray | None:
    regions = []
    for arm in arm_instances:
        circle = infer_hand_circle(arm)
        if circle is not None:
            regions.append(circle.mask)
    if not regions:
        return None
    return np.maximum.reduce(regions)
