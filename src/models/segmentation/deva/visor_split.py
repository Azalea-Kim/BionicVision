"""Split raw DEVA masks into VISOR-guided arm/object/scene masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import cv2
import numpy as np

from datasets.epic_kitchens.annotations import EpicFrame, is_hand, load_visor_annotations, rasterize_object
from simplification.masks import resize_like


@dataclass(frozen=True)
class DevaSplitConfig:
    hand_overlap_threshold: float = 0.05
    object_overlap_threshold: float = 0.05
    annotation_coverage_threshold: float = 0.20
    scene_min_area_fraction: float = 0.02


def split_deva_with_visor(
    *,
    deva_annotation_dir: Path,
    visor_annotation_path: Path,
    output_dir: Path,
    sampled_source_indices: list[int],
    config: DevaSplitConfig = DevaSplitConfig(),
) -> dict[str, list[Path]]:
    """Create arms/scenes/objects binary masks from raw DEVA IDs and VISOR masks."""

    if output_dir.exists():
        shutil.rmtree(output_dir)
    visor_frames = merge_duplicate_frames(load_visor_annotations(visor_annotation_path))
    visor_by_index = {frame.frame_index: frame for frame in visor_frames}
    deva_paths = sorted(deva_annotation_dir.glob("*.png"))
    if len(deva_paths) != len(sampled_source_indices):
        raise ValueError(f"Expected {len(sampled_source_indices)} DEVA masks, got {len(deva_paths)}")

    mask_paths = {
        "arms": output_dir / "arms",
        "objects": output_dir / "objects",
        "scenes": output_dir / "scenes",
    }
    for path in mask_paths.values():
        path.mkdir(parents=True, exist_ok=True)

    written: dict[str, list[Path]] = {name: [] for name in mask_paths}
    for output_index, (deva_path, source_index) in enumerate(zip(deva_paths, sampled_source_indices), start=1):
        frame = visor_by_index.get(source_index)
        raw_ids = load_deva_id_mask(deva_path)
        if frame is None:
            arms = np.zeros(raw_ids.shape, dtype=np.uint8)
            objects = np.zeros_like(arms)
            scenes = np.zeros_like(arms)
        else:
            arms, objects, scenes = split_frame_ids(raw_ids, frame, config=config)

        for name, image in (("arms", arms), ("objects", objects), ("scenes", scenes)):
            mask_path = mask_paths[name] / f"frame_{output_index:05d}.png"
            cv2.imwrite(str(mask_path), image)
            written[name].append(mask_path)

    return written


def split_frame_ids(
    raw_ids: np.ndarray,
    frame: EpicFrame,
    *,
    config: DevaSplitConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split one raw DEVA ID mask using VISOR overlap labels.

    DEVA is class-agnostic. We therefore use VISOR to decide which parts of a
    DEVA ID are hand/object foreground. Large DEVA IDs often include an entire
    annotated object but only devote a small fraction of their own area to it;
    classification must consider annotation coverage, not only instance area.
    """

    hand_mask = np.zeros(raw_ids.shape, dtype=np.uint8)
    object_mask = np.zeros(raw_ids.shape, dtype=np.uint8)
    for annotation in frame.annotations:
        mask = rasterize_object(annotation, raw_ids.shape)
        if mask.shape[:2] != raw_ids.shape[:2]:
            mask = resize_like(mask, raw_ids)
        if is_hand(annotation):
            hand_mask = np.maximum(hand_mask, mask)
        else:
            object_mask = np.maximum(object_mask, mask)

    arms = np.zeros(raw_ids.shape, dtype=np.uint8)
    objects = np.zeros_like(arms)
    scenes = np.zeros_like(arms)
    min_scene_area = int(round(raw_ids.size * config.scene_min_area_fraction))

    for raw_id in sorted(int(i) for i in np.unique(raw_ids) if i != 0):
        instance = raw_ids == raw_id
        area = int(np.count_nonzero(instance))
        if area == 0:
            continue
        hand_instance_overlap = _overlap_fraction(instance, hand_mask > 0)
        object_instance_overlap = _overlap_fraction(instance, object_mask > 0)
        hand_annotation_coverage = _reference_coverage(instance, hand_mask > 0)
        object_annotation_coverage = _reference_coverage(instance, object_mask > 0)
        hand_match = (
            hand_instance_overlap >= config.hand_overlap_threshold
            or hand_annotation_coverage >= config.annotation_coverage_threshold
        )
        object_match = (
            object_instance_overlap >= config.object_overlap_threshold
            or object_annotation_coverage >= config.annotation_coverage_threshold
        )

        assigned_foreground = np.zeros(raw_ids.shape, dtype=bool)
        if hand_match:
            hand_pixels = instance & (hand_mask > 0)
            if hand_instance_overlap >= config.hand_overlap_threshold and not object_match:
                hand_pixels = instance
            arms[hand_pixels] = 255
            assigned_foreground |= hand_pixels
        if object_match:
            object_pixels = instance & (object_mask > 0)
            if object_instance_overlap >= config.object_overlap_threshold and not hand_match:
                object_pixels = instance
            objects[object_pixels] = 255
            assigned_foreground |= object_pixels

        scene_pixels = instance & ~assigned_foreground
        if area >= min_scene_area and np.count_nonzero(scene_pixels) > 0:
            scenes[scene_pixels] = 255

    return arms, objects, scenes


def load_deva_id_mask(path: Path) -> np.ndarray:
    """Load DEVA's RGB long-ID PNG as an integer ID map."""

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        return image.astype(np.uint16)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint32)
    ids = rgb[:, :, 0] + 256 * rgb[:, :, 1] + 65536 * rgb[:, :, 2]
    return ids.astype(np.uint32)


def visible_instance_preview(instance_mask: np.ndarray) -> np.ndarray:
    """Convert an instance map to a high-contrast 8-bit preview image."""

    ids = [int(i) for i in np.unique(instance_mask) if i != 0]
    preview = np.zeros(instance_mask.shape[:2], dtype=np.uint8)
    if not ids:
        return preview
    values = np.linspace(96, 255, len(ids), dtype=np.uint8)
    for instance_id, value in zip(ids, values):
        preview[instance_mask == instance_id] = int(value)
    return preview


def merge_duplicate_frames(frames: list[EpicFrame]) -> list[EpicFrame]:
    """Merge duplicate VISOR frame records by frame index."""

    by_index: dict[int, list[EpicFrame]] = {}
    for frame in frames:
        by_index.setdefault(frame.frame_index, []).append(frame)

    merged = []
    for frame_index in sorted(by_index):
        group = by_index[frame_index]
        first = group[0]
        annotations = []
        seen = set()
        for frame in group:
            for annotation in frame.annotations:
                key = (
                    annotation.track_id,
                    annotation.name,
                    annotation.relation,
                    repr(annotation.segments),
                )
                if key in seen:
                    continue
                seen.add(key)
                annotations.append(annotation)
        merged.append(
            EpicFrame(
                video_id=first.video_id,
                frame_name=first.frame_name,
                frame_index=frame_index,
                image_path=first.image_path,
                annotations=tuple(annotations),
            )
        )
    return merged


def _overlap_fraction(mask: np.ndarray, reference: np.ndarray) -> float:
    area = int(np.count_nonzero(mask))
    if area == 0:
        return 0.0
    return float(np.count_nonzero(mask & reference) / area)


def _reference_coverage(mask: np.ndarray, reference: np.ndarray) -> float:
    reference_area = int(np.count_nonzero(reference))
    if reference_area == 0:
        return 0.0
    return float(np.count_nonzero(mask & reference) / reference_area)
