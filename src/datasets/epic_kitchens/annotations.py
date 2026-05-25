"""EPIC-KITCHENS VISOR annotation parsing and mask utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class EpicFrame:
    """One annotated VISOR frame."""

    video_id: str
    frame_name: str
    frame_index: int
    image_path: str
    annotations: tuple["VisorObject", ...]


@dataclass(frozen=True)
class VisorObject:
    """One VISOR object annotation in one frame."""

    name: str
    segments: tuple[tuple[tuple[float, float], ...], ...]
    track_id: str | None = None
    relation: str | None = None
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class ObjectTierFrame:
    """Foreground/background object sets for one frame."""

    frame: EpicFrame
    foreground_ids: frozenset[str]
    background_interactable_ids: frozenset[str]


def load_visor_annotations(path: str | Path) -> list[EpicFrame]:
    """Load VISOR JSON annotations into normalized frame records."""

    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    frames = []
    for datapoint in sorted(data.get("video_annotations", []), key=_image_sort_key):
        image = datapoint.get("image", {})
        frame_name = image.get("name", "")
        image_path = image.get("image_path", "")
        video_id = image.get("video") or _infer_video_id(frame_name, image_path, json_path)
        annotations = tuple(_parse_object_annotation(obj) for obj in datapoint.get("annotations", []))
        frames.append(
            EpicFrame(
                video_id=video_id,
                frame_name=frame_name,
                frame_index=parse_frame_index(frame_name),
                image_path=image_path,
                annotations=annotations,
            )
        )
    return frames


def build_clip_tiers(frames: Iterable[EpicFrame]) -> list[ObjectTierFrame]:
    """Create foreground/background-interactable sets for a VISOR clip/window.

    Foreground is any hand object plus objects with an explicit hand/contact
    relation in the current frame. Background interactables are objects that are
    interacted with somewhere in the clip but are not foreground in this frame.
    """

    frame_list = list(frames)
    interacted_ids = frozenset(
        object_id(obj)
        for frame in frame_list
        for obj in frame.annotations
        if is_interaction_object(obj)
    )
    tiers = []
    for frame in frame_list:
        visible_ids = frozenset(object_id(obj) for obj in frame.annotations)
        foreground_ids = frozenset(
            object_id(obj)
            for obj in frame.annotations
            if is_hand(obj) or is_interaction_object(obj)
        )
        background_ids = (visible_ids & interacted_ids) - foreground_ids
        tiers.append(
            ObjectTierFrame(
                frame=frame,
                foreground_ids=foreground_ids,
                background_interactable_ids=frozenset(background_ids),
            )
        )
    return tiers


def object_id(obj: VisorObject) -> str:
    """Stable object identity for track-level metrics."""

    if obj.track_id:
        return obj.track_id
    return normalize_label(obj.name)


def is_hand(obj: VisorObject) -> bool:
    return "hand" in normalize_label(obj.name)


def is_interaction_object(obj: VisorObject) -> bool:
    """Best-effort contact/in-hand detection from VISOR metadata."""

    if is_hand(obj):
        return False
    relation = normalize_label(obj.relation or "")
    if relation:
        return any(token in relation for token in ("hand", "contact", "hold", "held", "inhand", "in_hand"))
    raw_text = normalize_label(json.dumps(obj.raw or {}))
    return any(token in raw_text for token in ("contact", "in_hand", "inhand", "held", "hand_relation"))


def rasterize_object(obj: VisorObject, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize a VISOR polygon annotation into a binary mask."""

    mask = np.zeros(shape, dtype=np.uint8)
    polygons = []
    for segment in obj.segments:
        if not segment:
            continue
        polygon = np.asarray(segment, dtype=np.int32)
        if polygon.ndim == 2 and polygon.shape[0] >= 3:
            polygons.append(polygon)
    if polygons:
        cv2.fillPoly(mask, polygons, 255)
    return mask


def rasterize_objects(objects: Iterable[VisorObject], shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Rasterize objects into a mapping keyed by stable object ID."""

    masks: dict[str, np.ndarray] = {}
    for obj in objects:
        oid = object_id(obj)
        mask = rasterize_object(obj, shape)
        if oid in masks:
            masks[oid] = np.maximum(masks[oid], mask)
        else:
            masks[oid] = mask
    return masks


def normalize_label(label: str) -> str:
    return label.lower().strip().replace(" ", "_").replace("-", "_")


def parse_frame_index(frame_name: str) -> int:
    digits = "".join(ch for ch in Path(frame_name).stem.split("_")[-1] if ch.isdigit())
    return int(digits) if digits else -1


def _parse_object_annotation(data: dict[str, Any]) -> VisorObject:
    raw_segments = data.get("segments", [])
    segments = tuple(tuple(tuple(point) for point in polygon) for polygon in raw_segments)
    return VisorObject(
        name=str(data.get("name", data.get("class", "unknown"))),
        segments=segments,
        track_id=_first_present(data, ("track_id", "track", "id", "object_id", "key")),
        relation=_first_present(data, ("relation", "hand_relation", "state", "type")),
        raw=data,
    )


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in data and data[key] is not None:
            return str(data[key])
    return None


def _image_sort_key(datapoint: dict[str, Any]) -> tuple[str, int]:
    image = datapoint.get("image", {})
    return str(image.get("image_path", "")), parse_frame_index(str(image.get("name", "")))


def _infer_video_id(frame_name: str, image_path: str, json_path: Path) -> str:
    if image_path:
        return image_path.split("/")[0]
    if "_" in frame_name:
        return "_".join(frame_name.split("_")[:2])
    return json_path.stem

