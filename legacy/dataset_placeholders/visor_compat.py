"""Backward-compatible VISOR imports.

Prefer importing from `datasets.epic_kitchens`.
"""

from .epic_kitchens.annotations import (
    EpicFrame,
    ObjectTierFrame,
    VisorObject,
    build_clip_tiers,
    load_visor_annotations,
    object_id,
    rasterize_object,
    rasterize_objects,
)

__all__ = [
    "EpicFrame",
    "ObjectTierFrame",
    "VisorObject",
    "build_clip_tiers",
    "load_visor_annotations",
    "object_id",
    "rasterize_object",
    "rasterize_objects",
]

