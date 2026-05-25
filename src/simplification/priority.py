"""Brightness priority policy for simplified prosthetic stimuli."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PriorityConfig:
    hand_brightness: int = 255
    primary_brightness: int = 220
    gaze_brightness: int = 255
    secondary_with_hand_brightness: int = 160
    secondary_without_hand_brightness: int = 220
    scene_with_objects_brightness: int = 160
    scene_without_objects_brightness: int = 255
    near_hand_threshold_percent: float = 50.0


def object_brightness(
    *,
    near_hand_percent: float = 0.0,
    near_gaze: bool = False,
    hand_recently_seen: bool = False,
    no_hand_for_long: bool = False,
    config: PriorityConfig = PriorityConfig(),
) -> int:
    """Assign object brightness from hand/gaze context."""

    if near_gaze:
        return config.gaze_brightness
    if near_hand_percent > config.near_hand_threshold_percent:
        return config.primary_brightness
    if hand_recently_seen and not no_hand_for_long:
        return config.secondary_with_hand_brightness
    return config.secondary_without_hand_brightness


def scene_brightness(*, objects_present: bool, config: PriorityConfig = PriorityConfig()) -> int:
    """Assign scene-edge brightness."""

    return config.scene_with_objects_brightness if objects_present else config.scene_without_objects_brightness

