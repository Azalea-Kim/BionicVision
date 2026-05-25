"""Temporal persistence and smoothing helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque

import numpy as np


@dataclass(frozen=True)
class WeightedAverageConfig:
    window: int
    decay: float
    threshold: float


def exponential_weights(window: int, decay: float) -> np.ndarray:
    """Return oldest-to-newest exponential weights."""

    return np.array([np.exp(-decay * i) for i in range(window - 1, -1, -1)], dtype=np.float32)


def weighted_average(frames: list[np.ndarray], config: WeightedAverageConfig) -> np.ndarray:
    """Apply thresholded exponential averaging to recent mask frames."""

    if not frames:
        raise ValueError("weighted_average requires at least one frame.")
    recent = frames[-config.window :]
    stack = np.stack([frame.astype(np.float32) for frame in recent], axis=-1)
    weights = exponential_weights(config.window, config.decay)[-len(recent) :]
    averaged = np.average(stack, axis=-1, weights=weights)
    pixel_threshold = config.threshold * 255.0
    return np.where(averaged > pixel_threshold, averaged, 0).astype(np.uint8)


@dataclass
class TemporalMaskBuffer:
    """Fixed-size recent-frame buffer for temporal smoothing."""

    config: WeightedAverageConfig
    frames: deque[np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        self.frames = deque(maxlen=self.config.window)

    def update(self, frame: np.ndarray) -> np.ndarray:
        self.frames.append(frame)
        return weighted_average(list(self.frames), self.config)


@dataclass
class PersistenceTracker:
    """Remember last-seen instance masks for short dropouts."""

    max_missing_frames: int
    last_seen_frame: dict[int, int] = field(default_factory=dict)
    last_seen_mask: dict[int, np.ndarray] = field(default_factory=dict)

    def apply(self, instance_mask: np.ndarray, frame_index: int) -> np.ndarray:
        output = instance_mask.copy()
        current_ids = {int(i) for i in np.unique(instance_mask) if i != 0}

        for instance_id in current_ids:
            self.last_seen_frame[instance_id] = frame_index
            self.last_seen_mask[instance_id] = instance_mask == instance_id

        for instance_id, seen_frame in list(self.last_seen_frame.items()):
            if instance_id in current_ids:
                continue
            if frame_index - seen_frame <= self.max_missing_frames:
                mask = self.last_seen_mask[instance_id]
                output[mask & (output == 0)] = instance_id

        return output

