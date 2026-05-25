"""DEVA cached-mask adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..cached import load_mask_sequence


@dataclass(frozen=True)
class DevaPromptSet:
    """Prompt-grouped DEVA outputs used by the simplification pipeline."""

    arms: list[np.ndarray]
    scenes: list[np.ndarray]
    objects: list[np.ndarray]

    def __len__(self) -> int:
        return min(len(self.arms), len(self.scenes), len(self.objects))


def load_deva_prompt_set(
    *,
    arms_dir: str | Path,
    scenes_dir: str | Path,
    objects_dir: str | Path,
) -> DevaPromptSet:
    """Load prompt-grouped DEVA instance-mask sequences."""

    return DevaPromptSet(
        arms=load_mask_sequence(arms_dir),
        scenes=load_mask_sequence(scenes_dir),
        objects=load_mask_sequence(objects_dir),
    )
