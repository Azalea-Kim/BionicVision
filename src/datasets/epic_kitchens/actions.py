"""EPIC-KITCHENS action annotation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import ast
import csv
from pathlib import Path


@dataclass(frozen=True)
class EpicAction:
    video_id: str
    start_seconds: float
    stop_seconds: float
    narration: str
    verb: str
    noun: str
    all_nouns: tuple[str, ...]


def load_actions(annotation_root: str | Path) -> list[EpicAction]:
    """Load EPIC train/validation action annotations."""

    root = Path(annotation_root)
    actions: list[EpicAction] = []
    for name in ("EPIC_100_train.csv", "EPIC_100_validation.csv"):
        path = root / name
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                actions.append(_parse_action(row))
    return actions


def actions_for_window(
    actions: list[EpicAction],
    *,
    video_id: str,
    start_seconds: float,
    stop_seconds: float,
) -> list[EpicAction]:
    """Return actions overlapping a video time window."""

    return [
        action
        for action in actions
        if action.video_id == video_id
        and action.stop_seconds >= start_seconds
        and action.start_seconds <= stop_seconds
    ]


def object_labels_for_window(
    actions: list[EpicAction],
    *,
    video_id: str,
    start_seconds: float,
    stop_seconds: float,
    include_hands: bool = True,
) -> list[str]:
    """Extract normalized object labels for text-conditioned segmentation."""

    labels: list[str] = []
    if include_hands:
        labels.extend(("hand", "left hand", "right hand"))
    for action in actions_for_window(
        actions,
        video_id=video_id,
        start_seconds=start_seconds,
        stop_seconds=stop_seconds,
    ):
        for noun in action.all_nouns or (action.noun,):
            label = normalize_epic_noun(noun)
            if label and label not in labels:
                labels.append(label)
    return labels


def normalize_epic_noun(noun: str) -> str:
    """Normalize EPIC noun tokens into human-readable labels."""

    value = noun.strip().lower().replace("_", " ")
    if not value:
        return ""
    parts = [part for part in value.split(":") if part]
    if len(parts) > 1:
        value = " ".join(reversed(parts))
    return value


def timestamp_to_seconds(timestamp: str) -> float:
    hours, minutes, seconds = timestamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _parse_action(row: dict[str, str]) -> EpicAction:
    return EpicAction(
        video_id=row["video_id"],
        start_seconds=timestamp_to_seconds(row["start_timestamp"]),
        stop_seconds=timestamp_to_seconds(row["stop_timestamp"]),
        narration=row.get("narration", ""),
        verb=row.get("verb", ""),
        noun=row.get("noun", ""),
        all_nouns=tuple(_parse_list(row.get("all_nouns", ""))),
    )


def _parse_list(value: str) -> list[str]:
    if not value:
        return []
    parsed = ast.literal_eval(value)
    return [str(item) for item in parsed]
