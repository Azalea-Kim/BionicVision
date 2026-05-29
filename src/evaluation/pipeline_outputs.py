#!/usr/bin/env python3
"""Evaluate pipeline output directories against the local VISOR subset.

Pipeline outputs must use the shared visual-output contract:
`<clip>/frames/*.jpg` for sampled RGB inputs and
`<clip>/combination_frames/*.png` for simplified predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datasets.epic_kitchens.annotations import EpicFrame, VisorObject, load_visor_annotations
from evaluation.metrics import EvaluationConfig, evaluate_clip, flow_compensated_flicker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/epic_kitchens"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--target-fps", type=float, default=20.0)
    parser.add_argument("--strict-frame-count", action="store_true")
    args = parser.parse_args()

    summaries, frame_rows = evaluate_pipeline_outputs(
        args.output_root,
        args.data_root,
        args.target_fps,
        strict_frame_count=args.strict_frame_count,
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.results_dir / "summary.json"
    frame_path = args.results_dir / "frames.csv"
    summary_payload = {"clips": summaries, "aggregate": aggregate(summaries)}
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")
    with frame_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = frame_rows[0].keys() if frame_rows else ()
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(frame_rows)

    print(summary_path)
    print(frame_path)
    print(json.dumps(summary_payload, indent=2))


def evaluate_pipeline_outputs(
    output_root: Path,
    data_root: Path,
    target_fps: float,
    *,
    strict_frame_count: bool = False,
) -> tuple[list[dict[str, float | int | str | None]], list[dict[str, float | int | str | None]]]:
    """Evaluate every clip directory in a pipeline output root."""

    summaries = []
    frame_rows = []
    for clip_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        summary, rows = evaluate_pipeline_clip(
            clip_dir,
            data_root,
            target_fps,
            strict_frame_count=strict_frame_count,
        )
        summaries.append(summary)
        frame_rows.extend(rows)
    return summaries, frame_rows


def evaluate_pipeline_clip(
    clip_dir: Path,
    data_root: Path,
    target_fps: float,
    *,
    strict_frame_count: bool = False,
) -> tuple[dict[str, float | int | str | None], list[dict[str, float | int | str | None]]]:
    video_id, start_frame, end_frame = parse_clip_name(clip_dir.name)
    annotation_path = data_root / "visor" / "dense_annotations" / video_id / f"{video_id}_interpolations.json"
    annotations = merge_duplicate_frames(load_visor_annotations(annotation_path))
    annotations_by_index = {frame.frame_index: frame for frame in annotations}

    combination_paths = sorted((clip_dir / "combination_frames").glob("*.png"))
    rgb_paths = sorted((clip_dir / "frames").glob("*.jpg"))
    if len(combination_paths) != len(rgb_paths):
        raise ValueError(f"Expected equal RGB and combination counts in {clip_dir}")

    expected_output_frames, sampled_indices = sampled_source_indices(
        start_frame,
        end_frame,
        len(combination_paths),
        target_fps,
        strict=strict_frame_count,
    )
    selected_annotations = []
    selected_predictions = []
    selected_rgbs = []
    selected_output_indices = []
    for output_index, source_index in enumerate(sampled_indices):
        annotation = annotations_by_index.get(source_index)
        if annotation is None:
            continue
        selected_annotations.append(annotation)
        selected_predictions.append(combination_paths[output_index])
        selected_rgbs.append(rgb_paths[output_index])
        selected_output_indices.append(output_index)

    gold_result = evaluate_clip(
        selected_annotations,
        selected_predictions,
        config=EvaluationConfig(annotation_quality="gold"),
    )
    pseudo_result = evaluate_clip(
        selected_annotations,
        selected_predictions,
        config=EvaluationConfig(annotation_quality="pseudo"),
    )
    flicker = flow_compensated_flicker(selected_rgbs, selected_predictions)
    frame_rows = quality_frame_rows(
        clip_dir.name,
        video_id,
        selected_output_indices,
        "gold",
        gold_result.frames,
    ) + quality_frame_rows(
        clip_dir.name,
        video_id,
        selected_output_indices,
        "pseudo",
        pseudo_result.frames,
    )
    summary = {
        "clip": clip_dir.name,
        "video_id": video_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "annotation_quality_primary": "gold",
        "output_frames": len(combination_paths),
        "expected_output_frames": expected_output_frames,
        "evaluated_frames": len(selected_annotations),
        "skipped_missing_annotation_frames": len(combination_paths) - len(selected_annotations),
        "foreground_recall": gold_result.foreground_recall,
        "foreground_overlap": gold_result.foreground_overlap,
        "foreground_total": gold_result.foreground_total,
        "foreground_represented": gold_result.foreground_represented,
        "background_recall": gold_result.background_recall,
        "background_overlap": gold_result.background_overlap,
        "background_total": gold_result.background_total,
        "background_represented": gold_result.background_represented,
        "target_pixel_precision": gold_result.target_pixel_precision,
        "target_pixel_recall": gold_result.target_pixel_recall,
        "target_pixel_jaccard": gold_result.target_pixel_jaccard,
        "prediction_pixels": gold_result.prediction_pixels,
        "target_pixels": gold_result.target_pixels,
        "target_overlap_pixels": gold_result.target_overlap_pixels,
        "track_dropout_rate": gold_result.track_dropout_rate,
        "object_frame_miss_rate": gold_result.track_dropout_rate,
        "track_fragmentation_rate": gold_result.track_fragmentation_rate,
        "gold_foreground_recall": gold_result.foreground_recall,
        "gold_foreground_overlap": gold_result.foreground_overlap,
        "gold_foreground_total": gold_result.foreground_total,
        "gold_foreground_represented": gold_result.foreground_represented,
        "gold_background_recall": gold_result.background_recall,
        "gold_background_overlap": gold_result.background_overlap,
        "gold_background_total": gold_result.background_total,
        "gold_background_represented": gold_result.background_represented,
        "gold_target_pixel_precision": gold_result.target_pixel_precision,
        "gold_target_pixel_recall": gold_result.target_pixel_recall,
        "gold_target_pixel_jaccard": gold_result.target_pixel_jaccard,
        "gold_prediction_pixels": gold_result.prediction_pixels,
        "gold_target_pixels": gold_result.target_pixels,
        "gold_target_overlap_pixels": gold_result.target_overlap_pixels,
        "gold_track_dropout_rate": gold_result.track_dropout_rate,
        "gold_object_frame_miss_rate": gold_result.track_dropout_rate,
        "gold_track_fragmentation_rate": gold_result.track_fragmentation_rate,
        "pseudo_foreground_recall": pseudo_result.foreground_recall,
        "pseudo_foreground_overlap": pseudo_result.foreground_overlap,
        "pseudo_foreground_total": pseudo_result.foreground_total,
        "pseudo_foreground_represented": pseudo_result.foreground_represented,
        "pseudo_background_recall": pseudo_result.background_recall,
        "pseudo_background_overlap": pseudo_result.background_overlap,
        "pseudo_background_total": pseudo_result.background_total,
        "pseudo_background_represented": pseudo_result.background_represented,
        "pseudo_target_pixel_precision": pseudo_result.target_pixel_precision,
        "pseudo_target_pixel_recall": pseudo_result.target_pixel_recall,
        "pseudo_target_pixel_jaccard": pseudo_result.target_pixel_jaccard,
        "pseudo_prediction_pixels": pseudo_result.prediction_pixels,
        "pseudo_target_pixels": pseudo_result.target_pixels,
        "pseudo_target_overlap_pixels": pseudo_result.target_overlap_pixels,
        "pseudo_track_dropout_rate": pseudo_result.track_dropout_rate,
        "pseudo_object_frame_miss_rate": pseudo_result.track_dropout_rate,
        "pseudo_track_fragmentation_rate": pseudo_result.track_fragmentation_rate,
        "output_load": gold_result.output_load,
        "output_active_area": gold_result.output_active_area,
        "activity_load": gold_result.activity_load,
        "flow_compensated_flicker": flicker,
    }
    return summary, frame_rows


def quality_frame_rows(
    clip_name: str,
    video_id: str,
    output_indices: list[int],
    quality: str,
    frame_evaluations,
) -> list[dict[str, float | int | str | None]]:
    return [
        {
            "clip": clip_name,
            "video_id": video_id,
            "annotation_quality": quality,
            "output_frame": output_index,
            "visor_frame": frame_eval.frame_index,
            "foreground_total": frame_eval.foreground_total,
            "foreground_represented": frame_eval.foreground_represented,
            "foreground_recall": frame_eval.foreground_recall,
            "foreground_overlap": frame_eval.foreground_overlap,
            "background_total": frame_eval.background_total,
            "background_represented": frame_eval.background_represented,
            "background_recall": frame_eval.background_recall,
            "background_overlap": frame_eval.background_overlap,
            "prediction_active_area": frame_eval.prediction_active_area,
            "target_pixel_precision": frame_eval.target_pixel_precision,
            "target_pixel_recall": frame_eval.target_pixel_recall,
            "target_pixel_jaccard": frame_eval.target_pixel_jaccard,
            "prediction_pixels": frame_eval.prediction_pixels,
            "target_pixels": frame_eval.target_pixels,
            "target_overlap_pixels": frame_eval.target_overlap_pixels,
        }
        for output_index, frame_eval in zip(output_indices, frame_evaluations)
    ]


def parse_clip_name(name: str) -> tuple[str, int, int]:
    match = re.match(r"(?P<video>P\d+_\d+)_frames_(?P<start>\d+)_(?P<end>\d+)$", name)
    if not match:
        raise ValueError(f"Could not parse clip name: {name}")
    return match.group("video"), int(match.group("start")), int(match.group("end"))


def sampled_source_indices(
    start_frame: int,
    end_frame: int,
    output_count: int,
    target_fps: float,
    *,
    strict: bool = False,
) -> tuple[int, list[int]]:
    source_count = end_frame - start_frame + 1
    duration = source_count / 50.0
    expected_count = int(round(duration * target_fps))
    if strict and expected_count != output_count:
        raise ValueError(f"Expected {expected_count} output frames from {start_frame}-{end_frame}, got {output_count}")
    local_indices = np.linspace(0, max(source_count - 1, 0), output_count, dtype=int)
    return expected_count, [start_frame + int(index) for index in local_indices]


def merge_duplicate_frames(frames: list[EpicFrame]) -> list[EpicFrame]:
    by_index: dict[int, list[EpicFrame]] = defaultdict(list)
    for frame in frames:
        by_index[frame.frame_index].append(frame)

    merged = []
    for frame_index in sorted(by_index):
        group = by_index[frame_index]
        first = group[0]
        annotations: list[VisorObject] = []
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
                annotation_size=first.annotation_size,
            )
        )
    return merged


def aggregate(summaries: list[dict[str, float | int | str | None]]) -> dict[str, float | int | str | None]:
    if not summaries:
        return {}
    simple_keys = (
        "track_dropout_rate",
        "object_frame_miss_rate",
        "track_fragmentation_rate",
        "output_load",
        "output_active_area",
        "activity_load",
        "flow_compensated_flicker",
        "foreground_overlap",
        "background_overlap",
        "target_pixel_precision",
        "target_pixel_recall",
        "target_pixel_jaccard",
    )
    output: dict[str, float | int | str | None] = {
        "clips": len(summaries),
        "output_frames": sum(int(summary["output_frames"]) for summary in summaries),
        "evaluated_frames": sum(int(summary["evaluated_frames"]) for summary in summaries),
        "skipped_missing_annotation_frames": sum(int(summary["skipped_missing_annotation_frames"]) for summary in summaries),
        "foreground_total": sum(int(summary.get("foreground_total", 0)) for summary in summaries),
        "foreground_represented": sum(int(summary.get("foreground_represented", 0)) for summary in summaries),
        "background_total": sum(int(summary.get("background_total", 0)) for summary in summaries),
        "background_represented": sum(int(summary.get("background_represented", 0)) for summary in summaries),
        "gold_foreground_total": sum(int(summary.get("gold_foreground_total", 0)) for summary in summaries),
        "gold_foreground_represented": sum(int(summary.get("gold_foreground_represented", 0)) for summary in summaries),
        "gold_background_total": sum(int(summary.get("gold_background_total", 0)) for summary in summaries),
        "gold_background_represented": sum(int(summary.get("gold_background_represented", 0)) for summary in summaries),
        "pseudo_foreground_total": sum(int(summary.get("pseudo_foreground_total", 0)) for summary in summaries),
        "pseudo_foreground_represented": sum(int(summary.get("pseudo_foreground_represented", 0)) for summary in summaries),
        "pseudo_background_total": sum(int(summary.get("pseudo_background_total", 0)) for summary in summaries),
        "pseudo_background_represented": sum(int(summary.get("pseudo_background_represented", 0)) for summary in summaries),
        "prediction_pixels": sum(int(summary.get("prediction_pixels", 0)) for summary in summaries),
        "target_pixels": sum(int(summary.get("target_pixels", 0)) for summary in summaries),
        "target_overlap_pixels": sum(int(summary.get("target_overlap_pixels", 0)) for summary in summaries),
        "gold_prediction_pixels": sum(int(summary.get("gold_prediction_pixels", 0)) for summary in summaries),
        "gold_target_pixels": sum(int(summary.get("gold_target_pixels", 0)) for summary in summaries),
        "gold_target_overlap_pixels": sum(int(summary.get("gold_target_overlap_pixels", 0)) for summary in summaries),
        "pseudo_prediction_pixels": sum(int(summary.get("pseudo_prediction_pixels", 0)) for summary in summaries),
        "pseudo_target_pixels": sum(int(summary.get("pseudo_target_pixels", 0)) for summary in summaries),
        "pseudo_target_overlap_pixels": sum(int(summary.get("pseudo_target_overlap_pixels", 0)) for summary in summaries),
    }
    foreground_total = int(output["foreground_total"])
    background_total = int(output["background_total"])
    output["foreground_recall"] = (
        int(output["foreground_represented"]) / foreground_total if foreground_total else None
    )
    output["background_recall"] = (
        int(output["background_represented"]) / background_total if background_total else None
    )
    output["annotation_quality_primary"] = "gold"
    for quality in ("gold", "pseudo"):
        foreground_total = int(output[f"{quality}_foreground_total"])
        background_total = int(output[f"{quality}_background_total"])
        output[f"{quality}_foreground_recall"] = (
            int(output[f"{quality}_foreground_represented"]) / foreground_total if foreground_total else None
        )
        output[f"{quality}_background_recall"] = (
            int(output[f"{quality}_background_represented"]) / background_total if background_total else None
        )
        for key in (
            "foreground_overlap",
            "background_overlap",
        ):
            summary_key = f"{quality}_{key}"
            values = [float(summary[summary_key]) for summary in summaries if summary[summary_key] is not None]
            output[summary_key] = mean(values) if values else None
        prediction_pixels = int(output[f"{quality}_prediction_pixels"])
        target_pixels = int(output[f"{quality}_target_pixels"])
        overlap_pixels = int(output[f"{quality}_target_overlap_pixels"])
        output[f"{quality}_target_pixel_precision"] = (
            overlap_pixels / prediction_pixels if prediction_pixels else None
        )
        output[f"{quality}_target_pixel_recall"] = (
            overlap_pixels / target_pixels if target_pixels else None
        )
        union_pixels = prediction_pixels + target_pixels - overlap_pixels
        output[f"{quality}_target_pixel_jaccard"] = (
            overlap_pixels / union_pixels if union_pixels else None
        )
    for key in simple_keys:
        values = [float(summary[key]) for summary in summaries if summary[key] is not None]
        output[key] = mean(values) if values else None
    prediction_pixels = int(output["prediction_pixels"])
    target_pixels = int(output["target_pixels"])
    overlap_pixels = int(output["target_overlap_pixels"])
    output["target_pixel_precision"] = overlap_pixels / prediction_pixels if prediction_pixels else None
    output["target_pixel_recall"] = overlap_pixels / target_pixels if target_pixels else None
    union_pixels = prediction_pixels + target_pixels - overlap_pixels
    output["target_pixel_jaccard"] = overlap_pixels / union_pixels if union_pixels else None
    for quality in ("gold", "pseudo"):
        for key in ("track_dropout_rate", "object_frame_miss_rate", "track_fragmentation_rate"):
            summary_key = f"{quality}_{key}"
            values = [float(summary[summary_key]) for summary in summaries if summary[summary_key] is not None]
            output[summary_key] = mean(values) if values else None
    return output


if __name__ == "__main__":
    main()
