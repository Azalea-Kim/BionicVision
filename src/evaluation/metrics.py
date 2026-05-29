"""Metrics for EPIC-KITCHENS/VISOR scene-simplification evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import cv2
import numpy as np

from datasets.epic_kitchens.annotations import (
    EpicFrame,
    ObjectTierFrame,
    build_clip_tiers,
    rasterize_objects,
)
from datasets.frames import load_gray, load_rgb
from simplification.masks import as_binary, resize_like


@dataclass(frozen=True)
class EvaluationConfig:
    """Thresholds for object-level representation and output-load metrics."""

    prediction_threshold: int = 1
    min_overlap_fraction: float = 0.01
    min_overlap_pixels: int = 3
    boundary_kernel: int = 7
    active_threshold: float = 0.05
    annotation_quality: Literal["gold", "pseudo", "all"] = "gold"


@dataclass(frozen=True)
class FrameEvaluation:
    """Per-frame object-recall measurements.

    Background totals refer to visible non-hand context objects that are not
    foreground in the current frame.
    """

    frame_index: int
    foreground_total: int
    foreground_represented: int
    background_total: int
    background_represented: int
    represented_ids: frozenset[str]
    foreground_ids: frozenset[str]
    background_ids: frozenset[str]
    foreground_overlap: float | None
    background_overlap: float | None
    prediction_pixels: int
    target_pixels: int
    target_overlap_pixels: int
    frame_pixels: int

    @property
    def foreground_recall(self) -> float | None:
        return _ratio_or_none(self.foreground_represented, self.foreground_total)

    @property
    def background_recall(self) -> float | None:
        return _ratio_or_none(self.background_represented, self.background_total)

    @property
    def target_pixel_precision(self) -> float | None:
        return _ratio_or_none(self.target_overlap_pixels, self.prediction_pixels)

    @property
    def target_pixel_recall(self) -> float | None:
        return _ratio_or_none(self.target_overlap_pixels, self.target_pixels)

    @property
    def target_pixel_jaccard(self) -> float | None:
        union = self.prediction_pixels + self.target_pixels - self.target_overlap_pixels
        return _ratio_or_none(self.target_overlap_pixels, union)

    @property
    def prediction_active_area(self) -> float:
        return _ratio_or_none(self.prediction_pixels, self.frame_pixels) or 0.0


@dataclass(frozen=True)
class ClipEvaluation:
    """Aggregate metrics for one evaluated clip/window."""

    frames: tuple[FrameEvaluation, ...]
    foreground_recall: float | None
    background_recall: float | None
    foreground_overlap: float | None
    background_overlap: float | None
    target_pixel_precision: float | None
    target_pixel_recall: float | None
    target_pixel_jaccard: float | None
    prediction_pixels: int
    target_pixels: int
    target_overlap_pixels: int
    track_dropout_rate: float
    output_load: float
    output_active_area: float
    activity_load: float
    track_fragmentation_rate: float
    foreground_total: int
    foreground_represented: int
    background_total: int
    background_represented: int
    metadata: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(frozen=True)
class VariantEvaluation:
    """Evaluation result for one generated operating point."""

    name: str
    evaluation: ClipEvaluation


@dataclass(frozen=True)
class CurveEvaluation:
    """Recall-vs-activity evaluation across explicit output variants."""

    variants: tuple[VariantEvaluation, ...]
    foreground_auc: float | None
    background_auc: float | None
    load_cap: float
    metadata: dict[str, float | int | str] = field(default_factory=dict)


def evaluate_clip(
    annotation_frames: Sequence[EpicFrame],
    simplified_frames: Sequence[np.ndarray | str | Path],
    *,
    config: EvaluationConfig = EvaluationConfig(),
    percept_frames: Sequence[np.ndarray | str | Path] | None = None,
) -> ClipEvaluation:
    """Evaluate simplified frames against VISOR annotations.

    Foreground/background object tiers are inferred from each clip/window using
    `build_clip_tiers`. `percept_frames` may be a pulse2percept output sequence;
    when omitted, load is computed on the simplified frames themselves. By
    default, supervised recall is computed only on sparse VISOR masks and dense
    VISOR `type=1` filtered ground-truth masks; set
    `config.annotation_quality="pseudo"` to score dense `type=0` masks.
    """

    tiers = build_clip_tiers(annotation_frames)
    predictions = _load_gray_sequence(simplified_frames)
    if len(tiers) != len(predictions):
        raise ValueError(f"Expected equal annotation/prediction counts, got {len(tiers)} and {len(predictions)}.")

    frame_results = tuple(evaluate_frame(tier, prediction, config=config) for tier, prediction in zip(tiers, predictions))
    load_frames = _load_gray_sequence(percept_frames) if percept_frames is not None else predictions
    foreground_represented = sum(frame.foreground_represented for frame in frame_results)
    foreground_total = sum(frame.foreground_total for frame in frame_results)
    background_represented = sum(frame.background_represented for frame in frame_results)
    background_total = sum(frame.background_total for frame in frame_results)
    foreground = _ratio_or_none(foreground_represented, foreground_total)
    background = _ratio_or_none(background_represented, background_total)
    foreground_overlap = _weighted_frame_mean(
        (frame.foreground_overlap, frame.foreground_total) for frame in frame_results
    )
    background_overlap = _weighted_frame_mean(
        (frame.background_overlap, frame.background_total) for frame in frame_results
    )
    target_overlap_pixels = sum(frame.target_overlap_pixels for frame in frame_results)
    prediction_pixels = sum(frame.prediction_pixels for frame in frame_results)
    target_pixels = sum(frame.target_pixels for frame in frame_results)
    target_pixel_precision = _ratio_or_none(target_overlap_pixels, prediction_pixels)
    target_pixel_recall = _ratio_or_none(target_overlap_pixels, target_pixels)
    target_pixel_jaccard = _ratio_or_none(
        target_overlap_pixels,
        prediction_pixels + target_pixels - target_overlap_pixels,
    )
    activity = active_area(load_frames, threshold=config.active_threshold)

    return ClipEvaluation(
        frames=frame_results,
        foreground_recall=foreground,
        background_recall=background,
        foreground_overlap=foreground_overlap,
        background_overlap=background_overlap,
        target_pixel_precision=target_pixel_precision,
        target_pixel_recall=target_pixel_recall,
        target_pixel_jaccard=target_pixel_jaccard,
        prediction_pixels=prediction_pixels,
        target_pixels=target_pixels,
        target_overlap_pixels=target_overlap_pixels,
        track_dropout_rate=track_dropout_rate(frame_results),
        output_load=spv_load(load_frames),
        output_active_area=activity,
        activity_load=activity,
        track_fragmentation_rate=track_fragmentation_rate(frame_results),
        foreground_total=foreground_total,
        foreground_represented=foreground_represented,
        background_total=background_total,
        background_represented=background_represented,
        metadata={"frames": len(frame_results), "annotation_quality": config.annotation_quality},
    )


def evaluate_clip_variants(
    annotation_frames: Sequence[EpicFrame],
    variants: Mapping[str, Sequence[np.ndarray | str | Path]] | Sequence[tuple[str, Sequence[np.ndarray | str | Path]]],
    *,
    config: EvaluationConfig = EvaluationConfig(),
    load_cap: float | None = None,
) -> CurveEvaluation:
    """Evaluate explicit output variants as a recall-vs-active-area curve.

    The x-axis is `activity_load`, the mean fraction of active output pixels.
    This keeps the current unpercepted evaluation independent of brightness:
    different operating points should be generated by the pipeline itself, not
    by sweeping intensity thresholds inside one rendered video.
    """

    items = tuple(variants.items()) if isinstance(variants, Mapping) else tuple(variants)
    evaluated = tuple(
        VariantEvaluation(name=name, evaluation=evaluate_clip(annotation_frames, frames, config=config))
        for name, frames in items
    )
    cap = float(load_cap) if load_cap is not None else max(
        (variant.evaluation.activity_load for variant in evaluated),
        default=0.0,
    )
    return CurveEvaluation(
        variants=evaluated,
        foreground_auc=recall_auc(
            (
                (variant.evaluation.activity_load, variant.evaluation.foreground_recall)
                for variant in evaluated
            ),
            load_cap=cap,
        ),
        background_auc=recall_auc(
            (
                (variant.evaluation.activity_load, variant.evaluation.background_recall)
                for variant in evaluated
            ),
            load_cap=cap,
        ),
        load_cap=cap,
        metadata={"variants": len(evaluated)},
    )


def evaluate_frame(
    tier: ObjectTierFrame,
    simplified: np.ndarray,
    *,
    config: EvaluationConfig = EvaluationConfig(),
) -> FrameEvaluation:
    """Evaluate object representation for one annotated frame."""

    prediction = _prediction_mask(simplified, config.prediction_threshold)
    shape = prediction.shape[:2]
    annotations = _quality_filtered_annotations(tier.frame.annotations, config.annotation_quality)
    object_masks = rasterize_objects(annotations, shape)
    represented_ids = frozenset(
        oid
        for oid, mask in object_masks.items()
        if object_representation_score(mask, prediction, config=config) > 0
    )

    foreground_ids = tier.foreground_ids & frozenset(object_masks)
    background_ids = tier.background_interactable_ids & frozenset(object_masks)
    foreground_represented = len(foreground_ids & represented_ids)
    background_represented = len(background_ids & represented_ids)
    object_scores = {
        oid: object_overlap_score(mask, prediction, boundary_kernel=config.boundary_kernel)
        for oid, mask in object_masks.items()
    }
    foreground_overlap = _mean_or_none(object_scores[oid] for oid in foreground_ids)
    background_overlap = _mean_or_none(object_scores[oid] for oid in background_ids)
    target_mask = _union_masks(
        (mask for oid, mask in object_masks.items() if oid in (foreground_ids | background_ids)),
        shape,
    )
    pred = prediction > 0
    target = target_mask > 0

    return FrameEvaluation(
        frame_index=tier.frame.frame_index,
        foreground_total=len(foreground_ids),
        foreground_represented=foreground_represented,
        background_total=len(background_ids),
        background_represented=background_represented,
        represented_ids=represented_ids,
        foreground_ids=foreground_ids,
        background_ids=background_ids,
        foreground_overlap=foreground_overlap,
        background_overlap=background_overlap,
        prediction_pixels=int(np.count_nonzero(pred)),
        target_pixels=int(np.count_nonzero(target)),
        target_overlap_pixels=int(np.count_nonzero(pred & target)),
        frame_pixels=int(np.prod(shape)),
    )


def object_overlap_score(
    object_mask: np.ndarray,
    prediction_mask: np.ndarray,
    *,
    boundary_kernel: int = EvaluationConfig().boundary_kernel,
) -> float:
    """Return the best filled-mask or outline overlap without thresholding."""

    obj = as_binary(object_mask) > 0
    pred = as_binary(prediction_mask) > 0
    if pred.shape[:2] != obj.shape[:2]:
        pred = resize_like(pred.astype(np.uint8), obj.astype(np.uint8)) > 0
    if not np.any(obj):
        return 0.0

    fill_score = _overlap_score(obj, pred)
    boundary = object_boundary_band(obj.astype(np.uint8), kernel_size=boundary_kernel) > 0
    boundary_score = _overlap_score(boundary, pred)
    return max(fill_score, boundary_score)


def object_representation_score(
    object_mask: np.ndarray,
    prediction_mask: np.ndarray,
    *,
    config: EvaluationConfig = EvaluationConfig(),
) -> float:
    """Return nonzero representation score when output overlaps mask or outline.

    Filled-mask overlap is normalized by object area. Boundary overlap is
    normalized by a dilated boundary band, which lets outline renderers receive
    credit without requiring filled output.
    """

    obj = as_binary(object_mask) > 0
    pred = as_binary(prediction_mask) > 0
    if pred.shape[:2] != obj.shape[:2]:
        pred = resize_like(pred.astype(np.uint8), obj.astype(np.uint8)) > 0
    if not np.any(obj):
        return 0.0

    fill_score = _overlap_score(obj, pred)
    boundary = object_boundary_band(obj.astype(np.uint8), kernel_size=config.boundary_kernel) > 0
    boundary_score = _overlap_score(boundary, pred)
    score = max(fill_score, boundary_score)
    overlap_pixels = int(np.count_nonzero(pred & (obj | boundary)))
    if score >= config.min_overlap_fraction and overlap_pixels >= config.min_overlap_pixels:
        return score
    return 0.0


def object_boundary_band(mask: np.ndarray, *, kernel_size: int) -> np.ndarray:
    """Return a dilated object boundary band for outline-compatible matching."""

    binary = as_binary(mask)
    kernel_size = max(int(kernel_size), 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
    return as_binary(cv2.dilate(gradient, kernel, iterations=1))


def track_dropout_rate(frame_results: Sequence[FrameEvaluation]) -> float:
    """Return visible target object-frame miss rate.

    The score is the fraction of foreground/context object appearances that are
    visible in VISOR but not represented in the simplified output. It is bounded
    to [0, 1], where 0 means no visible target appearances were dropped and 1
    means every visible target appearance was dropped.
    """

    visible_count = 0
    missing_count = 0
    for result in frame_results:
        target_ids = result.foreground_ids | result.background_ids
        visible_count += len(target_ids)
        missing_count += len(target_ids - result.represented_ids)
    return missing_count / visible_count if visible_count else 0.0


def track_fragmentation_rate(frame_results: Sequence[FrameEvaluation]) -> float:
    """Return fraction of represented tracks with an internal representation gap.

    This complements object-frame miss rate. A track is fragmented when it is
    represented, then missed while still visible, then represented again.
    """

    sequences = _target_track_sequences(frame_results)
    represented_tracks = [states for states in sequences.values() if any(states)]
    if not represented_tracks:
        return 0.0
    fragmented = sum(1 for states in represented_tracks if _has_dropout(states))
    return fragmented / len(represented_tracks)


def recall_auc(points, *, load_cap: float) -> float | None:
    """Return normalized AUC for recall as a function of active-area load."""

    usable = sorted(
        (float(load), float(recall))
        for load, recall in points
        if recall is not None and np.isfinite(load) and np.isfinite(recall)
    )
    if load_cap <= 0 or not usable:
        return None

    clipped = [(0.0, 0.0)]
    for load, recall in usable:
        if load < 0:
            continue
        clipped.append((min(load, load_cap), np.clip(recall, 0.0, 1.0)))
    clipped.sort(key=lambda item: item[0])

    envelope = []
    best = 0.0
    for load, recall in clipped:
        best = max(best, recall)
        if envelope and np.isclose(envelope[-1][0], load):
            envelope[-1] = (load, max(envelope[-1][1], best))
        else:
            envelope.append((load, best))
    if envelope[-1][0] < load_cap:
        envelope.append((load_cap, envelope[-1][1]))

    area = 0.0
    previous_load, previous_recall = envelope[0]
    for load, recall in envelope[1:]:
        width = max(load - previous_load, 0.0)
        area += width * (previous_recall + recall) / 2.0
        previous_load, previous_recall = load, recall
    return float(area / load_cap)


def spv_load(frames: Sequence[np.ndarray | str | Path]) -> float:
    """Mean normalized intensity over a video-like frame sequence."""

    images = _load_gray_sequence(frames)
    if not images:
        return 0.0
    stack = np.stack([_normalize_fixed(frame) for frame in images], axis=0)
    return float(np.mean(stack))


def active_area(frames: Sequence[np.ndarray | str | Path], *, threshold: float = 0.05) -> float:
    """Mean fraction of pixels above a normalized intensity threshold."""

    images = _load_gray_sequence(frames)
    if not images:
        return 0.0
    stack = np.stack([_normalize_fixed(frame) for frame in images], axis=0)
    return float(np.mean(stack > threshold))


def flow_compensated_flicker(
    rgb_frames: Sequence[np.ndarray | str | Path],
    simplified_frames: Sequence[np.ndarray | str | Path],
) -> float:
    """Mean absolute simplified-output change after optical-flow warping."""

    rgbs = _load_rgb_sequence(rgb_frames)
    simplified = _load_gray_sequence(simplified_frames)
    if len(rgbs) != len(simplified):
        raise ValueError(f"Expected equal RGB/simplified counts, got {len(rgbs)} and {len(simplified)}.")
    if len(rgbs) < 2:
        return 0.0

    diffs = []
    for previous_rgb, current_rgb, previous_simplified, current_simplified in zip(
        rgbs[:-1],
        rgbs[1:],
        simplified[:-1],
        simplified[1:],
    ):
        previous_gray = cv2.cvtColor(previous_rgb, cv2.COLOR_RGB2GRAY)
        current_gray = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            current_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        warped = _warp_forward(previous_simplified, flow)
        diffs.append(float(np.mean(np.abs(_normalize_fixed(warped) - _normalize_fixed(current_simplified)))))
    return float(np.mean(diffs))


def _warp_forward(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    map_x = (xs - flow[:, :, 0]).astype(np.float32)
    map_y = (ys - flow[:, :, 1]).astype(np.float32)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _prediction_mask(frame: np.ndarray, threshold: int) -> np.ndarray:
    return np.where(np.asarray(frame) > threshold, 255, 0).astype(np.uint8)


def _overlap_score(support: np.ndarray, pred: np.ndarray) -> float:
    support_count = int(np.count_nonzero(support))
    if support_count == 0:
        return 0.0
    return float(np.count_nonzero(support & pred) / support_count)


def _has_dropout(states: Sequence[bool]) -> bool:
    seen_represented = False
    missing_after_seen = False
    for represented in states:
        if represented:
            if missing_after_seen:
                return True
            seen_represented = True
        elif seen_represented:
            missing_after_seen = True
    return False


def _target_track_sequences(frame_results: Sequence[FrameEvaluation]) -> dict[str, list[bool]]:
    sequences: dict[str, list[bool]] = {}
    for result in frame_results:
        for track_id in result.foreground_ids | result.background_ids:
            sequences.setdefault(track_id, []).append(track_id in result.represented_ids)
    return sequences


def _quality_filtered_annotations(annotations, quality: Literal["gold", "pseudo", "all"]):
    if quality == "all":
        return tuple(annotations)
    if quality == "gold":
        return tuple(annotation for annotation in annotations if _is_gold_annotation(annotation))
    if quality == "pseudo":
        return tuple(annotation for annotation in annotations if annotation.mask_type == 0)
    raise ValueError(f"Unknown annotation quality: {quality}")


def _is_gold_annotation(annotation) -> bool:
    return annotation.mask_type is None or annotation.mask_type == 1


def _ratio_or_none(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _mean_or_none(values) -> float | None:
    values = [float(value) for value in values]
    return float(np.mean(values)) if values else None


def _weighted_frame_mean(values) -> float | None:
    numerator = 0.0
    denominator = 0
    for value, weight in values:
        if value is None or weight <= 0:
            continue
        numerator += float(value) * int(weight)
        denominator += int(weight)
    return numerator / denominator if denominator else None


def _union_masks(masks, shape: tuple[int, int]) -> np.ndarray:
    masks = list(masks)
    if not masks:
        return np.zeros(shape, dtype=np.uint8)
    output = np.zeros(masks[0].shape[:2], dtype=np.uint8)
    for mask in masks:
        output = np.maximum(output, as_binary(mask))
    return output


def _normalize_fixed(frame: np.ndarray) -> np.ndarray:
    values = np.asarray(frame, dtype=np.float32)
    if values.size == 0:
        return values
    if values.max(initial=0) > 1.0:
        values = values / 255.0
    return np.clip(values, 0.0, 1.0)


def _load_gray_sequence(frames: Sequence[np.ndarray | str | Path] | None) -> list[np.ndarray]:
    if frames is None:
        return []
    return [_load_gray_frame(frame) for frame in frames]


def _load_gray_frame(frame: np.ndarray | str | Path) -> np.ndarray:
    if isinstance(frame, np.ndarray):
        image = frame
    else:
        image = load_gray(frame)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def _load_rgb_sequence(frames: Sequence[np.ndarray | str | Path]) -> list[np.ndarray]:
    images = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            image = frame
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = load_rgb(frame)
        images.append(image.astype(np.uint8) if image.dtype != np.uint8 else image)
    return images
