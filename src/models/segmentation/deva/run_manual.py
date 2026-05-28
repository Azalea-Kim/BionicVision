"""Run DEVA from EPIC/VISOR class-labeled manual masks."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys
import zlib

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from datasets.epic_kitchens.annotations import (
    EpicFrame,
    VisorObject,
    build_clip_tiers,
    is_hand,
    load_visor_annotations,
    object_id,
    rasterize_object,
)
from simplification.masks import resize_like

from .run_automatic import DEVA_ROOT, cpu_cuda_shim


SCENE_LABELS = {
    "cabinet",
    "ceiling",
    "counter",
    "counter top",
    "countertop",
    "cupboard",
    "door",
    "drawer",
    "floor",
    "fridge",
    "hob",
    "microwave",
    "oven",
    "refrigerator",
    "shelf",
    "sink",
    "stove",
    "table",
    "wall",
    "window",
}

GROUP_COLORS_RGB = {
    "arms": (255, 0, 0),
    "objects": (0, 255, 0),
    "scenes": (0, 0, 255),
}


@dataclass(frozen=True)
class ManualDevaOutputs:
    """Paths written by manual class-guided DEVA."""

    raw_annotation_dir: Path
    raw_visualization_dir: Path
    prompts_path: Path
    pred_json_path: Path


def run_deva_manual(
    *,
    frames_dir: Path,
    visor_annotation_path: Path,
    output_dir: Path,
    sampled_source_indices: list[int],
    size: int = 360,
    detection_every: int = 1,
    memory_reset_interval: int = 4,
) -> ManualDevaOutputs:
    """Run DEVA with class-labeled VISOR masks as manual detections.

    This is the prompted/manual variant used by combination1: each sampled frame
    supplies all VISOR classes present in that frame. DEVA tracks those labeled
    regions, and we save a semantic three-color render of arms/objects/scenes.
    """

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(DEVA_ROOT))
    from torch.utils.data import DataLoader
    from deva.inference.data.simple_video_reader import SimpleVideoReader, no_collate
    from deva.inference.demo_utils import get_input_frame_for_deva
    from deva.inference.eval_args import add_common_eval_args, get_model_and_config
    from deva.inference.inference_core import DEVAInferenceCore
    from deva.inference.object_info import ObjectInfo
    from deva.ext.ext_eval_args import add_text_default_args

    frames = merge_duplicate_frames(load_visor_annotations(visor_annotation_path))
    frames_by_index = {frame.frame_index: frame for frame in frames}
    tiers_by_index = {tier.frame.frame_index: tier for tier in build_clip_tiers(frames)}
    prompts = class_prompts_for_indices(frames_by_index, sampled_source_indices)
    category_by_label = {label: index for index, label in enumerate(prompts)}
    group_by_object_id = {
        stable_deva_object_id(frame.video_id, object_id(annotation), normalized_prompt(annotation.name)): label_group(annotation, tier)
        for source_index in sampled_source_indices
        for frame in [frames_by_index.get(source_index)]
        for tier in [tiers_by_index.get(source_index)]
        if frame is not None
        for annotation in frame.annotations
    }

    argv = [
        "deva-manual",
        "--model",
        str(DEVA_ROOT / "saves" / "DEVA-propagation.pth"),
        "--img_path",
        str(frames_dir),
        "--output",
        str(output_dir),
        "--size",
        str(size),
        "--temporal_setting",
        "online",
        "--detection_every",
        str(detection_every),
    ]
    parser = argparse.ArgumentParser()
    add_common_eval_args(parser)
    add_text_default_args(parser)

    old_argv = sys.argv
    sys.argv = argv
    use_cuda = torch.cuda.is_available()
    device_context = contextlib.nullcontext() if use_cuda else cpu_cuda_shim()
    try:
        with device_context:
            deva_model, cfg, args = get_model_and_config(parser)
            cfg["temporal_setting"] = args.temporal_setting.lower()
            cfg["prompt"] = ".".join(prompts)
            video_reader = SimpleVideoReader(cfg["img_path"])
            loader = DataLoader(video_reader, batch_size=None, collate_fn=no_collate, num_workers=0)
            video_length = len(loader)
            cfg["enable_long_term_count_usage"] = (
                cfg["enable_long_term"]
                and (
                    video_length / (cfg["max_mid_term_frames"] - cfg["min_mid_term_frames"])
                    * cfg["num_prototypes"]
                )
                >= cfg["max_long_term_elements"]
            )

            def new_deva_core() -> DEVAInferenceCore:
                core = DEVAInferenceCore(deva_model, config=cfg)
                core.next_voting_frame = args.num_voting_frames - 1
                core.enabled_long_id()
                return core

            deva = new_deva_core()
            annotation_dir = output_dir / "Annotations"
            visualization_dir = output_dir / "Visualizations"
            annotation_dir.mkdir(parents=True, exist_ok=True)
            visualization_dir.mkdir(parents=True, exist_ok=True)
            frame_records = []

            for ti, (frame_rgb, image_path) in enumerate(loader):
                if memory_reset_interval > 0 and ti > 0 and ti % memory_reset_interval == 0:
                    del deva
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    deva = new_deva_core()
                source_index = sampled_source_indices[ti]
                frame_record = frames_by_index.get(source_index)
                tier = tiers_by_index.get(source_index)
                height, width = frame_rgb.shape[:2]
                image = get_input_frame_for_deva(frame_rgb, cfg["size"])
                manual_mask, segments = manual_detection_for_frame(
                    frame_record,
                    tier,
                    shape=(height, width),
                    resized_shape=image.shape[1:],
                    category_by_label=category_by_label,
                    object_info_cls=ObjectInfo,
                )
                if ti % detection_every == 0 or not deva.memory.engaged:
                    prob = deva.incorporate_detection(image, manual_mask, segments)
                else:
                    prob = deva.step(image, None, None)
                grouped = group_probability_mask(
                    prob,
                    deva.object_manager.tmp_id_to_obj,
                    group_by_object_id,
                    shape=(height, width),
                    need_resize=cfg["size"] > 0,
                )
                annotation_rgb = semantic_annotation_rgb(grouped)
                visualization_rgb = semantic_visualization_rgb(frame_rgb, annotation_rgb)
                name = Path(image_path).name
                cv2.imwrite(str(annotation_dir / f"{Path(name).stem}.png"), cv2.cvtColor(annotation_rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(visualization_dir / f"{Path(name).stem}.png"), cv2.cvtColor(visualization_rgb, cv2.COLOR_RGB2BGR))
                frame_records.append(
                    {
                        "frame": name,
                        "source_frame": source_index,
                        "semantic_counts": {
                            group_name: int(np.count_nonzero(mask))
                            for group_name, mask in grouped.items()
                        },
                    }
                )
    finally:
        sys.argv = old_argv

    prompts_path = output_dir / "prompts.json"
    prompts_path.write_text(
        json.dumps(
            {
                "prompts": prompts,
                "semantic_colors_rgb": GROUP_COLORS_RGB,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    pred_json_path = output_dir / "pred.json"
    if "frame_records" in locals():
        pred_json_path.write_text(json.dumps({"frames": frame_records}, indent=2) + "\n", encoding="utf-8")

    return ManualDevaOutputs(
        raw_annotation_dir=output_dir / "Annotations",
        raw_visualization_dir=output_dir / "Visualizations",
        prompts_path=prompts_path,
        pred_json_path=pred_json_path,
    )


def manual_detection_for_frame(
    frame: EpicFrame | None,
    tier,
    *,
    shape: tuple[int, int],
    resized_shape: tuple[int, int],
    category_by_label: dict[str, int],
    object_info_cls,
) -> tuple[torch.Tensor, list]:
    """Build one DEVA detection mask and segment metadata from VISOR."""

    detection = np.zeros(shape, dtype=np.int64)
    segments = []
    if frame is not None:
        for annotation in frame.annotations:
            label = normalized_prompt(annotation.name)
            category_id = category_by_label[label]
            mask = rasterize_object(annotation, shape)
            if mask.shape[:2] != shape:
                mask = resize_like(mask, np.zeros(shape, dtype=np.uint8))
            stable_id = stable_deva_object_id(frame.video_id, object_id(annotation), label)
            detection[mask > 0] = stable_id
            group = label_group(annotation, tier)
            segments.append(
                object_info_cls(
                    id=stable_id,
                    category_id=category_id,
                    isthing=group != "scenes",
                    score=1.0,
                )
            )

    if resized_shape != shape:
        detection = cv2.resize(
            detection.astype(np.float32),
            (resized_shape[1], resized_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int64)
    tensor = torch.from_numpy(detection)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor, segments


def group_probability_mask(
    prob: torch.Tensor,
    tmp_id_to_obj: dict[int, object],
    group_by_object_id: dict[int, str],
    *,
    shape: tuple[int, int],
    need_resize: bool,
) -> dict[str, np.ndarray]:
    """Convert DEVA probability output into grouped binary masks."""

    if need_resize:
        prob = F.interpolate(prob.unsqueeze(1), shape, mode="bilinear", align_corners=False)[:, 0]
    tmp_mask = torch.argmax(prob, dim=0).detach().cpu().numpy()
    grouped = {name: np.zeros(shape, dtype=np.uint8) for name in ("arms", "objects", "scenes")}
    for tmp_id, obj in tmp_id_to_obj.items():
        group = group_by_object_id.get(obj.id)
        if group is None:
            continue
        grouped[group][tmp_mask == tmp_id] = 255
    return grouped


def semantic_annotation_rgb(grouped: dict[str, np.ndarray]) -> np.ndarray:
    """Render semantic arms/objects/scenes masks as a single RGB label image."""

    first = next(iter(grouped.values()))
    output = np.zeros((*first.shape[:2], 3), dtype=np.uint8)
    for group_name in ("scenes", "objects", "arms"):
        mask = grouped.get(group_name)
        if mask is None:
            continue
        output[mask > 0] = GROUP_COLORS_RGB[group_name]
    return output


def semantic_visualization_rgb(frame_rgb: np.ndarray, annotation_rgb: np.ndarray) -> np.ndarray:
    """Overlay the three semantic DEVA colors on the source RGB frame."""

    frame = frame_rgb.astype(np.uint8)
    if frame.shape[:2] != annotation_rgb.shape[:2]:
        annotation_rgb = cv2.resize(
            annotation_rgb,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    active = np.any(annotation_rgb > 0, axis=2)
    output = frame.copy()
    output[active] = np.clip(0.35 * frame[active] + 0.65 * annotation_rgb[active], 0, 255).astype(np.uint8)
    return output


def class_prompts_for_indices(frames_by_index: dict[int, EpicFrame], indices: list[int]) -> list[str]:
    labels = []
    seen = set()
    for index in indices:
        frame = frames_by_index.get(index)
        if frame is None:
            continue
        for annotation in frame.annotations:
            label = normalized_prompt(annotation.name)
            if label in seen:
                continue
            seen.add(label)
            labels.append(label)
    return labels


def label_group(annotation: VisorObject, tier) -> str:
    if is_hand(annotation):
        return "arms"
    oid = object_id(annotation)
    if tier is not None and oid in tier.foreground_ids:
        return "objects"
    label = normalized_prompt(annotation.name)
    if label in SCENE_LABELS:
        return "scenes"
    return "objects"


def normalized_prompt(label: str) -> str:
    return " ".join(str(label).lower().strip().replace("_", " ").replace("-", " ").split())


def stable_deva_object_id(video_id: str, oid: str, label: str) -> int:
    raw = f"{video_id}:{oid}:{label}".encode("utf-8")
    return 256 + (zlib.crc32(raw) % (256**3 - 256))


def merge_duplicate_frames(frames: list[EpicFrame]) -> list[EpicFrame]:
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
                annotation_size=first.annotation_size,
            )
        )
    return merged
