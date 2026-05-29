"""Run DEVA with fixed text-prompted class groups."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[4]
DEVA_ROOT = ROOT / "external" / "model_sources" / "segmentation" / "Tracking-Anything-with-DEVA"

GROUP_COLORS_RGB = {
    "arms": (255, 0, 0),
    "objects": (0, 255, 0),
    "scenes": (0, 0, 255),
}

ARM_PROMPTS = (
    "hand",
    "left hand",
    "right hand",
    "arm",
    "forearm",
)

OBJECT_PROMPTS = (
    "backpack",
    "handbag",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "dining table",
    "tv",
    "laptop",
    "remote",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "vase",
    "scissors",
    "toothbrush",
)

SCENE_PROMPTS = (
    "wall",
    "floor",
    "ceiling",
    "window",
    "cabinet",
    "door",
    "table",
    "shelf",
    "counter",
    "countertop",
    "stove",
    "microwave",
    "oven",
    "refrigerator",
    "sink",
)


@dataclass(frozen=True)
class PromptGroup:
    name: str
    prompts: tuple[str, ...]


DEFAULT_PROMPT_GROUPS = (
    PromptGroup("arms", ARM_PROMPTS),
    PromptGroup("objects", OBJECT_PROMPTS),
    PromptGroup("scenes", SCENE_PROMPTS),
)


@dataclass(frozen=True)
class ManualDevaOutputs:
    """Paths written by fixed-prompt DEVA."""

    raw_annotation_dir: Path
    raw_visualization_dir: Path
    prompts_path: Path
    pred_json_path: Path


@contextlib.contextmanager
def cpu_cuda_shim():
    """Let DEVA's CUDA-oriented loader run on CPU when CUDA is unavailable."""

    original_tensor_cuda = torch.Tensor.cuda
    original_module_cuda = torch.nn.Module.cuda
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self
    try:
        yield
    finally:
        torch.Tensor.cuda = original_tensor_cuda
        torch.nn.Module.cuda = original_module_cuda


@contextlib.contextmanager
def groundingdino_python_attention_fallback():
    """Use GroundingDINO's PyTorch attention path when compiled ops are absent."""

    try:
        from groundingdino.models.GroundingDINO import ms_deform_attn
    except ImportError:
        yield
        return

    if hasattr(ms_deform_attn, "_C"):
        yield
        return

    original_cuda_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        yield
    finally:
        torch.cuda.is_available = original_cuda_available


def run_deva_manual(
    *,
    frames_dir: Path,
    output_dir: Path,
    prompt_groups: tuple[PromptGroup, ...] = DEFAULT_PROMPT_GROUPS,
    size: int = 360,
    detection_every: int = 1,
    memory_reset_interval: int = 4,
    dino_threshold: float = 0.35,
    dino_nms_threshold: float = 0.8,
    sam_variant: str = "original",
) -> ManualDevaOutputs:
    """Run text-prompted DEVA with fixed class groups.

    This is the only DEVA mode used by pipelines. It uses GroundingDINO + SAM
    detections for fixed prompts and DEVA for propagation; it does not use
    automatic SAM grid prompting or VISOR ground-truth masks.
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
    from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args
    from deva.ext.grounding_dino import get_grounding_dino_model
    from deva.ext.with_text_processor import make_segmentation_with_text

    prompts, group_by_category_id = flatten_prompt_groups(prompt_groups)
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
        "--prompt",
        ".".join(prompts),
        "--GROUNDING_DINO_CONFIG_PATH",
        str(DEVA_ROOT / "saves" / "GroundingDINO_SwinT_OGC.py"),
        "--GROUNDING_DINO_CHECKPOINT_PATH",
        str(DEVA_ROOT / "saves" / "groundingdino_swint_ogc.pth"),
        "--SAM_CHECKPOINT_PATH",
        str(DEVA_ROOT / "saves" / "sam_vit_h_4b8939.pth"),
        "--DINO_THRESHOLD",
        str(dino_threshold),
        "--DINO_NMS_THRESHOLD",
        str(dino_nms_threshold),
        "--sam_variant",
        sam_variant,
    ]
    parser = argparse.ArgumentParser()
    add_common_eval_args(parser)
    add_ext_eval_args(parser)
    add_text_default_args(parser)

    old_argv = sys.argv
    sys.argv = argv
    use_cuda = torch.cuda.is_available()
    device_context = contextlib.nullcontext() if use_cuda else cpu_cuda_shim()
    detection_device = "cuda" if use_cuda else "cpu"
    try:
        with device_context:
            deva_model, cfg, args = get_model_and_config(parser)
            cfg["temporal_setting"] = args.temporal_setting.lower()
            cfg["prompt"] = ".".join(prompts)
            gd_model, sam_model = get_grounding_dino_model(cfg, detection_device)
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

            attention_context = groundingdino_python_attention_fallback() if use_cuda else contextlib.nullcontext()
            with attention_context:
                for ti, (frame_rgb, image_path) in enumerate(loader):
                    if memory_reset_interval > 0 and ti > 0 and ti % memory_reset_interval == 0:
                        del deva
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        deva = new_deva_core()
                    height, width = frame_rgb.shape[:2]
                    image = get_input_frame_for_deva(frame_rgb, cfg["size"])
                    if ti % detection_every == 0 or not deva.memory.engaged:
                        mask, segments_info = make_segmentation_with_text(
                            cfg,
                            frame_rgb,
                            gd_model,
                            sam_model,
                            prompts,
                            cfg["size"],
                        )
                        prob = deva.incorporate_detection(image, mask, segments_info)
                    else:
                        prob = deva.step(image, None, None)
                    grouped = group_probability_mask(
                        prob,
                        deva.object_manager.tmp_id_to_obj,
                        group_by_category_id,
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
                "prompt_groups": {
                    group.name: list(group.prompts)
                    for group in prompt_groups
                },
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


def flatten_prompt_groups(prompt_groups: tuple[PromptGroup, ...]) -> tuple[list[str], dict[int, str]]:
    prompts = []
    group_by_category_id = {}
    seen = set()
    for group in prompt_groups:
        if group.name not in GROUP_COLORS_RGB:
            raise ValueError(f"Unknown DEVA prompt group: {group.name}")
        for prompt in group.prompts:
            normalized = normalize_prompt(prompt)
            if normalized in seen:
                continue
            seen.add(normalized)
            group_by_category_id[len(prompts)] = group.name
            prompts.append(normalized)
    if not prompts:
        raise ValueError("DEVA prompt groups must contain at least one prompt.")
    return prompts, group_by_category_id


def normalize_prompt(label: str) -> str:
    return " ".join(str(label).lower().strip().replace("_", " ").replace("-", " ").split())


def group_probability_mask(
    prob: torch.Tensor,
    tmp_id_to_obj: dict[int, object],
    group_by_category_id: dict[int, str],
    *,
    shape: tuple[int, int],
    need_resize: bool,
) -> dict[str, np.ndarray]:
    """Convert DEVA probability output into grouped binary masks."""

    if need_resize:
        prob = F.interpolate(prob.unsqueeze(1), shape, mode="bilinear", align_corners=False)[:, 0]
    tmp_mask = torch.argmax(prob, dim=0).detach().cpu().numpy()
    grouped = {name: np.zeros(shape, dtype=np.uint8) for name in GROUP_COLORS_RGB}
    for tmp_id, obj in tmp_id_to_obj.items():
        if hasattr(obj, "vote_category_id"):
            category_id = obj.vote_category_id()
        else:
            category_id = getattr(obj, "category_id", None)
        if category_id is None:
            continue
        group = group_by_category_id.get(int(category_id))
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
