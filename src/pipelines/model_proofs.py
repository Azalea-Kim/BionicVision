"""Generate visual proof outputs for model adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw

from datasets.epic_kitchens.actions import load_actions, object_labels_for_window
from models.depth.transformer_depth import TransformerDepthEstimator
from models.saliency.spectral_residual import compute_spectral_residual_saliency
from models.segmentation.clipseg import ClipSegSegmenter


DEFAULT_CLIP_DIR = Path("data/epic_kitchens/clips_10s")
DEFAULT_ANNOTATION_ROOT = Path("data/epic_kitchens/epic-kitchens-100-annotations")
DEFAULT_OUTPUT_DIR = Path("outputs/model_proofs")


def run_model_proofs(
    *,
    clip_dir: Path = DEFAULT_CLIP_DIR,
    annotation_root: Path = DEFAULT_ANNOTATION_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    samples_per_clip: int = 2,
    max_labels: int = 8,
    depth_model: str | None = None,
    clipseg_model: str | None = None,
) -> dict[str, object]:
    """Run depth, saliency, and segmentation visual checks on clip frames."""

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    depth_dir = output_dir / "depth"
    saliency_dir = output_dir / "saliency"
    segmentation_dir = output_dir / "segmentation"
    sheet_dir = output_dir / "sheets"
    for folder in (frames_dir, depth_dir, saliency_dir, segmentation_dir, sheet_dir):
        folder.mkdir(parents=True, exist_ok=True)

    actions = load_actions(annotation_root)
    clips = sorted(clip_dir.glob("*.mp4"))
    depth = TransformerDepthEstimator(model_name=depth_model) if depth_model else TransformerDepthEstimator()
    segmenter = ClipSegSegmenter(model_name=clipseg_model) if clipseg_model else ClipSegSegmenter()

    manifest: dict[str, object] = {"clips": []}
    for clip_path in clips:
        clip_info = parse_clip_name(clip_path)
        labels = object_labels_for_window(
            actions,
            video_id=clip_info["video_id"],
            start_seconds=clip_info["start_seconds"],
            stop_seconds=clip_info["stop_seconds"],
        )[:max_labels]
        sample_paths = sample_video_frames(clip_path, frames_dir, samples_per_clip=samples_per_clip)
        clip_record = {
            "clip": str(clip_path),
            "video_id": clip_info["video_id"],
            "source_start_seconds": clip_info["start_seconds"],
            "source_stop_seconds": clip_info["stop_seconds"],
            "labels": labels,
            "samples": [],
        }

        for frame_path in sample_paths:
            image_bgr = cv2.imread(str(frame_path))
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            stem = frame_path.stem

            depth_map = depth.predict(pil_image)
            depth_path = depth_dir / f"{stem}_depth.png"
            write_heatmap(depth_map, depth_path, cv2.COLORMAP_TURBO)

            saliency = compute_spectral_residual_saliency(image_bgr)
            saliency_path = saliency_dir / f"{stem}_saliency.png"
            write_heatmap(saliency, saliency_path, cv2.COLORMAP_INFERNO)

            masks = segmenter.predict(pil_image, labels)
            segmentation_path = segmentation_dir / f"{stem}_clipseg_overlay.png"
            write_segmentation_overlay(pil_image, masks, segmentation_path)
            sheet_path = sheet_dir / f"{stem}_proof_sheet.jpg"
            write_proof_sheet(
                [
                    ("frame", frame_path),
                    ("depth", depth_path),
                    ("saliency", saliency_path),
                    ("clipseg", segmentation_path),
                ],
                sheet_path,
            )

            clip_record["samples"].append(
                {
                    "frame": str(frame_path),
                    "depth": str(depth_path),
                    "saliency": str(saliency_path),
                    "segmentation": str(segmentation_path),
                    "sheet": str(sheet_path),
                }
            )
        manifest["clips"].append(clip_record)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_clip_name(path: Path) -> dict[str, object]:
    """Parse filenames like `P22_113_0011_0021.mp4`."""

    match = re.match(r"(?P<participant>P\d+)_(?P<video>\d+)_(?P<start>\d{4})_(?P<end>\d{4})", path.stem)
    if not match:
        raise ValueError(f"Unexpected clip filename: {path.name}")
    start = mmss_to_seconds(match.group("start"))
    end = mmss_to_seconds(match.group("end"))
    return {
        "video_id": f"{match.group('participant')}_{match.group('video')}",
        "start_seconds": start,
        "stop_seconds": end,
    }


def mmss_to_seconds(value: str) -> int:
    return int(value[:2]) * 60 + int(value[2:])


def sample_video_frames(video_path: Path, output_dir: Path, *, samples_per_clip: int) -> list[Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return []
    indices = np.linspace(0, max(frame_count - 1, 0), num=samples_per_clip + 2, dtype=int)[1:-1]
    paths: list[Path] = []
    for sample_index, frame_index in enumerate(indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            continue
        output_path = output_dir / f"{video_path.stem}_sample{sample_index:02d}.jpg"
        cv2.imwrite(str(output_path), frame)
        paths.append(output_path)
    cap.release()
    return paths


def write_heatmap(values: np.ndarray, output_path: Path, colormap: int) -> None:
    image = np.clip(values * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(image, colormap)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), colored)


def write_segmentation_overlay(image: Image.Image, masks: dict[str, np.ndarray], output_path: Path) -> None:
    rgb = image.convert("RGBA")
    overlay = Image.new("RGBA", rgb.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    palette = [
        (255, 64, 64, 110),
        (64, 180, 255, 110),
        (90, 220, 120, 110),
        (255, 210, 64, 110),
        (190, 120, 255, 110),
        (255, 130, 60, 110),
        (80, 220, 220, 110),
        (255, 90, 180, 110),
    ]
    for index, (label, mask) in enumerate(masks.items()):
        binary = mask >= 0.5
        if not binary.any():
            continue
        color = palette[index % len(palette)]
        color_layer = np.zeros((rgb.height, rgb.width, 4), dtype=np.uint8)
        color_layer[binary] = color
        overlay = Image.alpha_composite(overlay, Image.fromarray(color_layer, mode="RGBA"))
        y, x = np.argwhere(binary).mean(axis=0)
        draw.text((float(x), float(y)), label, fill=(255, 255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(rgb, overlay).convert("RGB").save(output_path)


def write_proof_sheet(items: list[tuple[str, Path]], output_path: Path) -> None:
    thumbs: list[tuple[str, Image.Image]] = []
    for label, path in items:
        image = Image.open(path).convert("RGB")
        image.thumbnail((360, 240))
        tile = Image.new("RGB", (360, 270), "white")
        tile.paste(image, ((360 - image.width) // 2, 24))
        draw = ImageDraw.Draw(tile)
        draw.text((10, 6), label, fill=(0, 0, 0))
        thumbs.append((label, tile))
    sheet = Image.new("RGB", (360 * len(thumbs), 270), "white")
    for index, (_, tile) in enumerate(thumbs):
        sheet.paste(tile, (360 * index, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visual model proof outputs.")
    parser.add_argument("--clip-dir", type=Path, default=DEFAULT_CLIP_DIR)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples-per-clip", type=int, default=2)
    parser.add_argument("--max-labels", type=int, default=8)
    parser.add_argument("--depth-model", default=None)
    parser.add_argument("--clipseg-model", default=None)
    args = parser.parse_args()
    run_model_proofs(
        clip_dir=args.clip_dir,
        annotation_root=args.annotation_root,
        output_dir=args.output_dir,
        samples_per_clip=args.samples_per_clip,
        max_labels=args.max_labels,
        depth_model=args.depth_model,
        clipseg_model=args.clipseg_model,
    )


if __name__ == "__main__":
    main()
