"""Han et al. 2021 baseline pipeline on EPIC-KITCHENS clips.

This module ports the operations in bionicvisionlab/2021-han-scene-simplification
for local EPIC-KITCHENS 10-second clips. The implementation keeps Han's depth,
saliency, segmentation, and combination mechanics intact. The only dataset
adaptation is the class lists used to filter ADE20K scene parsing and COCO
Detectron2 detections for indoor EPIC-KITCHENS scenes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageFilter
from skimage import morphology
import torch
import torchvision.transforms as transforms


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "external" / "model_sources"
PYTHON = ROOT / ".venv-models" / "bin" / "python"
DEFAULT_CLIP_DIR = ROOT / "data" / "epic_kitchens" / "clips_10s"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "han_baseline_epic10"

ADE20K_STRUCTURE_CLASSES = (
    0,  # wall
    3,  # floor
    5,  # ceiling
    8,  # window
    10,  # cabinet
    14,  # door
    15,  # table
    20,  # chair
    24,  # shelf
    46,  # counter
    47,  # sink
    50,  # refrigerator
    71,  # countertop
    72,  # stove
    118,  # oven
    124,  # microwave
)
COCO_IMPORTANT_CLASSES = (
    0,  # person
    24,  # backpack
    26,  # handbag
    39,  # bottle
    40,  # wine glass
    41,  # cup
    42,  # fork
    43,  # knife
    44,  # spoon
    45,  # bowl
    46,  # banana
    47,  # apple
    48,  # sandwich
    49,  # orange
    50,  # broccoli
    51,  # carrot
    52,  # hot dog
    53,  # pizza
    54,  # donut
    55,  # cake
    56,  # chair
    60,  # dining table
    62,  # tv
    63,  # laptop
    65,  # remote
    67,  # cell phone
    68,  # microwave
    69,  # oven
    70,  # toaster
    71,  # sink
    72,  # refrigerator
    73,  # book
    75,  # vase
    76,  # scissors
    79,  # toothbrush
)


@dataclass(frozen=True)
class HanBaselineConfig:
    target_fps: float = 20.0
    max_frames: int | None = None
    device: str = "cuda"
    depth_mode: str = "flipped_quad"
    monodepth_model_name: str = "mono+stereo_640x192"
    depth_min_brightness: int = 0
    depth_max_brightness: int = 180
    depth_clip_percentile: float = 90.0
    saliency_bins: int = 8
    saliency_threshold_fraction: float = 0.90
    structure_class_ids: tuple[int, ...] = ADE20K_STRUCTURE_CLASSES
    important_coco_classes: tuple[int, ...] = COCO_IMPORTANT_CLASSES
    structure_min_region_area: int = 16000
    hough_window: int = 10
    hough_min_line_length: int = 200
    hough_max_line_gap: int = 1
    detectron_score_threshold: float = 0.5


def ensure_cuda_if_requested(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return torch.device(device)


def extract_video_frames(clip_path: Path, output_dir: Path, target_fps: float, max_frames: int | None) -> list[Path]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open clip: {clip_path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or target_fps
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / source_fps if source_fps > 0 else 0
    sample_count = int(round(duration * target_fps)) if duration > 0 else frame_count
    if max_frames is not None:
        sample_count = min(sample_count, max_frames)
    sample_count = max(sample_count, 1)
    indices = np.linspace(0, max(frame_count - 1, 0), sample_count, dtype=int)

    frame_paths: list[Path] = []
    for index, source_index in enumerate(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(source_index))
        ok, frame = capture.read()
        if not ok:
            continue
        output_path = output_dir / f"frame{index:05d}.jpg"
        cv2.imwrite(str(output_path), frame)
        frame_paths.append(output_path)
    capture.release()
    if not frame_paths:
        raise RuntimeError(f"No frames extracted from {clip_path}")
    return frame_paths


def restrict_x_value(d: np.ndarray, max_d: float, min_d: float, target_max_d: float, target_min_d: float) -> np.ndarray:
    if max_d == min_d:
        return np.full_like(d, target_min_d, dtype=np.float32)
    k = (target_min_d - target_max_d) / (min_d - max_d)
    b = target_min_d - k * min_d
    return k * d + b


def depth_exponential_map(depth_map: np.ndarray, max_depth: float, min_depth: float, min_y: float, max_y: float) -> np.ndarray:
    depth_map = restrict_x_value(depth_map, max_depth, min_depth, 10, 2)
    a = (max_y - min_y) / (math.exp(-2) - math.exp(-10))
    b = max_y - a * math.exp(-2)
    return a * np.exp(-depth_map) + b


def depth_linear_map(depth_map: np.ndarray, max_depth: float, min_depth: float, min_y: float, max_y: float) -> np.ndarray:
    depth_map = restrict_x_value(depth_map, max_depth, min_depth, 10, 2)
    k = (min_y - max_y) / (10 - 2)
    b = min_y - k * 10
    return k * depth_map + b


def gen_image_brightness(
    depth_map: np.ndarray,
    sidewalk_mask: np.ndarray,
    mode: str,
    min_brightness: int,
    max_brightness: int,
) -> np.ndarray:
    if np.count_nonzero(sidewalk_mask) == 0:
        return np.zeros((sidewalk_mask.shape[0], sidewalk_mask.shape[1], 3), dtype=np.uint8)
    max_depth = float(depth_map[sidewalk_mask].max())
    min_depth = float(depth_map[sidewalk_mask].min())
    if mode == "exponential":
        one_channel = depth_exponential_map(depth_map, max_depth, min_depth, min_brightness, max_brightness)
    else:
        # The Han combination notebook passes mode="flipped_quad", which falls
        # through to the default linear mapper in depth_to_image.py.
        one_channel = depth_linear_map(depth_map, max_depth, min_depth, min_brightness, max_brightness)
    one_channel[sidewalk_mask == False] = 0
    return np.dstack([one_channel] * 3).astype(np.uint8)


def build_depth_frames(frame_dir: Path, output_dir: Path, config: HanBaselineConfig) -> list[Path]:
    monodepth_root = EXTERNAL / "depth" / "monodepth2"
    work_dir = output_dir / "monodepth2_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    for frame in sorted(frame_dir.glob("*.jpg")):
        shutil.copy2(frame, work_dir / frame.name)

    command = [
        str(PYTHON),
        "test_simple.py",
        "--image_path",
        str(work_dir),
        "--model_name",
        config.monodepth_model_name,
        "--ext",
        "jpg",
        "--pred_metric_depth",
    ]
    if config.device == "cpu":
        command.append("--no_cuda")
    subprocess.run(command, cwd=monodepth_root, check=True)

    depth_frame_dir = output_dir / "frames"
    if depth_frame_dir.exists():
        shutil.rmtree(depth_frame_dir)
    depth_frame_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for depth_path in sorted(work_dir.glob("*_depth.npy")):
        depth = np.squeeze(np.load(depth_path)).astype(np.float32)
        vmax = np.percentile(depth, config.depth_clip_percentile)
        depth = depth.copy()
        depth[depth > vmax] = vmax
        image = gen_image_brightness(
            depth,
            np.ones(depth.shape, dtype=bool),
            mode=config.depth_mode,
            min_brightness=config.depth_min_brightness,
            max_brightness=config.depth_max_brightness,
        )
        out = depth_frame_dir / f"{depth_path.stem.removesuffix('_depth')}.png"
        cv2.imwrite(str(out), image)
        output_paths.append(out)
    return output_paths


def build_saliency_frames(frame_paths: list[Path], output_dir: Path, config: HanBaselineConfig) -> list[Path]:
    from deepgaze.saliency_map import FasaSaliencyMapping

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise FileNotFoundError(frame_path)
        h, w = frame.shape[:2]
        saliency = FasaSaliencyMapping(h, w).returnMask(frame, tot_bins=config.saliency_bins, format="BGR2LAB")
        saliency = cv2.GaussianBlur(saliency, (3, 3), 1)
        out = output_dir / f"{frame_path.stem}.png"
        cv2.imwrite(str(out), saliency)
        output_paths.append(out)
    return output_paths


def build_mit_scene_module(device: torch.device):
    from mit_semseg.models import ModelBuilder, SegmentationModule

    weights = ROOT / "data" / "model_weights" / "mit_scene_parsing" / "ade20k-resnet50dilated-ppm_deepsup"
    encoder = ModelBuilder.build_encoder(arch="resnet50dilated", fc_dim=2048, weights=str(weights / "encoder_epoch_20.pth"))
    decoder = ModelBuilder.build_decoder(
        arch="ppm_deepsup",
        fc_dim=2048,
        num_class=150,
        weights=str(weights / "decoder_epoch_20.pth"),
        use_softmax=True,
    )
    return SegmentationModule(encoder, decoder, torch.nn.NLLLoss(ignore_index=-1)).to(device).eval()


def build_detectron_predictor(device: torch.device, score_threshold: float):
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = device.type
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    return DefaultPredictor(cfg)


def get_houghlines(edges: np.ndarray, config: HanBaselineConfig) -> np.ndarray:
    kernel = np.ones((10, 10), np.uint8)
    lines = cv2.HoughLinesP(edges.astype("uint8"), 1, np.pi / 180, 15, minLineLength=config.hough_min_line_length, maxLineGap=config.hough_max_line_gap)
    edge_combined = np.zeros(edges.shape, dtype=np.uint8)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if np.abs(x1 - x2) > 5 and np.abs(y1 - y2) > 5:
                    cv2.line(edge_combined, (x1, y1), (x2, y2), color=255)
        edge_combined = cv2.dilate(edge_combined, kernel, iterations=1)
    return edge_combined


def build_segmentation_frames(frame_paths: list[Path], output_dir: Path, config: HanBaselineConfig) -> list[Path]:
    device = ensure_cuda_if_requested(config.device)
    segmentation_module = build_mit_scene_module(device)
    predictor = build_detectron_predictor(device, config.detectron_score_threshold)
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise FileNotFoundError(frame_paths[0])
    h, w = first.shape[:2]
    edge_rep = np.zeros((h, w, config.hough_window), dtype=np.uint8)
    kernel = np.ones((10, 10), np.uint8)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    output_paths: list[Path] = []
    for count, frame_path in enumerate(frame_paths, start=1):
        pil_image = Image.open(frame_path).convert("RGB")
        img_data = normalize(pil_image)
        singleton_batch = {"img_data": img_data[None].to(device)}
        with torch.no_grad():
            scores = segmentation_module(singleton_batch, segSize=img_data.shape[1:])
        _, pred = torch.max(scores, dim=1)
        pred = pred.cpu()[0].numpy()

        pred_clean = pred.copy()
        pred_clean[~np.isin(pred_clean, config.structure_class_ids)] = 0
        pred_clean2 = morphology.remove_small_objects(pred_clean.astype(bool), min_size=config.structure_min_region_area).astype(int) * 255
        pred_clean3 = np.minimum(pred_clean, pred_clean2)
        image_edge = Image.fromarray(np.uint8(pred_clean3 * 255), "L").filter(ImageFilter.FIND_EDGES)
        image_edge = cv2.dilate(np.array(image_edge), kernel, iterations=1)
        edges = get_houghlines(image_edge, config)

        if count <= config.hough_window:
            edge_rep[:, :, count - 1] = edges
        else:
            hist_curr = np.concatenate([edge_rep, np.expand_dims(edges, 2)], axis=2)
            hist_curr = np.max(hist_curr, axis=2)
            hist_curr = cv2.erode(get_houghlines(hist_curr, config), np.ones((10, 10), dtype=np.uint8))
            _, hist_curr = cv2.threshold(hist_curr, 0, 255, cv2.THRESH_BINARY)
            hist_curr = cv2.morphologyEx(hist_curr, cv2.MORPH_OPEN, kernel2, iterations=3)
            edges = cv2.erode(get_houghlines(hist_curr, config), np.ones((10, 10), dtype=np.uint8))

        image_bgr = cv2.imread(str(frame_path))
        if image_bgr is None:
            raise FileNotFoundError(frame_path)
        instances = predictor(image_bgr)["instances"]
        masks = np.asarray(instances.pred_masks.cpu().numpy()) if instances.has("pred_masks") else None
        classes = instances.pred_classes.cpu().numpy() if instances.has("pred_classes") else np.array([])
        selected = [index for index, class_id in enumerate(classes) if int(class_id) in config.important_coco_classes]
        if masks is None or not selected:
            masks_comb = edges
        else:
            masks_comb = np.max(masks[selected, :, :], axis=0).astype(np.uint8) * 255

        out = output_dir / f"{frame_path.stem}.png"
        cv2.imwrite(str(out), masks_comb.astype(np.uint8))
        output_paths.append(out)
    return output_paths


def read_gray_frame(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def combine_frames(saliency_paths: list[Path], segmentation_paths: list[Path], depth_paths: list[Path], output_dir: Path, config: HanBaselineConfig) -> list[Path]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for index, (sal_path, seg_path, dep_path) in enumerate(zip(saliency_paths, segmentation_paths, depth_paths)):
        sal = read_gray_frame(sal_path)
        seg = read_gray_frame(seg_path)
        dep = read_gray_frame(dep_path)
        threshold = float(np.max(sal)) * config.saliency_threshold_fraction
        sal_fil = sal.copy()
        sal_fil[sal_fil <= threshold] = 0
        sal_fil[sal_fil > 0] = 255
        sal_norm = cv2.normalize(sal_fil, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        seg_norm = cv2.normalize(seg, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        dep_norm = cv2.normalize(dep, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        seg_norm = cv2.resize(seg_norm, (sal_norm.shape[1], sal_norm.shape[0]))
        dep_norm = cv2.resize(dep_norm, (sal_norm.shape[1], sal_norm.shape[0]))
        seg_sal = np.max((sal_norm, seg_norm), axis=0)
        dep_seg_sal = dep_norm.copy()
        dep_seg_sal[seg_sal == 0] = 0
        result = (dep_seg_sal * 255).astype(np.uint8)
        out = output_dir / f"frame{index:05d}.png"
        cv2.imwrite(str(out), result)
        output_paths.append(out)
    return output_paths


def write_video(frame_paths: list[Path], output_path: Path, fps: float, is_color: bool = True) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first = _load_video_frame(frame_paths[0], is_color=is_color)
    height, width = first.shape[:2]
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        process.stdin.write(first.tobytes())
        for path in frame_paths[1:]:
            frame = _load_video_frame(path, is_color=is_color)
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed while writing {output_path}")


def _load_video_frame(path: Path, *, is_color: bool) -> np.ndarray:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR if is_color else cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise FileNotFoundError(path)
    if frame.ndim == 2:
        return np.stack([frame, frame, frame], axis=-1)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def run_han_baseline_on_clip(clip_path: Path, output_root: Path, config: HanBaselineConfig) -> dict[str, str | int]:
    clip_root = output_root / clip_path.stem
    frames = extract_video_frames(clip_path, clip_root / "frames", config.target_fps, config.max_frames)
    saliency = build_saliency_frames(frames, clip_root / "saliency_frames", config)
    segmentation = build_segmentation_frames(frames, clip_root / "segmentation_frames", config)
    depth = build_depth_frames(clip_root / "frames", clip_root / "depth", config)
    depth_by_name = {path.stem: path for path in depth}
    ordered_depth = [depth_by_name[frame.stem] for frame in frames]
    combination = combine_frames(saliency, segmentation, ordered_depth, clip_root / "combination_frames", config)
    videos_dir = clip_root / "videos"
    write_video(depth, videos_dir / "depth.mp4", config.target_fps, is_color=True)
    write_video(saliency, videos_dir / "saliency.mp4", config.target_fps, is_color=False)
    write_video(segmentation, videos_dir / "segmentation.mp4", config.target_fps, is_color=False)
    write_video(combination, videos_dir / "combination.mp4", config.target_fps, is_color=False)
    return {
        "clip": str(clip_path),
        "frames": len(frames),
        "depth_frames": len(depth),
        "saliency_frames": len(saliency),
        "segmentation_frames": len(segmentation),
        "combination_frames": len(combination),
        "output": str(clip_root),
    }


def run_han_baseline(clip_dir: Path = DEFAULT_CLIP_DIR, output_root: Path = DEFAULT_OUTPUT_ROOT, config: HanBaselineConfig = HanBaselineConfig()) -> list[dict[str, str | int]]:
    ensure_cuda_if_requested(config.device)
    clips = sorted(clip_dir.glob("*.mp4"))
    if not clips:
        raise FileNotFoundError(f"No MP4 clips found in {clip_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for clip in clips:
        print(f"processing {clip.name}", flush=True)
        summaries.append(run_han_baseline_on_clip(clip, output_root, config))
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Han et al. 2021 baseline on EPIC-KITCHENS clips.")
    parser.add_argument("--clip-dir", type=Path, default=DEFAULT_CLIP_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-fps", type=float, default=20.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    config = HanBaselineConfig(target_fps=args.target_fps, max_frames=args.max_frames, device=args.device)
    summaries = run_han_baseline(args.clip_dir.resolve(), args.output_root.resolve(), config)
    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()
