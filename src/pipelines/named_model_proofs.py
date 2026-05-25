"""Generate visual proof outputs for the named project models.

This runner intentionally uses the actual model packages/sources checked out
under external/model_sources and installed in .venv-models. It does not call
legacy code or substitute another model family when a requested model fails.
"""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
from torchvision.transforms import Compose


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "external" / "model_sources"
DEFAULT_FRAMES = ROOT / "outputs" / "model_proofs" / "frames"
DEFAULT_OUTPUT = ROOT / "outputs" / "named_model_proofs"
PYTHON = ROOT / ".venv-models" / "bin" / "python"


def selected_frames(frame_dir: Path, max_frames: int) -> list[Path]:
    frames = sorted(frame_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No JPG frames found in {frame_dir}")
    return frames[:max_frames]


def prepare_inputs(frames: list[Path], output_root: Path) -> Path:
    input_dir = output_root / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames, start=1):
        shutil.copy2(frame, input_dir / f"{index:05d}_{frame.name}")
    return input_dir


def save_heatmap_overlay(image_rgb: np.ndarray, score: np.ndarray, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    norm = normalize01(score)
    heat = (cm.inferno(norm)[:, :, :3] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(image_rgb, 0.55, heat, 0.45, 0)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_rgb)
    axes[0].set_title("input")
    axes[1].imshow(norm, cmap="inferno")
    axes[1].set_title(title)
    axes[2].imshow(overlay)
    axes[2].set_title("overlay")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_rgb(image_rgb: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb).save(output_path)


def normalize01(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.float32)
    lo = float(values[finite].min())
    hi = float(values[finite].max())
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def get_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return torch.device(name)


def run_deepgaze(frames: list[Path], output_root: Path, device: torch.device) -> None:
    import deepgaze_pytorch

    centerbias_path = ROOT / "src" / "models" / "saliency" / "deepgaze3" / "centerbias_mit1003.npy"
    centerbias_template = np.load(centerbias_path)

    models = {
        "deepgaze_ii": deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(device).eval(),
        "deepgaze_iii": deepgaze_pytorch.DeepGazeIII(pretrained=True).to(device).eval(),
    }

    for frame in frames:
        image = np.asarray(Image.open(frame).convert("RGB"))
        centerbias = zoom(
            centerbias_template,
            (image.shape[0] / centerbias_template.shape[0], image.shape[1] / centerbias_template.shape[1]),
            order=0,
            mode="nearest",
        )
        centerbias -= logsumexp(centerbias)

        image_tensor = torch.tensor([image.transpose(2, 0, 1)], dtype=torch.float32, device=device)
        centerbias_tensor = torch.tensor([centerbias], dtype=torch.float32, device=device)

        with torch.no_grad():
            dg2 = models["deepgaze_ii"](image_tensor, centerbias_tensor).detach().cpu().numpy()[0, 0]
        save_heatmap_overlay(
            image,
            dg2,
            output_root / "saliency" / "deepgaze_ii" / f"{frame.stem}.png",
            "DeepGaze II",
        )

        h, w = image.shape[:2]
        fixation_history_x = np.array([w // 2, w // 3, (2 * w) // 3, w // 2, w // 4, (3 * w) // 4])
        fixation_history_y = np.array([h // 2, h // 3, h // 3, (2 * h) // 3, h // 2, h // 2])
        model = models["deepgaze_iii"]
        x_hist = torch.tensor([fixation_history_x[model.included_fixations]], dtype=torch.float32, device=device)
        y_hist = torch.tensor([fixation_history_y[model.included_fixations]], dtype=torch.float32, device=device)
        with torch.no_grad():
            dg3 = model(image_tensor, centerbias_tensor, x_hist, y_hist).detach().cpu().numpy()[0, 0]
        save_heatmap_overlay(
            image,
            dg3,
            output_root / "saliency" / "deepgaze_iii" / f"{frame.stem}.png",
            "DeepGaze III",
        )


def run_detectron2(frames: list[Path], output_root: Path, device: torch.device) -> None:
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    from detectron2.engine import DefaultPredictor
    from detectron2.utils.visualizer import Visualizer

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = device.type
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    for frame in frames:
        image_bgr = cv2.imread(str(frame))
        if image_bgr is None:
            raise FileNotFoundError(frame)
        outputs = predictor(image_bgr)
        visualizer = Visualizer(image_bgr[:, :, ::-1], metadata=metadata, scale=1.0)
        rendered = visualizer.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
        save_rgb(rendered, output_root / "segmentation" / "detectron2" / f"{frame.stem}.png")


def run_mit_scene_parsing(frames: list[Path], output_root: Path, device: torch.device) -> None:
    from mit_semseg.models import ModelBuilder, SegmentationModule
    from mit_semseg.utils import colorEncode
    import torchvision.transforms as transforms

    source = EXTERNAL / "segmentation" / "semantic-segmentation-pytorch"
    weights = ROOT / "data" / "model_weights" / "mit_scene_parsing" / "ade20k-resnet50dilated-ppm_deepsup"
    colors = loadmat(source / "data" / "color150.mat")["colors"]

    encoder = ModelBuilder.build_encoder(
        arch="resnet50dilated",
        fc_dim=2048,
        weights=str(weights / "encoder_epoch_20.pth"),
    )
    decoder = ModelBuilder.build_decoder(
        arch="ppm_deepsup",
        fc_dim=2048,
        num_class=150,
        weights=str(weights / "decoder_epoch_20.pth"),
        use_softmax=True,
    )
    segmentation_module = SegmentationModule(encoder, decoder, torch.nn.NLLLoss(ignore_index=-1)).to(device).eval()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    to_tensor = transforms.ToTensor()

    for frame in frames:
        image = Image.open(frame).convert("RGB")
        image_np = np.asarray(image)
        tensor = normalize(to_tensor(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            scores = segmentation_module({"img_data": tensor}, segSize=image_np.shape[:2])
            _, prediction = torch.max(scores, dim=1)
        prediction_np = prediction.squeeze(0).cpu().numpy()
        color = colorEncode(prediction_np, colors).astype(np.uint8)
        overlay = cv2.addWeighted(image_np, 0.55, color, 0.45, 0)
        save_rgb(overlay, output_root / "segmentation" / "mit_scene_parsing" / f"{frame.stem}.png")


def run_monodepth2(input_dir: Path, output_root: Path, device: torch.device) -> None:
    monodepth_root = EXTERNAL / "depth" / "monodepth2"
    command = [
        str(PYTHON),
        "test_simple.py",
        "--image_path",
        str(input_dir),
        "--model_name",
        "mono_640x192",
        "--ext",
        "jpg",
    ]
    if device.type == "cpu":
        command.append("--no_cuda")
    subprocess.run(command, cwd=monodepth_root, check=True)
    dest = output_root / "depth" / "monodepth2"
    dest.mkdir(parents=True, exist_ok=True)
    for image in sorted(input_dir.glob("*_disp.jpeg")):
        shutil.copy2(image, dest / image.name)


def run_tc_monodepth(frames: list[Path], output_root: Path, device: torch.device) -> None:
    tc_root = EXTERNAL / "depth" / "TCMonoDepth"
    sys.path.insert(0, str(tc_root))
    from networks import TCSmallNet
    from networks.transforms import PrepareForNet, Resize

    args = SimpleNamespace()
    model = TCSmallNet(args)
    checkpoint = torch.load(tc_root / "weights" / "_ckpt_small.pt.tar", map_location="cpu")
    model.load_state_dict(checkpoint)
    model.to(device).eval()

    transform = Compose(
        [
            Resize(256, 256, resize_target=None, keep_aspect_ratio=True, ensure_multiple_of=32, resize_method="lower_bound"),
            PrepareForNet(),
        ]
    )
    dest = output_root / "depth" / "tc_monodepth"
    dest.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for frame in frames:
            image_bgr = cv2.imread(str(frame))
            if image_bgr is None:
                raise FileNotFoundError(frame)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            model_input = transform({"image": image_rgb})["image"]
            tensor = torch.from_numpy(model_input).to(device).unsqueeze(0)
            prediction = model(tensor)
            prediction = torch.nn.functional.interpolate(
                prediction,
                size=image_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()
            depth_vis = (normalize01(prediction) * 255).astype(np.uint8)
            color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
            cv2.imwrite(str(dest / f"{frame.stem}.png"), color)


@contextlib.contextmanager
def cpu_cuda_shim():
    original_tensor_cuda = torch.Tensor.cuda
    original_module_cuda = torch.nn.Module.cuda
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self
    try:
        yield
    finally:
        torch.Tensor.cuda = original_tensor_cuda
        torch.nn.Module.cuda = original_module_cuda


def run_deva(input_dir: Path, output_root: Path) -> None:
    deva_root = EXTERNAL / "segmentation" / "Tracking-Anything-with-DEVA"
    sys.path.insert(0, str(deva_root))

    from torch.utils.data import DataLoader
    from deva.inference.data.simple_video_reader import SimpleVideoReader, no_collate
    from deva.inference.demo_utils import flush_buffer
    from deva.inference.eval_args import add_common_eval_args, get_model_and_config
    from deva.inference.inference_core import DEVAInferenceCore
    from deva.inference.result_utils import ResultSaver
    from deva.ext.automatic_processor import process_frame_automatic
    from deva.ext.automatic_sam import get_sam_model
    from deva.ext.ext_eval_args import add_auto_default_args, add_ext_eval_args

    deva_output = output_root / "segmentation" / "deva"
    deva_output.mkdir(parents=True, exist_ok=True)
    argv = [
        "deva-proof",
        "--model",
        str(deva_root / "saves" / "DEVA-propagation.pth"),
        "--img_path",
        str(input_dir),
        "--output",
        str(deva_output),
        "--size",
        "360",
        "--temporal_setting",
        "online",
        "--detection_every",
        "1",
        "--SAM_NUM_POINTS_PER_SIDE",
        "16",
        "--SAM_NUM_POINTS_PER_BATCH",
        "16",
        "--SAM_CHECKPOINT_PATH",
        str(deva_root / "saves" / "sam_vit_h_4b8939.pth"),
    ]

    parser = argparse.ArgumentParser()
    add_common_eval_args(parser)
    add_ext_eval_args(parser)
    add_auto_default_args(parser)

    old_argv = sys.argv
    sys.argv = argv
    use_cuda = torch.cuda.is_available()
    device_context = contextlib.nullcontext() if use_cuda else cpu_cuda_shim()
    sam_device = "cuda" if use_cuda else "cpu"

    try:
        with device_context:
            deva_model, cfg, args = get_model_and_config(parser)
            sam_model = get_sam_model(cfg, sam_device)
            cfg["temporal_setting"] = args.temporal_setting.lower()
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
            deva = DEVAInferenceCore(deva_model, config=cfg)
            deva.next_voting_frame = args.num_voting_frames - 1
            deva.enabled_long_id()
            result_saver = ResultSaver(str(deva_output), None, dataset="demo", object_manager=deva.object_manager)
            for ti, (frame, im_path) in enumerate(loader):
                process_frame_automatic(deva, sam_model, im_path, result_saver, ti, image_np=frame)
            flush_buffer(deva, result_saver)
            result_saver.end()
    finally:
        sys.argv = old_argv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run proof images for the named project models.")
    parser.add_argument("--frame-dir", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-frames", type=int, default=3)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["deepgaze", "detectron2", "mit", "monodepth2", "tcmonodepth", "deva"],
        choices=["deepgaze", "detectron2", "mit", "monodepth2", "tcmonodepth", "deva"],
    )
    args = parser.parse_args()

    frame_dir = args.frame_dir.resolve()
    output_root = args.output_root.resolve()
    frames = selected_frames(frame_dir, args.max_frames)
    device = get_device(args.device)
    output_root.mkdir(parents=True, exist_ok=True)
    input_dir = prepare_inputs(frames, output_root)

    if "deepgaze" in args.models:
        run_deepgaze(frames, output_root, device)
    if "detectron2" in args.models:
        run_detectron2(frames, output_root, device)
    if "mit" in args.models:
        run_mit_scene_parsing(frames, output_root, device)
    if "monodepth2" in args.models:
        run_monodepth2(input_dir, output_root, device)
    if "tcmonodepth" in args.models:
        run_tc_monodepth(frames, output_root, device)
    if "deva" in args.models:
        run_deva(input_dir, output_root)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    main()
