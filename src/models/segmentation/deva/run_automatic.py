"""Run DEVA automatic segmentation on frame folders."""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import shutil
import sys

import torch


ROOT = Path(__file__).resolve().parents[4]
DEVA_ROOT = ROOT / "external" / "model_sources" / "segmentation" / "Tracking-Anything-with-DEVA"


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


def run_deva_automatic(
    *,
    frames_dir: Path,
    output_dir: Path,
    size: int = 360,
    detection_every: int = 5,
    points_per_side: int = 16,
    points_per_batch: int = 16,
) -> Path:
    """Run automatic SAM + DEVA propagation on a frame directory."""

    sys.path.insert(0, str(DEVA_ROOT))
    from torch.utils.data import DataLoader
    from deva.inference.data.simple_video_reader import SimpleVideoReader, no_collate
    from deva.inference.demo_utils import flush_buffer
    from deva.inference.eval_args import add_common_eval_args, get_model_and_config
    from deva.inference.inference_core import DEVAInferenceCore
    from deva.inference.result_utils import ResultSaver
    from deva.ext.automatic_processor import process_frame_automatic
    from deva.ext.automatic_sam import get_sam_model
    from deva.ext.ext_eval_args import add_auto_default_args, add_ext_eval_args

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        "deva-automatic",
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
        "--SAM_NUM_POINTS_PER_SIDE",
        str(points_per_side),
        "--SAM_NUM_POINTS_PER_BATCH",
        str(points_per_batch),
        "--SAM_CHECKPOINT_PATH",
        str(DEVA_ROOT / "saves" / "sam_vit_h_4b8939.pth"),
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
            result_saver = ResultSaver(str(output_dir), None, dataset="demo", object_manager=deva.object_manager)
            for ti, (frame, im_path) in enumerate(loader):
                process_frame_automatic(deva, sam_model, im_path, result_saver, ti, image_np=frame)
            flush_buffer(deva, result_saver)
            result_saver.end()
    finally:
        sys.argv = old_argv

    return output_dir / "Annotations"
