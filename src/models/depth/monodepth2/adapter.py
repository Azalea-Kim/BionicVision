"""Monodepth2 adapter backed by the checkout under `external/`."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from models.base import ModelSpec


ROOT = Path(__file__).resolve().parents[4]
MONODEPTH2_ROOT = ROOT / "external" / "model_sources" / "depth" / "monodepth2"
PYTHON = ROOT / ".venv-models" / "bin" / "python"

MONODEPTH2_SPEC = ModelSpec(
    name="monodepth2",
    task="depth",
    required_packages=("torch", "torchvision"),
)


def predict_depth_folder(
    *,
    frame_dir: Path,
    output_dir: Path,
    model_name: str,
    device: str = "cuda",
) -> list[Path]:
    """Run Monodepth2 on a folder of JPG frames and return depth `.npy` paths."""

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
        model_name,
        "--ext",
        "jpg",
        "--pred_metric_depth",
    ]
    if device == "cpu":
        command.append("--no_cuda")
    subprocess.run(command, cwd=MONODEPTH2_ROOT, check=True)
    return sorted(work_dir.glob("*_depth.npy"))
