"""Frame discovery and image I/O utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Sequence

import cv2
import imageio.v2 as imageio
import numpy as np

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class FrameSequence:
    """Ordered frame paths with lightweight metadata."""

    paths: tuple[Path, ...]
    fps: float | None = None

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self):
        return iter(self.paths)


def natural_sort_key(path: Path | str) -> list[int | str]:
    """Sort paths by embedded numbers, e.g. frame_2 before frame_10."""

    text = str(path)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def list_frames(
    folder: str | Path,
    *,
    extensions: Iterable[str] = IMAGE_EXTENSIONS,
    recursive: bool = False,
) -> list[Path]:
    """Return image frame paths in stable natural order."""

    root = Path(folder)
    allowed = {ext.lower() for ext in extensions}
    pattern = "**/*" if recursive else "*"
    return sorted(
        (p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in allowed),
        key=natural_sort_key,
    )


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an image as RGB uint8."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_gray(path: str | Path) -> np.ndarray:
    """Load an image as grayscale uint8."""

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def save_gray(path: str | Path, image: np.ndarray) -> Path:
    """Save a grayscale-compatible array as uint8."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output_path, normalize_to_uint8(image))
    return output_path


def normalize_to_uint8(values: np.ndarray) -> np.ndarray:
    """Normalize any numeric array into the 0-255 uint8 display range."""

    array = np.asarray(values)
    if array.dtype == np.uint8:
        return array
    array = array.astype(np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros(array.shape, dtype=np.uint8)
    min_value = float(array[finite].min())
    max_value = float(array[finite].max())
    if max_value <= min_value:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = (array - min_value) / (max_value - min_value)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def load_npy_sequence(paths: Sequence[str | Path]) -> list[np.ndarray]:
    """Load a sequence of NumPy arrays."""

    return [np.load(Path(path)) for path in paths]

