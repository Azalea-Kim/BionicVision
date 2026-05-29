#!/usr/bin/env python3
"""Create a contact sheet for quick visual inspection of frame variants."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datasets.frames import list_frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--cell-width", type=int, default=240)
    args = parser.parse_args()

    rows = [sample_row(frame_dir, args.samples, args.cell_width) for frame_dir in args.frame_dirs]
    sheet = np.concatenate(rows, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), sheet)
    print(args.output)


def sample_row(frame_dir: Path, samples: int, cell_width: int) -> np.ndarray:
    frames = list_frames(frame_dir)
    if not frames:
        raise FileNotFoundError(f"No image frames found in {frame_dir}")
    indices = np.linspace(0, len(frames) - 1, min(samples, len(frames)), dtype=int)
    cells = []
    for index in indices:
        image = cv2.imread(str(frames[int(index)]), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(frames[int(index)])
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        height, width = image.shape[:2]
        cell_height = max(1, int(round(height * (cell_width / width))))
        cells.append(cv2.resize(image, (cell_width, cell_height), interpolation=cv2.INTER_AREA))
    target_height = max(cell.shape[0] for cell in cells)
    padded = []
    for cell in cells:
        if cell.shape[0] < target_height:
            pad = np.zeros((target_height - cell.shape[0], cell.shape[1], 3), dtype=np.uint8)
            cell = np.concatenate([cell, pad], axis=0)
        padded.append(cell)
    return np.concatenate(padded, axis=1)


if __name__ == "__main__":
    main()
