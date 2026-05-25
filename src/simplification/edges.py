"""Edge extraction used by scene and large-object simplification."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .masks import as_binary, connected_component_filter


@dataclass(frozen=True)
class EdgeConfig:
    dilate_kernel: int = 10
    min_component_area: int = 1500
    hough_threshold: int = 15
    min_line_length: int = 30
    max_line_gap: int = 1
    border_margin: int = 10
    output_thickness: int = 10


def mask_to_edges(mask: np.ndarray, *, dilate_kernel: int = 10) -> np.ndarray:
    """Convert a binary/instance mask into a thick boundary map."""

    binary = as_binary(mask)
    kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
    return as_binary(gradient)


def extract_hough_edges(mask: np.ndarray, config: EdgeConfig = EdgeConfig()) -> np.ndarray:
    """Keep prominent straight edges after cleanup and Hough filtering."""

    edges = mask_to_edges(mask, dilate_kernel=config.dilate_kernel)
    edges = connected_component_filter(edges, min_area=config.min_component_area)

    lines = cv2.HoughLinesP(
        edges.astype(np.uint8),
        rho=1,
        theta=np.pi / 180,
        threshold=config.hough_threshold,
        minLineLength=config.min_line_length,
        maxLineGap=config.max_line_gap,
    )
    output = np.zeros_like(edges)
    if lines is None:
        return output

    height, width = output.shape[:2]
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = float(np.hypot(x2 - x1, y2 - y1))
        too_close_to_border = (
            x1 < config.border_margin
            or x2 < config.border_margin
            or y1 < config.border_margin
            or y2 < config.border_margin
            or x1 > width - config.border_margin
            or x2 > width - config.border_margin
            or y1 > height - config.border_margin
            or y2 > height - config.border_margin
        )
        if length >= config.min_line_length and not too_close_to_border:
            cv2.line(output, (x1, y1), (x2, y2), color=255, thickness=config.output_thickness)

    return connected_component_filter(output, min_area=config.min_component_area)


def large_mask_to_edges(mask: np.ndarray, *, brightness: int, kernel_size: int = 20) -> np.ndarray:
    """Render a large object as bright edges instead of a filled region."""

    edge_mask = mask_to_edges(mask, dilate_kernel=kernel_size).astype(np.float32)
    return np.clip(edge_mask * (float(brightness) / 255.0), 0, 255).astype(np.uint8)

