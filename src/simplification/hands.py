"""Hand/arm-derived interaction-region helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .masks import as_binary


@dataclass(frozen=True)
class HandCircle:
    center_x: float
    center_y: float
    radius: int
    mask: np.ndarray


def pca_angle(xs: np.ndarray, ys: np.ndarray) -> float:
    """Estimate arm orientation angle relative to the image y-axis."""

    points = np.column_stack((xs, ys)).astype(np.float32)
    centered = points - np.mean(points, axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eig(covariance)
    axis = eigen_vectors[:, int(np.argmax(eigen_values))]
    return float(np.degrees(np.arctan2(axis[0], axis[1])))


def exponential_circle_radius(
    shape: tuple[int, int],
    hand_x: float,
    hand_y: float,
    *,
    r_min: int = 100,
    r_max: int = 400,
    alpha: float = 2.0,
    squared_ratio: bool = True,
) -> int:
    """Radius is larger near image center and smaller near image borders."""

    height, width = shape[:2]
    center_x, center_y = width / 2.0, height / 2.0
    distance = float(np.hypot(hand_x - center_x, hand_y - center_y))
    max_distance = float(np.hypot(width / 2.0, height / 2.0))
    if max_distance == 0:
        return r_min
    normalized = distance / max_distance
    exponent = normalized**2 if squared_ratio else normalized
    ratio = float(np.exp(-alpha * exponent))
    return int(r_min + (r_max - r_min) * np.clip(ratio, 0, 1))


def infer_hand_circle(arm_mask: np.ndarray) -> HandCircle | None:
    """Infer a likely hand/action region from an arm mask."""

    binary = as_binary(arm_mask)
    ys, xs = np.where(binary == 255)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min = float(np.min(ys))
    angle = pca_angle(xs, ys)
    if angle < 0:
        hand_x, hand_y = x_min, y_min
    elif angle > 0:
        hand_x, hand_y = x_max, y_min
    else:
        hand_x, hand_y = (x_min + x_max) / 2.0, y_min

    radius = exponential_circle_radius(binary.shape, hand_x, hand_y)
    circle = np.zeros_like(binary)
    cv2.circle(circle, (int(hand_x), int(hand_y)), radius, 255, thickness=-1)
    return HandCircle(center_x=hand_x, center_y=hand_y, radius=radius, mask=circle)


def intersection_percentage(object_mask: np.ndarray, region_mask: np.ndarray) -> float:
    """Return percent of object pixels inside region."""

    object_active = np.asarray(object_mask) > 0
    region_active = np.asarray(region_mask) > 0
    object_count = int(np.count_nonzero(object_active))
    if object_count == 0:
        return 0.0
    intersection = int(np.count_nonzero(object_active & region_active))
    return 100.0 * intersection / object_count

