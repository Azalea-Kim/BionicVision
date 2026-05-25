"""Detectron2 Mask R-CNN adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from models.base import ModelSpec


DEFAULT_IMPORTANT_CLASSES = (0, 1, 2, 5, 7)
DETECTRON2_SPEC = ModelSpec(
    name="detectron2_mask_rcnn",
    task="segmentation",
    required_packages=("detectron2", "opencv-python"),
)


def build_predictor(score_threshold: float = 0.5):
    """Create a Detectron2 predictor lazily."""

    import detectron2  # noqa: F401
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.utils.logger import setup_logger

    setup_logger()
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    return DefaultPredictor(cfg)


def predict_important_mask(
    image_path: str | Path,
    *,
    important_classes: tuple[int, ...] = DEFAULT_IMPORTANT_CLASSES,
    score_threshold: float = 0.5,
) -> np.ndarray:
    """Predict a combined mask for selected COCO class IDs."""

    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    predictor = build_predictor(score_threshold=score_threshold)
    instances = predictor(image)["instances"]
    if not instances.has("pred_classes") or not instances.has("pred_masks"):
        return np.zeros(image.shape[:2], dtype=np.uint8)

    classes = instances.pred_classes.cpu().numpy()
    masks = np.asarray(instances.pred_masks.cpu().numpy())
    selected = [index for index, class_id in enumerate(classes) if int(class_id) in important_classes]
    if not selected:
        return np.zeros(image.shape[:2], dtype=np.uint8)

    return np.max(masks[selected, :, :], axis=0).astype(np.uint8) * 255


def save_mask(mask: np.ndarray, output_path: str | Path) -> Path:
    import cv2

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), mask)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Detectron2 baseline segmentation on one image.")
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--output-path", default="detectron_mask/frame_seg.png")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    args = parser.parse_args()

    mask = predict_important_mask(args.image_path, score_threshold=args.score_threshold)
    save_mask(mask, args.output_path)


if __name__ == "__main__":
    main()
