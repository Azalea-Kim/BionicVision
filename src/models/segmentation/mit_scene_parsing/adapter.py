"""ADE20K/MIT scene-parsing adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from models.base import ModelSpec


MIT_SCENE_PARSING_SPEC = ModelSpec(
    name="mit_scene_parsing",
    task="segmentation",
    required_packages=("torch", "torchvision", "mit_semseg"),
)


@dataclass(frozen=True)
class SceneParsingConfig:
    """Classes used by the baseline to render indoor structure."""

    structure_class_ids: tuple[int, ...] = (0, 14)
    min_region_area: int = 16000
    min_line_length: int = 30
    max_line_gap: int = 1


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_WEIGHTS = ROOT / "data" / "model_weights" / "mit_scene_parsing" / "ade20k-resnet50dilated-ppm_deepsup"


def build_scene_module(*, device: str, weights_dir: Path = DEFAULT_WEIGHTS):
    """Build the MIT semantic segmentation module lazily."""

    import torch
    from mit_semseg.models import ModelBuilder, SegmentationModule

    encoder = ModelBuilder.build_encoder(
        arch="resnet50dilated",
        fc_dim=2048,
        weights=str(weights_dir / "encoder_epoch_20.pth"),
    )
    decoder = ModelBuilder.build_decoder(
        arch="ppm_deepsup",
        fc_dim=2048,
        num_class=150,
        weights=str(weights_dir / "decoder_epoch_20.pth"),
        use_softmax=True,
    )
    return SegmentationModule(encoder, decoder, torch.nn.NLLLoss(ignore_index=-1)).to(device).eval()


def predict_labels(pil_image, segmentation_module, *, device: str):
    """Predict ADE20K class labels for one RGB PIL image."""

    import torch
    import torchvision.transforms as transforms

    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_data = normalize(pil_image)
    singleton_batch = {"img_data": img_data[None].to(device)}
    with torch.no_grad():
        scores = segmentation_module(singleton_batch, segSize=img_data.shape[1:])
    _, pred = torch.max(scores, dim=1)
    return pred.cpu()[0].numpy()
