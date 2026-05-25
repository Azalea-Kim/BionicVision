"""Lightweight Monodepth2 adapter metadata."""

from __future__ import annotations

from models.base import ModelSpec


MONODEPTH2_SPEC = ModelSpec(
    name="monodepth2",
    task="depth",
    required_packages=("torch", "torchvision"),
)
