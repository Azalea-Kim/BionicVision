"""Lightweight TCMonoDepth adapter metadata."""

from __future__ import annotations

from models.base import ModelSpec


TC_MONODEPTH_SPEC = ModelSpec(
    name="tc_monodepth",
    task="depth",
    required_packages=("torch", "torchvision"),
)
