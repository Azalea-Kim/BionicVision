"""Shared model adapter primitives."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Description of an optional model backend."""

    name: str
    task: str
    required_packages: tuple[str, ...] = ()


class ModelUnavailableError(RuntimeError):
    """Raised when an optional model backend is not installed."""
