"""Public Lambda0 API with lazy loading of optional Qwen3.5 dependencies."""

from __future__ import annotations

from typing import Any


__all__ = [
    "ACTION_CHUNK_SIZE",
    "LAMBDA0_VARIANTS",
    "Lambda0",
    "Lambda0Config",
    "Lambda0Processor",
]


def __getattr__(name: str) -> Any:
    if name in {"ACTION_CHUNK_SIZE", "LAMBDA0_VARIANTS", "Lambda0Config"}:
        from .configuration_lambda0 import (
            ACTION_CHUNK_SIZE,
            LAMBDA0_VARIANTS,
            Lambda0Config,
        )

        return {
            "ACTION_CHUNK_SIZE": ACTION_CHUNK_SIZE,
            "LAMBDA0_VARIANTS": LAMBDA0_VARIANTS,
            "Lambda0Config": Lambda0Config,
        }[name]
    if name == "Lambda0":
        from .modeling_lambda0 import Lambda0

        return Lambda0
    if name == "Lambda0Processor":
        from .processing_lambda0 import Lambda0Processor

        return Lambda0Processor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
