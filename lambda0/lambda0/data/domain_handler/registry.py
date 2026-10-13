"""Explicit loading of dataset adapters declared by metadata."""

from __future__ import annotations

import importlib
import inspect

from .base import DomainHandler


def get_handler_cls(reference: str) -> type[DomainHandler]:
    """Resolve a `package.module:ClassName` adapter reference."""
    if not isinstance(reference, str) or reference.count(":") != 1:
        raise ValueError(
            "Dataset handler must use the form 'package.module:ClassName'."
        )
    module_name, class_name = reference.split(":", maxsplit=1)
    if not module_name or not class_name:
        raise ValueError(
            "Dataset handler must use the form 'package.module:ClassName'."
        )
    module = importlib.import_module(module_name)
    handler_cls = getattr(module, class_name, None)
    if not isinstance(handler_cls, type) or not issubclass(
        handler_cls, DomainHandler
    ) or inspect.isabstract(handler_cls):
        raise TypeError(
            f"{reference!r} does not resolve to a concrete "
            "DomainHandler subclass."
        )
    return handler_cls


__all__ = ["get_handler_cls"]
