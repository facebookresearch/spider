"""Compatibility wrapper exposing example modules under the `spider.examples` namespace."""
from importlib import import_module
from types import ModuleType
from typing import Any


def _resolve(module: str) -> ModuleType:
    """Import a module from the top-level `examples` package."""
    return import_module(f"examples.{module}")


def load(name: str) -> Any:
    """Return the requested attribute from the proxied examples module."""
    module_name, _, attr = name.partition(":")
    module = _resolve(module_name)
    return getattr(module, attr) if attr else module

__all__ = ["load"]
