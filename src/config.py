"""Load config.yaml into a simple attribute-accessible object."""
from __future__ import annotations

import os
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


class Cfg(dict):
    """dict with attribute access, recursively."""

    def __getattr__(self, name: str) -> Any:
        try:
            v = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return Cfg(v) if isinstance(v, dict) else v


def load_config(path: str | None = None) -> Cfg:
    with open(path or DEFAULT_CONFIG_PATH) as f:
        return Cfg(yaml.safe_load(f))
