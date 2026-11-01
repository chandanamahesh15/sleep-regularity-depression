"""Configuration loading with typed coercion and dotted CLI overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def _coerce(value: Any) -> Any:
    """Turn YAML/CLI strings into numbers/bools/lists where unambiguous.

    PyYAML follows YAML 1.1, which parses ``1e-5`` (no dot) as a *string*.
    Coercing here means a silently-stringified learning rate can't reach the optimiser.
    """
    if isinstance(value, dict):
        return {k: _coerce(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce(v) for v in value]
    if not isinstance(value, str):
        return value
    low = value.strip().lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [_coerce(v.strip()) for v in inner.split(",")] if inner else []
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply ``a.b.c=value`` overrides to a nested dict (returns a copy)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.path=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(raw)
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as fh:
        cfg = _coerce(yaml.safe_load(fh))
    return apply_overrides(cfg, overrides)
