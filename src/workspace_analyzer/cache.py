"""Content-addressed, safe on-disk cache for workspace analysis results."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from .analyzer import AnalysisResult


class ResultCache:
    """Store versioned ``AnalysisResult`` objects by deterministic SHA-256 key."""

    def __init__(self, directory: str | Path, *, compressed: bool = True):
        if not isinstance(compressed, bool):
            raise ValueError("compressed must be Boolean")
        self.compressed = compressed
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        if (
            not isinstance(key, str)
            or len(key) != 64
            or any(char not in "0123456789abcdef" for char in key)
        ):
            raise ValueError("cache key must be a lowercase SHA-256 digest")
        return self.directory / key[:2] / f"{key}.npz"

    def get(self, key: str) -> AnalysisResult | None:
        path = self.path_for(key)
        if not path.is_file():
            return None
        try:
            result = AnalysisResult.load(path)
        except (BadZipFile, EOFError, KeyError, TypeError, ValueError, OSError):
            return None
        return result if result.metadata.get("cache_key") == key else None

    def put(self, key: str, result: AnalysisResult) -> Path:
        path = self.path_for(key)
        snapshot = replace(result, metadata={**result.metadata, "cache_key": key})
        snapshot.save(path, compressed=self.compressed)
        # Publish the key on the caller's result only after the atomic save succeeds.
        result.metadata["cache_key"] = key
        return path

    def get_or_compute(
        self, key: str, compute: Callable[[], AnalysisResult]
    ) -> tuple[AnalysisResult, bool]:
        cached = self.get(key)
        if cached is not None:
            return cached, True
        result = compute()
        self.put(key, result)
        return result, False


def analysis_cache_key(solver, mode: str, config) -> str:
    """Fingerprint model content, selected chain, solver, mode, and config."""
    payload = {
        "schema": 3,
        "model": _model_digest(solver.model),
        "base_link": solver.base_link,
        "tip_link": solver.tip_link,
        "solver": _normalize(solver.config),
        "runtime_backend": solver.backend,
        "runtime_device": solver.device,
        "mode": mode,
        "config": _normalize(config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _model_digest(model) -> str:
    # Hash the parsed model used for computation. Its source may have changed
    # or disappeared since construction, without changing this solver's model.
    description = {
        "name": model.name,
        "links": model.links,
        "joints": [
            {
                "name": joint.name,
                "kind": joint.kind,
                "parent": joint.parent,
                "child": joint.child,
                "origin": joint.origin,
                "axis": joint.axis,
                "limit": joint.limit,
            }
            for joint in model.joints
        ],
    }
    encoded = json.dumps(_normalize(description), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _normalize(value):
    if is_dataclass(value):
        return {
            field.name: _normalize(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        raise TypeError("callable configuration values cannot be cached")
    return value
