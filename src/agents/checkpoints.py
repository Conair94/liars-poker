"""Component-addressable checkpoint I/O (AR-0a §5).

A single canonical pair `save_component` / `load_component` so every
phase produces interoperable artifacts. `parent_run` is recorded for
provenance only — the loader does not follow it.

Optimizer state belongs in a separate file (`<path>.opt.pt`); this
module does not handle it.
"""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime
from typing import Any, Literal

import torch

SCHEMA_VERSION = 1

ComponentName = Literal["handmodel", "callpolicy", "bidpolicy", "unified"]
_VALID_COMPONENTS = {"handmodel", "callpolicy", "bidpolicy", "unified"}


def _git_sha_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=os.getcwd(),
            text=True,
        )
        return out.strip() or "?"
    except Exception:
        return "?"


def save_component(
    path: str,
    *,
    component: ComponentName,
    config: dict,
    state_dict: dict[str, Any],
    iter: int,
    parent_run: str | None = None,
) -> None:
    """Write a checkpoint blob to `path`. Atomic via tmp-file rename."""
    if component not in _VALID_COMPONENTS:
        raise ValueError(f"unknown component {component!r}; "
                         f"must be one of {sorted(_VALID_COMPONENTS)}")

    blob = {
        "schema_version": SCHEMA_VERSION,
        "component":      component,
        "config":         dict(config),
        "state_dict":     dict(state_dict),
        "iter":           int(iter),
        "git_sha":        _git_sha_short(),
        "parent_run":     parent_run,
        "saved_at":       datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)


def load_component(path: str) -> dict:
    """Load a checkpoint blob. Raises FileNotFoundError if path is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint schema version mismatch at {path}: "
            f"got {blob.get('schema_version')!r}, expected {SCHEMA_VERSION}"
        )
    if blob.get("component") not in _VALID_COMPONENTS:
        raise ValueError(f"checkpoint component {blob.get('component')!r} invalid")
    return blob
