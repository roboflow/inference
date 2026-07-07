"""Parity guard: the static mirror in mmp_translation must match the
inference_server handler registry. Skips where inference_server (and its
torch dependency chain) is not installed.
"""

import sys
from pathlib import Path

import pytest

from inference.core.managers.mmp_translation import NEW_WORLD_HANDLERS

_REPO_ROOT = Path(__file__).resolve().parents[5]


def _load_registry():
    sys.path.insert(0, str(_REPO_ROOT / "inference_server"))
    try:
        from inference_server.framework.registry import DYNAMIC_MODELS_HANDLERS

        return DYNAMIC_MODELS_HANDLERS
    finally:
        sys.path.pop(0)


def test_static_mirror_matches_inference_server_registry():
    try:
        registry = _load_registry()
    except ImportError as error:
        pytest.skip(f"inference_server not importable here: {error}")
    assert set(registry.keys()) == set(NEW_WORLD_HANDLERS)
