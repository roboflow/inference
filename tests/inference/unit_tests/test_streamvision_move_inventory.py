"""Retained host-module inventory for streamvision compatibility."""

from __future__ import annotations

import json
from pathlib import Path

_INVENTORY_PATH = Path(__file__).with_name("streamvision_move_inventory.json")
_INVENTORY = json.loads(_INVENTORY_PATH.read_text())

_RETAINED_LEGACY_NAMES = {
    "inference.core.interfaces.stream.inference_pipeline",
    "inference.core.interfaces.stream.stream",
    "inference.core.interfaces.stream.model_handlers.roboflow_models",
    "inference.core.interfaces.stream.model_handlers.yolo_world",
}


def test_retained_legacy_names_are_exactly_the_four_expected() -> None:
    retained = {entry["legacy"] for entry in _INVENTORY["retained_legacy"]}
    assert retained == _RETAINED_LEGACY_NAMES
    module_legacy_names = {entry["legacy"] for entry in _INVENTORY["modules"]}
    assert not (retained & module_legacy_names)


def test_legacy_stream_directory_matches_retained_inventory_plus_host() -> None:
    # Guards against a retained module going missing or an extra one appearing.
    repo_root = Path(__file__).resolve().parents[3]
    legacy_stream_root = (
        repo_root / "inference" / "core" / "interfaces" / "legacy_stream"
    )

    actual_modules = {
        ".".join(path.relative_to(repo_root).with_suffix("").parts)
        for path in legacy_stream_root.rglob("*.py")
        if path.name != "__init__.py"
    }

    expected_modules = {
        entry["retained"] for entry in _INVENTORY["retained_legacy"]
    } | {"inference.core.interfaces.legacy_stream.host"}

    assert actual_modules == expected_modules
