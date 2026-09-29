"""Move inventory for the future `streamvision` extraction.

`streamvision_move_inventory.json` (next to this file) lists every file
under `inference.core.interfaces.{camera,stream,stream_manager}` at the pinned
revision, and the `streamvision.*` name each one is expected to move to.
Four legacy modules are retained under `inference.core.interfaces.legacy_stream`
instead of moving. This test only checks the inventory against Git history; it
does not import `inference` or `streamvision`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_PINNED_REVISION = "8e14506749160290049e5c7e02c5ddb3ae1addc5"
_INVENTORY_PATH = Path(__file__).with_name("streamvision_move_inventory.json")
_INVENTORY = json.loads(_INVENTORY_PATH.read_text())

_RETAINED_LEGACY_NAMES = {
    "inference.core.interfaces.stream.inference_pipeline",
    "inference.core.interfaces.stream.stream",
    "inference.core.interfaces.stream.model_handlers.roboflow_models",
    "inference.core.interfaces.stream.model_handlers.yolo_world",
}


def test_inventory_pins_baseline_revision() -> None:
    assert _INVENTORY["generated_from_revision"] == _PINNED_REVISION


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


def test_inventory_matches_historical_git_tree() -> None:
    # Verifies the inventory against the pinned revision's tree, not the current tree.
    if shutil.which("git") is None:
        pytest.skip("git not available")

    repo_root = Path(__file__).resolve().parents[3]
    is_work_tree = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
    )
    if is_work_tree.returncode != 0:
        pytest.skip("not a git work tree")

    revision_check = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "cat-file",
            "-e",
            f"{_PINNED_REVISION}^{{commit}}",
        ],
        capture_output=True,
        text=True,
    )
    if revision_check.returncode != 0:
        pytest.skip("pinned revision unreachable from local git history")

    listing = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo_root),
            "ls-tree",
            "-r",
            "--name-only",
            _PINNED_REVISION,
            "--",
            "inference/core/interfaces/camera",
            "inference/core/interfaces/stream",
            "inference/core/interfaces/stream_manager",
        ],
        stderr=subprocess.STDOUT,
        text=True,
    )

    legacy_root = _INVENTORY["legacy_root"]
    canonical_root = _INVENTORY["canonical_root"]
    retained_root = _INVENTORY["retained_legacy_root"]

    def _to_canonical(legacy: str) -> str:
        if legacy in _RETAINED_LEGACY_NAMES:
            tail = legacy[len("inference.core.interfaces.stream.") :]
            return retained_root + "." + tail
        assert legacy == legacy_root or legacy.startswith(legacy_root + ".")
        return canonical_root + legacy[len(legacy_root) :]

    historical_packages: set[str] = set()
    historical_modules: set[str] = set()
    historical_retained: set[str] = set()
    for path in listing.splitlines():
        if not path.endswith(".py"):
            continue
        dotted = path[:-3].replace("/", ".")
        if dotted.endswith(".__init__"):
            historical_packages.add(dotted[: -len(".__init__")])
        elif dotted in _RETAINED_LEGACY_NAMES:
            historical_retained.add(dotted)
        else:
            historical_modules.add(dotted)

    inventory_packages = {entry["legacy"] for entry in _INVENTORY["packages"]}
    inventory_modules = {entry["legacy"] for entry in _INVENTORY["modules"]}
    inventory_retained = {entry["legacy"] for entry in _INVENTORY["retained_legacy"]}
    assert historical_packages == inventory_packages, (
        f"package mismatch: "
        f"history-only={historical_packages - inventory_packages!r}, "
        f"inventory-only={inventory_packages - historical_packages!r}"
    )
    assert historical_modules == inventory_modules, (
        f"module mismatch: "
        f"history-only={historical_modules - inventory_modules!r}, "
        f"inventory-only={inventory_modules - historical_modules!r}"
    )
    assert historical_retained == inventory_retained == _RETAINED_LEGACY_NAMES

    for entry in (
        _INVENTORY["packages"] + _INVENTORY["modules"] + _INVENTORY["retained_legacy"]
    ):
        legacy = entry["legacy"]
        expected = _to_canonical(legacy)
        actual = entry.get("canonical", entry.get("retained"))
        assert actual == expected, f"{legacy!r}: expected {expected!r}, got {actual!r}"
