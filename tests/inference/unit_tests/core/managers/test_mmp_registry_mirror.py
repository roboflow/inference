"""Parity guard: the static mirror in mmp_translation must match the
inference_server handler registry. The registry is source-parsed rather
than imported, so this runs without inference_server's runtime
dependencies installed.
"""

from __future__ import annotations

import ast
from pathlib import Path

from inference.core.managers.mmp_translation import NEW_WORLD_HANDLERS

# NOTE: parses registry.py's `import ...description` lines and each
# description module's `_register("type", "action", ...)` calls as source
# text. If registry.py's registration mechanism stops matching that literal
# shape, this parser needs a matching update.
_REPO_ROOT = Path(__file__).resolve().parents[5]
_PKG_ROOT = _REPO_ROOT / "inference_server"
_REGISTRY_PATH = _PKG_ROOT / "inference_server" / "framework" / "registry.py"


def _description_module_paths(registry_path: Path) -> list[Path]:
    tree = ast.parse(registry_path.read_text(), filename=str(registry_path))
    paths: list[Path] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Import):
            continue
        for alias in node.names:
            if alias.name.startswith(
                "inference_server.handlers."
            ) and alias.name.endswith(".description"):
                paths.append(
                    _PKG_ROOT / Path(*alias.name.split(".")).with_suffix(".py")
                )
    return paths


def _registered_pairs(module_path: Path) -> set[tuple[str, str]]:
    tree = ast.parse(module_path.read_text(), filename=str(module_path))
    pairs: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_register"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[1], ast.Constant)
        ):
            pairs.add((node.args[0].value, node.args[1].value))
    return pairs


def _load_registry() -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for module_path in _description_module_paths(_REGISTRY_PATH):
        pairs |= _registered_pairs(module_path)
    return pairs


def test_static_mirror_matches_inference_server_registry():
    assert _load_registry() == set(NEW_WORLD_HANDLERS)
