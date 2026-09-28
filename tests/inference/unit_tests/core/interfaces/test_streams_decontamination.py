"""Decontamination ratchet for camera/stream/stream_manager.

The frozen ``_ALLOWLIST`` below is empty and must stay empty: this scan
fails on any forbidden import found in these three trees. The four legacy
facade files are excluded from the scan by their exact source path, not by
prefix, because they self-replace in sys.modules.

Precedent: workflows/tests/unit_tests/test_decontamination_lint.py:1 (single
tree, single allowed prefix, already-zero target). This generalizes the same
AST + relative-import + quoted-import scan to three trees, with one extra
rule: the four facade modules are forbidden import sources for every *other*
host-neutral module in these trees, even though their dotted name lies under
an otherwise-allowed prefix.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pytest

from .conftest import require_git_baseline_history

# Baseline commit that every frozen manifest in this suite is pinned to.
BASELINE_SHA = "65ad2beaaca0825bffc2fbbe99199d3a40994324"

PROJECT_ROOT = Path(__file__).resolve().parents[5]
INTERFACES_ROOT = PROJECT_ROOT / "inference" / "core" / "interfaces"
TREES = ("camera", "stream", "stream_manager")
for _tree in TREES:
    assert (INTERFACES_ROOT / _tree).is_dir(), f"missing tree: {_tree}"

ALLOWED_PREFIXES = tuple(f"inference.core.interfaces.{tree}" for tree in TREES)

# Host-owned facade aliases; no host-neutral module in these trees may import them.
FACADE_MODULES = frozenset(
    {
        "inference.core.interfaces.stream.inference_pipeline",
        "inference.core.interfaces.stream.stream",
        "inference.core.interfaces.stream.model_handlers.roboflow_models",
        "inference.core.interfaces.stream.model_handlers.yolo_world",
    }
)

# Facade files self-replace in sys.modules; excluded from the scan by exact path only.
LEGACY_PREFIX = "inference.core.interfaces.legacy_stream"
FACADE_TARGETS = {
    "inference.core.interfaces.stream.inference_pipeline": (
        f"{LEGACY_PREFIX}.inference_pipeline"
    ),
    "inference.core.interfaces.stream.stream": f"{LEGACY_PREFIX}.stream",
    "inference.core.interfaces.stream.model_handlers.roboflow_models": (
        f"{LEGACY_PREFIX}.model_handlers.roboflow_models"
    ),
    "inference.core.interfaces.stream.model_handlers.yolo_world": (
        f"{LEGACY_PREFIX}.model_handlers.yolo_world"
    ),
}
assert set(FACADE_TARGETS) == FACADE_MODULES


def _source_path(module: str) -> str:
    return "/".join(module.split(".")) + ".py"


FACADE_SOURCE_PATHS = frozenset(_source_path(module) for module in FACADE_MODULES)

_STRING_IMPORT = re.compile(
    r"""["'](?:from|import)\s+(inference(?![A-Za-z0-9_])(?:\.[A-Za-z0-9_]+)*)"""
)


def _resolve_relative(module: str, level: int, path: Path) -> str:
    pkg_parts = path.relative_to(PROJECT_ROOT).with_suffix("").parts[:-1]
    base = pkg_parts[: len(pkg_parts) - (level - 1)] if level > 1 else pkg_parts
    return ".".join([*base, *(module.split(".") if module else [])])


def _module_exists_on_disk(module: str) -> bool:
    # Distinguishes a submodule import from a plain attribute import by disk check.
    if not module:
        return False
    rel = Path(*module.split("."))
    return (PROJECT_ROOT / rel.with_suffix(".py")).is_file() or (
        PROJECT_ROOT / rel / "__init__.py"
    ).is_file()


def _imported_modules(tree: ast.AST, path: Path) -> Set[str]:
    # ast.walk (not tree.body) so function-local imports are caught too.
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module
            else:
                base = _resolve_relative(node.module or "", node.level, path)
            if base:
                modules.add(base)
            for alias in node.names:
                candidate = f"{base}.{alias.name}" if base else alias.name
                if _module_exists_on_disk(candidate):
                    modules.add(candidate)
    return modules


def _is_forbidden(module: str) -> bool:
    if module in FACADE_MODULES:
        return True
    if module == "inference":
        return True
    if not module.startswith("inference."):
        return False
    # Boundary match: "streamx" must not match under the "stream" prefix.
    return not any(
        module == prefix or module.startswith(f"{prefix}.")
        for prefix in ALLOWED_PREFIXES
    )


def collect_violations(root: Path = INTERFACES_ROOT) -> Set[Tuple[str, str]]:
    violations = set()
    for tree_name in TREES:
        for path in sorted((root / tree_name).rglob("*.py")):
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            if relative in FACADE_SOURCE_PATHS:
                continue
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            for module in _imported_modules(tree, path):
                if _is_forbidden(module):
                    violations.add((relative, module))
            for module in _STRING_IMPORT.findall(source):
                if _is_forbidden(module):
                    violations.add((relative, f"{module} (quoted)"))
    return violations


# Generated by the ast scan above; shrink only alongside a real import removal.
_ALLOWLIST = frozenset()


def test_movable_files_have_no_allowlisted_production_dependency() -> None:
    assert _ALLOWLIST == frozenset()


def test_forbidden_imports_match_frozen_allowlist() -> None:
    actual = collect_violations()
    new = actual - _ALLOWLIST
    stale = _ALLOWLIST - actual
    assert (
        not new
    ), "new forbidden import(s) not in the frozen allowlist:\n  " + "\n  ".join(
        f"{p} -> {m}" for p, m in sorted(new)
    )
    assert not stale, (
        "allowlist entries no longer actually violated - shrink the "
        "allowlist to match real progress:\n  "
        + "\n  ".join(f"{p} -> {m}" for p, m in sorted(stale))
    )


def _parse_fixture(source: str, fake_path: Path) -> ast.AST:
    return ast.parse(source, filename=str(fake_path))


def test_checker_catches_function_local_forbidden_import() -> None:
    fake_path = INTERFACES_ROOT / "camera" / "_fixture_local_import.py"
    source = (
        "def helper():\n"
        "    from inference.core.roboflow_api import get_roboflow_model_data\n"
        "    return get_roboflow_model_data\n"
    )
    tree = _parse_fixture(source, fake_path)
    modules = _imported_modules(tree, fake_path)
    assert "inference.core.roboflow_api" in modules
    assert _is_forbidden("inference.core.roboflow_api")


def test_checker_catches_relative_forbidden_import() -> None:
    # Relative "from ...roboflow_api import x" resolves to inference.core.roboflow_api.
    fake_path = INTERFACES_ROOT / "camera" / "_fixture_relative_import.py"
    source = "from ...roboflow_api import get_roboflow_model_data\n"
    tree = _parse_fixture(source, fake_path)
    modules = _imported_modules(tree, fake_path)
    assert "inference.core.roboflow_api" in modules
    assert _is_forbidden("inference.core.roboflow_api")


def test_checker_catches_quoted_string_import() -> None:
    # Mirrors a string later exec()'d into code - invisible to ast.parse on this file.
    source = (
        "TEMPLATE = "
        '"from inference.core.roboflow_api import get_roboflow_model_data"\n'
    )
    found = _STRING_IMPORT.findall(source)
    assert found == ["inference.core.roboflow_api"]
    assert _is_forbidden(found[0])


def test_checker_resolves_submodule_import_but_not_attribute_import() -> None:
    # import logger from inference.core must resolve the submodule, not just the parent.
    fake_path = INTERFACES_ROOT / "camera" / "_fixture_submodule_import.py"
    submodule_source = "from inference.core import logger\n"
    modules = _imported_modules(_parse_fixture(submodule_source, fake_path), fake_path)
    assert "inference.core.logger" in modules
    assert "inference.core" in modules

    # Importing a class (not a module) must not be recorded as a module path.
    attribute_source = "from inference.core.exceptions import EngineExecutionError\n"
    modules = _imported_modules(_parse_fixture(attribute_source, fake_path), fake_path)
    assert "inference.core.exceptions" in modules
    assert "inference.core.exceptions.EngineExecutionError" not in modules


def test_module_exists_on_disk_matches_checker_resolution() -> None:
    assert _module_exists_on_disk("inference.core.logger")
    assert not _module_exists_on_disk("inference.core.logger.NotAModule")
    assert not _module_exists_on_disk("")


def test_checker_boundary_matches_allowed_prefixes_exactly() -> None:
    # "streamx" must not match "stream" via a naive startswith with no path boundary.
    assert _is_forbidden("inference.core.interfaces.streamx.entities")
    assert _is_forbidden("inference.core.interfaces.streamx")
    # The exact prefix itself, and genuine children of it, stay allowed.
    assert not _is_forbidden("inference.core.interfaces.stream")
    assert not _is_forbidden("inference.core.interfaces.stream.entities")


def test_checker_does_not_flag_inference_models_or_inference_sdk() -> None:
    # inference_models/inference_sdk are separate distributions, not "inference.*".
    assert not _is_forbidden("inference_models.models.base")
    assert not _is_forbidden("inference_sdk.http.client")


def test_checker_allows_same_tree_non_facade_imports() -> None:
    assert not _is_forbidden("inference.core.interfaces.camera.entities")
    assert not _is_forbidden("inference.core.interfaces.stream.entities")
    assert not _is_forbidden("inference.core.interfaces.stream_manager.api.entities")


def test_facade_modules_are_forbidden_even_under_an_allowed_prefix() -> None:
    # The four facades sit under an allowed prefix but stay forbidden as host-only.
    for module in FACADE_MODULES:
        assert module.startswith(ALLOWED_PREFIXES)
        assert _is_forbidden(module)


def test_facade_modules_are_exactly_four() -> None:
    assert len(FACADE_MODULES) == 4


def test_facade_source_paths_are_exactly_the_four_facade_files() -> None:
    # Exclusion is by exact file path, not directory - a sibling module still scanned.
    assert FACADE_SOURCE_PATHS == {
        "inference/core/interfaces/stream/inference_pipeline.py",
        "inference/core/interfaces/stream/stream.py",
        "inference/core/interfaces/stream/model_handlers/roboflow_models.py",
        "inference/core/interfaces/stream/model_handlers/yolo_world.py",
    }
    for path in FACADE_SOURCE_PATHS:
        assert (PROJECT_ROOT / path).is_file(), path


def _is_type_checking_guard(node: ast.If) -> bool:
    test = node.test
    return isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"


def _wildcard_import_nodes(tree: ast.AST) -> List[ast.ImportFrom]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and any(alias.name == "*" for alias in node.names)
    ]


def _wildcard_imports_guarded_by_type_checking(tree: ast.AST) -> Set[ast.ImportFrom]:
    guarded: Set[ast.ImportFrom] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_type_checking_guard(node):
            # node.body only: walking the If would also accept an else/elif branch.
            guarded_body = ast.Module(body=node.body, type_ignores=[])
            guarded.update(_wildcard_import_nodes(guarded_body))
    return guarded


def test_wildcard_import_in_type_checking_else_branch_is_not_guarded() -> None:
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    pass\n"
        "else:\n"
        "    from os import *\n"
    )
    tree = ast.parse(source)

    all_wildcard_imports = _wildcard_import_nodes(tree)
    guarded_wildcard_imports = _wildcard_imports_guarded_by_type_checking(tree)

    assert all_wildcard_imports
    assert not guarded_wildcard_imports


def test_wildcard_import_inside_type_checking_body_is_guarded() -> None:
    source = (
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from os import *\n"
    )
    tree = ast.parse(source)

    all_wildcard_imports = _wildcard_import_nodes(tree)
    guarded_wildcard_imports = _wildcard_imports_guarded_by_type_checking(tree)

    assert set(all_wildcard_imports) == guarded_wildcard_imports


@pytest.mark.parametrize("facade_module", sorted(FACADE_MODULES))
def test_facade_files_only_alias_their_exact_legacy_target(facade_module: str) -> None:
    # Only sys, typing (guarded wildcard) and the legacy target may be imported.
    path = PROJECT_ROOT / _source_path(facade_module)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    target = FACADE_TARGETS[facade_module]
    target_package = target.rsplit(".", 1)[0]

    assert _imported_modules(tree, path) == {"sys", "typing", target_package, target}
    assert "sys.modules[__name__] = _implementation" in ast.unparse(tree)

    # The wildcard import must exist and only ever sit under TYPE_CHECKING.
    all_wildcard_imports = _wildcard_import_nodes(tree)
    guarded_wildcard_imports = _wildcard_imports_guarded_by_type_checking(tree)
    assert all_wildcard_imports, "expected a TYPE_CHECKING-guarded wildcard import"
    assert set(all_wildcard_imports) == guarded_wildcard_imports


@pytest.mark.parametrize("facade_module", sorted(FACADE_MODULES))
def test_facade_name_resolves_to_the_legacy_module_object(
    facade_module: str, stub_ultralytics_if_missing
) -> None:
    import importlib

    facade = importlib.import_module(facade_module)
    target = importlib.import_module(FACADE_TARGETS[facade_module])
    parent_name, _, child_name = facade_module.rpartition(".")

    assert facade is target
    assert getattr(importlib.import_module(parent_name), child_name) is target


# The stream manager's pipeline host; new module, so it has no historical facade.
LEGACY_HOST_MODULE = f"{LEGACY_PREFIX}.host"


def test_legacy_stream_holds_exactly_the_four_facade_targets() -> None:
    # Host-owned side of the facades; nothing may grow under legacy_stream unnamed here.
    legacy_root = INTERFACES_ROOT / "legacy_stream"
    modules = {
        ".".join(path.relative_to(PROJECT_ROOT).with_suffix("").parts)
        for path in legacy_root.rglob("*.py")
        if path.name != "__init__.py"
    }
    assert modules == set(FACADE_TARGETS.values()) | {LEGACY_HOST_MODULE}


def test_host_neutral_trees_never_import_legacy_stream() -> None:
    # Imports into host-owned implementations are forbidden, directly or via facades.
    for target in FACADE_TARGETS.values():
        assert _is_forbidden(target)
    assert _is_forbidden(LEGACY_PREFIX)
    assert not any(
        module == LEGACY_PREFIX or module.startswith(f"{LEGACY_PREFIX}.")
        for _, module in collect_violations()
    )


@pytest.mark.parametrize("tree_name", TREES)
def test_manifest_source_paths_match_git_tree_at_baseline_sha(tree_name: str) -> None:
    import json
    import subprocess

    inventory_path = (
        PROJECT_ROOT
        / "tests"
        / "inference"
        / "unit_tests"
        / "streams_compat_inventory.json"
    )
    manifest = json.loads(inventory_path.read_text())
    assert manifest["baseline_sha"] == BASELINE_SHA
    require_git_baseline_history(BASELINE_SHA, project_root=PROJECT_ROOT)

    manifest_paths = {
        entry["source_path"]
        for entry in manifest["modules"]
        if entry["tree"] == tree_name
    }

    result = subprocess.run(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            BASELINE_SHA,
            "--",
            f"inference/core/interfaces/{tree_name}",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    git_paths = {line for line in result.stdout.splitlines() if line.endswith(".py")}

    assert manifest_paths == git_paths, (
        f"frozen manifest for {tree_name!r} drifted from the git tree at "
        f"{BASELINE_SHA}:\n  missing from manifest: {sorted(git_paths - manifest_paths)}\n"
        f"  missing from git tree: {sorted(manifest_paths - git_paths)}"
    )


def test_manifest_module_count_is_47() -> None:
    import json

    inventory_path = (
        PROJECT_ROOT
        / "tests"
        / "inference"
        / "unit_tests"
        / "streams_compat_inventory.json"
    )
    manifest = json.loads(inventory_path.read_text())
    assert manifest["module_count"] == 47
    assert len(manifest["modules"]) == 47


def test_manifest_marks_exactly_the_four_retained_facades() -> None:
    import json

    inventory_path = (
        PROJECT_ROOT
        / "tests"
        / "inference"
        / "unit_tests"
        / "streams_compat_inventory.json"
    )
    manifest = json.loads(inventory_path.read_text())
    retained = {
        entry["module"]
        for entry in manifest["modules"]
        if entry["retained_host_exception"]
    }
    assert retained == FACADE_MODULES


def test_manifest_expected_canonical_names_match_mapping_rule() -> None:
    # Every module maps to "roboflow_streams." except the 4 facades, which map to None.
    import json

    inventory_path = (
        PROJECT_ROOT
        / "tests"
        / "inference"
        / "unit_tests"
        / "streams_compat_inventory.json"
    )
    manifest = json.loads(inventory_path.read_text())
    legacy_prefix = "inference.core.interfaces."
    canonical_prefix = "roboflow_streams."

    for entry in manifest["modules"]:
        module = entry["module"]
        assert module.startswith(legacy_prefix), module
        if module in FACADE_MODULES:
            assert entry["expected_canonical_name"] is None, module
        else:
            expected = canonical_prefix + module[len(legacy_prefix) :]
            assert entry["expected_canonical_name"] == expected, module


def _module_dunder_all(tree: ast.Module) -> Optional[Set[str]]:
    # A declared __all__ overrides the underscore heuristic, not just adds to it.
    for node in tree.body:
        targets = None
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        if (
            targets
            and "__all__" in targets
            and isinstance(node.value, (ast.List, ast.Tuple))
        ):
            return {
                elt.value
                for elt in node.value.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            }
    return None


def _top_level_public_names(source: str) -> Set[str]:
    tree = ast.parse(source)
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(
                target.id
                for target in node.targets
                if isinstance(target, ast.Name) and not target.id.startswith("_")
            )
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and not node.target.id.startswith("_"):
                names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            # Re-exported imports count as public surface (e.g. sinks.py's VideoFrame).
            for alias in node.names:
                if alias.name == "*":
                    continue
                exposed = alias.asname or alias.name.split(".")[0]
                if not exposed.startswith("_"):
                    names.add(exposed)
    dunder_all = _module_dunder_all(tree)
    if dunder_all is not None:
        return dunder_all
    return names


@pytest.mark.parametrize("tree_name", TREES)
def test_manifest_public_top_level_names_match_git_tree_at_baseline_sha(
    tree_name: str,
) -> None:
    # Freezes each module's exact non-underscore top-level names, checked per module.
    import json
    import subprocess

    inventory_path = (
        PROJECT_ROOT
        / "tests"
        / "inference"
        / "unit_tests"
        / "streams_compat_inventory.json"
    )
    manifest = json.loads(inventory_path.read_text())
    require_git_baseline_history(BASELINE_SHA, project_root=PROJECT_ROOT)

    for entry in manifest["modules"]:
        if entry["tree"] != tree_name:
            continue
        result = subprocess.run(
            ["git", "show", f"{BASELINE_SHA}:{entry['source_path']}"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        actual_names = _top_level_public_names(result.stdout)
        assert actual_names == set(entry["public_top_level_names"]), entry[
            "source_path"
        ]


def test_top_level_public_names_includes_imported_public_names() -> None:
    # A name only re-exported via import (no local def) must still count as public.
    source = (
        "from .entities import VideoFrame\n"
        "from ..camera.entities import SinkHandler as _SinkHandlerAlias\n"
        "import inference.core.logger\n"
        "import numpy as np\n"
        "from . import _private_helper\n"
        "from .other import public_name, _hidden_name\n"
        "def local_public(): ...\n"
    )
    # The leading-underscore convention applies to aliases just like plain names.
    assert _top_level_public_names(source) == {
        "VideoFrame",
        "inference",
        "np",
        "public_name",
        "local_public",
    }


def test_top_level_public_names_respects_dunder_all_override() -> None:
    source = (
        "from .pipeline import build_gstreamer_rtsp_pipeline, gstreamer_rtsp_capture_available\n"
        "from .rtsp_tls import GST_SSL_CA_CERTIFICATE_ENV_VAR, is_rtsps_url\n"
        "__all__ = ['GStreamerRtspVideoFrameProducer', 'should_use_gstreamer_rtsp_producer']\n"
        "def should_use_gstreamer_rtsp_producer(): ...\n"
        "class GStreamerRtspVideoFrameProducer: ...\n"
    )
    assert _top_level_public_names(source) == {
        "GStreamerRtspVideoFrameProducer",
        "should_use_gstreamer_rtsp_producer",
    }


def test_sinks_manifest_entry_covers_its_imported_public_names() -> None:
    # Regression: sinks.py entry must list VideoFrame/SinkHandler, not just locals.
    import json

    inventory_path = (
        PROJECT_ROOT
        / "tests"
        / "inference"
        / "unit_tests"
        / "streams_compat_inventory.json"
    )
    manifest = json.loads(inventory_path.read_text())
    entry = next(
        entry
        for entry in manifest["modules"]
        if entry["source_path"] == "inference/core/interfaces/stream/sinks.py"
    )
    assert {"VideoFrame", "SinkHandler"} <= set(entry["public_top_level_names"])


# Closes the import graph so no facade, __init__, or loader hook leaks host code.
TESTS_ROOT = PROJECT_ROOT / "tests" / "inference" / "unit_tests" / "core" / "interfaces"
A_END_MANIFEST_PATH = (
    PROJECT_ROOT / "tests" / "inference" / "unit_tests" / "streams_a_end_manifest.json"
)

# _workflows_compat._HOST_EXPORTS binds this onto sinks at runtime, not via import.
HOST_EXPORT_HOOKS = {
    "inference.core.interfaces.stream.sinks": (
        ("ActiveLearningMiddleware", "inference.core.active_learning.middlewares"),
    ),
}

_HOST_MODULE_LITERAL = re.compile(
    r"""["'](inference(?:\.[A-Za-z0-9_]+)+)(?::[A-Za-z0-9_.]+)?["']"""
)


def _module_name(path: Path) -> str:
    parts = path.relative_to(PROJECT_ROOT).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def movable_source_paths() -> Set[str]:
    return {
        path.relative_to(PROJECT_ROOT).as_posix()
        for tree_name in TREES
        for path in (INTERFACES_ROOT / tree_name).rglob("*.py")
    } - FACADE_SOURCE_PATHS


def tree_test_paths_by_owner() -> Dict[str, Set[str]]:
    """Split the tree test files: movable ones import no host module."""
    owners: Dict[str, Set[str]] = {"move": set(), "host": set()}
    for tree_name in TREES:
        for path in (TESTS_ROOT / tree_name).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports_host = any(
                _is_forbidden(module) for module in _imported_modules(tree, path)
            )
            owner = "host" if imports_host else "move"
            owners[owner].add(path.relative_to(PROJECT_ROOT).as_posix())
    return owners


def test_movable_files_name_no_host_module_in_string_literals() -> None:
    # Catches importlib.import_module(...) string imports invisible to the AST scan.
    literals = set()
    for relative in sorted(movable_source_paths()):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        for module in _HOST_MODULE_LITERAL.findall(source):
            if _is_forbidden(module):
                literals.add((relative, module))
    assert literals == set()


def test_host_module_literal_check_catches_a_dynamic_import() -> None:
    source = 'importlib.import_module("inference.core.managers.base")'
    assert _HOST_MODULE_LITERAL.findall(source) == ["inference.core.managers.base"]
    assert _is_forbidden(_HOST_MODULE_LITERAL.findall(source)[0])


def test_host_export_hooks_are_exactly_the_declared_manifest() -> None:
    from inference import _workflows_compat

    assert _workflows_compat._HOST_EXPORTS == HOST_EXPORT_HOOKS
    for module, exports in HOST_EXPORT_HOOKS.items():
        assert _source_path(module) in movable_source_paths()
        for _, host_module in exports:
            assert _is_forbidden(host_module)


# Stubs inference's host-root packages; reports other inference.* modules loaded.
_STUBBED_ROOTS_PROBE = """
import importlib, json, sys, traceback, types
root, trees, modules = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3])
for name in ("inference", "inference.core", "inference.core.interfaces"):
    package = types.ModuleType(name)
    package.__path__ = [root + "/" + name.replace(".", "/")]
    sys.modules[name] = package
tree_prefixes = tuple("inference.core.interfaces." + tree for tree in trees)


def in_trees(name):
    return any(name == p or name.startswith(p + ".") for p in tree_prefixes)


loaded = {}


class Recorder:
    def find_spec(self, name, path=None, target=None):
        if name.startswith("inference.") and not in_trees(name):
            frames = traceback.extract_stack()[:-1]
            loaded.setdefault(name, [f"{f.filename}:{f.lineno}" for f in frames][-6:])
        return None


sys.meta_path.insert(0, Recorder())
failed = {}
for module in modules:
    try:
        importlib.import_module(module)
    except Exception as error:
        failed[module] = repr(error)
print(json.dumps({"failed": failed, "loaded": loaded}))
"""


def _import_with_stubbed_host_roots(modules: List[str]) -> dict:
    import json
    import os
    import subprocess
    import sys

    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "DISABLE_VERSION_CHECK": "True",
        "PYTHONPATH": os.pathsep.join(
            [str(PROJECT_ROOT / "workflows"), str(PROJECT_ROOT / "inference_models")]
        ),
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _STUBBED_ROOTS_PROBE,
            str(PROJECT_ROOT),
            json.dumps(TREES),
            json.dumps(modules),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_movable_import_closure_loads_no_host_module() -> None:
    modules = sorted(
        _module_name(PROJECT_ROOT / path) for path in movable_source_paths()
    )

    result = _import_with_stubbed_host_roots(modules)

    assert result["failed"] == {}
    assert result["loaded"] == {}


def test_import_closure_probe_sees_host_imports_behind_a_facade() -> None:
    # Positive control: allowed-looking facade resolves to host code the probe catches.
    facade = "inference.core.interfaces.stream.model_handlers.roboflow_models"

    result = _import_with_stubbed_host_roots([facade])

    assert "inference.core.interfaces.legacy_stream" in result["loaded"]


def _a_end_manifest() -> dict:
    import json

    return json.loads(A_END_MANIFEST_PATH.read_text())


def test_a_end_manifest_records_a_real_base_commit_and_no_candidate() -> None:
    manifest = _a_end_manifest()
    assert manifest["working_tree_dirty"] is True
    assert "candidate_sha" not in manifest
    # Checked before the history guard so a corrupted SHA fails without git history.
    assert manifest["base_head_sha"] == BASELINE_SHA

    require_git_baseline_history(manifest["base_head_sha"], project_root=PROJECT_ROOT)


def test_a_end_manifest_matches_the_movable_and_host_split() -> None:
    manifest = _a_end_manifest()
    tree_tests = tree_test_paths_by_owner()

    assert set(manifest["move"]["source"]) == movable_source_paths()
    assert set(manifest["move"]["tests"]) == tree_tests["move"]
    assert set(manifest["host"]["tree_tests"]) == tree_tests["host"]
    assert {
        module: entry["target"]
        for module, entry in manifest["host"]["retained_facades"].items()
    } == FACADE_TARGETS
    assert {
        entry["path"] for entry in manifest["host"]["retained_facades"].values()
    } == FACADE_SOURCE_PATHS
    assert {
        module: [list(export) for export in exports]
        for module, exports in HOST_EXPORT_HOOKS.items()
    } == manifest["host"]["host_export_hooks"]


def test_a_end_manifest_paths_exist_and_have_one_owner() -> None:
    manifest = _a_end_manifest()
    moved = set(manifest["move"]["source"]) | set(manifest["move"]["tests"])
    moved |= set(manifest["move"]["test_support"])
    host = set(manifest["host"]["source"]) | set(manifest["host"]["tree_tests"])
    host |= set(manifest["host"]["tests"])
    host |= {entry["path"] for entry in manifest["host"]["retained_facades"].values()}

    assert moved.isdisjoint(host)
    for path in moved | host:
        assert (PROJECT_ROOT / path).is_file(), path
    for path in host:
        assert (
            not path.startswith(
                tuple(f"inference/core/interfaces/{tree}/" for tree in TREES)
            )
            or path in FACADE_SOURCE_PATHS
        ), path
