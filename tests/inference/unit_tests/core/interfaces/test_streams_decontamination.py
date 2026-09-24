"""WP-A00 decontamination ratchet for camera/stream/stream_manager.

This is NOT a "zero forbidden imports" gate: at G0 (baseline SHA below) these
three trees are still fully contaminated with `inference.core`/`inference.models`
imports (see EXTRACT_INFERENCE_PIPELINE_AND_MANAGER_PLAN.MD, WP-A01..A05). The
frozen ``_ALLOWLIST`` below is an exact snapshot of the current violations, not
a target. `test_forbidden_imports_match_frozen_allowlist` fails on either a
*new* forbidden import (regression) or a *stale* allowlist entry (an import
that was actually removed, meaning the allowlist must shrink) - both directions
matter so later work packages can shrink this list without it silently going
out of date.

Precedent: workflows/tests/unit_tests/test_decontamination_lint.py:1 (single
tree, single allowed prefix, already-zero target). This generalizes the same
AST + relative-import + quoted-import scan to three trees with a non-empty
frozen allowlist, plus one extra rule: the four modules that WP-A02 will turn
into legacy facades (D:core/interfaces/legacy_stream in the plan) are forbidden
import sources for every *other* host-neutral module in these trees, even
though their dotted name lies under an otherwise-allowed prefix.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pytest

BASELINE_SHA = "65ad2beaaca0825bffc2fbbe99199d3a40994324"

PROJECT_ROOT = Path(__file__).resolve().parents[5]
INTERFACES_ROOT = PROJECT_ROOT / "inference" / "core" / "interfaces"
TREES = ("camera", "stream", "stream_manager")
for _tree in TREES:
    assert (INTERFACES_ROOT / _tree).is_dir(), f"missing tree: {_tree}"

ALLOWED_PREFIXES = tuple(f"inference.core.interfaces.{tree}" for tree in TREES)

# The four modules WP-A02 extracts into core/interfaces/legacy_stream, leaving
# these as thin facades (plan §"D. Other retained modules" and the B02 `git rm`
# list). Since A02 they are exact aliases of host-owned implementations, so
# no host-neutral module may import them (WP-A03 removed the manager's last
# one: its pipelines now get the workflow from an injected host).
FACADE_MODULES = frozenset(
    {
        "inference.core.interfaces.stream.inference_pipeline",
        "inference.core.interfaces.stream.stream",
        "inference.core.interfaces.stream.model_handlers.roboflow_models",
        "inference.core.interfaces.stream.model_handlers.yolo_world",
    }
)

# WP-A02: each facade file is now an exact alias of its host-owned
# implementation under core/interfaces/legacy_stream (it replaces itself in
# sys.modules). The four files are therefore host-owned code, excluded from
# the host-neutral scan by exact path - never by directory - and checked
# instead by test_facade_files_only_alias_their_exact_legacy_target.
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
    # Distinguishes `from pkg import submodule` (submodule is itself a module,
    # e.g. the FACADE_MODULES case) from `from pkg import some_name` (some_name
    # is just an attribute/class/function inside pkg, not a module at all).
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
    # Exact boundary match: "inference.core.interfaces.streamx" must not be
    # treated as living under the "inference.core.interfaces.stream" prefix.
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


# Started as the exact snapshot at BASELINE_SHA (`git show <sha>:<path>` for
# every entry), regenerated with the same ast.walk/_resolve_relative logic
# above, not typed by hand. Update only alongside a real A01+ import removal
# or addition; a diff here that isn't backed by a corresponding source change
# is a bug in this test, not license to widen the allowlist.
#
# WP-A02 removed every entry of the four facade files (now host-owned
# aliases, see FACADE_SOURCE_PATHS), sinks.py's active-learning import (the
# middleware is injected structurally) and the manager's two imports of the
# inference_pipeline facade (they only needed a core status constant, now
# imported from stream/pipeline.py). The five stream/pipeline.py entries are
# the host-neutral pipeline's share of the old inference_pipeline.py edges,
# relocated rather than added: exceptions and the Workflows modules are
# WP-A04's, the session module WP-A05's.
#
# WP-A03 removed inference_pipeline_manager.py's import of the
# inference_pipeline facade: the manager builds the host-neutral pipeline from
# what its injected host prepares.
#
# WP-A04 removed every utility, exception, HTTP serializer and
# `inference.core.workflows` edge: the helpers are stream-owned copies under
# stream/support, the exceptions and experimental warning are defined in
# stream/{exceptions,warnings}.py (or the Workflows platform-error module) and
# aliased by `inference.core.{exceptions,warnings}`, result serialisation is
# manager_app/result_serialization.py, and Workflows is imported by its
# canonical `roboflow_workflows` name. Only the session module (WP-A05's)
# remained.
#
# WP-A05 moved the session ContextVar into stream/session.py (the historical
# usage_tracking module re-exports it): movable files have no production
# dependency left. The allowlist is empty and must stay empty - the four
# retained facades are host code with their own manifest (FACADE_TARGETS and
# the A-end manifest), never an allowlisted dependency of movable files.
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
    # `from ...roboflow_api import x` inside camera/foo.py climbs to
    # `inference.core.roboflow_api`, exactly as contaminating as the
    # absolute form.
    fake_path = INTERFACES_ROOT / "camera" / "_fixture_relative_import.py"
    source = "from ...roboflow_api import get_roboflow_model_data\n"
    tree = _parse_fixture(source, fake_path)
    modules = _imported_modules(tree, fake_path)
    assert "inference.core.roboflow_api" in modules
    assert _is_forbidden("inference.core.roboflow_api")


def test_checker_catches_quoted_string_import() -> None:
    # Mirrors the real precedent this regex exists for: a source string later
    # exec()'d into an assembled code block, invisible to ast.parse on this
    # file (workflows/tests/unit_tests/test_decontamination_lint.py:29-34).
    source = (
        "TEMPLATE = "
        '"from inference.core.roboflow_api import get_roboflow_model_data"\n'
    )
    found = _STRING_IMPORT.findall(source)
    assert found == ["inference.core.roboflow_api"]
    assert _is_forbidden(found[0])


def test_checker_resolves_submodule_import_but_not_attribute_import() -> None:
    # `from inference.core import logger` imports the *submodule*
    # inference/core/logger.py - it must resolve to "inference.core.logger",
    # not just the parent "inference.core" (the bug this guards: alias
    # children of an ImportFrom were never checked against the filesystem).
    fake_path = INTERFACES_ROOT / "camera" / "_fixture_submodule_import.py"
    submodule_source = "from inference.core import logger\n"
    modules = _imported_modules(_parse_fixture(submodule_source, fake_path), fake_path)
    assert "inference.core.logger" in modules
    assert "inference.core" in modules

    # `from inference.core.exceptions import EngineExecutionError` imports an
    # attribute (a class), not a module - "inference.core.exceptions.
    # EngineExecutionError" does not exist on disk and must not be recorded.
    attribute_source = "from inference.core.exceptions import EngineExecutionError\n"
    modules = _imported_modules(_parse_fixture(attribute_source, fake_path), fake_path)
    assert "inference.core.exceptions" in modules
    assert "inference.core.exceptions.EngineExecutionError" not in modules


def test_module_exists_on_disk_matches_checker_resolution() -> None:
    assert _module_exists_on_disk("inference.core.logger")
    assert not _module_exists_on_disk("inference.core.logger.NotAModule")
    assert not _module_exists_on_disk("")


def test_checker_boundary_matches_allowed_prefixes_exactly() -> None:
    # A hypothetical sibling tree "streamx" must not be treated as living
    # under the "stream" prefix just because it shares that string prefix
    # (the bug this guards: a plain str.startswith check with no path
    # boundary would incorrectly allow this).
    assert _is_forbidden("inference.core.interfaces.streamx.entities")
    assert _is_forbidden("inference.core.interfaces.streamx")
    # The exact prefix itself, and genuine children of it, stay allowed.
    assert not _is_forbidden("inference.core.interfaces.stream")
    assert not _is_forbidden("inference.core.interfaces.stream.entities")


def test_checker_does_not_flag_inference_models_or_inference_sdk() -> None:
    # `inference_models`/`inference_sdk` are separate distributions; the
    # startswith("inference.") check must not treat them as `inference.*`.
    assert not _is_forbidden("inference_models.models.base")
    assert not _is_forbidden("inference_sdk.http.client")


def test_checker_allows_same_tree_non_facade_imports() -> None:
    assert not _is_forbidden("inference.core.interfaces.camera.entities")
    assert not _is_forbidden("inference.core.interfaces.stream.entities")
    assert not _is_forbidden("inference.core.interfaces.stream_manager.api.entities")


def test_facade_modules_are_forbidden_even_under_an_allowed_prefix() -> None:
    # The four modules land under `inference.core.interfaces.stream`, an
    # otherwise-allowed prefix, but WP-A02 turns them into host-only facades;
    # importing them from another host-neutral module must stay forbidden.
    for module in FACADE_MODULES:
        assert module.startswith(ALLOWED_PREFIXES)
        assert _is_forbidden(module)


def test_facade_modules_are_exactly_four() -> None:
    assert len(FACADE_MODULES) == 4


def test_facade_source_paths_are_exactly_the_four_facade_files() -> None:
    # The exclusion from the host-neutral scan is by exact file, never by
    # directory: model_handlers/workflows.py sits next to two facades and is
    # still scanned (see its allowlist entries).
    assert FACADE_SOURCE_PATHS == {
        "inference/core/interfaces/stream/inference_pipeline.py",
        "inference/core/interfaces/stream/stream.py",
        "inference/core/interfaces/stream/model_handlers/roboflow_models.py",
        "inference/core/interfaces/stream/model_handlers/yolo_world.py",
    }
    for path in FACADE_SOURCE_PATHS:
        assert (PROJECT_ROOT / path).is_file(), path


@pytest.mark.parametrize("facade_module", sorted(FACADE_MODULES))
def test_facade_files_only_alias_their_exact_legacy_target(facade_module: str) -> None:
    # A facade excluded from the scan must not smuggle anything else in: it
    # may import only `sys` and its own legacy target, which it installs as
    # its sys.modules entry.
    path = PROJECT_ROOT / _source_path(facade_module)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    target = FACADE_TARGETS[facade_module]
    target_package = target.rsplit(".", 1)[0]

    assert _imported_modules(tree, path) == {"sys", target_package, target}
    assert "sys.modules[__name__] = _implementation" in ast.unparse(tree)


@pytest.mark.parametrize("facade_module", sorted(FACADE_MODULES))
def test_facade_name_resolves_to_the_legacy_module_object(facade_module: str) -> None:
    import importlib

    facade = importlib.import_module(facade_module)
    target = importlib.import_module(FACADE_TARGETS[facade_module])
    parent_name, _, child_name = facade_module.rpartition(".")

    assert facade is target
    assert getattr(importlib.import_module(parent_name), child_name) is target


# WP-A03: the stream manager's `inference` pipeline host. New in A03, so it
# has no historical name and no facade.
LEGACY_HOST_MODULE = f"{LEGACY_PREFIX}.host"


def test_legacy_stream_holds_exactly_the_four_facade_targets() -> None:
    # The host-owned side of the facades: nothing else may grow under
    # legacy_stream without being named here (and, until Stage B, behind a
    # facade of its own).
    legacy_root = INTERFACES_ROOT / "legacy_stream"
    modules = {
        ".".join(path.relative_to(PROJECT_ROOT).with_suffix("").parts)
        for path in legacy_root.rglob("*.py")
        if path.name != "__init__.py"
    }
    assert modules == set(FACADE_TARGETS.values()) | {LEGACY_HOST_MODULE}


def test_host_neutral_trees_never_import_legacy_stream() -> None:
    # Core imports back into the host-owned implementations are forbidden,
    # directly or through the facades (covered by FACADE_MODULES above).
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
    # Every module is expected to move from the legacy "inference.core.interfaces."
    # namespace to "roboflow_streams." by prefix substitution, EXCEPT the 4
    # facade modules, which stay as host-only legacy facades and therefore have
    # no canonical name (None). A per-entry check (not just a count) guards
    # against a manifest that drifts on *which* modules got which name.
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
    # If a module declares __all__, that list *is* its public surface (the
    # convention `from module import *` and every other export tool honors),
    # overriding rather than adding to the underscore-based heuristic below -
    # see gstreamer_rtsp_producer.py, which imports several non-underscore
    # names it does not intend to re-export alongside its narrower __all__.
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
            # Re-exported imports (absolute or relative, aliased or not) are
            # part of a module's importable public surface too - e.g. sinks.py
            # re-exposes VideoFrame/SinkHandler this way with no local def.
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
    # "module exports" freeze: each manifest entry's public_top_level_names is
    # supposed to be the exact set of non-underscore top-level def/class/assign
    # names in that module. A per-module check (not just counts) guards against
    # the manifest drifting on *which* names a module exports - the same class
    # of gap as the canonical-name mapping check above.
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
    # The regression this guards: a module that only re-exports a name via
    # import (no local def/class/assign), e.g. sinks.py's `VideoFrame`/
    # `SinkHandler`, must still show up as part of its importable public
    # surface, exactly like the real fixture below.
    source = (
        "from .entities import VideoFrame\n"
        "from ..camera.entities import SinkHandler as _SinkHandlerAlias\n"
        "import inference.core.logger\n"
        "import numpy as np\n"
        "from . import _private_helper\n"
        "from .other import public_name, _hidden_name\n"
        "def local_public(): ...\n"
    )
    # `_SinkHandlerAlias` and `_hidden_name` stay excluded - the leading
    # underscore convention applies identically to aliases and plain names.
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
    # Focused regression for the exact gap being fixed: the real sinks.py
    # manifest entry must list VideoFrame/SinkHandler, not just its locally
    # defined names.
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


# WP-A05: the direct scan above sees only each file's own imports. The checks
# below close the import graph, so that no allowed-looking module - a facade,
# a package `__init__`, a dynamic import or a host-installed loader hook - can
# bring host modules back into what moves.
TESTS_ROOT = PROJECT_ROOT / "tests" / "inference" / "unit_tests" / "core" / "interfaces"
A_END_MANIFEST_PATH = (
    PROJECT_ROOT / "tests" / "inference" / "unit_tests" / "streams_a_end_manifest.json"
)

# The one historical export a host loader hook binds onto a movable module
# (inference/_workflows_compat.py `_HOST_EXPORTS`); it runs only in the
# `inference` host, never as an import made by the movable module itself.
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
    # Catches `importlib.import_module("inference.core....")`-style imports
    # the AST scan cannot see; the stream manager's host factory arrives as
    # a descriptor from the caller, never as a literal here.
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


# Imports the given modules with `inference`, `inference.core` and
# `inference.core.interfaces` replaced by empty packages - the host roots the
# moved package will not have - and reports every other `inference.*` module
# outside the three trees that got loaded, with the frames that loaded it.
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
    # Positive control: a facade is an allowed-looking name that resolves to
    # host code; the probe must report what it pulls in.
    facade = "inference.core.interfaces.stream.model_handlers.roboflow_models"

    result = _import_with_stubbed_host_roots([facade])

    assert "inference.core.interfaces.legacy_stream" in result["loaded"]


def _a_end_manifest() -> dict:
    import json

    return json.loads(A_END_MANIFEST_PATH.read_text())


def test_a_end_manifest_records_a_real_base_commit_and_no_candidate() -> None:
    import subprocess

    manifest = _a_end_manifest()
    subprocess.run(
        ["git", "cat-file", "-e", f"{manifest['base_head_sha']}^{{commit}}"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    assert manifest["working_tree_dirty"] is True
    assert "candidate_sha" not in manifest


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
