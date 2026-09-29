"""Decontamination lint for the `streamvision` package.

Fails on any import of the `inference` server package from `streamvision`:
AST imports (including function-local and relative ones), quoted import
statements, quoted dotted module names, and the runtime import closure. There
is no allowlist. `streamvision`, `roboflow_workflows`, `inference_sdk`,
`inference_models` and third-party packages are allowed.
"""

import ast
import re
from pathlib import Path
from typing import List, Set, Tuple

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PACKAGE_ROOT / "streamvision"
assert SOURCE_ROOT.is_dir(), f"missing package: {SOURCE_ROOT}"

FORBIDDEN_ROOT = "inference"

_STRING_IMPORT = re.compile(
    r"""["'](?:from|import)\s+(inference(?![A-Za-z0-9_])(?:\.[A-Za-z0-9_]+)*)"""
)


def _resolve_relative(module: str, level: int, path: Path) -> str:
    pkg_parts = path.relative_to(PACKAGE_ROOT).with_suffix("").parts[:-1]
    base = pkg_parts[: len(pkg_parts) - (level - 1)] if level > 1 else pkg_parts
    return ".".join([*base, *(module.split(".") if module else [])])


def _module_exists_on_disk(module: str) -> bool:
    # Distinguishes a submodule import from a plain attribute import by disk check.
    if not module:
        return False
    rel = Path(*module.split("."))
    return (PACKAGE_ROOT / rel.with_suffix(".py")).is_file() or (
        PACKAGE_ROOT / rel / "__init__.py"
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
    return module == FORBIDDEN_ROOT or module.startswith(f"{FORBIDDEN_ROOT}.")


def _source_paths() -> List[Path]:
    paths = sorted(SOURCE_ROOT.rglob("*.py"))

    return paths


def collect_violations() -> Set[Tuple[str, str]]:
    violations = set()
    for path in _source_paths():
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for module in _imported_modules(tree, path):
            if _is_forbidden(module):
                violations.add((relative, module))
        for module in _STRING_IMPORT.findall(source):
            if _is_forbidden(module):
                violations.add((relative, f"{module} (quoted)"))
    return violations


def test_package_has_no_forbidden_import() -> None:
    violations = collect_violations()
    assert not violations, "forbidden import(s):\n  " + "\n  ".join(
        f"{p} -> {m}" for p, m in sorted(violations)
    )


def _parse_fixture(source: str, fake_path: Path) -> ast.AST:
    return ast.parse(source, filename=str(fake_path))


def test_checker_catches_function_local_forbidden_import() -> None:
    fake_path = SOURCE_ROOT / "camera" / "_fixture_local_import.py"
    source = (
        "def helper():\n"
        "    from inference.host_module import host_function\n"
        "    return host_function\n"
    )
    tree = _parse_fixture(source, fake_path)
    modules = _imported_modules(tree, fake_path)
    assert "inference.host_module" in modules
    assert _is_forbidden("inference.host_module")


def test_checker_resolves_relative_import_inside_the_package() -> None:
    fake_path = SOURCE_ROOT / "camera" / "_fixture_relative_import.py"
    source = "from ..stream.entities import ModelConfig\n"
    tree = _parse_fixture(source, fake_path)
    modules = _imported_modules(tree, fake_path)
    assert "streamvision.stream.entities" in modules
    assert not _is_forbidden("streamvision.stream.entities")


def test_checker_catches_quoted_string_import() -> None:
    # Mirrors a string later exec()'d into code - invisible to ast.parse on this file.
    source = 'TEMPLATE = "from inference.host_module import host_function"\n'
    found = _STRING_IMPORT.findall(source)
    assert found == ["inference.host_module"]
    assert _is_forbidden(found[0])


def test_checker_resolves_submodule_import_but_not_attribute_import() -> None:
    # `from streamvision.camera import entities` must resolve the submodule too.
    fake_path = SOURCE_ROOT / "camera" / "_fixture_submodule_import.py"
    submodule_source = "from streamvision.camera import entities\n"
    modules = _imported_modules(_parse_fixture(submodule_source, fake_path), fake_path)
    assert "streamvision.camera.entities" in modules
    assert "streamvision.camera" in modules

    # Importing a class (not a module) must not be recorded as a module path.
    attribute_source = "from streamvision.camera.entities import VideoFrame\n"
    modules = _imported_modules(_parse_fixture(attribute_source, fake_path), fake_path)
    assert "streamvision.camera.entities" in modules
    assert "streamvision.camera.entities.VideoFrame" not in modules


def test_module_exists_on_disk_matches_checker_resolution() -> None:
    assert _module_exists_on_disk("streamvision.camera.entities")
    assert not _module_exists_on_disk("streamvision.camera.entities.NotAModule")
    assert not _module_exists_on_disk("")


def test_checker_boundary_matches_the_forbidden_root_exactly() -> None:
    assert _is_forbidden("inference")
    assert _is_forbidden("inference.host_module")
    assert not _is_forbidden("inferencex")
    assert not _is_forbidden("inferencex.host_module")


def test_checker_does_not_flag_inference_models_or_inference_sdk() -> None:
    # inference_models/inference_sdk are separate distributions, not "inference.*".
    assert not _is_forbidden("inference_models.models.base")
    assert not _is_forbidden("inference_sdk.http.client")


def test_checker_allows_package_and_workflows_imports() -> None:
    assert not _is_forbidden("streamvision.camera.entities")
    assert not _is_forbidden("streamvision.stream.entities")
    assert not _is_forbidden("streamvision.stream_manager.api.entities")
    assert not _is_forbidden("roboflow_workflows.execution_engine.core")


_HOST_MODULE_LITERAL = re.compile(
    r"""["'](inference(?:\.[A-Za-z0-9_]+)+)(?::[A-Za-z0-9_.]+)?["']"""
)


def test_package_names_no_host_module_in_string_literals() -> None:
    # Catches importlib.import_module(...) string imports invisible to the AST scan.
    literals = set()
    for path in _source_paths():
        source = path.read_text(encoding="utf-8")
        for module in _HOST_MODULE_LITERAL.findall(source):
            if _is_forbidden(module):
                literals.add((path.relative_to(PACKAGE_ROOT).as_posix(), module))
    assert literals == set()


def test_host_module_literal_check_catches_a_dynamic_import() -> None:
    source = 'importlib.import_module("inference.host_module")'
    assert _HOST_MODULE_LITERAL.findall(source) == ["inference.host_module"]
    assert _is_forbidden(_HOST_MODULE_LITERAL.findall(source)[0])


# Blocks and records every `inference` import while importing the given modules.
_BLOCKED_ROOT_PROBE = """
import importlib, json, sys, traceback
modules = json.loads(sys.argv[1])
loaded = {}


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "inference" or name.startswith("inference."):
            frames = traceback.extract_stack()[:-1]
            loaded.setdefault(name, [f"{f.filename}:{f.lineno}" for f in frames][-6:])
            raise ImportError(f"blocked: {name}")
        return None


sys.meta_path.insert(0, Blocker())
failed = {}
for module in modules:
    try:
        importlib.import_module(module)
    except Exception as error:
        failed[module] = repr(error)
print(json.dumps({"failed": failed, "loaded": loaded}))
"""


def _import_with_blocked_host_root(modules: List[str]) -> dict:
    import json
    import os
    import subprocess
    import sys

    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(
            [str(PACKAGE_ROOT), *filter(None, [os.environ.get("PYTHONPATH")])]
        ),
    }
    completed = subprocess.run(
        [sys.executable, "-c", _BLOCKED_ROOT_PROBE, json.dumps(modules)],
        cwd=PACKAGE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _module_name(path: Path) -> str:
    parts = path.relative_to(PACKAGE_ROOT).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def test_package_import_closure_loads_no_host_module() -> None:
    modules = [_module_name(path) for path in _source_paths()]

    result = _import_with_blocked_host_root(modules)

    assert result["failed"] == {}
    assert result["loaded"] == {}


def test_import_with_blocked_host_root_blocks_inference_itself() -> None:
    # Positive control: proves the blocker actually intercepts an inference import.
    result = _import_with_blocked_host_root(["inference"])

    assert "inference" in result["failed"]
    assert "inference" in result["loaded"]
