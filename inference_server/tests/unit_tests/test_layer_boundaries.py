from __future__ import annotations

import ast
from pathlib import Path

import pytest


_PKG_ROOT = (
    Path(__file__).resolve().parents[2] / "inference_server"
)


_L3_INTERNAL_MODULES = {
    "inference_server.proxies.mmp_client",
    "inference_server.proxies.mm_wrapper",
}

_FORBIDDEN_FOR_L1 = {
    "zmq",
    "zmq.asyncio",
    "multiprocessing.shared_memory",
    "inference_model_manager.model_manager_process",
} | _L3_INTERNAL_MODULES

_FORBIDDEN_FOR_L2 = {
    "zmq",
    "zmq.asyncio",
    "multiprocessing.shared_memory",
    "inference_model_manager.model_manager_process",
} | _L3_INTERNAL_MODULES

_FORBIDDEN_FOR_L3 = {
    "inference_server.handlers",
    "inference_server.routers",
}


def _iter_py_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def _module_imports(path: Path) -> set[str]:
    src = path.read_text()
    tree = ast.parse(src, filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
    return names


def _violates(imports: set[str], forbidden: set[str]) -> set[str]:
    hits: set[str] = set()
    for imp in imports:
        for bad in forbidden:
            if imp == bad or imp.startswith(bad + "."):
                hits.add(imp)
    return hits


@pytest.mark.parametrize(
    "path", _iter_py_files(_PKG_ROOT / "handlers"), ids=lambda p: str(p)
)
def test_l1_handlers_do_not_import_l3_internals_or_transport(path):
    imports = _module_imports(path)
    bad = _violates(imports, _FORBIDDEN_FOR_L1)
    assert not bad, f"{path} imports forbidden L1 dependency: {sorted(bad)}"


@pytest.mark.parametrize(
    "path", _iter_py_files(_PKG_ROOT / "framework"), ids=lambda p: str(p)
)
def test_l2_framework_does_not_import_l3_internals_or_transport(path):
    imports = _module_imports(path)
    bad = _violates(imports, _FORBIDDEN_FOR_L2)
    assert not bad, f"{path} imports forbidden L2 dependency: {sorted(bad)}"


@pytest.mark.parametrize(
    "path", _iter_py_files(_PKG_ROOT / "proxies"), ids=lambda p: str(p)
)
def test_l3_proxies_do_not_import_l1_or_routers(path):
    imports = _module_imports(path)
    bad = _violates(imports, _FORBIDDEN_FOR_L3)
    assert not bad, f"{path} imports forbidden L3 dependency: {sorted(bad)}"
