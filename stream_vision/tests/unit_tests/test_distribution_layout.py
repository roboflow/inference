"""File-inspection tests for the `streamvision` distribution layout.

No `inference` import: this package must be installable and importable on
its own, without the Inference server checkout on `sys.path`.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import pytest
from packaging.requirements import Requirement
from setuptools import find_packages

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT_PATH = _PACKAGE_ROOT / "pyproject.toml"
_INIT_PATH = _PACKAGE_ROOT / "streamvision" / "__init__.py"


def _load_pyproject() -> dict:
    with _PYPROJECT_PATH.open("rb") as handle:
        pyproject = tomllib.load(handle)

    return pyproject


def test_project_metadata() -> None:
    project = _load_pyproject()["project"]
    assert project["name"] == "streamvision"
    assert project["requires-python"] == ">=3.10,<3.14"


def test_base_dependencies_keep_gpu_and_workflow_support_optional() -> None:
    dependencies = {
        Requirement(value).name: Requirement(value)
        for value in _load_pyproject()["project"]["dependencies"]
    }
    assert {
        "numpy",
        "opencv-python",
        "pillow",
        "supervision",
        "pydantic",
        "psutil",
    } <= dependencies.keys()
    assert dependencies["pillow"].marker is None
    assert (
        not {"torch", "roboflow-workflows", "aiortc", "av", "pynvvideocodec"}
        & dependencies.keys()
    )


def test_workflows_and_webrtc_extras_declare_their_runtime_dependencies() -> None:
    extras = _load_pyproject()["project"]["optional-dependencies"]
    for extra, required in {
        "workflows": {"roboflow-workflows", "torch"},
        "webrtc": {"aiortc", "av", "orjson"},
    }.items():
        unconditional = {
            dependency.name
            for value in extras[extra]
            if (dependency := Requirement(value)).marker is None
        }
        assert required <= unconditional


@pytest.mark.parametrize(
    "platform, machine, supported",
    [
        ("linux", "x86_64", True),
        ("win32", "AMD64", True),
        ("darwin", "arm64", False),
        ("darwin", "x86_64", False),
        ("linux", "aarch64", False),
        ("win32", "ARM64", False),
    ],
)
def test_nvdec_declares_torch_and_codec_on_supported_platforms(
    platform, machine, supported
) -> None:
    dependencies = {
        Requirement(value).name: Requirement(value)
        for value in _load_pyproject()["project"]["optional-dependencies"]["nvdec"]
    }
    for name in ("torch", "pynvvideocodec"):
        marker = dependencies[name].marker
        assert marker is not None
        assert (
            marker.evaluate({"sys_platform": platform, "platform_machine": machine})
            is supported
        )


def test_find_packages_discovers_only_streamvision() -> None:
    config = _load_pyproject()["tool"]["setuptools"]["packages"]["find"]
    include = tuple(config.get("include", ("*",)))
    exclude = tuple(config.get("exclude", ()))
    where_dirs = config.get("where", ["."])

    discovered: set[str] = set()
    for where in where_dirs:
        discovered.update(
            find_packages(
                where=str(_PACKAGE_ROOT / where), include=include, exclude=exclude
            )
        )

    assert discovered, "find_packages discovered nothing"
    assert all(
        name == "streamvision" or name.startswith("streamvision.")
        for name in discovered
    )


def test_root_init_has_no_import_statements() -> None:
    tree = ast.parse(_INIT_PATH.read_text())
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert imports == []


def test_import_succeeds_standalone_without_inference() -> None:
    # Meta-path blocker raises on any `inference` import; PYTHONPATH has only the pkg.
    blocker = (
        "import sys\n"
        "class _Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'inference' or name.startswith('inference.'):\n"
        "            raise ImportError(f'blocked: {name}')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Blocker())\n"
        "import streamvision\n"
        "print('ok')\n"
    )
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(_PACKAGE_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", blocker],
        cwd=str(_PACKAGE_ROOT),
        env=child_env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip() == "ok"
