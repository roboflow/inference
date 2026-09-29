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
    assert project["version"] == "0.1.0rc4"
    assert project["requires-python"] == ">=3.10,<3.14"


def test_dependency_names() -> None:
    project = _load_pyproject()["project"]
    dependency_names = {
        dep.split(">=")[0].split("==")[0] for dep in project["dependencies"]
    }
    assert dependency_names == {
        "numpy",
        "opencv-python",
        "supervision",
        "pydantic",
        "psutil",
    }
    assert "roboflow-workflows" not in dependency_names


def test_optional_dependency_extras() -> None:
    extras = _load_pyproject()["project"]["optional-dependencies"]
    assert set(extras) == {"webrtc", "nvdec", "test", "workflows"}
    assert extras["webrtc"] == ["aiortc>=1.9.0", "av==14.2.0"]
    assert extras["workflows"] == ["roboflow-workflows>=0.2.4rc2"]
    assert extras["nvdec"] == [
        "pynvvideocodec>=2.1.0,<3.0.0; (sys_platform == 'linux' and "
        "platform_machine == 'x86_64') or (sys_platform == 'win32' and "
        "platform_machine == 'AMD64')"
    ]
    assert extras["test"] == [
        "pytest>=9.0.3,<10.0.0",
        "requests-mock~=1.12.1",
        "tomli>=2.0.0; python_version < '3.11'",
        "pytest-asyncio<=0.21.1",
        "pytest-timeout>=2.2.0",
    ]


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
