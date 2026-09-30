"""Requirements pins must track the version of the package they pin."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]

_PIN_CASES = [
    pytest.param(
        _REPO_ROOT / "stream_vision" / "pyproject.toml",
        _REPO_ROOT / "requirements" / "requirements.streamvision.txt",
        "streamvision[webrtc,workflows]==",
        id="streamvision",
    ),
    pytest.param(
        _REPO_ROOT / "workflows" / "pyproject.toml",
        _REPO_ROOT / "requirements" / "requirements.workflows.txt",
        "roboflow-workflows[enterprise]==",
        id="roboflow-workflows",
    ),
]


@pytest.mark.parametrize("pyproject_path, requirements_path, pin_prefix", _PIN_CASES)
def test_requirements_pin_matches_package_version(
    pyproject_path: Path, requirements_path: Path, pin_prefix: str
) -> None:
    with pyproject_path.open("rb") as handle:
        version = tomllib.load(handle)["project"]["version"]

    expected_line = f"{pin_prefix}{version}"
    requirements_lines = requirements_path.read_text().splitlines()

    assert expected_line in requirements_lines
