"""Exercise Jetson's package-build step on real Ubuntu 24.04 system Python."""

import os
import shutil
import subprocess
import sysconfig
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("allow_system_install", [False, True])
def test_jetson_standalone_build_preserves_system_python_policy(
    tmp_path, allow_system_install
):
    assert Path(sysconfig.get_path("stdlib"), "EXTERNALLY-MANAGED").exists()
    assert shutil.which("uv") is not None
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    for relative in ["Makefile", ".release/pypi/build-requirements.txt"]:
        destination = checkout / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    for project in ["workflows", "stream_vision"]:
        shutil.copytree(
            ROOT / project,
            checkout / project,
            ignore=shutil.ignore_patterns(
                ".venv", "dist", "build", "*.egg-info", "tests"
            ),
        )
    dockerfile = (ROOT / "docker/dockerfiles/Dockerfile.onnx.jetson.7.2.0").read_text()
    command = next(
        line.strip().removesuffix("&& \\").strip()
        for line in dockerfile.splitlines()
        if "make create_workflows_wheel create_streamvision_wheel" in line
    )
    if not allow_system_install:
        command = command.replace("PIP_BREAK_SYSTEM_PACKAGES=1 ", "")

    env = {**os.environ, "UV_CACHE_DIR": str(tmp_path / "uv-cache")}
    env.pop("PIP_BREAK_SYSTEM_PACKAGES", None)
    result = subprocess.run(
        ["sh", "-ec", command + " --old-file=download_fonts"],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
    )
    if allow_system_install:
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(list((checkout / "dist").glob("*.whl"))) == 2
    else:
        assert result.returncode != 0
        assert "externally-managed-environment" in result.stderr
