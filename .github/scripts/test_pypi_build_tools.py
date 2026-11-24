"""Run first, in a fresh release image, before suites that install build tools."""

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(".release/pypi/build-requirements.txt")


def _pins():
    pins = {}
    for line in (ROOT / MANIFEST).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue

        requirement = Requirement(line)
        specs = list(requirement.specifier)
        assert len(specs) == 1 and specs[0].operator == "=="
        pins[requirement.name] = specs[0].version
    return pins


def _versions(python):
    output = subprocess.check_output(
        [
            str(python),
            "-c",
            "import importlib.metadata as m, json; "
            "print(json.dumps({d.metadata['Name'].lower(): d.version "
            "for d in m.distributions()}))",
        ],
        text=True,
    )
    versions = json.loads(output)
    return {name: versions.get(name) for name in _pins()}


def _environment(tmp_path):
    environment = tmp_path / "caller"
    subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
    python = environment / "bin/python"
    env = {
        **os.environ,
        # Exclude the image's globally installed uv from PATH.
        "PATH": f"{python.parent}:/usr/bin:/bin",
        "PIP_NO_INDEX": "1",
        "PIP_FIND_LINKS": "/build-wheels",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "UV_OFFLINE": "1",
        "UV_NO_INDEX": "1",
        "UV_FIND_LINKS": "/build-wheels",
        "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
        "UV_PYTHON_DOWNLOADS": "never",
        "UV_PYTHON": str(python),
    }
    assert shutil.which("uv", path=env["PATH"]) is None
    return python, env


def test_release_image_starts_with_shared_pins():
    # This file runs in its own process before any Make invocation can repair
    # the image. Compare with source, not the runner's version as the oracle.
    assert (
        Path("/build-requirements.txt").read_bytes() == (ROOT / MANIFEST).read_bytes()
    )
    assert {name: importlib.metadata.version(name) for name in _pins()} == _pins()


@pytest.mark.parametrize("installer", ["backend", "frontend", "models", "sdk"])
def test_installers_select_pins_with_competing_versions(tmp_path, installer):
    python, env = _environment(tmp_path)
    if installer in {"backend", "frontend"}:
        target = (
            "install_pypi_build_tools"
            if installer == "backend"
            else "install_pypi_build_frontend"
        )
        argv = ["make", target, f"PYTHON={python}"]
        cwd = ROOT
    elif installer == "models":
        workflow = yaml.safe_load(
            (ROOT / ".github/workflows/publish.pypi.inference_exp.yml").read_text()
        )
        step = next(
            step
            for step in workflow["jobs"]["build"]["steps"]
            if step.get("name") == "📦 Install dependencies"
        )
        argv = ["sh", "-ec", step["run"]]
        cwd = ROOT / step["working-directory"]
    else:
        dockerfile = (ROOT / ".github/docker/Dockerfile.sdk-tests").read_text()
        assert f"COPY {MANIFEST} /build-requirements.txt" in dockerfile
        command = next(
            line.removeprefix("RUN ")
            for line in dockerfile.splitlines()
            if line.startswith("RUN python -m pip install")
        )
        argv = ["sh", "-ec", command]
        cwd = ROOT

    result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = _pins()
    if installer in {"frontend", "models"}:
        expected = {
            name: version if name == "uv" else None
            for name, version in expected.items()
        }
    assert _versions(python) == expected


@pytest.mark.parametrize(
    "target,installer",
    [
        ("create_workflows_wheel", "frontend"),
        ("create_streamvision_wheel", "frontend"),
        ("create_models_wheel", "frontend"),
        ("create_pypi_wheels", "tools"),
        ("create_pypi_sdk_wheel", "tools"),
        ("create_pypi_runtime_wheels", "tools"),
    ],
)
def test_build_target_runs_shared_installer_before_build(target, installer):
    result = subprocess.run(
        ["make", "--dry-run", target, "PYTHON=selected-python"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    option = "-c" if installer == "frontend" else "-r"
    install = f"selected-python -m pip install {option} {MANIFEST}"
    assert install in result.stdout
    build = "-m uv build" if installer == "frontend" else "bdist_wheel"
    assert result.stdout.index(install) < result.stdout.index(build)


@pytest.mark.parametrize(
    "target,project",
    [
        ("create_workflows_wheel", "workflows"),
        ("create_streamvision_wheel", "stream_vision"),
        ("create_models_wheel", "inference_models"),
    ],
)
def test_standalone_make_bootstraps_uv_and_preserves_caller(tmp_path, target, project):
    python, env = _environment(tmp_path)
    # Model an installed caller with the repository's runtime dependency bounds.
    caller_wheel = tmp_path / "caller_contract-1.0-py3-none-any.whl"
    with zipfile.ZipFile(caller_wheel, "w") as archive:
        prefix = "caller_contract-1.0.dist-info"
        archive.writestr(
            f"{prefix}/METADATA",
            "Metadata-Version: 2.1\nName: caller-contract\nVersion: 1.0\n"
            "Requires-Dist: packaging~=24.0\nRequires-Dist: wheel<=0.45.1\n",
        )
        archive.writestr(f"{prefix}/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-any\n")
        archive.writestr(f"{prefix}/RECORD", "")
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            str(caller_wheel),
            "setuptools==84.2.1",
            "wheel==0.45.1",
            "packaging==24.2",
        ],
        env=env,
        check=True,
        capture_output=True,
    )
    before = _versions(python)
    checkout = tmp_path / "checkout"
    (checkout / MANIFEST).parent.mkdir(parents=True)
    shutil.copyfile(ROOT / MANIFEST, checkout / MANIFEST)
    shutil.copyfile(ROOT / "Makefile", checkout / "Makefile")
    shutil.copytree(
        ROOT / project,
        checkout / project,
        ignore=shutil.ignore_patterns(".venv", "dist", "build", "*.egg-info", "tests"),
    )
    # Font downloads are unrelated to installer isolation and require network.
    result = subprocess.run(
        ["make", "--old-file=download_fonts", target, f"PYTHON={python}"],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert _versions(python) == {**before, "uv": _pins()["uv"]}
    subprocess.run([str(python), "-m", "pip", "check"], env=env, check=True)
    with zipfile.ZipFile(next((checkout / "dist").glob("*.whl"))) as archive:
        metadata = next(
            name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")
        )
        assert (
            f"Generator: setuptools ({_pins()['setuptools']})"
            in archive.read(metadata).decode()
        )
