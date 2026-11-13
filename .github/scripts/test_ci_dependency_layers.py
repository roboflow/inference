"""Check real package metadata used by the cached CI dependency layer."""

import email
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _metadata_source(tmp_path, *, family):
    name = "google.colab.gpu" if family == "colab" else "onnx.gpu.parallel"
    dockerfile = (ROOT / f"docker/dockerfiles/Dockerfile.{name}").read_text()
    stage = dockerfile.split(" AS ci-dependencies\n", 1)[1].split("\nFROM ", 1)[0]
    target = tmp_path / "metadata-source"
    target.mkdir()
    for line in stage.splitlines():
        if not line.startswith("COPY "):
            continue

        *sources, destination = shlex.split(line)[1:]
        relative = Path(destination).relative_to("/ci-dependencies")
        for source in sources:
            source = ROOT / source
            destination_path = target / relative
            if source.is_dir():
                shutil.copytree(source, destination_path, dirs_exist_ok=True)
            else:
                if destination.endswith("/"):
                    destination_path /= source.name

                destination_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination_path)

    (target / "inference_sdk").mkdir(exist_ok=True)
    (target / "inference_cli").mkdir(exist_ok=True)
    return target


def _metadata(path, *, setup=None):
    output = path / "distmeta"
    output.mkdir()
    command = (
        [sys.executable, setup, "dist_info", "--output-dir", str(output)]
        if setup
        else [
            sys.executable,
            "-c",
            "from setuptools.build_meta import prepare_metadata_for_build_wheel; "
            "prepare_metadata_for_build_wheel('distmeta')",
        ]
    )
    result = subprocess.run(command, cwd=path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    files = list(output.glob("*.dist-info/METADATA"))
    assert len(files) == 1
    metadata = email.message_from_string(files[0].read_text())
    return (
        metadata["Name"],
        metadata["Version"],
        sorted(metadata.get_all("Requires-Dist", [])),
        sorted(metadata.get_all("Provides-Extra", [])),
    )


@pytest.mark.parametrize("project", ["workflows", "stream_vision", "inference_models"])
def test_cached_project_metadata_matches_real_checkout(tmp_path, project):
    metadata_source = _metadata_source(tmp_path, family="colab")
    actual = tmp_path / "actual"
    shutil.copytree(
        ROOT / project,
        actual,
        ignore=shutil.ignore_patterns(".venv", "*.egg-info", "build", "dist", "__pycache__"),
    )
    assert _metadata(metadata_source / project) == _metadata(actual)


@pytest.mark.parametrize("package", ["core", "gpu", "sdk", "cli"])
def test_cached_legacy_metadata_matches_real_checkout(tmp_path, package):
    metadata_source = _metadata_source(tmp_path, family="parallel")
    actual = tmp_path / "actual"
    shutil.copytree(metadata_source, actual)
    for directory in ["inference", "inference_sdk", "inference_cli"]:
        shutil.copytree(
            ROOT / directory,
            actual / directory,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "node_modules"),
        )
    setup = f".release/pypi/inference.{package}.setup.py"
    assert _metadata(metadata_source, setup=setup) == _metadata(actual, setup=setup)


def test_metadata_layer_does_not_contain_application_source(tmp_path):
    for family in ["colab", "parallel"]:
        directory = tmp_path / family
        directory.mkdir()
        source = _metadata_source(directory, family=family)
        python_files = sorted(
            p.relative_to(source).as_posix() for p in source.glob("inference/**/*.py")
        )
        assert python_files == ["inference/core/version.py"]
        assert not list(source.glob("workflows/roboflow_workflows/**/*.py"))
        assert not list(source.glob("stream_vision/streamvision/**/*.py"))


@pytest.mark.parametrize("family", ["colab", "parallel"])
def test_priming_rejects_a_full_source_checkout_before_installing(tmp_path, family):
    (tmp_path / "inference").mkdir()
    (tmp_path / "inference/__init__.py").touch()
    result = subprocess.run(
        ["bash", str(ROOT / "docker/scripts/prepare_ci_dependencies.sh"), family],
        cwd=tmp_path,
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not (tmp_path / "dist").exists()

