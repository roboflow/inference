"""Keep CPU image dependencies on CPU wheels without flattening layers."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILES = [
    ROOT / f"docker/dockerfiles/Dockerfile.onnx.{suffix}"
    for suffix in ("cpu", "cpu.dev", "cpu.slim", "cpu.parallel")
]


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda path: path.name)
def test_cpu_torch_install_precedes_dependency_resolution(dockerfile):
    """Install pinned CPU wheels before resolving other requirements.

    Args:
        dockerfile (Path): CPU image recipe to check.
    """
    contents = dockerfile.read_text()
    install = (
        "pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "
        "-r /requirements.torch-cpu.txt"
    )
    assert install in contents
    assert contents.index(install) < contents.index("-r _requirements.txt")
    assert (
        "COPY requirements/requirements.torch-cpu.txt /requirements.torch-cpu.txt"
        in contents
    )
    assert "ARG PIP_CONSTRAINT=/requirements.torch-cpu.txt" in contents


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda path: path.name)
def test_cpu_images_preserve_dependency_layers(dockerfile):
    """Retain base layers so Docker can share and pull them independently.

    Args:
        dockerfile (Path): CPU image recipe to check.
    """
    contents = dockerfile.read_text()
    assert "FROM scratch" not in contents
    assert "COPY --from=base / /" not in contents


def test_cpu_torch_constraints_preserve_resolved_versions():
    """Pin the existing image versions and retain the required NVML binding."""
    constraints = ROOT / "requirements/requirements.torch-cpu.txt"
    assert constraints.read_text().splitlines() == [
        "torch==2.14.0+cpu",
        "torchvision==0.29.0+cpu",
    ]
    assert (
        "nvidia-ml-py<13.0.0"
        in (ROOT / "requirements/requirements.cpu.txt").read_text()
    )
