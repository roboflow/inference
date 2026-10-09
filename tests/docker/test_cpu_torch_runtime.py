"""Opt-in checks run inside a built CPU image with CPU_IMAGE_TEST=true."""

import os
from importlib.metadata import distributions, version

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("CPU_IMAGE_TEST", "false").lower() != "true",
    reason="Run inside the CPU image with CPU_IMAGE_TEST=true.",
)


def test_torch_has_no_cuda_runtime():
    """Require the CPU wheel even on a host where CUDA is unavailable."""
    import torch

    assert torch.version.cuda is None
    assert torch.__version__.endswith("+cpu")
    assert version("torchvision").endswith("+cpu")


def test_cuda_distributions_are_absent():
    """Retain NVML without the CUDA libraries and Triton pulled by CUDA torch."""
    names = {distribution.metadata["Name"].lower() for distribution in distributions()}
    assert "nvidia-ml-py" in names
    assert "triton" not in names
    assert {name for name in names if name.startswith("nvidia-")} == {"nvidia-ml-py"}


def test_torchvision_cpu_nms():
    """Exercise the native torchvision operator used in detection postprocessing."""
    import torch
    from torchvision.ops import nms

    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0]])
    scores = torch.tensor([0.9, 0.8])
    assert nms(boxes, scores, 0.5).tolist() == [0]
