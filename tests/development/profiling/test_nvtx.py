import torch

from development.profiling.nvtx import profiling_range_if_cuda


def test_profiling_range_if_cuda_is_noop_for_cpu_tensor():
    tensor = torch.ones((2, 2))

    with profiling_range_if_cuda("cpu range", tensor=tensor):
        result = tensor + 1

    assert result.sum().item() == 8


def test_profiling_range_if_cuda_is_noop_when_disabled():
    with profiling_range_if_cuda("disabled", enabled=False):
        result = "ok"

    assert result == "ok"
