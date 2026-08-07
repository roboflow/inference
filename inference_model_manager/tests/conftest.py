"""Top-level conftest — auto-skip markers."""

import pytest


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("cuda"):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
