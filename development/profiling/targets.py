from __future__ import annotations

from typing import Any

import torch

from development.profiling.data.base import DataRecord
from development.profiling.nvtx import profiling_range_if_cuda


class SmokeTensorTarget:
    """Small deterministic tensor target for profiling smoke tests."""

    name = "smoke-tensor"

    def prepare(self, record: DataRecord, *, device: torch.device) -> torch.Tensor:
        """Prepare a dummy tensor record.

        Args:
            record (DataRecord): Input data record.
            device (torch.device): Device selected for this profiling run.

        Returns:
            Tensor moved to the selected device.
        """
        tensor = record.metadata.get("tensor", record.image)
        if tensor is None:
            tensor = torch.ones((4, 4), dtype=torch.float32)

        prepared = torch.as_tensor(tensor, dtype=torch.float32, device=device)

        return prepared

    def run(self, prepared: torch.Tensor) -> torch.Tensor:
        """Run deterministic tensor operations.

        Args:
            prepared (torch.Tensor): Prepared tensor.

        Returns:
            Reduced tensor output.
        """
        with profiling_range_if_cuda("smoke multiply", tensor=prepared):
            result = prepared * 2.0

        with profiling_range_if_cuda("smoke reduction", tensor=prepared):
            output = result.sum(dim=-1)

        return output

    def validate(self, output: torch.Tensor) -> None:
        """Validate smoke target output shape.

        Args:
            output (torch.Tensor): Output returned by ``run``.

        Raises:
            ValueError: If the output is scalar.
        """
        if output.ndim == 0:
            raise ValueError(
                "Expected smoke target output to have at least one dimension."
            )

    def summarize(self, output: torch.Tensor) -> dict[str, Any]:
        """Summarize smoke target output for the manifest.

        Args:
            output (torch.Tensor): Output returned by ``run``.

        Returns:
            Manifest-safe output summary.
        """
        summary = {
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "sum": float(output.detach().cpu().sum().item()),
        }

        return summary


BUILTIN_TARGETS = {
    SmokeTensorTarget.name: SmokeTensorTarget,
}
