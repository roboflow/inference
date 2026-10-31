from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from development.profiling.data.base import DataRecord


@dataclass(frozen=True)
class DummyDataSource:
    """Deterministic tensor/image source for smoke profiling runs."""

    record_count: int = 1
    image_shape: tuple[int, int, int] = (64, 64, 3)
    tensor_shape: tuple[int, ...] = (4, 4)
    include_image: bool = True

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None) -> "DummyDataSource":
        """Build a dummy data source from config values.

        Args:
            config (Mapping[str, Any] | None): Optional source-specific config.

        Returns:
            Configured dummy data source.
        """
        config = config or {}
        data_source = cls(
            record_count=int(config.get("record_count", config.get("limit", 1))),
            image_shape=tuple(config.get("image_shape", (64, 64, 3))),
            tensor_shape=tuple(config.get("tensor_shape", (4, 4))),
            include_image=bool(config.get("include_image", True)),
        )

        return data_source

    def iter_records(self):
        """Iterate over deterministic dummy records.

        Returns:
            Generator of dummy records.
        """
        image_numel = _numel(self.image_shape)
        tensor_numel = _numel(self.tensor_shape)

        for index in range(self.record_count):
            image = None
            if self.include_image:
                image = (
                    torch.arange(image_numel, dtype=torch.float32)
                    .reshape(self.image_shape)
                    .add(index)
                )

            tensor = (
                torch.arange(tensor_numel, dtype=torch.float32)
                .reshape(self.tensor_shape)
                .add(index)
            )
            yield DataRecord(
                id=f"dummy-{index}",
                image=image,
                metadata={
                    "tensor": tensor,
                    "scores": torch.linspace(0.0, 1.0, steps=self.tensor_shape[0]),
                    "boxes": torch.arange(
                        max(self.tensor_shape[0], 1) * 4,
                        dtype=torch.float32,
                    ).reshape(max(self.tensor_shape[0], 1), 4),
                },
                source={
                    "type": "dummy",
                    "record_count": self.record_count,
                    "index": index,
                },
            )

    def describe(self) -> Mapping[str, Any]:
        """Describe this dummy data source.

        Returns:
            Manifest metadata for the dummy source.
        """
        description = {
            "type": "dummy",
            "record_count": self.record_count,
            "image_shape": list(self.image_shape),
            "tensor_shape": list(self.tensor_shape),
            "include_image": self.include_image,
        }

        return description


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result
