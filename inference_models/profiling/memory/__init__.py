"""GPU memory profiling harnesses aligned with ``docs/description.md``."""

from profiling.memory.schema import (
    OnnxMemoryProfileResult,
    TorchMemoryProfileResult,
)

__all__ = ["OnnxMemoryProfileResult", "TorchMemoryProfileResult"]
