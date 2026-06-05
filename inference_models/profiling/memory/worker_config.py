"""Typed worker payload for memory profiling subprocess harnesses."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from profiling.memory.metadata import ProfileTier, coerce_profile_tier


class MemoryProfilingWorkerPayload(BaseModel):
    """Configuration sent from the CLI to a profiling worker subprocess.

    The CLI builds a plain ``dict`` for pickling/JSON; workers coerce it with
    ``from_payload()`` for validation and documented fields.
    """

    model_config = ConfigDict(extra="ignore")

    module_name: str = Field(
        description="Registry model module resolved from architecture, task, and backend.",
    )
    class_name: str = Field(
        description="Registry model class resolved from architecture, task, and backend.",
    )
    package_path: Path = Field(
        description=(
            "Local model package directory passed to registry from_pretrained. "
            "Profiling CLI with --model-id resolves via resolve_package_directory "
            "before the worker starts; --model-path supplies this directly."
        ),
    )
    device_str: str = Field(
        default="cuda:0",
        description="Torch device string; workers require a CUDA device.",
    )
    method_name: str = Field(
        default="infer",
        description="Model method invoked with synthetic images during profiling.",
    )

    batch_size: int = Field(
        ge=1,
        description="Profiling batch size; must match static package constraints.",
    )
    height: int = Field(
        ge=1,
        description="Synthetic image height; must match static package constraints.",
    )
    width: int = Field(
        ge=1,
        description="Synthetic image width; must match static package constraints.",
    )

    warmup_iterations: int = Field(
        ge=0,
        description="Warmup infer iterations excluded from measured peak counters.",
    )
    measured_iterations: int = Field(
        ge=1,
        description="Measured infer iterations used for peak memory capture.",
    )

    from_pretrained_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra kwargs forwarded to from_pretrained in the worker.",
    )
    infer_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra kwargs merged into the profiling infer/prompt call.",
    )
    metadata_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Precomputed metadata sections assembled by the CLI.",
    )

    task_type: Optional[str] = Field(
        default=None,
        description="Registry task type label stored on the profile record.",
    )
    architecture: Optional[str] = Field(
        default=None,
        description="Registry architecture label stored on the profile record.",
    )
    backend: Optional[str] = Field(
        default=None,
        description="Profiling harness backend: torch, onnx, or trt.",
    )
    quantization: Optional[str] = Field(
        default=None,
        description="Package quantization label stored on the profile record.",
    )
    profile_tier: Optional[str] = Field(
        default=None,
        description=(
            "Profile tier label; falls back to metadata_context.profile_tier. "
            "Must be a valid ProfileTier value when resolved."
        ),
    )

    torch_profiler_memory: bool = Field(
        default=False,
        description="Torch harness only: enable torch.profiler profile_memory traces.",
    )
    in_process: bool = Field(
        default=False,
        description="True when the harness ran in the parent process (debug only).",
    )
    num_execution_contexts: int = Field(
        default=1,
        ge=1,
        description="Concurrent execution contexts recorded in profiling_run metadata.",
    )

    onnx_nvml_sampling_interval_seconds: float = Field(
        default=0.01,
        gt=0,
        description="ONNX harness NVML polling interval in seconds.",
    )
    trt_nvml_sampling_interval_seconds: float = Field(
        default=0.01,
        gt=0,
        description="TensorRT harness NVML polling interval in seconds.",
    )

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> MemoryProfilingWorkerPayload:
        """Validate and coerce a subprocess worker payload dict."""
        return cls.model_validate(payload)

    @property
    def profiling_shape(self) -> Tuple[int, int, int]:
        return self.batch_size, self.height, self.width

    @property
    def resolved_profile_tier(self) -> ProfileTier:
        raw_tier = self.profile_tier or self.metadata_context.get("profile_tier")

        return coerce_profile_tier(raw_tier)

    @property
    def onnx_nvml_sampling_interval(self) -> float:
        return self.onnx_nvml_sampling_interval_seconds

    @property
    def trt_nvml_sampling_interval(self) -> float:
        return (
            self.trt_nvml_sampling_interval_seconds
            or self.onnx_nvml_sampling_interval_seconds
        )
