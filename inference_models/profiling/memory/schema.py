from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from inference_models.models.auto_loaders.entities import BackendType


class ShapeProfile(BaseModel):
    """Input dimensions used for a profiling run."""

    batch_size: int = Field(default=1, ge=1)
    height: int = Field(default=640, ge=1)
    width: int = Field(default=640, ge=1)


class BaseMemoryProfileResult(BaseModel):
    """Common metadata for one memory profiling worker run."""

    model_id: str = Field(
        description="Weights path or logical model id used for this run"
    )
    backend: str = Field(default=BackendType.TORCH.value)
    gpu_name: Optional[str] = None
    quantization: Optional[str] = None
    shape_profile: ShapeProfile
    concurrency: int = Field(default=1, ge=1)

    warmup_iterations: int
    measured_iterations: int
    method_name: str

    shape_signature: str
    module_name: str
    class_name: str

    architecture: Optional[str] = None
    task_type: Optional[str] = None

    notes: List[str] = Field(default_factory=list)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )

    extra: Dict[str, Any] = Field(default_factory=dict)

    def as_json_dict(self) -> Dict[str, Any]:
        json_dict = self.model_dump(mode="json")

        return json_dict


class TorchMemoryProfileResult(BaseMemoryProfileResult):
    """Normalized metrics for one Torch profiling worker run."""

    backend: str = Field(default=BackendType.TORCH.value)

    idle_after_load_allocated_bytes: int
    idle_after_load_reserved_bytes: int

    peak_allocated_bytes: int
    peak_reserved_bytes: int
    end_reserved_bytes: int

    peak_incremental_allocated_bytes: int
    peak_incremental_reserved_bytes: int

    baseline_gpu_free_bytes_nvml: Optional[int] = None

    torch_profiler_memory_enabled: bool = False
    trace_files: List[str] = Field(default_factory=list)


class OnnxMemoryProfileResult(BaseMemoryProfileResult):
    """Normalized metrics for one ONNX Runtime profiling worker run."""

    backend: str = Field(default=BackendType.ONNX.value)

    baseline_process_gpu_bytes_nvml: Optional[int] = None
    idle_after_session_create_bytes: Optional[int] = None
    peak_process_gpu_bytes: Optional[int] = None
    delta_peak_bytes: Optional[int] = None

    execution_providers: List[str] = Field(default_factory=list)
    onnxruntime_version: Optional[str] = None
    trace_files: List[str] = Field(default_factory=list)
    nvml_sampling_interval_seconds: Optional[float] = None
