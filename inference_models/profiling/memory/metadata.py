"""Profiling result metadata models and assembly helpers."""

from __future__ import annotations

import platform
import uuid
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from pydantic import BaseModel, Field

from inference_models.models.auto_loaders.entities import BackendType
from profiling.memory.package_resolve import (
    extract_package_features,
    extract_trt_package_features,
)
from profiling.memory.package_input_profile import PackageProfilingShapeSpec
from profiling.memory.registry_profiles import resolve_registry_input_context


class ProfileTier(str, Enum):
    """Purpose and trust level of a profiling run for admission storage."""

    CUSTOMER = "customer"
    REGISTRY_TEMPLATE = "registry_template"
    VALIDATION = "validation"


PROFILE_TIER_DESCRIPTIONS = "\n".join(
    [
        "Purpose and trust level of this profiling run "
        "(label only; does not change harness behavior today).",
        (
            f"  {ProfileTier.CUSTOMER.value}: Profile for a specific registered "
            "model and package intended for production GPU admission."
        ),
        (
            f"  {ProfileTier.REGISTRY_TEMPLATE.value}: Standardized registry-driven "
            "baseline (template shapes or covering a registry entry), not tied to "
            "one customer deployment."
        ),
        (
            f"  {ProfileTier.VALIDATION.value}: QA or re-check run (regression after "
            "inference_models or package changes, CI, or manual verification)."
        ),
    ]
)


def profile_tier_field_description() -> str:
    return PROFILE_TIER_DESCRIPTIONS


def coerce_profile_tier(value: Any) -> ProfileTier:
    if isinstance(value, ProfileTier):
        return value
    if value is None:
        raise ValueError(
            "profile_tier is required; "
            f"expected one of: {', '.join(tier.value for tier in ProfileTier)}"
        )
    try:
        return ProfileTier(str(value))
    except ValueError as exc:
        raise ValueError(
            f"Invalid profile_tier {value!r}; "
            f"expected one of: {', '.join(tier.value for tier in ProfileTier)}"
        ) from exc


class InputAxisValue(BaseModel):
    """One resolved input axis used during profiling."""

    value: Optional[Union[int, float, str, bool]] = Field(
        default=None,
        description="Axis value used during the measured profiling iterations.",
        examples=[1, 640, "Describe this image."],
    )
    resolution: Optional[Literal["static", "dynamic"]] = Field(
        default=None,
        description="Whether the axis was fixed at profile time or may vary at inference.",
        examples=["static", "dynamic"],
    )
    source: Optional[str] = Field(
        default=None,
        description="Artifact or policy that supplied the effective value (e.g. inference_config.json path).",
        examples=[
            "inference_config.json#forward_pass.static_batch_size",
            "profiling_harness_default",
        ],
    )


class DeclaredInputSpec(BaseModel):
    """One input from the registry task inference profile (VLM, SAM, vision, etc.)."""

    name: str = Field(
        description="Input name from the task inference profile (e.g. images, prompt).",
        examples=["images", "prompt", "max_new_tokens"],
    )
    kind: str = Field(
        description="Registry input kind (e.g. image_batch, text, method_kwargs).",
        examples=["image_batch", "text", "method_kwargs"],
    )
    memory_impact: Optional[str] = Field(
        default=None,
        description="Registry memory impact level: low, medium, high, or critical.",
        examples=["high", "critical"],
    )
    profiled: bool = Field(
        default=False,
        description="True when this input was exercised by the profiling harness in this run.",
        examples=[True, False],
    )
    axes: Dict[str, InputAxisValue] = Field(
        default_factory=dict,
        description="Resolved axis values for profiled inputs; empty when not yet profiled.",
        examples=[
            {
                "batch": {
                    "value": 1,
                    "resolution": "static",
                    "source": "inference_config.json#forward_pass.static_batch_size",
                },
                "height": {"value": 640, "resolution": "static"},
            }
        ],
    )
    notes: Optional[str] = Field(
        default=None,
        description="Harness note when the input is declared but not profiled in this run.",
        examples=[
            "Declared by registry profile; not exercised in this harness run.",
        ],
    )


class InputMetadata(BaseModel):
    """Inputs exercised during the profiling run and the declared profile contract."""

    task_inference_profile: Optional[str] = Field(
        default=None,
        description="Registry task_inference_profiles key (e.g. vision_infer, vlm_prompt).",
        examples=["vision_infer", "vlm_prompt", "interactive_sam"],
    )
    profiling_method: Optional[str] = Field(
        default=None,
        description="Model method invoked for profiling (e.g. infer, prompt, segment_images).",
        examples=["infer", "prompt", "segment_images"],
    )
    inputs: Dict[str, Dict[str, InputAxisValue]] = Field(
        default_factory=dict,
        description="Profiled inputs keyed by registry input name, then axis name.",
        examples=[
            {
                "images": {
                    "batch": {"value": 1, "resolution": "static"},
                    "height": {"value": 640, "resolution": "static"},
                    "width": {"value": 640, "resolution": "static"},
                    "channels": {
                        "value": 3,
                        "resolution": "static",
                        "source": "profiling_harness_default",
                    },
                }
            }
        ],
    )
    declared_inputs: List[DeclaredInputSpec] = Field(
        default_factory=list,
        description="Full input contract from the registry profile, including unprofiled inputs.",
        examples=[
            [
                {
                    "name": "images",
                    "kind": "image_batch",
                    "memory_impact": "high",
                    "profiled": True,
                    "axes": {},
                },
                {
                    "name": "prompt",
                    "kind": "text",
                    "memory_impact": "high",
                    "profiled": False,
                    "notes": "Declared by registry profile; not exercised in this harness run.",
                },
            ]
        ],
    )
    infer_defaults: Dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments passed to the profiling method (memory-relevant kwargs).",
        examples=[{"max_new_tokens": 128}, {}],
    )


class RegisteredModelMetadata(BaseModel):
    """Roboflow registered model identity (one model, many backend packages)."""

    model_id: Optional[str] = Field(
        default=None,
        description="Registered model identifier (architecture, task, variant, training dataset).",
        examples=["my-workspace/yolov8-custom"],
    )
    architecture: Optional[str] = Field(
        default=None,
        description="Model architecture label (e.g. yolov8, florence-2).",
        examples=["yolov8", "florence-2", "sam2"],
    )
    task_type: Optional[str] = Field(
        default=None,
        description="Registry task type (e.g. object-detection, vlm).",
        examples=["object-detection", "vlm", "instance-segmentation"],
    )
    model_variant: Optional[str] = Field(
        default=None,
        description="Architecture variant slug (e.g. yolov8-n).",
        examples=["yolov8-n", "yolov8-s"],
    )
    registry_features: Dict[str, Any] = Field(
        default_factory=dict,
        description="Required and supported model features from REGISTERED_MODELS.",
        examples=[
            {
                "required_model_features": ["nms_fused"],
                "supported_model_features": ["nms_fused"],
            }
        ],
    )


class PackageMetadata(BaseModel):
    """One backend-prepared package artifact that was profiled."""

    package_id: Optional[str] = Field(
        default=None,
        description="Unique package identifier from model registry storage (API field packageId).",
        examples=["341349e5c9696c13be9f80d3dd53e463"],
    )
    package_path: Optional[str] = Field(
        default=None,
        description="Local filesystem path to the downloaded or user-supplied package directory.",
        examples=["/tmp/inference_model_packages/object-detection/yolov8/onnx/"],
    )
    backend: str = Field(
        description="Package backend: torch, onnx, trt, hugging-face, or torch-script.",
        examples=["onnx", "torch", "trt", "hugging-face"],
    )
    quantization: Optional[str] = Field(
        default=None,
        description="Package quantization (e.g. fp32, fp16, int8).",
        examples=["fp32", "fp16", "int8"],
    )
    package_features: Dict[str, Any] = Field(
        default_factory=dict,
        description="Memory-relevant features parsed from package artifacts (inference_config, trt_config).",
        examples=[{"static_batch_size": 1, "training_input_size": {"height": 640, "width": 640}}],
    )


class ModelMetadata(BaseModel):
    """Registered model plus the specific package profiled in this run."""

    registered_model: RegisteredModelMetadata = Field(
        description="Identity of the registered model (shared across backend packages).",
    )
    package: PackageMetadata = Field(
        description="The single backend package artifact profiled in this record.",
    )


class RuntimeMetadata(BaseModel):
    """Code path executed for this profiling run."""

    module_name: str = Field(
        description="Python module of the profiled model class.",
        examples=["inference_models.models.yolov8.yolov8_object_detection_onnx"],
    )
    class_name: str = Field(
        description="Profiled model class name.",
        examples=["Yolov8ObjectDetectionOnnx"],
    )
    method: str = Field(
        description="Method invoked during profiling (e.g. infer).",
        examples=["infer", "prompt"],
    )
    profiling_workflow: List[str] = Field(
        default_factory=list,
        description="Ordered steps executed by the harness for this run. Currently placeholder for multi-step runs.",
        examples=[["infer"], ["prompt"]],
    )
    inference_models_version: Optional[str] = Field(
        default=None,
        description="Installed inference-models package version at profile time.",
        examples=["0.54.2"],
    )


class EnvironmentMetadata(BaseModel):
    """Host and GPU environment for the profiling run."""

    os: Optional[str] = Field(
        default=None,
        description="Host operating system description.",
        examples=["Linux-6.8.0-x86_64-with-glibc2.39"],
    )
    cpu: Optional[str] = Field(
        default=None,
        description="Host CPU identifier or architecture string.",
        examples=["x86_64", "aarch64"],
    )
    gpu: Optional[str] = Field(
        default=None,
        description="CUDA device marketing name (e.g. NVIDIA L40S).",
        examples=["NVIDIA L40S", "NVIDIA A100-SXM4-80GB"],
    )
    gpu_driver: Optional[str] = Field(
        default=None,
        description="NVIDIA driver version from NVML when available.",
        examples=["550.54.15"],
    )
    cuda_version: Optional[str] = Field(
        default=None,
        description="CUDA toolkit version reported by PyTorch.",
        examples=["12.4"],
    )
    cudnn_version: Optional[str] = Field(
        default=None,
        description="cuDNN version reported by PyTorch.",
        examples=["8907"],
    )
    gpu_memory_total_bytes: Optional[int] = Field(
        default=None,
        description="Total GPU device memory in bytes.",
        examples=[48318382080],
    )
    compute_capability: Optional[str] = Field(
        default=None,
        description="CUDA compute capability major.minor for the profiled device.",
        examples=["8.9", "9.0"],
    )


class ProfilingRunMetadata(BaseModel):
    """Operational parameters of the profiling harness run."""

    warmup_iterations: int = Field(
        description="Number of inference iterations run before peak memory measurement.",
        examples=[2],
    )
    measured_iterations: int = Field(
        description="Number of iterations included in peak memory measurement.",
        examples=[5],
    )
    concurrency: int = Field(
        default=1,
        description="Concurrent execution contexts or requests assumed for this profile.",
        examples=[1, 4],
    )
    in_process: bool = Field(
        default=False,
        description="True when profiling ran in the parent process instead of an isolated worker.",
        examples=[False],
    )
    torch_profiler_memory_enabled: bool = Field(
        default=False,
        description="True when torch.profiler profile_memory was enabled for measured iterations.",
        examples=[False, True],
    )
    nvml_sampling_interval_seconds: Optional[float] = Field(
        default=None,
        description="NVML polling interval for ONNX/TensorRT process-memory peaks.",
        examples=[0.01],
    )
    trace_files: List[str] = Field(
        default_factory=list,
        description="Paths to profiler trace artifacts written during the run.",
        examples=[["/tmp/onnx-ort-profile-abc123.json"]],
    )


class TorchBackendMetadata(BaseModel):
    """PyTorch / TorchScript / HF runtime stack for this run."""

    torch_version: Optional[str] = Field(
        default=None,
        description="Installed PyTorch version.",
        examples=["2.5.1+cu124"],
    )
    torchvision_version: Optional[str] = Field(
        default=None,
        description="Installed torchvision version when present.",
        examples=["0.20.1+cu124"],
    )
    transformers_version: Optional[str] = Field(
        default=None,
        description="Installed transformers version when present (Hugging Face models).",
        examples=["4.48.0"],
    )


class OnnxBackendMetadata(BaseModel):
    """ONNX Runtime configuration for this run."""

    onnxruntime_version: Optional[str] = Field(
        default=None,
        description="ONNX Runtime library version.",
        examples=["1.20.1"],
    )
    execution_providers: List[str] = Field(
        default_factory=list,
        description="Ordered ONNX Runtime execution providers used for the session.",
        examples=[["CUDAExecutionProvider", "CPUExecutionProvider"]],
    )
    trace_dir: Optional[str] = Field(
        default=None,
        description="Directory where ONNX Runtime profiling traces were written.",
        examples=["/tmp/onnx-ort-profile-abc123"],
    )
    opset: Optional[int] = Field(
        default=None,
        description=(
            "ONNX opset of the exported graph from registry package metadata "
            "(``ModelPackageMetadata.onnx_package_details``) or package "
            "``model_config.json`` when profiling a local path."
        ),
        examples=[17, 19],
    )


class TensorRTBackendMetadata(BaseModel):
    """TensorRT runtime configuration for this run."""

    tensorrt_version: Optional[str] = Field(
        default=None,
        description="TensorRT library version.",
        examples=["10.7.0"],
    )
    engine_size_bytes: Optional[int] = Field(
        default=None,
        description="Serialized engine.plan size on disk in bytes.",
        examples=[52428800],
    )
    execution_context_bytes: Optional[int] = Field(
        default=None,
        description="GPU memory attributed to execution context creation when measured.",
        examples=[268435456],
    )
    optimization_profile: Optional[Dict[str, Any]] = Field(
        default=None,
        description="TensorRT optimization profile bounds used for this run.",
        examples=[
            {
                "min": {"batch": 1, "height": 640, "width": 640},
                "opt": {"batch": 1, "height": 640, "width": 640},
                "max": {"batch": 4, "height": 640, "width": 640},
            }
        ],
    )
    max_workspace_setting: Optional[int] = Field(
        default=None,
        description="Configured max workspace size for TensorRT execution.",
        examples=[1073741824],
    )
    num_contexts_profiled: int = Field(
        default=1,
        description="Number of execution contexts included in this profile.",
        examples=[1],
    )
    trt_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parsed trt_config.json fields relevant to batch and memory.",
        examples=[{"static_batch_size": 1, "dynamic_batch_size_max": 4}],
    )


class TorchMetrics(BaseModel):
    """Normalized GPU memory metrics for PyTorch CUDA profiling."""

    idle_after_load_allocated_bytes: int = Field(
        description="torch.cuda.memory_allocated immediately after model load.",
        examples=[536870912],
    )
    idle_after_load_reserved_bytes: int = Field(
        description="torch.cuda.memory_reserved immediately after model load.",
        examples=[1073741824],
    )
    peak_allocated_bytes: int = Field(
        description="Peak torch.cuda.memory_allocated during measured iterations.",
        examples=[805306368],
    )
    peak_reserved_bytes: int = Field(
        description="Peak torch.cuda.memory_reserved during measured iterations.",
        examples=[1610612736],
    )
    end_reserved_bytes: int = Field(
        description="torch.cuda.memory_reserved after measured iterations complete.",
        examples=[1610612736],
    )
    peak_incremental_allocated_bytes: int = Field(
        description="peak_allocated_bytes minus idle_after_load_allocated_bytes.",
        examples=[268435456],
    )
    peak_incremental_reserved_bytes: int = Field(
        description="peak_reserved_bytes minus idle_after_load_reserved_bytes.",
        examples=[536870912],
    )
    baseline_gpu_free_bytes_nvml: Optional[int] = Field(
        default=None,
        description="GPU free memory via NVML before load when available.",
        examples=[45097156608],
    )


class OnnxMetrics(BaseModel):
    """Normalized GPU memory metrics for ONNX Runtime profiling."""

    baseline_process_gpu_bytes_nvml: Optional[int] = Field(
        default=None,
        description="Process GPU memory via NVML before session creation.",
        examples=[104857600],
    )
    idle_after_session_create_bytes: Optional[int] = Field(
        default=None,
        description="Process GPU memory via NVML after ORT session creation.",
        examples=[536870912],
    )
    peak_process_gpu_bytes: Optional[int] = Field(
        default=None,
        description="Peak process GPU memory via NVML during measured inference.",
        examples=[805306368],
    )
    delta_peak_bytes: Optional[int] = Field(
        default=None,
        description="peak_process_gpu_bytes minus idle_after_session_create_bytes.",
        examples=[268435456],
    )


class TensorRTMetrics(BaseModel):
    """Normalized GPU memory metrics for TensorRT profiling."""

    baseline_process_gpu_bytes_nvml: Optional[int] = Field(
        default=None,
        description="Process GPU memory via NVML before engine deserialize.",
        examples=[104857600],
    )
    idle_after_deserialize_bytes: Optional[int] = Field(
        default=None,
        description="Process GPU memory via NVML after engine and context load.",
        examples=[1073741824],
    )
    peak_request_bytes: Optional[int] = Field(
        default=None,
        description="Peak process GPU memory via NVML during measured inference.",
        examples=[1610612736],
    )
    delta_peak_bytes: Optional[int] = Field(
        default=None,
        description="peak_request_bytes minus idle_after_deserialize_bytes.",
        examples=[536870912],
    )


class MemoryProfileRecord(BaseModel):
    """Full profiling record: metrics plus structured metadata."""

    schema_version: str = Field(
        default="1.1",
        description="Version of this profiling record JSON schema.",
        examples=["1.1"],
    )
    profile_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this profiling run result.",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    )
    profile_tier: ProfileTier = Field(
        default=ProfileTier.CUSTOMER,
        description=PROFILE_TIER_DESCRIPTIONS,
    )
    metrics: Union[TorchMetrics, OnnxMetrics, TensorRTMetrics] = Field(
        description="Backend-specific normalized GPU memory metrics.",
    )
    model_metadata: ModelMetadata = Field(
        description="Registered model identity and profiled package metadata.",
    )
    runtime_metadata: RuntimeMetadata = Field(
        description="Code path and inference-models version used for profiling.",
    )
    backend_metadata: Union[
        TorchBackendMetadata,
        OnnxBackendMetadata,
        TensorRTBackendMetadata,
    ] = Field(
        description="Runtime stack configuration for the profiling backend.",
    )
    input_metadata: InputMetadata = Field(
        description="Exercised and declared inference inputs for admission envelopes.",
    )
    environment_metadata: EnvironmentMetadata = Field(
        description="Host and GPU environment during the profiling run.",
    )
    profiling_run: ProfilingRunMetadata = Field(
        description="Harness iteration counts, concurrency, and trace settings.",
    )
    notes: List[str] = Field(
        default_factory=list,
        description="Human-readable caveats and alignment notes from the harness.",
        examples=[
            [
                "Package defines static_batch_size=1; profiling at that batch size.",
            ]
        ],
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        description="UTC timestamp when the record was finalized (ISO-8601).",
        examples=["2026-05-31T12:00:00Z"],
    )

    def as_json_dict(self) -> Dict[str, Any]:
        json_dict = self.model_dump(mode="json")

        return json_dict


def _distribution_version(package_name: str) -> Optional[str]:
    try:
        version = importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None

    return version


def collect_inference_models_version() -> Optional[str]:
    version = _distribution_version("inference-models")

    return version


def collect_environment_metadata(
    *,
    device: torch.device,
) -> EnvironmentMetadata:
    gpu_name: Optional[str] = None
    gpu_memory_total_bytes: Optional[int] = None
    compute_capability: Optional[str] = None
    cuda_version: Optional[str] = None
    cudnn_version: Optional[str] = None
    gpu_driver: Optional[str] = None

    if device.type == "cuda" and torch.cuda.is_available():
        device_index = device.index if device.index is not None else 0
        try:
            gpu_name = torch.cuda.get_device_name(device)
            props = torch.cuda.get_device_properties(device_index)
            gpu_memory_total_bytes = int(props.total_memory)
            compute_capability = f"{props.major}.{props.minor}"
        except Exception:
            pass

        try:
            cuda_version = torch.version.cuda
        except Exception:
            pass

        try:
            cudnn_version = str(torch.backends.cudnn.version())
        except Exception:
            pass

        try:
            import pynvml  # type: ignore[import-untyped]

            pynvml.nvmlInit()
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            gpu_driver = str(driver_version)
        except Exception:
            pass

    environment = EnvironmentMetadata(
        os=platform.platform(),
        cpu=platform.processor() or platform.machine(),
        gpu=gpu_name,
        gpu_driver=gpu_driver,
        cuda_version=cuda_version,
        cudnn_version=cudnn_version,
        gpu_memory_total_bytes=gpu_memory_total_bytes,
        compute_capability=compute_capability,
    )

    return environment


def build_image_input_metadata(
    *,
    batch_size: int,
    height: int,
    width: int,
    batch_source: Optional[str] = None,
    spatial_source: Optional[str] = None,
    batch_resolution: Optional[Literal["static", "dynamic"]] = None,
    spatial_resolution: Optional[Literal["static", "dynamic"]] = None,
) -> Dict[str, InputAxisValue]:
    image_axes = {
        "batch": InputAxisValue(
            value=batch_size,
            resolution=batch_resolution,
            source=batch_source,
        ),
        "height": InputAxisValue(
            value=height,
            resolution=spatial_resolution,
            source=spatial_source,
        ),
        "width": InputAxisValue(
            value=width,
            resolution=spatial_resolution,
            source=spatial_source,
        ),
        "channels": InputAxisValue(
            value=3,
            resolution="static",
            source="profiling_harness_default",
        ),
    }

    return image_axes


def _derive_axis_value_from_infer_input(axis_name: str, value: Any) -> Optional[Any]:
    if axis_name == "num_classes" and isinstance(value, list):
        return len(value)
    if axis_name in {"num_points", "num_boxes"} and hasattr(value, "__len__"):
        try:
            return len(value)
        except TypeError:
            return None
    if axis_name == "prompt_token_length":
        return None

    return value


def _axes_from_infer_input_value(
    input_spec: Dict[str, Any],
    value: Any,
    *,
    runtime_axes: Optional[Dict[str, Any]] = None,
) -> Dict[str, InputAxisValue]:
    axes: Dict[str, InputAxisValue] = {}
    axis_specs: List[Dict[str, Any]] = input_spec.get("axes") or []
    runtime_axes = runtime_axes or {}

    for axis_spec in axis_specs:
        if not isinstance(axis_spec, dict):
            continue

        axis_name = str(axis_spec.get("name") or "")
        if not axis_name:
            continue

        if axis_name in runtime_axes:
            derived = runtime_axes[axis_name]
            source = "runtime_tokenizer"
        else:
            derived = _derive_axis_value_from_infer_input(axis_name, value)
            source = "infer_kwargs"

        if derived is None:
            continue

        resolution = axis_spec.get("resolution")
        axes[axis_name] = InputAxisValue(
            value=derived,
            resolution=resolution if resolution in ("static", "dynamic") else "static",
            source=source,
        )

    if not axes:
        input_name = str(input_spec.get("name") or "value")
        axes[input_name] = InputAxisValue(
            value=value,
            resolution="static",
            source="infer_kwargs",
        )

    return axes


def _build_declared_inputs_from_profile(
    task_profile_spec: Optional[Dict[str, Any]],
    *,
    profiled_input_name: str,
    profiled_axes: Dict[str, InputAxisValue],
    infer_kwargs: Dict[str, Any],
    runtime_axis_values: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[DeclaredInputSpec]:
    if not task_profile_spec:
        return []

    declared: List[DeclaredInputSpec] = []
    profile_inputs: List[Dict[str, Any]] = task_profile_spec.get("inputs") or []
    method_kwargs_specs: List[Dict[str, Any]] = (
        task_profile_spec.get("method_kwargs_memory_relevant") or []
    )

    for input_spec in profile_inputs:
        if not isinstance(input_spec, dict):
            continue

        name = str(input_spec.get("name") or "")
        if not name:
            continue

        if name == profiled_input_name:
            is_profiled = True
            axes = profiled_axes
            notes = None
        else:
            value = infer_kwargs.get(name)
            is_profiled = value is not None
            axes = (
                _axes_from_infer_input_value(
                    input_spec,
                    value,
                    runtime_axes=(runtime_axis_values or {}).get(name),
                )
                if is_profiled
                else {}
            )
            notes = (
                None
                if is_profiled
                else (
                    "Declared by registry profile; harness defaults apply when "
                    "required, or set via --infer-kwargs-json to override."
                )
            )

        declared.append(
            DeclaredInputSpec(
                name=name,
                kind=str(input_spec.get("kind") or "unknown"),
                memory_impact=input_spec.get("memory_impact"),
                profiled=is_profiled,
                axes=axes,
                notes=notes,
            )
        )

    for kwarg_spec in method_kwargs_specs:
        if not isinstance(kwarg_spec, dict):
            continue

        kwarg_name = str(kwarg_spec.get("name") or "")
        if not kwarg_name:
            continue

        value = infer_kwargs.get(kwarg_name)
        axes: Dict[str, InputAxisValue] = {}
        if value is not None:
            axes[kwarg_name] = InputAxisValue(
                value=value,
                resolution="static",
                source="infer_kwargs",
            )

        declared.append(
            DeclaredInputSpec(
                name=kwarg_name,
                kind="method_kwargs",
                memory_impact=kwarg_spec.get("memory_impact"),
                profiled=value is not None,
                axes=axes,
                notes=(
                    None
                    if value is not None
                    else "Declared method kwarg; set via --infer-kwargs-json to profile."
                ),
            )
        )

    return declared


def build_input_metadata_with_registry(
    *,
    module_name: Optional[str],
    class_name: Optional[str],
    architecture: Optional[str],
    task_type: Optional[str],
    backend: Optional[str],
    batch_size: int,
    height: int,
    width: int,
    infer_kwargs: Dict[str, Any],
    profiled_input_name: str = "images",
    shape_spec: Optional[PackageProfilingShapeSpec] = None,
    runtime_axis_values: Optional[Dict[str, Dict[str, Any]]] = None,
) -> InputMetadata:
    """Build input metadata including registry-declared VLM/SAM/vision contracts."""
    registry_context = resolve_registry_input_context(
        module_name=module_name,
        class_name=class_name,
        architecture=architecture,
        task_type=task_type,
        backend=backend,
    )
    task_profile_spec = registry_context.get("task_profile_spec")
    task_inference_profile = registry_context.get("task_inference_profile")
    profiling_method = (
        task_profile_spec.get("profiling_method")
        if isinstance(task_profile_spec, dict)
        else None
    )

    batch_resolution: Optional[Literal["static", "dynamic"]] = None
    spatial_resolution: Optional[Literal["static", "dynamic"]] = None
    batch_source: Optional[str] = None
    spatial_source: Optional[str] = None

    if shape_spec is not None:
        if shape_spec.batch.resolution in ("static", "dynamic"):
            batch_resolution = shape_spec.batch.resolution
            batch_source = shape_spec.batch.source
        if shape_spec.height.resolution == shape_spec.width.resolution and (
            shape_spec.height.resolution in ("static", "dynamic")
        ):
            spatial_resolution = shape_spec.height.resolution
            spatial_source = shape_spec.height.source

    profiled_axes = build_image_input_metadata(
        batch_size=batch_size,
        height=height,
        width=width,
        batch_source=batch_source,
        spatial_source=spatial_source,
        batch_resolution=batch_resolution,
        spatial_resolution=spatial_resolution,
    )

    declared_inputs = _build_declared_inputs_from_profile(
        task_profile_spec if isinstance(task_profile_spec, dict) else None,
        profiled_input_name=profiled_input_name,
        profiled_axes=profiled_axes,
        infer_kwargs=infer_kwargs,
        runtime_axis_values=runtime_axis_values,
    )

    profiled_inputs: Dict[str, Dict[str, InputAxisValue]] = {
        profiled_input_name: profiled_axes,
    }
    for declared_input in declared_inputs:
        if declared_input.profiled and declared_input.name != profiled_input_name:
            profiled_inputs[declared_input.name] = declared_input.axes

    input_metadata = InputMetadata(
        task_inference_profile=(
            str(task_inference_profile) if task_inference_profile else None
        ),
        profiling_method=str(profiling_method) if profiling_method else None,
        inputs=profiled_inputs,
        declared_inputs=declared_inputs,
        infer_defaults=dict(infer_kwargs),
    )

    return input_metadata


def build_model_metadata_from_context(
    metadata_context: Dict[str, Any],
    *,
    package_dir: Optional[Path] = None,
) -> ModelMetadata:
    package_path = metadata_context.get("package_path")
    resolved_dir = package_dir or (
        Path(package_path) if package_path else None
    )

    package_features: Dict[str, Any] = {}
    if resolved_dir is not None and resolved_dir.is_dir():
        package_features = extract_package_features(resolved_dir)
        backend = metadata_context.get("backend")
        if backend == BackendType.TRT.value:
            package_features = {
                **package_features,
                **extract_trt_package_features(resolved_dir),
            }

    backend = metadata_context.get("backend") or BackendType.TORCH.value
    architecture = metadata_context.get("architecture")
    task_type = metadata_context.get("task_type")
    model_variant = metadata_context.get("model_variant")

    registered_model = RegisteredModelMetadata(
        model_id=metadata_context.get("model_id"),
        architecture=architecture,
        task_type=task_type,
        model_variant=model_variant,
        registry_features=metadata_context.get("registry_features") or {},
    )
    package_metadata = PackageMetadata(
        package_id=metadata_context.get("package_id"),
        package_path=str(resolved_dir) if resolved_dir is not None else package_path,
        backend=backend,
        quantization=metadata_context.get("quantization"),
        package_features=package_features,
    )
    model_metadata = ModelMetadata(
        registered_model=registered_model,
        package=package_metadata,
    )

    return model_metadata


def build_runtime_metadata(
    *,
    module_name: str,
    class_name: str,
    method_name: str,
    profiling_workflow: Optional[List[str]] = None,
) -> RuntimeMetadata:
    runtime_metadata = RuntimeMetadata(
        module_name=module_name,
        class_name=class_name,
        method=method_name,
        profiling_workflow=profiling_workflow or [method_name],
        inference_models_version=collect_inference_models_version(),
    )

    return runtime_metadata


def collect_torch_backend_metadata() -> TorchBackendMetadata:
    backend_metadata = TorchBackendMetadata(
        torch_version=torch.__version__,
        torchvision_version=_distribution_version("torchvision"),
        transformers_version=_distribution_version("transformers"),
    )

    return backend_metadata


def collect_onnx_backend_metadata(
    *,
    onnxruntime_version: str,
    execution_providers: List[str],
    trace_dir: Optional[str],
    opset: Optional[int] = None,
) -> OnnxBackendMetadata:
    backend_metadata = OnnxBackendMetadata(
        onnxruntime_version=onnxruntime_version,
        execution_providers=execution_providers,
        trace_dir=trace_dir,
        opset=opset,
    )

    return backend_metadata


def resolve_onnx_opset_from_metadata_context(
    metadata_context: Dict[str, Any],
) -> Optional[int]:
    """Read ONNX opset stored in the profiling ``metadata_context`` payload."""
    opset = metadata_context.get("onnx_opset")
    if opset is None:
        return None

    return int(opset)


def finalize_profile_record(
    *,
    profile_tier: ProfileTier,
    metrics: Union[TorchMetrics, OnnxMetrics, TensorRTMetrics],
    model_metadata: ModelMetadata,
    runtime_metadata: RuntimeMetadata,
    backend_metadata: Union[
        TorchBackendMetadata,
        OnnxBackendMetadata,
        TensorRTBackendMetadata,
    ],
    input_metadata: InputMetadata,
    environment_metadata: EnvironmentMetadata,
    profiling_run: ProfilingRunMetadata,
    notes: List[str],
) -> MemoryProfileRecord:
    record = MemoryProfileRecord(
        profile_tier=profile_tier,
        metrics=metrics,
        model_metadata=model_metadata,
        runtime_metadata=runtime_metadata,
        backend_metadata=backend_metadata,
        input_metadata=input_metadata,
        environment_metadata=environment_metadata,
        profiling_run=profiling_run,
        notes=notes,
    )

    return record
