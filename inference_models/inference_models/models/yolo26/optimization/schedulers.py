"""Selectable ONNX submission strategies for YOLO26 depth estimation."""

from __future__ import annotations

from numbers import Integral
from threading import Lock
from typing import List, Optional, Sequence, Union

import torch

from inference_models.errors import ModelRuntimeError
from inference_models.models.common.onnx import (
    ort_tensor_type_to_torch_tensor_type,
    run_onnx_session_with_batch_size_limit,
    torch_tensor_type_to_onnx_type,
)
from inference_models.models.common.streams import get_cuda_stream
from inference_models.models.optimization.contracts import (
    DeviceCompatibility,
    ExecutionContext,
    InputCompatibility,
    OptimizationMetadata,
    OptimizationStage,
    immutable_mapping,
    metadata_supports_context,
)
from inference_models.models.yolo26.optimization.ids import (
    YOLO26_DEPTH_ONNX_SCHEDULER_BASE,
    YOLO26_DEPTH_ONNX_SCHEDULER_ORT_CUDA_GRAPH_V1,
)

try:
    import onnxruntime
except ImportError:
    onnxruntime = None

OnnxProvider = Union[str, tuple]


def configure_providers_for_scheduler(
    *,
    providers: List[OnnxProvider],
    scheduler_id: str,
) -> List[OnnxProvider]:
    """Enable CUDA Graph support only for the explicit graph scheduler."""
    if scheduler_id != YOLO26_DEPTH_ONNX_SCHEDULER_ORT_CUDA_GRAPH_V1:
        return providers

    configured: List[OnnxProvider] = []
    for provider in providers:
        provider_name = provider if isinstance(provider, str) else provider[0]
        if provider_name != "CUDAExecutionProvider":
            configured.append(provider)
            continue
        provider_options = {} if isinstance(provider, str) else dict(provider[1])
        provider_options["enable_cuda_graph"] = "1"
        configured.append((provider_name, provider_options))
    return configured


def build_base_scheduler_metadata(*, device_kind: str) -> OptimizationMetadata:
    """Build base metadata for the model's effective CPU or GPU device."""
    return OptimizationMetadata(
        implementation_id=YOLO26_DEPTH_ONNX_SCHEDULER_BASE,
        stage=OptimizationStage.SCHEDULER,
        version="1",
        target=DeviceCompatibility(device_kind=device_kind),
        inputs=InputCompatibility(
            scenarios=("*",),
            axis_constraints=immutable_mapping({"batch": ">=1"}),
            dtypes=("model input dtype",),
            layouts=("contiguous NCHW",),
        ),
        dependencies=("onnxruntime", "torch"),
        fallback_id=YOLO26_DEPTH_ONNX_SCHEDULER_BASE,
        changes_numerics=False,
        supports_concurrency=True,
        supports_cuda_graphs=False,
        output_contract=immutable_mapping(
            {
                "allocation": "ONNX Runtime IO binding output per dispatch",
                "ownership": "returned tensor owns or references its ORT output",
                "lifetime": "independent of later model requests",
            }
        ),
        numerical_behavior="preserves the existing ONNX IO-binding implementation",
        stream_behavior=(
            "uses the calling thread's reusable inference stream and preserves the "
            "existing synchronization behavior"
        ),
    )


class BaseOnnxExecutionScheduler:
    """Preserve the existing ONNX IO-binding execution path."""

    def __init__(
        self,
        *,
        session,
        input_name: str,
        input_batch_size: Optional[int],
        device: torch.device,
        metadata: OptimizationMetadata,
    ) -> None:
        self.metadata = metadata
        self._session = session
        self._input_name = input_name
        self._input_batch_size = input_batch_size
        self._device = device

    def is_compatible(self, context: ExecutionContext) -> bool:
        """Return whether the preserved path supports the effective device."""
        return metadata_supports_context(self.metadata, context)

    def execute(self, pre_processed_images: torch.Tensor) -> torch.Tensor:
        """Execute the original batch-limited ONNX helper."""
        return run_onnx_session_with_batch_size_limit(
            session=self._session,
            inputs={self._input_name: pre_processed_images},
            min_batch_size=self._input_batch_size,
            max_batch_size=self._input_batch_size,
            stream=get_cuda_stream(device=self._device, purpose="inference"),
        )[0]


class OrtCudaGraphExecutionScheduler:
    """Replay one fixed-shape ONNX Runtime CUDA Graph with stable IO buffers."""

    metadata = OptimizationMetadata(
        implementation_id=YOLO26_DEPTH_ONNX_SCHEDULER_ORT_CUDA_GRAPH_V1,
        stage=OptimizationStage.SCHEDULER,
        version="1",
        target=DeviceCompatibility(
            device_kind="gpu",
            minimum_compute_capability=(7, 0),
        ),
        inputs=InputCompatibility(
            scenarios=(
                "camera_640x480_batch_1_base",
                "camera_3840x2160_batch_1_high",
            ),
            axis_constraints=immutable_mapping(
                {
                    "batch": 1,
                    "network_shape": "static",
                    "model_inputs": 1,
                    "model_outputs": 1,
                }
            ),
            dtypes=("model input dtype",),
            layouts=("contiguous NCHW",),
        ),
        dependencies=("onnxruntime", "torch"),
        fallback_id=YOLO26_DEPTH_ONNX_SCHEDULER_BASE,
        changes_numerics=False,
        supports_concurrency=False,
        supports_cuda_graphs=True,
        output_contract=immutable_mapping(
            {
                "accepted_input": (
                    "one contiguous CUDA tensor with the session's static shape and dtype"
                ),
                "allocation": (
                    "one persistent input, output, IO binding, and ORT graph per model"
                ),
                "aliasing": "returned raw output aliases the persistent graph output",
                "lifetime": (
                    "raw output is valid until the next dispatch; model clones final "
                    "postprocessed results before releasing request serialization"
                ),
                "per_call_allocation": "independent final result tensors only",
                "graph_cache": "one fixed graph; no dynamic-shape cache",
            }
        ),
        numerical_behavior=(
            "replays the unchanged ONNX graph with identical input and output dtypes"
        ),
        stream_behavior=(
            "copies into stable storage on one reusable CUDA stream, synchronizes that "
            "producer before ORT replay, and relies on synchronous RunWithBinding "
            "completion before postprocessing"
        ),
    )

    def __init__(
        self,
        *,
        session,
        input_name: str,
        device: torch.device,
    ) -> None:
        self._session = session
        self._input_name = input_name
        self._device = device
        self._lock = Lock()
        self._stream = torch.cuda.Stream(device=device)
        self._binding = None
        self._stable_input: Optional[torch.Tensor] = None
        self._stable_output: Optional[torch.Tensor] = None
        self._run_options = onnxruntime.RunOptions()
        self._run_options.add_run_config_entry("gpu_graph_id", "0")
        self._compatibility_error = self._inspect_session_compatibility()
        if self._compatibility_error is None:
            self._enable_cuda_graph()

    def is_compatible(self, context: ExecutionContext) -> bool:
        """Require a fixed-shape, CUDA-first, single-input/output ORT session."""
        return (
            metadata_supports_context(self.metadata, context)
            and self._compatibility_error is None
        )

    @property
    def compatibility_error(self) -> Optional[str]:
        """Return a concrete static-session incompatibility reason, if any."""
        return self._compatibility_error

    def execute(self, pre_processed_images: torch.Tensor) -> torch.Tensor:
        """Capture on first use and replay the fixed ONNX graph thereafter."""
        if self._compatibility_error is not None:
            raise self._runtime_error(self._compatibility_error)
        self._validate_request(pre_processed_images)
        with self._lock:
            if self._binding is None:
                self._initialize_stable_binding(pre_processed_images)
            with torch.cuda.stream(self._stream):
                pre_processed_images.record_stream(self._stream)
                self._stable_input.copy_(pre_processed_images)
            self._stream.synchronize()
            self._session.run_with_iobinding(self._binding, self._run_options)
            return self._stable_output

    def _inspect_session_compatibility(self) -> Optional[str]:
        providers = self._session.get_providers()
        if not providers or providers[0] != "CUDAExecutionProvider":
            return (
                "requires CUDAExecutionProvider to be the session's highest-priority "
                f"provider, received {providers!r}"
            )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            return (
                "requires exactly one tensor input and one tensor output, received "
                f"{len(inputs)} inputs and {len(outputs)} outputs"
            )
        if inputs[0].name != self._input_name:
            return f"session input is {inputs[0].name!r}, expected {self._input_name!r}"
        if self._static_shape(inputs[0].shape) is None:
            return f"requires a static input shape, received {inputs[0].shape!r}"
        if self._static_shape(outputs[0].shape) is None:
            return f"requires a static output shape, received {outputs[0].shape!r}"
        return None

    def _enable_cuda_graph(self) -> None:
        provider_options = self._session.get_provider_options()
        providers = [
            (provider, provider_options.get(provider, {}))
            for provider in self._session.get_providers()
        ]
        configured = configure_providers_for_scheduler(
            providers=providers,
            scheduler_id=YOLO26_DEPTH_ONNX_SCHEDULER_ORT_CUDA_GRAPH_V1,
        )
        self._session.set_providers(configured)

    def _validate_request(self, tensor: torch.Tensor) -> None:
        expected = self._session.get_inputs()[0]
        expected_shape = self._static_shape(expected.shape)
        expected_dtype = ort_tensor_type_to_torch_tensor_type(expected.type)
        device_index_mismatch = (
            self._device.index is not None and tensor.device.index != self._device.index
        )
        if tensor.device.type != self._device.type or device_index_mismatch:
            raise self._runtime_error(
                f"requires input on {self._device}, received {tensor.device}"
            )
        if tuple(tensor.shape) != expected_shape:
            raise self._runtime_error(
                f"requires input shape {expected_shape}, received {tuple(tensor.shape)}"
            )
        if tensor.dtype != expected_dtype:
            raise self._runtime_error(
                f"requires input dtype {expected_dtype}, received {tensor.dtype}"
            )
        if not tensor.is_contiguous():
            raise self._runtime_error("requires a contiguous NCHW input tensor")

    def _initialize_stable_binding(self, tensor: torch.Tensor) -> None:
        output = self._session.get_outputs()[0]
        output_shape = self._static_shape(output.shape)
        output_dtype = ort_tensor_type_to_torch_tensor_type(output.type)
        with torch.cuda.stream(self._stream):
            self._stable_input = torch.empty_like(tensor)
            self._stable_output = torch.empty(
                output_shape,
                dtype=output_dtype,
                device=self._device,
            )
        self._stream.synchronize()
        binding = self._session.io_binding()
        binding.bind_input(
            name=self._input_name,
            device_type="cuda",
            device_id=self._device.index or 0,
            element_type=torch_tensor_type_to_onnx_type(self._stable_input.dtype),
            shape=tuple(self._stable_input.shape),
            buffer_ptr=self._stable_input.data_ptr(),
        )
        binding.bind_output(
            name=output.name,
            device_type="cuda",
            device_id=self._device.index or 0,
            element_type=torch_tensor_type_to_onnx_type(self._stable_output.dtype),
            shape=tuple(self._stable_output.shape),
            buffer_ptr=self._stable_output.data_ptr(),
        )
        self._binding = binding

    @staticmethod
    def _static_shape(shape: Sequence) -> Optional[tuple]:
        if any(
            not isinstance(dimension, Integral) or dimension <= 0 for dimension in shape
        ):
            return None
        return tuple(int(dimension) for dimension in shape)

    @staticmethod
    def _runtime_error(reason: str) -> ModelRuntimeError:
        return ModelRuntimeError(
            message=(
                "YOLO26 depth ONNX scheduler 'ort-cuda-graph-v1' cannot execute: "
                f"{reason}."
            ),
            help_url=(
                "https://inference-models.roboflow.com/errors/models-runtime/"
                "#modelruntimeerror"
            ),
        )
