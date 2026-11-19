import threading
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.developer_tools import align_device_with_onnx_session
from inference_models.entities import Confidence
from inference_models.errors import (
    EnvironmentConfigurationError,
    MissingDependencyError,
)
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.onnx import (
    run_onnx_session_with_batch_size_limit,
    set_onnx_execution_provider_defaults,
)
from inference_models.utils.onnx_introspection import (
    get_selected_onnx_execution_providers,
)

try:
    import onnxruntime
except ImportError as import_error:
    raise MissingDependencyError(
        message="Running MediaPipe hand models with the ONNX backend requires onnxruntime, which is brought "
        "with the `onnx-*` extras of `inference-models`. If you see this error running locally, please follow "
        "our installation guide: https://inference-models.roboflow.com/getting-started/installation/",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error

DEFAULT_MAX_BATCH_SIZE = 16


class OnnxGraph:
    """A converted MediaPipe graph: one NHWC float input in [0, 1], several outputs."""

    @classmethod
    def from_package(
        cls,
        model_name_or_path: str,
        onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
        default_onnx_trt_options: bool = True,
        device: torch.device = DEFAULT_DEVICE,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    ) -> "OnnxGraph":
        if onnx_execution_providers is None:
            onnx_execution_providers = get_selected_onnx_execution_providers()
        if not onnx_execution_providers:
            raise EnvironmentConfigurationError(
                message="Could not initialize a MediaPipe hand model - the ONNX backend requires execution "
                "providers, given explicitly in `from_pretrained(...)` or through the "
                "`ONNXRUNTIME_EXECUTION_PROVIDERS` environment variable.",
                help_url="https://inference-models.roboflow.com/errors/runtime-environment/#environmentconfigurationerror",
            )
        onnx_execution_providers = set_onnx_execution_provider_defaults(
            providers=onnx_execution_providers,
            model_package_path=model_name_or_path,
            device=device,
            default_onnx_trt_options=default_onnx_trt_options,
        )
        contents = get_model_package_contents(
            model_package_dir=model_name_or_path, elements=["weights.onnx"]
        )
        session = onnxruntime.InferenceSession(
            path_or_bytes=contents["weights.onnx"], providers=onnx_execution_providers
        )
        device = align_device_with_onnx_session(session=session, device=device)
        return cls(session=session, device=device, max_batch_size=max_batch_size)

    def __init__(
        self,
        session: "onnxruntime.InferenceSession",
        device: torch.device,
        max_batch_size: int,
    ):
        self._session = session
        self._lock = threading.Lock()
        self._device = device
        self._max_batch_size = max_batch_size
        self._input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        self.input_size: Tuple[int, int] = (int(input_shape[1]), int(input_shape[2]))

    @property
    def device(self) -> torch.device:
        return self._device

    def run(self, batch: np.ndarray) -> List[np.ndarray]:
        """Run a (N, H, W, 3) float32 batch, chunked to the maximum batch size."""
        inputs = {self._input_name: torch.from_numpy(batch).to(self._device)}
        with self._lock:
            outputs = run_onnx_session_with_batch_size_limit(
                session=self._session,
                inputs=inputs,
                max_batch_size=self._max_batch_size,
            )
        return [output.detach().cpu().numpy() for output in outputs]


def graph_kwargs(kwargs: dict) -> dict:
    """The ``from_pretrained`` arguments that configure the ONNX session."""
    return {
        key: kwargs[key]
        for key in (
            "onnx_execution_providers",
            "default_onnx_trt_options",
            "device",
            "max_batch_size",
        )
        if key in kwargs
    }


def resolve_confidence(confidence: Confidence, default: float) -> float:
    if isinstance(confidence, (int, float)):
        return float(confidence)
    return default
