from dataclasses import dataclass
from threading import Lock
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torchvision import transforms

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.entities import ColorFormat
from inference_models.errors import (
    EnvironmentConfigurationError,
    MissingDependencyError,
    ModelInputError,
    ModelRuntimeError,
)
from inference_models.models.base.types import PreprocessedInputs
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
        message="Running L2CS model with ONNX backend requires pycuda installation, which is brought with "
                "`onnx-*` extras of `inference-models` library. If you see this error running locally, "
                "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/"
                " If you see this error using Roboflow infrastructure, make sure the service you use does support the "
                f"model, You can also contact Roboflow to get support."
                "Additionally - if AutoModel.from_pretrained(...) "
                f"automatically selects model package which does not match your environment - that's a serious problem and "
                f"we will really appreciate letting us know - https://github.com/roboflow/inference/issues",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error



DEFAULT_GAZE_MAX_BATCH_SIZE = 8


@dataclass
class L2CSGazeDetection:
    yaw: torch.Tensor
    pitch: torch.Tensor


class L2CSNetOnnx:

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
        default_onnx_trt_options: bool = True,
        device: torch.device = DEFAULT_DEVICE,
        max_batch_size: int = DEFAULT_GAZE_MAX_BATCH_SIZE,
        **kwargs,
    ):
        if onnx_execution_providers is None:
            onnx_execution_providers = get_selected_onnx_execution_providers()
        if not onnx_execution_providers:
            raise EnvironmentConfigurationError(
                message=f"Could not initialize model - selected backend is ONNX which requires execution provider to "
                f"be specified - explicitly in `from_pretrained(...)` method or via env variable "
                f"`ONNXRUNTIME_EXECUTION_PROVIDERS`. If you run model locally - adjust your setup, otherwise "
                f"contact the platform support.",
                help_url="https://inference-models.roboflow.com/errors/runtime-environment/#environmentconfigurationerror",
            )
        onnx_execution_providers = set_onnx_execution_provider_defaults(
            providers=onnx_execution_providers,
            model_package_path=model_name_or_path,
            device=device,
            default_onnx_trt_options=default_onnx_trt_options,
        )
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=["weights.onnx"],
        )
        session = onnxruntime.InferenceSession(
            path_or_bytes=model_package_content["weights.onnx"],
            providers=onnx_execution_providers,
        )
        input_name = session.get_inputs()[0].name
        return cls(
            session=session,
            max_batch_size=max_batch_size,
            device=device,
            input_name=input_name,
        )

    def __init__(
        self,
        session: onnxruntime.InferenceSession,
        max_batch_size: int,
        device: torch.device,
        input_name: str,
    ):
        self._session = session
        self._max_batch_size = max_batch_size
        self._device = device
        self._input_name = input_name
        self._session_thread_lock = Lock()
        self._numpy_transformations = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize([448, 448]),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self._tensors_transformations = transforms.Compose(
            [
                lambda x: x / 255.0,
                transforms.Resize([448, 448]),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @property
    def device(self) -> torch.device:
        return self._device

    def infer(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> L2CSGazeDetection:
        pre_processed_images = self.pre_process(images, **kwargs)
        model_results = self.forward(pre_processed_images, **kwargs)
        return self.post_process(model_results, **kwargs)

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(images, np.ndarray):
            input_color_format = input_color_format or "bgr"
            if input_color_format != "rgb":
                images = np.ascontiguousarray(images[:, :, ::-1])
            pre_processed = self._numpy_transformations(images)
            return torch.unsqueeze(pre_processed, dim=0).to(self._device)
        if isinstance(images, torch.Tensor):
            input_color_format = input_color_format or "rgb"
            if len(images.shape) == 3:
                images = torch.unsqueeze(images, dim=0)
            images = images.to(self._device)
            if input_color_format != "rgb":
                images = images[:, [2, 1, 0], :, :]
            return self._tensors_transformations(images.float())
        if not isinstance(images, list):
            raise ModelInputError(
                message="Pre-processing supports only np.array or torch.Tensor or list of above.",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
        if not len(images):
            raise ModelInputError(
                message="Detected empty input to the model",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror"
            )
        if isinstance(images[0], np.ndarray):
            input_color_format = input_color_format or "bgr"
            pre_processed = []
            for image in images:
                if input_color_format != "rgb":
                    image = np.ascontiguousarray(image[:, :, ::-1])
                pre_processed.append(self._numpy_transformations(image))
            return torch.stack(pre_processed, dim=0).to(self._device)
        if isinstance(images[0], torch.Tensor):
            input_color_format = input_color_format or "rgb"
            pre_processed = []
            for image in images:
                if len(image.shape) == 3:
                    image = torch.unsqueeze(image, dim=0)
                if input_color_format != "rgb":
                    image = image[:, [2, 1, 0], :, :]
                pre_processed.append(self._tensors_transformations(image.float()))
            return torch.cat(pre_processed, dim=0).to(self._device)
        raise ModelInputError(
            message=f"Detected unknown input batch element: {type(images[0])}",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )

    def forward(
        self,
        pre_processed_images: PreprocessedInputs,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with self._session_thread_lock:
            yaw, pitch = run_onnx_session_with_batch_size_limit(
                session=self._session, inputs={self._input_name: pre_processed_images}
            )
            return yaw, pitch

    def post_process(
        self,
        model_results: Tuple[torch.Tensor, torch.Tensor],
        **kwargs,
    ) -> L2CSGazeDetection:
        return L2CSGazeDetection(yaw=model_results[0], pitch=model_results[1])

    def __call__(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> L2CSGazeDetection:
        return self.infer(images, **kwargs)
