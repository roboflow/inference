from dataclasses import dataclass
from threading import Lock
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.entities import ColorFormat
from inference_exp.errors import (
    EnvironmentConfigurationError,
    MissingDependencyError,
    ModelRuntimeError,
)
from inference_exp.models.base.types import PreprocessedInputs
from inference_exp.models.common.model_packages import get_model_package_contents
from inference_exp.models.common.onnx import (
    run_session_via_iobinding,
    set_execution_provider_defaults,
)
from inference_exp.utils.onnx_introspection import get_selected_onnx_execution_providers
from torchvision import transforms

try:
    import onnxruntime
except ImportError as import_error:
    raise MissingDependencyError(
        message=f"Could not import L2CS model with ONNX backend - this error means that some additional dependencies "
        f"are not installed in the environment. If you run the `inference` library directly in your Python "
        f"program, make sure the following extras of the package are installed: \n"
        f"\t* `onnx-cpu` - when you wish to use library with CPU support only\n"
        f"\t* `onnx-cu12` - for running on GPU with Cuda 12 installed\n"
        f"\t* `onnx-cu118` - for running on GPU with Cuda 11.8 installed\n"
        f"\t* `onnx-jp6-cu126` - for running on Jetson with Jetpack 6\n"
        f"If you see this error using Roboflow infrastructure, make sure the service you use does support the model. "
        f"You can also contact Roboflow to get support.",
        help_url="https://todo",
    ) from import_error


DEFAULT_GAZE_MAX_BATCH_SIZE = 8


@dataclass
class GazeDetection:
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
                help_url="https://todo",
            )
        onnx_execution_providers = set_execution_provider_defaults(
            providers=onnx_execution_providers,
            model_package_path=model_name_or_path,
            device=device,
            default_onnx_trt_options=default_onnx_trt_options,
        )
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=["L2CSNet_gaze360_resnet50_90bins.onnx"],
        )
        session = onnxruntime.InferenceSession(
            path_or_bytes=model_package_content["L2CSNet_gaze360_resnet50_90bins.onnx"],
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

    def infer(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> GazeDetection:
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
            raise ModelRuntimeError(
                message="Pre-processing supports only np.array or torch.Tensor or list of above.",
                help_url="https://todo",
            )
        if not len(images):
            raise ModelRuntimeError(
                message="Detected empty input to the model", help_url="https://todo"
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
        raise ModelRuntimeError(
            message=f"Detected unknown input batch element: {type(images[0])}",
            help_url="https://todo",
        )

    def forward(
        self,
        pre_processed_images: PreprocessedInputs,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with self._session_thread_lock:
            yaw, pitch = [], []
            for i in range(0, pre_processed_images.shape[0], self._max_batch_size):
                batch_input = pre_processed_images[
                    i : i + self._max_batch_size
                ].contiguous()
                batch_yaw, batch_pitch = run_session_via_iobinding(
                    session=self._session, inputs={self._input_name: batch_input}
                )
                yaw.append(batch_yaw)
                pitch.append(batch_pitch)
            return torch.cat(yaw, dim=0), torch.cat(pitch, dim=0)

    def post_process(
        self,
        model_results: Tuple[torch.Tensor, torch.Tensor],
        **kwargs,
    ) -> GazeDetection:
        return GazeDetection(yaw=model_results[0], pitch=model_results[1])

    def __call__(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> GazeDetection:
        return self.infer(images, **kwargs)
