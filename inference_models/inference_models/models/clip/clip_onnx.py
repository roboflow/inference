from threading import Lock
from typing import List, Optional, Union

import clip
import numpy as np
import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.entities import ColorFormat
from inference_models.errors import (
    EnvironmentConfigurationError,
    MissingDependencyError,
)
from inference_models.models.base.embeddings import TextImageEmbeddingModel
from inference_models.models.clip.preprocessing import create_clip_preprocessor
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
        message=f"Could not import CLIP model with ONNX backend - this error means that some additional dependencies "
        f"are not installed in the environment. If you run the `inference-models` library directly in your Python "
        f"program, make sure the following extras of the package are installed: \n"
        f"\t* `onnx-cpu` - when you wish to use library with CPU support only\n"
        f"\t* `onnx-cu12` - for running on GPU with Cuda 12 installed\n"
        f"\t* `onnx-cu118` - for running on GPU with Cuda 11.8 installed\n"
        f"\t* `onnx-jp6-cu126` - for running on Jetson with Jetpack 6\n"
        f"If you see this error using Roboflow infrastructure, make sure the service you use does support the model. "
        f"You can also contact Roboflow to get support.",
        help_url="https://todo",
    ) from import_error


MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


class ClipOnnx(TextImageEmbeddingModel):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
        default_onnx_trt_options: bool = True,
        device: torch.device = DEFAULT_DEVICE,
        max_batch_size: int = 32,
        **kwargs,
    ) -> "ClipOnnx":
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
        onnx_execution_providers = set_onnx_execution_provider_defaults(
            providers=onnx_execution_providers,
            model_package_path=model_name_or_path,
            device=device,
            default_onnx_trt_options=default_onnx_trt_options,
        )
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "textual.onnx",
                "visual.onnx",
            ],
        )
        visual_onnx_session = onnxruntime.InferenceSession(
            path_or_bytes=model_package_content["visual.onnx"],
            providers=onnx_execution_providers,
        )
        textual_onnx_session = onnxruntime.InferenceSession(
            path_or_bytes=model_package_content["textual.onnx"],
            providers=onnx_execution_providers,
        )
        image_size = visual_onnx_session.get_inputs()[0].shape[2]
        visual_input_name = visual_onnx_session.get_inputs()[0].name
        textual_input_name = textual_onnx_session.get_inputs()[0].name
        return cls(
            visual_onnx_session=visual_onnx_session,
            textual_onnx_session=textual_onnx_session,
            image_size=image_size,
            visual_input_name=visual_input_name,
            textual_input_name=textual_input_name,
            device=device,
            max_batch_size=max_batch_size,
        )

    def __init__(
        self,
        visual_onnx_session: onnxruntime.InferenceSession,
        textual_onnx_session: onnxruntime.InferenceSession,
        image_size: int,
        visual_input_name: str,
        textual_input_name: str,
        device: torch.device,
        max_batch_size: int,
    ):
        self._visual_onnx_session = visual_onnx_session
        self._textual_onnx_session = textual_onnx_session
        self._image_size = image_size
        self._visual_input_name = visual_input_name
        self._textual_input_name = textual_input_name
        self._device = device
        self._max_batch_size = max_batch_size
        self._visual_session_thread_lock = Lock()
        self._textual_session_thread_lock = Lock()
        self._preprocessor = create_clip_preprocessor(image_size=image_size)

    def embed_images(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> torch.Tensor:
        pre_processed_images = self._preprocessor(
            images, input_color_format, self._device
        )
        with self._visual_session_thread_lock:
            return run_onnx_session_with_batch_size_limit(
                session=self._visual_onnx_session,
                inputs={self._visual_input_name: pre_processed_images},
                max_batch_size=self._max_batch_size,
            )[0]

    def embed_text(self, texts: Union[str, List[str]], **kwargs) -> torch.Tensor:
        if not isinstance(texts, list):
            texts = [texts]
        tokenized_batch = clip.tokenize(texts)
        with self._textual_session_thread_lock:
            return run_onnx_session_with_batch_size_limit(
                session=self._textual_onnx_session,
                inputs={self._textual_input_name: tokenized_batch},
                max_batch_size=self._max_batch_size,
            )[0]
