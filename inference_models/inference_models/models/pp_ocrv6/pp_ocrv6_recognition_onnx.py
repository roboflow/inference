from threading import Lock
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.entities import ColorFormat
from inference_models.errors import EnvironmentConfigurationError
from inference_models.models.base.documents_parsing import TextOnlyOCRModel
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.onnx import (
    align_cuda_device_with_onnx_session,
    run_onnx_session_with_batch_size_limit,
    set_onnx_execution_provider_defaults,
)
from inference_models.models.pp_ocrv6.pp_ocrv6_common import (
    is_torch_input,
    normalize_input_images,
    normalize_torch_images_to_device,
)
from inference_models.models.pp_ocrv6.pp_ocrv6_recognition_utils import (
    ctc_decode_indices,
    load_inference_config,
    preprocess_text_lines,
    preprocess_text_lines_torch,
)
from inference_models.utils.onnx_introspection import (
    get_selected_onnx_execution_providers,
)

try:
    import onnxruntime
except ImportError as import_error:
    from inference_models.errors import MissingDependencyError

    raise MissingDependencyError(
        message=(
            "Running PP-OCRv6 recognition with ONNX backend requires "
            "onnxruntime installation, which is brought with `onnx-*` extras of "
            "`inference-models` library."
        ),
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error


PP_OCRV6_RECOGNITION_MODEL_FILE = "inference.onnx"
PP_OCRV6_RECOGNITION_CONFIG_FILE = "inference.yml"
DEFAULT_RECOGNITION_MAX_BATCH_SIZE = 8


class PPOCRv6RecognitionOnnx(TextOnlyOCRModel[torch.Tensor, torch.Tensor]):
    """PP-OCRv6 text recognition model.

    This model runs recognition only. Inputs should be cropped text-line images.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
        default_onnx_trt_options: bool = True,
        device: torch.device = DEFAULT_DEVICE,
        max_batch_size: int = DEFAULT_RECOGNITION_MAX_BATCH_SIZE,
        **kwargs,
    ) -> "TextOnlyOCRModel":
        if onnx_execution_providers is None:
            onnx_execution_providers = get_selected_onnx_execution_providers()
        if not onnx_execution_providers:
            raise EnvironmentConfigurationError(
                message=(
                    "Could not initialize PP-OCRv6 recognition model - ONNX backend "
                    "requires an execution provider, either explicitly in "
                    "`from_pretrained(...)` or via `ONNXRUNTIME_EXECUTION_PROVIDERS`."
                ),
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
            elements=[
                PP_OCRV6_RECOGNITION_MODEL_FILE,
                PP_OCRV6_RECOGNITION_CONFIG_FILE,
            ],
        )
        image_shape, characters = load_inference_config(
            config_path=model_package_content[PP_OCRV6_RECOGNITION_CONFIG_FILE]
        )
        session = onnxruntime.InferenceSession(
            path_or_bytes=model_package_content[PP_OCRV6_RECOGNITION_MODEL_FILE],
            providers=onnx_execution_providers,
        )
        device = align_cuda_device_with_onnx_session(session=session, device=device)
        return cls(
            session=session,
            input_name=session.get_inputs()[0].name,
            image_shape=image_shape,
            characters=characters,
            device=device,
            max_batch_size=max_batch_size,
        )

    def __init__(
        self,
        session: onnxruntime.InferenceSession,
        input_name: str,
        image_shape: Tuple[int, int, int],
        characters: List[str],
        device: torch.device,
        max_batch_size: int = DEFAULT_RECOGNITION_MAX_BATCH_SIZE,
    ):
        self._session = session
        self._input_name = input_name
        self._input_channels, self._input_height, self._input_width = image_shape
        self._characters = characters
        self._device = device
        self._max_batch_size = max_batch_size
        self._session_thread_lock = Lock()

    @property
    def device(self) -> torch.device:
        return self._device

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> torch.Tensor:
        # torch.Tensor inputs are pre-processed on their own device (resize +
        # normalize + pad run with torch ops), so they are never copied out to
        # numpy and back. numpy inputs keep the cv2 pipeline.
        if is_torch_input(images):
            device_images = normalize_torch_images_to_device(
                images=images,
                input_color_format=input_color_format,
                device=self._device,
            )
            return preprocess_text_lines_torch(
                images=device_images,
                target_height=self._input_height,
                min_width=self._input_width,
            )
        images = normalize_input_images(
            images=images,
            input_color_format=input_color_format,
        )
        pre_processed_images = preprocess_text_lines(
            images=images,
            target_height=self._input_height,
            min_width=self._input_width,
        )
        return torch.from_numpy(pre_processed_images).to(self._device)

    def forward(self, pre_processed_images: torch.Tensor, **kwargs) -> torch.Tensor:
        with self._session_thread_lock:
            return run_onnx_session_with_batch_size_limit(
                session=self._session,
                inputs={self._input_name: pre_processed_images},
                max_batch_size=self._max_batch_size,
            )[0]

    def post_process(self, model_results: torch.Tensor, **kwargs) -> List[str]:
        probs, indices = torch.max(model_results, dim=2)
        return [
            text
            for text, _ in ctc_decode_indices(
                indices=indices.detach().cpu().numpy(),
                probs=probs.detach().cpu().numpy(),
                characters=self._characters,
            )
        ]
