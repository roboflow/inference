from threading import Lock
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.entities import ColorFormat
from inference_models.errors import EnvironmentConfigurationError, ModelInputError
from inference_models.models.base.documents_parsing import TextOnlyOCRModel
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.onnx import set_onnx_execution_provider_defaults
from inference_models.models.pp_ocrv6.pp_ocrv6_recognition_utils import (
    ctc_decode,
    load_inference_config,
    preprocess_text_lines,
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


class PPOCRv6RecognitionOnnx(TextOnlyOCRModel[np.ndarray, np.ndarray]):
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
        return cls(
            session=session,
            input_name=session.get_inputs()[0].name,
            image_shape=image_shape,
            characters=characters,
        )

    def __init__(
        self,
        session: onnxruntime.InferenceSession,
        input_name: str,
        image_shape: Tuple[int, int, int],
        characters: List[str],
    ):
        self._session = session
        self._input_name = input_name
        self._input_channels, self._input_height, self._input_width = image_shape
        self._characters = characters
        self._session_thread_lock = Lock()

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> np.ndarray:
        images = normalize_input_images(
            images=images,
            input_color_format=input_color_format,
        )
        return preprocess_text_lines(
            images=images,
            target_height=self._input_height,
            min_width=self._input_width,
        )

    def forward(self, pre_processed_images: np.ndarray, **kwargs) -> np.ndarray:
        with self._session_thread_lock:
            return self._session.run(
                None,
                {self._input_name: pre_processed_images},
            )[0]

    def post_process(self, model_results: np.ndarray, **kwargs) -> List[str]:
        return [
            text
            for text, _ in ctc_decode(
                predictions=model_results,
                characters=self._characters,
            )
        ]


def normalize_input_images(
    images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
    input_color_format: Optional[ColorFormat] = None,
) -> List[np.ndarray]:
    if isinstance(images, np.ndarray):
        input_color_format = input_color_format or "bgr"
        return [
            convert_numpy_image_to_bgr(image=images, color_format=input_color_format)
        ]
    if isinstance(images, torch.Tensor):
        input_color_format = input_color_format or "rgb"
        return [
            convert_numpy_image_to_bgr(
                image=image,
                color_format=input_color_format,
            )
            for image in torch_images_to_numpy_list(images=images)
        ]
    if not isinstance(images, list):
        raise ModelInputError(
            message="PP-OCRv6 recognition supports np.ndarray, torch.Tensor, or lists of those inputs.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    if not images:
        raise ModelInputError(
            message="Detected empty input to PP-OCRv6 recognition model.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    if isinstance(images[0], np.ndarray):
        input_color_format = input_color_format or "bgr"
        return [
            convert_numpy_image_to_bgr(image=image, color_format=input_color_format)
            for image in images
        ]
    if isinstance(images[0], torch.Tensor):
        input_color_format = input_color_format or "rgb"
        return [
            convert_numpy_image_to_bgr(image=image, color_format=input_color_format)
            for image in torch_images_to_numpy_list(images=torch.stack(images, dim=0))
        ]
    raise ModelInputError(
        message=f"Detected unsupported PP-OCRv6 recognition input type: {type(images[0])}",
        help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
    )


def torch_images_to_numpy_list(images: torch.Tensor) -> List[np.ndarray]:
    if len(images.shape) == 3:
        images = torch.unsqueeze(images, dim=0)
    if len(images.shape) != 4:
        raise ModelInputError(
            message=(
                "PP-OCRv6 recognition expects torch images in CHW or BCHW format."
            ),
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    return [image.permute(1, 2, 0).detach().cpu().numpy() for image in images]


def convert_numpy_image_to_bgr(
    image: np.ndarray, color_format: ColorFormat
) -> np.ndarray:
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ModelInputError(
            message="PP-OCRv6 recognition expects images with 3 color channels.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    if color_format == "rgb":
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image)
