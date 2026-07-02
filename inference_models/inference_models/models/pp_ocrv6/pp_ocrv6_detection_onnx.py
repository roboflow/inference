from threading import Lock
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.entities import ColorFormat
from inference_models.errors import EnvironmentConfigurationError, ModelInputError
from inference_models.models.base.object_detection import (
    Detections,
    ObjectDetectionModel,
)
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.onnx import set_onnx_execution_provider_defaults
from inference_models.models.pp_ocrv6.pp_ocrv6_detection_utils import (
    DBNetConfig,
    boxes_from_probability_map,
    load_detection_config,
    normalize_detection_image,
    resize_for_detection,
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
            "Running PP-OCRv6 detection with ONNX backend requires "
            "onnxruntime installation, which is brought with `onnx-*` extras of "
            "`inference-models` library."
        ),
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error


PP_OCRV6_DETECTION_MODEL_FILE = "inference.onnx"
PP_OCRV6_DETECTION_CONFIG_FILE = "inference.yml"
TEXT_CLASS_NAME = "text"


class PPOCRv6DetectionOnnx(
    ObjectDetectionModel[List[np.ndarray], List[dict], List[np.ndarray]]
):
    """PP-OCRv6 text detection model (DBNet).

    Detects text regions and returns axis-aligned bounding boxes. The tight
    four-point quadrilateral for each region is preserved in
    ``Detections.bboxes_metadata`` under the ``"polygon"`` key so downstream
    recognition can crop rotated text lines accurately.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
        default_onnx_trt_options: bool = True,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "ObjectDetectionModel":
        if onnx_execution_providers is None:
            onnx_execution_providers = get_selected_onnx_execution_providers()
        if not onnx_execution_providers:
            raise EnvironmentConfigurationError(
                message=(
                    "Could not initialize PP-OCRv6 detection model - ONNX backend "
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
                PP_OCRV6_DETECTION_MODEL_FILE,
                PP_OCRV6_DETECTION_CONFIG_FILE,
            ],
        )
        config = load_detection_config(
            config_path=model_package_content[PP_OCRV6_DETECTION_CONFIG_FILE]
        )
        session = onnxruntime.InferenceSession(
            path_or_bytes=model_package_content[PP_OCRV6_DETECTION_MODEL_FILE],
            providers=onnx_execution_providers,
        )
        return cls(
            session=session,
            input_name=session.get_inputs()[0].name,
            config=config,
        )

    def __init__(
        self,
        session: onnxruntime.InferenceSession,
        input_name: str,
        config: DBNetConfig,
    ):
        self._session = session
        self._input_name = input_name
        self._config = config
        self._session_thread_lock = Lock()

    @property
    def class_names(self) -> List[str]:
        return [TEXT_CLASS_NAME]

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> Tuple[List[np.ndarray], List[dict]]:
        images = normalize_input_images(
            images=images,
            input_color_format=input_color_format,
        )
        pre_processed_images = []
        pre_processing_meta = []
        for image in images:
            source_height, source_width = image.shape[:2]
            resized, _, _ = resize_for_detection(
                image=image,
                limit_side_len=self._config.limit_side_len,
                limit_type=self._config.limit_type,
            )
            pre_processed_images.append(
                normalize_detection_image(image_bgr=resized, config=self._config)
            )
            pre_processing_meta.append(
                {"source_height": source_height, "source_width": source_width}
            )
        return pre_processed_images, pre_processing_meta

    def forward(
        self, pre_processed_images: List[np.ndarray], **kwargs
    ) -> List[np.ndarray]:
        probability_maps = []
        with self._session_thread_lock:
            for pre_processed_image in pre_processed_images:
                output = self._session.run(
                    None,
                    {self._input_name: pre_processed_image},
                )[0]
                probability_maps.append(output[0, 0])
        return probability_maps

    def post_process(
        self,
        model_results: List[np.ndarray],
        pre_processing_meta: List[dict],
        **kwargs,
    ) -> List[Detections]:
        detections = []
        for probability_map, meta in zip(model_results, pre_processing_meta):
            quads_with_scores = boxes_from_probability_map(
                probability_map=probability_map,
                source_height=meta["source_height"],
                source_width=meta["source_width"],
                config=self._config,
            )
            detections.append(_quads_to_detections(quads_with_scores))
        return detections


def _quads_to_detections(
    quads_with_scores: List[Tuple[np.ndarray, float]],
) -> Detections:
    if not quads_with_scores:
        return Detections(
            xyxy=torch.zeros((0, 4), dtype=torch.float32),
            class_id=torch.zeros((0,), dtype=torch.int64),
            confidence=torch.zeros((0,), dtype=torch.float32),
            bboxes_metadata=[],
        )
    xyxy = []
    scores = []
    bboxes_metadata = []
    for quad, score in quads_with_scores:
        x_min, y_min = quad.min(axis=0)
        x_max, y_max = quad.max(axis=0)
        xyxy.append([float(x_min), float(y_min), float(x_max), float(y_max)])
        scores.append(score)
        bboxes_metadata.append({"polygon": quad.tolist()})
    return Detections(
        xyxy=torch.tensor(xyxy, dtype=torch.float32),
        class_id=torch.zeros((len(xyxy),), dtype=torch.int64),
        confidence=torch.tensor(scores, dtype=torch.float32),
        bboxes_metadata=bboxes_metadata,
    )


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
            convert_numpy_image_to_bgr(image=image, color_format=input_color_format)
            for image in torch_images_to_numpy_list(images=images)
        ]
    if not isinstance(images, list):
        raise ModelInputError(
            message="PP-OCRv6 detection supports np.ndarray, torch.Tensor, or lists of those inputs.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    if not images:
        raise ModelInputError(
            message="Detected empty input to PP-OCRv6 detection model.",
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
        message=f"Detected unsupported PP-OCRv6 detection input type: {type(images[0])}",
        help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
    )


def torch_images_to_numpy_list(images: torch.Tensor) -> List[np.ndarray]:
    if len(images.shape) == 3:
        images = torch.unsqueeze(images, dim=0)
    if len(images.shape) != 4:
        raise ModelInputError(
            message="PP-OCRv6 detection expects torch images in CHW or BCHW format.",
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
            message="PP-OCRv6 detection expects images with 3 color channels.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    if color_format == "rgb":
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image)
