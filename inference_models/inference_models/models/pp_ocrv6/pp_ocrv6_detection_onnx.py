from threading import Lock
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.developer_tools import (
    align_device_with_onnx_session,
    run_onnx_session_via_iobinding,
    set_onnx_execution_provider_defaults,
)
from inference_models.entities import ColorFormat
from inference_models.errors import EnvironmentConfigurationError
from inference_models.models.base.object_detection import (
    Detections,
    ObjectDetectionModel,
)
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.pp_ocrv6.pp_ocrv6_common import (
    PreProcessingMetadata,
    is_torch_input,
    normalize_input_images,
    normalize_torch_images_to_device,
)
from inference_models.models.pp_ocrv6.pp_ocrv6_detection_utils import (
    DBNetConfig,
    boxes_from_probability_map,
    load_detection_config,
    normalize_detection_image,
    normalize_detection_image_torch,
    resize_for_detection,
    resize_for_detection_torch,
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
    ObjectDetectionModel[
        List[torch.Tensor], List[PreProcessingMetadata], List[torch.Tensor]
    ]
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
        device = align_device_with_onnx_session(session=session, device=device)
        return cls(
            session=session,
            input_name=session.get_inputs()[0].name,
            config=config,
            device=device,
        )

    def __init__(
        self,
        session: onnxruntime.InferenceSession,
        input_name: str,
        config: DBNetConfig,
        device: torch.device,
    ):
        self._session = session
        self._input_name = input_name
        self._config = config
        self._device = device
        self._session_thread_lock = Lock()

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def class_names(self) -> List[str]:
        return [TEXT_CLASS_NAME]

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> Tuple[List[torch.Tensor], List[PreProcessingMetadata]]:
        # torch.Tensor inputs are pre-processed on their own device (resize +
        # normalize run with torch ops), so they are never copied out to numpy
        # and back. numpy inputs keep the cv2 pipeline.
        if is_torch_input(images):
            return self._pre_process_torch(images, input_color_format)
        return self._pre_process_numpy(images, input_color_format)

    @property
    def _model_color_format(self) -> ColorFormat:
        return "rgb" if self._config.to_rgb else "bgr"

    def _pre_process_numpy(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat],
    ) -> Tuple[List[torch.Tensor], List[PreProcessingMetadata]]:
        images = normalize_input_images(
            images=images,
            input_color_format=input_color_format,
            target_color_format=self._model_color_format,
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
            normalized = normalize_detection_image(image=resized, config=self._config)
            pre_processed_images.append(torch.from_numpy(normalized).to(self._device))
            pre_processing_meta.append(
                PreProcessingMetadata(
                    source_height=source_height, source_width=source_width
                )
            )
        return pre_processed_images, pre_processing_meta

    def _pre_process_torch(
        self,
        images: Union[torch.Tensor, List[torch.Tensor]],
        input_color_format: Optional[ColorFormat],
    ) -> Tuple[List[torch.Tensor], List[PreProcessingMetadata]]:
        device_images = normalize_torch_images_to_device(
            images=images,
            input_color_format=input_color_format,
            device=self._device,
            target_color_format=self._model_color_format,
        )
        pre_processed_images = []
        pre_processing_meta = []
        for image in device_images:
            source_height, source_width = int(image.shape[1]), int(image.shape[2])
            resized = resize_for_detection_torch(
                image=image,
                limit_side_len=self._config.limit_side_len,
                limit_type=self._config.limit_type,
            )
            normalized = normalize_detection_image_torch(
                image=resized, config=self._config
            )
            pre_processed_images.append(normalized)
            pre_processing_meta.append(
                PreProcessingMetadata(
                    source_height=source_height, source_width=source_width
                )
            )
        return pre_processed_images, pre_processing_meta

    def forward(
        self, pre_processed_images: List[torch.Tensor], **kwargs
    ) -> List[torch.Tensor]:
        # Detection inputs have per-image spatial shapes (each image is resized
        # to its own multiple-of-32 size), so images run through the session one
        # by one instead of as a single stacked batch.
        probability_maps = []
        with self._session_thread_lock:
            for pre_processed_image in pre_processed_images:
                output = run_onnx_session_via_iobinding(
                    session=self._session,
                    inputs={self._input_name: pre_processed_image},
                )[0]
                probability_maps.append(output[0, 0])
        return probability_maps

    def post_process(
        self,
        model_results: List[torch.Tensor],
        pre_processing_meta: List[PreProcessingMetadata],
        **kwargs,
    ) -> List[Detections]:
        detections = []
        for probability_map, meta in zip(model_results, pre_processing_meta):
            quads_with_scores = boxes_from_probability_map(
                probability_map=probability_map.detach().cpu().numpy(),
                source_height=meta.source_height,
                source_width=meta.source_width,
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
