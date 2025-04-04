from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from dataclasses_json import DataClassJsonMixin
from PIL import Image

from inference_sdk.http.errors import ModelTaskTypeNotSupportedError
from inference_sdk.http.utils.iterables import remove_empty_values

ImagesReference = Union[np.ndarray, Image.Image, str]

DEFAULT_IMAGE_EXTENSIONS = ["jpg", "jpeg", "JPG", "JPEG", "png", "PNG"]

TaskType = str
CLASSIFICATION_TASK = "classification"
OBJECT_DETECTION_TASK = "object-detection"
INSTANCE_SEGMENTATION_TASK = "instance-segmentation"
KEYPOINTS_DETECTION_TASK = "keypoint-detection"
DEFAULT_MAX_INPUT_SIZE = 1024

ALL_ROBOFLOW_API_URLS = {
    "https://detect.roboflow.com",
    "https://outline.roboflow.com",
    "https://classify.roboflow.com",
    "https://infer.roboflow.com",
    "https://serverless.roboflow.com",
    "https://serverless.roboflow.one",
}


@dataclass(frozen=True)
class ServerInfo(DataClassJsonMixin):
    """Dataclass for Information about the inference server.

    Attributes:
        name: The name of the inference server.
        version: The version of the inference server.
        uuid: The unique identifier of the inference server instance.
    """

    name: str
    version: str
    uuid: str


@dataclass(frozen=True)
class ModelDescription(DataClassJsonMixin):
    """Dataclass for model description.

    Attributes:
        model_id: The unique identifier of the model.
        task_type: The type of task the model is designed for.
        batch_size: The batch size for the model.
        input_height: The height of the input image.
        input_width: The width of the input image.
    """

    model_id: str
    task_type: TaskType
    batch_size: Optional[Union[int, str]] = None
    input_height: Optional[int] = None
    input_width: Optional[int] = None


@dataclass(frozen=True)
class RegisteredModels(DataClassJsonMixin):
    """Dataclass for registered models.

    Attributes:
        models: A list of model descriptions.
    """

    models: List[ModelDescription]


class HTTPClientMode(str, Enum):
    """Enum for the HTTP client mode.

    Attributes:
        V0: The version 0 of the HTTP client.
        V1: The version 1 of the HTTP client.
    """

    V0 = "v0"
    V1 = "v1"


class VisualisationResponseFormat(str, Enum):
    """Enum for the visualisation response format.

    Attributes:
        BASE64: The base64 format.
        NUMPY: The numpy format.
        PILLOW: The pillow format.
    """

    BASE64 = "base64"
    NUMPY = "numpy"
    PILLOW = "pillow"


@dataclass(frozen=True)
class InferenceConfiguration:
    """Dataclass for inference configuration.

    Attributes:
        confidence_threshold: The confidence threshold for the inference.
        keypoint_confidence_threshold: The keypoint confidence threshold for the inference.
        format: The format for the inference.
        mask_decode_mode: The mask decode mode for the inference.
        tradeoff_factor: The tradeoff factor for the inference.
        max_candidates: The maximum number of candidates for the inference.
        max_detections: The maximum number of detections for the inference.
        iou_threshold: The intersection over union threshold for the inference.
        stroke_width: The stroke width for the inference.
    """

    confidence_threshold: Optional[float] = None
    keypoint_confidence_threshold: Optional[float] = None
    format: Optional[str] = None
    mask_decode_mode: Optional[str] = None
    tradeoff_factor: Optional[float] = None
    max_candidates: Optional[int] = None
    max_detections: Optional[int] = None
    iou_threshold: Optional[float] = None
    stroke_width: Optional[int] = None
    count_inference: Optional[bool] = None
    service_secret: Optional[str] = None
    disable_preproc_auto_orientation: Optional[bool] = None
    disable_preproc_contrast: Optional[bool] = None
    disable_preproc_grayscale: Optional[bool] = None
    disable_preproc_static_crop: Optional[bool] = None
    class_agnostic_nms: Optional[bool] = None
    class_filter: Optional[List[str]] = None
    fix_batch_size: Optional[bool] = None
    visualize_predictions: bool = False
    visualize_labels: Optional[bool] = None
    output_visualisation_format: VisualisationResponseFormat = (
        VisualisationResponseFormat.BASE64
    )
    image_extensions_for_directory_scan: Optional[List[str]] = field(
        default_factory=lambda: DEFAULT_IMAGE_EXTENSIONS,
    )
    client_downsizing_disabled: bool = True
    default_max_input_size: int = DEFAULT_MAX_INPUT_SIZE
    disable_active_learning: bool = False
    active_learning_target_dataset: Optional[str] = None
    max_concurrent_requests: int = 1
    max_batch_size: int = 1
    source: Optional[str] = None
    source_info: Optional[str] = None
    profiling_directory: str = "./inference_profiling"

    @classmethod
    def init_default(cls) -> "InferenceConfiguration":
        return cls()

    def to_api_call_parameters(
        self, client_mode: HTTPClientMode, task_type: TaskType
    ) -> Dict[str, Any]:
        """Convert the current configuration to API call parameters.

        Args:
            client_mode: The HTTP client mode.
            task_type: The type of task the model is designed for.

        Returns:
            Dict[str, Any]: The API call parameters.
        """
        if client_mode is HTTPClientMode.V0:
            return self.to_legacy_call_parameters()
        if task_type == OBJECT_DETECTION_TASK:
            return self.to_object_detection_parameters()
        if task_type == INSTANCE_SEGMENTATION_TASK:
            return self.to_instance_segmentation_parameters()
        if task_type == CLASSIFICATION_TASK:
            return self.to_classification_parameters()
        if task_type == KEYPOINTS_DETECTION_TASK:
            return self.to_keypoints_detection_parameters()
        raise ModelTaskTypeNotSupportedError(
            f"Model task {task_type} is not supported by API v1 client."
        )

    def to_object_detection_parameters(self) -> Dict[str, Any]:
        """Convert the current configuration to object detection parameters.

        Returns:
            Dict[str, Any]: The object detection parameters.
        """
        parameters_specs = [
            ("disable_preproc_auto_orientation", "disable_preproc_auto_orient"),
            ("disable_preproc_contrast", "disable_preproc_contrast"),
            ("disable_preproc_grayscale", "disable_preproc_grayscale"),
            ("disable_preproc_static_crop", "disable_preproc_static_crop"),
            ("class_agnostic_nms", "class_agnostic_nms"),
            ("class_filter", "class_filter"),
            ("confidence_threshold", "confidence"),
            ("fix_batch_size", "fix_batch_size"),
            ("iou_threshold", "iou_threshold"),
            ("max_detections", "max_detections"),
            ("max_candidates", "max_candidates"),
            ("visualize_labels", "visualization_labels"),
            ("stroke_width", "visualization_stroke_width"),
            ("visualize_predictions", "visualize_predictions"),
            ("disable_active_learning", "disable_active_learning"),
            ("active_learning_target_dataset", "active_learning_target_dataset"),
            ("source", "source"),
            ("source_info", "source_info"),
        ]
        return get_non_empty_attributes(
            source_object=self,
            specification=parameters_specs,
        )

    def to_keypoints_detection_parameters(self) -> Dict[str, Any]:
        """Convert the current configuration to keypoints detection parameters.

        Returns:
            Dict[str, Any]: The keypoints detection parameters.
        """
        parameters = self.to_object_detection_parameters()
        parameters["keypoint_confidence"] = self.keypoint_confidence_threshold
        return remove_empty_values(dictionary=parameters)

    def to_instance_segmentation_parameters(self) -> Dict[str, Any]:
        """Convert the current configuration to instance segmentation parameters.

        Returns:
            Dict[str, Any]: The instance segmentation parameters.
        """
        parameters = self.to_object_detection_parameters()
        parameters_specs = [
            ("mask_decode_mode", "mask_decode_mode"),
            ("tradeoff_factor", "tradeoff_factor"),
        ]
        for internal_name, external_name in parameters_specs:
            parameters[external_name] = getattr(self, internal_name)
        return remove_empty_values(dictionary=parameters)

    def to_classification_parameters(self) -> Dict[str, Any]:
        """Convert the current configuration to classification parameters.

        Returns:
            Dict[str, Any]: The classification parameters.
        """
        parameters_specs = [
            ("disable_preproc_auto_orientation", "disable_preproc_auto_orient"),
            ("disable_preproc_contrast", "disable_preproc_contrast"),
            ("disable_preproc_grayscale", "disable_preproc_grayscale"),
            ("disable_preproc_static_crop", "disable_preproc_static_crop"),
            ("confidence_threshold", "confidence"),
            ("visualize_predictions", "visualize_predictions"),
            ("stroke_width", "visualization_stroke_width"),
            ("disable_active_learning", "disable_active_learning"),
            ("source", "source"),
            ("source_info", "source_info"),
            ("active_learning_target_dataset", "active_learning_target_dataset"),
        ]
        return get_non_empty_attributes(
            source_object=self,
            specification=parameters_specs,
        )

    def to_legacy_call_parameters(self) -> Dict[str, Any]:
        """Convert the current configuration to legacy call parameters.

        Returns:
            Dict[str, Any]: The legacy call parameters.
        """
        parameters_specs = [
            ("confidence_threshold", "confidence"),
            ("keypoint_confidence_threshold", "keypoint_confidence"),
            ("format", "format"),
            ("visualize_labels", "labels"),
            ("mask_decode_mode", "mask_decode_mode"),
            ("tradeoff_factor", "tradeoff_factor"),
            ("max_detections", "max_detections"),
            ("iou_threshold", "overlap"),
            ("stroke_width", "stroke"),
            ("count_inference", "countinference"),
            ("service_secret", "service_secret"),
            ("disable_preproc_auto_orientation", "disable_preproc_auto_orient"),
            ("disable_preproc_contrast", "disable_preproc_contrast"),
            ("disable_preproc_grayscale", "disable_preproc_grayscale"),
            ("disable_preproc_static_crop", "disable_preproc_static_crop"),
            ("disable_active_learning", "disable_active_learning"),
            ("active_learning_target_dataset", "active_learning_target_dataset"),
            ("source", "source"),
            ("source_info", "source_info"),
        ]
        return get_non_empty_attributes(
            source_object=self,
            specification=parameters_specs,
        )


def get_non_empty_attributes(
    source_object: object, specification: List[Tuple[str, str]]
) -> Dict[str, Any]:
    """Get non-empty attributes from the source object.

    Args:
        source_object: The source object.
        specification: The specification of the attributes.

    Returns:
        Dict[str, Any]: The non-empty attributes.
    """
    attributes = {
        external_name: getattr(source_object, internal_name)
        for internal_name, external_name in specification
    }
    return remove_empty_values(dictionary=attributes)
