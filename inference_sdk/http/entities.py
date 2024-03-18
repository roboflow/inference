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
}


@dataclass(frozen=True)
class ServerInfo(DataClassJsonMixin):
    name: str
    version: str
    uuid: str


@dataclass(frozen=True)
class ModelDescription(DataClassJsonMixin):
    model_id: str
    task_type: TaskType
    batch_size: Optional[Union[int, str]] = None
    input_height: Optional[int] = None
    input_width: Optional[int] = None


@dataclass(frozen=True)
class RegisteredModels(DataClassJsonMixin):
    models: List[ModelDescription]


class HTTPClientMode(str, Enum):
    V0 = "v0"
    V1 = "v1"


class VisualisationResponseFormat(str, Enum):
    BASE64 = "base64"
    NUMPY = "numpy"
    PILLOW = "pillow"


@dataclass(frozen=True)
class InferenceConfiguration:
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
    client_downsizing_disabled: bool = False
    default_max_input_size: int = DEFAULT_MAX_INPUT_SIZE
    disable_active_learning: bool = False
    active_learning_target_dataset: Optional[str] = None
    max_concurrent_requests: int = 1
    max_batch_size: int = 1
    source: Optional[str] = None
    source_info: Optional[str] = None

    @classmethod
    def init_default(cls) -> "InferenceConfiguration":
        return cls()

    def to_api_call_parameters(
        self, client_mode: HTTPClientMode, task_type: TaskType
    ) -> Dict[str, Any]:
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
        parameters = self.to_object_detection_parameters()
        parameters["keypoint_confidence"] = self.keypoint_confidence_threshold
        return remove_empty_values(dictionary=parameters)

    def to_instance_segmentation_parameters(self) -> Dict[str, Any]:
        parameters = self.to_object_detection_parameters()
        parameters_specs = [
            ("mask_decode_mode", "mask_decode_mode"),
            ("tradeoff_factor", "tradeoff_factor"),
        ]
        for internal_name, external_name in parameters_specs:
            parameters[external_name] = getattr(self, internal_name)
        return remove_empty_values(dictionary=parameters)

    def to_classification_parameters(self) -> Dict[str, Any]:
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
    attributes = {
        external_name: getattr(source_object, internal_name)
        for internal_name, external_name in specification
    }
    return remove_empty_values(dictionary=attributes)
