from dataclasses import dataclass, replace
from enum import Enum
from typing import List, Optional, Union, Dict, Any, Tuple

import numpy as np
from PIL import Image

from dataclasses_json import DataClassJsonMixin

from clients.http.utils.iterables import remove_empty_values

ImagesReference = Union[np.ndarray, Image.Image, str]


@dataclass(frozen=True)
class ServerInfo(DataClassJsonMixin):
    name: str
    version: str
    uuid: str


@dataclass(frozen=True)
class RegisteredModels(DataClassJsonMixin):
    model_ids: List[str]

class HTTPClientMode(Enum):
    LEGACY = "legacy"
    NEW = "new"


class ModelType(Enum):
    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object_detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"


class VisualisationResponseFormat:
    BASE64 = "base64"
    NUMPY = "numpy"
    PILLOW = "pillow"


@dataclass(frozen=True)
class InferenceConfiguration:
    # Here we could add methods like with_confidence_threshold(...) - nice but a lot of code
    confidence_threshold: Optional[float] = None
    format: Optional[str] = None
    show_labels: Optional[bool] = None
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
    visualize_predictions: Optional[bool] = None
    visualize_labels: Optional[bool] = None
    output_visualisation_format: VisualisationResponseFormat = VisualisationResponseFormat.BASE64

    @classmethod
    def init_default(cls) -> "InferenceConfiguration":
        return cls()

    def to_api_call_parameters(self, client_mode: HTTPClientMode, model_type: ModelType) -> Dict[str, Any]:
        if client_mode is HTTPClientMode.LEGACY:
            return self.to_legacy_call_parameters()
        if model_type is ModelType.OBJECT_DETECTION:
            return self.to_object_detection_parameters()
        if model_type is ModelType.INSTANCE_SEGMENTATION:
            return self.to_instance_segmentation_parameters()
        return self.to_classification_parameters()

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
        ]
        return get_non_empty_attributes(
            source_object=self,
            specification=parameters_specs,
        )

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
        ]
        return get_non_empty_attributes(
            source_object=self,
            specification=parameters_specs,
        )

    def to_legacy_call_parameters(self) -> Dict[str, Any]:
        parameters_specs = [
            ("confidence_threshold", "confidence"),
            ("format", "format"),
            ("show_labels", "labels"),
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
        ]
        return get_non_empty_attributes(
            source_object=self,
            specification=parameters_specs,
        )


def get_non_empty_attributes(
    source_object: object,
    specification: List[Tuple[str, str]]
) -> Dict[str, Any]:
    attributes = {
        external_name: getattr(source_object, internal_name)
        for internal_name, external_name in specification
    }
    return remove_empty_values(dictionary=attributes)
