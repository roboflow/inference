from dataclasses import dataclass
from typing import Dict, Optional, Union

from inference.core.env import (
    CLASS_AGNOSTIC_NMS_ENV,
    DEFAULT_CLASS_AGNOSTIC_NMS,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MAX_DETECTIONS,
    IOU_THRESHOLD_ENV,
    MAX_CANDIDATES_ENV,
    MAX_DETECTIONS_ENV,
)
from inference.core.utils.environment import safe_env_to_type, str2bool

ObjectDetectionPrediction = dict


@dataclass(frozen=True)
class ObjectDetectionInferenceConfig:
    class_agnostic_nms: Optional[bool]
    confidence: Optional[float]
    iou_threshold: Optional[float]
    max_candidates: Optional[int]
    max_detections: Optional[int]

    @classmethod
    def init(
        cls,
        class_agnostic_nms: Optional[bool] = None,
        confidence: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_candidates: Optional[int] = None,
        max_detections: Optional[int] = None,
    ) -> "ObjectDetectionInferenceConfig":
        if class_agnostic_nms is None:
            class_agnostic_nms = safe_env_to_type(
                variable_name=CLASS_AGNOSTIC_NMS_ENV,
                default_value=DEFAULT_CLASS_AGNOSTIC_NMS,
                type_constructor=str2bool,
            )
        if confidence is None:
            confidence = safe_env_to_type(
                variable_name=CLASS_AGNOSTIC_NMS_ENV,
                default_value=DEFAULT_CONFIDENCE,
                type_constructor=float,
            )
        if iou_threshold is None:
            iou_threshold = safe_env_to_type(
                variable_name=IOU_THRESHOLD_ENV,
                default_value=DEFAULT_IOU_THRESHOLD,
                type_constructor=float,
            )
        if max_candidates is None:
            max_candidates = safe_env_to_type(
                variable_name=MAX_CANDIDATES_ENV,
                default_value=DEFAULT_MAX_CANDIDATES,
                type_constructor=int,
            )
        if max_detections is None:
            max_detections = safe_env_to_type(
                variable_name=MAX_DETECTIONS_ENV,
                default_value=DEFAULT_MAX_DETECTIONS,
                type_constructor=int,
            )
        return ObjectDetectionInferenceConfig(
            class_agnostic_nms=class_agnostic_nms,
            confidence=confidence,
            iou_threshold=iou_threshold,
            max_candidates=max_candidates,
            max_detections=max_detections,
        )

    def to_postprocessing_params(self) -> Dict[str, Union[bool, float, int]]:
        result = {}
        for field in [
            "class_agnostic_nms",
            "confidence",
            "iou_threshold",
            "max_candidates",
            "max_detections",
        ]:
            result[field] = getattr(self, field, None)
        return {name: value for name, value in result.items() if value is not None}
