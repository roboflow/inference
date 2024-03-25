from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

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
from inference.core.interfaces.camera.entities import StatusUpdate, VideoFrame
from inference.core.interfaces.camera.video_source import SourceMetadata
from inference.core.utils.environment import safe_env_to_type, str2bool

AnyPrediction = Any
ObjectDetectionPrediction = dict


@dataclass(frozen=True)
class ModelConfig:
    class_agnostic_nms: Optional[bool]
    confidence: Optional[float]
    iou_threshold: Optional[float]
    max_candidates: Optional[int]
    max_detections: Optional[int]
    mask_decode_mode: Optional[str]
    tradeoff_factor: Optional[float]

    @classmethod
    def init(
        cls,
        class_agnostic_nms: Optional[bool] = None,
        confidence: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_candidates: Optional[int] = None,
        max_detections: Optional[int] = None,
        mask_decode_mode: Optional[str] = None,
        tradeoff_factor: Optional[float] = None,
    ) -> "ModelConfig":
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
        return ModelConfig(
            class_agnostic_nms=class_agnostic_nms,
            confidence=confidence,
            iou_threshold=iou_threshold,
            max_candidates=max_candidates,
            max_detections=max_detections,
            mask_decode_mode=mask_decode_mode,
            tradeoff_factor=tradeoff_factor,
        )

    def to_postprocessing_params(self) -> Dict[str, Union[bool, float, int]]:
        result = {}
        for field in [
            "class_agnostic_nms",
            "confidence",
            "iou_threshold",
            "max_candidates",
            "max_detections",
            "mask_decode_mode",
            "tradeoff_factor",
        ]:
            result[field] = getattr(self, field, None)
        return {name: value for name, value in result.items() if value is not None}


@dataclass(frozen=True)
class ModelActivityEvent:
    frame_decoding_timestamp: datetime
    event_timestamp: datetime
    frame_id: int


@dataclass(frozen=True)
class LatencyMonitorReport:
    source_id: Optional[int] = None
    frame_decoding_latency: Optional[float] = None
    inference_latency: Optional[float] = None
    e2e_latency: Optional[float] = None


@dataclass(frozen=True)
class PipelineStateReport:
    video_source_status_updates: List[StatusUpdate]
    latency_reports: List[LatencyMonitorReport]
    inference_throughput: float
    sources_metadata: List[SourceMetadata]


InferenceHandler = Callable[[List[VideoFrame]], List[AnyPrediction]]
SinkHandler = Optional[
    Union[
        Callable[[AnyPrediction, VideoFrame], None],
        Callable[[List[Optional[AnyPrediction]], List[Optional[VideoFrame]]], None],
    ]
]
