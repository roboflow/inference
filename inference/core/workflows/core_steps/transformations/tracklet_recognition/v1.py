from __future__ import annotations

from collections import OrderedDict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite
from typing import Any, Deque, Dict, List, Literal, Optional, Type, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.workflows.core_steps.fusion.detections_classes_replacement.v1 import (
    extract_leading_class_from_prediction,
)
from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    FLOAT_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
    STILL_IMAGE_INPUT_SOFT_RESTRICTION,
    BlockResult,
    RuntimeRestriction,
    WorkflowBlock,
    WorkflowBlockManifest,
)

TRACKLET_RECOGNITION_PREDICTION_KINDS = [
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
]

RECOGNITION_CLASS_NAME_KEY = "recognition_class_name"
RECOGNITION_CLASS_ID_KEY = "recognition_class_id"
RECOGNITION_CONFIDENCE_KEY = "recognition_confidence"
RECOGNITION_LOCKED_KEY = "recognition_locked"
RECOGNITION_UPDATED_KEY = "recognition_updated_this_frame"
RECOGNITION_AGE_SECONDS_KEY = "recognition_age_seconds"
ORIGINAL_CLASS_NAME_KEY = "original_class_name"
ORIGINAL_CLASS_ID_KEY = "original_class_id"
ORIGINAL_CONFIDENCE_KEY = "original_confidence"

MAX_STATE_IDS = 1024
MAX_VIDEOS_PER_STATE = 256
MAX_TRACKLETS_PER_VIDEO = 16384
TRACKLET_STATE_TTL_SECONDS = 300.0

SHORT_GATE_DESCRIPTION = "Select tracked detections that are due for recognition."
SHORT_CACHE_DESCRIPTION = (
    "Cache recognition results by tracklet and attach them to every frame."
)


@dataclass
class TrackletRecognition:
    class_name: str
    class_id: int
    confidence: float
    updated_at_seconds: float
    last_seen_at_seconds: float
    history: Deque[str] = field(default_factory=deque)
    locked: bool = False


class TrackletRecognitionStore:
    def __init__(
        self,
        max_videos: int = MAX_VIDEOS_PER_STATE,
        max_tracklets_per_video: int = MAX_TRACKLETS_PER_VIDEO,
        tracklet_state_ttl_seconds: float = TRACKLET_STATE_TTL_SECONDS,
    ) -> None:
        self._per_video: (
            "OrderedDict[str, OrderedDict[Union[int, str], TrackletRecognition]]"
        ) = OrderedDict()
        self._max_videos = max_videos
        self._max_tracklets_per_video = max_tracklets_per_video
        self._tracklet_state_ttl_seconds = tracklet_state_ttl_seconds

    def get(
        self,
        video_id: str,
        tracker_id: Union[int, str],
        timestamp_seconds: Optional[float] = None,
    ) -> Optional[TrackletRecognition]:
        per_video = self._per_video.get(video_id)
        if per_video is None:
            return None
        self._per_video.move_to_end(video_id)
        recognition = per_video.get(tracker_id)
        if recognition is None:
            return None
        if timestamp_seconds is not None and is_stale(
            last_seen_at_seconds=recognition.last_seen_at_seconds,
            timestamp_seconds=timestamp_seconds,
            ttl_seconds=self._tracklet_state_ttl_seconds,
        ):
            del per_video[tracker_id]
            if not per_video:
                del self._per_video[video_id]
            return None
        per_video.move_to_end(tracker_id)
        if timestamp_seconds is not None:
            recognition.last_seen_at_seconds = timestamp_seconds
        return recognition

    def set(
        self,
        video_id: str,
        tracker_id: Union[int, str],
        class_name: str,
        class_id: int,
        confidence: float,
        timestamp_seconds: float,
        consistency_window: int,
    ) -> TrackletRecognition:
        self.prune(timestamp_seconds=timestamp_seconds)
        per_video = self._per_video.setdefault(video_id, OrderedDict())
        self._per_video.move_to_end(video_id)
        existing = per_video.get(tracker_id)
        if existing is None:
            history: Deque[str] = deque(maxlen=max(1, consistency_window))
            existing = TrackletRecognition(
                class_name=class_name,
                class_id=class_id,
                confidence=confidence,
                updated_at_seconds=timestamp_seconds,
                last_seen_at_seconds=timestamp_seconds,
                history=history,
            )
            per_video[tracker_id] = existing
        elif existing.history.maxlen != max(1, consistency_window):
            existing.history = deque(
                existing.history, maxlen=max(1, consistency_window)
            )

        existing.class_name = class_name
        existing.class_id = class_id
        existing.confidence = confidence
        existing.updated_at_seconds = timestamp_seconds
        existing.last_seen_at_seconds = timestamp_seconds
        existing.history.append(class_name)
        existing.locked = (
            len(existing.history) == existing.history.maxlen
            and len(set(existing.history)) == 1
        )
        per_video.move_to_end(tracker_id)
        self.enforce_limits()
        return existing

    def prune(self, timestamp_seconds: float) -> None:
        empty_video_ids = []
        for video_id, per_video in self._per_video.items():
            stale_tracker_ids = [
                tracker_id
                for tracker_id, recognition in per_video.items()
                if is_stale(
                    last_seen_at_seconds=recognition.last_seen_at_seconds,
                    timestamp_seconds=timestamp_seconds,
                    ttl_seconds=self._tracklet_state_ttl_seconds,
                )
            ]
            for tracker_id in stale_tracker_ids:
                del per_video[tracker_id]
            if not per_video:
                empty_video_ids.append(video_id)
        for video_id in empty_video_ids:
            del self._per_video[video_id]

    def enforce_limits(self) -> None:
        while len(self._per_video) > self._max_videos:
            self._per_video.popitem(last=False)
        for per_video in self._per_video.values():
            while len(per_video) > self._max_tracklets_per_video:
                per_video.popitem(last=False)


_STORES: "OrderedDict[str, TrackletRecognitionStore]" = OrderedDict()


def _get_store(state_id: str) -> TrackletRecognitionStore:
    if state_id not in _STORES:
        _STORES[state_id] = TrackletRecognitionStore()
    _STORES.move_to_end(state_id)
    while len(_STORES) > MAX_STATE_IDS:
        _STORES.popitem(last=False)
    return _STORES[state_id]


class TrackletRecognitionGateManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Tracklet Recognition Gate",
            "version": "v1",
            "short_description": SHORT_GATE_DESCRIPTION,
            "long_description": "Filters tracked detections so downstream recognition only runs for tracklets that are not locked and whose recognition interval has elapsed. State is process-local, bounded, and intended for stateful video workflows; it is not durable or shared across server instances.",
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-timer",
                "blockPriority": 5,
            },
        }
    )
    type: Literal["roboflow_core/tracklet_recognition_gate@v1"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        description="Video frame image. The embedded video metadata identifies the video stream and timestamp used for per-tracklet throttling.",
    )
    detections: Selector(kind=TRACKLET_RECOGNITION_PREDICTION_KINDS) = Field(
        description="Tracked detections. Each detection must have tracker_id information.",
        examples=["$steps.byte_tracker.tracked_detections"],
    )
    recognition_interval_seconds: Union[
        float, Selector(kind=[FLOAT_KIND, FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        default=0.2,
        ge=0.0,
        description="Minimum video-time interval between recognition attempts for each unlocked tracklet.",
    )
    state_id: Union[str, Selector(kind=[STRING_KIND])] = Field(
        default="default",
        description="Shared process-local state namespace. Use the same low-cardinality value in the Tracklet Recognition Cache block connected after recognition.",
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="detections_to_recognize",
                kind=TRACKLET_RECOGNITION_PREDICTION_KINDS,
            ),
            OutputDefinition(
                name="all_detections",
                kind=TRACKLET_RECOGNITION_PREDICTION_KINDS,
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        return [
            STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
            STILL_IMAGE_INPUT_SOFT_RESTRICTION,
        ]


class TrackletRecognitionGateBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return TrackletRecognitionGateManifest

    def run(
        self,
        image: WorkflowImageData,
        detections: sv.Detections,
        recognition_interval_seconds: float,
        state_id: str,
    ) -> BlockResult:
        detections = ensure_detection_ids(detections=deepcopy(detections))
        if detections.tracker_id is None:
            raise ValueError(
                f"tracker_id not initialized, {self.__class__.__name__} requires detections to be tracked"
            )
        timestamp_seconds = get_video_time_seconds(image=image)
        video_id = image.video_metadata.video_identifier
        store = _get_store(state_id=state_id)
        store.prune(timestamp_seconds=timestamp_seconds)

        due_mask = []
        for tracker_id in detections.tracker_id.tolist():
            recognition = store.get(
                video_id=video_id,
                tracker_id=tracker_id,
                timestamp_seconds=timestamp_seconds,
            )
            is_due = is_due_for_recognition(
                recognition=recognition,
                timestamp_seconds=timestamp_seconds,
                recognition_interval_seconds=recognition_interval_seconds,
            )
            due_mask.append(is_due)

        return {
            "detections_to_recognize": detections[due_mask],
            "all_detections": detections,
        }


class TrackletRecognitionCacheManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Tracklet Recognition Cache",
            "version": "v1",
            "short_description": SHORT_CACHE_DESCRIPTION,
            "long_description": "Stores recognition results by tracker ID, locks stable tracklets, and emits all current detections enriched with cached recognition metadata. State is process-local and bounded by state namespace, video stream, tracklet count, and stale-track TTL; it is not durable or shared across server instances.",
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-lock-keyhole",
                "blockPriority": 6,
            },
        }
    )
    type: Literal["roboflow_core/tracklet_recognition_cache@v1"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        description="Video frame image. The embedded video metadata identifies the video stream and timestamp used for recognition cache updates.",
    )
    detections: Selector(kind=TRACKLET_RECOGNITION_PREDICTION_KINDS) = Field(
        description="All current tracked detections, typically from Tracklet Recognition Gate's all_detections output.",
        examples=["$steps.tracklet_recognition_gate.all_detections"],
    )
    recognized_detections: Selector(kind=TRACKLET_RECOGNITION_PREDICTION_KINDS) = Field(
        description="The subset sent to recognition, typically from Tracklet Recognition Gate's detections_to_recognize output. Used to map positional recognition results back to tracker IDs.",
        examples=["$steps.tracklet_recognition_gate.detections_to_recognize"],
    )
    recognition_predictions: Selector(
        kind=[CLASSIFICATION_PREDICTION_KIND, STRING_KIND, LIST_OF_VALUES_KIND]
    ) = Field(
        description="Recognition results produced from crops of recognized_detections. Results with parent_id are matched by detection_id; plain strings/lists are matched positionally.",
        examples=["$steps.recognition_model.predictions"],
    )
    lock_after_consistent_results: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=5,
        ge=1,
        description="Number of consecutive identical recognition results required before a tracklet is locked.",
    )
    state_id: Union[str, Selector(kind=[STRING_KIND])] = Field(
        default="default",
        description="Shared process-local state namespace. Must match the Tracklet Recognition Gate block and should use a stable, low-cardinality value.",
    )
    update_detection_class: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="When true, replace detection class/confidence with the latest cached recognition result so existing visualization blocks can display it.",
    )
    update_class_id_from_cache_status: Union[bool, Selector(kind=[BOOLEAN_KIND])] = (
        Field(
            default=False,
            description="Advanced setting. When true, replace detection class_id with cache status for visualization coloring: 0 for unlocked cached results, 1 for results updated in the current frame, and 2 for locked results. This does not change the displayed recognition text.",
            json_schema_extra={
                "advanced": True,
            },
        )
    )

    @classmethod
    def accepts_empty_values(cls) -> bool:
        return True

    @classmethod
    def get_input_dimensionality_offsets(cls) -> Dict[str, int]:
        return {
            "detections": 0,
            "recognized_detections": 0,
            "recognition_predictions": 1,
        }

    @classmethod
    def get_dimensionality_reference_property(cls) -> Optional[str]:
        return "detections"

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="recognized_detections",
                kind=TRACKLET_RECOGNITION_PREDICTION_KINDS,
            ),
            OutputDefinition(
                name="recognition_results",
                kind=[LIST_OF_VALUES_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        return [
            STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
            STILL_IMAGE_INPUT_SOFT_RESTRICTION,
        ]


class TrackletRecognitionCacheBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return TrackletRecognitionCacheManifest

    def run(
        self,
        image: WorkflowImageData,
        detections: Optional[sv.Detections],
        recognized_detections: Optional[sv.Detections],
        recognition_predictions: Optional[
            Union[Batch[Optional[Union[dict, str, List[str]]], dict, str, List[str]]]
        ],
        lock_after_consistent_results: int,
        state_id: str,
        update_detection_class: bool,
        update_class_id_from_cache_status: bool = False,
    ) -> BlockResult:
        if detections is None:
            return {"recognized_detections": None, "recognition_results": None}
        detections = ensure_detection_ids(detections=deepcopy(detections))
        if detections.tracker_id is None:
            raise ValueError(
                f"tracker_id not initialized, {self.__class__.__name__} requires detections to be tracked"
            )

        timestamp_seconds = get_video_time_seconds(image=image)
        video_id = image.video_metadata.video_identifier
        store = _get_store(state_id=state_id)
        store.prune(timestamp_seconds=timestamp_seconds)
        updated_tracker_ids = update_store_from_predictions(
            store=store,
            video_id=video_id,
            timestamp_seconds=timestamp_seconds,
            recognized_detections=recognized_detections,
            recognition_predictions=recognition_predictions,
            consistency_window=lock_after_consistent_results,
        )
        enriched = enrich_detections_with_recognition(
            detections=detections,
            store=store,
            video_id=video_id,
            timestamp_seconds=timestamp_seconds,
            updated_tracker_ids=updated_tracker_ids,
            update_detection_class=update_detection_class,
            update_class_id_from_cache_status=update_class_id_from_cache_status,
        )
        recognition_results = build_recognition_results(
            detections=enriched,
            store=store,
            video_id=video_id,
            timestamp_seconds=timestamp_seconds,
            updated_tracker_ids=updated_tracker_ids,
        )
        return {
            "recognized_detections": enriched,
            "recognition_results": recognition_results,
        }


def is_due_for_recognition(
    recognition: Optional[TrackletRecognition],
    timestamp_seconds: float,
    recognition_interval_seconds: float,
) -> bool:
    if recognition is None:
        return True
    if recognition.locked:
        return False
    return (
        timestamp_seconds - recognition.updated_at_seconds
    ) >= recognition_interval_seconds


def is_stale(
    last_seen_at_seconds: float,
    timestamp_seconds: float,
    ttl_seconds: float,
) -> bool:
    return (
        timestamp_seconds >= last_seen_at_seconds
        and (timestamp_seconds - last_seen_at_seconds) > ttl_seconds
    )


def update_store_from_predictions(
    store: TrackletRecognitionStore,
    video_id: str,
    timestamp_seconds: float,
    recognized_detections: Optional[sv.Detections],
    recognition_predictions: Optional[
        Union[Batch[Optional[Union[dict, str, List[str]]], dict, str, List[str]]]
    ],
    consistency_window: int,
) -> set:
    recognition_predictions = normalize_recognition_predictions(
        recognition_predictions=recognition_predictions
    )
    if (
        recognized_detections is None
        or len(recognized_detections) == 0
        or len(recognition_predictions) == 0
    ):
        return set()
    if recognized_detections.tracker_id is None:
        raise ValueError("recognized_detections must include tracker_id")
    recognized_detections = ensure_detection_ids(detections=recognized_detections)
    detection_id_to_tracker_id = {
        str(detection_id): tracker_id
        for detection_id, tracker_id in zip(
            recognized_detections[DETECTION_ID_KEY],
            recognized_detections.tracker_id.tolist(),
        )
    }
    updated_tracker_ids = set()
    positional_tracker_ids = recognized_detections.tracker_id.tolist()
    for index, prediction in enumerate(recognition_predictions):
        if prediction is None:
            continue
        extracted = extract_leading_class_from_prediction(prediction=prediction)
        if extracted is None:
            continue
        class_name, class_id, confidence = extracted
        tracker_id = resolve_tracker_id_for_prediction(
            prediction=prediction,
            prediction_index=index,
            detection_id_to_tracker_id=detection_id_to_tracker_id,
            positional_tracker_ids=positional_tracker_ids,
        )
        if tracker_id is None:
            continue
        store.set(
            video_id=video_id,
            tracker_id=tracker_id,
            class_name=class_name,
            class_id=class_id,
            confidence=confidence,
            timestamp_seconds=timestamp_seconds,
            consistency_window=consistency_window,
        )
        updated_tracker_ids.add(tracker_id)
    return updated_tracker_ids


def normalize_recognition_predictions(
    recognition_predictions: Optional[
        Union[Batch[Optional[Union[dict, str, List[str]]], dict, str, List[str]]]
    ],
) -> List[Optional[Union[dict, str, List[str]]]]:
    if recognition_predictions is None:
        return []
    if isinstance(recognition_predictions, (str, dict)):
        return [recognition_predictions]
    return list(recognition_predictions)


def resolve_tracker_id_for_prediction(
    prediction: Union[dict, str, List[str]],
    prediction_index: int,
    detection_id_to_tracker_id: Dict[str, Union[int, str]],
    positional_tracker_ids: List[Union[int, str]],
) -> Optional[Union[int, str]]:
    if isinstance(prediction, dict):
        parent_id = prediction.get(PARENT_ID_KEY)
        if parent_id is not None:
            return detection_id_to_tracker_id.get(str(parent_id))
    if prediction_index < len(positional_tracker_ids):
        return positional_tracker_ids[prediction_index]
    return None


def enrich_detections_with_recognition(
    detections: sv.Detections,
    store: TrackletRecognitionStore,
    video_id: str,
    timestamp_seconds: float,
    updated_tracker_ids: set,
    update_detection_class: bool,
    update_class_id_from_cache_status: bool,
) -> sv.Detections:
    if len(detections) == 0:
        add_empty_recognition_fields(detections=detections)
        return detections

    class_names = []
    class_ids = []
    confidences = []
    locked = []
    updated = []
    ages = []
    for tracker_id in detections.tracker_id.tolist():
        recognition = store.get(
            video_id=video_id,
            tracker_id=tracker_id,
            timestamp_seconds=timestamp_seconds,
        )
        if recognition is None:
            class_names.append("")
            class_ids.append(-1)
            confidences.append(np.nan)
            locked.append(False)
            updated.append(False)
            ages.append(np.nan)
            continue
        class_names.append(recognition.class_name)
        class_ids.append(recognition.class_id)
        confidences.append(recognition.confidence)
        locked.append(recognition.locked)
        updated.append(tracker_id in updated_tracker_ids)
        ages.append(timestamp_seconds - recognition.updated_at_seconds)

    detections[RECOGNITION_CLASS_NAME_KEY] = np.asarray(class_names, dtype=object)
    detections[RECOGNITION_CLASS_ID_KEY] = np.asarray(class_ids, dtype=int)
    detections[RECOGNITION_CONFIDENCE_KEY] = np.asarray(confidences, dtype=float)
    detections[RECOGNITION_LOCKED_KEY] = np.asarray(locked, dtype=bool)
    detections[RECOGNITION_UPDATED_KEY] = np.asarray(updated, dtype=bool)
    detections[RECOGNITION_AGE_SECONDS_KEY] = np.asarray(ages, dtype=float)

    if update_detection_class:
        preserve_original_detection_class(detections=detections)
        has_recognition = detections[RECOGNITION_CLASS_NAME_KEY] != ""
        if CLASS_NAME_DATA_FIELD not in detections.data:
            detections.data[CLASS_NAME_DATA_FIELD] = np.asarray(
                [""] * len(detections), dtype=object
            )
        detections.data[CLASS_NAME_DATA_FIELD][has_recognition] = detections[
            RECOGNITION_CLASS_NAME_KEY
        ][has_recognition]
        if detections.class_id is None:
            detections.class_id = np.full(len(detections), -1, dtype=int)
        detections.class_id[has_recognition] = detections[RECOGNITION_CLASS_ID_KEY][
            has_recognition
        ]
        if detections.confidence is None:
            detections.confidence = np.full(len(detections), np.nan, dtype=float)
        detections.confidence[has_recognition] = detections[RECOGNITION_CONFIDENCE_KEY][
            has_recognition
        ]

    if update_class_id_from_cache_status:
        preserve_original_detection_class(detections=detections)
        if detections.class_id is None:
            detections.class_id = np.full(len(detections), 0, dtype=int)
        detections.class_id[:] = build_cache_status_class_ids(
            locked=detections[RECOGNITION_LOCKED_KEY],
            updated=detections[RECOGNITION_UPDATED_KEY],
        )

    return detections


def build_cache_status_class_ids(locked: np.ndarray, updated: np.ndarray) -> np.ndarray:
    status_class_ids = np.zeros(len(locked), dtype=int)
    status_class_ids[updated] = 1
    status_class_ids[locked] = 2
    return status_class_ids


def preserve_original_detection_class(detections: sv.Detections) -> None:
    if ORIGINAL_CLASS_NAME_KEY not in detections.data:
        if CLASS_NAME_DATA_FIELD in detections.data:
            detections[ORIGINAL_CLASS_NAME_KEY] = detections.data[
                CLASS_NAME_DATA_FIELD
            ].copy()
        else:
            detections[ORIGINAL_CLASS_NAME_KEY] = np.asarray(
                [""] * len(detections), dtype=object
            )
    if ORIGINAL_CLASS_ID_KEY not in detections.data:
        if detections.class_id is not None:
            detections[ORIGINAL_CLASS_ID_KEY] = detections.class_id.copy()
        else:
            detections[ORIGINAL_CLASS_ID_KEY] = np.full(len(detections), -1, dtype=int)
    if ORIGINAL_CONFIDENCE_KEY not in detections.data:
        if detections.confidence is not None:
            detections[ORIGINAL_CONFIDENCE_KEY] = detections.confidence.copy()
        else:
            detections[ORIGINAL_CONFIDENCE_KEY] = np.full(
                len(detections), np.nan, dtype=float
            )


def add_empty_recognition_fields(detections: sv.Detections) -> None:
    detections[RECOGNITION_CLASS_NAME_KEY] = np.asarray([], dtype=object)
    detections[RECOGNITION_CLASS_ID_KEY] = np.asarray([], dtype=int)
    detections[RECOGNITION_CONFIDENCE_KEY] = np.asarray([], dtype=float)
    detections[RECOGNITION_LOCKED_KEY] = np.asarray([], dtype=bool)
    detections[RECOGNITION_UPDATED_KEY] = np.asarray([], dtype=bool)
    detections[RECOGNITION_AGE_SECONDS_KEY] = np.asarray([], dtype=float)


def build_recognition_results(
    detections: sv.Detections,
    store: TrackletRecognitionStore,
    video_id: str,
    timestamp_seconds: float,
    updated_tracker_ids: set,
) -> List[Dict[str, Any]]:
    results = []
    for tracker_id in detections.tracker_id.tolist():
        recognition = store.get(
            video_id=video_id,
            tracker_id=tracker_id,
            timestamp_seconds=timestamp_seconds,
        )
        if recognition is None:
            continue
        results.append(
            {
                "tracker_id": to_native_value(tracker_id),
                "text": recognition.class_name,
                "confidence": recognition.confidence,
                "class_id": recognition.class_id,
                "locked": recognition.locked,
                "updated_this_frame": tracker_id in updated_tracker_ids,
                "age_seconds": timestamp_seconds - recognition.updated_at_seconds,
            }
        )
    return results


def to_native_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def ensure_detection_ids(detections: sv.Detections) -> sv.Detections:
    if DETECTION_ID_KEY in detections.data and len(detections[DETECTION_ID_KEY]) == len(
        detections
    ):
        return detections
    detections[DETECTION_ID_KEY] = np.asarray(
        [f"tracklet_recognition.{i}" for i in range(len(detections))],
        dtype=object,
    )
    return detections


def get_video_time_seconds(image: WorkflowImageData) -> float:
    metadata = image.video_metadata
    if metadata.comes_from_video_file and metadata.fps:
        return metadata.frame_number / metadata.fps
    timestamp = metadata.frame_timestamp
    if isinstance(timestamp, datetime):
        return timestamp.timestamp()
    if isinstance(timestamp, (int, float)) and isfinite(timestamp):
        return float(timestamp)
    return datetime.now().timestamp()
