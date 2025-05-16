from collections import deque
from typing import Any, Dict, Literal, Union

import supervision as sv
from pydantic import Field

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    Selector,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INTEGER_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)

__all__ = [
    "InstanceCache",
    "BaseTrackerBlockManifest",
    "BaseTrackerBlock",
    "BaseReIDTrackerBlockManifest",
    "BaseReIDTrackerBlock",
]


class InstanceCache:
    """Tracks which tracker_ids have already been seen for a given video."""

    def __init__(self, size: int = 16_384) -> None:
        self._size = size
        self._dq: deque[int] = deque(maxlen=size)
        self._set: set[int] = set()

    def record(self, tracker_id: int) -> bool:
        """Returns **True** if *tracker_id* was seen before, else **False** and records it."""
        if tracker_id == -1:
            return True  # ignore unconfirmed tracks
        if tracker_id in self._set:
            return True
        if len(self._dq) == self._size:
            old = self._dq.popleft()
            self._set.discard(old)
        self._dq.append(tracker_id)
        self._set.add(tracker_id)
        return False


# ---------------------------------------------------------------------------
# Base tracker (algorithm‑agnostic)
# ---------------------------------------------------------------------------


class BaseTrackerBlockManifest(WorkflowBlockManifest):
    """Common schema for any tracker workflow block."""

    type: Literal["roboflow_core/base_tracker@v1"] = Field(
        ...,
        description="Abstract base – do not instantiate directly",
    )

    # Inputs
    image: Selector(kind=[IMAGE_KIND])
    detections: Selector(kind=[OBJECT_DETECTION_PREDICTION_KIND])

    # Shared params
    track_activation_threshold: Union[
        float, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = 0.25
    lost_track_buffer: Union[int, Selector(kind=[INTEGER_KIND])] = 30
    minimum_consecutive_frames: Union[int, Selector(kind=[INTEGER_KIND])] = 3
    minimum_iou_threshold: Union[float, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = 0.3
    instances_cache_size: Union[int, Selector(kind=[INTEGER_KIND])] = 16_384

    @classmethod
    def describe_outputs(cls):
        return [
            OutputDefinition(
                name="tracked_detections",
                kind=[OBJECT_DETECTION_PREDICTION_KIND],
            ),
            OutputDefinition(
                name="new_instances",
                kind=[OBJECT_DETECTION_PREDICTION_KIND],
            ),
            OutputDefinition(
                name="already_seen_instances",
                kind=[OBJECT_DETECTION_PREDICTION_KIND],
            ),
        ]


class BaseTrackerBlock(WorkflowBlock):
    """Algorithm‑agnostic tracker block implementing the standard I/O contract."""

    @classmethod
    def get_manifest(cls):  # noqa: D401 (pydantic style)
        return BaseTrackerBlockManifest

    # ---------------------------------------------------------------------
    # Lifecycle helpers
    # ---------------------------------------------------------------------

    def __init__(self) -> None:  # noqa: D401 – override
        # one tracker + cache per video stream
        self._trackers: Dict[str, Any] = {}
        self._per_video_cache: Dict[str, InstanceCache] = {}

    # ---------------------- abstract hooks ------------------------------

    def _instantiate_tracker(self, *, video_id: str, frame_rate: float, **kwargs):
        """Return a *new* underlying tracker instance.  Must be provided by subclass."""
        raise NotImplementedError

    def _update_tracker(
        self,
        tracker,
        detections: sv.Detections,
        image: WorkflowImageData | None = None,
    ) -> sv.Detections:
        """Default update calls *tracker.update(detections)*; override if frame needed."""
        return tracker.update(detections)

    # ---------------------------- run -----------------------------------

    def run(
        self,
        image: WorkflowImageData,
        detections: sv.Detections,
        track_activation_threshold: float,
        lost_track_buffer: int,
        minimum_consecutive_frames: int,
        minimum_iou_threshold: float,
        instances_cache_size: int,
        **extra_params,
    ):
        video_id = getattr(image.video_metadata, "video_identifier", "default")
        fps = getattr(image.video_metadata, "fps", 0.0) or 0.0

        # create tracker lazily
        if video_id not in self._trackers:
            self._trackers[video_id] = self._instantiate_tracker(
                video_id=video_id,
                frame_rate=fps,
                track_activation_threshold=track_activation_threshold,
                lost_track_buffer=lost_track_buffer,
                minimum_consecutive_frames=minimum_consecutive_frames,
                minimum_iou_threshold=minimum_iou_threshold,
                **extra_params,
            )
        tracker = self._trackers[video_id]

        tracked = self._update_tracker(tracker, detections, image=image)

        # cache bookkeeping ------------------------------------------------
        if video_id not in self._per_video_cache:
            self._per_video_cache[video_id] = InstanceCache(size=instances_cache_size)
        cache = self._per_video_cache[video_id]

        ids = tracked.tracker_id or []
        new_mask, seen_mask = [], []
        for tid in ids:
            seen_before = cache.record(tid)
            new_mask.append(not seen_before)
            seen_mask.append(seen_before)

        new_instances = tracked[new_mask] if new_mask else sv.Detections.empty()
        already_seen = tracked[seen_mask] if seen_mask else sv.Detections.empty()

        return {
            "tracked_detections": tracked,
            "new_instances": new_instances,
            "already_seen_instances": already_seen,
        }


# ---------------------------------------------------------------------------
# Base for ReID‑based trackers (needs frame & embedding model)
# ---------------------------------------------------------------------------


class BaseReIDTrackerBlockManifest(BaseTrackerBlockManifest):
    type: Literal["roboflow_core/reid_tracker@v1"]

    embedding_model: Union[str, Selector(kind=[STRING_KIND])] = "clip/RN101"
    appearance_threshold: Union[float, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = 0.7
    appearance_weight: Union[float, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = 0.5
    distance_metric: Union[str, Selector(kind=[STRING_KIND])] = "cosine"
    device: Union[str, Selector(kind=[STRING_KIND]), None] = None


class BaseReIDTrackerBlock(BaseTrackerBlock):
    @classmethod
    def get_manifest(cls):
        return BaseReIDTrackerBlockManifest

    # lazy single model shared across all videos
    _reid_model = None
    _reid_model_name = None

    # ------------------------------------------------------------------

    def _get_reid_model(self, model_name: str, device: str | None):
        if self._reid_model is None or model_name != self._reid_model_name:
            from trackers.core.reid.model import ReIDModel

            self._reid_model = ReIDModel.from_timm(
                model_name=model_name, device=device or "auto"
            )
            self._reid_model_name = model_name
        return self._reid_model

    # Override BaseTrackerBlock behaviour
    def _update_tracker(
        self, tracker, detections: sv.Detections, image: WorkflowImageData
    ):
        return tracker.update(detections, image.numpy_image)
