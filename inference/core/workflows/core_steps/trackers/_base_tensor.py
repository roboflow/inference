"""Shared base classes for tracker workflow blocks (tensor-native sibling).

Tensor-native counterpart of ``_base.py``. Under
``ENABLE_TENSOR_DATA_REPRESENTATION`` the loader swaps each concrete tracker
block to its ``*_tensor.py`` sibling; those siblings import ``TrackerBlockBase``
/ ``TRACKER_PREDICTION_KINDS`` / ``tracker_describe_outputs`` from here instead
of from ``_base.py``.

The only representation-specific change versus ``_base.py`` is the native
input/output handling in ``_run_tracker``: detection predictions are native
``inference_models`` objects (``Detections`` / ``InstanceDetections`` / the
``(KeyPoints, Optional[Detections])`` keypoint tuple) rather than
``sv.Detections``. The third-party tracker libraries (``trackers`` package) are
``sv.Detections``-based, so ``_run_tracker`` materialises a minimal
``sv.Detections`` (bounding boxes only, with a stashed row index) as the
transport to/from the tracker, then maps the surviving rows back onto the
ORIGINAL native input — preserving masks / keypoints / all native metadata —
and writes the assigned ``tracker_id`` into ``bboxes_metadata``. Block outputs
are always native ``inference_models`` objects; ``sv.Detections`` is used only
as the algorithm boundary to the tracker library.

Each concrete tracker block (ByteTrack, BoT-SORT, SORT, OC-SORT) inherits from
``TrackerBlockBase`` and implements ``_create_tracker`` and ``get_manifest``.
Sub-classes may override ``_tracker_update`` when the underlying tracker needs
extra per-frame context (e.g. a video frame for camera motion compensation).
``_tracker_update`` / ``_create_tracker`` stay ``sv.Detections``-based /
library-based and identical to ``_base.py`` (the third-party trackers are
``sv``-based) — only ``_run_tracker`` does the native↔sv conversion.
"""

import os
from abc import abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import supervision as sv
import torch

from inference.core import logger
from inference.core.workflows.core_steps.common.tensor_native import (
    split_key_point_prediction,
    take_prediction_by_indices,
)
from inference.core.workflows.core_steps.trackers.batch_scheduler import (
    get_tracker_batch_scheduler,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
    TENSOR_NATIVE_RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections

OUTPUT_KEY: str = "tracked_detections"

#: Key under which the per-detection row index into the original native input is
#: stashed inside the transport ``sv.Detections.data`` dict. The third-party
#: tracker libraries index back into the input ``sv.Detections`` (preserving the
#: ``.data`` dict) so this row index travels through ``tracker.update`` and lets
#: us slice the ORIGINAL native input by the surviving rows afterwards.
_TRACKER_ROW_INDEX_KEY: str = "__tracker_row_index__"

#: Detection kinds accepted as tracker input and declared on tracker output.
#: Trackers only use bounding boxes for association and preserve all other
#: fields (masks, keypoints, custom data) via native indexing back into the
#: original ``inference_models`` prediction.
TRACKER_PREDICTION_KINDS = [
    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
    TENSOR_NATIVE_RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
]


class InstanceCache:
    """FIFO cache that tracks which object track IDs have been seen before.

    Used to categorize tracked detections as new (first appearance) or
    already seen (reappearance) across video frames.
    """

    def __init__(self, size: int):
        size = max(1, size)
        self._cache_inserts_track = deque(maxlen=size)
        self._cache = set()

    def record_instance(self, tracker_id: int) -> bool:
        """Record a tracker ID and return whether it was previously seen.

        Returns:
            True if the tracker_id was already in the cache (seen before),
            False if this is its first appearance.
        """
        in_cache = tracker_id in self._cache
        if not in_cache:
            self._cache_new_tracker_id(tracker_id=tracker_id)
        return in_cache

    def _cache_new_tracker_id(self, tracker_id: int) -> None:
        while len(self._cache) >= self._cache_inserts_track.maxlen:
            to_drop = self._cache_inserts_track.popleft()
            self._cache.remove(to_drop)
        self._cache_inserts_track.append(tracker_id)
        self._cache.add(tracker_id)


# Native prediction shapes accepted by tracker blocks (object detection,
# instance segmentation, RLE instance segmentation, or the keypoint-detection
# tuple). Mirrors TRACKER_PREDICTION_KINDS.
TensorNativeTrackerPrediction = Union[
    Detections,
    InstanceDetections,
    Tuple[KeyPoints, Optional[Detections]],
]


class TrackerBlockBase(WorkflowBlock):
    """Common run-loop shared by every tracker block.

    Sub-classes implement ``_create_tracker`` and ``get_manifest``.  Override
    ``_tracker_update`` only when the tracker API requires additional context
    beyond ``sv.Detections`` (e.g. BoT-SORT with camera motion compensation).
    """

    def __init__(self) -> None:
        self._trackers: Dict[str, Any] = {}
        self._per_video_cache: Dict[str, InstanceCache] = {}

    @classmethod
    @abstractmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]: ...

    @abstractmethod
    def _create_tracker(self, fps: int, **kwargs: Any) -> Any:
        """Instantiate the concrete tracker with algorithm-specific params."""
        ...

    def _tracker_update(
        self,
        tracker: Any,
        detections: sv.Detections,
        image: WorkflowImageData,
    ) -> sv.Detections:
        """Invoke the tracker for one frame.

        Must call ``tracker.update`` only with arguments that library trackers
        define for the per-frame step (typically detections, optionally a frame
        tensor).  Do **not** pass workflow/block kwargs used in ``_create_tracker``.
        """
        return tracker.update(detections)

    def _tracker_batch_frame(
        self,
        tracker: Any,
        image: WorkflowImageData,
    ) -> Any | None:
        """Return optional per-frame context passed to batched Tracktors."""
        return None

    @staticmethod
    def _can_batch_tracker_update(detections: sv.Detections) -> bool:
        """Return whether this call can enter the CUDA micro-batch scheduler."""
        if os.getenv("TRACKTORS_DISABLE_BATCHING", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            return False
        boxes = detections.xyxy
        if not isinstance(boxes, torch.Tensor) or boxes.device.type != "cuda":
            return False
        try:
            import tracktors
        except ImportError:
            return False
        return callable(getattr(tracktors, "update_batch", None))

    def _run_tracker(
        self,
        image: WorkflowImageData,
        detections: TensorNativeTrackerPrediction,
        instances_cache_size: int,
        **tracker_kwargs: Any,
    ) -> BlockResult:
        """Run one frame through the tracker.

        Note: tracker parameters (``tracker_kwargs``) are only used when the
        tracker is **first created** for a given ``video_identifier``.
        Changing parameter values on subsequent frames has no effect because
        the tracker instance is cached for the lifetime of the video stream.

        Tensor-native note: ``detections`` is a native ``inference_models``
        prediction. Its bounding-box tensors are wrapped by ``sv.Detections``
        without moving them off-device. Surviving rows are mapped back onto the
        original native input so masks, keypoints, and metadata are preserved.
        """
        metadata = image.video_metadata
        fps = metadata.fps
        if not fps:
            fps = 30
            logger.warning(
                f"fps not available in VideoMetadata for {self.__class__.__name__}, "
                "defaulting to 30 fps for tracker initialisation"
            )
        video_id = metadata.video_identifier

        if video_id not in self._trackers:
            self._trackers[video_id] = self._create_tracker(fps=fps, **tracker_kwargs)

        tracker = self._trackers[video_id]

        _, bbox = split_key_point_prediction(detections)
        n = int(bbox.xyxy.shape[0])
        sv_input = sv.Detections(
            xyxy=bbox.xyxy,
            confidence=bbox.confidence,
            class_id=bbox.class_id,
            data={
                _TRACKER_ROW_INDEX_KEY: torch.arange(
                    n,
                    dtype=torch.long,
                    device=bbox.xyxy.device,
                )
            },
        )

        if self._can_batch_tracker_update(sv_input):
            tracked_sv = get_tracker_batch_scheduler().update(
                tracker,
                sv_input,
                frame=self._tracker_batch_frame(tracker, image),
            )
        else:
            tracked_sv = self._tracker_update(tracker, sv_input, image)

        # Filter out immature / unmatched tracks (tracker_id == -1). Mirror the
        # numpy guard exactly: only filter when tracker_id is present and there
        # is at least one tracked detection.
        if tracked_sv.tracker_id is not None and len(tracked_sv) > 0:
            valid_mask = tracked_sv.tracker_id != -1
            tracked_sv = tracked_sv[valid_mask]

        # Recover the surviving native-input row indices (stashed in .data and
        # preserved by the tracker library's internal sv indexing) and the
        # assigned tracker ids. The empty / library-emptied case yields no rows.
        if (
            tracked_sv.data
            and _TRACKER_ROW_INDEX_KEY in tracked_sv.data
            and tracked_sv.tracker_id is not None
            and len(tracked_sv) > 0
        ):
            surviving = torch.as_tensor(
                tracked_sv.data[_TRACKER_ROW_INDEX_KEY],
                dtype=torch.long,
                device=bbox.xyxy.device,
            )
            tracker_ids_tensor = torch.as_tensor(
                tracked_sv.tracker_id,
                dtype=torch.long,
                device=bbox.xyxy.device,
            )
        else:
            surviving = torch.empty(
                0,
                dtype=torch.long,
                device=bbox.xyxy.device,
            )
            tracker_ids_tensor = torch.empty(
                0,
                dtype=torch.long,
                device=bbox.xyxy.device,
            )

        # Slice the ORIGINAL native input by the surviving rows (handles
        # Detections / InstanceDetections / the keypoint tuple, dense + RLE
        # masks). This preserves every native field for the surviving rows.
        tracked_detections = take_prediction_by_indices(detections, surviving)

        # Tracker IDs cross to Python only for the object metadata and cache.
        tracker_ids = tracker_ids_tensor.detach().to("cpu").tolist()
        _patch_tracker_ids(tracked_detections, tracker_ids)

        if video_id not in self._per_video_cache:
            self._per_video_cache[video_id] = InstanceCache(size=instances_cache_size)
        cache = self._per_video_cache[video_id]

        not_seen_indices, seen_indices = [], []
        for position, tracker_id in enumerate(tracker_ids):
            already_seen = cache.record_instance(tracker_id=tracker_id)
            if already_seen:
                seen_indices.append(position)
            else:
                not_seen_indices.append(position)

        return {
            OUTPUT_KEY: tracked_detections,
            "new_instances": take_prediction_by_indices(
                tracked_detections, not_seen_indices
            ),
            "already_seen_instances": take_prediction_by_indices(
                tracked_detections, seen_indices
            ),
        }


def _bbox_component(
    prediction: TensorNativeTrackerPrediction,
) -> Union[Detections, InstanceDetections]:
    """Return the bounding-box-bearing component of a tracker prediction (the
    Detections/InstanceDetections itself, or the bbox element of the keypoint
    tuple)."""
    _, bbox = split_key_point_prediction(prediction)
    return bbox


def _patch_tracker_ids(
    prediction: TensorNativeTrackerPrediction,
    tracker_ids: List[int],
) -> None:
    """Write ``tracker_id`` into the bbox component's ``bboxes_metadata``.

    Builds ``bboxes_metadata`` as a list of dicts when it is ``None``, and
    copies any existing per-detection dict so caller-owned state is not mutated.
    Mutates the bbox component of ``prediction`` in place (the native objects
    produced by ``take_prediction_by_indices`` are freshly allocated for this
    block, so this is safe).
    """
    bbox = _bbox_component(prediction)
    n = int(bbox.xyxy.shape[0])
    existing = bbox.bboxes_metadata
    new_meta: List[dict] = []
    for i in range(n):
        base = dict(existing[i]) if existing is not None and existing[i] else {}
        base["tracker_id"] = int(tracker_ids[i])
        new_meta.append(base)
    bbox.bboxes_metadata = new_meta if new_meta else None


def tracker_describe_outputs() -> List[OutputDefinition]:
    """Output definitions shared by all tracker blocks.

    Trackers preserve all detection fields (masks, keypoints, custom data) —
    they only use bounding boxes for association then index back into the
    original native prediction.  The output kinds therefore mirror the input
    kinds accepted by every tracker manifest.
    """
    return [
        OutputDefinition(name=OUTPUT_KEY, kind=TRACKER_PREDICTION_KINDS),
        OutputDefinition(name="new_instances", kind=TRACKER_PREDICTION_KINDS),
        OutputDefinition(
            name="already_seen_instances",
            kind=TRACKER_PREDICTION_KINDS,
        ),
    ]
