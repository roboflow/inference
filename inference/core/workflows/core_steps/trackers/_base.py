"""Shared base classes for tracker workflow blocks.

Each concrete tracker block (ByteTrack, SORT, OC-SORT) inherits from
``TrackerBlockBase`` and only needs to implement ``_create_tracker`` and
``get_manifest``.
"""

from abc import abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional, Type

import supervision as sv

from inference.core import logger
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    OBJECT_DETECTION_PREDICTION_KIND,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "tracked_detections"


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


class TrackerBlockBase(WorkflowBlock):
    """Common run-loop shared by every tracker block.

    Sub-classes only need to override ``_create_tracker`` and ``get_manifest``.
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

    def _run_tracker(
        self,
        image: WorkflowImageData,
        detections: sv.Detections,
        instances_cache_size: int,
        **tracker_kwargs: Any,
    ) -> BlockResult:
        """Run one frame through the tracker.

        Note: tracker parameters (``tracker_kwargs``) are only used when the
        tracker is **first created** for a given ``video_identifier``.
        Changing parameter values on subsequent frames has no effect because
        the tracker instance is cached for the lifetime of the video stream.
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
        tracked_detections = tracker.update(detections)

        # Filter out immature / unmatched tracks (tracker_id == -1)
        if tracked_detections.tracker_id is not None and len(tracked_detections) > 0:
            valid_mask = tracked_detections.tracker_id != -1
            tracked_detections = tracked_detections[valid_mask]

        if video_id not in self._per_video_cache:
            self._per_video_cache[video_id] = InstanceCache(size=instances_cache_size)
        cache = self._per_video_cache[video_id]

        not_seen_mask, seen_mask = [], []
        for tracker_id in tracked_detections.tracker_id.tolist():
            already_seen = cache.record_instance(tracker_id=tracker_id)
            not_seen_mask.append(not already_seen)
            seen_mask.append(already_seen)

        return {
            OUTPUT_KEY: tracked_detections,
            "new_instances": tracked_detections[not_seen_mask],
            "already_seen_instances": tracked_detections[seen_mask],
        }


def tracker_describe_outputs() -> List[OutputDefinition]:
    """Output definitions shared by all tracker blocks."""
    return [
        OutputDefinition(name=OUTPUT_KEY, kind=[OBJECT_DETECTION_PREDICTION_KIND]),
        OutputDefinition(name="new_instances", kind=[OBJECT_DETECTION_PREDICTION_KIND]),
        OutputDefinition(
            name="already_seen_instances",
            kind=[OBJECT_DETECTION_PREDICTION_KIND],
        ),
    ]
