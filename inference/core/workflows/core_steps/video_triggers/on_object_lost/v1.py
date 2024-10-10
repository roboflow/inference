from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    VideoMetadata,
)
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    StepOutputSelector,
    WorkflowParameterSelector,
    WorkflowVideoMetadataSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "On Object Lost",
            "version": "v1",
            "short_description": "",
            "long_description": "",
            "license": "Apache-2.0",
            "block_type": "video_trigger",
        }
    )
    type: Literal["roboflow_core/on_object_lost@v1"]
    video_metadata: WorkflowVideoMetadataSelector
    predictions: StepOutputSelector(
        kind=[OBJECT_DETECTION_PREDICTION_KIND, INSTANCE_SEGMENTATION_PREDICTION_KIND]
    )
    forget_after: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(
        description="Number of seconds to forget the instance",
        default=5,
    )
    per_video_cache_size: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = (
        Field(
            default=16384,
        )
    )
    assumed_video_fps: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = (
        Field(
            description="If video source does not manifest FPS, value will be assumed",
            default=30,
        )
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="tracker_ids",
                kind=[LIST_OF_VALUES_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class OnObjectLostBlockV1(WorkflowBlock):

    def __init__(self):
        self._per_video_cache: Dict[str, InstanceCache] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        video_metadata: VideoMetadata,
        predictions: sv.Detections,
        forget_after: int,
        per_video_cache_size: int,
        assumed_video_fps: int,
    ) -> BlockResult:
        if predictions.tracker_id is None:
            return {
                "tracker_ids": None,
            }
        if video_metadata.video_identifier not in self._per_video_cache:
            self._per_video_cache[video_metadata.video_identifier] = InstanceCache(
                size=per_video_cache_size
            )
        cache = self._per_video_cache[video_metadata.video_identifier]
        for tracker_id in predictions.tracker_id.tolist():
            cache.record_instance(
                tracker_id=tracker_id,
                frame_timestamp=video_metadata.frame_timestamp,
                frame_number=video_metadata.frame_number,
            )
        older_than, sequence_number_lower_than = None, None
        if video_metadata.comes_from_video_file is True:
            forgetting_duration_frames = (
                int(video_metadata.fps or assumed_video_fps) * forget_after
            )
            sequence_number_lower_than = (
                video_metadata.frame_number - forgetting_duration_frames
            )
        else:
            older_than = video_metadata.frame_timestamp - timedelta(
                seconds=forget_after
            )
        lost_traces = cache.pop_lost_instances(
            older_than=older_than,
            sequence_number_lower_than=sequence_number_lower_than,
        )
        if not lost_traces:
            return {
                "tracker_ids": None,
            }
        return {
            "tracker_ids": lost_traces,
        }


class InstanceCache:

    def __init__(self, size: int):
        size = max(1, size)
        self._cache_inserts_track = deque(maxlen=size)
        self._cache: Dict[int, Tuple[datetime, int]] = {}

    def record_instance(
        self,
        tracker_id: int,
        frame_timestamp: datetime,
        frame_number: int,
    ) -> None:
        if tracker_id in self._cache:
            self._cache[tracker_id] = (frame_timestamp, frame_number)
            return None
        while len(self._cache) >= self._cache_inserts_track.maxlen:
            to_drop = self._cache_inserts_track.popleft()
            del self._cache[to_drop]
        self._cache_inserts_track.append(tracker_id)
        self._cache[tracker_id] = (frame_timestamp, frame_number)
        return None

    def pop_lost_instances(
        self,
        older_than: Optional[datetime] = None,
        sequence_number_lower_than: Optional[int] = None,
    ) -> List[int]:
        if older_than is None and sequence_number_lower_than is None:
            return []
        survivor_dequeue = deque(maxlen=self._cache_inserts_track.maxlen)
        removed = []
        for tracker_id in self._cache_inserts_track:
            last_seen_timestamp, last_seen_frame_id = self._cache[tracker_id]
            to_be_removed = False
            if older_than is not None and last_seen_timestamp < older_than:
                to_be_removed = True
            if (
                sequence_number_lower_than is not None
                and last_seen_frame_id < sequence_number_lower_than
            ):
                to_be_removed = True
            if to_be_removed:
                del self._cache[tracker_id]
                removed.append(tracker_id)
            else:
                survivor_dequeue.append(tracker_id)
        self._cache_inserts_track = survivor_dequeue
        return removed
