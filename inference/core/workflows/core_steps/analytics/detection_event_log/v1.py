from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Type, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    VideoMetadata,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    DICTIONARY_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    FLOAT_KIND,
    Selector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


OUTPUT_KEY = "event_log"
DETECTIONS_OUTPUT_KEY = "detections"


@dataclass
class DetectionEvent:
    """Stores event data for a tracked detection."""
    tracker_id: int
    class_name: str
    first_seen_frame: int
    first_seen_timestamp: float
    most_recent_frame: int
    most_recent_timestamp: float
    frame_count: int = 1
    logged: bool = False


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Detection Event Log",
            "version": "v1",
            "short_description": "Tracks detection events over time, logging when objects first appear and persist.",
            "long_description": (
                "This block maintains a log of detection events from tracked objects. "
                "It records when each object was first seen, its class, and the most recent time it was seen. "
                "Objects must be seen for a minimum number of frames (frame_threshold) before being logged. "
                "The log is periodically cleared based on flush_time."
            ),
            "license": "Apache-2.0",
            "block_type": "analytics",
            "ui_manifest": {
                "section": "analytics",
                "icon": "fal fa-list-timeline",
                "blockPriority": 3,
            },
        }
    )
    type: Literal["roboflow_core/detection_event_log@v1"]

    image: WorkflowImageSelector = Field(
        description="Reference to the image for video metadata (frame number, timestamp).",
        examples=["$inputs.image"],
    )

    detections: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Tracked detections from byte tracker (must have tracker_id).",
        examples=["$steps.byte_tracker.tracked_detections"],
    )

    frame_threshold: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(
        default=5,
        description="Number of frames an object must be seen before being logged.",
        examples=[5, 10],
        ge=1,
    )

    flush_time: Union[float, WorkflowParameterSelector(kind=[FLOAT_KIND])] = Field(
        default=60.0,
        description="Time in seconds after which the event log is cleared.",
        examples=[30.0, 60.0],
        ge=0,
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_KEY,
                kind=[DICTIONARY_KIND],
            ),
            OutputDefinition(
                name=DETECTIONS_OUTPUT_KEY,
                kind=[OBJECT_DETECTION_PREDICTION_KIND, INSTANCE_SEGMENTATION_PREDICTION_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class DetectionEventLogBlockV1(WorkflowBlock):
    """
    Block that tracks detection events over time.

    Maintains a dictionary of tracked objects with:
    - First seen timestamp and frame
    - Most recent timestamp and frame
    - Class name
    - Frame count (number of frames the object has been seen)

    Only logs objects that have been seen for at least frame_threshold frames.
    Clears the log periodically based on flush_time.
    """

    def __init__(self):
        # Dict[video_id, Dict[tracker_id, DetectionEvent]]
        self._event_logs: Dict[str, Dict[int, DetectionEvent]] = {}
        # Dict[video_id, last_flush_timestamp]
        self._last_flush: Dict[str, float] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def _get_timestamp(self, metadata: VideoMetadata) -> float:
        """Extract timestamp from video metadata."""
        if metadata.comes_from_video_file and metadata.fps and metadata.fps != 0:
            return metadata.frame_number / metadata.fps
        return metadata.frame_timestamp.timestamp()

    def _should_flush(self, video_id: str, current_time: float, flush_time: float) -> bool:
        """Check if the event log should be flushed."""
        if flush_time <= 0:
            return False
        last_flush = self._last_flush.get(video_id, current_time)
        return (current_time - last_flush) >= flush_time

    def run(
        self,
        image: WorkflowImageData,
        detections: sv.Detections,
        frame_threshold: int,
        flush_time: float,
    ) -> BlockResult:
        metadata = image.video_metadata
        video_id = metadata.video_identifier
        current_frame = metadata.frame_number
        current_time = self._get_timestamp(metadata)

        # Initialize event log for this video if needed
        event_log = self._event_logs.setdefault(video_id, {})

        # Check if we should flush the log
        if self._should_flush(video_id, current_time, flush_time):
            event_log.clear()
            self._last_flush[video_id] = current_time

        # Initialize last flush time if not set
        if video_id not in self._last_flush:
            self._last_flush[video_id] = current_time

        # Process detections
        if detections.tracker_id is None or len(detections.tracker_id) == 0:
            # No tracked detections, return current log
            return {
                OUTPUT_KEY: self._format_event_log(event_log, frame_threshold),
                DETECTIONS_OUTPUT_KEY: detections,
            }

        # Get class names
        class_names = detections.data.get("class_name", [])
        if len(class_names) == 0 and hasattr(detections, "class_id") and detections.class_id is not None:
            class_names = [f"class_{cid}" for cid in detections.class_id]

        # Update event log for each tracked detection
        for i, tracker_id in enumerate(detections.tracker_id):
            tracker_id = int(tracker_id)
            class_name = class_names[i] if i < len(class_names) else "unknown"

            if tracker_id in event_log:
                # Update existing event
                event = event_log[tracker_id]
                event.most_recent_frame = current_frame
                event.most_recent_timestamp = current_time
                event.frame_count += 1

                # Mark as logged once threshold is reached
                if event.frame_count >= frame_threshold and not event.logged:
                    event.logged = True
                    print(f"[Detection Event Log] Object {tracker_id} ({class_name}) logged after {event.frame_count} frames")
            else:
                # Create new event
                event_log[tracker_id] = DetectionEvent(
                    tracker_id=tracker_id,
                    class_name=class_name,
                    first_seen_frame=current_frame,
                    first_seen_timestamp=current_time,
                    most_recent_frame=current_frame,
                    most_recent_timestamp=current_time,
                    frame_count=1,
                    logged=False,
                )

        return {
            OUTPUT_KEY: self._format_event_log(event_log, frame_threshold),
            DETECTIONS_OUTPUT_KEY: detections,
        }

    def _format_event_log(
        self,
        event_log: Dict[int, DetectionEvent],
        frame_threshold: int,
    ) -> Dict[str, Any]:
        """Format the event log for output."""
        logged_events = {}
        pending_events = {}

        for tracker_id, event in event_log.items():
            event_data = {
                "tracker_id": event.tracker_id,
                "class_name": event.class_name,
                "first_seen_frame": event.first_seen_frame,
                "first_seen_timestamp": event.first_seen_timestamp,
                "most_recent_frame": event.most_recent_frame,
                "most_recent_timestamp": event.most_recent_timestamp,
                "frame_count": event.frame_count,
            }

            if event.frame_count >= frame_threshold:
                logged_events[str(tracker_id)] = event_data
            else:
                pending_events[str(tracker_id)] = event_data

        return {
            "logged": logged_events,
            "pending": pending_events,
            "total_logged": len(logged_events),
            "total_pending": len(pending_events),
        }
