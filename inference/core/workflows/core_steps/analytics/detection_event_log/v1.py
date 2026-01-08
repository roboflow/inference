from dataclasses import dataclass
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
    FLOAT_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
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
MAX_VIDEOS = 100  # Maximum number of video streams to track before evicting oldest


@dataclass
class DetectionEvent:
    """Stores event data for a tracked detection."""
    tracker_id: int
    class_name: str
    first_seen_frame: int
    first_seen_timestamp: float
    last_seen_frame: int
    last_seen_timestamp: float
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
                "It records when each object was first seen, its class, and the last time it was seen."
                "Objects must be seen for a minimum number of frames (frame_threshold) before being logged. "
                "Stale events (not seen for stale_frames) are removed during periodic cleanup (every flush_interval frames)."
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

    flush_interval: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(
        default=30,
        description="How often (in frames) to run the cleanup operation for stale events.",
        examples=[30, 60],
        ge=1,
    )

    stale_frames: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(
        default=300,
        description="Remove events that haven't been seen for this many frames.",
        examples=[150, 300],
        ge=1,
    )

    reference_timestamp: Optional[Union[float, WorkflowParameterSelector(kind=[FLOAT_KIND])]] = Field(
        default=None,
        description="Unix timestamp to use as reference for calculating relative times (first_seen_relative, last_seen_relative).",
        examples=[1726570875.0],
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
            OutputDefinition(
                name="total_logged",
                kind=[INTEGER_KIND],
            ),
            OutputDefinition(
                name="total_pending",
                kind=[INTEGER_KIND],
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
    - Last seen timestamp and frame
    - Class name
    - Frame count (number of frames the object has been seen)

    Only logs objects that have been seen for at least frame_threshold frames.
    Runs cleanup every flush_interval frames, removing events not seen for stale_frames.
    """

    def __init__(self):
        # Dict[video_id, Dict[tracker_id, DetectionEvent]]
        self._event_logs: Dict[str, Dict[int, DetectionEvent]] = {}
        # Dict[video_id, last_flush_frame]
        self._last_flush_frame: Dict[str, int] = {}
        # Dict[video_id, frame_count] - internal frame counter (increments each run)
        self._frame_count: Dict[str, int] = {}
        # Dict[video_id, last_access_frame] - tracks when each video was last accessed (global frame count)
        self._last_access: Dict[str, int] = {}
        # Global frame counter for tracking video access order
        self._global_frame: int = 0

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def _get_timestamp(self, metadata: VideoMetadata) -> float:
        """Extract timestamp from video metadata."""
        if metadata.comes_from_video_file and metadata.fps and metadata.fps != 0:
            return metadata.frame_number / metadata.fps
        return metadata.frame_timestamp.timestamp()

    def _should_flush(self, video_id: str, current_frame: int, flush_interval: int) -> bool:
        """Check if it's time to run cleanup."""
        last_flush = self._last_flush_frame.get(video_id, 0)
        return (current_frame - last_flush) >= flush_interval

    def _evict_oldest_video(self) -> None:
        """Remove the oldest video stream data when MAX_VIDEOS is exceeded."""
        if len(self._event_logs) <= MAX_VIDEOS:
            return

        # Find the video with the oldest last access time
        oldest_video_id = min(self._last_access, key=self._last_access.get)

        # Remove all data for this video
        self._event_logs.pop(oldest_video_id, None)
        self._last_flush_frame.pop(oldest_video_id, None)
        self._frame_count.pop(oldest_video_id, None)
        self._last_access.pop(oldest_video_id, None)

    def _remove_stale_events(
        self,
        event_log: Dict[int, DetectionEvent],
        current_frame: int,
        stale_frames: int,
    ) -> List[DetectionEvent]:
        """Remove events that haven't been seen for stale_frames.

        Returns list of removed events for logging purposes.
        """
        stale_tracker_ids = []
        removed_events = []

        for tracker_id, event in event_log.items():
            frames_since_seen = current_frame - event.last_seen_frame
            if frames_since_seen > stale_frames:
                stale_tracker_ids.append(tracker_id)
                removed_events.append(event)

        for tracker_id in stale_tracker_ids:
            del event_log[tracker_id]

        return removed_events

    def run(
        self,
        image: WorkflowImageData,
        detections: sv.Detections,
        frame_threshold: int,
        flush_interval: int,
        stale_frames: int,
        reference_timestamp: Optional[float] = None,
    ) -> BlockResult:
        metadata = image.video_metadata
        video_id = metadata.video_identifier

        # Track global frame count and video access for eviction
        self._global_frame += 1
        self._last_access[video_id] = self._global_frame

        # Evict oldest video if we've exceeded MAX_VIDEOS
        self._evict_oldest_video()

        # Increment internal frame counter
        current_frame = self._frame_count.get(video_id, 0) + 1
        self._frame_count[video_id] = current_frame

        current_time = self._get_timestamp(metadata)

        # Initialize event log for this video if needed
        event_log = self._event_logs.setdefault(video_id, {})

        # Initialize last flush frame if not set
        if video_id not in self._last_flush_frame:
            self._last_flush_frame[video_id] = current_frame

        # Check if it's time to run cleanup
        if self._should_flush(video_id, current_frame, flush_interval):
            removed_events = self._remove_stale_events(event_log, current_frame, stale_frames)
            self._last_flush_frame[video_id] = current_frame
            for event in removed_events:
                if event.logged:
                    print(f"[Detection Event Log] Removed stale object {event.tracker_id} ({event.class_name}) - not seen for {current_frame - event.last_seen_frame} frames")

        # Process detections
        if detections.tracker_id is None or len(detections.tracker_id) == 0:
            # No tracked detections, return current log
            event_log_dict, total_logged, total_pending = self._format_event_log(event_log, frame_threshold, reference_timestamp)
            return {
                OUTPUT_KEY: event_log_dict,
                DETECTIONS_OUTPUT_KEY: detections,
                "total_logged": total_logged,
                "total_pending": total_pending,
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
                event.last_seen_frame = current_frame
                event.last_seen_timestamp = current_time
                event.frame_count += 1

                # Mark as logged once threshold is reached
                if event.frame_count >= frame_threshold and not event.logged:
                    event.logged = True
                    print(f"[Detection Event Log] Object {tracker_id} ({event.class_name}) logged after {event.frame_count} frames")
            else:
                # Create new event
                event_log[tracker_id] = DetectionEvent(
                    tracker_id=tracker_id,
                    class_name=class_name,
                    first_seen_frame=current_frame,
                    first_seen_timestamp=current_time,
                    last_seen_frame=current_frame,
                    last_seen_timestamp=current_time,
                    frame_count=1,
                    logged=False,
                )

        event_log_dict, total_logged, total_pending = self._format_event_log(event_log, frame_threshold, reference_timestamp)
        return {
            OUTPUT_KEY: event_log_dict,
            DETECTIONS_OUTPUT_KEY: detections,
            "total_logged": total_logged,
            "total_pending": total_pending,
        }

    def _format_event_log(
        self,
        event_log: Dict[int, DetectionEvent],
        frame_threshold: int,
        reference_timestamp: Optional[float] = None,
    ) -> tuple:
        """Format the event log for output.

        Returns:
            Tuple of (event_log_dict, total_logged, total_pending)
        """
        logged_events = {}
        pending_events = {}

        for tracker_id, event in event_log.items():
            event_data = {
                "tracker_id": event.tracker_id,
                "class_name": event.class_name,
                "first_seen_frame": event.first_seen_frame,
                "first_seen_timestamp": event.first_seen_timestamp,
                "last_seen_frame": event.last_seen_frame,
                "last_seen_timestamp": event.last_seen_timestamp,
                "frame_count": event.frame_count,
            }

            # Add relative timestamps if reference_timestamp is provided
            if reference_timestamp is not None:
                event_data["first_seen_relative"] = event.first_seen_timestamp - reference_timestamp
                event_data["last_seen_relative"] = event.last_seen_timestamp - reference_timestamp

            if event.frame_count >= frame_threshold:
                logged_events[str(tracker_id)] = event_data
            else:
                pending_events[str(tracker_id)] = event_data

        event_log_dict = {
            "logged": logged_events,
            "pending": pending_events,
        }

        return event_log_dict, len(logged_events), len(pending_events)
