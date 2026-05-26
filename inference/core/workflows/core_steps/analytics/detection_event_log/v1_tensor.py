import heapq
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core import logger
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    VideoMetadata,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    DICTIONARY_KIND,
    FLOAT_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
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
    first_seen_timestamp: float  # Unix wall-clock time (frame_timestamp or time.time())
    last_seen_frame: int
    last_seen_timestamp: float  # Unix wall-clock time (frame_timestamp or time.time())
    first_seen_relative: float = 0.0  # seconds since video start
    last_seen_relative: float = 0.0  # seconds since video start
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
                "For each tracked object it records: class name, first and last seen frame numbers, "
                "absolute wall-clock timestamps (Unix epoch floats derived from frame_timestamp metadata, "
                "or time.time() as fallback), and relative timestamps in seconds since the video started. "
                "Objects must be seen for a minimum number of frames (frame_threshold) before being moved "
                "from 'pending' to 'logged' status. "
                "Stale events (not seen for stale_frames frames) are removed during periodic cleanup "
                "(every flush_interval frames). When a logged event goes stale it is emitted in the "
                "complete_events output, which contains the full event data for objects that were tracked "
                "long enough to be logged and have since left the scene. "
                "The reference_timestamp parameter is deprecated and no longer used."
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
        default=30,
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

    reference_timestamp: Optional[Union[float, Selector(kind=[FLOAT_KIND])]] = Field(
        default=None,
        description="Deprecated, no longer used. Absolute timestamps are now taken directly from frame_timestamp metadata (or time.time() as fallback).",
        examples=[1726570875.0],
        deprecated=True,
    )

    fallback_fps: Union[float, WorkflowParameterSelector(kind=[FLOAT_KIND])] = Field(
        default=1.0,
        description="Fallback FPS to use when video metadata does not provide FPS information. Used to calculate relative timestamps.",
        examples=[1.0, 30.0],
        gt=0,
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
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                ],
            ),
            OutputDefinition(
                name="total_logged",
                kind=[INTEGER_KIND],
            ),
            OutputDefinition(
                name="total_pending",
                kind=[INTEGER_KIND],
            ),
            OutputDefinition(
                name="complete_events",
                kind=[DICTIONARY_KIND],
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
        # Dict[video_id, first_frame_timestamp] - stores the first frame's wall-clock timestamp
        # Used as the anchor for frame_timestamp-based relative time calculation
        self._first_frame_timestamps: Dict[str, float] = {}
        # Global frame counter for tracking video access order
        self._global_frame: int = 0
        # Min-heap of (last_access_frame, video_id) for efficient oldest video lookup
        self._access_heap: List[Tuple[int, str]] = []

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def _get_relative_time(
        self,
        video_id: str,
        metadata: VideoMetadata,
        fallback_fps: float,
    ) -> float:
        """Calculate relative time in seconds since video started.

        Uses frame_timestamp from metadata when available for accurate timing,
        even when inference doesn't run at the camera's reported FPS (e.g. due
        to dropped frames or processing lag). Falls back to metadata.frame_number
        / FPS when frame_timestamp is not available.
        """
        if metadata.frame_timestamp is not None:
            frame_ts = metadata.frame_timestamp.timestamp()
            if video_id not in self._first_frame_timestamps:
                self._first_frame_timestamps[video_id] = frame_ts
            return frame_ts - self._first_frame_timestamps[video_id]

        # Fallback: use actual video frame number (not internal counter) to
        # correctly account for dropped/skipped frames during inference.
        # frame_number=0 is a sentinel for static/non-video images, treat as first frame.
        fps = metadata.fps if metadata.fps and metadata.fps != 0 else fallback_fps
        return max(metadata.frame_number - 1, 0) / fps

    def _evict_oldest_video(self) -> None:
        """Remove the oldest video stream data when MAX_VIDEOS is exceeded."""
        if len(self._event_logs) <= MAX_VIDEOS:
            return

        # Rebuild heap if out of sync with current state
        if len(self._access_heap) < len(self._last_access):
            self._access_heap[:] = [
                (frame, vid) for vid, frame in self._last_access.items()
            ]
            heapq.heapify(self._access_heap)

        # Pop stale entries until we find a valid current entry
        while self._access_heap:
            frame, vid = heapq.heappop(self._access_heap)
            if self._last_access.get(vid) == frame and vid in self._event_logs:
                oldest_video_id = vid
                break
        else:
            # If heap is empty but we have event_logs, use fallback
            oldest_video_id = min(self._last_access, key=self._last_access.get)

        # Remove all data for this video
        self._event_logs.pop(oldest_video_id, None)
        self._last_flush_frame.pop(oldest_video_id, None)
        self._frame_count.pop(oldest_video_id, None)
        self._last_access.pop(oldest_video_id, None)
        self._first_frame_timestamps.pop(oldest_video_id, None)

    def _remove_stale_events(
        self,
        event_log: Dict[int, DetectionEvent],
        current_frame: int,
        stale_frames: int,
        frame_threshold: int,
    ) -> List[DetectionEvent]:
        """Remove events that haven't been seen for stale_frames.

        Returns list of removed LOGGED events (events that met frame_threshold).
        These are "complete" events - objects that were tracked long enough
        to be logged and have now left the scene.
        """
        stale_tracker_ids = []
        complete_events = []

        for tracker_id, event in event_log.items():
            frames_since_seen = current_frame - event.last_seen_frame
            if frames_since_seen > stale_frames:
                stale_tracker_ids.append(tracker_id)
                # Only return logged events as "complete" - pending events are just discarded
                if event.frame_count >= frame_threshold:
                    complete_events.append(event)

        for tracker_id in stale_tracker_ids:
            del event_log[tracker_id]

        return complete_events

    def run(
        self,
        image: WorkflowImageData,
        detections: sv.Detections,
        frame_threshold: int,
        flush_interval: int,
        stale_frames: int,
        fallback_fps: float = 1.0,
        reference_timestamp: Optional[float] = None,
    ) -> BlockResult:
        """Process detections and update the event log.

        Args:
            image: Workflow image data containing video metadata.
            detections: Tracked detections with tracker_id from ByteTracker.
            frame_threshold: Minimum frames an object must be seen before logging.
            flush_interval: How often to run stale event cleanup.
            stale_frames: Remove events not seen for this many frames.
            fallback_fps: FPS to use when video metadata doesn't provide FPS.
            reference_timestamp: Unused, kept for backward compatibility.

        Returns:
            Dictionary containing event_log, detections, total_logged, and total_pending.
        """
        metadata = image.video_metadata
        video_id = metadata.video_identifier

        # Track global frame count and video access for eviction
        self._global_frame += 1
        self._last_access[video_id] = self._global_frame

        # Increment internal frame counter
        current_frame = self._frame_count.get(video_id, 0) + 1
        self._frame_count[video_id] = current_frame

        current_time = self._get_relative_time(video_id, metadata, fallback_fps)

        # Use frame_timestamp for absolute time when available (reflects actual capture
        # time, not inference processing time). Falls back to time.time().
        current_absolute_time = (
            metadata.frame_timestamp.timestamp()
            if metadata.frame_timestamp is not None
            else time.time()
        )

        # Initialize event log for this video if needed
        event_log = self._event_logs.setdefault(video_id, {})

        # Evict oldest video if we've exceeded MAX_VIDEOS (after adding current video)
        self._evict_oldest_video()

        # Initialize last flush frame if not set
        if video_id not in self._last_flush_frame:
            self._last_flush_frame[video_id] = current_frame

        # Check if it's time to run cleanup
        complete_events_list = []
        last_flush = self._last_flush_frame.get(video_id, 0)
        if (current_frame - last_flush) >= flush_interval:
            complete_events_list = self._remove_stale_events(
                event_log, current_frame, stale_frames, frame_threshold
            )
            self._last_flush_frame[video_id] = current_frame

        # Format complete events
        complete_events = self._format_complete_events(complete_events_list)

        # Process detections
        if detections.tracker_id is None or len(detections.tracker_id) == 0:
            # No tracked detections, return current log
            event_log_dict, total_logged, total_pending = self._format_event_log(
                event_log, frame_threshold
            )
            return {
                OUTPUT_KEY: event_log_dict,
                DETECTIONS_OUTPUT_KEY: detections,
                "total_logged": total_logged,
                "total_pending": total_pending,
                "complete_events": complete_events,
            }

        # Get class names
        class_names = detections.data.get("class_name", [])
        if (
            len(class_names) == 0
            and hasattr(detections, "class_id")
            and detections.class_id is not None
        ):
            class_names = [f"class_{cid}" for cid in detections.class_id]

        # Update event log for each tracked detection
        for i, tracker_id in enumerate(detections.tracker_id):
            tracker_id = int(tracker_id)
            class_name = str(class_names[i]) if len(class_names) > 0 else "unknown"

            if tracker_id in event_log:
                # Update existing event
                event = event_log[tracker_id]
                event.last_seen_frame = current_frame
                event.last_seen_timestamp = current_absolute_time
                event.last_seen_relative = current_time
                event.frame_count += 1

                # Mark as logged once threshold is reached
                if event.frame_count >= frame_threshold and not event.logged:
                    event.logged = True
                    logger.debug(
                        f"Object {tracker_id} ({event.class_name}) logged after {event.frame_count} frames"
                    )
            else:
                # Create new event
                event_log[tracker_id] = DetectionEvent(
                    tracker_id=tracker_id,
                    class_name=class_name,
                    first_seen_frame=current_frame,
                    first_seen_timestamp=current_absolute_time,
                    last_seen_frame=current_frame,
                    last_seen_timestamp=current_absolute_time,
                    first_seen_relative=current_time,
                    last_seen_relative=current_time,
                    frame_count=1,
                    logged=False,
                )

        event_log_dict, total_logged, total_pending = self._format_event_log(
            event_log, frame_threshold
        )
        return {
            OUTPUT_KEY: event_log_dict,
            DETECTIONS_OUTPUT_KEY: detections,
            "total_logged": total_logged,
            "total_pending": total_pending,
            "complete_events": complete_events,
        }

    def _format_complete_events(
        self,
        complete_events: List[DetectionEvent],
    ) -> Dict[str, Any]:
        """Format complete events for output.

        Args:
            complete_events: List of DetectionEvent objects that have completed (gone stale).

        Returns:
            Dictionary with tracker_id as key and event data as value.
        """
        formatted = {}
        for event in complete_events:
            event_data = event.__dict__.copy()
            del event_data["logged"]
            formatted[str(event.tracker_id)] = event_data

        return formatted

    def _format_event_log(
        self,
        event_log: Dict[int, DetectionEvent],
        frame_threshold: int,
    ) -> tuple:
        """Format the event log for output.

        Returns:
            Tuple of (event_log_dict, total_logged, total_pending)
        """
        logged_events = {}
        pending_events = {}

        for tracker_id, event in event_log.items():
            event_data = event.__dict__.copy()
            del event_data["logged"]

            if event.frame_count >= frame_threshold:
                logged_events[str(tracker_id)] = event_data
            else:
                pending_events[str(tracker_id)] = event_data

        event_log_dict = {
            "logged": logged_events,
            "pending": pending_events,
        }

        return event_log_dict, len(logged_events), len(pending_events)
