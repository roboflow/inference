from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field
from typing_extensions import Literal, Type

from inference.core.workflows.execution_engine.constants import (
    TIME_IN_ZONE_KEY_IN_SV_DETECTIONS,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    VideoMetadata,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    VIDEO_METADATA_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "timed_detections"
SHORT_DESCRIPTION = "Track object time in zone."
LONG_DESCRIPTION = """
Calculate and track the time spent by tracked objects within a defined polygon zone, measure duration of object presence in specific areas, filter detections based on zone membership, reset time tracking when objects leave zones, and enable zone-based analytics, dwell time analysis, and presence monitoring workflows.

## How This Block Works

This block measures how long each tracked object has been inside a defined polygon zone by tracking entry and exit times for each unique track ID. The block:

1. Receives tracked detection predictions with track IDs, an image, video metadata, and a polygon zone definition
2. Validates that detections have track IDs (tracker_id must be present):
   - Requires detections to come from a tracking block (e.g., Byte Tracker)
   - Each object must have a unique tracker_id that persists across frames
   - Raises an error if tracker_id is missing
3. Initializes or retrieves a polygon zone for the video:
   - Creates a PolygonZone object from zone coordinates for each unique video
   - Validates zone coordinates (must be a list of at least 3 points, each with 2 coordinates)
   - Stores zone configuration per video using video_identifier
   - Configures triggering anchor point (e.g., CENTER, BOTTOM_CENTER) for zone detection
4. Initializes or retrieves time tracking state for the video:
   - Maintains a dictionary tracking when each track_id entered the zone
   - Stores entry timestamps per video using video_identifier
   - Maintains separate tracking state for each video
5. Calculates current timestamp for time measurement:
   - For video files: Calculates timestamp as frame_number / fps
   - For streamed video: Uses frame_timestamp from metadata
   - Provides accurate time measurement for duration calculation
6. Checks which detections are in the zone:
   - Uses polygon zone trigger to test if each detection's anchor point is inside the zone
   - The triggering_anchor determines which point on the bounding box is checked (CENTER, BOTTOM_CENTER, etc.)
   - Returns boolean for each detection indicating zone membership
7. Updates time tracking for each tracked object:
   - **For objects entering the zone**: Records entry timestamp if not already tracked
   - **For objects in the zone**: Calculates time spent as current_timestamp - entry_timestamp
   - **For objects leaving the zone**: 
     - If reset_out_of_zone_detections is True: Removes entry timestamp (resets to 0)
     - If reset_out_of_zone_detections is False: Keeps entry timestamp (continues tracking)
8. Handles out-of-zone detections:
   - **If remove_out_of_zone_detections is True**: Filters out detections outside the zone from output
   - **If remove_out_of_zone_detections is False**: Includes out-of-zone detections with time = 0
9. Adds time_in_zone information to each detection:
   - Attaches time_in_zone value (in seconds) to each detection as metadata
   - Objects in zone: Time represents duration spent in zone
   - Objects outside zone: Time is 0 (if not reset) or undefined (if removed)
10. Returns detections with time_in_zone information:
    - Outputs tracked detections enhanced with time_in_zone metadata
    - Filtered or unfiltered based on remove_out_of_zone_detections setting
    - Maintains all original detection properties plus time tracking information

The block maintains persistent tracking state across frames, allowing accurate cumulative time measurement for objects that remain in the zone over multiple frames. Time is measured from when an object first enters the zone (based on its track_id) until the current frame, providing real-time duration tracking. The zone is defined as a polygon with multiple points, allowing flexible area definitions. The triggering anchor determines which part of the bounding box is used for zone detection, enabling different zone entry/exit behaviors based on object position.

## Common Use Cases

- **Dwell Time Analysis**: Measure how long objects remain in specific areas for behavior analysis (e.g., measure customer dwell time in store sections, track time spent in parking spaces, analyze time in waiting areas), enabling dwell time analytics workflows
- **Zone-Based Monitoring**: Monitor object presence in defined areas for security and safety (e.g., detect loitering in restricted areas, monitor time in danger zones, track presence in secure zones), enabling zone monitoring workflows
- **Retail Analytics**: Track customer time in different store sections for retail insights (e.g., measure time in product aisles, analyze shopping patterns, track department engagement), enabling retail analytics workflows
- **Occupancy Management**: Measure time objects spend in spaces for space utilization (e.g., track vehicle parking duration, measure table occupancy time, analyze space usage patterns), enabling occupancy management workflows
- **Safety Compliance**: Monitor time violations in restricted or time-limited zones (e.g., detect extended stays in hazardous areas, monitor time limit violations, track safety compliance), enabling safety monitoring workflows
- **Traffic Analysis**: Measure time vehicles spend in traffic zones or intersections (e.g., track time at intersections, measure queue waiting time, analyze traffic flow patterns), enabling traffic analytics workflows

## Connecting to Other Blocks

This block receives tracked detections, image, video metadata, and zone coordinates, and produces timed_detections with time_in_zone metadata:

- **After Byte Tracker blocks** to measure time for tracked objects (e.g., track time in zones for tracked objects, measure dwell time with consistent IDs, analyze tracked object presence), enabling tracking-to-time workflows
- **After zone definition blocks** to apply time tracking to defined areas (e.g., measure time in polygon zones, track duration in custom zones, analyze zone-based presence), enabling zone-to-time workflows
- **Before logic blocks** like Continue If to make decisions based on time in zone (e.g., continue if time exceeds threshold, filter based on dwell time, trigger actions on time violations), enabling time-based decision workflows
- **Before analysis blocks** to analyze time-based metrics (e.g., analyze dwell time patterns, process time-in-zone data, work with duration metrics), enabling time analysis workflows
- **Before notification blocks** to alert on time violations or thresholds (e.g., alert on extended stays, notify on time limit violations, trigger time-based alerts), enabling time-based notification workflows
- **Before data storage blocks** to record time metrics (e.g., store dwell time data, log time-in-zone metrics, record duration measurements), enabling time metrics logging workflows

## Requirements

This block requires tracked detections with tracker_id information (detections must come from a tracking block like Byte Tracker). The zone must be defined as a list of at least 3 points, where each point is a list or tuple of exactly 2 coordinates (x, y). The block requires video metadata with frame rate (fps) for video files or frame timestamps for streamed video to calculate accurate time measurements. The block maintains persistent tracking state across frames for each video, so it should be used in video workflows where frames are processed sequentially. For accurate time measurement, detections should be provided consistently across frames with valid track IDs.
"""


class TimeInZoneManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Time in Zone",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "analytics",
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-timer",
                "blockPriority": 1,
            },
        }
    )
    type: Literal["roboflow_core/time_in_zone@v1"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Image",
        description="Input image for the current video frame. Used for zone visualization and reference. The block uses the image dimensions to validate zone coordinates. The image metadata may be used for time calculation if frame timestamps are needed.",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    metadata: Selector(kind=[VIDEO_METADATA_KIND]) = Field(
        description="Video metadata containing frame rate (fps), frame number, frame timestamp, video identifier, and video source information required for time calculation and state management. The fps and frame_number are used for video files to calculate timestamps (timestamp = frame_number / fps). For streamed video, frame_timestamp is used directly. The video_identifier is used to maintain separate tracking state and zone configurations for different videos. The metadata must include valid fps for video files or frame_timestamp for streams to enable accurate time measurement.",
    )
    detections: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Tracked detection predictions (object detection or instance segmentation) with tracker_id information. Detections must come from a tracking block (e.g., Byte Tracker) that has assigned unique tracker_id values that persist across frames. Each detection must have a tracker_id to enable time tracking. The block calculates time_in_zone for each tracked object based on when its track_id first entered the zone. The output will include the same detections enhanced with time_in_zone metadata (duration in seconds). If remove_out_of_zone_detections is True, only detections inside the zone are included in the output.",
        examples=["$steps.object_detection_model.predictions"],
    )
    zone: Union[list, Selector(kind=[LIST_OF_VALUES_KIND]), Selector(kind=[LIST_OF_VALUES_KIND])] = Field(  # type: ignore
        description="Polygon zone coordinates defining the area for time measurement. Must be a list of at least 3 points, where each point is a list or tuple of exactly 2 coordinates [x, y] or (x, y). Coordinates should be in pixel space matching the image dimensions. Example: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] for a quadrilateral zone. The zone defines the polygon area where time tracking occurs. Objects are considered 'in zone' when their triggering_anchor point is inside this polygon. Zone coordinates are validated and a PolygonZone object is created for each video.",
        examples=["$inputs.zones"],
    )
    triggering_anchor: Union[str, Selector(kind=[STRING_KIND]), Literal[tuple(sv.Position.list())]] = Field(  # type: ignore
        description="Point on the detection bounding box that must be inside the zone to consider the object 'in zone'. Options include: 'CENTER' (default, center of bounding box), 'BOTTOM_CENTER' (bottom center point), 'TOP_CENTER' (top center point), 'CENTER_LEFT' (center left point), 'CENTER_RIGHT' (center right point), and other Position enum values. The triggering anchor determines which part of the object's bounding box is checked against the zone polygon. Use CENTER for standard zone detection, BOTTOM_CENTER for ground-level zones (e.g., tracking feet/vehicle base), or other anchors based on detection needs. Default is 'CENTER'.",
        default="CENTER",
        examples=["CENTER"],
    )
    remove_out_of_zone_detections: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(  # type: ignore
        description="If True (default), detections found outside the zone are filtered out and not included in the output. Only detections inside the zone are returned. If False, all detections are included in the output, with time_in_zone = 0 for objects outside the zone. Use True to focus analysis only on objects in the zone, or False to maintain all detections with zone status. Default is True for cleaner output focused on zone activity.",
        default=True,
        examples=[True, False],
    )
    reset_out_of_zone_detections: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(  # type: ignore
        description="If True (default), when a tracked object leaves the zone, its time tracking is reset (entry timestamp is cleared). When the object re-enters the zone, time tracking starts from 0 again. If False, time tracking continues even after leaving the zone, and re-entry maintains cumulative time. Use True to measure current continuous time in zone (resets on exit), or False to measure cumulative time across multiple entries. Default is True for measuring continuous presence duration.",
        default=True,
        examples=[True, False],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_KEY,
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class TimeInZoneBlockV1(WorkflowBlock):
    def __init__(self):
        self._batch_of_tracked_ids_in_zone: Dict[str, Dict[Union[int, str], float]] = {}
        self._batch_of_polygon_zones: Dict[str, sv.PolygonZone] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return TimeInZoneManifest

    def run(
        self,
        image: WorkflowImageData,
        detections: sv.Detections,
        metadata: VideoMetadata,
        zone: List[Tuple[int, int]],
        triggering_anchor: str,
        remove_out_of_zone_detections: bool,
        reset_out_of_zone_detections: bool,
    ) -> BlockResult:
        if detections.tracker_id is None:
            raise ValueError(
                f"tracker_id not initialized, {self.__class__.__name__} requires detections to be tracked"
            )
        if metadata.video_identifier not in self._batch_of_polygon_zones:
            if not isinstance(zone, list) or len(zone) < 3:
                raise ValueError(
                    f"{self.__class__.__name__} requires zone to be a list containing more than 2 points"
                )
            if any(
                (not isinstance(e, list) and not isinstance(e, tuple)) or len(e) != 2
                for e in zone
            ):
                raise ValueError(
                    f"{self.__class__.__name__} requires each point of zone to be a list containing exactly 2 coordinates"
                )
            if any(
                not isinstance(e[0], (int, float)) or not isinstance(e[1], (int, float))
                for e in zone
            ):
                raise ValueError(
                    f"{self.__class__.__name__} requires each coordinate of zone to be a number"
                )
            self._batch_of_polygon_zones[metadata.video_identifier] = sv.PolygonZone(
                polygon=np.array(zone),
                triggering_anchors=(sv.Position(triggering_anchor),),
            )
        polygon_zone = self._batch_of_polygon_zones[metadata.video_identifier]
        tracked_ids_in_zone = self._batch_of_tracked_ids_in_zone.setdefault(
            metadata.video_identifier, {}
        )
        result_detections = []
        if metadata.comes_from_video_file and metadata.fps != 0:
            ts_end = metadata.frame_number / metadata.fps
        else:
            ts_end = metadata.frame_timestamp.timestamp()
        for i, is_in_zone, tracker_id in zip(
            range(len(detections)),
            polygon_zone.trigger(detections),
            detections.tracker_id,
        ):
            if (
                not is_in_zone
                and tracker_id in tracked_ids_in_zone
                and reset_out_of_zone_detections
            ):
                del tracked_ids_in_zone[tracker_id]
            if not is_in_zone and remove_out_of_zone_detections:
                continue

            # copy
            detection = detections[i]

            detection[TIME_IN_ZONE_KEY_IN_SV_DETECTIONS] = np.array(
                [0], dtype=np.float64
            )
            if is_in_zone:
                ts_start = tracked_ids_in_zone.setdefault(tracker_id, ts_end)
                detection[TIME_IN_ZONE_KEY_IN_SV_DETECTIONS] = np.array(
                    [ts_end - ts_start], dtype=np.float64
                )
            elif tracker_id in tracked_ids_in_zone:
                del tracked_ids_in_zone[tracker_id]
            result_detections.append(detection)
        return {OUTPUT_KEY: sv.Detections.merge(result_detections)}
