import itertools
from collections import OrderedDict
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
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    Selector,
    WorkflowImageSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

PolygonAsNestedList = List[List[int]]
PolygonAsArray = np.ndarray
PolygonAsListOfArrays = List[np.ndarray]
PolygonAsListOfTuples = List[Tuple[int, int]]
Polygon = Union[
    PolygonAsNestedList, PolygonAsArray, PolygonAsListOfArrays, PolygonAsListOfTuples
]


OUTPUT_KEY: str = "timed_detections"
SHORT_DESCRIPTION = "Track object time in zone."
LONG_DESCRIPTION = """
Calculate and track the time spent by tracked objects within one or more defined polygon zones, measure duration of object presence in specific areas (supporting multiple zones where objects are considered 'in zone' if present in any zone), filter detections based on zone membership, reset time tracking when objects leave zones, and enable zone-based analytics, dwell time analysis, and presence monitoring workflows.

## How This Block Works

This block measures how long each tracked object has been inside one or more defined polygon zones by tracking entry and exit times for each unique track ID. The block supports multiple zones, treating objects as 'in zone' if they are present in any of the defined zones. The block:

1. Receives tracked detection predictions with track IDs, an image with embedded video metadata, and polygon zone definition(s) (single zone or list of zones)
2. Extracts video metadata from the image:
   - Accesses video_metadata from the WorkflowImageData object
   - Extracts fps, frame_number, frame_timestamp, video_identifier, and video source information
   - Uses video_identifier to maintain separate tracking state for different videos
3. Validates that detections have track IDs (tracker_id must be present):
   - Requires detections to come from a tracking block (e.g., Byte Tracker)
   - Each object must have a unique tracker_id that persists across frames
   - Raises an error if tracker_id is missing
4. Normalizes zone input to a list of polygons:
   - Accepts a single polygon zone or a list of polygon zones
   - Automatically wraps single polygons in a list for consistent processing
   - Validates nesting depth and coordinate format for all zones
   - Enables flexible zone input formats (single zone or multiple zones)
5. Initializes or retrieves polygon zones for the video:
   - Creates a list of PolygonZone objects from zone coordinates for each unique zone combination
   - Validates zone coordinates (each zone must be a list of at least 3 points, each with 2 coordinates)
   - Stores zone configurations in an OrderedDict with zone cache (max 100 zone combinations)
   - Uses zone key combining video_identifier and zone coordinates for cache lookup
   - Implements FIFO eviction when cache exceeds 100 zone combinations
   - Configures triggering anchor point (e.g., CENTER, BOTTOM_CENTER) for zone detection
6. Initializes or retrieves time tracking state for the video:
   - Maintains a dictionary tracking when each track_id entered any zone
   - Stores entry timestamps per video using video_identifier
   - Maintains separate tracking state for each video
7. Calculates current timestamp for time measurement:
   - For video files: Calculates timestamp as frame_number / fps
   - For streamed video: Uses frame_timestamp from metadata
   - Provides accurate time measurement for duration calculation
8. Checks which detections are in any zone:
   - Tests each detection against all polygon zones using polygon zone triggers
   - Creates a matrix of zone membership (zones x detections)
   - Uses logical OR operation: objects are considered 'in zone' if they're in ANY of the zones
   - The triggering_anchor determines which point on the bounding box is checked (CENTER, BOTTOM_CENTER, etc.)
   - Returns boolean for each detection indicating zone membership in any zone
9. Updates time tracking for each tracked object:
   - **For objects entering any zone**: Records entry timestamp if not already tracked
   - **For objects in any zone**: Calculates time spent as current_timestamp - entry_timestamp
   - **For objects leaving all zones**: 
     - If reset_out_of_zone_detections is True: Removes entry timestamp (resets to 0)
     - If reset_out_of_zone_detections is False: Keeps entry timestamp (continues tracking)
10. Handles out-of-zone detections:
    - **If remove_out_of_zone_detections is True**: Filters out detections outside all zones from output
    - **If remove_out_of_zone_detections is False**: Includes out-of-zone detections with time = 0
11. Adds time_in_zone information to each detection:
    - Attaches time_in_zone value (in seconds) to each detection as metadata
    - Objects in any zone: Time represents duration spent in any zone
    - Objects outside all zones: Time is 0 (if not reset) or undefined (if removed)
12. Returns detections with time_in_zone information:
    - Outputs tracked detections enhanced with time_in_zone metadata
    - Filtered or unfiltered based on remove_out_of_zone_detections setting
    - Maintains all original detection properties plus time tracking information

The block maintains persistent tracking state across frames, allowing accurate cumulative time measurement for objects that remain in any zone over multiple frames. Time is measured from when an object first enters any zone (based on its track_id) until the current frame, providing real-time duration tracking. When multiple zones are provided, objects are considered 'in zone' if their anchor point is inside any of the zones, allowing tracking across multiple areas as a single combined zone. The zone cache efficiently manages multiple zone configurations per video using FIFO eviction to limit memory usage. The triggering anchor determines which part of the bounding box is used for zone detection, enabling different zone entry/exit behaviors based on object position.

## Common Use Cases

- **Multi-Zone Dwell Time Analysis**: Measure how long objects remain in any of multiple areas for behavior analysis (e.g., measure customer time in any store section, track time spent in multiple parking areas, analyze time in overlapping zones), enabling multi-zone dwell time analytics workflows
- **Zone-Based Monitoring**: Monitor object presence across multiple defined areas for security and safety (e.g., detect loitering in any restricted area, monitor time in multiple danger zones, track presence across secure zones), enabling multi-zone monitoring workflows
- **Retail Analytics**: Track customer time across multiple store sections for retail insights (e.g., measure time in any product aisle, analyze shopping patterns across departments, track engagement in multiple zones), enabling multi-zone retail analytics workflows
- **Occupancy Management**: Measure time objects spend in any of multiple spaces for space utilization (e.g., track vehicle parking duration in multiple lots, measure table occupancy across zones, analyze space usage in multiple areas), enabling multi-zone occupancy management workflows
- **Safety Compliance**: Monitor time violations across multiple restricted or time-limited zones (e.g., detect extended stays in any hazardous area, monitor time limit violations across zones, track safety compliance in multiple areas), enabling multi-zone safety monitoring workflows
- **Traffic Analysis**: Measure time vehicles spend in any of multiple traffic zones or intersections (e.g., track time at multiple intersections, measure queue waiting time across zones, analyze traffic flow in multiple areas), enabling multi-zone traffic analytics workflows

## Connecting to Other Blocks

This block receives an image with embedded video metadata, tracked detections, and zone coordinates (single or multiple zones), and produces timed_detections with time_in_zone metadata:

- **After Byte Tracker blocks** to measure time for tracked objects across multiple zones (e.g., track time in multiple zones for tracked objects, measure dwell time with consistent IDs across areas, analyze tracked object presence in multiple zones), enabling tracking-to-time workflows
- **After zone definition blocks** to apply time tracking to multiple defined areas (e.g., measure time across multiple polygon zones, track duration in custom multi-zone configurations, analyze zone-based presence across areas), enabling zone-to-time workflows
- **Before logic blocks** like Continue If to make decisions based on time in any zone (e.g., continue if time exceeds threshold in any zone, filter based on dwell time across zones, trigger actions on time violations in multiple areas), enabling time-based decision workflows
- **Before analysis blocks** to analyze time-based metrics across multiple zones (e.g., analyze dwell time patterns across zones, process time-in-zone data for multiple areas, work with duration metrics across zones), enabling time analysis workflows
- **Before notification blocks** to alert on time violations or thresholds in any zone (e.g., alert on extended stays in any zone, notify on time limit violations across areas, trigger time-based alerts for multiple zones), enabling time-based notification workflows
- **Before data storage blocks** to record time metrics across multiple zones (e.g., store dwell time data for multiple areas, log time-in-zone metrics across zones, record duration measurements for multiple zones), enabling time metrics logging workflows

## Version Differences

**Enhanced from v2:**

- **Multiple Zone Support**: Supports tracking time across multiple polygon zones simultaneously, where objects are considered 'in zone' if they're present in any of the defined zones, enabling multi-zone time tracking and analysis
- **Flexible Zone Input**: Accepts either a single polygon zone or a list of polygon zones, automatically normalizing the input to handle both formats seamlessly
- **Zone Cache Management**: Implements a zone cache with FIFO eviction (max 100 zone combinations) to efficiently manage multiple zone configurations per video while limiting memory usage
- **Combined Zone Logic**: Uses logical OR operation across all zones, allowing tracking across multiple areas as a unified zone system for comprehensive presence monitoring
- **Enhanced Zone Key System**: Uses combined zone keys (video_identifier + zone coordinates) for cache lookup, enabling efficient storage and retrieval of zone configurations

## Requirements

This block requires tracked detections with tracker_id information (detections must come from a tracking block like Byte Tracker). The zone can be a single polygon or a list of polygons, where each polygon must be defined as a list of at least 3 points, with each point being a list or tuple of exactly 2 coordinates (x, y). The image's video_metadata should include frame rate (fps) for video files or frame timestamps for streamed video to calculate accurate time measurements. The block maintains persistent tracking state across frames for each video, so it should be used in video workflows where frames are processed sequentially. For accurate time measurement, detections should be provided consistently across frames with valid track IDs. When multiple zones are provided, objects are considered 'in zone' if they're present in any of the zones.
"""
ZONE_CACHE_SIZE = 100


class TimeInZoneManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Time in Zone",
            "version": "v3",
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
    type: Literal["roboflow_core/time_in_zone@v3"]
    image: Union[WorkflowImageSelector] = Field(
        title="Image",
        description="Input image for the current video frame containing embedded video metadata (fps, frame_number, frame_timestamp, video_identifier, video source) required for time calculation and state management. The block extracts video_metadata from the WorkflowImageData object. The fps and frame_number are used for video files to calculate timestamps (timestamp = frame_number / fps). For streamed video, frame_timestamp is used directly. The video_identifier is used to maintain separate tracking state and zone configurations for different videos. Used for zone visualization and reference. The image dimensions are used to validate zone coordinates. This version supports multiple zones per video with efficient zone cache management.",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    detections: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Tracked detection predictions (object detection or instance segmentation) with tracker_id information. Detections must come from a tracking block (e.g., Byte Tracker) that has assigned unique tracker_id values that persist across frames. Each detection must have a tracker_id to enable time tracking. The block calculates time_in_zone for each tracked object based on when its track_id first entered any of the zones. Objects are considered 'in zone' if their anchor point is inside any of the provided zones. The output will include the same detections enhanced with time_in_zone metadata (duration in seconds). If remove_out_of_zone_detections is True, only detections inside any zone are included in the output.",
        examples=["$steps.object_detection_model.predictions"],
    )
    zone: Union[list, Selector(kind=[LIST_OF_VALUES_KIND])] = Field(  # type: ignore
        description="Polygon zone coordinates defining one or more areas for time measurement. Can be a single polygon zone or a list of polygon zones. Each zone must be a list of at least 3 points, where each point is a list or tuple of exactly 2 coordinates [x, y] or (x, y). Coordinates should be in pixel space matching the image dimensions. Example for single zone: [(100, 100), (100, 200), (300, 200), (300, 100)]. Example for multiple zones: [[(100, 100), (100, 200), (300, 200), (300, 100)], [(400, 400), (400, 500), (600, 500), (600, 400)]]. Objects are considered 'in zone' if their triggering_anchor point is inside ANY of the provided zones. Zone coordinates are validated and PolygonZone objects are created for each zone. Zone configurations are cached (max 100 combinations) with FIFO eviction.",
        examples=[[(100, 100), (100, 200), (300, 200), (300, 100)], "$inputs.zones"],
    )
    triggering_anchor: Union[
        str, Selector(kind=[STRING_KIND]), Literal[tuple(sv.Position.list())]
    ] = Field(  # type: ignore
        description="Point on the detection bounding box that must be inside the zone to consider the object 'in zone'. Options include: 'CENTER' (default, center of bounding box), 'BOTTOM_CENTER' (bottom center point), 'TOP_CENTER' (top center point), 'CENTER_LEFT' (center left point), 'CENTER_RIGHT' (center right point), and other Position enum values. The triggering anchor determines which part of the object's bounding box is checked against the zone polygon(s). When multiple zones are provided, the object is considered 'in zone' if its anchor point is inside ANY of the zones. Use CENTER for standard zone detection, BOTTOM_CENTER for ground-level zones (e.g., tracking feet/vehicle base), or other anchors based on detection needs. Default is 'CENTER'.",
        default="CENTER",
        examples=["CENTER"],
    )
    remove_out_of_zone_detections: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(  # type: ignore
        description="If True (default), detections found outside all zones are filtered out and not included in the output. Only detections inside at least one zone are returned. If False, all detections are included in the output, with time_in_zone = 0 for objects outside all zones. Use True to focus analysis only on objects in any zone, or False to maintain all detections with zone status. When multiple zones are provided, objects are considered 'in zone' if present in any zone. Default is True for cleaner output focused on zone activity.",
        default=True,
        examples=[True, False],
    )
    reset_out_of_zone_detections: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(  # type: ignore
        description="If True (default), when a tracked object leaves all zones, its time tracking is reset (entry timestamp is cleared). When the object re-enters any zone, time tracking starts from 0 again. If False, time tracking continues even after leaving all zones, and re-entry maintains cumulative time. Use True to measure current continuous time in any zone (resets on exit from all zones), or False to measure cumulative time across multiple entries. When multiple zones are provided, time is reset only when the object leaves all zones. Default is True for measuring continuous presence duration.",
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


class TimeInZoneBlockV3(WorkflowBlock):
    def __init__(self):
        self._batch_of_tracked_ids_in_zone: Dict[str, Dict[Union[int, str], float]] = {}
        self._batch_of_polygon_zones: OrderedDict[str, List[sv.PolygonZone]] = (
            OrderedDict()
        )

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return TimeInZoneManifest

    def run(
        self,
        image: WorkflowImageData,
        detections: sv.Detections,
        zone: List[List[Tuple[int, int]]],
        triggering_anchor: str,
        remove_out_of_zone_detections: bool,
        reset_out_of_zone_detections: bool,
    ) -> BlockResult:
        if detections.tracker_id is None:
            raise ValueError(
                f"tracker_id not initialized, {self.__class__.__name__} requires detections to be tracked"
            )
        metadata = image.video_metadata
        zones = ensure_zone_is_list_of_polygons(zone)
        zone_key = f"{metadata.video_identifier}_{str(zones)}"
        if zone_key not in self._batch_of_polygon_zones:
            if len(zones) > 0 and (not isinstance(zones[0], list) or len(zones[0]) < 3):
                raise ValueError(
                    f"{self.__class__.__name__} requires zone to be a list containing more than 2 points"
                )
            if any(
                (not isinstance(e, list) and not isinstance(e, tuple)) or len(e) != 2
                for e in itertools.chain.from_iterable(zones)
            ):
                raise ValueError(
                    f"{self.__class__.__name__} requires each point of zone to be a list containing exactly 2 coordinates"
                )
            if any(
                not isinstance(e[0], (int, float)) or not isinstance(e[1], (int, float))
                for e in itertools.chain.from_iterable(zones)
            ):
                raise ValueError(
                    f"{self.__class__.__name__} requires each coordinate of zone to be a number"
                )
            self._batch_of_polygon_zones[zone_key] = [
                sv.PolygonZone(
                    polygon=np.array(zone),
                    triggering_anchors=(sv.Position(triggering_anchor),),
                )
                for zone in zones
            ]
            # keeps the cache size at ZONE_CACHE_SIZE
            if len(self._batch_of_polygon_zones) > ZONE_CACHE_SIZE:
                self._batch_of_polygon_zones.popitem(last=False)
        polygon_zones = self._batch_of_polygon_zones[zone_key]
        tracked_ids_in_zone = self._batch_of_tracked_ids_in_zone.setdefault(
            metadata.video_identifier, {}
        )
        result_detections = []
        if metadata.comes_from_video_file and metadata.fps != 0:
            ts_end = metadata.frame_number / metadata.fps
        else:
            ts_end = metadata.frame_timestamp.timestamp()

        # get trigger for all zones. It is a matrix of shape (len(zones), len(detections))
        polygon_triggers = [
            polygon_zone.trigger(detections) for polygon_zone in polygon_zones
        ]
        is_in_any_zone = (
            np.any(polygon_triggers, axis=0)
            if len(polygon_triggers) > 0
            else np.array([False] * len(detections))
        )

        for i, is_in_zone, tracker_id in zip(
            range(len(detections)),
            is_in_any_zone,
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


def ensure_zone_is_list_of_polygons(
    zone: Union[Polygon, List[Polygon]],
) -> List[Polygon]:
    nesting_depth = calculate_nesting_depth(zone=zone, max_depth=3)
    if nesting_depth > 3:
        raise ValueError(
            "roboflow_core/time_in_zone@v2 block requires `zone` input to be list of points, but "
            "input with excessive nesting depth found. If you created the `zone` input manually, verify it's "
            "correctness. If the input is constructed by another Workflow block - raise an issue: "
            "https://github.com/roboflow/inference/issues"
        )
    if nesting_depth == 2:
        return [zone]
    return zone


def calculate_nesting_depth(
    zone: Union[Polygon, List[Polygon]], max_depth: int, current_depth: int = 0
) -> int:
    remaining_depth = max_depth - current_depth
    if isinstance(zone, np.ndarray):
        array_depth = len(zone.shape)
        if array_depth > remaining_depth:
            raise ValueError(
                "While processing polygon zone detected an instance of the zone which is invalid, as "
                "the input is nested beyond limits - the block supports single and multiple "
                "lists of zone points. If you created the `zone` input manually, verify it's correctness. If "
                "the input is constructed by another Workflow block - raise an issue: "
                "https://github.com/roboflow/inference/issues"
            )
        return current_depth + array_depth
    if isinstance(zone, (list, tuple)):
        if remaining_depth < 1:
            raise ValueError(
                "While processing polygon zone detected an instance of the zone which is invalid, as "
                "the input is nested beyond limits - the block supports single and multiple "
                "lists of zone points. If you created the `zone` input manually, verify it's correctness. If "
                "the input is constructed by another Workflow block - raise an issue: "
                "https://github.com/roboflow/inference/issues"
            )
        depths = {
            calculate_nesting_depth(
                zone=e, max_depth=max_depth, current_depth=current_depth + 1
            )
            for e in zone
        }
        if not depths:
            return current_depth + 1
        if len(depths) != 1:
            raise ValueError(
                "While processing polygon zone detected an instance of the zone which is invalid, as "
                "the input is nested in irregular way. If you created the `zone` input manually, verify it's correctness. "
                "If the input is constructed by another Workflow block - raise an issue: "
                "https://github.com/roboflow/inference/issues"
            )
        return min(depths)
    return current_depth
