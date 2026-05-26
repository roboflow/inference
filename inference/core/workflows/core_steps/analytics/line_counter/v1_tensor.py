from typing import Dict, List, Optional, Tuple, Union

import supervision as sv
from pydantic import ConfigDict, Field
from typing_extensions import Literal, Type

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    VideoMetadata,
)
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
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

OUTPUT_KEY_COUNT_IN: str = "count_in"
OUTPUT_KEY_COUNT_OUT: str = "count_out"
IN: str = "in"
OUT: str = "out"
DETECTIONS_IN_OUT_PARAM: str = "in_out"
SHORT_DESCRIPTION = "Count detections passing a line."
LONG_DESCRIPTION = """
Count objects crossing a defined line segment in video using tracked detections, maintaining separate counts for objects crossing in opposite directions (in and out) for traffic analysis, people counting, entry/exit monitoring, and directional flow measurement workflows.

## How This Block Works

This block counts objects that cross a line segment by tracking their movement across video frames. The block:

1. Receives tracked detection predictions with unique tracker IDs and video metadata
2. Validates that detections have tracker IDs (required for tracking object movement across frames)
3. Initializes or retrieves a line zone for the video:
   - Creates a LineZone from two coordinate points defining the line segment
   - Stores line zone configuration per video using video_identifier
   - Maintains separate counting state for each video
4. Monitors object positions across frames:
   - Tracks each object's position using its unique tracker_id
   - Detects when an object's triggering anchor point (default: CENTER of bounding box) crosses the line
   - Determines crossing direction based on which side of the line the object approaches from
5. Counts line crossings:
   - **In Direction**: Objects crossing the line in one direction increment the count_in counter
   - **Out Direction**: Objects crossing the line in the opposite direction increment the count_out counter
   - Each unique tracker_id is counted only once per crossing (prevents duplicate counting if object oscillates near line)
6. Maintains persistent counting state:
   - Counts accumulate across frames for the entire video
   - State persists for each video until workflow execution completes
   - Separate counters for each unique video_identifier
7. Returns two count values:
   - **count_in**: Total number of objects that crossed the line in the "in" direction
   - **count_out**: Total number of objects that crossed the line in the "out" direction

The line segment defines a virtual boundary in the video frame. The direction (in/out) is determined by which side of the line objects approach from - for a horizontal line, objects coming from above might count as "in" while objects from below count as "out" (or vice versa, depending on line orientation). The triggering anchor determines which point on the bounding box must cross the line for the crossing to be counted - using CENTER ensures the object is substantially across the line before counting.

## Common Use Cases

- **People Counting**: Count people entering and exiting buildings, stores, or events (e.g., count visitors entering store, track people entering/exiting building, monitor event attendance), enabling entry/exit counting workflows
- **Traffic Analysis**: Count vehicles passing through intersections or road segments (e.g., count vehicles crossing intersection, track traffic flow in specific directions, monitor vehicle passage at checkpoints), enabling traffic flow analysis workflows
- **Retail Analytics**: Track customer movement and foot traffic in retail spaces (e.g., count customers entering store sections, track movement between departments, monitor shopping flow patterns), enabling retail foot traffic analytics workflows
- **Security Monitoring**: Monitor entry and exit at secure areas or checkpoints (e.g., track entries to restricted areas, count people at access points, monitor checkpoint crossings), enabling security access monitoring workflows
- **Occupancy Management**: Track occupancy changes by counting objects entering and leaving spaces (e.g., count entries/exits to manage room capacity, track vehicle arrivals/departures in parking, monitor space occupancy changes), enabling occupancy tracking workflows
- **Wildlife Monitoring**: Count animals crossing defined paths or boundaries (e.g., track animal migration patterns, count wildlife crossing roads, monitor animal movement in habitats), enabling wildlife behavior analysis workflows

## Connecting to Other Blocks

This block receives tracked detections and video metadata, and produces count_in and count_out values:

- **After Byte Tracker blocks** to count tracked objects crossing lines (e.g., count tracked people crossing line, track vehicle crossings with consistent IDs, monitor tracked object movements), enabling tracking-to-counting workflows
- **After object detection or instance segmentation blocks** with tracking enabled to count detected objects (e.g., count detected vehicles, track people crossings, monitor object movements), enabling detection-to-counting workflows
- **Before visualization blocks** to display line counter information (e.g., visualize line and counts, display crossing statistics, show counting results), enabling counting visualization workflows
- **Before data storage blocks** to record counting data (e.g., log entry/exit counts, store traffic statistics, record occupancy metrics), enabling counting data logging workflows
- **Before notification blocks** to alert on count thresholds or events (e.g., alert when count exceeds limit, notify on occupancy changes, trigger actions based on counts), enabling count-based notification workflows
- **Before analysis blocks** to process counting metrics (e.g., analyze traffic patterns, process occupancy data, work with counting statistics), enabling counting analysis workflows

## Requirements

This block requires tracked detections with tracker_id information (detections must come from a tracking block like Byte Tracker). The line must be defined as a list of exactly 2 points, where each point is a list or tuple of exactly 2 coordinates (x, y). The block requires video metadata with video_identifier to maintain separate counting state for different videos. The block maintains persistent counting state across frames for each video, so it should be used in video workflows where frames are processed sequentially. For accurate counting, detections should be provided consistently across frames with valid tracker IDs.
"""


class LineCounterManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Line Counter",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "analytics",
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-arrow-down-up-across-line",
                "blockPriority": 2,
            },
        }
    )
    type: Literal["roboflow_core/line_counter@v1"]
    metadata: Selector(kind=[VIDEO_METADATA_KIND]) = Field(
        description="Video metadata containing video_identifier to maintain separate counting state for different videos. Required for persistent counting across frames.",
    )
    detections: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Tracked object detection or instance segmentation predictions. Must include tracker_id information from a tracking block. Objects are counted when their triggering anchor point crosses the line segment.",
        examples=["$steps.object_detection_model.predictions"],
    )

    line_segment: Union[list, Selector(kind=[LIST_OF_VALUES_KIND]), Selector(kind=[LIST_OF_VALUES_KIND])] = Field(  # type: ignore
        description="Line segment defined by exactly two points, each with [x, y] coordinates. Objects crossing from one side count as 'in', objects crossing from the other side count as 'out'. Example: [[0, 100], [500, 100]] creates a horizontal line at y=100. Crossing direction depends on which side objects approach from.",
        examples=[[[0, 50], [500, 50]], "$inputs.zones"],
    )
    triggering_anchor: Union[str, Selector(kind=[STRING_KIND]), Literal[tuple(sv.Position.list())]] = Field(  # type: ignore
        description="Point on the bounding box that must cross the line for counting. Options: CENTER (default), BOTTOM_CENTER, TOP_CENTER, CENTER_LEFT, CENTER_RIGHT, etc. CENTER ensures the object is substantially across the line before counting, reducing false positives from objects near but not fully crossing the line.",
        default="CENTER",
        examples=["CENTER"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_KEY_COUNT_IN,
                kind=[INTEGER_KIND],
            ),
            OutputDefinition(
                name=OUTPUT_KEY_COUNT_OUT,
                kind=[INTEGER_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class LineCounterBlockV1(WorkflowBlock):
    def __init__(self):
        self._batch_of_line_zones: Dict[str, sv.LineZone] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return LineCounterManifest

    def run(
        self,
        detections: sv.Detections,
        metadata: VideoMetadata,
        line_segment: List[Tuple[int, int]],
        triggering_anchor: str = "CENTER",
    ) -> BlockResult:
        if detections.tracker_id is None:
            raise ValueError(
                f"tracker_id not initialized, {self.__class__.__name__} requires detections to be tracked"
            )
        if metadata.video_identifier not in self._batch_of_line_zones:
            if not isinstance(line_segment, list) or len(line_segment) != 2:
                raise ValueError(
                    f"{self.__class__.__name__} requires line zone to be a list containing exactly 2 points"
                )
            if any(not isinstance(e, list) or len(e) != 2 for e in line_segment):
                raise ValueError(
                    f"{self.__class__.__name__} requires each point of line zone to be a list containing exactly 2 coordinates"
                )
            if any(
                not isinstance(e[0], (int, float)) or not isinstance(e[1], (int, float))
                for e in line_segment
            ):
                raise ValueError(
                    f"{self.__class__.__name__} requires each coordinate of line zone to be a number"
                )
            self._batch_of_line_zones[metadata.video_identifier] = sv.LineZone(
                start=sv.Point(*line_segment[0]),
                end=sv.Point(*line_segment[1]),
                triggering_anchors=[sv.Position(triggering_anchor)],
            )
        line_zone = self._batch_of_line_zones[metadata.video_identifier]

        line_zone.trigger(detections=detections)

        return {
            OUTPUT_KEY_COUNT_IN: line_zone.in_count,
            OUTPUT_KEY_COUNT_OUT: line_zone.out_count,
        }
