from typing import Dict, List, Optional, Tuple, Union

import supervision as sv
from pydantic import ConfigDict, Field
from typing_extensions import Literal, Type

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
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

OUTPUT_KEY_COUNT_IN: str = "count_in"
OUTPUT_KEY_COUNT_OUT: str = "count_out"
OUTPUT_KEY_DETECTIONS_IN: str = "detections_in"
OUTPUT_KEY_DETECTIONS_OUT: str = "detections_out"
IN: str = "in"
OUT: str = "out"
DETECTIONS_IN_OUT_PARAM: str = "in_out"
SHORT_DESCRIPTION = "Count detections passing a line."
LONG_DESCRIPTION = """
Count objects crossing a defined line segment in video using tracked detections, maintaining separate counts for objects crossing in opposite directions (in and out), and outputting both count values and the actual detection objects that crossed the line for traffic analysis, people counting, entry/exit monitoring, and directional flow measurement workflows.

## How This Block Works

This block counts objects that cross a line segment by tracking their movement across video frames. The block:

1. Receives tracked detection predictions with unique tracker IDs and an image with embedded video metadata
2. Extracts video metadata from the image:
   - Accesses video_metadata from the WorkflowImageData object
   - Extracts video_identifier to maintain separate counting state for different videos
   - Uses video metadata to initialize and manage line zone state per video
3. Validates that detections have tracker IDs (required for tracking object movement across frames)
4. Initializes or retrieves a line zone for the video:
   - Creates a LineZone from two coordinate points defining the line segment
   - Configures triggering anchor point if specified (optional - if not specified, uses default anchor behavior)
   - Stores line zone configuration per video using video_identifier
   - Maintains separate counting state for each video
5. Monitors object positions across frames:
   - Tracks each object's position using its unique tracker_id
   - Detects when an object's triggering anchor point (if specified) or default anchor crosses the line
   - Determines crossing direction based on which side of the line the object approaches from
6. Counts line crossings:
   - **In Direction**: Objects crossing the line in one direction increment the count_in counter
   - **Out Direction**: Objects crossing the line in the opposite direction increment the count_out counter
   - Each unique tracker_id is counted only once per crossing (prevents duplicate counting if object oscillates near line)
7. Identifies crossing detections:
   - Creates masks identifying which detections crossed in each direction in the current frame
   - Filters detections to separate those that crossed "in" from those that crossed "out"
   - Returns the actual detection objects (not just counts) for further processing
8. Maintains persistent counting state:
   - Counts accumulate across frames for the entire video
   - State persists for each video until workflow execution completes
   - Separate counters for each unique video_identifier
9. Returns four outputs:
   - **count_in**: Total number of objects that crossed the line in the "in" direction (cumulative across video)
   - **count_out**: Total number of objects that crossed the line in the "out" direction (cumulative across video)
   - **detections_in**: Detection objects that crossed the line in the "in" direction (current frame crossings)
   - **detections_out**: Detection objects that crossed the line in the "out" direction (current frame crossings)

The line segment defines a virtual boundary in the video frame. The direction (in/out) is determined by which side of the line objects approach from - for a horizontal line, objects coming from above might count as "in" while objects from below count as "out" (or vice versa, depending on line orientation). The triggering anchor (if specified) determines which point on the bounding box must cross the line for the crossing to be counted - if not specified, the line zone uses its default anchor behavior. The count outputs provide cumulative totals across the video, while the detection outputs provide the actual objects that crossed in the current frame, enabling further analysis or visualization of crossing events.

## Common Use Cases

- **People Counting**: Count people entering and exiting buildings, stores, or events (e.g., count visitors entering store, track people entering/exiting building, monitor event attendance), enabling entry/exit counting workflows
- **Traffic Analysis**: Count vehicles passing through intersections or road segments (e.g., count vehicles crossing intersection, track traffic flow in specific directions, monitor vehicle passage at checkpoints), enabling traffic flow analysis workflows
- **Retail Analytics**: Track customer movement and foot traffic in retail spaces (e.g., count customers entering store sections, track movement between departments, monitor shopping flow patterns), enabling retail foot traffic analytics workflows
- **Security Monitoring**: Monitor entry and exit at secure areas or checkpoints (e.g., track entries to restricted areas, count people at access points, monitor checkpoint crossings), enabling security access monitoring workflows
- **Occupancy Management**: Track occupancy changes by counting objects entering and leaving spaces (e.g., count entries/exits to manage room capacity, track vehicle arrivals/departures in parking, monitor space occupancy changes), enabling occupancy tracking workflows
- **Wildlife Monitoring**: Count animals crossing defined paths or boundaries (e.g., track animal migration patterns, count wildlife crossing roads, monitor animal movement in habitats), enabling wildlife behavior analysis workflows

## Connecting to Other Blocks

This block receives tracked detections and an image with embedded video metadata, and produces count_in, count_out, detections_in, and detections_out:

- **After Byte Tracker blocks** to count tracked objects crossing lines (e.g., count tracked people crossing line, track vehicle crossings with consistent IDs, monitor tracked object movements), enabling tracking-to-counting workflows
- **After object detection or instance segmentation blocks** with tracking enabled to count detected objects (e.g., count detected vehicles, track people crossings, monitor object movements), enabling detection-to-counting workflows
- **Using detections_in or detections_out outputs** to process or visualize objects that crossed the line (e.g., visualize objects that crossed, analyze crossing objects, filter for crossing events), enabling crossing object analysis workflows
- **Before visualization blocks** to display line counter information and crossing objects (e.g., visualize line and counts, display crossing statistics, show crossing objects with annotations), enabling counting visualization workflows
- **Before data storage blocks** to record counting data and crossing events (e.g., log entry/exit counts, store traffic statistics, record crossing objects with metadata), enabling counting data logging workflows
- **Before notification blocks** to alert on count thresholds or crossing events (e.g., alert when count exceeds limit, notify on specific object crossings, trigger actions based on counts), enabling count-based notification workflows

## Version Differences

**Enhanced from v1:**

- **Detection Outputs**: Adds two new outputs (`detections_in` and `detections_out`) that provide the actual detection objects that crossed the line in each direction, not just count totals, enabling downstream processing and visualization of crossing objects
- **Simplified Input**: Uses `image` input that contains embedded video metadata instead of requiring a separate `metadata` field, simplifying workflow connections and reducing input complexity
- **Optional Triggering Anchor**: Makes `triggering_anchor` optional (default None) instead of required, allowing the line zone to use its default anchor behavior when no specific anchor is needed
- **Improved Integration**: Better integration with image-based workflows since video metadata is accessed directly from the image object rather than requiring separate metadata input

## Requirements

This block requires tracked detections with tracker_id information (detections must come from a tracking block like Byte Tracker). The line must be defined as a list of exactly 2 points, where each point is a list or tuple of exactly 2 coordinates (x, y). The image's video_metadata should include video_identifier to maintain separate counting state for different videos. The block maintains persistent counting state across frames for each video, so it should be used in video workflows where frames are processed sequentially. For accurate counting, detections should be provided consistently across frames with valid tracker IDs.
"""


class LineCounterManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Line Counter",
            "version": "v2",
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
    type: Literal["roboflow_core/line_counter@v2"]
    image: WorkflowImageSelector = Field(
        description="Image with embedded video metadata. The video_metadata contains video_identifier to maintain separate counting state for different videos. Required for persistent counting across frames.",
    )
    detections: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Tracked object detection or instance segmentation predictions. Must include tracker_id information from a tracking block. Objects are counted when their triggering anchor point (if specified) crosses the line segment. The detections_in and detections_out outputs provide the actual detection objects that crossed in each direction.",
        examples=["$steps.object_detection_model.predictions"],
    )

    line_segment: Union[list, Selector(kind=[LIST_OF_VALUES_KIND]), Selector(kind=[LIST_OF_VALUES_KIND])] = Field(  # type: ignore
        description="Line segment defined by exactly two points, each with [x, y] coordinates. Objects crossing from one side count as 'in', objects crossing from the other side count as 'out'. Example: [[0, 100], [500, 100]] creates a horizontal line at y=100. Crossing direction depends on which side objects approach from.",
        examples=[[[0, 50], [500, 50]], "$inputs.zones"],
    )
    triggering_anchor: Optional[Union[str, Selector(kind=[STRING_KIND]), Literal[tuple(sv.Position.list())]]] = Field(  # type: ignore
        description="Optional point on the bounding box that must cross the line for counting. If not specified (None), the line zone uses its default anchor behavior. Options when specified: CENTER, BOTTOM_CENTER, TOP_CENTER, CENTER_LEFT, CENTER_RIGHT, etc. Specifying CENTER ensures the object is substantially across the line before counting, reducing false positives from objects near but not fully crossing the line.",
        default=None,
        examples=["CENTER", None],
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
            OutputDefinition(
                name=OUTPUT_KEY_DETECTIONS_IN,
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                ],
            ),
            OutputDefinition(
                name=OUTPUT_KEY_DETECTIONS_OUT,
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class LineCounterBlockV2(WorkflowBlock):
    def __init__(self):
        self._batch_of_line_zones: Dict[str, sv.LineZone] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return LineCounterManifest

    def run(
        self,
        detections: sv.Detections,
        image: WorkflowImageData,
        line_segment: List[Tuple[int, int]],
        triggering_anchor: Optional[str] = None,
    ) -> BlockResult:
        if detections.tracker_id is None:
            raise ValueError(
                f"tracker_id not initialized, {self.__class__.__name__} requires detections to be tracked"
            )
        metadata = image.video_metadata
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
            if triggering_anchor is not None:
                self._batch_of_line_zones[metadata.video_identifier] = sv.LineZone(
                    start=sv.Point(*line_segment[0]),
                    end=sv.Point(*line_segment[1]),
                    triggering_anchors=[sv.Position(triggering_anchor)],
                )
            else:
                self._batch_of_line_zones[metadata.video_identifier] = sv.LineZone(
                    start=sv.Point(*line_segment[0]),
                    end=sv.Point(*line_segment[1]),
                )
        line_zone = self._batch_of_line_zones[metadata.video_identifier]

        mask_in, mask_out = line_zone.trigger(detections=detections)
        detections_in = detections[mask_in]
        detections_out = detections[mask_out]

        return {
            OUTPUT_KEY_COUNT_IN: line_zone.in_count,
            OUTPUT_KEY_COUNT_OUT: line_zone.out_count,
            OUTPUT_KEY_DETECTIONS_IN: detections_in,
            OUTPUT_KEY_DETECTIONS_OUT: detections_out,
        }
