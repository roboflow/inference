import hashlib
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import cv2 as cv
import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
    VisualizationBlock,
    VisualizationManifest,
)
from inference.core.workflows.core_steps.visualizations.common.utils import str_to_color
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    FloatZeroToOne,
    Selector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

TYPE: str = "roboflow_core/line_counter_visualization@v1"
SHORT_DESCRIPTION = "Apply a mask over a line zone in an image."
LONG_DESCRIPTION = """
Draw a line zone on an image to visualize counting boundaries, displaying a colored line overlay with in/out count labels for line counter workflows that track objects crossing a specified line.

## How This Block Works

This block takes an image and line zone coordinates (two points defining a line) and draws a visual representation of the counting line with count statistics. The block:

1. Takes an image and line zone coordinates (two points: [x1, y1] and [x2, y2]) as input
2. Creates a line mask from the zone coordinates using the specified color and thickness
3. Overlays the line onto the image with the specified opacity, creating a semi-transparent line visualization
4. Displays text labels showing the count_in (objects that crossed into the zone) and count_out (objects that crossed out of the zone) values
5. Positions the count text at the starting point of the line (x1, y1) with customizable text styling
6. Returns an annotated image with the line zone and count statistics overlaid on the original image

The block visualizes line counting zones used to track object movement across a defined boundary line. The line is drawn between the two specified points with customizable color, thickness, and opacity. Count statistics (in and out) are displayed as text labels, typically connected from a Line Counter block that tracks object crossings. The visualization helps users see the counting boundary and monitor counting results in real-time. Note: This block should typically be placed before other visualization blocks in the workflow, as the line zone provides a background reference layer for object detection visualizations.

## Common Use Cases

- **Line Counter Visualization**: Visualize line counting zones for people counting, vehicle counting, or object tracking workflows where objects cross a defined line boundary, displaying the counting line and in/out statistics
- **Traffic and Movement Monitoring**: Display counting lines for traffic monitoring, pedestrian flow analysis, or entry/exit tracking applications where you need to visualize the counting boundary and current counts
- **Checkpoint and Access Control**: Visualize counting lines at checkpoints, gates, or access points to show the monitoring boundary and track entry/exit counts for security or access control workflows
- **Retail and Business Analytics**: Display counting lines for foot traffic analysis, customer flow monitoring, or occupancy tracking in retail, hospitality, or business intelligence applications
- **Crowd Management and Safety**: Visualize counting lines for crowd management, capacity monitoring, or safety workflows where tracking object movement across boundaries is critical
- **Real-Time Counting Dashboards**: Create visual overlays for real-time counting dashboards, monitoring interfaces, or live video feeds where the counting line and statistics need to be clearly visible

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Line Counter blocks** to receive count_in and count_out values that are displayed on the visualization
- **Other visualization blocks** (e.g., Bounding Box Visualization, Label Visualization, Polygon Visualization) to add object detection annotations on top of the line zone visualization
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save images with line zone visualizations for documentation, reporting, or analysis
- **Webhook blocks** to send visualized results with line zones to external systems, APIs, or web applications for display in dashboards or monitoring tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with line zones as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with line zone visualizations for live monitoring, counting visualization, or post-processing analysis
"""


class LineCounterZoneVisualizationManifest(VisualizationManifest):
    type: Literal[f"{TYPE}"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Line Counter Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "search_keywords": ["annotator"],
            "ui_manifest": {
                "section": "visualization",
                "icon": "far fa-arrow-down-up-across-line",
                "blockPriority": 15,
                "supervision": True,
                "warnings": [
                    {
                        "property": "copy_image",
                        "value": False,
                        "message": "This setting will mutate its input image. If the input is used by other blocks, it may cause unexpected behavior.",
                    }
                ],
            },
        }
    )
    zone: Union[list, Selector(kind=[LIST_OF_VALUES_KIND]), Selector(kind=[LIST_OF_VALUES_KIND])] = Field(  # type: ignore
        description="Line zone coordinates in the format [[x1, y1], [x2, y2]] consisting of exactly two points that define the counting line. The line is drawn between these two points, and objects crossing this line are counted. Typically connected from a Line Counter block's zone output.",
        examples=[[[0, 50], [500, 50]], "$inputs.zones"],
    )
    color: Union[str, Selector(kind=[STRING_KIND])] = Field(  # type: ignore
        description="Color of the line zone. Can be specified as a color name (e.g., 'WHITE', 'RED'), hex color code (e.g., '#5bb573', '#FFFFFF'), or RGB format (e.g., 'rgb(255, 255, 255)'). The line is drawn in this color with the specified opacity.",
        default="#5bb573",
        examples=["WHITE", "#FFFFFF", "rgb(255, 255, 255)", "$inputs.background_color"],
    )
    thickness: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Thickness of the line zone in pixels. Controls how thick the counting line appears. Higher values create thicker, more visible lines, while lower values create thinner lines. Typical values range from 1 to 10 pixels.",
        default=2,
        examples=[2, "$inputs.thickness"],
    )
    text_thickness: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Thickness of the count text labels in pixels. Controls how bold the text appears (line width of text characters). Higher values create thicker, bolder text, while lower values create thinner text. Typical values range from 1 to 3.",
        default=1,
        examples=[1, "$inputs.text_thickness"],
    )
    text_scale: Union[float, Selector(kind=[FLOAT_KIND])] = Field(  # type: ignore
        description="Scale factor for the count text labels. Controls the size of the text displaying count_in and count_out values. Values greater than 1.0 make text larger, values less than 1.0 make text smaller. Typical values range from 0.5 to 2.0.",
        default=1.0,
        examples=[1.0, "$inputs.text_scale"],
    )
    count_in: Union[int, Selector(kind=[INTEGER_KIND]), Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Number of objects that crossed into the line zone (crossing from one side to the other in the 'in' direction). Typically connected from a Line Counter block's count_in output (e.g., '$steps.line_counter.count_in'). This value is displayed in the visualization text label.",
        default=0,
        examples=["$steps.line_counter.count_in"],
        json_schema_extra={"always_visible": True},
    )
    count_out: Union[int, Selector(kind=[INTEGER_KIND]), Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Number of objects that crossed out of the line zone (crossing from one side to the other in the 'out' direction). Typically connected from a Line Counter block's count_out output (e.g., '$steps.line_counter.count_out'). This value is displayed in the visualization text label.",
        default=0,
        examples=["$steps.line_counter.count_out"],
        json_schema_extra={"always_visible": True},
    )
    opacity: Union[FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        description="Opacity of the line zone overlay, ranging from 0.0 (fully transparent) to 1.0 (fully opaque). Controls how transparent the counting line appears over the image. Lower values create more transparent lines that blend with the background, while higher values create more opaque, visible lines. Typical values range from 0.2 to 0.5 for balanced visibility.",
        default=0.3,
        examples=[0.3, "$inputs.opacity"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class LineCounterZoneVisualizationBlockV1(VisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache: Dict[str, np.ndarray] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return LineCounterZoneVisualizationManifest

    def getAnnotator(
        self,
        **kwargs,
    ) -> sv.PolygonZoneAnnotator:
        pass

    def run(
        self,
        image: WorkflowImageData,
        zone: List[Tuple[int, int]],
        copy_image: bool,
        color: str,
        thickness: int,
        text_thickness: int,
        text_scale: int,
        count_in: int,
        count_out: int,
        opacity: float,
    ) -> BlockResult:
        h, w, *_ = image.numpy_image.shape
        zone_fingerprint = hashlib.md5(str(zone).encode()).hexdigest()
        key = f"{zone_fingerprint}_{color}_{opacity}_{w}_{h}"
        x1, y1 = zone[0]
        x2, y2 = zone[1]
        if key not in self._cache:
            mask = np.zeros(
                shape=image.numpy_image.shape,
                dtype=image.numpy_image.dtype,
            )
            mask = cv.line(
                img=mask,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=str_to_color(color).as_bgr(),
                thickness=thickness,
            )
            self._cache[key] = mask
        mask = self._cache[key].copy()

        np_image = image.numpy_image
        if copy_image:
            np_image = np_image.copy()
        annotated_image = cv.addWeighted(
            src1=mask,
            alpha=opacity,
            src2=np_image,
            beta=1,
            gamma=0,
        )
        annotated_image = sv.draw_text(
            scene=annotated_image,
            text=f"in: {count_in}, out: {count_out}",
            text_anchor=sv.Point(x1, y1),
            text_thickness=text_thickness,
            text_scale=text_scale,
            background_color=sv.Color.WHITE,
            text_padding=0,
        )
        return {
            OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                origin_image_data=image, numpy_image=annotated_image
            )
        }
