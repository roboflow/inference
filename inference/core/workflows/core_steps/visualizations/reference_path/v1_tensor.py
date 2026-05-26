from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import cv2
import numpy as np
from pydantic import ConfigDict, Field, field_validator

from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
    VisualizationManifest,
)
from inference.core.workflows.core_steps.visualizations.common.utils import str_to_color
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.execution_engine.entities.types import (
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = "Draw a reference path in the image."
LONG_DESCRIPTION = """
Draw a static reference path on an image to visualize an expected or ideal route, displaying a predefined polyline path that can be compared against actual object trajectories for path deviation analysis and route compliance monitoring.

## How This Block Works

This block takes an image and reference path coordinates (a list of points defining a path) and draws a static polyline path representing an expected route or ideal trajectory. The block:

1. Takes an image and reference path coordinates (a list of points: [(x1, y1), (x2, y2), (x3, y3), ...]) as input
2. Converts the coordinate list into a polyline path connecting the points in sequence
3. Draws the reference path as a polyline using the specified color and thickness
4. Returns an annotated image with the reference path overlaid on the original image

The block visualizes a static, predefined reference path that represents where objects should ideally move or what route they should follow. Unlike Trace Visualization (which draws dynamic paths based on actual tracked object movement), Reference Path Visualization draws a fixed path that remains constant. This reference path serves as a baseline for comparison, allowing you to visualize the expected route alongside actual object trajectories. The path is drawn as a continuous line connecting all the specified points, creating a visual guide for route compliance, path deviation analysis, or navigation workflows. This block is commonly used with Path Deviation analytics blocks to visually display the reference path that actual object trajectories will be compared against.

## Common Use Cases

- **Path Deviation Visualization**: Visualize a reference path alongside actual object trajectories to compare expected routes against actual movement for path deviation detection, route compliance monitoring, or navigation validation workflows
- **Route Planning and Navigation**: Display predefined routes, navigation paths, or expected travel routes that objects should follow for route planning, navigation systems, or waypoint visualization applications
- **Compliance and Safety Monitoring**: Visualize expected paths for safety monitoring, compliance workflows, or route validation where objects need to follow specific paths (e.g., vehicles on designated lanes, robots on expected routes)
- **Industrial and Logistics Applications**: Display reference paths for conveyor systems, automated guided vehicles (AGVs), or manufacturing workflows where objects must follow predefined routes for process control or quality assurance
- **Security and Access Control**: Visualize expected movement paths for security monitoring, access control, or surveillance workflows where deviations from expected routes need to be identified
- **Training and Documentation**: Display reference paths in training materials, documentation, or demonstrations to show expected object behavior, routes, or movement patterns for educational or reference purposes

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Path Deviation analytics blocks** to compare tracked object trajectories against the visualized reference path for deviation analysis
- **Other visualization blocks** (e.g., Trace Visualization, Bounding Box Visualization, Label Visualization) to combine reference path visualization with actual object tracking visualizations for comprehensive path comparison
- **Tracking blocks** (e.g., Byte Tracker) where the reference path can serve as a visual baseline for comparing actual tracked object trajectories
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save images with reference paths for documentation, reporting, or analysis
- **Webhook blocks** to send visualized results with reference paths to external systems, APIs, or web applications for display in dashboards or monitoring tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with reference paths as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with reference paths for live monitoring, path visualization, or post-processing analysis
"""


class ReferencePathVisualizationManifest(VisualizationManifest):
    type: Literal["roboflow_core/reference_path_visualization@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Reference Path Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "search_keywords": ["annotator"],
            "ui_manifest": {
                "section": "visualization",
                "icon": "fas fa-road",
                "blockPriority": 18,
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
    reference_path: Union[
        list,
        Selector(kind=[LIST_OF_VALUES_KIND]),
        Selector(kind=[LIST_OF_VALUES_KIND]),
    ] = Field(  # type: ignore
        description="Reference path coordinates in the format [(x1, y1), (x2, y2), (x3, y3), ...] defining the expected or ideal route. The path is drawn as a polyline connecting these points in sequence, creating a continuous line representing the reference trajectory. Typically connected from Path Deviation analytics blocks or defined manually as an expected route. Must contain at least two points to form a valid path.",
        examples=["$inputs.expected_path"],
    )
    color: Union[str, Selector(kind=[STRING_KIND])] = Field(  # type: ignore
        description="Color of the reference path. Can be specified as a color name (e.g., 'WHITE', 'GREEN', 'BLUE'), hex color code (e.g., '#5bb573', '#FFFFFF'), or RGB format (e.g., 'rgb(91, 181, 115)'). The reference path is drawn in this color with the specified thickness.",
        default="#5bb573",
        examples=["WHITE", "#FFFFFF", "rgb(255, 255, 255)", "$inputs.background_color"],
    )
    thickness: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Thickness of the reference path line in pixels. Controls how thick the reference path appears. Higher values create thicker, more visible paths, while lower values create thinner, more subtle paths. Must be greater than or equal to zero. Typical values range from 1 to 5 pixels.",
        default=2,
        examples=[2, "$inputs.thickness"],
    )

    @field_validator("thickness")
    @classmethod
    def validate_thickness_greater_than_zero(
        cls, value: Union[int, str]
    ) -> Union[int, str]:
        if isinstance(value, int) and value <= 0:
            raise ValueError("Thickness must be greater or equal to zero")
        return value

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class ReferencePathVisualizationBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache: Dict[str, np.ndarray] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ReferencePathVisualizationManifest

    def run(
        self,
        image: WorkflowImageData,
        reference_path: List[Union[Tuple[int, int], List[int]]],
        copy_image: bool,
        color: str,
        thickness: int,
    ) -> BlockResult:
        reference_path_array = np.array(reference_path)[:, :2].astype(np.int32)
        numpy_image = image.numpy_image
        result_image = cv2.polylines(
            numpy_image if not copy_image else numpy_image.copy(),
            [reference_path_array],
            False,
            str_to_color(color).as_bgr(),
            thickness,
        )
        return {
            OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                origin_image_data=image,
                numpy_image=result_image,
            )
        }
