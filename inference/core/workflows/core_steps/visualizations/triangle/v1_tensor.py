from typing import List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
)
from inference.core.workflows.core_steps.visualizations.common.base_colorable import (
    ColorableVisualizationBlock,
    ColorableVisualizationManifest,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.execution_engine.entities.types import (
    INTEGER_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

TYPE: str = "roboflow_core/triangle_visualization@v1"
SHORT_DESCRIPTION = "Draw triangle markers on an image at specific coordinates based on provided detections."
LONG_DESCRIPTION = """
Draw triangular markers on an image to mark specific points on detected objects, with customizable position, size, color, and outline styling, providing directional indicators and geometric markers for visual annotations.

## How This Block Works

This block takes an image and detection predictions and draws triangular markers at specified anchor positions on each detected object. The block:

1. Takes an image and predictions as input
2. Determines the triangle position for each detection based on the selected anchor point (center, corners, edges, or center of mass)
3. Applies color styling based on the selected color palette, with colors assigned by class, index, or track ID
4. Draws triangular markers with the specified base width and height dimensions, with optional outline thickness using Supervision's TriangleAnnotator
5. Returns an annotated image with triangular markers overlaid on the original image

The block supports various position options including the center of the bounding box, any of the four corners, edge midpoints, or the center of mass (useful for objects with irregular shapes). Triangles can be customized with different sizes (base width and height), optional outlines for better visibility, and various color palettes. Triangular markers provide a distinctive geometric shape that can serve as directional indicators (pointing in a specific direction) or geometric markers, offering an alternative to circular dots or square markers. This provides a clean visualization style that marks detection locations with directional or geometric emphasis, making it ideal for applications requiring directional indicators, geometric markers, or distinctive point markers.

## Common Use Cases

- **Directional Object Marking**: Mark detected objects with triangular markers that can indicate direction or orientation, useful for tracking applications, motion analysis, or directional workflows where the triangle's pointed shape provides directional information
- **Geometric Marker Visualization**: Use triangular markers as distinctive geometric shapes to mark detection locations, providing visual variety compared to circular dots or rectangular bounding boxes for design purposes or geometric emphasis
- **Minimal Object Marking**: Mark detected objects with small triangular markers instead of bounding boxes for cleaner, less cluttered visualizations when working with dense scenes or many detections, with the triangular shape providing visual distinction
- **Tracking Visualization**: Use triangular markers to visualize object trajectories or tracking IDs over time, creating a cleaner alternative to bounding boxes for tracking workflows, with triangles potentially indicating movement direction
- **Point of Interest Highlighting**: Mark specific anchor points (corners, center, center of mass) on detected objects with triangular markers for applications like object tracking, spatial analysis, or geometric annotation workflows
- **Design and Aesthetic Applications**: Create triangular markers for design purposes, user interfaces, dashboards, or artistic visualizations where geometric shapes provide distinctive visual style or aesthetic appeal

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Other visualization blocks** (e.g., Bounding Box Visualization, Label Visualization, Trace Visualization) to combine triangular markers with additional annotations for comprehensive visualization
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save annotated images with triangular markers for documentation, reporting, or analysis
- **Webhook blocks** to send visualized results with triangular markers to external systems, APIs, or web applications for display in dashboards or monitoring tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with triangular markers as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with triangular markers for live monitoring, tracking visualization, or post-processing analysis
"""


class TriangleManifest(ColorableVisualizationManifest):
    type: Literal[f"{TYPE}", "TriangleVisualization"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Triangle Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "search_keywords": ["annotator"],
            "ui_manifest": {
                "section": "visualization",
                "icon": "far fa-triangle",
                "blockPriority": 14,
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

    position: Union[
        Literal[
            "CENTER",
            "CENTER_LEFT",
            "CENTER_RIGHT",
            "TOP_CENTER",
            "TOP_LEFT",
            "TOP_RIGHT",
            "BOTTOM_LEFT",
            "BOTTOM_CENTER",
            "BOTTOM_RIGHT",
            "CENTER_OF_MASS",
        ],
        Selector(kind=[STRING_KIND]),
    ] = Field(  # type: ignore
        default="TOP_CENTER",
        description="Anchor position for placing the triangle marker relative to the detection's bounding box. Options include: CENTER (center of box), corners (TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT), edge midpoints (TOP_CENTER, CENTER_LEFT, CENTER_RIGHT, BOTTOM_CENTER), or CENTER_OF_MASS (center of mass of the object, useful for irregular shapes).",
        examples=["CENTER", "$inputs.position"],
    )

    base: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Base width of the triangle in pixels. Controls the horizontal width of the triangular marker at its base. Larger values create wider triangles, while smaller values create narrower triangles. Works together with height to control the overall triangle size and shape.",
        default=10,
        examples=[10, "$inputs.base"],
    )

    height: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Height of the triangle in pixels. Controls the vertical height of the triangular marker from base to tip. Larger values create taller triangles, while smaller values create shorter triangles. Works together with base to control the overall triangle size and shape.",
        default=10,
        examples=[10, "$inputs.height"],
    )

    outline_thickness: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Thickness of the triangle outline in pixels. A value of 0 creates a filled triangle with no outline. Higher values create thicker outlines around the triangle border, improving visibility and contrast. Useful for making triangular markers more visible against complex backgrounds.",
        default=0,
        examples=[2, "$inputs.outline_thickness"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class TriangleVisualizationBlockV1(ColorableVisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotatorCache = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return TriangleManifest

    def getAnnotator(
        self,
        color_palette: str,
        palette_size: int,
        custom_colors: List[str],
        color_axis: str,
        position: str,
        base: int,
        height: int,
        outline_thickness: int,
    ) -> sv.annotators.base.BaseAnnotator:
        key = "_".join(
            map(
                str,
                [
                    color_palette,
                    palette_size,
                    color_axis,
                    position,
                    base,
                    height,
                    outline_thickness,
                ],
            )
        )

        if key not in self.annotatorCache:
            palette = self.getPalette(color_palette, palette_size, custom_colors)

            self.annotatorCache[key] = sv.TriangleAnnotator(
                color=palette,
                color_lookup=getattr(sv.ColorLookup, color_axis),
                position=getattr(sv.Position, position),
                base=base,
                height=height,
                outline_thickness=outline_thickness,
            )

        return self.annotatorCache[key]

    def run(
        self,
        image: WorkflowImageData,
        predictions: sv.Detections,
        copy_image: bool,
        color_palette: Optional[str],
        palette_size: Optional[int],
        custom_colors: Optional[List[str]],
        color_axis: Optional[str],
        position: Optional[str],
        base: Optional[int],
        height: Optional[int],
        outline_thickness: Optional[int],
    ) -> BlockResult:
        annotator = self.getAnnotator(
            color_palette,
            palette_size,
            custom_colors,
            color_axis,
            position,
            base,
            height,
            outline_thickness,
        )
        annotated_image = annotator.annotate(
            scene=image.numpy_image.copy() if copy_image else image.numpy_image,
            detections=predictions,
        )
        return {
            OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                origin_image_data=image, numpy_image=annotated_image
            )
        }
