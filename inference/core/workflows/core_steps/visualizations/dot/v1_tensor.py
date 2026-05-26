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

TYPE: str = "roboflow_core/dot_visualization@v1"
SHORT_DESCRIPTION = (
    "Draw dots on an image at specific coordinates based on provided detections."
)
LONG_DESCRIPTION = """
Draw circular dots on an image to mark specific points on detected objects, with customizable position, size, color, and outline styling.

## How This Block Works

This block takes an image and detection predictions and draws circular dot markers at specified anchor positions on each detected object. The block:

1. Takes an image and predictions as input
2. Determines the dot position for each detection based on the selected anchor point (center, corners, edges, or center of mass)
3. Applies color styling based on the selected color palette, with colors assigned by class, index, or track ID
4. Draws circular dots with the specified radius and optional outline thickness using Supervision's DotAnnotator
5. Returns an annotated image with dots overlaid on the original image

The block supports various position options including the center of the bounding box, any of the four corners, edge midpoints, or the center of mass (useful for objects with irregular shapes). Dots can be customized with different sizes (radius), optional outlines for better visibility, and various color palettes. This provides a minimal, clean visualization style that marks detection locations without the visual clutter of full bounding boxes, making it ideal for dense scenes or when you need to highlight specific points of interest.

## Common Use Cases

- **Minimal Object Marking**: Mark detected objects with small dots instead of bounding boxes for cleaner, less cluttered visualizations when working with dense scenes or many detections
- **Point of Interest Highlighting**: Mark specific anchor points (corners, center, center of mass) on detected objects for applications like object tracking, pose estimation, or spatial analysis
- **Tracking Visualization**: Use dots to visualize object trajectories or tracking IDs over time, creating a cleaner alternative to bounding boxes for tracking workflows
- **Crowd Counting and Density Analysis**: Mark people or objects with dots to visualize density patterns, crowd distribution, or object counts without overlapping bounding boxes
- **Keypoint and Landmark Marking**: Mark specific points on objects (such as the center of mass for irregular shapes) for physics simulations, measurement workflows, or spatial relationship analysis
- **Minimal UI Overlays**: Create clean, unobtrusive visual overlays for user interfaces, dashboards, or mobile applications where full bounding boxes would be too visually intrusive

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Other visualization blocks** (e.g., Bounding Box Visualization, Label Visualization, Trace Visualization) to combine dot markers with additional annotations for comprehensive visualization
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save annotated images with dot markers for documentation, reporting, or analysis
- **Webhook blocks** to send visualized results with dot markers to external systems, APIs, or web applications for display in dashboards or monitoring tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with dot markers as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with dot markers for live monitoring, tracking visualization, or post-processing analysis
"""


class DotManifest(ColorableVisualizationManifest):
    type: Literal[f"{TYPE}", "DotVisualization"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Dot Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "search_keywords": ["annotator"],
            "ui_manifest": {
                "section": "visualization",
                "icon": "far fa-palette",
                "blockPriority": 1,
                "opencv": True,
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
        description="Anchor position for placing the dot relative to each detection's bounding box. Options include: CENTER (center of box), corners (TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT), edge midpoints (TOP_CENTER, CENTER_LEFT, CENTER_RIGHT, BOTTOM_CENTER), or CENTER_OF_MASS (center of mass of the object, useful for irregular shapes).",
        default="CENTER",
        examples=["CENTER", "$inputs.position"],
    )

    radius: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Radius of the dot in pixels. Higher values create larger, more visible dots.",
        default=4,
        examples=[4, "$inputs.radius"],
    )

    outline_thickness: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Thickness of the dot outline in pixels. Set to 0 for no outline (filled dots only). Higher values create thicker outlines around the dot for better visibility against varying backgrounds.",
        default=0,
        examples=[2, "$inputs.outline_thickness"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class DotVisualizationBlockV1(ColorableVisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotatorCache = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return DotManifest

    def getAnnotator(
        self,
        color_palette: str,
        palette_size: int,
        custom_colors: List[str],
        color_axis: str,
        position: str,
        radius: int,
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
                    radius,
                    outline_thickness,
                ],
            )
        )

        if key not in self.annotatorCache:
            palette = self.getPalette(color_palette, palette_size, custom_colors)

            self.annotatorCache[key] = sv.DotAnnotator(
                color=palette,
                color_lookup=getattr(sv.ColorLookup, color_axis),
                position=getattr(sv.Position, position),
                radius=radius,
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
        radius: Optional[int],
        outline_thickness: Optional[int],
    ) -> BlockResult:
        annotator = self.getAnnotator(
            color_palette,
            palette_size,
            custom_colors,
            color_axis,
            position,
            radius,
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
