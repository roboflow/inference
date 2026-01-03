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
    Selector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

TYPE: str = "roboflow_core/ellipse_visualization@v1"
SHORT_DESCRIPTION = "Draw ellipses that highlight detected objects in an image."
LONG_DESCRIPTION = """
Draw elliptical outlines around detected objects, providing oval-shaped annotations that can be customized to full ellipses or partial arcs, offering more flexible geometry than circular visualizations.

## How This Block Works

This block takes an image and detection predictions and draws elliptical (oval-shaped) outlines around each detected object. The block:

1. Takes an image and predictions as input
2. Identifies bounding box coordinates for each detected object
3. Calculates the center point and dimensions (width and height) for each detection based on its bounding box
4. Applies color styling based on the selected color palette, with colors assigned by class, index, or track ID
5. Draws elliptical outlines around each detected object using Supervision's EllipseAnnotator
6. Applies the specified thickness to control the line width of the elliptical outlines
7. Uses start and end angle parameters to draw either full ellipses or partial elliptical arcs
8. Returns an annotated image with elliptical outlines overlaid on the original image

The block draws ellipses that are typically fitted to each detection's bounding box, creating oval-shaped outlines that adapt to the object's aspect ratio (unlike circles which are always round). Ellipses provide more flexible geometry than circles, as they can represent both round and elongated objects more accurately. The start and end angle parameters allow you to draw full ellipses (360 degrees) or partial arcs, providing additional visual style options. Unlike circle visualization (which always draws complete circles), ellipse visualization can draw partial arcs, creating distinctive visual markers while still clearly indicating object locations.

## Common Use Cases

- **Oval and Elongated Object Highlighting**: Highlight detected objects with elliptical outlines when objects are oval-shaped or elongated (e.g., vehicles, elongated products, elliptical objects) where ellipses provide a better fit than circular or rectangular shapes
- **Flexible Geometric Visualization**: Use elliptical shapes as a more geometrically flexible alternative to circles or bounding boxes, adapting to object aspect ratios for more accurate visual representation
- **Partial Arc Annotations**: Draw partial elliptical arcs (using start and end angles) to create distinctive, stylized visual markers that indicate object locations with a unique visual style
- **Aesthetic Visualization Alternatives**: Create visually distinct annotations compared to standard bounding boxes or circles for design purposes, artistic visualizations, or when elliptical shapes better match design aesthetics
- **Object Shape Adaptation**: Use ellipses when objects have non-square aspect ratios where circular outlines would be less accurate, providing better visual fit for rectangular or elongated objects
- **Design and UI Applications**: Integrate elliptical outlines into user interfaces, dashboards, or interactive applications where oval shapes provide a softer, more organic visual style than angular rectangles

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Other visualization blocks** (e.g., Label Visualization, Dot Visualization, Bounding Box Visualization) to combine elliptical outlines with additional annotations for comprehensive visualization
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save annotated images with elliptical outlines for documentation, reporting, or analysis
- **Webhook blocks** to send visualized results with elliptical outlines to external systems, APIs, or web applications for display in dashboards or monitoring tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with elliptical outlines as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with elliptical outlines for live monitoring, tracking visualization, or post-processing analysis
"""


class EllipseManifest(ColorableVisualizationManifest):
    type: Literal[f"{TYPE}", "EllipseVisualization"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Ellipse Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "search_keywords": ["annotator"],
            "ui_manifest": {
                "section": "visualization",
                "icon": "fad fa-dot-circle",
                "blockPriority": 10,
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

    thickness: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Thickness of the ellipse outline in pixels. Higher values create thicker, more visible elliptical outlines.",
        default=2,
        examples=[2, "$inputs.thickness"],
    )

    start_angle: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Starting angle for drawing the ellipse arc in degrees. Used together with end_angle to control whether a full ellipse (360 degrees) or partial arc is drawn. Angles are measured from the positive x-axis (0 degrees = right, 90 degrees = down, 180 degrees = left, 270 degrees = up).",
        default=-45,
        examples=[-45, "$inputs.start_angle"],
    )

    end_angle: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Ending angle for drawing the ellipse arc in degrees. Used together with start_angle to control whether a full ellipse (360 degrees) or partial arc is drawn. To draw a full ellipse, set end_angle = start_angle + 360 (or equivalent). Angles are measured from the positive x-axis (0 degrees = right, 90 degrees = down, 180 degrees = left, 270 degrees = up).",
        default=235,
        examples=[235, "$inputs.end_angle"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class EllipseVisualizationBlockV1(ColorableVisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotatorCache = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return EllipseManifest

    def getAnnotator(
        self,
        color_palette: str,
        palette_size: int,
        custom_colors: List[str],
        color_axis: str,
        thickness: int,
        start_angle: int,
        end_angle: int,
    ) -> sv.annotators.base.BaseAnnotator:
        key = "_".join(
            map(
                str,
                [
                    color_palette,
                    palette_size,
                    color_axis,
                    thickness,
                    start_angle,
                    end_angle,
                ],
            )
        )

        if key not in self.annotatorCache:
            palette = self.getPalette(color_palette, palette_size, custom_colors)

            self.annotatorCache[key] = sv.EllipseAnnotator(
                color=palette,
                color_lookup=getattr(sv.ColorLookup, color_axis),
                thickness=thickness,
                start_angle=start_angle,
                end_angle=end_angle,
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
        thickness: Optional[int],
        start_angle: Optional[int],
        end_angle: Optional[int],
    ) -> BlockResult:
        annotator = self.getAnnotator(
            color_palette,
            palette_size,
            custom_colors,
            color_axis,
            thickness,
            start_angle,
            end_angle,
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
