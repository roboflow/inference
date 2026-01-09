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

TYPE: str = "roboflow_core/circle_visualization@v1"
SHORT_DESCRIPTION = "Draw a circle around detected objects in an image."
LONG_DESCRIPTION = """
Draw circular outlines around detected objects, providing an alternative to rectangular bounding boxes with a softer, more rounded visualization style.

## How This Block Works

This block takes an image and detection predictions and draws circular outlines around each detected object. The block:

1. Takes an image and predictions as input
2. Calculates the center point and size for each detection based on its bounding box
3. Applies color styling based on the selected color palette, with colors assigned by class, index, or track ID
4. Draws circular outlines around each detected object using Supervision's CircleAnnotator
5. Applies the specified circle thickness to control the line width of the circular outlines
6. Returns an annotated image with circular outlines overlaid on the original image

The block draws circles that are typically centered on each detection's bounding box, with the circle size determined by the detection dimensions. Circles provide a softer, more organic visual style compared to rectangular bounding boxes, while still clearly marking the location and extent of detected objects. Unlike dot visualization (which marks specific points), circle visualization draws full circular outlines that encompass the detected objects, making it useful when you want a rounded geometric shape that's less angular than bounding boxes but more prominent than small dot markers.

## Common Use Cases

- **Soft Geometric Visualization**: Use circular outlines instead of rectangular bounding boxes for a softer, more organic visual style in presentations, dashboards, or user interfaces where rounded shapes are preferred
- **Object Highlighting with Rounded Shapes**: Highlight detected objects with circular outlines when working with circular or spherical objects (e.g., balls, coins, circular logos, round products) where circles naturally fit the object shape
- **Aesthetic Visualization Alternatives**: Create visually distinct annotations compared to standard bounding boxes for design purposes, artistic visualizations, or when circular shapes better match the overall design aesthetic
- **Detection Visualization with Variation**: Provide an alternative visualization style to bounding boxes for comparison, experimentation, or when multiple visualization types are used together to distinguish different detection sets
- **User Interface Design**: Use circular outlines in user interfaces, mobile apps, or interactive displays where rounded shapes are more visually appealing or match design guidelines
- **Scientific and Medical Imaging**: Visualize detections with circular outlines in scientific or medical imaging contexts where rounded shapes may be more appropriate than angular bounding boxes

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Other visualization blocks** (e.g., Label Visualization, Dot Visualization, Bounding Box Visualization) to combine circular outlines with additional annotations for comprehensive visualization
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save annotated images with circular outlines for documentation, reporting, or analysis
- **Webhook blocks** to send visualized results with circular outlines to external systems, APIs, or web applications for display in dashboards or monitoring tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with circular outlines as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with circular outlines for live monitoring, tracking visualization, or post-processing analysis
"""


class CircleManifest(ColorableVisualizationManifest):
    type: Literal[f"{TYPE}", "CircleVisualization"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Circle Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "search_keywords": ["annotator"],
            "ui_manifest": {
                "section": "visualization",
                "icon": "far fa-circle",
                "blockPriority": 5,
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
        description="Thickness of the circle outline in pixels. Higher values create thicker, more visible circular outlines.",
        default=2,
        examples=[2, "$inputs.thickness"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class CircleVisualizationBlockV1(ColorableVisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotatorCache = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return CircleManifest

    def getAnnotator(
        self,
        color_palette: str,
        palette_size: int,
        custom_colors: List[str],
        color_axis: str,
        thickness: int,
    ) -> sv.annotators.base.BaseAnnotator:
        key = "_".join(
            map(
                str,
                [
                    color_palette,
                    palette_size,
                    color_axis,
                    thickness,
                ],
            )
        )

        if key not in self.annotatorCache:
            palette = self.getPalette(color_palette, palette_size, custom_colors)

            self.annotatorCache[key] = sv.CircleAnnotator(
                color=palette,
                color_lookup=getattr(sv.ColorLookup, color_axis),
                thickness=thickness,
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
    ) -> BlockResult:
        annotator = self.getAnnotator(
            color_palette,
            palette_size,
            custom_colors,
            color_axis,
            thickness,
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
