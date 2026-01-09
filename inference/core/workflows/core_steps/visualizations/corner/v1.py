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

TYPE: str = "roboflow_core/corner_visualization@v1"
SHORT_DESCRIPTION = "Draw the corners of detected objects in an image."
LONG_DESCRIPTION = """
Draw corner markers at the four corners of detected object bounding boxes, providing a minimal, clean visualization style that marks object locations without full bounding box outlines.

## How This Block Works

This block takes an image and detection predictions and draws corner markers at the four corners of each detected object's bounding box. The block:

1. Takes an image and predictions as input
2. Identifies bounding box coordinates for each detected object
3. Calculates the four corner positions (top-left, top-right, bottom-left, bottom-right) of each bounding box
4. Applies color styling based on the selected color palette, with colors assigned by class, index, or track ID
5. Draws corner markers (typically L-shaped lines or corner indicators) at each corner position using Supervision's BoxCornerAnnotator
6. Applies the specified thickness and corner length to control the appearance of the corner markers
7. Returns an annotated image with corner markers overlaid on the original image

The block draws minimal corner markers instead of full bounding boxes, creating a clean, unobtrusive visualization style. This approach marks object locations clearly while maintaining a minimal aesthetic that doesn't overwhelm the image with full rectangular outlines. The corner markers can be customized with different thickness and length values, and colors can be assigned based on object class, index, or tracking ID, making it easy to distinguish between different objects or object types.

## Common Use Cases

- **Minimal Object Marking**: Mark detected objects with corner indicators instead of full bounding boxes for a clean, unobtrusive visualization style that preserves image clarity while still indicating object locations
- **Aesthetic Visualization Design**: Create visually minimal annotations for presentations, dashboards, or user interfaces where full bounding boxes would be too visually intrusive but corner markers provide sufficient location indication
- **Dense Scene Visualization**: Use corner markers when working with many detected objects in dense scenes where full bounding boxes would overlap excessively and create visual clutter
- **Design-Oriented Applications**: Apply corner markers in design workflows, artistic visualizations, or creative applications where a minimal, modern aesthetic is preferred over traditional bounding box outlines
- **Subtle Object Highlighting**: Mark object locations subtly without drawing attention away from the main image content, useful for background annotations or when object location indication is needed without visual prominence
- **UI and Dashboard Integration**: Integrate corner markers into user interfaces, dashboards, or interactive applications where minimal visual indicators are preferred for better user experience and reduced visual noise

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Other visualization blocks** (e.g., Label Visualization, Dot Visualization, Bounding Box Visualization) to combine corner markers with additional annotations for comprehensive visualization
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save annotated images with corner markers for documentation, reporting, or analysis
- **Webhook blocks** to send visualized results with corner markers to external systems, APIs, or web applications for display in dashboards or monitoring tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with corner markers as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with corner markers for live monitoring, tracking visualization, or post-processing analysis
"""


class CornerManifest(ColorableVisualizationManifest):
    type: Literal[f"{TYPE}", "CornerVisualization"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Corner Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "search_keywords": ["annotator"],
            "ui_manifest": {
                "section": "visualization",
                "icon": "far fa-expand",
                "blockPriority": 7,
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
        description="Thickness of the corner marker lines in pixels. Higher values create thicker, more visible corner markers.",
        default=4,
        examples=[4, "$inputs.thickness"],
    )

    corner_length: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Length of each corner marker line segment in pixels. This controls how long the corner indicators extend from each corner point. Higher values create longer, more prominent corner markers.",
        default=15,
        examples=[15, "$inputs.corner_length"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class CornerVisualizationBlockV1(ColorableVisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotatorCache = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return CornerManifest

    def getAnnotator(
        self,
        color_palette: str,
        palette_size: int,
        custom_colors: List[str],
        color_axis: str,
        thickness: int,
        corner_length: int,
    ) -> sv.annotators.base.BaseAnnotator:
        key = "_".join(
            map(
                str,
                [
                    color_palette,
                    palette_size,
                    color_axis,
                    thickness,
                    corner_length,
                ],
            )
        )

        if key not in self.annotatorCache:
            palette = self.getPalette(color_palette, palette_size, custom_colors)

            self.annotatorCache[key] = sv.BoxCornerAnnotator(
                color=palette,
                color_lookup=getattr(sv.ColorLookup, color_axis),
                thickness=thickness,
                corner_length=corner_length,
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
        corner_length: Optional[int],
    ) -> BlockResult:
        annotator = self.getAnnotator(
            color_palette,
            palette_size,
            custom_colors,
            color_axis,
            thickness,
            corner_length,
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
