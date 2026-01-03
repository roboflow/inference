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
    FLOAT_ZERO_TO_ONE_KIND,
    INTEGER_KIND,
    FloatZeroToOne,
    Selector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

TYPE: str = "roboflow_core/bounding_box_visualization@v1"
SHORT_DESCRIPTION = "Draw a box around detected objects in an image."
LONG_DESCRIPTION = """
Draw bounding boxes around detected objects in an image, with customizable colors, thickness, and corner roundness.

## How This Block Works

This block takes an image and detection predictions (from object detection, instance segmentation, or keypoint detection models) and draws rectangular bounding boxes around each detected object. The block:

1. Takes an image and predictions as input
2. Applies color styling based on the selected color palette, with colors assigned by class, index, or track ID
3. Draws bounding boxes using Supervision's BoxAnnotator (for square corners) or RoundBoxAnnotator (for rounded corners) based on the roundness setting
4. Applies the specified box thickness to control the line width of the bounding boxes
5. Returns an annotated image with bounding boxes overlaid on the original image

The block supports various color palettes (default, Roboflow, Matplotlib palettes, or custom colors) and can color boxes based on detection class, index, or tracker ID. When roundness is set to 0, square corners are used; when roundness is greater than 0, rounded corners are applied for a softer visual appearance. You can choose whether to modify the original image or create a copy for visualization, which is useful when stacking multiple visualization blocks.

## Common Use Cases

- **Model Validation and Debugging**: Visualize detection results to verify model performance, check bounding box accuracy, identify false positives or false negatives, and debug model outputs
- **Results Presentation**: Create annotated images for reports, dashboards, or presentations showing what objects were detected in images or video frames
- **Quality Control**: Overlay bounding boxes on production line images to visualize detected defects, products, or components for quality assurance workflows
- **Monitoring and Alerting**: Generate visual outputs showing detected objects for security monitoring, surveillance systems, or compliance tracking with annotated evidence
- **Training Data Review**: Review and validate training datasets by visualizing annotations and bounding boxes to ensure labeling accuracy and consistency
- **Interactive Applications**: Create user interfaces that display real-time detection results with bounding boxes for object tracking, counting, or identification applications

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Other visualization blocks** (e.g., Label Visualization, Polygon Visualization, Mask Visualization) to stack multiple annotations on the same image for comprehensive visualization
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save annotated images for documentation, archiving, or training data preparation
- **Webhook blocks** to send visualized results to external systems, APIs, or web applications for display in dashboards or monitoring tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with bounding boxes for live monitoring or post-processing analysis
"""


class BoundingBoxManifest(ColorableVisualizationManifest):
    type: Literal[f"{TYPE}", "BoundingBoxVisualization"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Bounding Box Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "search_keywords": ["annotator"],
            "ui_manifest": {
                "section": "visualization",
                "icon": "far fa-object-group",
                "blockPriority": 0,
                "supervision": True,
                "popular": True,
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
        description="Thickness of the bounding box edges in pixels. Higher values create thicker, more visible box outlines.",
        default=2,
        examples=[2, "$inputs.thickness"],
    )

    roundness: Union[FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        description="Roundness of the bounding box corners, ranging from 0.0 (square corners) to 1.0 (fully rounded corners). When set to 0.0, square-cornered boxes are used; higher values create progressively more rounded corners.",
        default=0.0,
        examples=[0.0, "$inputs.roundness"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class BoundingBoxVisualizationBlockV1(ColorableVisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotatorCache = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BoundingBoxManifest

    def getAnnotator(
        self,
        color_palette: str,
        palette_size: int,
        custom_colors: List[str],
        color_axis: str,
        thickness: int,
        roundness: float,
    ) -> sv.annotators.base.BaseAnnotator:
        key = "_".join(
            map(str, [color_palette, palette_size, color_axis, thickness, roundness])
        )

        if key not in self.annotatorCache:
            palette = self.getPalette(color_palette, palette_size, custom_colors)

            if roundness == 0:
                self.annotatorCache[key] = sv.BoxAnnotator(
                    color=palette,
                    color_lookup=getattr(sv.ColorLookup, color_axis),
                    thickness=thickness,
                )
            else:
                self.annotatorCache[key] = sv.RoundBoxAnnotator(
                    color=palette,
                    color_lookup=getattr(sv.ColorLookup, color_axis),
                    thickness=thickness,
                    roundness=roundness,
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
        roundness: Optional[float],
    ) -> BlockResult:
        annotator = self.getAnnotator(
            color_palette,
            palette_size,
            custom_colors,
            color_axis,
            thickness,
            roundness,
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
