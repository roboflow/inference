from typing import List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field
from supervision import PolygonAnnotator

from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
)
from inference.core.workflows.core_steps.visualizations.common.base_colorable import (
    ColorableVisualizationBlock,
    ColorableVisualizationManifest,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

TYPE: str = "roboflow_core/polygon_visualization@v2"
SHORT_DESCRIPTION = "Draw a polygon around detected objects in an image."
LONG_DESCRIPTION = """
Draw polygon outlines around detected objects that follow the exact shape of object masks, providing precise boundary visualization for instance segmentation results.

## How This Block Works

This block takes an image and instance segmentation predictions (which include segmentation masks) and draws polygon outlines that precisely follow the shape of each detected object. The block:

1. Takes an image and instance segmentation predictions as input (predictions must include mask data)
2. Converts segmentation masks to polygon coordinates that trace the object boundaries
3. Applies color styling based on the selected color palette, with colors assigned by class, index, or track ID
4. Draws polygon outlines with the specified thickness using the PolygonAnnotator
5. Returns an annotated image with polygon outlines overlaid on the original image

The block extracts the exact shape of each object from its segmentation mask and draws polygon outlines that follow these precise boundaries. This provides much more accurate visualization than bounding boxes, as polygons conform to the actual object shape rather than enclosing them in rectangles. If mask data is not available, the block falls back to drawing bounding boxes. The polygon outlines can be customized with different thickness values and color palettes, allowing you to clearly distinguish between different objects or object classes.

## Common Use Cases

- **Precise Object Boundary Visualization**: Visualize the exact shape and boundaries of segmented objects for applications requiring accurate object outlines, such as medical imaging, manufacturing quality control, or precise measurement workflows
- **Instance Segmentation Model Validation**: Verify and debug instance segmentation model performance by visualizing how well polygon predictions match object boundaries, identify segmentation errors, and validate mask quality
- **Irregular Shape Analysis**: Visualize objects with irregular or non-rectangular shapes (e.g., people, animals, complex machinery parts) where bounding boxes would be inaccurate or misleading
- **Overlapping Object Visualization**: Clearly show object boundaries when multiple objects overlap, as polygons accurately represent each object's shape without the ambiguity of overlapping bounding boxes
- **Shape-Based Quality Control**: Inspect object shapes and boundaries in manufacturing, agriculture, or quality assurance workflows where precise object contours are critical for defect detection or classification
- **Scientific and Medical Imaging**: Visualize segmented regions in medical imaging, microscopy, or scientific analysis where accurate boundary representation is essential for measurement, analysis, or diagnosis

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Other visualization blocks** (e.g., Label Visualization, Mask Visualization, Bounding Box Visualization) to combine polygon outlines with additional annotations for comprehensive visualization
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save annotated images with polygon outlines for documentation, reporting, or training data validation
- **Webhook blocks** to send visualized results with polygon outlines to external systems, APIs, or web applications for display in dashboards or analysis tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with polygon outlines as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with polygon outlines for live monitoring, tracking, or post-processing analysis
"""


class PolygonManifest(ColorableVisualizationManifest):
    type: Literal[TYPE]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Polygon Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "search_keywords": ["annotator"],
            "ui_manifest": {
                "section": "visualization",
                "icon": "far fa-shapes",
                "blockPriority": 1,
                "popular": True,
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

    predictions: Selector(
        kind=[
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Instance segmentation predictions containing mask data. The block converts masks to polygon outlines that follow the exact shape of each detected object.",
        examples=["$steps.instance_segmentation_model.predictions"],
    )

    thickness: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Thickness of the polygon outline in pixels. Higher values create thicker, more visible outlines.",
        default=2,
        examples=[2, "$inputs.thickness"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class PolygonVisualizationBlockV2(ColorableVisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotatorCache = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return PolygonManifest

    def getAnnotator(
        self,
        color_palette: str,
        palette_size: int,
        custom_colors: List[str],
        color_axis: str,
        thickness: int,
    ) -> PolygonAnnotator:
        key = "_".join(
            map(
                str,
                [
                    color_palette,
                    palette_size,
                    color_axis,
                    thickness,
                    "_".join(custom_colors) if custom_colors else "",
                ],
            )
        )

        if key not in self.annotatorCache:
            palette = self.getPalette(color_palette, palette_size, custom_colors)

            self.annotatorCache[key] = PolygonAnnotator(
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
