from typing import Dict, List, Literal, Optional, Type, Union

import numpy as np
import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field, model_validator

from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
    VisualizationBlock,
    VisualizationManifest,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

TYPE: str = "roboflow_core/icon_visualization@v1"
SHORT_DESCRIPTION = "Draw icons on an image either at specific static coordinates or dynamically based on detections."
LONG_DESCRIPTION = """
Place custom icon images on images either at fixed positions (static mode) or dynamically positioned on detected objects (dynamic mode), useful for watermarks, labels, badges, or visual markers.

## How This Block Works

This block takes an image and optionally detection predictions, then places a custom icon image on the image. The block supports two modes:

**Static Mode** (for watermarks and fixed positioning):
1. Takes an image and an icon image as input
2. Places the icon at fixed x and y coordinates on the image
3. Supports negative coordinates for positioning from the right or bottom edges
4. Returns an annotated image with the icon at the specified static location

**Dynamic Mode** (for detection-based positioning):
1. Takes an image, an icon image, and detection predictions as input
2. Positions the icon on each detected object based on the selected anchor point (center, corners, edges, or center of mass)
3. Places the icon at the same position relative to each detection
4. Returns an annotated image with icons overlaid on detected objects

The block supports PNG images with transparency (alpha channel), allowing icons to blend naturally with the background. Icons can be resized to any width and height, making them suitable for various use cases from small badges to large watermarks. In static mode, icons are placed at fixed coordinates, making it ideal for watermarks or branding. In dynamic mode, icons automatically follow detected objects, making it useful for labeling, categorizing, or marking detected items with custom visual indicators.

## Common Use Cases

- **Watermarks and Branding**: Place logos, watermarks, or branding elements at fixed positions (static mode) on images or videos for content protection, copyright marking, or brand identification
- **Object Labeling with Icons**: Place custom icons on detected objects (dynamic mode) to categorize, label, or mark objects with visual indicators (e.g., warning icons on unsafe objects, category icons for products, status badges)
- **Visual Status Indicators**: Display status icons (e.g., checkmarks, warning signs, information badges) on detected objects based on classification results, confidence levels, or custom logic for quick visual feedback
- **Product Marking and Categorization**: Place category icons, product type indicators, or custom markers on detected products in retail, e-commerce, or inventory management workflows
- **Custom Annotation Systems**: Create custom annotation workflows with specialized icons for quality control, defect marking, or compliance tracking in manufacturing or inspection workflows
- **Interactive UI Elements**: Add icon-based visual elements to images or videos for user interfaces, dashboards, or interactive applications where custom icons provide intuitive visual cues

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Other visualization blocks** (e.g., Label Visualization, Bounding Box Visualization, Polygon Visualization) to combine icon placement with additional annotations for comprehensive visualization
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save images with icons for documentation, reporting, or archiving
- **Webhook blocks** to send visualized results with icons to external systems, APIs, or web applications for display in dashboards or monitoring tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with icons as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with icons for live monitoring, tracking visualization, or post-processing analysis
"""


class IconManifest(VisualizationManifest):
    type: Literal[f"{TYPE}", "IconVisualization"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Icon Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "search_keywords": ["annotator", "icon", "watermark"],
            "ui_manifest": {
                "section": "visualization",
                "icon": "far fa-image",
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

    icon: Selector(kind=[IMAGE_KIND]) = Field(
        title="Icon Image",
        description="The icon image to place on the input image. PNG format with transparency (alpha channel) is recommended for best results, as it allows the icon to blend naturally with the background. The icon will be resized to the specified width and height.",
        examples=["$inputs.icon", "$steps.image_loader.image"],
        json_schema_extra={
            "always_visible": True,
            "order": 3,
        },
    )

    mode: Union[
        Literal["static", "dynamic"],
        Selector(kind=[STRING_KIND]),
    ] = Field(
        default="dynamic",
        description="Mode for placing icons. 'static' mode places the icon at fixed x,y coordinates (useful for watermarks or fixed-position elements). 'dynamic' mode places icons on detected objects based on their positions (useful for object labeling or categorization).",
        examples=["static", "dynamic", "$inputs.mode"],
        json_schema_extra={
            "always_visible": True,
            "order": 1,
        },
    )

    predictions: Optional[
        Selector(
            kind=[
                OBJECT_DETECTION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
                RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
                KEYPOINT_DETECTION_PREDICTION_KIND,
            ]
        )
    ] = Field(
        default=None,
        description="Model predictions to place icons on (required for dynamic mode). Icons will be positioned on each detected object based on the selected position anchor point.",
        examples=["$steps.object_detection_model.predictions"],
        json_schema_extra={
            "relevant_for": {
                "mode": {"values": ["dynamic"], "required": True},
            },
            "order": 4,
        },
    )

    icon_width: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=64,
        description="Width of the icon in pixels. The icon image will be resized to this width while maintaining aspect ratio if height is also specified.",
        examples=[64, "$inputs.icon_width"],
        json_schema_extra={
            "always_visible": True,
        },
    )

    icon_height: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=64,
        description="Height of the icon in pixels. The icon image will be resized to this height while maintaining aspect ratio if width is also specified.",
        examples=[64, "$inputs.icon_height"],
        json_schema_extra={
            "always_visible": True,
        },
    )

    position: Optional[
        Union[
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
        ]
    ] = Field(
        default="TOP_CENTER",
        description="Anchor position for placing icons relative to each detection's bounding box (dynamic mode only). Options include: CENTER (center of box), corners (TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT), edge midpoints (TOP_CENTER, CENTER_LEFT, CENTER_RIGHT, BOTTOM_CENTER), or CENTER_OF_MASS (center of mass of the object).",
        examples=["TOP_CENTER", "$inputs.position"],
        json_schema_extra={
            "relevant_for": {
                "mode": {"values": ["dynamic"], "required": False},
            },
        },
    )

    x_position: Optional[Union[int, Selector(kind=[INTEGER_KIND])]] = Field(
        default=10,
        description="X coordinate for static mode positioning. Positive values position from the left edge of the image. Negative values position from the right edge (e.g., -10 places the icon 10 pixels from the right edge).",
        examples=[10, -10, "$inputs.x_position"],
        json_schema_extra={
            "relevant_for": {
                "mode": {"values": ["static"], "required": True},
            },
        },
    )

    y_position: Optional[Union[int, Selector(kind=[INTEGER_KIND])]] = Field(
        default=10,
        description="Y coordinate for static mode positioning. Positive values position from the top edge of the image. Negative values position from the bottom edge (e.g., -10 places the icon 10 pixels from the bottom edge).",
        examples=[10, -10, "$inputs.y_position"],
        json_schema_extra={
            "relevant_for": {
                "mode": {"values": ["static"], "required": True},
            },
        },
    )

    @model_validator(mode="after")
    def validate_mode_parameters(self) -> "IconManifest":
        if self.mode == "dynamic":
            if self.predictions is None:
                raise ValueError("The 'predictions' field is required for dynamic mode")
        return self

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class IconVisualizationBlockV1(VisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotatorCache = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return IconManifest

    def getAnnotator(
        self,
        icon_width: int,
        icon_height: int,
        position: Optional[str] = None,
    ) -> Optional[sv.annotators.base.BaseAnnotator]:
        if position is not None:
            key = f"dynamic_{icon_width}_{icon_height}_{position}"
            if key not in self.annotatorCache:
                self.annotatorCache[key] = sv.IconAnnotator(
                    icon_resolution_wh=(icon_width, icon_height),
                    icon_position=getattr(sv.Position, position),
                )
            return self.annotatorCache[key]
        return None

    def run(
        self,
        image: WorkflowImageData,
        copy_image: bool,
        mode: str,
        icon: WorkflowImageData,
        predictions: Optional[sv.Detections],
        icon_width: int,
        icon_height: int,
        position: Optional[str],
        x_position: Optional[int],
        y_position: Optional[int],
    ) -> BlockResult:
        annotated_image = image.numpy_image.copy() if copy_image else image.numpy_image
        icon_np = icon.numpy_image.copy()

        import os
        import tempfile

        import cv2

        # WorkflowImageData loses alpha channels when loading images.
        # Try to recover them from the original source.
        if icon_np.shape[2] == 3:
            # Try reloading from file with IMREAD_UNCHANGED
            if (
                hasattr(icon, "_image_reference")
                and icon._image_reference
                and not icon._image_reference.startswith("http")
            ):
                try:
                    icon_with_alpha = cv2.imread(
                        icon._image_reference, cv2.IMREAD_UNCHANGED
                    )
                    if icon_with_alpha is not None and icon_with_alpha.shape[2] == 4:
                        icon_np = icon_with_alpha
                except:
                    pass

            # Try decoding base64 with alpha preserved
            if (
                icon_np.shape[2] == 3
                and hasattr(icon, "_base64_image")
                and icon._base64_image
            ):
                try:
                    import base64

                    image_bytes = base64.b64decode(icon._base64_image)
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    decoded = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                    if decoded is not None and len(decoded.shape) >= 2:
                        if len(decoded.shape) == 2:
                            decoded = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)
                        if decoded.shape[2] == 4:
                            icon_np = decoded
                except:
                    pass

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Ensure proper format for IconAnnotator
            if len(icon_np.shape) == 2:
                icon_np = cv2.cvtColor(icon_np, cv2.COLOR_GRAY2BGR)
                alpha = (
                    np.ones(
                        (icon_np.shape[0], icon_np.shape[1], 1), dtype=icon_np.dtype
                    )
                    * 255
                )
                icon_np = np.concatenate([icon_np, alpha], axis=2)
            elif icon_np.shape[2] == 3:
                alpha = (
                    np.ones(
                        (icon_np.shape[0], icon_np.shape[1], 1), dtype=icon_np.dtype
                    )
                    * 255
                )
                icon_np = np.concatenate([icon_np, alpha], axis=2)

            cv2.imwrite(f.name, icon_np)
            icon_path = f.name

        try:
            if mode == "static":
                img_height, img_width = annotated_image.shape[:2]

                # Handle negative positioning (from right/bottom edges)
                if x_position < 0:
                    actual_x = img_width + x_position - icon_width
                else:
                    actual_x = x_position

                if y_position < 0:
                    actual_y = img_height + y_position - icon_height
                else:
                    actual_y = y_position

                # IconAnnotator expects a detection, so create one at the desired position
                center_x = actual_x + icon_width // 2
                center_y = actual_y + icon_height // 2

                static_detections = sv.Detections(
                    xyxy=np.array(
                        [[center_x - 1, center_y - 1, center_x + 1, center_y + 1]],
                        dtype=np.float64,
                    ),
                    class_id=np.array([0]),
                    confidence=np.array([1.0]),
                )

                annotator = sv.IconAnnotator(
                    icon_resolution_wh=(icon_width, icon_height),
                    icon_position=sv.Position.CENTER,
                )

                annotated_image = annotator.annotate(
                    scene=annotated_image,
                    detections=static_detections,
                    icon_path=icon_path,
                )

            elif mode == "dynamic" and predictions is not None and len(predictions) > 0:
                annotator = self.getAnnotator(
                    icon_width=icon_width,
                    icon_height=icon_height,
                    position=position,
                )

                if annotator is not None:
                    annotated_image = annotator.annotate(
                        scene=annotated_image,
                        detections=predictions,
                        icon_path=icon_path,
                    )
        finally:
            os.unlink(icon_path)

        return {
            OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                origin_image_data=image, numpy_image=annotated_image
            )
        }
