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
The `IconVisualization` block draws icons on an image using Supervision's `sv.IconAnnotator`.
It supports two modes:
1. **Static Mode**: Position an icon at a fixed location (e.g., for watermarks)
2. **Dynamic Mode**: Position icons based on detection coordinates
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
        description="The icon image to place on the input image (PNG with transparency recommended)",
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
        description="Mode for placing icons: 'static' for fixed position (watermark), 'dynamic' for detection-based",
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
        description="Model predictions to place icons on (required for dynamic mode)",
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
        description="Width of the icon in pixels",
        examples=[64, "$inputs.icon_width"],
        json_schema_extra={
            "always_visible": True,
        },
    )

    icon_height: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=64,
        description="Height of the icon in pixels",
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
        description="Position relative to detection for dynamic mode",
        examples=["TOP_CENTER", "$inputs.position"],
        json_schema_extra={
            "relevant_for": {
                "mode": {"values": ["dynamic"], "required": False},
            },
        },
    )

    x_position: Optional[Union[int, Selector(kind=[INTEGER_KIND])]] = Field(
        default=10,
        description="X coordinate for static mode. Positive values from left edge, negative from right edge",
        examples=[10, -10, "$inputs.x_position"],
        json_schema_extra={
            "relevant_for": {
                "mode": {"values": ["static"], "required": True},
            },
        },
    )

    y_position: Optional[Union[int, Selector(kind=[INTEGER_KIND])]] = Field(
        default=10,
        description="Y coordinate for static mode. Positive values from top edge, negative from bottom edge",
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
