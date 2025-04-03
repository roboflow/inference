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
The `TriangleVisualization` block draws triangle markers on an image at specific coordinates
based on provided detections using Supervision's `sv.TriangleAnnotator`.
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
        description="The anchor position for placing the triangle.",
        examples=["CENTER", "$inputs.position"],
    )

    base: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Base width of the triangle in pixels.",
        default=10,
        examples=[10, "$inputs.base"],
    )

    height: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Height of the triangle in pixels.",
        default=10,
        examples=[10, "$inputs.height"],
    )

    outline_thickness: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Thickness of the outline of the triangle in pixels.",
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
