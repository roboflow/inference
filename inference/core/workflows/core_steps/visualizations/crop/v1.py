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
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

TYPE: str = "roboflow_core/crop_visualization@v1"
SHORT_DESCRIPTION = "Draws scaled up crops of detections on the scene."
LONG_DESCRIPTION = """
The `CropVisualization` block draws scaled up crops of detections
on the scene using Supervision's `sv.CropAnnotator`.
"""


class CropManifest(ColorableVisualizationManifest):
    type: Literal[f"{TYPE}", "CropVisualization"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Crop Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
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
        WorkflowParameterSelector(kind=[STRING_KIND]),
    ] = Field(  # type: ignore
        default="TOP_CENTER",
        description="The anchor position for placing the crop.",
        examples=["CENTER", "$inputs.position"],
    )

    scale_factor: Union[float, WorkflowParameterSelector(kind=[FLOAT_KIND])] = Field(  # type: ignore
        description="The factor by which to scale the cropped image part. A factor of 2, for example, would double the size of the cropped area, allowing for a closer view of the detection.",
        default=2.0,
        examples=[2.0, "$inputs.scale_factor"],
    )

    border_thickness: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Thickness of the outline in pixels.",
        default=2,
        examples=[2, "$inputs.border_thickness"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class CropVisualizationBlockV1(ColorableVisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotatorCache = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return CropManifest

    def getAnnotator(
        self,
        color_palette: str,
        palette_size: int,
        custom_colors: List[str],
        color_axis: str,
        position: str,
        scale_factor: float,
        border_thickness: int,
    ) -> sv.annotators.base.BaseAnnotator:
        key = "_".join(
            map(
                str,
                [
                    color_palette,
                    palette_size,
                    color_axis,
                    position,
                    scale_factor,
                    border_thickness,
                ],
            )
        )

        if key not in self.annotatorCache:
            palette = self.getPalette(color_palette, palette_size, custom_colors)

            self.annotatorCache[key] = sv.CropAnnotator(
                border_color=palette,
                border_color_lookup=getattr(sv.ColorLookup, color_axis),
                position=getattr(sv.Position, position),
                scale_factor=scale_factor,
                border_thickness=border_thickness,
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
        scale_factor: Optional[float],
        border_thickness: Optional[int],
    ) -> BlockResult:
        annotator = self.getAnnotator(
            color_palette,
            palette_size,
            custom_colors,
            color_axis,
            position,
            scale_factor,
            border_thickness,
        )

        annotated_image = annotator.annotate(
            scene=image.numpy_image.copy() if copy_image else image.numpy_image,
            detections=predictions,
        )

        output = WorkflowImageData(
            parent_metadata=image.parent_metadata,
            workflow_root_ancestor_metadata=image.workflow_root_ancestor_metadata,
            numpy_image=annotated_image,
        )

        return {OUTPUT_IMAGE_KEY: output}
