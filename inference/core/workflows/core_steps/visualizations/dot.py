from typing import List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.visualizations.base import (
    VisualizationBlock,
    VisualizationManifest,
)
from inference.core.workflows.entities.base import WorkflowImageData
from inference.core.workflows.entities.types import (
    INTEGER_KIND,
    STRING_KIND,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

OUTPUT_IMAGE_KEY: str = "image"

TYPE: str = "DotVisualization"
SHORT_DESCRIPTION = (
    "Draws dots on an image at specific coordinates based on provided detections."
)
LONG_DESCRIPTION = """
The `DotVisualization` block draws dots on an image at specific coordinates
based on provided detections using Supervision's `sv.DotAnnotator`.
"""


class DotManifest(VisualizationManifest):
    type: Literal[f"{TYPE}"]
    model_config = ConfigDict(
        json_schema_extra={
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
        default="CENTER",
        description="The anchor position for placing the dot.",
        examples=["CENTER", "$inputs.position"],
    )

    radius: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Radius of the dot in pixels.",
        default=4,
        examples=[4, "$inputs.radius"],
    )

    outline_thickness: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Thickness of the outline of the dot in pixels.",
        default=0,
        examples=[2, "$inputs.outline_thickness"],
    )


class DotVisualizationBlock(VisualizationBlock):
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

    async def run(
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

        output = WorkflowImageData(
            parent_metadata=image.parent_metadata,
            workflow_root_ancestor_metadata=image.workflow_root_ancestor_metadata,
            numpy_image=annotated_image,
        )

        return {OUTPUT_IMAGE_KEY: output}
