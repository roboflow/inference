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
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

TYPE: str = "roboflow_core/circle_visualization@v1"
SHORT_DESCRIPTION = "Draws a circle around detected objects in an image."
LONG_DESCRIPTION = """
The `CircleVisualization` block draws a circle around detected
objects in an image using Supervision's `sv.CircleAnnotator`.
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
        }
    )

    thickness: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Thickness of the lines in pixels.",
        default=2,
        examples=[2, "$inputs.thickness"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


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

        output = WorkflowImageData(
            parent_metadata=image.parent_metadata,
            workflow_root_ancestor_metadata=image.workflow_root_ancestor_metadata,
            numpy_image=annotated_image,
        )

        return {OUTPUT_IMAGE_KEY: output}
