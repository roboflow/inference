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
The `BoundingBoxVisualization` block draws a box around detected
objects in an image using Supervision's `sv.RoundBoxAnnotator`.
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
        description="Set the thickness of the bounding box edges.",
        default=2,
        examples=[2, "$inputs.thickness"],
    )

    roundness: Union[FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        description="Define the roundness of the bounding box corners.",
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
