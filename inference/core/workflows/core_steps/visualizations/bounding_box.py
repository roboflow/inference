from inference.core.workflows.core_steps.visualizations.base import (
    VisualizationManifest,
    VisualizationBlock
)

from typing import Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.entities.base import (
    WorkflowImageData,
)
from inference.core.workflows.entities.types import (
    INTEGER_KIND,
    FloatZeroToOne,
    FLOAT_ZERO_TO_ONE_KIND,
    STRING_KIND,
    WorkflowParameterSelector
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlockManifest,
)

OUTPUT_IMAGE_KEY: str = "image"

TYPE: str = "BoundingBoxVisualization"
SHORT_DESCRIPTION = (
    "Draws a box around detected objects in an image."
)
LONG_DESCRIPTION = """
The `BoundingBoxVisualization` block draws a box around detected
objects in an image using Supervision's `sv.RoundBoxAnnotator`.
"""

class BoundingBoxManifest(VisualizationManifest):
    type: Literal[f"{TYPE}"]
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
        }
    )
    
    thickness: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field( # type: ignore
        description="Thickness of the bounding box in pixels.",
        default=2,
    )

    roundness: Union[FloatZeroToOne, WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field( # type: ignore
        description="Roundness of the corners of the bounding box.",
        default=0.0,
    )

class BoundingBoxVisualizationBlock(VisualizationBlock):
    def __init__(self):
        self.annotatorCache = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BoundingBoxManifest

    def getAnnotator(
        self,
        color_lookup:str,
        thickness:int,
        roundness: float
    ) -> sv.annotators.base.BaseAnnotator:
        key = f"{color_lookup}_{thickness}_{roundness}"
        if key not in self.annotatorCache:
            if roundness == 0:
                self.annotatorCache[key] = sv.BoxAnnotator(
                    color_lookup=getattr(sv.annotators.utils.ColorLookup, color_lookup),
                    thickness=thickness
                )
            else:
                self.annotatorCache[key] = sv.RoundBoxAnnotator(
                    color_lookup=getattr(sv.annotators.utils.ColorLookup, color_lookup),
                    thickness=thickness,
                    roundness=roundness
                )
        return self.annotatorCache[key]

    async def run(
        self,
        image: WorkflowImageData,
        predictions: sv.Detections,
        copy_image: bool,
        color_lookup: Optional[str],
        thickness: Optional[int],
        roundness: Optional[float],
    ) -> BlockResult:
        annotator = self.getAnnotator(color_lookup, thickness, roundness)

        annotated_image = annotator.annotate(
            scene=image.numpy_image.copy() if copy_image else image.numpy_image,
            detections=predictions
        )

        output = WorkflowImageData(
            parent_metadata=image.parent_metadata,
            workflow_root_ancestor_metadata=image.workflow_root_ancestor_metadata,
            numpy_image=annotated_image,
        )

        return {
            OUTPUT_IMAGE_KEY: output
        }
