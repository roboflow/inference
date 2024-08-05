from typing import List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.visualizations.annotators.background_color import (
    BackgroundColorAnnotator,
)
from inference.core.workflows.core_steps.visualizations.base import (
    OUTPUT_IMAGE_KEY,
    VisualizationBlock,
    VisualizationManifest,
)
from inference.core.workflows.core_steps.visualizations.utils import str_to_color
from inference.core.workflows.entities.base import WorkflowImageData
from inference.core.workflows.entities.types import (
    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    STRING_KIND,
    FloatZeroToOne,
    StepOutputSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

TYPE: str = "BackgroundColorVisualization"
SHORT_DESCRIPTION = (
    "Paints a mask over all areas outside of detected regions in an image."
)
LONG_DESCRIPTION = """
The `BackgroundColorVisualization` block draws all areas
outside of detected regions in an image with a specified
color.
"""


class BackgroundColorManifest(VisualizationManifest):
    type: Literal[f"{TYPE}"]
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
        }
    )

    color: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = Field(  # type: ignore
        description="Color of the background.",
        default="BLACK",
        examples=["WHITE", "#FFFFFF", "rgb(255, 255, 255)" "$inputs.background_color"],
    )

    opacity: Union[FloatZeroToOne, WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        description="Transparency of the Mask overlay.",
        default=0.5,
        examples=[0.5, "$inputs.opacity"],
    )

    @classmethod
    def get_block_version(cls) -> int:
        return 1

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return "~=1.0.0"


class BackgroundColorVisualizationBlock(VisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotatorCache = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BackgroundColorManifest

    def getAnnotator(
        self,
        color: str,
        opacity: float,
    ) -> sv.annotators.base.BaseAnnotator:
        key = "_".join(
            map(
                str,
                [
                    color,
                    opacity,
                ],
            )
        )

        if key not in self.annotatorCache:
            background_color = str_to_color(color)
            self.annotatorCache[key] = BackgroundColorAnnotator(
                color=background_color,
                opacity=opacity,
            )

        return self.annotatorCache[key]

    async def run(
        self,
        image: WorkflowImageData,
        predictions: sv.Detections,
        copy_image: bool,
        color: str,
        opacity: Optional[float],
    ) -> BlockResult:
        annotator = self.getAnnotator(color, opacity)

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
