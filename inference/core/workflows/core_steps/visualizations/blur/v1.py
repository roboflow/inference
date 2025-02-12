from typing import Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
    PredictionsVisualizationBlock,
    PredictionsVisualizationManifest,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.execution_engine.entities.types import (
    INTEGER_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

TYPE: str = "roboflow_core/blur_visualization@v1"
SHORT_DESCRIPTION = "Blur detected objects in an image."
LONG_DESCRIPTION = """
The `BlurVisualization` block blurs detected
objects in an image using Supervision's `sv.BlurAnnotator`.
"""


class BlurManifest(PredictionsVisualizationManifest):
    type: Literal[f"{TYPE}", "BlurVisualization"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Blur Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "search_keywords": ["annotator"],
            "ui_manifest": {
                "section": "visualization",
                "icon": "fad fa-glasses",
                "blockPriority": 4,
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

    kernel_size: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Size of the average pooling kernel used for blurring.",
        default=15,
        examples=[15, "$inputs.kernel_size"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class BlurVisualizationBlockV1(PredictionsVisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotatorCache = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlurManifest

    def getAnnotator(
        self,
        kernel_size: int,
    ) -> sv.annotators.base.BaseAnnotator:
        key = "_".join(map(str, [kernel_size]))

        if key not in self.annotatorCache:
            self.annotatorCache[key] = sv.BlurAnnotator(kernel_size=kernel_size)
        return self.annotatorCache[key]

    def run(
        self,
        image: WorkflowImageData,
        predictions: sv.Detections,
        copy_image: bool,
        kernel_size: Optional[int],
    ) -> BlockResult:
        annotator = self.getAnnotator(kernel_size)
        annotated_image = annotator.annotate(
            scene=image.numpy_image.copy() if copy_image else image.numpy_image,
            detections=predictions,
        )
        return {
            OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                origin_image_data=image, numpy_image=annotated_image
            )
        }
